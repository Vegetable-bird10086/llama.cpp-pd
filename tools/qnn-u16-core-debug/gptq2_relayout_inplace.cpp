#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr char kManifestMagic[8] = {'G', '2', 'R', 'L', 'V', '1', '\0', '\0'};
constexpr char kOldLayout[] = "gs32_source_v1";
constexpr char kNewLayout[] = "i8mm_native_v1";

struct ManifestHeader {
  char magic[8];
  uint32_t version;
  uint32_t count;
};

struct TensorRecord {
  uint64_t offset;
  uint64_t n_bytes;
  uint32_t columns;
  uint32_t rows;
};

bool read_exact(int fd, void* dst, size_t size) {
  auto* out = static_cast<uint8_t*>(dst);
  while (size != 0) {
    const ssize_t got = read(fd, out, size);
    if (got <= 0) {
      return false;
    }
    out += got;
    size -= static_cast<size_t>(got);
  }
  return true;
}

uint8_t canonical_byte(
    const uint8_t* old, uint32_t groups, uint32_t row,
    uint32_t group, uint32_t byte_index) {
  (void) groups;
  const uint32_t a = row / 32;
  const uint32_t d = (row / 8) % 4;
  const uint32_t e = row % 8;
  const uint32_t c = byte_index / 2;
  const uint32_t f = byte_index % 2;
  const size_t index = static_cast<size_t>(group) * 768 +
      (((((static_cast<size_t>(a) * 4 + c) * 4 + d) * 8 + e) * 2 + f));
  return old[index];
}

void convert_block(uint8_t* block, uint32_t groups) {
  const size_t block_bytes = static_cast<size_t>(groups) * 768;
  const size_t native_bytes = static_cast<size_t>(groups) * 512;
  std::vector<uint8_t> old(block, block + block_bytes);

  for (uint32_t row = 0; row < 64; ++row) {
    const uint32_t tile = row / 16;
    const uint32_t row_in_tile = row % 16;
    for (uint32_t group = 0; group < groups; ++group) {
      uint8_t* native =
          block + (static_cast<size_t>(tile) * groups + group) * 128 +
          row_in_tile * 8;
      for (uint32_t lane = 0; lane < 8; ++lane) {
        uint8_t packed = 0;
        for (uint32_t plane = 0; plane < 4; ++plane) {
          const uint32_t code_index = plane * 8 + lane;
          const uint32_t byte_index = code_index / 4;
          const uint32_t shift = (code_index % 4) * 2;
          const uint8_t code =
              (canonical_byte(old.data(), groups, row, group, byte_index) >> shift) & 3;
          packed |= static_cast<uint8_t>(code << (plane * 2));
        }
        native[lane] = packed;
      }
    }
  }
  for (uint32_t group = 0; group < groups; ++group) {
    std::memcpy(
        block + native_bytes + static_cast<size_t>(group) * 256,
        old.data() + static_cast<size_t>(group) * 768 + 512,
        256);
  }
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s MODEL.gguf MANIFEST.bin\n", argv[0]);
    return 2;
  }
  const int manifest_fd = open(argv[2], O_RDONLY);
  if (manifest_fd < 0) {
    std::fprintf(stderr, "open manifest: %s\n", std::strerror(errno));
    return 1;
  }
  ManifestHeader header{};
  if (!read_exact(manifest_fd, &header, sizeof(header)) ||
      std::memcmp(header.magic, kManifestMagic, sizeof(header.magic)) != 0 ||
      header.version != 1 || header.count == 0) {
    std::fprintf(stderr, "invalid relayout manifest\n");
    close(manifest_fd);
    return 1;
  }
  std::vector<TensorRecord> records(header.count);
  if (!read_exact(manifest_fd, records.data(), records.size() * sizeof(TensorRecord))) {
    std::fprintf(stderr, "truncated relayout manifest\n");
    close(manifest_fd);
    return 1;
  }
  close(manifest_fd);

  const int model_fd = open(argv[1], O_RDWR);
  struct stat st {};
  if (model_fd < 0 || fstat(model_fd, &st) != 0) {
    std::fprintf(stderr, "open model: %s\n", std::strerror(errno));
    return 1;
  }
  auto* model = static_cast<uint8_t*>(
      mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, model_fd, 0));
  if (model == MAP_FAILED) {
    std::fprintf(stderr, "mmap model: %s\n", std::strerror(errno));
    close(model_fd);
    return 1;
  }

  size_t marker_count = 0;
  for (size_t i = 0; i + sizeof(kOldLayout) - 1 <= static_cast<size_t>(st.st_size); ++i) {
    if (std::memcmp(model + i, kOldLayout, sizeof(kOldLayout) - 1) == 0) {
      std::memcpy(model + i, kNewLayout, sizeof(kNewLayout) - 1);
      ++marker_count;
    }
  }
  if (marker_count != 1) {
    std::fprintf(stderr, "expected one layout marker, found %zu\n", marker_count);
    munmap(model, st.st_size);
    close(model_fd);
    return 1;
  }

  for (size_t tensor_index = 0; tensor_index < records.size(); ++tensor_index) {
    const TensorRecord& tensor = records[tensor_index];
    if (tensor.columns % 32 != 0 || tensor.rows % 64 != 0 ||
        tensor.offset + tensor.n_bytes > static_cast<uint64_t>(st.st_size)) {
      std::fprintf(stderr, "invalid tensor record %zu\n", tensor_index);
      return 1;
    }
    const uint32_t groups = tensor.columns / 32;
    const size_t block_bytes = static_cast<size_t>(groups) * 768;
    if (tensor.n_bytes != static_cast<uint64_t>(tensor.rows / 64) * block_bytes) {
      std::fprintf(stderr, "unexpected tensor size %zu\n", tensor_index);
      return 1;
    }
    for (uint32_t row_block = 0; row_block < tensor.rows / 64; ++row_block) {
      convert_block(
          model + tensor.offset + static_cast<size_t>(row_block) * block_bytes,
          groups);
    }
    std::fprintf(stderr, "[%zu/%zu] bytes=%llu\n", tensor_index + 1, records.size(),
        static_cast<unsigned long long>(tensor.n_bytes));
  }
  const int sync_status = msync(model, st.st_size, MS_SYNC);
  munmap(model, st.st_size);
  close(model_fd);
  if (sync_status != 0) {
    std::fprintf(stderr, "msync: %s\n", std::strerror(errno));
    return 1;
  }
  return 0;
}
