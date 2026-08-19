#include "llama-qnn-quant-profile.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr std::array<uint8_t, 8> kMagic = {'L', 'Q', 'N', 'N', 'P', 'R', 'F', '\0'};
constexpr uint32_t kVersion = 5;
constexpr uint32_t kEndianTag = 0x01020304U;
constexpr size_t kHeaderBytes = 8 + 4 + 4 + 8 + 8;
constexpr uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);

uint64_t payload_checksum(const uint8_t * data, size_t size) {
    uint64_t result = kFnvOffset;
    for (size_t index = 0; index < size; ++index) {
        result = (result ^ data[index]) * kFnvPrime;
    }
    return result;
}

[[noreturn]] void binary_fail(const std::string & message) {
    throw std::runtime_error("invalid binary QNN quantization profile: " + message);
}

std::string shard_sidecar_path(
        const std::string & index_path,
        size_t shard_index) {
    return index_path + ".shard" + std::to_string(shard_index) + ".bin";
}

std::string sidecar_data_path(const std::string & meta_path) {
    static constexpr const char * suffix = ".meta";
    if (meta_path.size() >= std::strlen(suffix) &&
        meta_path.compare(
            meta_path.size() - std::strlen(suffix),
            std::strlen(suffix),
            suffix) == 0) {
        return meta_path.substr(0, meta_path.size() - std::strlen(suffix)) +
            ".bin";
    }
    return meta_path + ".sidecar.bin";
}

class shard_payload_writer {
public:
    shard_payload_writer(
            const std::string & meta_path,
            std::vector<size_t> shard_sizes) :
        stream(
            sidecar_data_path(meta_path),
            std::ios::binary | std::ios::trunc),
        sizes(std::move(shard_sizes)) {
        if (sizes.empty()) {
            throw std::runtime_error("sharded QNN profile requires at least one shard");
        }
        if (!stream) {
            throw std::runtime_error(
                "cannot create QNN sidecar payload: " +
                sidecar_data_path(meta_path));
        }
        offsets.assign(sizes.size(), 0);
        bases.resize(sizes.size(), 0);
        uint64_t end = 0;
        for (size_t index = 0; index < sizes.size(); ++index) {
            end = (end + 4095) & ~UINT64_C(4095);
            bases[index] = end;
            end += sizes[index];
        }
        if (end != 0) {
            stream.seekp(static_cast<std::streamoff>(end - 1));
            stream.put(0);
            stream.seekp(0);
            if (!stream) {
                throw std::runtime_error(
                    "cannot size QNN sidecar payload file");
            }
        }
    }

    uint64_t append(
            int32_t requested_shard,
            const void * data,
            size_t size,
            size_t alignment) {
        const size_t shard = requested_shard < 0
            ? 0
            : static_cast<size_t>(requested_shard);
        if (shard >= sizes.size()) {
            throw std::runtime_error(
                "QNN profile record references an invalid shard " +
                std::to_string(requested_shard));
        }
        const uint64_t padding =
            (alignment - offsets[shard] % alignment) % alignment;
        offsets[shard] += padding;
        const uint64_t result = offsets[shard];
        if (size > sizes[shard] - offsets[shard]) {
            throw std::runtime_error(
                "QNN sidecar sizing pass disagrees with serialization");
        }
        if (size != 0) {
            if (data == nullptr) {
                throw std::runtime_error(
                    "cannot export a nonresident QNN sidecar buffer");
            }
            stream.seekp(static_cast<std::streamoff>(bases[shard] + result));
            stream.write(
                static_cast<const char *>(data),
                static_cast<std::streamsize>(size));
            offsets[shard] += size;
        }
        if (!stream) {
            throw std::runtime_error("failed while writing sharded QNN sidecar");
        }
        return result;
    }

    void finish() {
        for (size_t index = 0; index < sizes.size(); ++index) {
            if (offsets[index] != sizes[index]) {
                throw std::runtime_error(
                    "QNN sidecar shard size disagrees with serialization: shard=" +
                    std::to_string(index));
            }
        }
        stream.flush();
        if (!stream) {
            throw std::runtime_error("failed to finalize QNN sidecar payload");
        }
    }

    size_t shard_count() const {
        return sizes.size();
    }

    uint64_t file_offset(size_t shard) const {
        return bases.at(shard);
    }

    uint64_t shard_size(size_t shard) const {
        return sizes.at(shard);
    }

private:
    std::ofstream stream;
    std::vector<size_t> sizes;
    std::vector<uint64_t> bases;
    std::vector<uint64_t> offsets;
};

class binary_writer {
public:
    explicit binary_writer(
            const std::string & path,
            uint32_t version = 3,
            std::shared_ptr<shard_payload_writer> shard_payloads = nullptr) :
        stream(path, std::ios::binary),
        shard_payloads(std::move(shard_payloads)) {
        if (!stream) {
            throw std::runtime_error("cannot create binary QNN quantization profile: " + path);
        }
        bytes(kMagic.data(), kMagic.size());
        scalar(version);
        scalar(kEndianTag);
        scalar(uint64_t{0});
        scalar(uint64_t{0});
        writing_payload = true;
    }

    ~binary_writer() = default;

    template <typename T>
    void scalar(T value) {
        static_assert(std::is_integral<T>::value, "binary scalar must be integral");
        using U = typename std::make_unsigned<T>::type;
        U bits = static_cast<U>(value);
        for (size_t index = 0; index < sizeof(T); ++index) {
            const uint8_t byte = static_cast<uint8_t>(bits >> (index * 8));
            bytes(&byte, 1);
        }
    }

    void boolean(bool value) {
        scalar<uint8_t>(value ? 1 : 0);
    }

    void real64(double value) {
        uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "binary64 is required");
        std::memcpy(&bits, &value, sizeof(bits));
        scalar(bits);
    }

    void string(const std::string & value) {
        scalar<uint64_t>(value.size());
        bytes(value.data(), value.size());
    }

    template <typename Container, typename Fn>
    void vector(const Container & values, Fn && write_one) {
        scalar<uint64_t>(values.size());
        for (const auto & value : values) {
            write_one(value);
        }
    }

    template <typename Container>
    void scalar_vector(const Container & values) {
        using T = typename Container::value_type;
        vector(values, [&](const T value) { scalar<T>(value); });
    }

    template <typename Container>
    void byte_vector(const Container & values) {
        scalar<uint64_t>(values.size());
        bytes(values.data(), values.size());
    }

    template <typename Container>
    void aligned_scalar_buffer(const Container & values) {
        using T = typename Container::value_type;
        scalar<uint64_t>(values.size());
        if (shard_payloads) {
            scalar<uint32_t>(normalized_shard());
            const uint64_t offset = shard_payloads->append(
                current_shard,
                values.data(),
                values.size() * sizeof(T),
                64);
            scalar(offset);
            return;
        }
        align(64);
        for (const T value : values) {
            scalar<T>(value);
        }
    }

    template <typename Container>
    void aligned_byte_buffer(const Container & values) {
        scalar<uint64_t>(values.size());
        if (shard_payloads) {
            scalar<uint32_t>(normalized_shard());
            const uint64_t offset = shard_payloads->append(
                current_shard,
                values.data(),
                values.size(),
                64);
            scalar(offset);
            return;
        }
        align(64);
        bytes(values.data(), values.size());
    }

    void set_current_shard(int32_t shard) {
        current_shard = shard;
    }

    void finish() {
        if (!stream) {
            throw std::runtime_error("failed while writing binary QNN quantization profile");
        }
        const auto end = stream.tellp();
        if (end < static_cast<std::streamoff>(kHeaderBytes)) {
            throw std::runtime_error("failed to determine binary QNN profile size");
        }
        const uint64_t payload_bytes =
            static_cast<uint64_t>(end - static_cast<std::streamoff>(kHeaderBytes));
        writing_payload = false;
        stream.seekp(16);
        for (size_t index = 0; index < sizeof(payload_bytes); ++index) {
            stream.put(static_cast<char>(payload_bytes >> (index * 8)));
        }
        for (size_t index = 0; index < sizeof(checksum); ++index) {
            stream.put(static_cast<char>(checksum >> (index * 8)));
        }
        stream.flush();
        if (!stream) {
            throw std::runtime_error("failed to finalize binary QNN quantization profile");
        }
        if (shard_payloads) {
            shard_payloads->finish();
        }
    }

private:
    uint32_t normalized_shard() const {
        const size_t shard = current_shard < 0
            ? 0
            : static_cast<size_t>(current_shard);
        if (!shard_payloads || shard >= shard_payloads->shard_count()) {
            throw std::runtime_error(
                "QNN profile record references an invalid shard " +
                std::to_string(current_shard));
        }
        return static_cast<uint32_t>(shard);
    }

    void align(size_t alignment) {
        const auto position = stream.tellp();
        if (position < 0) {
            throw std::runtime_error("failed to align binary QNN profile");
        }
        const size_t padding =
            (alignment - static_cast<size_t>(position) % alignment) % alignment;
        static constexpr uint8_t zeros[64] = {};
        bytes(zeros, padding);
    }

    void bytes(const void * data, size_t size) {
        if (size != 0) {
            stream.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
            if (writing_payload) {
                const auto * input = static_cast<const uint8_t *>(data);
                for (size_t index = 0; index < size; ++index) {
                    checksum = (checksum ^ input[index]) * kFnvPrime;
                }
            }
        }
    }

    std::ofstream stream;
    uint64_t checksum = kFnvOffset;
    bool writing_payload = false;
    std::shared_ptr<shard_payload_writer> shard_payloads;
    int32_t current_shard = 0;
};

class mapped_file {
public:
    explicit mapped_file(const std::string & path) {
        fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            throw std::runtime_error(
                "cannot open binary QNN quantization profile: " + path +
                ": " + std::strerror(errno));
        }
        struct stat status {};
        if (fstat(fd, &status) != 0 || status.st_size < 0) {
            const std::string error = std::strerror(errno);
            close(fd);
            fd = -1;
            throw std::runtime_error(
                "cannot stat binary QNN quantization profile: " + path + ": " + error);
        }
        size = static_cast<size_t>(status.st_size);
        if (size == 0) {
            close(fd);
            fd = -1;
            binary_fail("file is empty");
        }
        const char * anonymous_value =
            std::getenv("LLAMA_QNN_U16_PROFILE_ANONYMOUS_BUFFER");
        anonymous = anonymous_value != nullptr &&
            std::strcmp(anonymous_value, "1") == 0;
        if (anonymous) {
            void * allocation = mmap(
                nullptr, size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (allocation == MAP_FAILED) {
                const std::string error = std::strerror(errno);
                close(fd);
                fd = -1;
                throw std::runtime_error(
                    "cannot allocate anonymous QNN profile buffer: " + path +
                    ": " + error);
            }
            size_t offset = 0;
            while (offset < size) {
                const ssize_t count = pread(
                    fd,
                    static_cast<uint8_t *>(allocation) + offset,
                    size - offset,
                    static_cast<off_t>(offset));
                if (count > 0) {
                    offset += static_cast<size_t>(count);
                    continue;
                }
                if (count < 0 && errno == EINTR) {
                    continue;
                }
                const std::string error = count == 0
                    ? "unexpected EOF"
                    : std::strerror(errno);
                munmap(allocation, size);
                close(fd);
                fd = -1;
                throw std::runtime_error(
                    "cannot read anonymous QNN profile buffer: " + path +
                    ": " + error);
            }
            if (mprotect(allocation, size, PROT_READ) != 0) {
                const std::string error = std::strerror(errno);
                munmap(allocation, size);
                close(fd);
                fd = -1;
                throw std::runtime_error(
                    "cannot protect anonymous QNN profile buffer: " + path +
                    ": " + error);
            }
            data = static_cast<const uint8_t *>(allocation);
            close(fd);
            fd = -1;
            std::fprintf(
                stderr,
                "qnn-profile: anonymous binary backing ready bytes=%zu path=%s\n",
                size,
                path.c_str());
        } else {
            data = static_cast<const uint8_t *>(
                mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
            if (data == MAP_FAILED) {
                data = nullptr;
                const std::string error = std::strerror(errno);
                close(fd);
                fd = -1;
                throw std::runtime_error(
                    "cannot mmap binary QNN quantization profile: " + path +
                    ": " + error);
            }
            madvise(const_cast<uint8_t *>(data), size, MADV_SEQUENTIAL);
        }
    }

    ~mapped_file() {
        if (data != nullptr) {
            if (!anonymous) {
                madvise(const_cast<uint8_t *>(data), size, MADV_DONTNEED);
            }
            munmap(const_cast<uint8_t *>(data), size);
        }
        if (fd >= 0) {
            close(fd);
        }
    }

    mapped_file(const mapped_file &) = delete;
    mapped_file & operator=(const mapped_file &) = delete;

    bool stream_prepare(size_t shard_count) {
        if (anonymous || streaming || fd < 0 || shard_count == 0) {
            return false;
        }
        void * replacement = mmap(
            const_cast<uint8_t *>(data),
            size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
            -1,
            0);
        if (replacement == MAP_FAILED || replacement != data) {
            return false;
        }
        anonymous = true;
        streaming = true;
        streaming_shard_count = shard_count;
        streaming_completed.assign(shard_count, false);
        return true;
    }

    bool stream_fill(size_t shard_index, size_t * bytes_loaded) {
        if (!streaming || fd < 0 || shard_index >= streaming_shard_count ||
            streaming_completed[shard_index]) {
            return false;
        }
        const long page_size_long = sysconf(_SC_PAGESIZE);
        if (page_size_long <= 0) {
            return false;
        }
        const size_t page_size = static_cast<size_t>(page_size_long);
        const size_t total_pages = (size + page_size - 1) / page_size;
        const size_t first_page =
            total_pages * shard_index / streaming_shard_count;
        const size_t last_page =
            total_pages * (shard_index + 1) / streaming_shard_count;
        const size_t offset = first_page * page_size;
        const size_t limit = std::min(size, last_page * page_size);
        size_t position = offset;
        while (position < limit) {
            const ssize_t count = pread(
                fd,
                const_cast<uint8_t *>(data) + position,
                limit - position,
                static_cast<off_t>(position));
            if (count > 0) {
                position += static_cast<size_t>(count);
                continue;
            }
            if (count < 0 && errno == EINTR) {
                continue;
            }
            return false;
        }
        streaming_completed[shard_index] = true;
        if (bytes_loaded != nullptr) {
            *bytes_loaded = limit - offset;
        }
        return true;
    }

    bool stream_finish() {
        if (!streaming ||
            std::find(streaming_completed.begin(), streaming_completed.end(), false) !=
                streaming_completed.end()) {
            return false;
        }
        if (mprotect(const_cast<uint8_t *>(data), size, PROT_READ) != 0) {
            return false;
        }
        streaming = false;
        streaming_ready = true;
        close(fd);
        fd = -1;
        return true;
    }

    const uint8_t * data = nullptr;
    size_t size = 0;
    bool anonymous = false;
    bool streaming = false;
    bool streaming_ready = false;

private:
    int fd = -1;
    size_t streaming_shard_count = 0;
    std::vector<bool> streaming_completed;
};

class sharded_file_set {
public:
    sharded_file_set(
            std::shared_ptr<mapped_file> index_mapping,
            const std::string & index_path,
            size_t shard_count) :
        index_mapping_(std::move(index_mapping)) {
        parts.reserve(shard_count);
        for (size_t index = 0; index < shard_count; ++index) {
            part value;
            value.path = shard_sidecar_path(index_path, index);
            value.storage = std::make_shared<llama_qnn_buffer_storage>();
            load_part(value);
            parts.push_back(std::move(value));
        }
    }

    sharded_file_set(
            std::shared_ptr<mapped_file> index_mapping,
            const std::string & meta_path,
            const std::vector<std::pair<uint64_t, uint64_t>> & ranges) :
        index_mapping_(std::move(index_mapping)),
        shared_path_(sidecar_data_path(meta_path)) {
        shared_fd_ = open(shared_path_.c_str(), O_RDONLY | O_CLOEXEC);
        if (shared_fd_ < 0) {
            throw std::runtime_error(
                "cannot open QNN sidecar payload: " + shared_path_ +
                ": " + std::strerror(errno));
        }
        struct stat status {};
        if (fstat(shared_fd_, &status) != 0 || status.st_size < 0) {
            const std::string error = std::strerror(errno);
            close(shared_fd_);
            shared_fd_ = -1;
            throw std::runtime_error(
                "cannot stat QNN sidecar payload: " + shared_path_ +
                ": " + error);
        }
        const uint64_t file_size = static_cast<uint64_t>(status.st_size);
        parts.reserve(ranges.size());
        for (const auto & range : ranges) {
            if (range.first > file_size ||
                range.second > file_size - range.first ||
                range.second > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error(
                    "QNN sidecar shard range exceeds payload file");
            }
            part value;
            value.path = shared_path_;
            value.file_offset = range.first;
            value.size = static_cast<size_t>(range.second);
            value.range_known = true;
            value.storage = std::make_shared<llama_qnn_buffer_storage>();
            load_part(value);
            parts.push_back(std::move(value));
        }
    }

    ~sharded_file_set() {
        if (shared_fd_ >= 0) {
            close(shared_fd_);
        }
    }

    std::vector<std::shared_ptr<llama_qnn_buffer_storage>> storages() const {
        std::vector<std::shared_ptr<llama_qnn_buffer_storage>> result;
        result.reserve(parts.size());
        for (const auto & value : parts) {
            result.push_back(value.storage);
        }
        return result;
    }

    bool stream_prepare(size_t shard_count) {
        if (streaming || shard_count != parts.size()) {
            return false;
        }
        for (auto & value : parts) {
            // Keep the independently allocated virtual address stable because
            // the already-reserved TG graph contains raw pointers into this
            // shard.  Drop only its anonymous physical pages; stream_fill()
            // repopulates the same malloc buffer with ordinary file read().
            // This avoids both file-backed mmap/page-cache residency and a
            // graph rebuild/rebind at the Prefill -> Decode boundary.
            if (value.storage->data != nullptr && value.allocation_size != 0 &&
                    madvise(
                        const_cast<uint8_t *>(value.storage->data),
                        value.allocation_size,
                        MADV_DONTNEED) != 0) {
                return false;
            }
            value.ready = false;
        }
        streaming = true;
        return true;
    }

    bool stream_fill(size_t shard_index, size_t * bytes_loaded) {
        if (!streaming || shard_index >= parts.size() || parts[shard_index].ready) {
            return false;
        }
        try {
            load_part(parts[shard_index]);
        } catch (const std::exception & error) {
            std::fprintf(
                stderr,
                "qnn-profile: failed to load sidecar shard=%zu error=%s\n",
                shard_index,
                error.what());
            return false;
        }
        if (bytes_loaded != nullptr) {
            *bytes_loaded = parts[shard_index].size;
        }
        return true;
    }

    bool stream_finish() {
        if (!streaming) {
            return false;
        }
        for (const auto & value : parts) {
            if (!value.ready) {
                return false;
            }
        }
        streaming = false;
        return true;
    }

private:
    struct part {
        std::string path;
        uint64_t file_offset = 0;
        size_t size = 0;
        size_t allocation_size = 0;
        bool range_known = false;
        std::shared_ptr<llama_qnn_buffer_storage> storage;
        bool ready = false;
    };

    void load_part(part & value) {
        const bool shared = shared_fd_ >= 0;
        const int fd = shared
            ? shared_fd_
            : open(value.path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            throw std::runtime_error(
                "cannot open sharded QNN sidecar: " + value.path +
                ": " + std::strerror(errno));
        }
        struct stat status {};
        if (fstat(fd, &status) != 0 || status.st_size < 0) {
            const std::string error = std::strerror(errno);
            if (!shared) {
                close(fd);
            }
            throw std::runtime_error(
                "cannot stat sharded QNN sidecar: " + value.path +
                ": " + error);
        }
        const size_t size = value.range_known
            ? value.size
            : static_cast<size_t>(status.st_size);
        if (value.file_offset > static_cast<uint64_t>(status.st_size) ||
            size > static_cast<uint64_t>(status.st_size) - value.file_offset ||
            lseek(fd, static_cast<off_t>(value.file_offset), SEEK_SET) < 0) {
            const std::string error = std::strerror(errno);
            if (!shared) {
                close(fd);
            }
            throw std::runtime_error(
                "cannot seek sharded QNN sidecar: " + value.path +
                ": " + error);
        }
        const long page_size_long = sysconf(_SC_PAGESIZE);
        if (page_size_long <= 0) {
            if (!shared) {
                close(fd);
            }
            throw std::runtime_error("cannot query page size for QNN sidecar");
        }
        const size_t page_size = static_cast<size_t>(page_size_long);
        if (size > std::numeric_limits<size_t>::max() - (page_size - 1)) {
            if (!shared) {
                close(fd);
            }
            throw std::runtime_error("QNN sidecar shard allocation overflows");
        }
        const size_t allocation_size =
            size == 0 ? 0 : ((size + page_size - 1) / page_size) * page_size;
        void * allocation = value.storage->allocation.get();
        const bool new_allocation = allocation == nullptr;
        if (new_allocation && allocation_size != 0) {
            const int allocation_error =
                posix_memalign(&allocation, page_size, allocation_size);
            if (allocation_error != 0 || allocation == nullptr) {
                if (!shared) {
                    close(fd);
                }
                throw std::runtime_error(
                    "cannot allocate sharded QNN sidecar buffer: " +
                    value.path + ": " + std::strerror(allocation_error));
            }
        }
        size_t offset = 0;
        while (offset < size) {
            const ssize_t count = read(
                fd,
                static_cast<uint8_t *>(allocation) + offset,
                size - offset);
            if (count > 0) {
                offset += static_cast<size_t>(count);
                continue;
            }
            if (count < 0 && errno == EINTR) {
                continue;
            }
            const std::string error = count == 0
                ? "unexpected EOF"
                : std::strerror(errno);
            if (new_allocation) {
                free(allocation);
            }
            if (!shared) {
                close(fd);
            }
            throw std::runtime_error(
                "cannot read sharded QNN sidecar: " + value.path +
                ": " + error);
        }
        if (!shared) {
            close(fd);
        }
        value.size = size;
        value.allocation_size = allocation_size;
        if (new_allocation) {
            value.storage->allocation = std::shared_ptr<void>(
                allocation,
                [](void * pointer) { free(pointer); });
        }
        value.storage->data = static_cast<const uint8_t *>(allocation);
        value.storage->size = size;
        value.ready = true;
    }

    std::shared_ptr<mapped_file> index_mapping_;
    std::string shared_path_;
    int shared_fd_ = -1;
    std::vector<part> parts;
    bool streaming = false;
};

class binary_reader {
public:
    binary_reader(
            const uint8_t * data,
            size_t size,
            bool mapped_buffers = false) :
        current(data), end(data + size), mapped_buffers(mapped_buffers) {}

    void set_sharded_buffers(
            std::vector<std::shared_ptr<llama_qnn_buffer_storage>> storages) {
        sharded_storages = std::move(storages);
        mapped_buffers = true;
    }

    template <typename T>
    T scalar() {
        static_assert(std::is_integral<T>::value, "binary scalar must be integral");
        require(sizeof(T));
        using U = typename std::make_unsigned<T>::type;
        U value = 0;
        for (size_t index = 0; index < sizeof(T); ++index) {
            value |= static_cast<U>(current[index]) << (index * 8);
        }
        current += sizeof(T);
        return static_cast<T>(value);
    }

    bool boolean() {
        const uint8_t value = scalar<uint8_t>();
        if (value > 1) {
            binary_fail("boolean value is outside {0,1}");
        }
        return value != 0;
    }

    double real64() {
        const uint64_t bits = scalar<uint64_t>();
        double value = 0;
        static_assert(sizeof(bits) == sizeof(value), "binary64 is required");
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    std::string string() {
        const size_t size = count(1);
        const char * begin = reinterpret_cast<const char *>(current);
        current += size;
        return std::string(begin, size);
    }

    template <typename T, typename Fn>
    std::vector<T> vector(Fn && read_one, size_t minimum_item_bytes = 1) {
        const size_t size = count(minimum_item_bytes);
        std::vector<T> result;
        result.reserve(size);
        for (size_t index = 0; index < size; ++index) {
            result.push_back(read_one());
        }
        return result;
    }

    template <typename T>
    std::vector<T> scalar_vector() {
        return vector<T>([&]() { return scalar<T>(); }, sizeof(T));
    }

    std::vector<uint8_t> byte_vector() {
        const size_t size = count(1);
        std::vector<uint8_t> result(current, current + size);
        current += size;
        return result;
    }

    template <typename T>
    llama_qnn_buffer<T> scalar_buffer() {
        if (!mapped_buffers) {
            return scalar_vector<T>();
        }
        if (!sharded_storages.empty()) {
            const size_t size = external_count();
            const size_t shard = scalar<uint32_t>();
            const uint64_t raw_offset = scalar<uint64_t>();
            if (shard >= sharded_storages.size() ||
                raw_offset > std::numeric_limits<size_t>::max() ||
                size > std::numeric_limits<size_t>::max() / sizeof(T)) {
                binary_fail("sharded scalar buffer descriptor is invalid");
            }
            const size_t offset = static_cast<size_t>(raw_offset);
            const size_t bytes = size * sizeof(T);
            const auto & storage = sharded_storages[shard];
            if (offset > storage->size || bytes > storage->size - offset ||
                offset % alignof(T) != 0) {
                binary_fail("sharded scalar buffer exceeds its sidecar");
            }
            llama_qnn_buffer<T> result;
            result.set_sharded(storage, offset, size);
            return result;
        }
        const size_t size = count(sizeof(T));
        align(64);
        require(size * sizeof(T));
        if (reinterpret_cast<uintptr_t>(current) % alignof(T) != 0) {
            binary_fail("mapped scalar buffer is misaligned");
        }
        llama_qnn_buffer<T> result;
        result.set_mapped(reinterpret_cast<const T *>(current), size);
        current += size * sizeof(T);
        return result;
    }

    llama_qnn_buffer<uint8_t> byte_buffer() {
        if (!mapped_buffers) {
            return byte_vector();
        }
        if (!sharded_storages.empty()) {
            const size_t size = external_count();
            const size_t shard = scalar<uint32_t>();
            const uint64_t raw_offset = scalar<uint64_t>();
            if (shard >= sharded_storages.size() ||
                raw_offset > std::numeric_limits<size_t>::max()) {
                binary_fail("sharded byte buffer descriptor is invalid");
            }
            const size_t offset = static_cast<size_t>(raw_offset);
            const auto & storage = sharded_storages[shard];
            if (offset > storage->size || size > storage->size - offset) {
                binary_fail("sharded byte buffer exceeds its sidecar");
            }
            llama_qnn_buffer<uint8_t> result;
            result.set_sharded(storage, offset, size);
            return result;
        }
        const size_t size = count(1);
        align(64);
        require(size);
        llama_qnn_buffer<uint8_t> result;
        result.set_mapped(current, size);
        current += size;
        return result;
    }

    void expect_end() const {
        if (current != end) {
            binary_fail("payload has trailing bytes");
        }
    }

private:
    void align(size_t alignment) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(current);
        const size_t padding = (alignment - address % alignment) % alignment;
        require(padding);
        for (size_t index = 0; index < padding; ++index) {
            if (current[index] != 0) {
                binary_fail("mapped buffer padding is not zero");
            }
        }
        current += padding;
    }

    void require(size_t size) const {
        if (size > static_cast<size_t>(end - current)) {
            binary_fail("payload is truncated");
        }
    }

    size_t count(size_t minimum_item_bytes) {
        const size_t result = external_count();
        const size_t remaining = static_cast<size_t>(end - current);
        if (result != 0 && minimum_item_bytes > remaining / result) {
            binary_fail("array or string length exceeds the mapped payload");
        }
        return result;
    }

    size_t external_count() {
        const uint64_t raw = scalar<uint64_t>();
        if (raw > std::numeric_limits<size_t>::max()) {
            binary_fail("array length overflows size_t");
        }
        return static_cast<size_t>(raw);
    }

    const uint8_t * current;
    const uint8_t * end;
    bool mapped_buffers;
    std::vector<std::shared_ptr<llama_qnn_buffer_storage>> sharded_storages;
};

void write_affine(binary_writer & writer, const llama_qnn_affine_qparams & value) {
    writer.scalar(value.scale_bits);
    writer.scalar(value.offset);
    writer.scalar(value.zero_point);
}

llama_qnn_affine_qparams read_affine(binary_reader & reader) {
    llama_qnn_affine_qparams value;
    value.scale_bits = reader.scalar<uint32_t>();
    std::memcpy(&value.scale, &value.scale_bits, sizeof(value.scale));
    value.offset = reader.scalar<int32_t>();
    value.zero_point = reader.scalar<int32_t>();
    return value;
}

void write_qparams(binary_writer & writer, const llama_qnn_tensor_qparams & value) {
    writer.scalar<int32_t>(value.encoding);
    writer.scalar(value.axis);
    writer.vector(value.scale_offsets, [&](const auto & item) { write_affine(writer, item); });
    writer.scalar(value.block_scale_bitwidth);
    writer.scalar(value.block_scale_element_bytes);
    writer.scalar(value.num_blocks_per_axis);
    writer.aligned_byte_buffer(value.block_scale_codes);
}

llama_qnn_tensor_qparams read_qparams(binary_reader & reader) {
    llama_qnn_tensor_qparams value;
    const int32_t encoding = reader.scalar<int32_t>();
    if (encoding < LLAMA_QNN_QUANTIZATION_UNDEFINED ||
        encoding > LLAMA_QNN_QUANTIZATION_BLOCKWISE_EXPANSION) {
        binary_fail("qparam encoding is invalid");
    }
    value.encoding = static_cast<llama_qnn_quantization_encoding>(encoding);
    value.axis = reader.scalar<int32_t>();
    value.scale_offsets = reader.vector<llama_qnn_affine_qparams>(
        [&]() { return read_affine(reader); }, 12);
    value.block_scale_bitwidth = reader.scalar<int32_t>();
    value.block_scale_element_bytes = reader.scalar<int32_t>();
    value.num_blocks_per_axis = reader.scalar<int32_t>();
    value.block_scale_codes = reader.byte_buffer();
    return value;
}

void write_use(binary_writer & writer, const llama_qnn_u16_tensor_use & value) {
    writer.string(value.operation_name);
    writer.string(value.role);
    writer.scalar(value.position);
}

llama_qnn_u16_tensor_use read_use(binary_reader & reader) {
    llama_qnn_u16_tensor_use value;
    value.operation_name = reader.string();
    value.role = reader.string();
    value.position = reader.scalar<int32_t>();
    return value;
}

void write_binding(binary_writer & writer, const llama_qnn_decoder_binding & value) {
    writer.scalar_vector(value.layer_ids);
    writer.vector(value.module_paths, [&](const auto & item) { writer.string(item); });
    writer.string(value.projection);
}

llama_qnn_decoder_binding read_binding(binary_reader & reader) {
    llama_qnn_decoder_binding value;
    value.layer_ids = reader.scalar_vector<int32_t>();
    value.module_paths =
        reader.vector<std::string>([&]() { return reader.string(); });
    value.projection = reader.string();
    return value;
}

void write_u16_tensor(binary_writer & writer, const llama_qnn_u16_tensor & value) {
    writer.scalar(value.shard_index);
    writer.string(value.scope);
    writer.string(value.name);
    writer.string(value.data_type);
    writer.scalar_vector(value.dimensions);
    write_qparams(writer, value.qparams);
    writer.aligned_scalar_buffer(value.static_data);
    writer.string(value.static_data_sha256);
    writer.vector(value.operation_uses, [&](const auto & item) { write_use(writer, item); });
    writer.vector(value.decoder_bindings, [&](const auto & item) { write_binding(writer, item); });
}

llama_qnn_u16_tensor read_u16_tensor(binary_reader & reader) {
    llama_qnn_u16_tensor value;
    value.shard_index = reader.scalar<int32_t>();
    value.scope = reader.string();
    value.name = reader.string();
    value.data_type = reader.string();
    value.dimensions = reader.scalar_vector<int64_t>();
    value.qparams = read_qparams(reader);
    value.static_data = reader.scalar_buffer<uint16_t>();
    value.static_data_sha256 = reader.string();
    value.operation_uses =
        reader.vector<llama_qnn_u16_tensor_use>([&]() { return read_use(reader); });
    value.decoder_bindings =
        reader.vector<llama_qnn_decoder_binding>([&]() { return read_binding(reader); });
    return value;
}

void write_aux_tensor(binary_writer & writer, const llama_qnn_aux_quantized_tensor & value) {
    writer.scalar(value.shard_index);
    writer.string(value.scope);
    writer.string(value.name);
    writer.string(value.data_type);
    writer.scalar(value.element_bytes);
    writer.scalar_vector(value.dimensions);
    write_qparams(writer, value.qparams);
    writer.aligned_byte_buffer(value.static_data);
    writer.string(value.static_data_sha256);
    writer.vector(value.operation_uses, [&](const auto & item) { write_use(writer, item); });
    writer.vector(value.decoder_bindings, [&](const auto & item) { write_binding(writer, item); });
}

llama_qnn_aux_quantized_tensor read_aux_tensor(binary_reader & reader) {
    llama_qnn_aux_quantized_tensor value;
    value.shard_index = reader.scalar<int32_t>();
    value.scope = reader.string();
    value.name = reader.string();
    value.data_type = reader.string();
    value.element_bytes = reader.scalar<uint32_t>();
    value.dimensions = reader.scalar_vector<int64_t>();
    value.qparams = read_qparams(reader);
    value.static_data = reader.byte_buffer();
    value.static_data_sha256 = reader.string();
    value.operation_uses =
        reader.vector<llama_qnn_u16_tensor_use>([&]() { return read_use(reader); });
    value.decoder_bindings =
        reader.vector<llama_qnn_decoder_binding>([&]() { return read_binding(reader); });
    return value;
}

void write_linear(binary_writer & writer, const llama_qnn_linear_qparams & value) {
    writer.scalar(value.layer_id);
    writer.scalar(value.shard_index);
    writer.string(value.scope);
    writer.string(value.projection);
    write_affine(writer, value.input);
    write_affine(writer, value.output);
    writer.scalar(value.activation_to_output_q20);
    writer.scalar(value.qnn_weight_block_size);
    writer.scalar(value.qnn_weight_blocks_per_row);
    writer.aligned_scalar_buffer(value.qnn_channel_scale_to_output_q31);
    writer.aligned_byte_buffer(value.qnn_weight_block_scale_codes);
    writer.aligned_scalar_buffer(value.qnn_prepared_weight_sums);
    writer.boolean(value.qnn_weight_block_codes_prepared);
    writer.boolean(value.weights_gs32_source);
    writer.scalar(value.qnn_weight_block_code_layout);
    writer.scalar(value.output_bias_q7);
    writer.vector(value.operation_names, [&](const auto & item) { writer.string(item); });
}

llama_qnn_linear_qparams read_linear(binary_reader & reader) {
    llama_qnn_linear_qparams value;
    value.layer_id = reader.scalar<int32_t>();
    value.shard_index = reader.scalar<int32_t>();
    value.scope = reader.string();
    value.projection = reader.string();
    value.input = read_affine(reader);
    value.output = read_affine(reader);
    value.activation_to_output_q20 = reader.scalar<int64_t>();
    value.qnn_weight_block_size = reader.scalar<int32_t>();
    value.qnn_weight_blocks_per_row = reader.scalar<int32_t>();
    value.qnn_channel_scale_to_output_q31 = reader.scalar_buffer<int64_t>();
    value.qnn_weight_block_scale_codes = reader.byte_buffer();
    value.qnn_prepared_weight_sums = reader.scalar_buffer<int64_t>();
    value.qnn_weight_block_codes_prepared = reader.boolean();
    value.weights_gs32_source = reader.boolean();
    value.qnn_weight_block_code_layout = reader.scalar<int32_t>();
    value.output_bias_q7 = reader.scalar<int32_t>();
    value.operation_names =
        reader.vector<std::string>([&]() { return reader.string(); });
    return value;
}

void write_u16_operand(binary_writer & writer, const llama_qnn_operation_u16_operand & value) {
    writer.string(value.name);
    writer.string(value.role);
    writer.scalar(value.position);
    writer.scalar<uint64_t>(value.tensor_index);
}

llama_qnn_operation_u16_operand read_u16_operand(binary_reader & reader) {
    llama_qnn_operation_u16_operand value;
    value.name = reader.string();
    value.role = reader.string();
    value.position = reader.scalar<int32_t>();
    const uint64_t index = reader.scalar<uint64_t>();
    if (index > std::numeric_limits<size_t>::max()) {
        binary_fail("U16 operand tensor index overflows size_t");
    }
    value.tensor_index = static_cast<size_t>(index);
    return value;
}

void write_aux_operand(binary_writer & writer, const llama_qnn_operation_aux_operand & value) {
    writer.string(value.name);
    writer.string(value.role);
    writer.scalar(value.position);
    writer.scalar<uint64_t>(value.tensor_index);
}

llama_qnn_operation_aux_operand read_aux_operand(binary_reader & reader) {
    llama_qnn_operation_aux_operand value;
    value.name = reader.string();
    value.role = reader.string();
    value.position = reader.scalar<int32_t>();
    const uint64_t index = reader.scalar<uint64_t>();
    if (index > std::numeric_limits<size_t>::max()) {
        binary_fail("aux operand tensor index overflows size_t");
    }
    value.tensor_index = static_cast<size_t>(index);
    return value;
}

void write_operation(binary_writer & writer, const llama_qnn_operation & value) {
    writer.scalar(value.shard_index);
    writer.string(value.scope);
    writer.string(value.name);
    writer.string(value.type_name);
    writer.string(value.fx_node_name);
    write_binding(writer, value.decoder_binding);
    writer.vector(value.inputs, [&](const auto & item) { writer.string(item); });
    writer.vector(value.outputs, [&](const auto & item) { writer.string(item); });
    writer.vector(value.u16_operands, [&](const auto & item) { write_u16_operand(writer, item); });
    writer.vector(value.aux_operands, [&](const auto & item) { write_aux_operand(writer, item); });
    writer.aligned_scalar_buffer(value.input_to_output_q20);
    writer.scalar(value.product_to_output_q20);
    writer.scalar(value.product_to_output_q31);
    writer.scalar(value.product_requant_nudge_q31);
    writer.scalar(value.matmul_product_to_output_q31);
    writer.scalar(value.unary_input_to_output_q20);
    writer.scalar(value.unary_input_to_output_q31);
    writer.scalar(value.a8_add_subtract.coefficients[0]);
    writer.scalar(value.a8_add_subtract.coefficients[1]);
    writer.scalar(value.a8_add_subtract.bias);
    writer.scalar(value.a8_add_subtract.shift);
    writer.scalar<uint8_t>(value.a8_add_subtract.valid ? 1 : 0);
    writer.scalar(value.a8_multiply.q15_coefficient);
    writer.scalar(value.a8_multiply.packed_zero_points);
    writer.scalar(value.a8_multiply.negative_translation);
    writer.scalar(value.a8_multiply.pre_multiply_bias);
    writer.scalar(value.a8_multiply.output_bias);
    writer.scalar(value.a8_multiply.shift);
    writer.scalar<uint8_t>(value.a8_multiply.valid ? 1 : 0);
    writer.aligned_scalar_buffer(value.unary_lut);
    writer.scalar(value.softmax_scale_over_ln2_q24);
    writer.scalar(value.softmax_unit_code);
    writer.aligned_scalar_buffer(value.softmax_exp2_lut_q31);
    writer.real64(value.rms_epsilon);
    writer.scalar(value.rms_epsilon_in_codes_q16);
}

llama_qnn_operation read_operation(binary_reader & reader, uint32_t version) {
    llama_qnn_operation value;
    value.shard_index = reader.scalar<int32_t>();
    value.scope = reader.string();
    value.name = reader.string();
    value.type_name = reader.string();
    value.fx_node_name = reader.string();
    value.decoder_binding = read_binding(reader);
    value.inputs = reader.vector<std::string>([&]() { return reader.string(); });
    value.outputs = reader.vector<std::string>([&]() { return reader.string(); });
    value.u16_operands =
        reader.vector<llama_qnn_operation_u16_operand>([&]() { return read_u16_operand(reader); });
    value.aux_operands =
        reader.vector<llama_qnn_operation_aux_operand>([&]() { return read_aux_operand(reader); });
    value.input_to_output_q20 = reader.scalar_buffer<int64_t>();
    value.product_to_output_q20 = reader.scalar<int64_t>();
    value.product_to_output_q31 = reader.scalar<int64_t>();
    value.product_requant_nudge_q31 = reader.scalar<int64_t>();
    value.matmul_product_to_output_q31 = reader.scalar<int64_t>();
    value.unary_input_to_output_q20 = reader.scalar<int64_t>();
    value.unary_input_to_output_q31 = reader.scalar<int64_t>();
    if (version >= 3) {
        value.a8_add_subtract.coefficients[0] = reader.scalar<int16_t>();
        value.a8_add_subtract.coefficients[1] = reader.scalar<int16_t>();
        value.a8_add_subtract.bias = reader.scalar<int16_t>();
        value.a8_add_subtract.shift = reader.scalar<int32_t>();
        value.a8_add_subtract.valid = reader.scalar<uint8_t>() != 0;
        value.a8_multiply.q15_coefficient = reader.scalar<int16_t>();
        value.a8_multiply.packed_zero_points = reader.scalar<int16_t>();
        value.a8_multiply.negative_translation = reader.scalar<int16_t>();
        value.a8_multiply.pre_multiply_bias = reader.scalar<int16_t>();
        value.a8_multiply.output_bias = reader.scalar<int16_t>();
        value.a8_multiply.shift = reader.scalar<int32_t>();
        value.a8_multiply.valid = reader.scalar<uint8_t>() != 0;
    }
    value.unary_lut = reader.scalar_buffer<uint16_t>();
    value.softmax_scale_over_ln2_q24 = reader.scalar<int64_t>();
    value.softmax_unit_code = reader.scalar<int64_t>();
    value.softmax_exp2_lut_q31 = reader.scalar_buffer<uint32_t>();
    value.rms_epsilon = reader.real64();
    value.rms_epsilon_in_codes_q16 = reader.scalar<uint64_t>();
    return value;
}

void add_shard_payload_size(
        std::vector<size_t> & sizes,
        int32_t requested_shard,
        size_t bytes) {
    const size_t shard = requested_shard < 0
        ? 0
        : static_cast<size_t>(requested_shard);
    if (shard >= sizes.size()) {
        throw std::runtime_error(
            "QNN profile record references an invalid shard " +
            std::to_string(requested_shard));
    }
    const size_t aligned = (sizes[shard] + 63) & ~size_t{63};
    if (bytes > std::numeric_limits<size_t>::max() - aligned) {
        throw std::runtime_error("QNN sidecar shard size overflows size_t");
    }
    sizes[shard] = aligned + bytes;
}

std::vector<size_t> calculate_shard_payload_sizes(
        const llama_qnn_quant_profile & profile,
        size_t shard_count) {
    std::vector<size_t> sizes(shard_count, 0);
    for (const auto & tensor : profile.u16_tensors) {
        add_shard_payload_size(
            sizes, tensor.shard_index, tensor.qparams.block_scale_codes.size());
        add_shard_payload_size(
            sizes,
            tensor.shard_index,
            tensor.static_data.size() * sizeof(uint16_t));
    }
    for (const auto & tensor : profile.aux_quantized_tensors) {
        add_shard_payload_size(
            sizes, tensor.shard_index, tensor.qparams.block_scale_codes.size());
        add_shard_payload_size(
            sizes, tensor.shard_index, tensor.static_data.size());
    }
    for (const auto & linear : profile.linear_qparams) {
        add_shard_payload_size(
            sizes,
            linear.shard_index,
            linear.qnn_channel_scale_to_output_q31.size() * sizeof(int64_t));
        add_shard_payload_size(
            sizes,
            linear.shard_index,
            linear.qnn_weight_block_scale_codes.size());
        add_shard_payload_size(
            sizes,
            linear.shard_index,
            linear.qnn_prepared_weight_sums.size() * sizeof(int64_t));
    }
    for (const auto & operation : profile.operations) {
        add_shard_payload_size(
            sizes,
            operation.shard_index,
            operation.input_to_output_q20.size() * sizeof(int64_t));
        add_shard_payload_size(
            sizes,
            operation.shard_index,
            operation.unary_lut.size() * sizeof(uint16_t));
        add_shard_payload_size(
            sizes,
            operation.shard_index,
            operation.softmax_exp2_lut_q31.size() * sizeof(uint32_t));
    }
    return sizes;
}

std::string sharded_key(int32_t shard_index, const std::string & name) {
    return std::to_string(shard_index) + ':' + name;
}

std::string scoped_key(const std::string & scope, const std::string & name) {
    return std::to_string(scope.size()) + ':' + scope + name;
}

std::string linear_key(int32_t layer_id, const std::string & projection) {
    return std::to_string(layer_id) + ':' + projection;
}

std::string operation_fx_key_local(int32_t layer_id, const std::string & fx_node_name) {
    return std::to_string(layer_id) + ':' + fx_node_name;
}

} // namespace

bool llama_qnn_quant_profile::binary_backing_info(
        const void ** data,
        size_t * size,
        bool * anonymous) const {
    if (!binary_mapping || binary_mapping_is_sharded) {
        return false;
    }
    const auto * mapping =
        static_cast<const mapped_file *>(binary_mapping.get());
    if (data != nullptr) {
        *data = mapping->data;
    }
    if (size != nullptr) {
        *size = mapping->size;
    }
    if (anonymous != nullptr) {
        *anonymous = mapping->anonymous;
    }
    return mapping->data != nullptr && mapping->size != 0;
}

bool llama_qnn_quant_profile::binary_stream_prepare(size_t shard_count) const {
    if (!binary_mapping) {
        return false;
    }
    if (binary_mapping_is_sharded) {
        auto * mapping = static_cast<sharded_file_set *>(binary_mapping.get());
        return mapping->stream_prepare(shard_count);
    }
    auto * mapping = static_cast<mapped_file *>(binary_mapping.get());
    return mapping->stream_prepare(shard_count);
}

bool llama_qnn_quant_profile::binary_stream_fill(
        size_t shard_index,
        size_t * bytes_loaded) const {
    if (!binary_mapping) {
        return false;
    }
    if (binary_mapping_is_sharded) {
        auto * mapping = static_cast<sharded_file_set *>(binary_mapping.get());
        return mapping->stream_fill(shard_index, bytes_loaded);
    }
    auto * mapping = static_cast<mapped_file *>(binary_mapping.get());
    return mapping->stream_fill(shard_index, bytes_loaded);
}

bool llama_qnn_quant_profile::binary_stream_finish() const {
    if (!binary_mapping) {
        return false;
    }
    if (binary_mapping_is_sharded) {
        auto * mapping = static_cast<sharded_file_set *>(binary_mapping.get());
        return mapping->stream_finish();
    }
    auto * mapping = static_cast<mapped_file *>(binary_mapping.get());
    return mapping->stream_finish();
}

bool llama_qnn_quant_profile_is_binary_file(const std::string & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }
    std::array<uint8_t, kMagic.size()> magic {};
    stream.read(reinterpret_cast<char *>(magic.data()), magic.size());
    return stream.gcount() == static_cast<std::streamsize>(magic.size()) && magic == kMagic;
}

void llama_qnn_quant_profile_save_binary_file(
        const llama_qnn_quant_profile & profile,
        const std::string & path) {
    binary_writer writer(path);
    writer.string(profile.quantization_formula);
    writer.scalar(profile.num_decoder_layers);
    writer.scalar(profile.source_weight_bits);
    writer.scalar(profile.source_group_size);
    writer.scalar(profile.activation_bits);
    writer.scalar(profile.activation_code_max);
    writer.string(profile.weight_layout);
    writer.string(profile.lm_head_tensor_name);
    writer.string(profile.lm_head_type);
    writer.string(profile.lm_head_layout);
    writer.scalar(profile.lm_head_input_elements);
    writer.scalar(profile.lm_head_output_rows);
    writer.vector(profile.u16_tensors, [&](const auto & item) { write_u16_tensor(writer, item); });
    writer.vector(
        profile.aux_quantized_tensors,
        [&](const auto & item) { write_aux_tensor(writer, item); });
    writer.vector(profile.linear_qparams, [&](const auto & item) { write_linear(writer, item); });
    writer.vector(profile.operations, [&](const auto & item) { write_operation(writer, item); });
    writer.finish();
}

void llama_qnn_quant_profile_save_sharded_binary_file(
        const llama_qnn_quant_profile & profile,
        const std::string & index_path,
        size_t shard_count) {
    if (shard_count == 0 || shard_count > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("invalid QNN sidecar shard count");
    }
    auto payloads = std::make_shared<shard_payload_writer>(
        index_path,
        calculate_shard_payload_sizes(profile, shard_count));
    binary_writer writer(index_path, kVersion, payloads);
    writer.string(profile.quantization_formula);
    writer.scalar(profile.num_decoder_layers);
    writer.scalar(profile.source_weight_bits);
    writer.scalar(profile.source_group_size);
    writer.scalar(profile.activation_bits);
    writer.scalar(profile.activation_code_max);
    writer.string(profile.weight_layout);
    writer.string(profile.lm_head_tensor_name);
    writer.string(profile.lm_head_type);
    writer.string(profile.lm_head_layout);
    writer.scalar(profile.lm_head_input_elements);
    writer.scalar(profile.lm_head_output_rows);
    writer.scalar<uint32_t>(static_cast<uint32_t>(shard_count));
    for (size_t shard = 0; shard < shard_count; ++shard) {
        writer.scalar<uint64_t>(payloads->file_offset(shard));
        writer.scalar<uint64_t>(payloads->shard_size(shard));
    }
    writer.vector(profile.u16_tensors, [&](const auto & item) {
        writer.set_current_shard(item.shard_index);
        write_u16_tensor(writer, item);
    });
    writer.vector(profile.aux_quantized_tensors, [&](const auto & item) {
        writer.set_current_shard(item.shard_index);
        write_aux_tensor(writer, item);
    });
    writer.vector(profile.linear_qparams, [&](const auto & item) {
        writer.set_current_shard(item.shard_index);
        write_linear(writer, item);
    });
    writer.vector(profile.operations, [&](const auto & item) {
        writer.set_current_shard(item.shard_index);
        write_operation(writer, item);
    });
    writer.finish();
}

std::shared_ptr<llama_qnn_quant_profile>
llama_qnn_quant_profile_load_binary_file(const std::string & path) {
    auto mapping = std::make_shared<mapped_file>(path);
    if (mapping->size < kHeaderBytes ||
        !std::equal(kMagic.begin(), kMagic.end(), mapping->data)) {
        binary_fail("magic is invalid");
    }
    binary_reader header(
        mapping->data + kMagic.size(), mapping->size - kMagic.size());
    const uint32_t version = header.scalar<uint32_t>();
    if (version < 2 || version > kVersion) {
        binary_fail("version is unsupported");
    }
    if (header.scalar<uint32_t>() != kEndianTag) {
        binary_fail("endian tag is invalid");
    }
    const uint64_t payload_bytes = header.scalar<uint64_t>();
    const uint64_t expected_checksum = header.scalar<uint64_t>();
    if (payload_bytes != mapping->size - kHeaderBytes) {
        binary_fail("payload size disagrees with the file size");
    }
    const char * verify_checksum =
        std::getenv("LLAMA_QNN_U16_VERIFY_BINARY_CHECKSUM");
    if (verify_checksum != nullptr && std::strcmp(verify_checksum, "1") == 0) {
        if (payload_checksum(
                mapping->data + kHeaderBytes, static_cast<size_t>(payload_bytes)) !=
            expected_checksum) {
            binary_fail("payload checksum is invalid");
        }
    }

    binary_reader reader(
        mapping->data + kHeaderBytes, static_cast<size_t>(payload_bytes), true);
    auto result = std::make_shared<llama_qnn_quant_profile>();
    result->binary_mapping = mapping;
    result->source_path = path;
    result->quantization_formula = reader.string();
    result->num_decoder_layers = reader.scalar<int32_t>();
    result->source_weight_bits = reader.scalar<int32_t>();
    result->source_group_size = reader.scalar<int32_t>();
    if (version >= 3) {
        result->activation_bits = reader.scalar<int32_t>();
        result->activation_code_max = reader.scalar<uint16_t>();
    }
    result->weight_layout = reader.string();
    result->lm_head_tensor_name = reader.string();
    result->lm_head_type = reader.string();
    result->lm_head_layout = reader.string();
    result->lm_head_input_elements = reader.scalar<int64_t>();
    result->lm_head_output_rows = reader.scalar<int64_t>();
    std::shared_ptr<sharded_file_set> sharded_mapping;
    if (version >= 4) {
        const uint32_t shard_count = reader.scalar<uint32_t>();
        if (shard_count == 0) {
            binary_fail("sharded profile has no sidecars");
        }
        if (version >= 5) {
            std::vector<std::pair<uint64_t, uint64_t>> ranges;
            ranges.reserve(shard_count);
            for (size_t shard = 0; shard < shard_count; ++shard) {
                const uint64_t file_offset = reader.scalar<uint64_t>();
                const uint64_t shard_bytes = reader.scalar<uint64_t>();
                if (file_offset % 4096 != 0) {
                    binary_fail("sidecar shard file offset is not page aligned");
                }
                ranges.emplace_back(file_offset, shard_bytes);
            }
            sharded_mapping = std::make_shared<sharded_file_set>(
                mapping, path, ranges);
        } else {
            sharded_mapping = std::make_shared<sharded_file_set>(
                mapping, path, shard_count);
        }
        reader.set_sharded_buffers(sharded_mapping->storages());
        result->binary_mapping = sharded_mapping;
        result->binary_mapping_is_sharded = true;
    }
    result->u16_tensors =
        reader.vector<llama_qnn_u16_tensor>([&]() { return read_u16_tensor(reader); });
    result->aux_quantized_tensors =
        reader.vector<llama_qnn_aux_quantized_tensor>([&]() { return read_aux_tensor(reader); });
    result->linear_qparams =
        reader.vector<llama_qnn_linear_qparams>([&]() { return read_linear(reader); });
    result->operations =
        reader.vector<llama_qnn_operation>([&]() { return read_operation(reader, version); });
    reader.expect_end();

    if (version < 4) {
        result->binary_mapping = mapping;
    }

    std::unordered_set<std::string> ambiguous_u16_names;
    for (size_t index = 0; index < result->u16_tensors.size(); ++index) {
        const auto & tensor = result->u16_tensors[index];
        if (!result->u16_tensor_shard_index.emplace(
                sharded_key(tensor.shard_index, tensor.name), index).second ||
            !result->u16_tensor_scope_index.emplace(
                scoped_key(tensor.scope, tensor.name), index).second) {
            binary_fail("duplicate U16 tensor identity");
        }
        if (ambiguous_u16_names.count(tensor.name) == 0) {
            const auto inserted = result->u16_tensor_index.emplace(tensor.name, index);
            if (!inserted.second) {
                result->u16_tensor_index.erase(inserted.first);
                ambiguous_u16_names.insert(tensor.name);
            }
        }
    }

    std::unordered_set<std::string> ambiguous_aux_names;
    for (size_t index = 0; index < result->aux_quantized_tensors.size(); ++index) {
        const auto & tensor = result->aux_quantized_tensors[index];
        if (!result->aux_tensor_shard_index.emplace(
                sharded_key(tensor.shard_index, tensor.name), index).second ||
            !result->aux_tensor_scope_index.emplace(
                scoped_key(tensor.scope, tensor.name), index).second) {
            binary_fail("duplicate auxiliary tensor identity");
        }
        if (ambiguous_aux_names.count(tensor.name) == 0) {
            const auto inserted = result->aux_tensor_index.emplace(tensor.name, index);
            if (!inserted.second) {
                result->aux_tensor_index.erase(inserted.first);
                ambiguous_aux_names.insert(tensor.name);
            }
        }
    }

    for (size_t index = 0; index < result->linear_qparams.size(); ++index) {
        const auto & linear = result->linear_qparams[index];
        if (!result->linear_qparams_index.emplace(
                linear_key(linear.layer_id, linear.projection), index).second) {
            binary_fail("duplicate linear qparams identity");
        }
    }

    for (size_t index = 0; index < result->operations.size(); ++index) {
        const auto & operation = result->operations[index];
        if (!result->operation_shard_index.emplace(
                sharded_key(operation.shard_index, operation.name), index).second) {
            binary_fail("duplicate operation identity");
        }
        for (const auto & output : operation.outputs) {
            if (!result->operation_output_shard_index.emplace(
                    sharded_key(operation.shard_index, output), index).second) {
                binary_fail("duplicate operation output producer");
            }
        }
        if (operation.type_name == "Convert" && operation.inputs.size() == 1) {
            // Match the JSON loader and the historical first matching scan.
            result->operation_convert_input_shard_index.emplace(
                sharded_key(operation.shard_index, operation.inputs.front()), index);
        }
        if (operation.decoder_binding.layer_ids.size() == 1 &&
            !result->operation_fx_index.emplace(
                operation_fx_key_local(
                    operation.decoder_binding.layer_ids.front(), operation.fx_node_name),
                index).second) {
            binary_fail("duplicate decoder FX operation identity");
        }
        for (const auto & operand : operation.u16_operands) {
            if (operand.tensor_index >= result->u16_tensors.size()) {
                binary_fail("U16 operand tensor index is invalid");
            }
        }
        for (const auto & operand : operation.aux_operands) {
            if (operand.tensor_index >= result->aux_quantized_tensors.size()) {
                binary_fail("aux operand tensor index is invalid");
            }
        }
    }

    if (result->quantization_formula != "real=(integer_code+offset)*scale" ||
        result->num_decoder_layers <= 0 ||
        result->source_weight_bits != 2 ||
        result->source_group_size < 32 ||
        result->u16_tensors.empty() ||
        result->linear_qparams.size() !=
            static_cast<size_t>(result->num_decoder_layers) * 7) {
        binary_fail("runtime contract is incomplete");
    }
    return result;
}
