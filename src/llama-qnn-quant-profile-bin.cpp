#include "llama-qnn-quant-profile.h"

#include <array>
#include <cerrno>
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
constexpr uint32_t kVersion = 2;
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

class binary_writer {
public:
    explicit binary_writer(const std::string & path) : stream(path, std::ios::binary) {
        if (!stream) {
            throw std::runtime_error("cannot create binary QNN quantization profile: " + path);
        }
        bytes(kMagic.data(), kMagic.size());
        scalar(kVersion);
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
        align(64);
        for (const T value : values) {
            scalar<T>(value);
        }
    }

    template <typename Container>
    void aligned_byte_buffer(const Container & values) {
        scalar<uint64_t>(values.size());
        align(64);
        bytes(values.data(), values.size());
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
    }

private:
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
        data = static_cast<const uint8_t *>(
            mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (data == MAP_FAILED) {
            data = nullptr;
            const std::string error = std::strerror(errno);
            close(fd);
            fd = -1;
            throw std::runtime_error(
                "cannot mmap binary QNN quantization profile: " + path + ": " + error);
        }
        madvise(const_cast<uint8_t *>(data), size, MADV_SEQUENTIAL);
    }

    ~mapped_file() {
        if (data != nullptr) {
            madvise(const_cast<uint8_t *>(data), size, MADV_DONTNEED);
            munmap(const_cast<uint8_t *>(data), size);
        }
        if (fd >= 0) {
            close(fd);
        }
    }

    mapped_file(const mapped_file &) = delete;
    mapped_file & operator=(const mapped_file &) = delete;

    const uint8_t * data = nullptr;
    size_t size = 0;

private:
    int fd = -1;
};

class binary_reader {
public:
    binary_reader(
            const uint8_t * data,
            size_t size,
            bool mapped_buffers = false) :
        current(data), end(data + size), mapped_buffers(mapped_buffers) {}

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
        const uint64_t raw = scalar<uint64_t>();
        const size_t remaining = static_cast<size_t>(end - current);
        if (raw > std::numeric_limits<size_t>::max() ||
            (raw != 0 && minimum_item_bytes > remaining / static_cast<size_t>(raw))) {
            binary_fail("array or string length exceeds the mapped payload");
        }
        return static_cast<size_t>(raw);
    }

    const uint8_t * current;
    const uint8_t * end;
    bool mapped_buffers;
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
    writer.aligned_scalar_buffer(value.unary_lut);
    writer.scalar(value.softmax_scale_over_ln2_q24);
    writer.scalar(value.softmax_unit_code);
    writer.aligned_scalar_buffer(value.softmax_exp2_lut_q31);
    writer.real64(value.rms_epsilon);
    writer.scalar(value.rms_epsilon_in_codes_q16);
}

llama_qnn_operation read_operation(binary_reader & reader) {
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
    value.unary_lut = reader.scalar_buffer<uint16_t>();
    value.softmax_scale_over_ln2_q24 = reader.scalar<int64_t>();
    value.softmax_unit_code = reader.scalar<int64_t>();
    value.softmax_exp2_lut_q31 = reader.scalar_buffer<uint32_t>();
    value.rms_epsilon = reader.real64();
    value.rms_epsilon_in_codes_q16 = reader.scalar<uint64_t>();
    return value;
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

std::shared_ptr<llama_qnn_quant_profile>
llama_qnn_quant_profile_load_binary_file(const std::string & path) {
    auto mapping = std::make_shared<mapped_file>(path);
    if (mapping->size < kHeaderBytes ||
        !std::equal(kMagic.begin(), kMagic.end(), mapping->data)) {
        binary_fail("magic is invalid");
    }
    binary_reader header(
        mapping->data + kMagic.size(), mapping->size - kMagic.size());
    if (header.scalar<uint32_t>() != kVersion) {
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
    result->weight_layout = reader.string();
    result->lm_head_tensor_name = reader.string();
    result->lm_head_type = reader.string();
    result->lm_head_layout = reader.string();
    result->lm_head_input_elements = reader.scalar<int64_t>();
    result->lm_head_output_rows = reader.scalar<int64_t>();
    result->u16_tensors =
        reader.vector<llama_qnn_u16_tensor>([&]() { return read_u16_tensor(reader); });
    result->aux_quantized_tensors =
        reader.vector<llama_qnn_aux_quantized_tensor>([&]() { return read_aux_tensor(reader); });
    result->linear_qparams =
        reader.vector<llama_qnn_linear_qparams>([&]() { return read_linear(reader); });
    result->operations =
        reader.vector<llama_qnn_operation>([&]() { return read_operation(reader); });
    reader.expect_end();

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
