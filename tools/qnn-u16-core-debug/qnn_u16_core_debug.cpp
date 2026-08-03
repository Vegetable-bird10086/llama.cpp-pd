#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstring>

extern "C" {
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include "gguf.h"
}
#include "ggml-cpu/repack.h"
#include "llama-qnn-quant-profile.h"
#include "llama-qnn-u16.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <json.hpp>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define QNN_U16_HAVE_NEON 1
#else
#define QNN_U16_HAVE_NEON 0
#endif

namespace {

constexpr int kRequantShift = 20;

struct affine_u16 {
    double scale;
    int32_t zero_point;
};

struct fixed_multiplier {
    int64_t value;
    int shift;
};

using json = nlohmann::json;

void repack_lm_head_q8_0(
        const block_q8_0 * source,
        block_q8_0x4 * destination,
        int64_t rows,
        int64_t columns,
        int interleave) {
    const int64_t blocks = columns / QK8_0;
    for (int64_t row = 0; row < rows; row += 4) {
        for (int64_t block = 0; block < blocks; ++block) {
            block_q8_0x4 & output = *destination++;
            for (int lane = 0; lane < 4; ++lane) {
                output.d[lane] = source[(row + lane) * blocks + block].d;
            }
            const int chunks = QK8_0 / interleave;
            for (int chunk = 0; chunk < chunks; ++chunk) {
                for (int lane = 0; lane < 4; ++lane) {
                    const block_q8_0 & input =
                        source[(row + lane) * blocks + block];
                    std::memcpy(
                        output.qs + (chunk * 4 + lane) * interleave,
                        input.qs + chunk * interleave,
                        interleave);
                }
            }
        }
    }
}

void repack_lm_head_q6_k(
        const block_q6_K * source,
        block_q6_Kx8 * destination,
        int64_t rows,
        int64_t columns,
        int interleave) {
    const int64_t blocks = columns / QK_K;
    for (int64_t row = 0; row < rows; row += 8) {
        for (int64_t block = 0; block < blocks; ++block) {
            block_q6_Kx8 & output = *destination++;
            for (int lane = 0; lane < 8; ++lane) {
                output.d[lane] = source[(row + lane) * blocks + block].d;
            }
            for (size_t offset = 0; offset < sizeof(block_q6_K::ql);
                 offset += interleave) {
                for (int lane = 0; lane < 8; ++lane) {
                    const block_q6_K & input =
                        source[(row + lane) * blocks + block];
                    std::memcpy(
                        output.ql +
                            (offset / interleave * 8 + lane) * interleave,
                        input.ql + offset, interleave);
                }
            }
            for (size_t offset = 0; offset < sizeof(block_q6_K::qh);
                 offset += interleave) {
                for (int lane = 0; lane < 8; ++lane) {
                    const block_q6_K & input =
                        source[(row + lane) * blocks + block];
                    std::memcpy(
                        output.qh +
                            (offset / interleave * 8 + lane) * interleave,
                        input.qh + offset, interleave);
                }
            }
            for (int scale = 0; scale < QK_K / 16; ++scale) {
                for (int lane = 0; lane < 8; ++lane) {
                    output.scales[scale * 8 + lane] =
                        source[(row + lane) * blocks + block].scales[scale];
                }
            }
        }
    }
}

int run_lm_head_gemv_test(ggml_type weight_type, int n_threads, int64_t rows) {
    constexpr int64_t columns = 2048;
    constexpr int iterations = 4;
    if ((weight_type != GGML_TYPE_Q8_0 && weight_type != GGML_TYPE_Q6_K) ||
        n_threads <= 0 || rows <= 0 || rows % 8 != 0) {
        return 2;
    }

    ggml_init_params weight_params {
        /* .mem_size = */ ggml_tensor_overhead() * 8 +
            ggml_graph_overhead_custom(8, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(weight_params);
    if (ctx == nullptr) {
        return 3;
    }
    ggml_tensor * weights =
        ggml_new_tensor_2d(ctx, weight_type, columns, rows);
    ggml_set_name(weights, "output.weight");
    ggml_backend_buffer_t weight_buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(
            ctx, ggml_backend_cpu_repack_buffer_type());
    if (weight_buffer == nullptr) {
        ggml_free(ctx);
        return 4;
    }

    std::vector<uint8_t> raw_weights(ggml_nbytes(weights));
    uint32_t random_state = 0x6d2b79f5u;
    auto next_random = [&random_state]() {
        random_state ^= random_state << 13;
        random_state ^= random_state >> 17;
        random_state ^= random_state << 5;
        return random_state;
    };
    if (weight_type == GGML_TYPE_Q8_0) {
        auto * blocks = reinterpret_cast<block_q8_0 *>(raw_weights.data());
        const size_t count = raw_weights.size() / sizeof(block_q8_0);
        for (size_t index = 0; index < count; ++index) {
            blocks[index].d = GGML_FP32_TO_FP16(1.0f / 127.0f);
            std::memset(
                blocks[index].qs,
                static_cast<int>(index * 37u + 11u), QK8_0);
        }
    } else {
        auto * blocks = reinterpret_cast<block_q6_K *>(raw_weights.data());
        const size_t count = raw_weights.size() / sizeof(block_q6_K);
        for (size_t index = 0; index < count; ++index) {
            blocks[index].d = GGML_FP32_TO_FP16(1.0f / 32.0f);
            std::memset(
                blocks[index].ql,
                static_cast<int>(index * 29u + 7u),
                sizeof(blocks[index].ql));
            std::memset(
                blocks[index].qh,
                static_cast<int>(index * 43u + 3u),
                sizeof(blocks[index].qh));
            for (size_t lane = 0; lane < sizeof(blocks[index].scales); ++lane) {
                blocks[index].scales[lane] =
                    static_cast<int8_t>(
                        ((index + lane * 5u) % 31u) - 15);
            }
        }
    }
    const int interleave = ggml_cpu_has_matmul_int8() ? 8 : 4;
    if (weight_type == GGML_TYPE_Q8_0) {
        repack_lm_head_q8_0(
            reinterpret_cast<const block_q8_0 *>(raw_weights.data()),
            reinterpret_cast<block_q8_0x4 *>(weights->data),
            rows, columns, interleave);
    } else {
        repack_lm_head_q6_k(
            reinterpret_cast<const block_q6_K *>(raw_weights.data()),
            reinterpret_cast<block_q6_Kx8 *>(weights->data),
            rows, columns, interleave);
    }
    raw_weights.clear();
    raw_weights.shrink_to_fit();

    ggml_tensor * input =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, 1);
    ggml_set_name(input, "lm_head_input");
    ggml_tensor * output = ggml_mul_mat(ctx, weights, input);
    ggml_set_name(output, "lm_head_output");
    ggml_backend_buffer_t compute_buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(
            ctx, ggml_backend_cpu_buffer_type());
    if (compute_buffer == nullptr) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        return 5;
    }

    std::vector<float> input_data(columns);
    for (int64_t index = 0; index < columns; ++index) {
        input_data[index] =
            static_cast<float>(static_cast<int>(next_random() % 2001u) - 1000) /
            1000.0f;
    }
    ggml_backend_tensor_set(
        input, input_data.data(), 0, input_data.size() * sizeof(float));

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_threadpool_params threadpool_params =
        ggml_threadpool_params_default(n_threads);
    ggml_threadpool * threadpool = ggml_threadpool_new(&threadpool_params);
    if (threadpool == nullptr) {
        ggml_backend_free(backend);
        ggml_backend_buffer_free(compute_buffer);
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        return 6;
    }
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    ggml_backend_cpu_set_threadpool(backend, threadpool);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, output);

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_free(backend);
        ggml_threadpool_free(threadpool);
        ggml_backend_buffer_free(compute_buffer);
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        return 7;
    }

    std::array<double, iterations> elapsed_ms {};
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        const auto end = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_free(backend);
            ggml_threadpool_free(threadpool);
            ggml_backend_buffer_free(compute_buffer);
            ggml_backend_buffer_free(weight_buffer);
            ggml_free(ctx);
            return 8;
        }
        elapsed_ms[iteration] =
            std::chrono::duration<double, std::milli>(end - start).count();
    }
    std::vector<float> output_data(rows);
    ggml_backend_tensor_get(
        output, output_data.data(), 0, output_data.size() * sizeof(float));
    double checksum = 0.0;
    for (int64_t index = 0; index < rows; index += 997) {
        checksum += output_data[index];
    }
    double average_ms = 0.0;
    for (double value : elapsed_ms) {
        average_ms += value;
    }
    average_ms /= iterations;
    std::printf(
        "lm-head-gemv: type=%s rows=%" PRId64 " columns=%" PRId64
        " threads=%d iterations=%d times_ms=%.6f,%.6f,%.6f,%.6f"
        " average_ms=%.6f checksum=%.9f dotprod=%d i8mm=%d\n",
        ggml_type_name(weight_type), rows, columns, n_threads, iterations,
        elapsed_ms[0], elapsed_ms[1], elapsed_ms[2], elapsed_ms[3],
        average_ms, checksum, ggml_cpu_has_dotprod(), ggml_cpu_has_matmul_int8());

    ggml_backend_free(backend);
    ggml_threadpool_free(threadpool);
    ggml_backend_buffer_free(compute_buffer);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);
    return 0;
}

struct qnn_profile_inspection {
    size_t tensor_count = 0;
    size_t operation_count = 0;
    size_t qnn_parameter_count = 0;
    size_t blockwise_weight_count = 0;
    size_t embedded_static_tensor_count = 0;
    size_t embedded_static_bytes = 0;
    size_t block_scale_bytes = 0;
    size_t logical_tensor_count = 0;
    size_t logical_tensor_with_decoder_module_count = 0;
    std::map<std::string, size_t> operation_types;
};

std::vector<uint8_t> decode_base64(std::string_view encoded) {
    static const std::array<int8_t, 256> kDecodeTable = []() {
        std::array<int8_t, 256> table {};
        table.fill(-1);
        for (int index = 0; index < 26; ++index) {
            table[static_cast<uint8_t>('A' + index)] = index;
            table[static_cast<uint8_t>('a' + index)] = 26 + index;
        }
        for (int index = 0; index < 10; ++index) {
            table[static_cast<uint8_t>('0' + index)] = 52 + index;
        }
        table[static_cast<uint8_t>('+')] = 62;
        table[static_cast<uint8_t>('/')] = 63;
        return table;
    }();
    if (encoded.size() % 4 != 0) {
        throw std::runtime_error("base64 payload has an invalid length");
    }
    std::vector<uint8_t> decoded;
    decoded.reserve(encoded.size() / 4 * 3);
    for (size_t offset = 0; offset < encoded.size(); offset += 4) {
        uint32_t accumulator = 0;
        int padding = 0;
        for (size_t index = 0; index < 4; ++index) {
            const char character = encoded[offset + index];
            if (character == '=') {
                if (index < 2 || offset + 4 != encoded.size()) {
                    throw std::runtime_error("base64 payload has invalid padding");
                }
                ++padding;
                accumulator <<= 6;
                continue;
            }
            if (padding != 0) {
                throw std::runtime_error("base64 payload has data after padding");
            }
            const int8_t value = kDecodeTable[static_cast<uint8_t>(character)];
            if (value < 0) {
                throw std::runtime_error("base64 payload has an invalid character");
            }
            accumulator = (accumulator << 6) | static_cast<uint8_t>(value);
        }
        decoded.push_back(static_cast<uint8_t>(accumulator >> 16));
        if (padding < 2) {
            decoded.push_back(static_cast<uint8_t>(accumulator >> 8));
        }
        if (padding == 0) {
            decoded.push_back(static_cast<uint8_t>(accumulator));
        }
    }
    return decoded;
}

void require_exact_scale_offset(const json & value, const std::string & location) {
    if (!value.is_object() || !value.contains("scale_f32_le_hex") ||
        !value.contains("scale") || !value.contains("offset") ||
        !value.contains("zero_point")) {
        throw std::runtime_error(location + " lacks exact scale/offset qparams");
    }
    const std::string raw_scale = value.at("scale_f32_le_hex").get<std::string>();
    if (raw_scale.size() != 8 || raw_scale.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) {
        throw std::runtime_error(location + " lacks four raw f32 scale bytes");
    }
    if (value.at("zero_point").get<int64_t>() != -value.at("offset").get<int64_t>()) {
        throw std::runtime_error(location + " has inconsistent offset and zero point");
    }
}

void require_profile_capabilities(const json & capabilities, const std::string & location) {
    static constexpr std::array<const char *, 7> kRequired = {
        "exact_f32_scale_bits",
        "raw_operator_parameters",
        "operation_tensor_sources",
        "embedded_static_affine_tensor_bytes",
        "blockwise_weight_scale_payload",
        "blockwise_weight_payload_digest",
        "structured_decoder_tensor_bindings",
    };
    if (!capabilities.is_object()) {
        throw std::runtime_error(location + " has no exact-decode capabilities");
    }
    for (const char * capability : kRequired) {
        if (!capabilities.value(capability, false)) {
            throw std::runtime_error(location + " lacks capability " + capability);
        }
    }
}

void require_decoder_binding(
        const json & source,
        const std::string & location,
        bool require_module_path) {
    if (!source.is_object() || !source.contains("fx_node_name") ||
        !source.at("fx_node_name").is_string() ||
        source.at("fx_node_name").get<std::string>().empty()) {
        throw std::runtime_error(location + " lacks an FX node binding");
    }
    const auto & binding = source.at("decoder_binding");
    if (!binding.is_object() || !binding.contains("module_paths") ||
        !binding.contains("layer_ids") || !binding.contains("projection")) {
        throw std::runtime_error(location + " lacks a structured decoder binding");
    }
    const auto & module_paths = binding.at("module_paths");
    const auto & layer_ids = binding.at("layer_ids");
    if (!module_paths.is_array() || !layer_ids.is_array()) {
        throw std::runtime_error(location + " has malformed decoder binding arrays");
    }
    for (const auto & module_path : module_paths) {
        if (!module_path.is_string() || module_path.get<std::string>().empty()) {
            throw std::runtime_error(location + " has an invalid decoder module path");
        }
    }
    if (require_module_path && module_paths.empty()) {
        throw std::runtime_error(location + " lacks a decoder module path");
    }
    for (const auto & layer_id : layer_ids) {
        if (!layer_id.is_number_integer() || layer_id.get<int64_t>() < 0) {
            throw std::runtime_error(location + " has an invalid decoder layer ID");
        }
    }
    if (!binding.at("projection").is_null() &&
        (!binding.at("projection").is_string() ||
         binding.at("projection").get<std::string>().empty())) {
        throw std::runtime_error(location + " has an invalid decoder projection");
    }
}

void inspect_qnn_tensor(
        const std::string & name,
        const json & tensor,
        qnn_profile_inspection * inspection) {
    if (!tensor.is_object() || tensor.value("name", "") != name || tensor.value("version", 0) != 2) {
        throw std::runtime_error("invalid QNN tensor profile: " + name);
    }
    const std::string encoding = tensor.value("quantization_encoding", "");
    if (encoding == "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET") {
        require_exact_scale_offset(tensor.at("scale_offset"), "tensor " + name);
    } else if (encoding == "QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET") {
        const auto & axis_payload = tensor.at("axis_scale_offset");
        const auto & scale_offsets = axis_payload.at("scale_offsets");
        if (!scale_offsets.is_array() || scale_offsets.empty()) {
            throw std::runtime_error("axis qparams are empty for tensor " + name);
        }
        for (size_t index = 0; index < scale_offsets.size(); ++index) {
            require_exact_scale_offset(scale_offsets[index], "axis qparams " + name);
        }
    } else if (encoding == "QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION") {
        const auto & blockwise = tensor.at("blockwise_expansion");
        const auto & scale_offsets = blockwise.at("scale_offsets");
        if (!scale_offsets.is_array() || scale_offsets.empty()) {
            throw std::runtime_error("blockwise qparams are empty for tensor " + name);
        }
        for (size_t index = 0; index < scale_offsets.size(); ++index) {
            require_exact_scale_offset(scale_offsets[index], "blockwise qparams " + name);
        }
        const std::vector<uint8_t> block_codes = decode_base64(
            blockwise.at("block_scales_base64").get<std::string>());
        if (block_codes.size() != blockwise.at("block_scales_bytes").get<size_t>()) {
            throw std::runtime_error("block-scale byte count mismatch for tensor " + name);
        }
        ++inspection->blockwise_weight_count;
        inspection->block_scale_bytes += block_codes.size();
    } else if (encoding != "QNN_QUANTIZATION_ENCODING_UNDEFINED") {
        throw std::runtime_error("unsupported QNN qparam encoding for tensor " + name);
    }

    if (tensor.value("tensor_type", "") != "QNN_TENSOR_TYPE_STATIC") {
        return;
    }
    const auto & static_payload = tensor.at("static_payload");
    const std::string storage = static_payload.at("storage").get<std::string>();
    if (storage == "external_gptq_int2_source_reconstruction") {
        if (static_payload.at("qnn_payload_bytes").get<size_t>() == 0 ||
            static_payload.at("qnn_payload_sha256").get<std::string>().size() != 64) {
            throw std::runtime_error("incomplete GPTQ source contract for tensor " + name);
        }
    } else if (storage == "embedded_exact_bytes") {
        const std::vector<uint8_t> raw = decode_base64(
            static_payload.at("data_le_base64").get<std::string>());
        if (raw.size() != static_payload.at("data_bytes").get<size_t>() ||
            raw.empty() || static_payload.at("sha256").get<std::string>().size() != 64) {
            throw std::runtime_error("embedded static payload mismatch for tensor " + name);
        }
        ++inspection->embedded_static_tensor_count;
        inspection->embedded_static_bytes += raw.size();
    } else {
        throw std::runtime_error("unsupported static storage contract for tensor " + name);
    }
}

void inspect_qnn_parameters(const json & operation, qnn_profile_inspection * inspection) {
    const auto & parameters = operation.at("params");
    if (!parameters.is_array()) {
        throw std::runtime_error("QNN operation parameter list is invalid");
    }
    for (const auto & parameter : parameters) {
        const std::string type = parameter.at("param_type").get<std::string>();
        if (type == "QNN_PARAMTYPE_SCALAR") {
            const auto & scalar = parameter.at("scalar");
            if (decode_base64(scalar.at("value_le_base64").get<std::string>()).size() !=
                scalar.at("value_bytes").get<size_t>()) {
                throw std::runtime_error("QNN scalar parameter byte count mismatch");
            }
        } else if (type == "QNN_PARAMTYPE_TENSOR") {
            const auto & tensor = parameter.at("tensor");
            inspect_qnn_tensor(tensor.at("name").get<std::string>(), tensor, inspection);
            if (decode_base64(parameter.at("tensor_data_base64").get<std::string>()).size() !=
                parameter.at("tensor_data_bytes").get<size_t>()) {
                throw std::runtime_error("QNN tensor parameter byte count mismatch");
            }
        } else if (type == "QNN_PARAMTYPE_UNSUPPORTED_SCALAR") {
            const auto & unsupported = parameter.at("unsupported_scalar");
            if (!unsupported.is_object() ||
                unsupported.value("data_type", "").empty() ||
                unsupported.value("reason", "").empty()) {
                throw std::runtime_error("invalid unsupported QNN scalar parameter metadata");
            }
        } else if (type == "QNN_PARAMTYPE_UNSUPPORTED_TENSOR") {
            const auto & unsupported = parameter.at("unsupported_tensor");
            if (!unsupported.is_object() ||
                !unsupported.value("unsupported_version", false)) {
                throw std::runtime_error("invalid unsupported QNN tensor parameter metadata");
            }
        } else {
            throw std::runtime_error("unsupported QNN operation parameter type");
        }
        ++inspection->qnn_parameter_count;
    }
}

int run_profile_inspect(const char * manifest_path, std::optional<size_t> selected_shard) {
    try {
        const auto runtime_profile = llama_qnn_quant_profile_load_file(manifest_path);
        std::printf(
            "qnn-profile-runtime-load: u16_tensors=%zu linear_pairs=%zu source_bits=%d group_size=%d manifest=%s\n",
            runtime_profile->u16_tensor_count(),
            runtime_profile->linear_qparams_count(),
            runtime_profile->source_weight_bits,
            runtime_profile->source_group_size,
            runtime_profile->source_path.c_str());
        std::unordered_map<std::string, size_t> runtime_name_counts;
        for (const auto & tensor : runtime_profile->u16_tensors) {
            ++runtime_name_counts[tensor.name];
            if (runtime_profile->find_u16_tensor(tensor.shard_index, tensor.name) != &tensor ||
                runtime_profile->find_u16_tensor(tensor.scope, tensor.name) != &tensor) {
                throw std::runtime_error("runtime QNN profile has a broken shard/scope tensor index");
            }
        }
        for (const auto & tensor : runtime_profile->u16_tensors) {
            const auto * unqualified = runtime_profile->find_u16_tensor(tensor.name);
            if ((runtime_name_counts.at(tensor.name) == 1 && unqualified != &tensor) ||
                (runtime_name_counts.at(tensor.name) > 1 && unqualified != nullptr)) {
                throw std::runtime_error("runtime QNN profile has an unsafe unqualified tensor index");
            }
        }
        std::ifstream stream(manifest_path);
        if (!stream) {
            throw std::runtime_error("cannot open manifest");
        }
        const json manifest = json::parse(stream);
        const auto & profile = manifest.at("graphs").at("prefill_forward").at("llama_qnn_quant_profile");
        if (profile.value("schema_version", 0) != 2 ||
            profile.value("format", "") != "llama-qnn-quant-profile-v2") {
            throw std::runtime_error("manifest does not contain a v2 QNN quantization profile");
        }
        require_profile_capabilities(profile.at("capabilities"), "QNN profile");
        const auto & recipe = profile.at("gptq_source_recipe");
        const size_t group_size = recipe.at("group_size").get<size_t>();
        const auto & contract = recipe.at("qnn_weight_code_contract");
        if (recipe.at("source_weight_bits").get<int>() != 2 || group_size == 0 ||
            group_size % 32 != 0 ||
            contract.at("source_group_size").get<size_t>() != group_size ||
            contract.at("source_group_code_bytes").get<size_t>() != group_size / 4 ||
            contract.at("source_code_packing").get<std::string>() !=
                "four_int2_codes_per_byte_lsb_first") {
            throw std::runtime_error("manifest has an incompatible GPTQ INT2 source contract");
        }

        const auto & shards = profile.at("shards");
        if (!shards.is_array() || shards.empty()) {
            throw std::runtime_error("QNN quantization profile has no shards");
        }
        qnn_profile_inspection inspection;
        for (size_t shard_index = 0; shard_index < shards.size(); ++shard_index) {
            if (selected_shard.has_value() && selected_shard.value() != shard_index) {
                continue;
            }
            const auto & shard = shards[shard_index];
            require_profile_capabilities(shard.at("capabilities"), "QNN profile shard");
            const auto & tensors = shard.at("tensors");
            const auto & operations = shard.at("operations");
            const auto & logical_tensors = shard.at("logical_tensors");
            if (!tensors.is_object() || !operations.is_array() || !logical_tensors.is_object()) {
                throw std::runtime_error("QNN shard has invalid tensor or operation records");
            }
            for (auto iterator = tensors.begin(); iterator != tensors.end(); ++iterator) {
                inspect_qnn_tensor(iterator.key(), iterator.value(), &inspection);
                const auto & tensor = iterator.value();
                if (tensor.value("tensor_type", "") == "QNN_TENSOR_TYPE_STATIC" &&
                    tensor.value("quantization_encoding", "") ==
                        "QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION") {
                    const auto source = logical_tensors.find(iterator.key());
                    if (source != logical_tensors.end() &&
                        !source.value().at("decoder_binding").at("module_paths").empty()) {
                        require_decoder_binding(
                            source.value(),
                            "static QNN blockwise tensor " + iterator.key(),
                            true);
                    }
                }
                ++inspection.tensor_count;
            }
            for (auto iterator = logical_tensors.begin(); iterator != logical_tensors.end(); ++iterator) {
                if (tensors.find(iterator.key()) == tensors.end()) {
                    throw std::runtime_error("logical QNN tensor is outside the QNN tensor ABI");
                }
                require_decoder_binding(
                    iterator.value(), "logical QNN tensor " + iterator.key(), false);
                ++inspection.logical_tensor_count;
                if (!iterator.value().at("decoder_binding").at("module_paths").empty()) {
                    ++inspection.logical_tensor_with_decoder_module_count;
                }
            }
            for (const auto & operation : operations) {
                const std::string type_name = operation.at("type_name").get<std::string>();
                for (const char * field_name : {"input_sources", "output_sources"}) {
                    const auto & bindings = operation.at(field_name);
                    if (!bindings.is_object()) {
                        throw std::runtime_error("QNN operation has invalid tensor source bindings");
                    }
                    for (auto iterator = bindings.begin(); iterator != bindings.end(); ++iterator) {
                        require_decoder_binding(
                            iterator.value(),
                            "QNN operation tensor " + iterator.key(),
                            false);
                    }
                }
                inspect_qnn_parameters(operation, &inspection);
                ++inspection.operation_types[type_name];
                ++inspection.operation_count;
            }
        }
        if (inspection.tensor_count == 0) {
            throw std::runtime_error("selected QNN shard is absent");
        }
        std::printf(
            "qnn-profile-inspect: source_bits=2 group_size=%zu tensors=%zu operations=%zu "
            "params=%zu blockwise_weights=%zu block_scale_bytes=%zu "
            "embedded_static_tensors=%zu embedded_static_bytes=%zu "
            "logical_tensors=%zu logical_with_module=%zu\n",
            group_size,
            inspection.tensor_count,
            inspection.operation_count,
            inspection.qnn_parameter_count,
            inspection.blockwise_weight_count,
            inspection.block_scale_bytes,
            inspection.embedded_static_tensor_count,
            inspection.embedded_static_bytes,
            inspection.logical_tensor_count,
            inspection.logical_tensor_with_decoder_module_count);
        for (const auto & [type_name, count] : inspection.operation_types) {
            std::printf("qnn-profile-inspect-op: type=%s count=%zu\n", type_name.c_str(), count);
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "qnn-profile-inspect failed: %s\n", error.what());
        return 12;
    }
}

// GPTQ2 source groups may span one or more GS32 tiles. QNN INT4 code values
// exist only in vector registers; the source byte stream remains packed INT2.
struct qnn_a16w2_packed_plan {
    size_t input_size;
    size_t output_size;
    size_t group_size;
    affine_u16 input;
    affine_u16 output;
    std::vector<uint8_t> packed_weight_codes;
    std::vector<int8_t> weight_zero_points;
    std::vector<fixed_multiplier> group_multipliers;
};

enum class kernel_backend {
    scalar,
    neon,
    compare,
};

bool initialize_ggml_cpu_backend() {
    static ggml_backend_t backend = ggml_backend_cpu_init();
    return backend != nullptr;
}

bool u16_activations_enabled() {
    const char * value = std::getenv("GGML_QNN_U16_ACTIVATIONS");
    return value != nullptr && std::string_view(value) == "1";
}

int64_t floor_shift_i64(int64_t value, int shift);

int64_t round_shift_away_from_zero(int64_t value, int shift) {
    if (shift == 0) {
        return value;
    }

    const int64_t half = int64_t{1} << (shift - 1);
    if (value >= 0) {
        return (value + half) >> shift;
    }
    return -(((-value) + half) >> shift);
}

int64_t round_shift_htp(int64_t value, int shift) {
    if (shift == 0) {
        return value;
    }

    const int64_t half = int64_t{1} << (shift - 1);
    return floor_shift_i64(value + half, shift);
}

fixed_multiplier make_multiplier(double ratio) {
    const double scaled = std::ldexp(ratio, kRequantShift);
    if (scaled > static_cast<double>(std::numeric_limits<int64_t>::max()) ||
        scaled < static_cast<double>(std::numeric_limits<int64_t>::min())) {
        std::fprintf(stderr, "requant multiplier overflow: %.17g\n", ratio);
        std::exit(2);
    }
    return { static_cast<int64_t>(std::llround(scaled)), kRequantShift };
}

uint16_t saturate_u16(int64_t value) {
    return static_cast<uint16_t>(std::clamp<int64_t>(value, 0, UINT16_MAX));
}

int64_t dot_a16w2_packed_scalar(
        const uint16_t * activations,
        int32_t activation_zero_point,
        const uint8_t * packed_weight_codes,
        int8_t weight_zero_point,
        size_t count) {
    int64_t dot = 0;
    for (size_t i = 0; i < count; ++i) {
        const int32_t activation = static_cast<int32_t>(activations[i]) - activation_zero_point;
        const int32_t weight_code =
            static_cast<int32_t>((packed_weight_codes[i >> 2U] >> ((i & 3U) * 2U)) & 0x3U);
        const int32_t weight = weight_code - weight_zero_point;
        dot += static_cast<int64_t>(activation) * weight;
    }
    return dot;
}

bool dot_a16w2_packed_neon(
        const uint16_t * activations,
        int32_t activation_zero_point,
        const uint8_t * packed_weight_codes,
        int8_t weight_zero_point,
        size_t count,
        int64_t * result) {
#if QNN_U16_HAVE_NEON
    const int32x4_t activation_zero = vdupq_n_s32(activation_zero_point);
    const int32x4_t weight_zero = vdupq_n_s32(weight_zero_point);
    const uint8x8_t code_mask = vdup_n_u8(0x3);
    size_t i = 0;
    int64_t dot = 0;

    for (; i + 32 <= count; i += 32) {
        // Four INT2 codes live in each source byte. vld4q de-interleaves
        // matching activations; unpacked QNN INT4 codes remain in registers.
        const uint16x8x4_t activation_lanes = vld4q_u16(activations + i);
        const uint8x8_t packed_codes = vld1_u8(packed_weight_codes + (i >> 2U));
        const uint8x8_t code_lanes[4] = {
            vand_u8(packed_codes, code_mask),
            vand_u8(vshr_n_u8(packed_codes, 2), code_mask),
            vand_u8(vshr_n_u8(packed_codes, 4), code_mask),
            vand_u8(vshr_n_u8(packed_codes, 6), code_mask),
        };
        int32x4_t accumulators[8];
        for (auto & accumulator : accumulators) {
            accumulator = vdupq_n_s32(0);
        }
        for (size_t lane = 0; lane < 4; ++lane) {
            const int32x4_t activation_lo = vreinterpretq_s32_u32(
                vmovl_u16(vget_low_u16(activation_lanes.val[lane])));
            const int32x4_t activation_hi = vreinterpretq_s32_u32(
                vmovl_u16(vget_high_u16(activation_lanes.val[lane])));
            const int16x8_t code_as_int4 =
                vreinterpretq_s16_u16(vmovl_u8(code_lanes[lane]));
            const int32x4_t weight_lo = vmovl_s16(vget_low_s16(code_as_int4));
            const int32x4_t weight_hi = vmovl_s16(vget_high_s16(code_as_int4));
            accumulators[lane * 2] = vmlaq_s32(
                accumulators[lane * 2],
                vsubq_s32(activation_lo, activation_zero),
                vsubq_s32(weight_lo, weight_zero));
            accumulators[lane * 2 + 1] = vmlaq_s32(
                accumulators[lane * 2 + 1],
                vsubq_s32(activation_hi, activation_zero),
                vsubq_s32(weight_hi, weight_zero));
        }
        for (const int32x4_t accumulator : accumulators) {
            dot += static_cast<int64_t>(vaddvq_s32(accumulator));
        }
    }

    dot += dot_a16w2_packed_scalar(
        activations + i,
        activation_zero_point,
        packed_weight_codes + (i >> 2U),
        weight_zero_point,
        count - i);
    *result = dot;
    return true;
#else
    (void) activations;
    (void) activation_zero_point;
    (void) packed_weight_codes;
    (void) weight_zero_point;
    (void) count;
    (void) result;
    return false;
#endif
}

void qnn_a16w2_packed_gemv_scalar(
        const qnn_a16w2_packed_plan & plan,
        const uint16_t * input,
        uint16_t * output) {
    const size_t groups = plan.input_size / plan.group_size;
    for (size_t row = 0; row < plan.output_size; ++row) {
        int64_t scaled_accumulator = 0;
        for (size_t group = 0; group < groups; ++group) {
            const size_t group_index = row * groups + group;
            const size_t base = group_index * (plan.group_size / 4U);
            const int64_t dot = dot_a16w2_packed_scalar(
                input + group * plan.group_size,
                plan.input.zero_point,
                plan.packed_weight_codes.data() + base,
                plan.weight_zero_points[group_index],
                plan.group_size);
            scaled_accumulator += dot * plan.group_multipliers[group_index].value;
        }
        output[row] = saturate_u16(
            round_shift_away_from_zero(scaled_accumulator, kRequantShift) + plan.output.zero_point);
    }
}

bool qnn_a16w2_packed_gemv_neon(
        const qnn_a16w2_packed_plan & plan,
        const uint16_t * input,
        uint16_t * output) {
#if QNN_U16_HAVE_NEON
    const size_t groups = plan.input_size / plan.group_size;
    for (size_t row = 0; row < plan.output_size; ++row) {
        int64_t scaled_accumulator = 0;
        for (size_t group = 0; group < groups; ++group) {
            const size_t group_index = row * groups + group;
            const size_t base = group_index * (plan.group_size / 4U);
            int64_t dot = 0;
            if (!dot_a16w2_packed_neon(
                    input + group * plan.group_size,
                    plan.input.zero_point,
                    plan.packed_weight_codes.data() + base,
                    plan.weight_zero_points[group_index],
                    plan.group_size,
                    &dot)) {
                return false;
            }
            scaled_accumulator += dot * plan.group_multipliers[group_index].value;
        }
        output[row] = saturate_u16(
            round_shift_away_from_zero(scaled_accumulator, kRequantShift) + plan.output.zero_point);
    }
    return true;
#else
    (void) plan;
    (void) input;
    (void) output;
    return false;
#endif
}

void qnn_u16_add_scalar(
        const uint16_t * lhs,
        affine_u16 lhs_qparams,
        const uint16_t * rhs,
        affine_u16 rhs_qparams,
        affine_u16 output_qparams,
        size_t count,
        uint16_t * output) {
    const fixed_multiplier lhs_multiplier = make_multiplier(lhs_qparams.scale / output_qparams.scale);
    const fixed_multiplier rhs_multiplier = make_multiplier(rhs_qparams.scale / output_qparams.scale);
    for (size_t i = 0; i < count; ++i) {
        const int64_t lhs_value = static_cast<int64_t>(lhs[i]) - lhs_qparams.zero_point;
        const int64_t rhs_value = static_cast<int64_t>(rhs[i]) - rhs_qparams.zero_point;
        const int64_t sum = lhs_value * lhs_multiplier.value + rhs_value * rhs_multiplier.value;
        output[i] = saturate_u16(
            round_shift_away_from_zero(sum, kRequantShift) + output_qparams.zero_point);
    }
}

bool qnn_u16_add_neon(
        const uint16_t * lhs,
        affine_u16 lhs_qparams,
        const uint16_t * rhs,
        affine_u16 rhs_qparams,
        affine_u16 output_qparams,
        size_t count,
        uint16_t * output) {
#if QNN_U16_HAVE_NEON
    const fixed_multiplier lhs_multiplier = make_multiplier(lhs_qparams.scale / output_qparams.scale);
    const fixed_multiplier rhs_multiplier = make_multiplier(rhs_qparams.scale / output_qparams.scale);
    const int32x4_t lhs_zero = vdupq_n_s32(lhs_qparams.zero_point);
    const int32x4_t rhs_zero = vdupq_n_s32(rhs_qparams.zero_point);
    const int32x2_t lhs_mul = vdup_n_s32(static_cast<int32_t>(lhs_multiplier.value));
    const int32x2_t rhs_mul = vdup_n_s32(static_cast<int32_t>(rhs_multiplier.value));
    size_t i = 0;

    for (; i + 4 <= count; i += 4) {
        const int32x4_t lhs_values =
            vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vld1_u16(lhs + i))), lhs_zero);
        const int32x4_t rhs_values =
            vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vld1_u16(rhs + i))), rhs_zero);
        const int64x2_t sum_lo = vaddq_s64(
            vmull_s32(vget_low_s32(lhs_values), lhs_mul),
            vmull_s32(vget_low_s32(rhs_values), rhs_mul));
        const int64x2_t sum_hi = vaddq_s64(
            vmull_s32(vget_high_s32(lhs_values), lhs_mul),
            vmull_s32(vget_high_s32(rhs_values), rhs_mul));
        std::array<int64_t, 4> sums {};
        vst1q_s64(sums.data(), sum_lo);
        vst1q_s64(sums.data() + 2, sum_hi);
        for (size_t lane = 0; lane < sums.size(); ++lane) {
            output[i + lane] = saturate_u16(
                round_shift_away_from_zero(sums[lane], kRequantShift) + output_qparams.zero_point);
        }
    }

    qnn_u16_add_scalar(
        lhs + i,
        lhs_qparams,
        rhs + i,
        rhs_qparams,
        output_qparams,
        count - i,
        output + i);
    return true;
#else
    (void) lhs;
    (void) lhs_qparams;
    (void) rhs;
    (void) rhs_qparams;
    (void) output_qparams;
    (void) count;
    (void) output;
    return false;
#endif
}

void qnn_u16_rms_norm_affine_scalar(
        const uint16_t * input,
        affine_u16 input_qparams,
        const uint16_t * weight,
        affine_u16 weight_qparams,
        double epsilon,
        affine_u16 output_qparams,
        size_t count,
        uint16_t * output) {
    int64_t square_sum = 0;
    for (size_t i = 0; i < count; ++i) {
        const int64_t centered = static_cast<int32_t>(input[i]) - input_qparams.zero_point;
        square_sum += centered * centered;
    }
    const double epsilon_in_codes = epsilon / (input_qparams.scale * input_qparams.scale);
    const double inverse_rms = 1.0 / std::sqrt(
        static_cast<double>(square_sum) / static_cast<double>(count) + epsilon_in_codes);
    const fixed_multiplier multiplier = make_multiplier(
        weight_qparams.scale * inverse_rms / output_qparams.scale);
    for (size_t i = 0; i < count; ++i) {
        const int64_t input_value = static_cast<int32_t>(input[i]) - input_qparams.zero_point;
        const int64_t weight_value = static_cast<int32_t>(weight[i]) - weight_qparams.zero_point;
        output[i] = saturate_u16(
            round_shift_away_from_zero(
                input_value * weight_value * multiplier.value, multiplier.shift) +
            output_qparams.zero_point);
    }
}

uint64_t fnv1a_u16(const uint16_t * values, size_t count, uint64_t hash = 1469598103934665603ULL) {
    for (size_t i = 0; i < count; ++i) {
        hash ^= static_cast<uint8_t>(values[i] & 0xffU);
        hash *= 1099511628211ULL;
        hash ^= static_cast<uint8_t>(values[i] >> 8);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool same_values(const uint16_t * lhs, const uint16_t * rhs, size_t count) {
    return std::equal(lhs, lhs + count, rhs);
}

uint64_t fnv1a_bytes(
        const uint8_t * values,
        size_t count,
        uint64_t hash = 1469598103934665603ULL) {
    for (size_t i = 0; i < count; ++i) {
        hash ^= values[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint16_t gptq2_u16_scalar_reference(
        int n,
        const uint8_t * packed_weights,
        const uint16_t * activations,
        float activation_scale,
        int32_t activation_zero_point,
        float output_scale,
        int32_t output_zero_point,
        int group_size) {
    constexpr int requant_shift = 20;
    const int code_bytes = group_size / 4;
    const int source_block_bytes = code_bytes + 4;
    int64_t scaled_accumulator = 0;
    for (int group_index = 0; group_index < n / group_size; ++group_index) {
        const uint8_t * const block = packed_weights + group_index * source_block_bytes;
        ggml_fp16_t scale_fp16;
        ggml_fp16_t zero_bias_fp16;
        std::memcpy(&scale_fp16, block + code_bytes, sizeof(scale_fp16));
        std::memcpy(&zero_bias_fp16, block + code_bytes + 2, sizeof(zero_bias_fp16));
        const float weight_scale = std::max(ggml_fp16_to_fp32(scale_fp16), 1.0e-4f);
        const int8_t weight_zero_point = static_cast<int8_t>(std::clamp<long>(
            std::lround(static_cast<double>(ggml_fp16_to_fp32(zero_bias_fp16)) / weight_scale),
            0,
            3));
        const int64_t dot = dot_a16w2_packed_scalar(
            activations + group_index * group_size,
            activation_zero_point,
            block,
            weight_zero_point,
            group_size);
        const fixed_multiplier multiplier = make_multiplier(
            static_cast<double>(activation_scale) * weight_scale / output_scale);
        scaled_accumulator += dot * multiplier.value;
    }
    return saturate_u16(
        round_shift_away_from_zero(scaled_accumulator, requant_shift) + output_zero_point);
}

uint16_t gptq2_u16_blockwise_scalar_reference(
        int n,
        const uint8_t * packed_weights,
        const uint16_t * activations,
        const uint8_t * block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int group_size) {
    const int code_bytes = group_size / 4;
    const int source_block_bytes = code_bytes + 4;
    const int chunks_per_source_group = group_size / 32;
    int64_t scaled_accumulator = 0;
    for (int qnn_block = 0; qnn_block < n / 32; ++qnn_block) {
        const int source_group = qnn_block / chunks_per_source_group;
        const int chunk = qnn_block % chunks_per_source_group;
        const uint8_t * const block = packed_weights + source_group * source_block_bytes;
        ggml_fp16_t scale_fp16;
        ggml_fp16_t zero_bias_fp16;
        std::memcpy(&scale_fp16, block + code_bytes, sizeof(scale_fp16));
        std::memcpy(&zero_bias_fp16, block + code_bytes + 2, sizeof(zero_bias_fp16));
        const float source_scale = std::max(ggml_fp16_to_fp32(scale_fp16), 1.0e-4f);
        const int8_t weight_zero_point = static_cast<int8_t>(std::clamp<long>(
            std::lround(static_cast<double>(ggml_fp16_to_fp32(zero_bias_fp16)) / source_scale),
            0,
            3));
        const int64_t dot = dot_a16w2_packed_scalar(
            activations + qnn_block * 32,
            activation_zero_point,
            block + chunk * 8,
            weight_zero_point,
            32);
        scaled_accumulator += dot * channel_scale_to_output_q31 * block_scale_codes[qnn_block];
    }
    return saturate_u16(
        round_shift_away_from_zero(scaled_accumulator, 31) + output_zero_point);
}

template <typename T>
std::vector<T> read_binary_vector(const char * path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error(std::string("failed to open binary vector: ") + path);
    }
    const std::streamoff bytes = stream.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(T)) != 0) {
        throw std::runtime_error(std::string("binary vector has an invalid size: ") + path);
    }
    std::vector<T> values(static_cast<size_t>(bytes / sizeof(T)));
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(values.data()), bytes);
    if (!stream || stream.gcount() != bytes) {
        throw std::runtime_error(std::string("failed to read binary vector: ") + path);
    }
    return values;
}

int64_t floor_shift_i64(const int64_t value, const int shift) {
    GGML_ASSERT(shift > 0 && shift < 63);
    if (value >= 0) {
        return value >> shift;
    }
    return -static_cast<int64_t>(
        (static_cast<uint64_t>(-(value + 1)) + 1U + ((UINT64_C(1) << shift) - 1U)) >> shift);
}

int64_t ceil_shift_i64(const int64_t value, const int shift) {
    GGML_ASSERT(shift > 0 && shift < 63);
    return -floor_shift_i64(-value, shift);
}

int64_t trunc_shift_i64(const int64_t value, const int shift) {
    GGML_ASSERT(shift > 0 && shift < 63);
    return value / (INT64_C(1) << shift);
}

int64_t gptq2_u16_blockwise_scaled_accumulator(
        int n,
        const uint8_t * packed_weights,
        const uint16_t * activations,
        const uint8_t * block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int group_size) {
    const int code_bytes = group_size / 4;
    const int source_block_bytes = code_bytes + 4;
    const int chunks_per_source_group = group_size / 32;
    int64_t scaled_accumulator = 0;
    for (int qnn_block = 0; qnn_block < n / 32; ++qnn_block) {
        const int source_group = qnn_block / chunks_per_source_group;
        const int chunk = qnn_block % chunks_per_source_group;
        const uint8_t * const block = packed_weights + source_group * source_block_bytes;
        ggml_fp16_t scale_fp16;
        ggml_fp16_t zero_bias_fp16;
        std::memcpy(&scale_fp16, block + code_bytes, sizeof(scale_fp16));
        std::memcpy(&zero_bias_fp16, block + code_bytes + 2, sizeof(zero_bias_fp16));
        const float source_scale = std::max(ggml_fp16_to_fp32(scale_fp16), 1.0e-4f);
        const int8_t weight_zero_point = static_cast<int8_t>(std::clamp<long>(
            std::lround(static_cast<double>(ggml_fp16_to_fp32(zero_bias_fp16)) / source_scale),
            0,
            3));
        const int64_t dot = dot_a16w2_packed_scalar(
            activations + qnn_block * 32,
            activation_zero_point,
            block + chunk * 8,
            weight_zero_point,
            32);
        scaled_accumulator += dot * channel_scale_to_output_q31 * block_scale_codes[qnn_block];
    }
    return scaled_accumulator;
}

std::vector<int64_t> gptq2_u16_blockwise_scaled_block_accumulators(
        int n,
        const uint8_t * packed_weights,
        const uint16_t * activations,
        const uint8_t * block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int group_size) {
    const int code_bytes = group_size / 4;
    const int source_block_bytes = code_bytes + 4;
    const int chunks_per_source_group = group_size / 32;
    std::vector<int64_t> result(static_cast<size_t>(n / 32));
    for (int qnn_block = 0; qnn_block < n / 32; ++qnn_block) {
        const int source_group = qnn_block / chunks_per_source_group;
        const int chunk = qnn_block % chunks_per_source_group;
        const uint8_t * const block = packed_weights + source_group * source_block_bytes;
        ggml_fp16_t scale_fp16;
        ggml_fp16_t zero_bias_fp16;
        std::memcpy(&scale_fp16, block + code_bytes, sizeof(scale_fp16));
        std::memcpy(&zero_bias_fp16, block + code_bytes + 2, sizeof(zero_bias_fp16));
        const float source_scale = std::max(ggml_fp16_to_fp32(scale_fp16), 1.0e-4f);
        const int8_t weight_zero_point = static_cast<int8_t>(std::clamp<long>(
            std::lround(static_cast<double>(ggml_fp16_to_fp32(zero_bias_fp16)) / source_scale),
            0,
            3));
        const int64_t dot = dot_a16w2_packed_scalar(
            activations + qnn_block * 32,
            activation_zero_point,
            block + chunk * 8,
            weight_zero_point,
            32);
        result[static_cast<size_t>(qnn_block)] =
            dot * channel_scale_to_output_q31 * block_scale_codes[qnn_block];
    }
    return result;
}

struct split_zero_point_block_accumulators {
    std::vector<int64_t> raw;
    std::vector<int64_t> correction;
};

split_zero_point_block_accumulators
gptq2_u16_blockwise_split_zero_point_block_accumulators(
        int n,
        const uint8_t * packed_weights,
        const uint16_t * activations,
        const uint8_t * block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int group_size) {
    const int code_bytes = group_size / 4;
    const int source_block_bytes = code_bytes + 4;
    const int chunks_per_source_group = group_size / 32;
    split_zero_point_block_accumulators result;
    result.raw.resize(static_cast<size_t>(n / 32));
    result.correction.resize(static_cast<size_t>(n / 32));
    for (int qnn_block = 0; qnn_block < n / 32; ++qnn_block) {
        const int source_group = qnn_block / chunks_per_source_group;
        const int chunk = qnn_block % chunks_per_source_group;
        const uint8_t * const block =
            packed_weights + source_group * source_block_bytes;
        ggml_fp16_t scale_fp16;
        ggml_fp16_t zero_bias_fp16;
        std::memcpy(&scale_fp16, block + code_bytes, sizeof(scale_fp16));
        std::memcpy(
            &zero_bias_fp16, block + code_bytes + 2,
            sizeof(zero_bias_fp16));
        const float source_scale =
            std::max(ggml_fp16_to_fp32(scale_fp16), 1.0e-4f);
        const int32_t weight_zero_point = static_cast<int32_t>(
            std::clamp<long>(
                std::lround(
                    static_cast<double>(ggml_fp16_to_fp32(zero_bias_fp16)) /
                    source_scale),
                0,
                3));
        int64_t raw_dot = 0;
        int64_t weight_sum = 0;
        for (int index = 0; index < 32; ++index) {
            const uint8_t packed_code = block[chunk * 8 + index / 4];
            const int32_t weight =
                static_cast<int32_t>(
                    (packed_code >> ((index & 3) * 2)) & 0x3) -
                weight_zero_point;
            raw_dot += static_cast<int64_t>(
                activations[qnn_block * 32 + index]) * weight;
            weight_sum += weight;
        }
        const int64_t multiplier =
            channel_scale_to_output_q31 * block_scale_codes[qnn_block];
        result.raw[static_cast<size_t>(qnn_block)] = raw_dot * multiplier;
        result.correction[static_cast<size_t>(qnn_block)] =
            static_cast<int64_t>(activation_zero_point) * weight_sum *
            multiplier;
    }
    return result;
}

int64_t gptq2_blockwise_weight_sum(
        int n,
        const uint8_t * packed_weights,
        const uint8_t * block_scale_codes,
        int group_size) {
    const int code_bytes = group_size / 4;
    const int source_block_bytes = code_bytes + 4;
    const int chunks_per_source_group = group_size / 32;
    int64_t weighted_sum = 0;
    for (int qnn_block = 0; qnn_block < n / 32; ++qnn_block) {
        const int source_group = qnn_block / chunks_per_source_group;
        const int chunk = qnn_block % chunks_per_source_group;
        const uint8_t * const block = packed_weights + source_group * source_block_bytes;
        ggml_fp16_t scale_fp16;
        ggml_fp16_t zero_bias_fp16;
        std::memcpy(&scale_fp16, block + code_bytes, sizeof(scale_fp16));
        std::memcpy(&zero_bias_fp16, block + code_bytes + 2, sizeof(zero_bias_fp16));
        const float source_scale = std::max(ggml_fp16_to_fp32(scale_fp16), 1.0e-4f);
        const int32_t weight_zero_point = static_cast<int32_t>(std::clamp<long>(
            std::lround(static_cast<double>(ggml_fp16_to_fp32(zero_bias_fp16)) / source_scale),
            0,
            3));
        int64_t block_sum = 0;
        for (int index = 0; index < 32; ++index) {
            const uint8_t packed_code = block[chunk * 8 + index / 4];
            const int32_t code =
                (packed_code >> ((index & 3) * 2)) & 0x3;
            block_sum += code - weight_zero_point;
        }
        weighted_sum += block_sum * block_scale_codes[qnn_block];
    }
    return weighted_sum;
}

int64_t requantize_block_chunks(
        const std::vector<int64_t> & block_accumulators,
        size_t chunk_blocks,
        bool htp_round) {
    int64_t result = 0;
    for (size_t first = 0; first < block_accumulators.size(); first += chunk_blocks) {
        const size_t end = std::min(first + chunk_blocks, block_accumulators.size());
        int64_t chunk_accumulator = 0;
        for (size_t index = first; index < end; ++index) {
            chunk_accumulator += block_accumulators[index];
        }
        result += htp_round
            ? round_shift_htp(chunk_accumulator, 31)
            : floor_shift_i64(chunk_accumulator, 31);
    }
    return result;
}

int64_t requantize_blocks_fractional(
        const std::vector<int64_t> & block_accumulators,
        int fractional_bits,
        bool intermediate_round,
        bool final_round) {
    GGML_ASSERT(fractional_bits > 0 && fractional_bits < 31);
    const int intermediate_shift = 31 - fractional_bits;
    int64_t fixed_sum = 0;
    for (const int64_t block_accumulator : block_accumulators) {
        fixed_sum += intermediate_round
            ? round_shift_htp(block_accumulator, intermediate_shift)
            : floor_shift_i64(block_accumulator, intermediate_shift);
    }
    return final_round
        ? round_shift_htp(fixed_sum, fractional_bits)
        : floor_shift_i64(fixed_sum, fractional_bits);
}

int64_t requantize_split_zero_point_blocks_fractional(
        const split_zero_point_block_accumulators & blocks,
        int fractional_bits,
        bool raw_round,
        bool correction_round) {
    GGML_ASSERT(
        blocks.raw.size() == blocks.correction.size() &&
        fractional_bits > 0 && fractional_bits < 31);
    const int intermediate_shift = 31 - fractional_bits;
    int64_t fixed_sum = 0;
    for (size_t index = 0; index < blocks.raw.size(); ++index) {
        const int64_t raw = raw_round
            ? round_shift_htp(blocks.raw[index], intermediate_shift)
            : floor_shift_i64(blocks.raw[index], intermediate_shift);
        const int64_t correction = correction_round
            ? round_shift_htp(blocks.correction[index], intermediate_shift)
            : floor_shift_i64(blocks.correction[index], intermediate_shift);
        fixed_sum += raw - correction;
    }
    return floor_shift_i64(fixed_sum, fractional_bits);
}

int64_t requantize_blocks_float32(
        const std::vector<int64_t> & block_accumulators,
        bool final_round) {
    float sum = 0.0f;
    for (const int64_t block_accumulator : block_accumulators) {
        sum += static_cast<float>(block_accumulator) * 0x1p-31f;
    }
    return final_round
        ? static_cast<int64_t>(std::floor(sum + 0.5f))
        : static_cast<int64_t>(std::floor(sum));
}

long double quantize_binary_significand(long double value, int significand_bits) {
    GGML_ASSERT(value > 0.0L && significand_bits > 0 && significand_bits < 63);
    int exponent = 0;
    const long double significand = std::frexp(value, &exponent);
    const long double quantized = std::nearbyint(
        std::ldexp(significand, significand_bits));
    return std::ldexp(quantized, exponent - significand_bits);
}

struct normalized_q31_multiplier {
    int32_t multiplier;
    int exponent;
};

normalized_q31_multiplier make_normalized_q31_multiplier(long double ratio) {
    GGML_ASSERT(ratio > 0.0L);
    int exponent = 0;
    const long double significand = std::frexp(ratio, &exponent);
    int64_t multiplier = static_cast<int64_t>(std::llround(
        std::ldexp(significand, 31)));
    if (multiplier == (INT64_C(1) << 31)) {
        multiplier /= 2;
        ++exponent;
    }
    GGML_ASSERT(multiplier > 0 && multiplier <= INT32_MAX);
    return { static_cast<int32_t>(multiplier), exponent };
}

int32_t saturating_rounding_doubling_high_mul(int32_t lhs, int32_t rhs) {
    if (lhs == INT32_MIN && rhs == INT32_MIN) {
        return INT32_MAX;
    }
    const int64_t product = static_cast<int64_t>(lhs) * rhs;
    const int64_t nudge =
        product >= 0 ? (INT64_C(1) << 30) : (1 - (INT64_C(1) << 30));
    return static_cast<int32_t>((product + nudge) / (INT64_C(1) << 31));
}

int32_t rounding_divide_by_power_of_two(int32_t value, int exponent) {
    GGML_ASSERT(exponent >= 0 && exponent < 31);
    if (exponent == 0) {
        return value;
    }
    const uint32_t mask = (UINT32_C(1) << exponent) - 1;
    const int32_t remainder = static_cast<int32_t>(
        static_cast<uint32_t>(value) & mask);
    const int32_t threshold =
        static_cast<int32_t>(mask >> 1) + (value < 0 ? 1 : 0);
    return (value >> exponent) + (remainder > threshold);
}

int32_t multiply_by_normalized_q31(
        int32_t value,
        const normalized_q31_multiplier multiplier) {
    if (multiplier.exponent > 0) {
        const int64_t shifted = static_cast<int64_t>(value) << multiplier.exponent;
        return saturating_rounding_doubling_high_mul(
            static_cast<int32_t>(std::clamp<int64_t>(
                shifted, INT32_MIN, INT32_MAX)),
            multiplier.multiplier);
    }
    return rounding_divide_by_power_of_two(
        saturating_rounding_doubling_high_mul(value, multiplier.multiplier),
        -multiplier.exponent);
}

int64_t multiply_by_normalized_q31_single_shift(
        int32_t value,
        const normalized_q31_multiplier multiplier,
        bool htp_round) {
    const int shift = 31 - multiplier.exponent;
    GGML_ASSERT(shift > 0 && shift < 63);
    const int64_t product =
        static_cast<int64_t>(value) * multiplier.multiplier;
    return htp_round
        ? round_shift_htp(product, shift)
        : floor_shift_i64(product, shift);
}

std::vector<int8_t> gptq2_qnn_requantized_codes(
        int n,
        const uint8_t * packed_weights,
        const uint8_t * block_scale_codes,
        int group_size,
        long double channel_base_weight_scale) {
    const int code_bytes = group_size / 4;
    const int source_block_bytes = code_bytes + 4;
    const int chunks_per_source_group = group_size / 32;
    std::vector<int8_t> result(static_cast<size_t>(n));
    for (int qnn_block = 0; qnn_block < n / 32; ++qnn_block) {
        const int source_group = qnn_block / chunks_per_source_group;
        const int chunk = qnn_block % chunks_per_source_group;
        const uint8_t * const block = packed_weights + source_group * source_block_bytes;
        ggml_fp16_t scale_fp16;
        ggml_fp16_t zero_bias_fp16;
        std::memcpy(&scale_fp16, block + code_bytes, sizeof(scale_fp16));
        std::memcpy(&zero_bias_fp16, block + code_bytes + 2, sizeof(zero_bias_fp16));
        const long double source_scale =
            std::max<long double>(ggml_fp16_to_fp32(scale_fp16), 1.0e-4L);
        const int32_t source_zero_point = static_cast<int32_t>(std::clamp<long>(
            std::lround(static_cast<double>(ggml_fp16_to_fp32(zero_bias_fp16)) /
                static_cast<double>(source_scale)),
            0,
            3));
        const long double effective_qnn_scale =
            channel_base_weight_scale * block_scale_codes[qnn_block];
        for (int index = 0; index < 32; ++index) {
            const int packed_index = chunk * 8 + index / 4;
            const int shift = (index & 3) * 2;
            const int32_t source_code = (block[packed_index] >> shift) & 3;
            const int32_t source_signed_code = source_code - source_zero_point;
            const int32_t qnn_signed_code = static_cast<int32_t>(std::clamp<long>(
                std::lround(static_cast<long double>(source_signed_code) * source_scale /
                    effective_qnn_scale),
                -8,
                7));
            result[static_cast<size_t>(qnn_block * 32 + index)] =
                static_cast<int8_t>(qnn_signed_code);
        }
    }
    return result;
}

int64_t qnn_requantized_codes_scaled_accumulator(
        const std::vector<int8_t> & weight_codes,
        const uint16_t * activations,
        const uint8_t * block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point) {
    int64_t scaled_accumulator = 0;
    for (size_t qnn_block = 0; qnn_block < weight_codes.size() / 32; ++qnn_block) {
        int64_t dot = 0;
        for (size_t index = 0; index < 32; ++index) {
            const size_t offset = qnn_block * 32 + index;
            dot +=
                (static_cast<int32_t>(activations[offset]) - activation_zero_point) *
                static_cast<int32_t>(weight_codes[offset]);
        }
        scaled_accumulator +=
            dot * channel_scale_to_output_q31 * block_scale_codes[qnn_block];
    }
    return scaled_accumulator;
}

struct code_error_stats {
    size_t count = 0;
    size_t exact = 0;
    size_t within_one = 0;
    uint64_t total_delta = 0;
    int64_t total_signed_delta = 0;
    uint32_t max_delta = 0;

    void add(uint16_t actual, uint16_t expected) {
        const uint32_t delta = actual > expected ? actual - expected : expected - actual;
        ++count;
        exact += delta == 0;
        within_one += delta <= 1;
        total_delta += delta;
        total_signed_delta += static_cast<int64_t>(actual) - static_cast<int64_t>(expected);
        max_delta = std::max(max_delta, delta);
    }

    void merge(const code_error_stats & other) {
        count += other.count;
        exact += other.exact;
        within_one += other.within_one;
        total_delta += other.total_delta;
        total_signed_delta += other.total_signed_delta;
        max_delta = std::max(max_delta, other.max_delta);
    }

    double mean_delta() const {
        return count == 0 ? 0.0 : static_cast<double>(total_delta) / static_cast<double>(count);
    }

    double mean_signed_delta() const {
        return count == 0 ? 0.0 :
            static_cast<double>(total_signed_delta) / static_cast<double>(count);
    }
};

int run_gguf_test(const char * gguf_path, const char * requested_tensor) {
    if (!initialize_ggml_cpu_backend()) {
        std::fprintf(stderr, "failed to initialize GGML CPU backend.\n");
        return 6;
    }

    gguf_init_params params {};
    params.no_alloc = true;
    params.ctx = nullptr;
    gguf_context * const gguf = gguf_init_from_file(gguf_path, params);
    if (gguf == nullptr) {
        std::fprintf(stderr, "failed to read GGUF metadata: %s\n", gguf_path);
        return 7;
    }

    int64_t tensor_id = -1;
    if (requested_tensor != nullptr) {
        tensor_id = gguf_find_tensor(gguf, requested_tensor);
    } else {
        const int64_t n_tensors = gguf_get_n_tensors(gguf);
        for (int64_t index = 0; index < n_tensors; ++index) {
            const enum ggml_type tensor_type = gguf_get_tensor_type(gguf, index);
            if (tensor_type == GGML_TYPE_GPTQ2_32 ||
                tensor_type == GGML_TYPE_GPTQ2_64 ||
                tensor_type == GGML_TYPE_GPTQ2_128) {
                tensor_id = index;
                break;
            }
        }
    }
    if (tensor_id < 0) {
        std::fprintf(stderr, "no requested GPTQ2_{32,64,128} tensor in GGUF: %s\n", gguf_path);
        gguf_free(gguf);
        return 8;
    }
    const enum ggml_type tensor_type = gguf_get_tensor_type(gguf, tensor_id);
    size_t group_size = 0;
    switch (tensor_type) {
        case GGML_TYPE_GPTQ2_32:
            group_size = 32;
            break;
        case GGML_TYPE_GPTQ2_64:
            group_size = 64;
            break;
        case GGML_TYPE_GPTQ2_128:
            group_size = 128;
            break;
        default:
            break;
    }
    if (group_size == 0) {
        std::fprintf(stderr, "tensor is not a supported GPTQ2 tensor: %s\n",
            gguf_get_tensor_name(gguf, tensor_id));
        gguf_free(gguf);
        return 9;
    }

    constexpr size_t max_blocks = 64;
    const size_t tensor_size = gguf_get_tensor_size(gguf, tensor_id);
    const size_t source_block_bytes = group_size / 4U + 4U;
    const size_t available_blocks = tensor_size / source_block_bytes;
    const size_t block_count = std::min(max_blocks, available_blocks);
    if (tensor_size % source_block_bytes != 0 || block_count == 0) {
        std::fprintf(stderr, "invalid GPTQ2_%zu tensor size: %zu\n", group_size, tensor_size);
        gguf_free(gguf);
        return 10;
    }
    const size_t packed_bytes = block_count * source_block_bytes;
    const size_t activation_count = block_count * group_size;
    std::unique_ptr<uint8_t[]> packed_weights(new uint8_t[packed_bytes]);
    std::unique_ptr<uint16_t[]> activations(new uint16_t[activation_count]);

    std::ifstream file(gguf_path, std::ios::binary);
    const std::streamoff data_offset = static_cast<std::streamoff>(gguf_get_data_offset(gguf));
    const std::streamoff tensor_offset = static_cast<std::streamoff>(gguf_get_tensor_offset(gguf, tensor_id));
    file.seekg(data_offset + tensor_offset);
    file.read(reinterpret_cast<char *>(packed_weights.get()), static_cast<std::streamsize>(packed_bytes));
    if (!file || static_cast<size_t>(file.gcount()) != packed_bytes) {
        std::fprintf(stderr, "failed to read GPTQ2_%zu blocks from GGUF: %s\n", group_size, gguf_path);
        gguf_free(gguf);
        return 11;
    }

    for (size_t i = 0; i < activation_count; ++i) {
        activations[i] = static_cast<uint16_t>((i * 7919U + 913U) & 0xffffU);
    }

    constexpr float activation_scale = 3.075e-5f;
    constexpr int32_t activation_zero_point = 31241;
    constexpr float output_scale = 1.937e-4f;
    constexpr int32_t output_zero_point = 28719;
    const uint16_t scalar = gptq2_u16_scalar_reference(
        static_cast<int>(activation_count),
        packed_weights.get(),
        activations.get(),
        activation_scale,
        activation_zero_point,
        output_scale,
        output_zero_point,
        static_cast<int>(group_size));
    uint16_t production = 0;
    switch (group_size) {
        case 32:
            ggml_vec_dot_gptq2_32_u16_qnn(
                static_cast<int>(activation_count), &production, packed_weights.get(),
                activations.get(), activation_scale, activation_zero_point,
                output_scale, output_zero_point);
            break;
        case 64:
            ggml_vec_dot_gptq2_64_u16_qnn(
                static_cast<int>(activation_count), &production, packed_weights.get(),
                activations.get(), activation_scale, activation_zero_point,
                output_scale, output_zero_point);
            break;
        case 128:
            ggml_vec_dot_gptq2_128_u16_qnn(
                static_cast<int>(activation_count), &production, packed_weights.get(),
                activations.get(), activation_scale, activation_zero_point,
                output_scale, output_zero_point);
            break;
    }
    const char * const tensor_name = gguf_get_tensor_name(gguf, tensor_id);
    const uint64_t source_checksum = fnv1a_bytes(packed_weights.get(), packed_bytes);
    std::printf(
        "u16-core-gguf-test: tensor=%s group_size=%zu tensor_bytes=%zu sampled_blocks=%zu n=%zu "
        "source_checksum=0x%016llx scalar=%u production=%u scalar_production_match=%d "
        "activation_dequant_buffers=0 packed_int4_buffers=0 kernel_backend=%s\n",
        tensor_name,
        group_size,
        tensor_size,
        block_count,
        activation_count,
        static_cast<unsigned long long>(source_checksum),
        static_cast<unsigned>(scalar),
        static_cast<unsigned>(production),
        scalar == production ? 1 : 0,
        QNN_U16_HAVE_NEON ? "neon" : "scalar");
    gguf_free(gguf);
    return scalar == production ? 0 : 5;
}

const char * projection_tensor_component(const std::string & projection) {
    if (projection == "self_attn.q_proj") return "attn_q";
    if (projection == "self_attn.k_proj") return "attn_k";
    if (projection == "self_attn.v_proj") return "attn_v";
    if (projection == "self_attn.o_proj") return "attn_output";
    if (projection == "mlp.gate_proj") return "ffn_gate";
    if (projection == "mlp.up_proj") return "ffn_up";
    if (projection == "mlp.down_proj") return "ffn_down";
    return nullptr;
}

std::vector<uint8_t> qnn_linear_row_block_codes(
        const llama_qnn_linear_qparams & qparams,
        size_t row) {
    const size_t blocks =
        static_cast<size_t>(qparams.qnn_weight_blocks_per_row);
    std::vector<uint8_t> result(blocks);
    if (qparams.qnn_weight_block_code_layout ==
        LLAMA_QNN_BLOCK_CODES_GS32_TILE8_BLOCK_MAJOR) {
        const size_t tile = row / 8;
        const size_t lane = row % 8;
        for (size_t block = 0; block < blocks; ++block) {
            result[block] =
                qparams.qnn_weight_block_scale_codes[
                    (tile * blocks + block) * 8 + lane] & 0x1f;
        }
    } else {
        std::copy_n(
            qparams.qnn_weight_block_scale_codes.begin() + row * blocks,
            blocks, result.begin());
        for (uint8_t & code : result) {
            code &= 0x1f;
        }
    }
    return result;
}

int run_profile_gguf_linear_test(const char * profile_path, const char * gguf_path) {
    const auto profile = llama_qnn_quant_profile_load_file(profile_path);
    ggml_context * tensor_context = nullptr;
    gguf_init_params gguf_params {};
    gguf_params.no_alloc = true;
    gguf_params.ctx = &tensor_context;
    gguf_context * gguf = gguf_init_from_file(gguf_path, gguf_params);
    if (gguf == nullptr || tensor_context == nullptr) {
        std::fprintf(stderr, "failed to load GGUF metadata for profile linear test: %s\n", gguf_path);
        if (gguf != nullptr) gguf_free(gguf);
        if (tensor_context != nullptr) ggml_free(tensor_context);
        return 13;
    }
    std::ifstream file(gguf_path, std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "failed to open GGUF for profile linear test: %s\n", gguf_path);
        gguf_free(gguf);
        ggml_free(tensor_context);
        return 13;
    }
    const int64_t layout_key = gguf_find_key(gguf, "general.gptq2_32.layout");
    const bool weights_gs32_source =
        layout_key >= 0 &&
        std::string_view(gguf_get_val_str(gguf, layout_key)) == "gs32_source_v1";

    size_t checked = 0;
    size_t checked_pairs = 0;
    size_t checked_gs32_tiles = 0;
    size_t exact = 0;
    size_t prepared_exact = 0;
    size_t saturated = 0;
    uint32_t max_code_delta = 0;
    uint64_t total_code_delta = 0;
    uint64_t checksum = 1469598103934665603ULL;
    double layer0_q_kernel_us = 0.0;
    double layer0_q_prepared_kernel_us = 0.0;
    uint64_t layer0_q_kernel_checksum = 0;
    uint64_t layer0_q_prepared_kernel_checksum = 0;
    const std::streamoff data_offset = static_cast<std::streamoff>(gguf_get_data_offset(gguf));
    for (const auto & qparams : profile->linear_qparams) {
        const char * component = projection_tensor_component(qparams.projection);
        if (component == nullptr) {
            std::fprintf(stderr, "unsupported projection in runtime profile: %s\n", qparams.projection.c_str());
            gguf_free(gguf);
            ggml_free(tensor_context);
            return 14;
        }
        char tensor_name[128];
        std::snprintf(tensor_name, sizeof(tensor_name), "blk.%d.%s.weight", qparams.layer_id, component);
        const int64_t tensor_id = gguf_find_tensor(gguf, tensor_name);
        const ggml_tensor * tensor = ggml_get_tensor(tensor_context, tensor_name);
        if (tensor_id < 0 || tensor == nullptr || tensor->ne[0] <= 0 || tensor->ne[1] <= 0) {
            std::fprintf(stderr, "profile linear tensor is missing from GGUF: %s\n", tensor_name);
            gguf_free(gguf);
            ggml_free(tensor_context);
            return 14;
        }
        int group_size = 0;
        switch (tensor->type) {
            case GGML_TYPE_GPTQ2_32: group_size = 32; break;
            case GGML_TYPE_GPTQ2_64: group_size = 64; break;
            case GGML_TYPE_GPTQ2_128: group_size = 128; break;
            default: break;
        }
        if (group_size != profile->source_group_size || tensor->ne[0] % group_size != 0) {
            std::fprintf(stderr, "profile/GGUF group-size mismatch for %s\n", tensor_name);
            gguf_free(gguf);
            ggml_free(tensor_context);
            return 14;
        }
        if (qparams.qnn_weight_block_size != 32 ||
            qparams.qnn_weight_blocks_per_row != tensor->ne[0] / 32 ||
            qparams.qnn_channel_scale_to_output_q31.size() != static_cast<size_t>(tensor->ne[1]) ||
            qparams.qnn_weight_block_scale_codes.size() !=
                static_cast<size_t>(tensor->ne[1] * qparams.qnn_weight_blocks_per_row)) {
            std::fprintf(stderr, "profile blockwise geometry mismatch for %s\n", tensor_name);
            gguf_free(gguf);
            ggml_free(tensor_context);
            return 14;
        }
        const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
        std::vector<uint8_t> packed(row_bytes);
        std::vector<uint8_t> gs32_row_block;
        if (weights_gs32_source) {
            if (tensor->type != GGML_TYPE_GPTQ2_32 || tensor->ne[1] % 64 != 0) {
                throw std::runtime_error(
                    std::string("invalid gs32_source_v1 tensor geometry: ") +
                    tensor_name);
            }
            gs32_row_block.resize(64 * row_bytes);
        }
        std::vector<uint8_t> prepared_block_codes(
            static_cast<size_t>(qparams.qnn_weight_blocks_per_row));
        std::vector<uint16_t> activations(static_cast<size_t>(tensor->ne[0]));
        for (size_t index = 0; index < activations.size(); ++index) {
            const int32_t delta = static_cast<int32_t>(
                (index * 73U + static_cast<size_t>(qparams.layer_id) * 29U + checked * 11U) % 511U) - 255;
            activations[index] = static_cast<uint16_t>(std::clamp<int32_t>(
                qparams.input.zero_point + delta, 0, UINT16_MAX));
        }

        std::vector<int64_t> rows = {0, tensor->ne[1] / 2, tensor->ne[1] - 1};
        if ((qparams.projection == "self_attn.q_proj" ||
             qparams.projection == "self_attn.k_proj" ||
             qparams.projection == "self_attn.v_proj") && tensor->ne[1] > 128) {
            rows.push_back(127);
            rows.push_back(128);
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        for (const int64_t row : rows) {
            file.clear();
            const std::streamoff tensor_data =
                data_offset +
                static_cast<std::streamoff>(gguf_get_tensor_offset(gguf, tensor_id));
            if (weights_gs32_source) {
                file.seekg(tensor_data + static_cast<std::streamoff>(
                    (row / 64) * 64 * row_bytes));
                file.read(
                    reinterpret_cast<char *>(gs32_row_block.data()),
                    static_cast<std::streamsize>(gs32_row_block.size()));
                if (file &&
                    static_cast<size_t>(file.gcount()) == gs32_row_block.size()) {
                    ggml_gptq2_32_gs32_restore_rows(
                        static_cast<int>(tensor->ne[0]), gs32_row_block.data(),
                        row % 64, 1, packed.data(), row_bytes);
                    if (row % 64 == 0) {
                        std::vector<uint8_t> tile(8 * row_bytes);
                        std::vector<uint8_t> single(row_bytes);
                        ggml_gptq2_32_gs32_restore_rows(
                            static_cast<int>(tensor->ne[0]),
                            gs32_row_block.data(), 0, 8,
                            tile.data(), row_bytes);
                        for (int tile_row = 0; tile_row < 8; ++tile_row) {
                            ggml_gptq2_32_gs32_restore_rows(
                                static_cast<int>(tensor->ne[0]),
                                gs32_row_block.data(), tile_row, 1,
                                single.data(), row_bytes);
                            if (std::memcmp(
                                    tile.data() + tile_row * row_bytes,
                                    single.data(), row_bytes) != 0) {
                                throw std::runtime_error(
                                    std::string("GS32 8-row tile mismatch: ") +
                                    tensor_name);
                            }
                        }
                        const size_t blocks =
                            static_cast<size_t>(qparams.qnn_weight_blocks_per_row);
                        std::vector<uint8_t> prepared(8 * blocks);
                        std::vector<int64_t> weight_sums(8);
                        std::vector<int32_t> activation_sums(blocks);
                        bool activations_fit_i16 = true;
                        for (size_t block = 0; block < blocks; ++block) {
                            int32_t sum = 0;
                            for (size_t lane = 0; lane < 32; ++lane) {
                                const int32_t centered =
                                    static_cast<int32_t>(
                                        activations[block * 32 + lane]) -
                                    qparams.input.zero_point;
                                sum += centered;
                                activations_fit_i16 =
                                    activations_fit_i16 &&
                                    centered >= std::numeric_limits<int16_t>::min() &&
                                    centered <= std::numeric_limits<int16_t>::max();
                            }
                            activation_sums[block] = sum;
                        }
                        for (int tile_row = 0; tile_row < 8; ++tile_row) {
                            uint8_t * row_prepared =
                                prepared.data() + tile_row * blocks;
                            const std::vector<uint8_t> row_codes =
                                qnn_linear_row_block_codes(
                                    qparams, row + tile_row);
                            ggml_gptq2_prepare_qnn_block_codes(
                                static_cast<int>(tensor->ne[0]), row_prepared,
                                tile.data() + tile_row * row_bytes,
                                row_codes.data(),
                                32);
                            weight_sums[tile_row] =
                                ggml_gptq2_32_qnn_prepared_weight_sum(
                                    static_cast<int>(tensor->ne[0]),
                                    tile.data() + tile_row * row_bytes,
                                    row_prepared);
                        }
                        uint16_t row_major_output[8];
                        uint16_t gs32_output[8];
                        std::vector<uint8_t> activation_low(tensor->ne[0]);
                        std::vector<int8_t> activation_high(tensor->ne[0]);
                        ggml_gptq2_32_prepare_u16_activation(
                            static_cast<int>(tensor->ne[0]), activations.data(),
                            qparams.input.zero_point,
                            activation_sums.data(),
                            activation_low.data(), activation_high.data(), nullptr);
                        ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_8rows(
                            static_cast<int>(tensor->ne[0]), row_major_output,
                            tile.data(), row_bytes, activations.data(),
                            activation_sums.data(), activations_fit_i16,
                            prepared.data(), blocks,
                            qparams.qnn_channel_scale_to_output_q31.data() + row,
                            weight_sums.data(), qparams.input.zero_point,
                            qparams.output.zero_point, 0, 0, 0);
                        ggml_vec_dot_gptq2_32_gs32_u16_qnn_blockwise_affine_8rows(
                            static_cast<int>(tensor->ne[0]), gs32_output,
                            gs32_row_block.data(), 0, activations.data(),
                            activation_low.data(), activation_high.data(),
                            activation_sums.data(), activations_fit_i16,
                            prepared.data(), blocks,
                            qparams.qnn_channel_scale_to_output_q31.data() + row,
                            weight_sums.data(), qparams.input.zero_point,
                            qparams.output.zero_point, 0, 0, 0);
                        if (std::memcmp(
                                row_major_output, gs32_output,
                                sizeof(row_major_output)) != 0) {
                            throw std::runtime_error(
                                std::string("GS32 direct GEMV mismatch: ") +
                                tensor_name);
                        }
                        ++checked_gs32_tiles;
                    }
                }
            } else {
                file.seekg(tensor_data + static_cast<std::streamoff>(row * row_bytes));
                file.read(
                    reinterpret_cast<char *>(packed.data()),
                    static_cast<std::streamsize>(packed.size()));
            }
            const size_t expected_read =
                weights_gs32_source ? gs32_row_block.size() : packed.size();
            if (!file || static_cast<size_t>(file.gcount()) != expected_read) {
                std::fprintf(stderr, "failed to read packed row %lld for %s\n",
                    static_cast<long long>(row), tensor_name);
                gguf_free(gguf);
                ggml_free(tensor_context);
                return 14;
            }
            const std::vector<uint8_t> row_block_scale_codes =
                qnn_linear_row_block_codes(qparams, row);
            const uint8_t * const block_scale_codes =
                row_block_scale_codes.data();
            const int64_t channel_scale_to_output_q31 =
                qparams.qnn_channel_scale_to_output_q31[static_cast<size_t>(row)];
            const uint16_t reference = gptq2_u16_blockwise_scalar_reference(
                static_cast<int>(activations.size()), packed.data(), activations.data(),
                block_scale_codes, channel_scale_to_output_q31,
                qparams.input.zero_point, qparams.output.zero_point, group_size);
            uint16_t fixed = 0;
            switch (group_size) {
                case 32:
                    ggml_vec_dot_gptq2_32_u16_qnn_blockwise_fixed(
                        static_cast<int>(activations.size()), &fixed, packed.data(), activations.data(),
                        block_scale_codes, channel_scale_to_output_q31,
                        qparams.input.zero_point, qparams.output.zero_point);
                    break;
                case 64:
                    ggml_vec_dot_gptq2_64_u16_qnn_blockwise_fixed(
                        static_cast<int>(activations.size()), &fixed, packed.data(), activations.data(),
                        block_scale_codes, channel_scale_to_output_q31,
                        qparams.input.zero_point, qparams.output.zero_point);
                    break;
                case 128:
                    ggml_vec_dot_gptq2_128_u16_qnn_blockwise_fixed(
                        static_cast<int>(activations.size()), &fixed, packed.data(), activations.data(),
                        block_scale_codes, channel_scale_to_output_q31,
                        qparams.input.zero_point, qparams.output.zero_point);
                    break;
            }
            ggml_gptq2_prepare_qnn_block_codes(
                static_cast<int>(activations.size()),
                prepared_block_codes.data(),
                packed.data(),
                block_scale_codes,
                group_size);
            uint16_t prepared_fixed = 0;
            switch (group_size) {
                case 32:
                    ggml_vec_dot_gptq2_32_u16_qnn_blockwise_prepared(
                        static_cast<int>(activations.size()), &prepared_fixed,
                        packed.data(), activations.data(),
                        prepared_block_codes.data(), channel_scale_to_output_q31,
                        qparams.input.zero_point, qparams.output.zero_point);
                    break;
                case 64:
                    ggml_vec_dot_gptq2_64_u16_qnn_blockwise_prepared(
                        static_cast<int>(activations.size()), &prepared_fixed,
                        packed.data(), activations.data(),
                        prepared_block_codes.data(), channel_scale_to_output_q31,
                        qparams.input.zero_point, qparams.output.zero_point);
                    break;
                case 128:
                    ggml_vec_dot_gptq2_128_u16_qnn_blockwise_prepared(
                        static_cast<int>(activations.size()), &prepared_fixed,
                        packed.data(), activations.data(),
                        prepared_block_codes.data(), channel_scale_to_output_q31,
                        qparams.input.zero_point, qparams.output.zero_point);
                    break;
            }
            prepared_exact += prepared_fixed == fixed;
            if (qparams.layer_id == 0 &&
                qparams.projection == "self_attn.q_proj" &&
                row == 0) {
                constexpr int benchmark_warmup = 128;
                constexpr int benchmark_iterations = 20000;
                const auto run_production = [&]() {
                    switch (group_size) {
                        case 32:
                            ggml_vec_dot_gptq2_32_u16_qnn_blockwise_fixed(
                                static_cast<int>(activations.size()), &fixed,
                                packed.data(), activations.data(), block_scale_codes,
                                channel_scale_to_output_q31, qparams.input.zero_point,
                                qparams.output.zero_point);
                            break;
                        case 64:
                            ggml_vec_dot_gptq2_64_u16_qnn_blockwise_fixed(
                                static_cast<int>(activations.size()), &fixed,
                                packed.data(), activations.data(), block_scale_codes,
                                channel_scale_to_output_q31, qparams.input.zero_point,
                                qparams.output.zero_point);
                            break;
                        case 128:
                            ggml_vec_dot_gptq2_128_u16_qnn_blockwise_fixed(
                                static_cast<int>(activations.size()), &fixed,
                                packed.data(), activations.data(), block_scale_codes,
                                channel_scale_to_output_q31, qparams.input.zero_point,
                                qparams.output.zero_point);
                            break;
                    }
                };
                const auto run_prepared = [&]() {
                    ggml_vec_dot_gptq2_32_u16_qnn_blockwise_prepared(
                        static_cast<int>(activations.size()), &fixed,
                        packed.data(), activations.data(),
                        prepared_block_codes.data(),
                        channel_scale_to_output_q31,
                        qparams.input.zero_point,
                        qparams.output.zero_point);
                };
                const uint8_t max_block_scale_code = *std::max_element(
                    block_scale_codes,
                    block_scale_codes + qparams.qnn_weight_blocks_per_row);
                std::printf(
                    "qnn-profile-gguf-linear-prepared-check: fixed=%u prepared=%u "
                    "first_scale=%u first_prepared=%u max_scale=%u\n",
                    static_cast<unsigned>(fixed),
                    static_cast<unsigned>(prepared_fixed),
                    static_cast<unsigned>(block_scale_codes[0]),
                    static_cast<unsigned>(prepared_block_codes[0]),
                    static_cast<unsigned>(max_block_scale_code));
                for (int iteration = 0; iteration < benchmark_warmup; ++iteration) {
                    run_production();
                    layer0_q_kernel_checksum += fixed;
                }
                const auto benchmark_start = std::chrono::steady_clock::now();
                for (int iteration = 0; iteration < benchmark_iterations; ++iteration) {
                    run_production();
                    layer0_q_kernel_checksum += fixed;
                }
                const auto benchmark_end = std::chrono::steady_clock::now();
                layer0_q_kernel_us =
                    std::chrono::duration<double, std::micro>(
                        benchmark_end - benchmark_start).count() /
                    benchmark_iterations;
                if (group_size == 32) {
                    for (int iteration = 0; iteration < benchmark_warmup; ++iteration) {
                        run_prepared();
                        layer0_q_prepared_kernel_checksum += fixed;
                    }
                    const auto prepared_start = std::chrono::steady_clock::now();
                    for (int iteration = 0; iteration < benchmark_iterations; ++iteration) {
                        run_prepared();
                        layer0_q_prepared_kernel_checksum += fixed;
                    }
                    const auto prepared_end = std::chrono::steady_clock::now();
                    layer0_q_prepared_kernel_us =
                        std::chrono::duration<double, std::micro>(
                            prepared_end - prepared_start).count() /
                        benchmark_iterations;
                }
            }
            const uint32_t delta = reference > fixed ? reference - fixed : fixed - reference;
            max_code_delta = std::max(max_code_delta, delta);
            total_code_delta += delta;
            exact += delta == 0;
            saturated += fixed == 0 || fixed == UINT16_MAX;
            checksum = fnv1a_u16(&fixed, 1, checksum);
            ++checked;
        }
        ++checked_pairs;
    }

    const double mean_code_delta = checked == 0 ? 0.0 :
        static_cast<double>(total_code_delta) / static_cast<double>(checked);
    std::printf(
        "qnn-profile-gguf-linear-test: pairs=%zu sampled_rows=%zu gs32_tiles=%zu "
        "exact=%zu max_code_delta=%u "
        "mean_code_delta=%.6f prepared_exact=%zu saturated=%zu source_bits=%d group_size=%d "
        "layer0_q_kernel_us=%.6f layer0_q_kernel_checksum=%llu "
        "layer0_q_prepared_kernel_us=%.6f layer0_q_prepared_kernel_checksum=%llu "
        "activation_dequant_buffers=0 packed_int4_buffers=0 layout=%s "
        "checksum=0x%016llx status=%s\n",
        checked_pairs, checked, checked_gs32_tiles,
        exact, max_code_delta, mean_code_delta,
        prepared_exact, saturated,
        profile->source_weight_bits, profile->source_group_size,
        layer0_q_kernel_us,
        static_cast<unsigned long long>(layer0_q_kernel_checksum),
        layer0_q_prepared_kernel_us,
        static_cast<unsigned long long>(layer0_q_prepared_kernel_checksum),
        weights_gs32_source ? "gs32_source_v1" : "row_major",
        static_cast<unsigned long long>(checksum),
        checked_pairs == profile->linear_qparams_count() &&
            prepared_exact == checked && max_code_delta <= 2 ? "pass" : "fail");
    gguf_free(gguf);
    ggml_free(tensor_context);
    return checked_pairs == profile->linear_qparams_count() &&
        prepared_exact == checked && max_code_delta <= 2 ? 0 : 15;
}

int run_profile_gguf_linear_vector_test(
        const char * profile_path,
        const char * gguf_path,
        int32_t layer_id,
        const char * projection,
        const char * input_path,
        const char * expected_path) {
    const auto profile = llama_qnn_quant_profile_load_file(profile_path);
    const llama_qnn_linear_qparams * qparams = profile->find_linear_qparams(layer_id, projection);
    if (qparams == nullptr) {
        throw std::runtime_error(
            "runtime profile has no linear qparams for layer " + std::to_string(layer_id) +
            " projection " + projection);
    }
    std::printf(
        "qnn-profile-gguf-linear-qparams: layer=%d projection=%s "
        "input_scale=%.9g input_zero_point=%d output_scale=%.9g output_zero_point=%d "
        "input_output_ratio=%.9g input_output_ratio_exponent=%d\n",
        layer_id,
        projection,
        qparams->input.scale,
        qparams->input.zero_point,
        qparams->output.scale,
        qparams->output.zero_point,
        qparams->input.scale / qparams->output.scale,
        std::ilogb(qparams->input.scale / qparams->output.scale));
    const char * component = projection_tensor_component(qparams->projection);
    if (component == nullptr) {
        throw std::runtime_error(std::string("unsupported projection: ") + projection);
    }

    ggml_context * tensor_context = nullptr;
    gguf_init_params gguf_params {};
    gguf_params.no_alloc = true;
    gguf_params.ctx = &tensor_context;
    gguf_context * gguf = gguf_init_from_file(gguf_path, gguf_params);
    if (gguf == nullptr || tensor_context == nullptr) {
        if (gguf != nullptr) gguf_free(gguf);
        if (tensor_context != nullptr) ggml_free(tensor_context);
        throw std::runtime_error(std::string("failed to load GGUF metadata: ") + gguf_path);
    }

    char tensor_name[128];
    std::snprintf(tensor_name, sizeof(tensor_name), "blk.%d.%s.weight", layer_id, component);
    const int64_t tensor_id = gguf_find_tensor(gguf, tensor_name);
    const ggml_tensor * tensor = ggml_get_tensor(tensor_context, tensor_name);
    if (tensor_id < 0 || tensor == nullptr) {
        gguf_free(gguf);
        ggml_free(tensor_context);
        throw std::runtime_error(std::string("GGUF tensor is missing: ") + tensor_name);
    }
    int group_size = 0;
    switch (tensor->type) {
        case GGML_TYPE_GPTQ2_32: group_size = 32; break;
        case GGML_TYPE_GPTQ2_64: group_size = 64; break;
        case GGML_TYPE_GPTQ2_128: group_size = 128; break;
        default: break;
    }
    if (group_size == 0 || tensor->ne[0] % group_size != 0 ||
        qparams->qnn_weight_block_size != 32 ||
        qparams->qnn_weight_blocks_per_row != tensor->ne[0] / 32) {
        gguf_free(gguf);
        ggml_free(tensor_context);
        throw std::runtime_error(std::string("unsupported linear geometry for ") + tensor_name);
    }

    const std::vector<uint16_t> input = read_binary_vector<uint16_t>(input_path);
    const std::vector<uint16_t> expected = read_binary_vector<uint16_t>(expected_path);
    const size_t input_width = static_cast<size_t>(tensor->ne[0]);
    if (input.empty() || input.size() % input_width != 0 || expected.empty()) {
        gguf_free(gguf);
        ggml_free(tensor_context);
        throw std::runtime_error(
            "linear vector dimensions do not match GGUF tensor " + std::string(tensor_name));
    }
    const size_t token_count = input.size() / input_width;
    if (expected.size() % token_count != 0 ||
        expected.size() / token_count > static_cast<size_t>(tensor->ne[1])) {
        gguf_free(gguf);
        ggml_free(tensor_context);
        throw std::runtime_error(
            "expected linear vector dimensions do not match GGUF tensor " + std::string(tensor_name));
    }
    const size_t output_rows = expected.size() / token_count;
    if (qparams->qnn_channel_scale_to_output_q31.size() != static_cast<size_t>(tensor->ne[1]) ||
        qparams->qnn_weight_block_scale_codes.size() !=
            static_cast<size_t>(tensor->ne[1] * qparams->qnn_weight_blocks_per_row)) {
        gguf_free(gguf);
        ggml_free(tensor_context);
        throw std::runtime_error(std::string("profile blockwise qparams mismatch for ") + tensor_name);
    }

    std::ifstream file(gguf_path, std::ios::binary);
    if (!file) {
        gguf_free(gguf);
        ggml_free(tensor_context);
        throw std::runtime_error(std::string("failed to open GGUF data: ") + gguf_path);
    }
    const std::streamoff data_offset = static_cast<std::streamoff>(gguf_get_data_offset(gguf));
    const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
    std::vector<uint8_t> packed(row_bytes);
    code_error_stats production_stats;
    code_error_stats floor_stats;
    code_error_stats htp_round_stats;
    code_error_stats away_stats;
    code_error_stats trunc_stats;
    code_error_stats fitted_scale_stats;
    code_error_stats production_holdout_stats;
    code_error_stats fitted_scale_holdout_stats;
    code_error_stats interval_scale_stats;
    code_error_stats qnn_weight_requant_stats;
    code_error_stats production_requant29_stats;
    size_t production_requant29_mismatches = 0;
    static constexpr std::array<size_t, 6> chunk_block_counts = { 1, 2, 4, 8, 16, 32 };
    std::array<code_error_stats, chunk_block_counts.size()> chunk_floor_stats;
    std::array<code_error_stats, chunk_block_counts.size()> chunk_round_stats;
    static constexpr size_t fractional_candidate_count = 24;
    std::array<code_error_stats, fractional_candidate_count> fractional_floor_floor_stats;
    std::array<code_error_stats, fractional_candidate_count> fractional_round_floor_stats;
    std::array<code_error_stats, fractional_candidate_count> fractional_round_round_stats;
    std::array<size_t, fractional_candidate_count> row_best_floor_floor_histogram {};
    std::array<size_t, fractional_candidate_count> row_best_round_floor_histogram {};
    code_error_stats row_oracle_floor_floor_stats;
    code_error_stats row_oracle_round_floor_stats;
    static constexpr int first_multiplier_exponent_constant = 20;
    static constexpr size_t multiplier_exponent_candidate_count = 36;
    std::array<code_error_stats, multiplier_exponent_candidate_count>
        multiplier_exponent_floor_floor_stats;
    std::array<code_error_stats, multiplier_exponent_candidate_count>
        multiplier_exponent_round_floor_stats;
    std::array<code_error_stats, multiplier_exponent_candidate_count>
        multiplier_exponent_floor_floor_holdout_stats;
    std::array<code_error_stats, multiplier_exponent_candidate_count>
        multiplier_exponent_round_floor_holdout_stats;
    code_error_stats float32_floor_stats;
    code_error_stats float32_round_stats;
    static constexpr int first_scale_significand_bits = 4;
    static constexpr size_t scale_significand_candidate_count = 17;
    std::array<code_error_stats, scale_significand_candidate_count>
        scale_significand_floor_stats;
    std::array<code_error_stats, scale_significand_candidate_count>
        scale_significand_round_stats;
    code_error_stats fp16_base_scale_floor_stats;
    code_error_stats fp16_base_scale_round_stats;
    code_error_stats normalized_q31_stats;
    code_error_stats normalized_q31_single_floor_stats;
    code_error_stats normalized_q31_single_round_stats;
    code_error_stats fp16_real_floor_stats;
    code_error_stats fp16_real_round_stats;
    code_error_stats fp16_block_scale_floor_stats;
    code_error_stats fp16_block_scale_round_stats;
    static constexpr int first_block_multiplier_bits = 8;
    static constexpr size_t block_multiplier_candidate_count = 23;
    std::array<code_error_stats, block_multiplier_candidate_count>
        block_multiplier_floor_stats;
    std::array<code_error_stats, block_multiplier_candidate_count>
        block_multiplier_round_stats;
    static constexpr int output_bias_denominator = 128;
    static constexpr int output_bias_radius = 256;
    static constexpr size_t output_bias_candidate_count =
        output_bias_radius * 2 + 1;
    std::array<code_error_stats, output_bias_candidate_count> output_bias_stats;
    std::array<code_error_stats, output_bias_candidate_count> output_bias_fit_stats;
    std::array<code_error_stats, output_bias_candidate_count> output_bias_holdout_stats;
    static constexpr int accumulator_bias_radius = 512;
    static constexpr size_t accumulator_bias_candidate_count =
        accumulator_bias_radius * 2 + 1;
    std::array<code_error_stats, accumulator_bias_candidate_count>
        accumulator_bias_stats;
    std::array<code_error_stats, accumulator_bias_candidate_count>
        accumulator_bias_fit_stats;
    std::array<code_error_stats, accumulator_bias_candidate_count>
        accumulator_bias_holdout_stats;
    code_error_stats accumulator_bias_row_oracle_stats;
    std::array<size_t, accumulator_bias_candidate_count>
        accumulator_bias_row_histogram {};
    code_error_stats split_zp_floor_floor_stats;
    code_error_stats split_zp_floor_round_stats;
    code_error_stats split_zp_round_round_stats;
    code_error_stats split_zp_round_floor_stats;
    code_error_stats htp_a16s8_reduced_accumulator_stats;
    static constexpr size_t split_block_fractional_candidate_count = 30;
    std::array<code_error_stats, split_block_fractional_candidate_count>
        split_block_floor_floor_stats;
    std::array<code_error_stats, split_block_fractional_candidate_count>
        split_block_floor_round_stats;
    std::array<code_error_stats, split_block_fractional_candidate_count>
        split_block_round_floor_stats;
    std::array<code_error_stats, split_block_fractional_candidate_count>
        split_block_round_round_stats;
    std::array<code_error_stats, split_block_fractional_candidate_count>
        split_block_floor_ceil_stats;
    std::array<code_error_stats, split_block_fractional_candidate_count>
        split_block_trunc_ceil_stats;
    std::array<code_error_stats, split_block_fractional_candidate_count>
        split_block_trunc_round_stats;
    size_t production_floor_differences = 0;
    size_t weighted_dot_i32_overflows = 0;
    std::array<size_t, 4> source_zero_point_counts {};
    int64_t min_weighted_dot = INT64_MAX;
    int64_t max_weighted_dot = INT64_MIN;
    long double fitted_scale_ppm_sum = 0.0L;
    long double fitted_scale_ppm_abs_sum = 0.0L;
    long double fitted_scale_ppm_min = std::numeric_limits<long double>::infinity();
    long double fitted_scale_ppm_max = -std::numeric_limits<long double>::infinity();
    size_t fitted_scale_rows = 0;
    size_t interval_explainable_rows = 0;
    size_t nearest_interval_explainable_rows = 0;
    const size_t fit_token_count = std::max<size_t>(1, token_count / 2);
    for (size_t row = 0; row < output_rows; ++row) {
        file.clear();
        file.seekg(
            data_offset + static_cast<std::streamoff>(gguf_get_tensor_offset(gguf, tensor_id)) +
            static_cast<std::streamoff>(row * row_bytes));
        file.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
        if (!file || static_cast<size_t>(file.gcount()) != packed.size()) {
            gguf_free(gguf);
            ggml_free(tensor_context);
            throw std::runtime_error("failed to read packed GGUF row " + std::to_string(row));
        }
        const std::vector<uint8_t> row_block_scale_codes =
            qnn_linear_row_block_codes(*qparams, row);
        const uint8_t * block_scale_codes = row_block_scale_codes.data();
        const size_t source_code_bytes = static_cast<size_t>(group_size) / 4;
        const size_t source_block_bytes = source_code_bytes + 4;
        const uint8_t * source_row = packed.data();
        for (size_t group = 0; group < input_width / static_cast<size_t>(group_size); ++group) {
            const uint8_t * source_block = source_row + group * source_block_bytes;
            ggml_fp16_t scale_fp16;
            ggml_fp16_t zero_bias_fp16;
            std::memcpy(&scale_fp16, source_block + source_code_bytes, sizeof(scale_fp16));
            std::memcpy(
                &zero_bias_fp16, source_block + source_code_bytes + sizeof(scale_fp16),
                sizeof(zero_bias_fp16));
            const float scale = std::max(ggml_fp16_to_fp32(scale_fp16), 0.0001f);
            const long source_zero_point = std::clamp<long>(
                std::lround(static_cast<double>(ggml_fp16_to_fp32(zero_bias_fp16)) /
                    static_cast<double>(scale)),
                0, 3);
            ++source_zero_point_counts[static_cast<size_t>(source_zero_point)];
        }
        const int64_t channel_multiplier = qparams->qnn_channel_scale_to_output_q31[row];
        const long double channel_base_weight_scale =
            (static_cast<long double>(channel_multiplier) /
                static_cast<long double>(UINT64_C(1) << 31)) *
            static_cast<long double>(qparams->output.scale) /
            static_cast<long double>(qparams->input.scale);
        const long double nominal_ratio =
            static_cast<long double>(channel_multiplier) /
            static_cast<long double>(UINT64_C(1) << 31);
        const normalized_q31_multiplier normalized_multiplier =
            make_normalized_q31_multiplier(nominal_ratio);
        const float fp16_base_weight_scale = ggml_fp16_to_fp32(
            ggml_fp32_to_fp16(static_cast<float>(channel_base_weight_scale)));
        const int64_t fp16_base_multiplier = static_cast<int64_t>(std::llround(
            std::ldexp(
                nominal_ratio *
                    (static_cast<long double>(fp16_base_weight_scale) /
                        channel_base_weight_scale),
                31)));
        std::array<int64_t, scale_significand_candidate_count>
            scale_significand_multipliers;
        for (size_t candidate = 0;
             candidate < scale_significand_candidate_count;
             ++candidate) {
            const int significand_bits =
                first_scale_significand_bits + static_cast<int>(candidate);
            const long double quantized_base = quantize_binary_significand(
                channel_base_weight_scale, significand_bits);
            scale_significand_multipliers[candidate] =
                static_cast<int64_t>(std::llround(std::ldexp(
                    nominal_ratio * (quantized_base / channel_base_weight_scale), 31)));
        }
        if (row < 8) {
            std::printf(
                "qnn-profile-gguf-linear-row-scale: tensor=%s row=%zu "
                "channel_multiplier=%lld base_weight_scale=%.12Lg first_block_code=%u\n",
                tensor_name,
                row,
                static_cast<long long>(channel_multiplier),
                channel_base_weight_scale,
                static_cast<unsigned>(block_scale_codes[0]));
        }
        const std::vector<int8_t> qnn_requantized_codes = gptq2_qnn_requantized_codes(
            static_cast<int>(input_width), packed.data(), block_scale_codes, group_size,
            channel_base_weight_scale);
        const int64_t weighted_weight_sum = gptq2_blockwise_weight_sum(
            static_cast<int>(input_width), packed.data(), block_scale_codes,
            group_size);
        std::vector<int64_t> row_weighted_dots(token_count);
        long double fit_numerator = 0.0L;
        long double fit_denominator = 0.0L;
        long double scale_interval_lower = -std::numeric_limits<long double>::infinity();
        long double scale_interval_upper = std::numeric_limits<long double>::infinity();
        long double nearest_interval_lower = -std::numeric_limits<long double>::infinity();
        long double nearest_interval_upper = std::numeric_limits<long double>::infinity();
        std::array<code_error_stats, fractional_candidate_count>
            row_fractional_floor_floor_stats;
        std::array<code_error_stats, fractional_candidate_count>
            row_fractional_round_floor_stats;
        code_error_stats row_production_stats;
        std::array<code_error_stats, accumulator_bias_candidate_count>
            row_accumulator_bias_stats;
        const int channel_multiplier_exponent =
            63 - __builtin_clzll(static_cast<unsigned long long>(channel_multiplier));
        for (size_t token = 0; token < token_count; ++token) {
            const uint16_t * token_input = input.data() + token * input_width;
            uint16_t production = 0;
            ggml_vec_dot_gptq2_u16_qnn_blockwise_affine(
                static_cast<int>(input_width), &production, packed.data(), token_input,
                block_scale_codes, channel_multiplier,
                qparams->input.zero_point, qparams->output.zero_point,
                group_size, 0, 0,
                qparams->projection == "self_attn.o_proj" &&
                    u16_activations_enabled() &&
                    std::getenv("GGML_QNN_U16_OPROJ_NEAREST_REQUANT") != nullptr,
                0);
            const int64_t accumulator = gptq2_u16_blockwise_scaled_accumulator(
                static_cast<int>(input_width), packed.data(), token_input, block_scale_codes,
                channel_multiplier, qparams->input.zero_point, group_size);
            const std::vector<int64_t> block_accumulators =
                gptq2_u16_blockwise_scaled_block_accumulators(
                    static_cast<int>(input_width), packed.data(), token_input,
                    block_scale_codes, channel_multiplier, qparams->input.zero_point,
                    group_size);
            const split_zero_point_block_accumulators split_block_accumulators =
                gptq2_u16_blockwise_split_zero_point_block_accumulators(
                    static_cast<int>(input_width), packed.data(), token_input,
                    block_scale_codes, channel_multiplier,
                    qparams->input.zero_point, group_size);
            GGML_ASSERT(channel_multiplier > 0 && accumulator % channel_multiplier == 0);
            const int64_t weighted_dot = accumulator / channel_multiplier;
            row_weighted_dots[token] = weighted_dot;
            min_weighted_dot = std::min(min_weighted_dot, weighted_dot);
            max_weighted_dot = std::max(max_weighted_dot, weighted_dot);
            weighted_dot_i32_overflows +=
                weighted_dot < INT32_MIN || weighted_dot > INT32_MAX;
            const uint16_t floor_output = saturate_u16(
                floor_shift_i64(accumulator, 31) + qparams->output.zero_point);
            const uint16_t htp_round_output = saturate_u16(
                round_shift_htp(accumulator, 31) + qparams->output.zero_point);
            const uint16_t away_output = saturate_u16(
                round_shift_away_from_zero(accumulator, 31) + qparams->output.zero_point);
            const uint16_t trunc_output = saturate_u16(
                accumulator / (INT64_C(1) << 31) + qparams->output.zero_point);
            const int64_t qnn_weight_requant_accumulator =
                qnn_requantized_codes_scaled_accumulator(
                    qnn_requantized_codes, token_input, block_scale_codes,
                    channel_multiplier, qparams->input.zero_point);
            const uint16_t qnn_weight_requant_output = saturate_u16(
                floor_shift_i64(qnn_weight_requant_accumulator, 31) +
                qparams->output.zero_point);
            const uint16_t expected_output = expected[token * output_rows + row];
            uint16_t production_requant29 = 0;
            ggml_vec_dot_gptq2_u16_qnn_blockwise_requant(
                static_cast<int>(input_width), &production_requant29,
                packed.data(), token_input, block_scale_codes, channel_multiplier,
                qparams->input.zero_point, qparams->output.zero_point,
                group_size, 0, 29);
            const int fractional_bits29 = std::clamp(
                29 - channel_multiplier_exponent,
                1,
                static_cast<int>(fractional_candidate_count));
            const uint16_t reference_requant29 = saturate_u16(
                requantize_blocks_fractional(
                    block_accumulators, fractional_bits29, false, false) +
                qparams->output.zero_point);
            production_requant29_mismatches +=
                production_requant29 != reference_requant29;
            production_requant29_stats.add(production_requant29, expected_output);
            if (token < fit_token_count &&
                expected_output != 0 && expected_output != UINT16_MAX && weighted_dot != 0) {
                const long double expected_code =
                    static_cast<long double>(expected_output) - qparams->output.zero_point;
                const long double expected_center = expected_code + 0.5L;
                fit_numerator += static_cast<long double>(weighted_dot) * expected_center;
                fit_denominator +=
                    static_cast<long double>(weighted_dot) * static_cast<long double>(weighted_dot);
                const long double edge0 = expected_code / weighted_dot;
                const long double edge1 = (expected_code + 1.0L) / weighted_dot;
                scale_interval_lower =
                    std::max(scale_interval_lower, std::min(edge0, edge1));
                scale_interval_upper =
                    std::min(scale_interval_upper, std::max(edge0, edge1));
                const long double nearest_edge0 = (expected_code - 0.5L) / weighted_dot;
                const long double nearest_edge1 = (expected_code + 0.5L) / weighted_dot;
                nearest_interval_lower =
                    std::max(nearest_interval_lower, std::min(nearest_edge0, nearest_edge1));
                nearest_interval_upper =
                    std::min(nearest_interval_upper, std::max(nearest_edge0, nearest_edge1));
            }
            production_stats.add(production, expected_output);
            if (token >= fit_token_count) {
                production_holdout_stats.add(production, expected_output);
            }
            row_production_stats.add(production, expected_output);
            floor_stats.add(floor_output, expected_output);
            htp_round_stats.add(htp_round_output, expected_output);
            away_stats.add(away_output, expected_output);
            trunc_stats.add(trunc_output, expected_output);
            qnn_weight_requant_stats.add(qnn_weight_requant_output, expected_output);
            for (size_t candidate = 0;
                 candidate < accumulator_bias_candidate_count;
                 ++candidate) {
                const int accumulator_bias =
                    static_cast<int>(candidate) - accumulator_bias_radius;
                const uint16_t accumulator_bias_output = saturate_u16(
                    round_shift_htp(
                        accumulator + accumulator_bias * channel_multiplier,
                        31) +
                    qparams->output.zero_point);
                accumulator_bias_stats[candidate].add(
                    accumulator_bias_output, expected_output);
                row_accumulator_bias_stats[candidate].add(
                    accumulator_bias_output, expected_output);
                if (token < fit_token_count) {
                    accumulator_bias_fit_stats[candidate].add(
                        accumulator_bias_output, expected_output);
                } else {
                    accumulator_bias_holdout_stats[candidate].add(
                        accumulator_bias_output, expected_output);
                }
            }
            for (size_t candidate = 0; candidate < chunk_block_counts.size(); ++candidate) {
                const uint16_t chunk_floor_output = saturate_u16(
                    requantize_block_chunks(
                        block_accumulators, chunk_block_counts[candidate], false) +
                    qparams->output.zero_point);
                const uint16_t chunk_round_output = saturate_u16(
                    requantize_block_chunks(
                        block_accumulators, chunk_block_counts[candidate], true) +
                    qparams->output.zero_point);
                chunk_floor_stats[candidate].add(chunk_floor_output, expected_output);
                chunk_round_stats[candidate].add(chunk_round_output, expected_output);
            }
            for (size_t candidate = 0; candidate < fractional_candidate_count; ++candidate) {
                const int fractional_bits = static_cast<int>(candidate + 1);
                const uint16_t fractional_floor_floor_output = saturate_u16(
                    requantize_blocks_fractional(
                        block_accumulators, fractional_bits, false, false) +
                    qparams->output.zero_point);
                const uint16_t fractional_round_floor_output = saturate_u16(
                    requantize_blocks_fractional(
                        block_accumulators, fractional_bits, true, false) +
                    qparams->output.zero_point);
                const uint16_t fractional_round_round_output = saturate_u16(
                    requantize_blocks_fractional(
                        block_accumulators, fractional_bits, true, true) +
                    qparams->output.zero_point);
                fractional_floor_floor_stats[candidate].add(
                    fractional_floor_floor_output, expected_output);
                fractional_round_floor_stats[candidate].add(
                    fractional_round_floor_output, expected_output);
                fractional_round_round_stats[candidate].add(
                    fractional_round_round_output, expected_output);
                row_fractional_floor_floor_stats[candidate].add(
                    fractional_floor_floor_output, expected_output);
                row_fractional_round_floor_stats[candidate].add(
                    fractional_round_floor_output, expected_output);
            }
            for (size_t candidate = 0;
                 candidate < multiplier_exponent_candidate_count;
                 ++candidate) {
                const int constant =
                    first_multiplier_exponent_constant + static_cast<int>(candidate);
                const int fractional_bits = std::clamp(
                    constant - channel_multiplier_exponent,
                    1,
                    static_cast<int>(fractional_candidate_count));
                const uint16_t floor_floor_output = saturate_u16(
                    requantize_blocks_fractional(
                        block_accumulators, fractional_bits, false, false) +
                    qparams->output.zero_point);
                const uint16_t round_floor_output = saturate_u16(
                    requantize_blocks_fractional(
                        block_accumulators, fractional_bits, true, false) +
                    qparams->output.zero_point);
                multiplier_exponent_floor_floor_stats[candidate].add(
                    floor_floor_output, expected_output);
                multiplier_exponent_round_floor_stats[candidate].add(
                    round_floor_output, expected_output);
                if (token >= fit_token_count) {
                    multiplier_exponent_floor_floor_holdout_stats[candidate].add(
                        floor_floor_output, expected_output);
                    multiplier_exponent_round_floor_holdout_stats[candidate].add(
                        round_floor_output, expected_output);
                }
            }
            float32_floor_stats.add(
                saturate_u16(
                    requantize_blocks_float32(block_accumulators, false) +
                    qparams->output.zero_point),
                expected_output);
            float32_round_stats.add(
                saturate_u16(
                    requantize_blocks_float32(block_accumulators, true) +
                    qparams->output.zero_point),
                expected_output);
            const int64_t fp16_base_accumulator =
                weighted_dot * fp16_base_multiplier;
            fp16_base_scale_floor_stats.add(
                saturate_u16(
                    floor_shift_i64(fp16_base_accumulator, 31) +
                    qparams->output.zero_point),
                expected_output);
            fp16_base_scale_round_stats.add(
                saturate_u16(
                    round_shift_htp(fp16_base_accumulator, 31) +
                    qparams->output.zero_point),
                expected_output);
            for (size_t candidate = 0;
                 candidate < scale_significand_candidate_count;
                 ++candidate) {
                const int64_t candidate_accumulator =
                    weighted_dot * scale_significand_multipliers[candidate];
                scale_significand_floor_stats[candidate].add(
                    saturate_u16(
                        floor_shift_i64(candidate_accumulator, 31) +
                        qparams->output.zero_point),
                    expected_output);
                scale_significand_round_stats[candidate].add(
                    saturate_u16(
                        round_shift_htp(candidate_accumulator, 31) +
                        qparams->output.zero_point),
                    expected_output);
            }
            normalized_q31_stats.add(
                saturate_u16(
                    multiply_by_normalized_q31(
                        static_cast<int32_t>(weighted_dot), normalized_multiplier) +
                    qparams->output.zero_point),
                expected_output);
            normalized_q31_single_floor_stats.add(
                saturate_u16(
                    multiply_by_normalized_q31_single_shift(
                        static_cast<int32_t>(weighted_dot),
                        normalized_multiplier,
                        false) +
                    qparams->output.zero_point),
                expected_output);
            normalized_q31_single_round_stats.add(
                saturate_u16(
                    multiply_by_normalized_q31_single_shift(
                        static_cast<int32_t>(weighted_dot),
                        normalized_multiplier,
                        true) +
                    qparams->output.zero_point),
                expected_output);
            const float real_output = static_cast<float>(
                static_cast<long double>(weighted_dot) *
                channel_base_weight_scale *
                static_cast<long double>(qparams->input.scale));
            const float fp16_real_output = ggml_fp16_to_fp32(
                ggml_fp32_to_fp16(real_output));
            const long double fp16_output_code =
                static_cast<long double>(fp16_real_output) /
                static_cast<long double>(qparams->output.scale);
            fp16_real_floor_stats.add(
                saturate_u16(
                    static_cast<int64_t>(std::floor(fp16_output_code)) +
                    qparams->output.zero_point),
                expected_output);
            fp16_real_round_stats.add(
                saturate_u16(
                    static_cast<int64_t>(std::floor(fp16_output_code + 0.5L)) +
                    qparams->output.zero_point),
                expected_output);
            long double fp16_block_real_output = 0.0L;
            for (size_t block_index = 0;
                 block_index < block_accumulators.size();
                 ++block_index) {
                const int64_t block_code = block_scale_codes[block_index];
                const int64_t divisor = channel_multiplier * block_code;
                GGML_ASSERT(divisor > 0 &&
                    block_accumulators[block_index] % divisor == 0);
                const int64_t block_dot = block_accumulators[block_index] / divisor;
                const float effective_scale = static_cast<float>(
                    channel_base_weight_scale * block_code);
                const float fp16_effective_scale = ggml_fp16_to_fp32(
                    ggml_fp32_to_fp16(effective_scale));
                fp16_block_real_output +=
                    static_cast<long double>(block_dot) * fp16_effective_scale;
            }
            const long double fp16_block_output_code =
                fp16_block_real_output *
                static_cast<long double>(qparams->input.scale) /
                static_cast<long double>(qparams->output.scale);
            fp16_block_scale_floor_stats.add(
                saturate_u16(
                    static_cast<int64_t>(std::floor(fp16_block_output_code)) +
                    qparams->output.zero_point),
                expected_output);
            fp16_block_scale_round_stats.add(
                saturate_u16(
                    static_cast<int64_t>(std::floor(
                        fp16_block_output_code + 0.5L)) +
                    qparams->output.zero_point),
                expected_output);
            for (size_t candidate = 0;
                 candidate < block_multiplier_candidate_count;
                 ++candidate) {
                const int multiplier_bits =
                    first_block_multiplier_bits + static_cast<int>(candidate);
                int64_t block_multiplier_accumulator = 0;
                for (size_t block_index = 0;
                     block_index < block_accumulators.size();
                     ++block_index) {
                    const int64_t block_code = block_scale_codes[block_index];
                    const int64_t divisor = channel_multiplier * block_code;
                    GGML_ASSERT(divisor > 0 &&
                        block_accumulators[block_index] % divisor == 0);
                    const int64_t block_dot =
                        block_accumulators[block_index] / divisor;
                    const int64_t block_multiplier =
                        static_cast<int64_t>(std::llround(std::ldexp(
                            nominal_ratio * block_code, multiplier_bits)));
                    block_multiplier_accumulator += block_dot * block_multiplier;
                }
                block_multiplier_floor_stats[candidate].add(
                    saturate_u16(
                        floor_shift_i64(
                            block_multiplier_accumulator, multiplier_bits) +
                        qparams->output.zero_point),
                    expected_output);
                block_multiplier_round_stats[candidate].add(
                    saturate_u16(
                        round_shift_htp(
                            block_multiplier_accumulator, multiplier_bits) +
                        qparams->output.zero_point),
                    expected_output);
            }
            for (size_t candidate = 0;
                 candidate < output_bias_candidate_count;
                 ++candidate) {
                const int bias_numerator =
                    static_cast<int>(candidate) - output_bias_radius;
                const int64_t biased_accumulator =
                    accumulator +
                    static_cast<int64_t>(bias_numerator) *
                        (INT64_C(1) << 31) / output_bias_denominator;
                output_bias_stats[candidate].add(
                    saturate_u16(
                        floor_shift_i64(biased_accumulator, 31) +
                        qparams->output.zero_point),
                    expected_output);
                auto & split_stats = token < fit_token_count
                    ? output_bias_fit_stats[candidate]
                    : output_bias_holdout_stats[candidate];
                split_stats.add(
                    saturate_u16(
                        floor_shift_i64(biased_accumulator, 31) +
                        qparams->output.zero_point),
                    expected_output);
            }
            const int64_t correction_weighted_dot =
                static_cast<int64_t>(qparams->input.zero_point) *
                weighted_weight_sum;
            const int64_t raw_weighted_dot =
                weighted_dot + correction_weighted_dot;
            const int64_t raw_accumulator =
                raw_weighted_dot * channel_multiplier;
            const int64_t correction_accumulator =
                correction_weighted_dot * channel_multiplier;
            const int64_t htp_a16s8_reduced_dot =
                (floor_shift_i64(raw_weighted_dot, 8) -
                    round_shift_htp(correction_weighted_dot, 8)) *
                (INT64_C(1) << 8);
            htp_a16s8_reduced_accumulator_stats.add(
                saturate_u16(
                    round_shift_htp(
                        htp_a16s8_reduced_dot * channel_multiplier, 31) +
                    qparams->output.zero_point),
                expected_output);
            split_zp_floor_floor_stats.add(
                saturate_u16(
                    floor_shift_i64(raw_accumulator, 31) -
                    floor_shift_i64(correction_accumulator, 31) +
                    qparams->output.zero_point),
                expected_output);
            split_zp_floor_round_stats.add(
                saturate_u16(
                    floor_shift_i64(raw_accumulator, 31) -
                    round_shift_htp(correction_accumulator, 31) +
                    qparams->output.zero_point),
                expected_output);
            split_zp_round_round_stats.add(
                saturate_u16(
                    round_shift_htp(raw_accumulator, 31) -
                    round_shift_htp(correction_accumulator, 31) +
                    qparams->output.zero_point),
                expected_output);
            split_zp_round_floor_stats.add(
                saturate_u16(
                    round_shift_htp(raw_accumulator, 31) -
                    floor_shift_i64(correction_accumulator, 31) +
                    qparams->output.zero_point),
                expected_output);
            for (size_t candidate = 0;
                 candidate < split_block_fractional_candidate_count;
                 ++candidate) {
                const int fractional_bits = static_cast<int>(candidate + 1);
                split_block_floor_floor_stats[candidate].add(
                    saturate_u16(
                        requantize_split_zero_point_blocks_fractional(
                            split_block_accumulators, fractional_bits,
                            false, false) +
                        qparams->output.zero_point),
                    expected_output);
                split_block_floor_round_stats[candidate].add(
                    saturate_u16(
                        requantize_split_zero_point_blocks_fractional(
                            split_block_accumulators, fractional_bits,
                            false, true) +
                        qparams->output.zero_point),
                    expected_output);
                split_block_round_floor_stats[candidate].add(
                    saturate_u16(
                        requantize_split_zero_point_blocks_fractional(
                            split_block_accumulators, fractional_bits,
                            true, false) +
                        qparams->output.zero_point),
                    expected_output);
                split_block_round_round_stats[candidate].add(
                    saturate_u16(
                        requantize_split_zero_point_blocks_fractional(
                            split_block_accumulators, fractional_bits,
                            true, true) +
                        qparams->output.zero_point),
                    expected_output);
                const int intermediate_shift = 31 - fractional_bits;
                int64_t floor_ceil_sum = 0;
                int64_t trunc_ceil_sum = 0;
                int64_t trunc_round_sum = 0;
                for (size_t block_index = 0;
                     block_index < split_block_accumulators.raw.size();
                     ++block_index) {
                    const int64_t raw =
                        split_block_accumulators.raw[block_index];
                    const int64_t correction =
                        split_block_accumulators.correction[block_index];
                    floor_ceil_sum +=
                        floor_shift_i64(raw, intermediate_shift) -
                        ceil_shift_i64(correction, intermediate_shift);
                    trunc_ceil_sum +=
                        trunc_shift_i64(raw, intermediate_shift) -
                        ceil_shift_i64(correction, intermediate_shift);
                    trunc_round_sum +=
                        trunc_shift_i64(raw, intermediate_shift) -
                        round_shift_htp(correction, intermediate_shift);
                }
                split_block_floor_ceil_stats[candidate].add(
                    saturate_u16(
                        floor_shift_i64(floor_ceil_sum, fractional_bits) +
                        qparams->output.zero_point),
                    expected_output);
                split_block_trunc_ceil_stats[candidate].add(
                    saturate_u16(
                        floor_shift_i64(trunc_ceil_sum, fractional_bits) +
                        qparams->output.zero_point),
                    expected_output);
                split_block_trunc_round_stats[candidate].add(
                    saturate_u16(
                        floor_shift_i64(trunc_round_sum, fractional_bits) +
                        qparams->output.zero_point),
                    expected_output);
            }
            production_floor_differences += production != floor_output;
        }
        size_t best_row_floor_floor = 0;
        size_t best_row_round_floor = 0;
        for (size_t candidate = 1; candidate < fractional_candidate_count; ++candidate) {
            if (row_fractional_floor_floor_stats[candidate].mean_delta() <
                row_fractional_floor_floor_stats[best_row_floor_floor].mean_delta()) {
                best_row_floor_floor = candidate;
            }
            if (row_fractional_round_floor_stats[candidate].mean_delta() <
                row_fractional_round_floor_stats[best_row_round_floor].mean_delta()) {
                best_row_round_floor = candidate;
            }
        }
        ++row_best_floor_floor_histogram[best_row_floor_floor];
        ++row_best_round_floor_histogram[best_row_round_floor];
        row_oracle_floor_floor_stats.merge(
            row_fractional_floor_floor_stats[best_row_floor_floor]);
        row_oracle_round_floor_stats.merge(
            row_fractional_round_floor_stats[best_row_round_floor]);
        const long double row_fitted_scale_ppm = fit_denominator > 0.0L
            ? ((fit_numerator / fit_denominator) /
                (static_cast<long double>(channel_multiplier) /
                    static_cast<long double>(UINT64_C(1) << 31)) -
                1.0L) *
                1000000.0L
            : 0.0L;
        std::printf(
            "qnn-profile-gguf-linear-row-error: tensor=%s row=%zu "
            "mae=%.6f bias=%.6f max=%u weighted_weight_sum=%lld "
            "channel_multiplier=%lld fitted_scale_ppm=%.6Lf\n",
            tensor_name,
            row,
            row_production_stats.mean_delta(),
            row_production_stats.mean_signed_delta(),
            row_production_stats.max_delta,
            static_cast<long long>(weighted_weight_sum),
            static_cast<long long>(channel_multiplier),
            row_fitted_scale_ppm);
        if (row < 8) {
            std::printf(
                "qnn-profile-gguf-linear-row-fractional: tensor=%s row=%zu "
                "multiplier_exponent=%d floor_floor_bits=%zu floor_floor_mae=%.6f "
                "round_floor_bits=%zu round_floor_mae=%.6f\n",
                tensor_name,
                row,
                channel_multiplier_exponent,
                best_row_floor_floor + 1,
                row_fractional_floor_floor_stats[best_row_floor_floor].mean_delta(),
                best_row_round_floor + 1,
                row_fractional_round_floor_stats[best_row_round_floor].mean_delta());
        }
        if (fit_denominator > 0.0L) {
            const long double fitted_scale = fit_numerator / fit_denominator;
            const long double nominal_scale =
                static_cast<long double>(channel_multiplier) /
                static_cast<long double>(UINT64_C(1) << 31);
            const long double ppm = (fitted_scale / nominal_scale - 1.0L) * 1000000.0L;
            fitted_scale_ppm_sum += ppm;
            fitted_scale_ppm_abs_sum += std::abs(ppm);
            fitted_scale_ppm_min = std::min(fitted_scale_ppm_min, ppm);
            fitted_scale_ppm_max = std::max(fitted_scale_ppm_max, ppm);
            ++fitted_scale_rows;
            for (size_t token = 0; token < token_count; ++token) {
                const int64_t centered = static_cast<int64_t>(
                    std::floor(static_cast<long double>(row_weighted_dots[token]) * fitted_scale));
                const uint16_t fitted_output =
                    saturate_u16(centered + qparams->output.zero_point);
                fitted_scale_stats.add(fitted_output, expected[token * output_rows + row]);
                if (token >= fit_token_count) {
                    fitted_scale_holdout_stats.add(
                        fitted_output, expected[token * output_rows + row]);
                }
            }
        if (scale_interval_lower < scale_interval_upper) {
                ++interval_explainable_rows;
                const long double interval_scale = std::clamp(
                    nominal_scale, scale_interval_lower, scale_interval_upper);
                for (size_t token = 0; token < token_count; ++token) {
                    const int64_t centered = static_cast<int64_t>(
                        std::floor(static_cast<long double>(row_weighted_dots[token]) * interval_scale));
                    const uint16_t interval_output =
                        saturate_u16(centered + qparams->output.zero_point);
                    interval_scale_stats.add(
                        interval_output, expected[token * output_rows + row]);
                }
            }
            nearest_interval_explainable_rows +=
                nearest_interval_lower < nearest_interval_upper;
        size_t best_row_accumulator_bias = 0;
        for (size_t candidate = 1;
             candidate < row_accumulator_bias_stats.size();
             ++candidate) {
            if (row_accumulator_bias_stats[candidate].mean_delta() <
                row_accumulator_bias_stats[best_row_accumulator_bias].mean_delta()) {
                best_row_accumulator_bias = candidate;
            }
        }
        accumulator_bias_row_oracle_stats.merge(
            row_accumulator_bias_stats[best_row_accumulator_bias]);
        ++accumulator_bias_row_histogram[best_row_accumulator_bias];
        }
    }

    std::printf(
        "qnn-profile-gguf-linear-vector-test: tensor=%s tokens=%zu rows=%zu input_width=%zu group_size=%d "
        "production_exact=%zu production_within1=%zu production_mae=%.6f production_bias=%.6f production_max=%u "
        "floor_exact=%zu floor_within1=%zu floor_mae=%.6f floor_bias=%.6f floor_max=%u "
        "htp_round_exact=%zu htp_round_within1=%zu htp_round_mae=%.6f htp_round_bias=%.6f htp_round_max=%u "
        "away_exact=%zu away_within1=%zu away_mae=%.6f away_bias=%.6f away_max=%u "
        "trunc_exact=%zu trunc_within1=%zu trunc_mae=%.6f trunc_bias=%.6f trunc_max=%u "
        "qnn_weight_requant_exact=%zu qnn_weight_requant_within1=%zu "
        "qnn_weight_requant_mae=%.6f qnn_weight_requant_bias=%.6f qnn_weight_requant_max=%u "
        "fitted_exact=%zu fitted_within1=%zu fitted_mae=%.6f fitted_bias=%.6f fitted_max=%u "
        "holdout_tokens=%zu production_holdout_mae=%.6f production_holdout_bias=%.6f "
        "fitted_holdout_mae=%.6f fitted_holdout_bias=%.6f "
        "fitted_scale_ppm_mean=%.3Lf fitted_scale_ppm_abs_mean=%.3Lf "
        "fitted_scale_ppm_min=%.3Lf fitted_scale_ppm_max=%.3Lf "
        "floor_interval_rows=%zu nearest_interval_rows=%zu "
        "interval_exact=%zu interval_mae=%.6f "
        "production_floor_differences=%zu weighted_dot_min=%lld weighted_dot_max=%lld "
        "weighted_dot_i32_overflows=%zu\n",
        tensor_name, token_count, output_rows, input_width, group_size,
        production_stats.exact, production_stats.within_one,
        production_stats.mean_delta(), production_stats.mean_signed_delta(), production_stats.max_delta,
        floor_stats.exact, floor_stats.within_one,
        floor_stats.mean_delta(), floor_stats.mean_signed_delta(), floor_stats.max_delta,
        htp_round_stats.exact, htp_round_stats.within_one,
        htp_round_stats.mean_delta(), htp_round_stats.mean_signed_delta(), htp_round_stats.max_delta,
        away_stats.exact, away_stats.within_one,
        away_stats.mean_delta(), away_stats.mean_signed_delta(), away_stats.max_delta,
        trunc_stats.exact, trunc_stats.within_one,
        trunc_stats.mean_delta(), trunc_stats.mean_signed_delta(), trunc_stats.max_delta,
        qnn_weight_requant_stats.exact, qnn_weight_requant_stats.within_one,
        qnn_weight_requant_stats.mean_delta(),
        qnn_weight_requant_stats.mean_signed_delta(),
        qnn_weight_requant_stats.max_delta,
        fitted_scale_stats.exact, fitted_scale_stats.within_one,
        fitted_scale_stats.mean_delta(), fitted_scale_stats.mean_signed_delta(),
        fitted_scale_stats.max_delta,
        token_count - fit_token_count,
        production_holdout_stats.mean_delta(),
        production_holdout_stats.mean_signed_delta(),
        fitted_scale_holdout_stats.mean_delta(),
        fitted_scale_holdout_stats.mean_signed_delta(),
        fitted_scale_rows == 0 ? 0.0L : fitted_scale_ppm_sum / fitted_scale_rows,
        fitted_scale_rows == 0 ? 0.0L : fitted_scale_ppm_abs_sum / fitted_scale_rows,
        fitted_scale_rows == 0 ? 0.0L : fitted_scale_ppm_min,
        fitted_scale_rows == 0 ? 0.0L : fitted_scale_ppm_max,
        interval_explainable_rows,
        nearest_interval_explainable_rows,
        interval_scale_stats.exact,
        interval_scale_stats.mean_delta(),
        production_floor_differences,
        static_cast<long long>(min_weighted_dot),
        static_cast<long long>(max_weighted_dot),
        weighted_dot_i32_overflows);
    std::printf(
        "qnn-profile-gguf-linear-source-zp: tensor=%s zp0=%zu zp1=%zu zp2=%zu zp3=%zu\n",
        tensor_name,
        source_zero_point_counts[0],
        source_zero_point_counts[1],
        source_zero_point_counts[2],
        source_zero_point_counts[3]);
    std::printf(
        "qnn-profile-gguf-linear-production-requant29: tensor=%s exact=%zu "
        "within1=%zu mae=%.6f bias=%.6f max=%u reference_mismatches=%zu\n",
        tensor_name,
        production_requant29_stats.exact,
        production_requant29_stats.within_one,
        production_requant29_stats.mean_delta(),
        production_requant29_stats.mean_signed_delta(),
        production_requant29_stats.max_delta,
        production_requant29_mismatches);
    for (size_t candidate = 0; candidate < chunk_block_counts.size(); ++candidate) {
        std::printf(
            "qnn-profile-gguf-linear-vector-chunk-test: tensor=%s chunk_blocks=%zu "
            "floor_mae=%.6f floor_bias=%.6f floor_max=%u "
            "round_mae=%.6f round_bias=%.6f round_max=%u\n",
            tensor_name,
            chunk_block_counts[candidate],
            chunk_floor_stats[candidate].mean_delta(),
            chunk_floor_stats[candidate].mean_signed_delta(),
            chunk_floor_stats[candidate].max_delta,
            chunk_round_stats[candidate].mean_delta(),
            chunk_round_stats[candidate].mean_signed_delta(),
            chunk_round_stats[candidate].max_delta);
    }
    auto print_best_fractional = [&](const char * mode, const auto & stats) {
        size_t best = 0;
        for (size_t candidate = 1; candidate < stats.size(); ++candidate) {
            if (stats[candidate].mean_delta() < stats[best].mean_delta()) {
                best = candidate;
            }
        }
        std::printf(
            "qnn-profile-gguf-linear-vector-fractional-test: tensor=%s mode=%s "
            "best_fractional_bits=%zu mae=%.6f bias=%.6f max=%u exact=%zu\n",
            tensor_name, mode, best + 1, stats[best].mean_delta(),
            stats[best].mean_signed_delta(), stats[best].max_delta,
            stats[best].exact);
    };
    print_best_fractional("floor_floor", fractional_floor_floor_stats);
    print_best_fractional("round_floor", fractional_round_floor_stats);
    print_best_fractional("round_round", fractional_round_round_stats);
    auto print_row_fractional_histogram = [&](const char * mode, const auto & histogram) {
        std::printf(
            "qnn-profile-gguf-linear-row-fractional-histogram: tensor=%s mode=%s",
            tensor_name, mode);
        for (size_t candidate = 0; candidate < histogram.size(); ++candidate) {
            if (histogram[candidate] != 0) {
                std::printf(" bits%zu=%zu", candidate + 1, histogram[candidate]);
            }
        }
        std::printf("\n");
    };
    print_row_fractional_histogram("floor_floor", row_best_floor_floor_histogram);
    print_row_fractional_histogram("round_floor", row_best_round_floor_histogram);
    std::printf(
        "qnn-profile-gguf-linear-row-fractional-oracle: tensor=%s "
        "floor_floor_mae=%.6f floor_floor_bias=%.6f floor_floor_max=%u "
        "round_floor_mae=%.6f round_floor_bias=%.6f round_floor_max=%u\n",
        tensor_name,
        row_oracle_floor_floor_stats.mean_delta(),
        row_oracle_floor_floor_stats.mean_signed_delta(),
        row_oracle_floor_floor_stats.max_delta,
        row_oracle_round_floor_stats.mean_delta(),
        row_oracle_round_floor_stats.mean_signed_delta(),
        row_oracle_round_floor_stats.max_delta);
    auto print_best_multiplier_exponent_rule = [&](const char * mode, const auto & stats) {
        size_t best = 0;
        for (size_t candidate = 1; candidate < stats.size(); ++candidate) {
            if (stats[candidate].mean_delta() < stats[best].mean_delta()) {
                best = candidate;
            }
        }
        std::printf(
            "qnn-profile-gguf-linear-multiplier-exponent-rule: tensor=%s mode=%s "
            "constant=%d mae=%.6f bias=%.6f max=%u exact=%zu\n",
            tensor_name,
            mode,
            first_multiplier_exponent_constant + static_cast<int>(best),
            stats[best].mean_delta(),
            stats[best].mean_signed_delta(),
            stats[best].max_delta,
            stats[best].exact);
    };
    print_best_multiplier_exponent_rule(
        "floor_floor", multiplier_exponent_floor_floor_stats);
    print_best_multiplier_exponent_rule(
        "round_floor", multiplier_exponent_round_floor_stats);
    for (const int constant : {28, 29, 30, 31}) {
        const size_t candidate = static_cast<size_t>(
            constant - first_multiplier_exponent_constant);
        const auto & floor_floor =
            multiplier_exponent_floor_floor_stats[candidate];
        const auto & round_floor =
            multiplier_exponent_round_floor_stats[candidate];
        const auto & floor_floor_holdout =
            multiplier_exponent_floor_floor_holdout_stats[candidate];
        const auto & round_floor_holdout =
            multiplier_exponent_round_floor_holdout_stats[candidate];
        std::printf(
            "qnn-profile-gguf-linear-multiplier-exponent-candidate: tensor=%s "
            "constant=%d mae=%.6f bias=%.6f max=%u exact=%zu "
            "holdout_mae=%.6f holdout_bias=%.6f\n",
            tensor_name,
            constant,
            floor_floor.mean_delta(),
            floor_floor.mean_signed_delta(),
            floor_floor.max_delta,
            floor_floor.exact,
            floor_floor_holdout.mean_delta(),
            floor_floor_holdout.mean_signed_delta());
        std::printf(
            "qnn-profile-gguf-linear-multiplier-exponent-round-floor-candidate: "
            "tensor=%s constant=%d mae=%.6f bias=%.6f max=%u exact=%zu "
            "holdout_mae=%.6f holdout_bias=%.6f\n",
            tensor_name,
            constant,
            round_floor.mean_delta(),
            round_floor.mean_signed_delta(),
            round_floor.max_delta,
            round_floor.exact,
            round_floor_holdout.mean_delta(),
            round_floor_holdout.mean_signed_delta());
    }
    std::printf(
        "qnn-profile-gguf-linear-vector-float32-test: tensor=%s "
        "floor_mae=%.6f floor_bias=%.6f floor_max=%u "
        "round_mae=%.6f round_bias=%.6f round_max=%u\n",
        tensor_name,
        float32_floor_stats.mean_delta(), float32_floor_stats.mean_signed_delta(),
        float32_floor_stats.max_delta,
        float32_round_stats.mean_delta(), float32_round_stats.mean_signed_delta(),
        float32_round_stats.max_delta);
    auto print_best_scale_significand = [&](const char * mode, const auto & stats) {
        size_t best = 0;
        for (size_t candidate = 1; candidate < stats.size(); ++candidate) {
            if (stats[candidate].mean_delta() < stats[best].mean_delta()) {
                best = candidate;
            }
        }
        std::printf(
            "qnn-profile-gguf-linear-vector-scale-significand-test: tensor=%s mode=%s "
            "best_significand_bits=%d mae=%.6f bias=%.6f max=%u exact=%zu\n",
            tensor_name, mode,
            first_scale_significand_bits + static_cast<int>(best),
            stats[best].mean_delta(), stats[best].mean_signed_delta(),
            stats[best].max_delta, stats[best].exact);
    };
    print_best_scale_significand("floor", scale_significand_floor_stats);
    print_best_scale_significand("round", scale_significand_round_stats);
    std::printf(
        "qnn-profile-gguf-linear-vector-fp16-base-test: tensor=%s "
        "floor_mae=%.6f floor_bias=%.6f floor_max=%u floor_exact=%zu "
        "round_mae=%.6f round_bias=%.6f round_max=%u round_exact=%zu\n",
        tensor_name,
        fp16_base_scale_floor_stats.mean_delta(),
        fp16_base_scale_floor_stats.mean_signed_delta(),
        fp16_base_scale_floor_stats.max_delta,
        fp16_base_scale_floor_stats.exact,
        fp16_base_scale_round_stats.mean_delta(),
        fp16_base_scale_round_stats.mean_signed_delta(),
        fp16_base_scale_round_stats.max_delta,
        fp16_base_scale_round_stats.exact);
    std::printf(
        "qnn-profile-gguf-linear-vector-normalized-q31-test: tensor=%s "
        "double_round_mae=%.6f double_round_bias=%.6f double_round_max=%u "
        "double_round_exact=%zu double_round_within1=%zu "
        "single_floor_mae=%.6f single_floor_bias=%.6f single_floor_max=%u "
        "single_floor_exact=%zu single_floor_within1=%zu "
        "single_round_mae=%.6f single_round_bias=%.6f single_round_max=%u "
        "single_round_exact=%zu single_round_within1=%zu\n",
        tensor_name,
        normalized_q31_stats.mean_delta(),
        normalized_q31_stats.mean_signed_delta(),
        normalized_q31_stats.max_delta,
        normalized_q31_stats.exact,
        normalized_q31_stats.within_one,
        normalized_q31_single_floor_stats.mean_delta(),
        normalized_q31_single_floor_stats.mean_signed_delta(),
        normalized_q31_single_floor_stats.max_delta,
        normalized_q31_single_floor_stats.exact,
        normalized_q31_single_floor_stats.within_one,
        normalized_q31_single_round_stats.mean_delta(),
        normalized_q31_single_round_stats.mean_signed_delta(),
        normalized_q31_single_round_stats.max_delta,
        normalized_q31_single_round_stats.exact,
        normalized_q31_single_round_stats.within_one);
    std::printf(
        "qnn-profile-gguf-linear-vector-fp16-real-test: tensor=%s "
        "floor_mae=%.6f floor_bias=%.6f floor_max=%u floor_exact=%zu "
        "round_mae=%.6f round_bias=%.6f round_max=%u round_exact=%zu\n",
        tensor_name,
        fp16_real_floor_stats.mean_delta(),
        fp16_real_floor_stats.mean_signed_delta(),
        fp16_real_floor_stats.max_delta,
        fp16_real_floor_stats.exact,
        fp16_real_round_stats.mean_delta(),
        fp16_real_round_stats.mean_signed_delta(),
        fp16_real_round_stats.max_delta,
        fp16_real_round_stats.exact);
    std::printf(
        "qnn-profile-gguf-linear-vector-fp16-block-scale-test: tensor=%s "
        "floor_mae=%.6f floor_bias=%.6f floor_max=%u floor_exact=%zu "
        "round_mae=%.6f round_bias=%.6f round_max=%u round_exact=%zu\n",
        tensor_name,
        fp16_block_scale_floor_stats.mean_delta(),
        fp16_block_scale_floor_stats.mean_signed_delta(),
        fp16_block_scale_floor_stats.max_delta,
        fp16_block_scale_floor_stats.exact,
        fp16_block_scale_round_stats.mean_delta(),
        fp16_block_scale_round_stats.mean_signed_delta(),
        fp16_block_scale_round_stats.max_delta,
        fp16_block_scale_round_stats.exact);
    auto print_best_block_multiplier = [&](const char * mode, const auto & stats) {
        size_t best = 0;
        for (size_t candidate = 1; candidate < stats.size(); ++candidate) {
            if (stats[candidate].mean_delta() < stats[best].mean_delta()) {
                best = candidate;
            }
        }
        std::printf(
            "qnn-profile-gguf-linear-vector-block-multiplier-test: tensor=%s "
            "mode=%s best_multiplier_bits=%d mae=%.6f bias=%.6f max=%u exact=%zu\n",
            tensor_name, mode,
            first_block_multiplier_bits + static_cast<int>(best),
            stats[best].mean_delta(), stats[best].mean_signed_delta(),
            stats[best].max_delta, stats[best].exact);
    };
    print_best_block_multiplier("floor", block_multiplier_floor_stats);
    print_best_block_multiplier("round", block_multiplier_round_stats);
    size_t best_accumulator_bias = 0;
    size_t best_fit_accumulator_bias = 0;
    for (size_t candidate = 1;
         candidate < accumulator_bias_stats.size();
         ++candidate) {
        if (accumulator_bias_stats[candidate].mean_delta() <
            accumulator_bias_stats[best_accumulator_bias].mean_delta()) {
            best_accumulator_bias = candidate;
        }
        if (accumulator_bias_fit_stats[candidate].mean_delta() <
            accumulator_bias_fit_stats[best_fit_accumulator_bias].mean_delta()) {
            best_fit_accumulator_bias = candidate;
        }
    }
    std::printf(
        "qnn-profile-gguf-linear-vector-accumulator-bias-test: tensor=%s "
        "best_bias=%d mae=%.6f bias=%.6f max=%u exact=%zu "
        "fit_best_bias=%d fit_mae=%.6f holdout_mae=%.6f "
        "holdout_bias=%.6f holdout_max=%u row_oracle_mae=%.6f "
        "row_oracle_bias=%.6f row_oracle_max=%u\n",
        tensor_name,
        static_cast<int>(best_accumulator_bias) - accumulator_bias_radius,
        accumulator_bias_stats[best_accumulator_bias].mean_delta(),
        accumulator_bias_stats[best_accumulator_bias].mean_signed_delta(),
        accumulator_bias_stats[best_accumulator_bias].max_delta,
        accumulator_bias_stats[best_accumulator_bias].exact,
        static_cast<int>(best_fit_accumulator_bias) - accumulator_bias_radius,
        accumulator_bias_fit_stats[best_fit_accumulator_bias].mean_delta(),
        accumulator_bias_holdout_stats[best_fit_accumulator_bias].mean_delta(),
        accumulator_bias_holdout_stats[best_fit_accumulator_bias].mean_signed_delta(),
        accumulator_bias_holdout_stats[best_fit_accumulator_bias].max_delta,
        accumulator_bias_row_oracle_stats.mean_delta(),
        accumulator_bias_row_oracle_stats.mean_signed_delta(),
        accumulator_bias_row_oracle_stats.max_delta);
    std::printf(
        "qnn-profile-gguf-linear-row-accumulator-bias-histogram: tensor=%s",
        tensor_name);
    for (size_t candidate = 0;
         candidate < accumulator_bias_row_histogram.size();
         ++candidate) {
        if (accumulator_bias_row_histogram[candidate] == 0) {
            continue;
        }
        std::printf(
            " bias%d=%zu",
            static_cast<int>(candidate) - accumulator_bias_radius,
            accumulator_bias_row_histogram[candidate]);
    }
    std::printf("\n");
    size_t best_output_bias = 0;
    size_t best_fit_output_bias = 0;
    for (size_t candidate = 1;
         candidate < output_bias_stats.size();
         ++candidate) {
        if (output_bias_stats[candidate].mean_delta() <
            output_bias_stats[best_output_bias].mean_delta()) {
            best_output_bias = candidate;
        }
        if (output_bias_fit_stats[candidate].mean_delta() <
            output_bias_fit_stats[best_fit_output_bias].mean_delta()) {
            best_fit_output_bias = candidate;
        }
    }
    const int best_bias_numerator =
        static_cast<int>(best_output_bias) - output_bias_radius;
    std::printf(
        "qnn-profile-gguf-linear-vector-output-bias-test: tensor=%s "
        "best_bias=%+.6f mae=%.6f bias=%.6f max=%u exact=%zu within1=%zu\n",
        tensor_name,
        static_cast<double>(best_bias_numerator) / output_bias_denominator,
        output_bias_stats[best_output_bias].mean_delta(),
        output_bias_stats[best_output_bias].mean_signed_delta(),
        output_bias_stats[best_output_bias].max_delta,
        output_bias_stats[best_output_bias].exact,
        output_bias_stats[best_output_bias].within_one);
    const int best_fit_bias_numerator =
        static_cast<int>(best_fit_output_bias) - output_bias_radius;
    std::printf(
        "qnn-profile-gguf-linear-vector-output-bias-holdout: tensor=%s "
        "fit_bias=%+.6f fit_mae=%.6f holdout_mae=%.6f holdout_bias=%.6f "
        "holdout_max=%u holdout_exact=%zu holdout_within1=%zu\n",
        tensor_name,
        static_cast<double>(best_fit_bias_numerator) / output_bias_denominator,
        output_bias_fit_stats[best_fit_output_bias].mean_delta(),
        output_bias_holdout_stats[best_fit_output_bias].mean_delta(),
        output_bias_holdout_stats[best_fit_output_bias].mean_signed_delta(),
        output_bias_holdout_stats[best_fit_output_bias].max_delta,
        output_bias_holdout_stats[best_fit_output_bias].exact,
        output_bias_holdout_stats[best_fit_output_bias].within_one);
    std::printf(
        "qnn-profile-gguf-linear-vector-split-zp-test: tensor=%s "
        "floor_floor_mae=%.6f floor_floor_bias=%.6f floor_floor_max=%u "
        "floor_round_mae=%.6f floor_round_bias=%.6f floor_round_max=%u "
        "round_round_mae=%.6f round_round_bias=%.6f round_round_max=%u "
        "round_floor_mae=%.6f round_floor_bias=%.6f round_floor_max=%u\n",
        tensor_name,
        split_zp_floor_floor_stats.mean_delta(),
        split_zp_floor_floor_stats.mean_signed_delta(),
        split_zp_floor_floor_stats.max_delta,
        split_zp_floor_round_stats.mean_delta(),
        split_zp_floor_round_stats.mean_signed_delta(),
        split_zp_floor_round_stats.max_delta,
        split_zp_round_round_stats.mean_delta(),
        split_zp_round_round_stats.mean_signed_delta(),
        split_zp_round_round_stats.max_delta,
        split_zp_round_floor_stats.mean_delta(),
        split_zp_round_floor_stats.mean_signed_delta(),
        split_zp_round_floor_stats.max_delta);
    std::printf(
        "qnn-profile-gguf-linear-vector-htp-a16s8-reduced-accumulator-test: "
        "tensor=%s mae=%.6f bias=%.6f max=%u exact=%zu within1=%zu\n",
        tensor_name,
        htp_a16s8_reduced_accumulator_stats.mean_delta(),
        htp_a16s8_reduced_accumulator_stats.mean_signed_delta(),
        htp_a16s8_reduced_accumulator_stats.max_delta,
        htp_a16s8_reduced_accumulator_stats.exact,
        htp_a16s8_reduced_accumulator_stats.within_one);
    auto print_best_split_block_fractional =
        [&](const char * mode, const auto & stats) {
            size_t best = 0;
            for (size_t candidate = 1; candidate < stats.size(); ++candidate) {
                if (stats[candidate].mean_delta() <
                    stats[best].mean_delta()) {
                    best = candidate;
                }
            }
            std::printf(
                "qnn-profile-gguf-linear-vector-split-block-zp-test: "
                "tensor=%s mode=%s fractional_bits=%zu mae=%.6f "
                "bias=%.6f max=%u exact=%zu within1=%zu\n",
                tensor_name, mode, best + 1,
                stats[best].mean_delta(),
                stats[best].mean_signed_delta(),
                stats[best].max_delta,
                stats[best].exact,
                stats[best].within_one);
        };
    print_best_split_block_fractional(
        "floor_floor", split_block_floor_floor_stats);
    print_best_split_block_fractional(
        "floor_round", split_block_floor_round_stats);
    print_best_split_block_fractional(
        "round_floor", split_block_round_floor_stats);
    print_best_split_block_fractional(
        "round_round", split_block_round_round_stats);
    print_best_split_block_fractional(
        "floor_ceil", split_block_floor_ceil_stats);
    print_best_split_block_fractional(
        "trunc_ceil", split_block_trunc_ceil_stats);
    print_best_split_block_fractional(
        "trunc_round", split_block_trunc_round_stats);
    gguf_free(gguf);
    ggml_free(tensor_context);
    return 0;
}

kernel_backend parse_backend(std::string_view value) {
    if (value == "scalar") {
        return kernel_backend::scalar;
    }
    if (value == "neon") {
        return kernel_backend::neon;
    }
    if (value == "compare") {
        return kernel_backend::compare;
    }
    std::fprintf(stderr, "unknown backend: %.*s\n", static_cast<int>(value.size()), value.data());
    std::exit(2);
}

int run_u16_source_group_kernel_test(size_t group_size) {
    constexpr size_t input_size = 128;
    constexpr size_t output_size = 5;
    const size_t groups = input_size / group_size;
    const size_t code_bytes = group_size / 4U;
    const size_t source_block_bytes = code_bytes + 4U;

    qnn_a16w2_packed_plan plan {
        input_size,
        output_size,
        group_size,
        { 3.075e-5, 31241 },
        { 1.937e-4, 28719 },
        std::vector<uint8_t>(output_size * groups * code_bytes),
        std::vector<int8_t>(output_size * groups),
        std::vector<fixed_multiplier>(output_size * groups),
    };
    plan.input.scale = static_cast<double>(static_cast<float>(plan.input.scale));
    plan.output.scale = static_cast<double>(static_cast<float>(plan.output.scale));

    std::vector<uint16_t> input(input_size);
    std::vector<uint8_t> source(output_size * groups * source_block_bytes);
    for (size_t column = 0; column < input.size(); ++column) {
        input[column] = static_cast<uint16_t>((column * 7919U + 913U) & 0xffffU);
    }
    for (size_t row = 0; row < output_size; ++row) {
        for (size_t group = 0; group < groups; ++group) {
            const size_t group_index = row * groups + group;
            uint8_t * const source_block =
                source.data() + group_index * source_block_bytes;
            plan.weight_zero_points[group_index] =
                static_cast<int8_t>((row + group) % 4U);
            for (size_t column = 0; column < group_size; ++column) {
                const uint8_t code =
                    static_cast<uint8_t>((row * 11U + (group * group_size + column) * 7U) % 4U);
                const size_t byte_index = column / 4U;
                source_block[byte_index] |=
                    static_cast<uint8_t>(code << ((column & 3U) * 2U));
            }
            std::memcpy(
                plan.packed_weight_codes.data() + group_index * code_bytes,
                source_block,
                code_bytes);
            const float source_scale =
                0.00115f + 0.00017f * static_cast<float>(group_index + 1U);
            const ggml_fp16_t scale_fp16 = ggml_fp32_to_fp16(source_scale);
            const ggml_fp16_t zero_bias_fp16 = ggml_fp32_to_fp16(
                source_scale * static_cast<float>(plan.weight_zero_points[group_index]));
            std::memcpy(source_block + code_bytes, &scale_fp16, sizeof(scale_fp16));
            std::memcpy(source_block + code_bytes + 2U, &zero_bias_fp16, sizeof(zero_bias_fp16));
            const double qnn_weight_scale = std::max(
                static_cast<double>(ggml_fp16_to_fp32(scale_fp16)), 1.0e-4);
            plan.group_multipliers[group_index] = make_multiplier(
                plan.input.scale * qnn_weight_scale / plan.output.scale);
        }
    }

    std::vector<uint16_t> reference(output_size);
    std::vector<uint16_t> production(output_size);
    qnn_a16w2_packed_gemv_scalar(plan, input.data(), reference.data());
    for (size_t row = 0; row < output_size; ++row) {
        const void * const weights = source.data() + row * groups * source_block_bytes;
        switch (group_size) {
            case 64:
                ggml_vec_dot_gptq2_64_u16_qnn(
                    static_cast<int>(input_size), production.data() + row, weights,
                    input.data(), static_cast<float>(plan.input.scale), plan.input.zero_point,
                    static_cast<float>(plan.output.scale), plan.output.zero_point);
                break;
            case 128:
                ggml_vec_dot_gptq2_128_u16_qnn(
                    static_cast<int>(input_size), production.data() + row, weights,
                    input.data(), static_cast<float>(plan.input.scale), plan.input.zero_point,
                    static_cast<float>(plan.output.scale), plan.output.zero_point);
                break;
            default:
                std::fprintf(stderr, "unsupported source group test: %zu\n", group_size);
                return 8;
        }
    }

    const bool match = same_values(reference.data(), production.data(), output_size);
    std::printf(
        "u16-core-source-group-test: group_size=%zu source_block_bytes=%zu "
        "result_match=%d activation_dequant_buffers=0 packed_int4_buffers=0\n",
        group_size, source_block_bytes, match ? 1 : 0);
    return match ? 0 : 9;
}

int run_gptq2_u16_gemv_4row_test(int multithread_test_threads = 0) {
    constexpr int input_size = 2048;
    constexpr int output_size = 4096;
    constexpr int blocks = input_size / 32;
    constexpr size_t weight_row_stride = blocks * 12U;
    constexpr size_t metadata_row_stride = blocks;
    constexpr int activation_zero_point = 32711;
    constexpr int output_zero_point = 30119;
    constexpr int fractional_constant = 29;

    std::vector<uint16_t> activations(input_size);
    std::vector<float> activations_f32(input_size);
    std::vector<uint8_t> weights(output_size * weight_row_stride);
    std::vector<uint8_t> gs32_weights(output_size * weight_row_stride);
    std::vector<uint8_t> metadata(output_size * metadata_row_stride);
    std::vector<int64_t> channel_scales(output_size);
    std::vector<int64_t> prepared_weight_sums(output_size);
    std::vector<int32_t> activation_block_sums(blocks);
    std::vector<uint16_t> scalar_outputs(output_size);
    std::vector<uint16_t> row4_outputs(output_size);
    std::vector<uint16_t> row8_outputs(output_size);
    std::vector<uint16_t> row16_outputs(output_size);
    std::vector<uint16_t> gs32_row16_outputs(output_size);
    std::vector<float> f32_outputs(output_size);
    constexpr int q2_k_blocks_per_row = input_size / QK_K;
    std::vector<block_q2_K> q2_k_weights(
        static_cast<size_t>(output_size) * q2_k_blocks_per_row);
    std::vector<block_q8_K> q8_k_activations(input_size / QK_K);
    std::vector<float> q2_k_outputs(output_size);
    std::vector<float> q2_k_weight_row(input_size);

    for (int index = 0; index < input_size; ++index) {
        const int32_t centered = (index * 1229 + 811) % 8191 - 4095;
        activations[index] = static_cast<uint16_t>(activation_zero_point + centered);
        activations_f32[index] = centered * 0.00025f;
    }
    for (int row = 0; row < output_size; ++row) {
        channel_scales[row] = 350000 + (row * 7919) % 900000;
        for (int block = 0; block < blocks; ++block) {
            uint8_t * codes = weights.data() + row * weight_row_stride + block * 12U;
            for (int packed = 0; packed < 8; ++packed) {
                uint8_t value = 0;
                for (int lane = 0; lane < 4; ++lane) {
                    const int column = block * 32 + packed * 4 + lane;
                    const uint8_t code = static_cast<uint8_t>(
                        (row * 13 + column * 7 + block * 3) & 0x3);
                    value |= static_cast<uint8_t>(code << (lane * 2));
                }
                codes[packed] = value;
            }
            const uint8_t zero_point = static_cast<uint8_t>((row + block) & 0x3);
            const uint8_t block_scale = static_cast<uint8_t>(1 + (row * 5 + block * 3) % 16);
            metadata[row * metadata_row_stride + block] =
                static_cast<uint8_t>((zero_point << 5) | block_scale);
            const ggml_fp16_t scale_fp16 =
                ggml_fp32_to_fp16(0.001f * block_scale);
            const ggml_fp16_t zero_fp16 =
                ggml_fp32_to_fp16(0.001f * block_scale * zero_point);
            std::memcpy(codes + 8, &scale_fp16, sizeof(scale_fp16));
            std::memcpy(codes + 10, &zero_fp16, sizeof(zero_fp16));
        }
        prepared_weight_sums[row] =
            ggml_gptq2_32_qnn_prepared_weight_sum(
                input_size,
                weights.data() + row * weight_row_stride,
                metadata.data() + row * metadata_row_stride);
        for (int column = 0; column < input_size; ++column) {
            q2_k_weight_row[column] =
                0.015625f * static_cast<float>(
                    ((row * 13 + column * 7 + (column / 32) * 3) & 0x3) - 2);
        }
        quantize_row_q2_K(
            q2_k_weight_row.data(),
            q2_k_weights.data() + static_cast<size_t>(row) * q2_k_blocks_per_row,
            input_size);
    }
    quantize_row_q8_K(
        activations_f32.data(), q8_k_activations.data(), input_size);

    // Match gs32_source_v1 exactly: each 64-row/32-column source group keeps
    // four 2-byte qcode pairs in row-interleaved 16-byte tiles, followed by
    // the four metadata bytes for all 64 rows.
    for (int row_block = 0; row_block < output_size / 64; ++row_block) {
        uint8_t * gs32_row_block = gs32_weights.data() +
            static_cast<size_t>(row_block) * 64 * weight_row_stride;
        for (int block = 0; block < blocks; ++block) {
            uint8_t * source_group =
                gs32_row_block + static_cast<size_t>(block) * 768;
            for (int row = 0; row < 64; ++row) {
                const uint8_t * row_source = weights.data() +
                    static_cast<size_t>(row_block * 64 + row) *
                        weight_row_stride +
                    static_cast<size_t>(block) * 12;
                const size_t row_outer = static_cast<size_t>(row) / 32;
                const size_t row_middle =
                    (static_cast<size_t>(row) % 32) / 8;
                const size_t row_inner = static_cast<size_t>(row) % 8;
                for (size_t pair = 0; pair < 4; ++pair) {
                    const size_t destination_offset =
                        ((((row_outer * 4 + pair) * 4 + row_middle) * 8 +
                            row_inner) * 2);
                    std::memcpy(
                        source_group + destination_offset,
                        row_source + pair * 2, 2);
                }
                std::memcpy(source_group + 512 + row * 4, row_source + 8, 4);
            }
        }
    }
    std::vector<uint8_t> tiled_metadata(metadata.size());
    for (int row = 0; row < output_size; ++row) {
        const size_t tile = static_cast<size_t>(row) / 8;
        const size_t lane = static_cast<size_t>(row) % 8;
        for (int block = 0; block < blocks; ++block) {
            tiled_metadata[
                (tile * static_cast<size_t>(blocks) + block) * 8 + lane] =
                metadata[
                    static_cast<size_t>(row) * metadata_row_stride + block];
        }
    }
    int activation_range = 1;
    for (int block = 0; block < blocks; ++block) {
        int32_t sum = 0;
        for (int lane = 0; lane < 32; ++lane) {
            const int32_t centered =
                static_cast<int32_t>(activations[block * 32 + lane]) -
                activation_zero_point;
            sum += centered;
            activation_range = activation_range &&
                centered >= std::numeric_limits<int16_t>::min() &&
                centered <= std::numeric_limits<int16_t>::max();
        }
        activation_block_sums[block] = sum;
    }
    std::vector<uint8_t> activation_low(input_size);
    std::vector<int8_t> activation_high(input_size);

    const auto run_scalar = [&]() {
        for (int row = 0; row < output_size; ++row) {
            ggml_vec_dot_gptq2_u16_qnn_blockwise_affine(
                input_size, scalar_outputs.data() + row,
                weights.data() + row * weight_row_stride,
                activations.data(),
                metadata.data() + row * metadata_row_stride,
                channel_scales[row], activation_zero_point, output_zero_point,
                32, 1, fractional_constant, 0, 0);
        }
    };
    const auto run_4rows = [&]() {
        for (int row = 0; row < output_size; row += 4) {
            ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_4rows(
                input_size, row4_outputs.data() + row,
                weights.data() + row * weight_row_stride,
                weight_row_stride, activations.data(),
                activation_block_sums.data(), activation_range,
                metadata.data() + row * metadata_row_stride,
                metadata_row_stride, channel_scales.data() + row,
                prepared_weight_sums.data() + row,
                activation_zero_point, output_zero_point,
                fractional_constant, 0, 0);
        }
    };
    const auto run_8rows = [&]() {
        for (int row = 0; row < output_size; row += 8) {
            ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_8rows(
                input_size, row8_outputs.data() + row,
                weights.data() + row * weight_row_stride,
                weight_row_stride, activations.data(),
                activation_block_sums.data(), activation_range,
                metadata.data() + row * metadata_row_stride,
                metadata_row_stride, channel_scales.data() + row,
                prepared_weight_sums.data() + row,
                activation_zero_point, output_zero_point,
                fractional_constant, 0, 0);
        }
    };
    const auto run_gs32_8rows = [&]() {
        activation_range = ggml_gptq2_32_prepare_u16_activation(
            input_size, activations.data(), activation_zero_point,
            activation_block_sums.data(),
            activation_low.data(), activation_high.data(), nullptr);
        for (int row = 0; row < output_size; row += 8) {
            ggml_vec_dot_gptq2_32_gs32_u16_qnn_blockwise_affine_8rows(
                input_size, row8_outputs.data() + row,
                gs32_weights.data(), row, activations.data(),
                activation_low.data(),
                activation_range == GGML_GPTQ2_U16_ACTIVATION_I16
                    ? activation_high.data()
                    : nullptr,
                activation_block_sums.data(), activation_range,
                tiled_metadata.data() + row * metadata_row_stride,
                0, channel_scales.data() + row,
                prepared_weight_sums.data() + row,
                activation_zero_point, output_zero_point,
                fractional_constant, 0, 0);
        }
    };
    const auto run_gs32_16rows = [&]() {
        int32_t activation_abs_sum = 0;
        activation_range = ggml_gptq2_32_prepare_u16_activation(
            input_size, activations.data(), activation_zero_point,
            activation_block_sums.data(),
            activation_low.data(), activation_high.data(), &activation_abs_sum);
        const bool accumulation_fits_i32 =
            static_cast<int64_t>(activation_abs_sum) * 7 * 31 <= INT32_MAX;
        for (int row = 0; row < output_size; row += 16) {
            ggml_vec_dot_gptq2_32_gs32_u16_qnn_blockwise_affine_16rows(
                input_size, gs32_row16_outputs.data() + row,
                gs32_weights.data(), row, activations.data(),
                activation_low.data(),
                activation_range == GGML_GPTQ2_U16_ACTIVATION_I16
                    ? activation_high.data()
                    : nullptr,
                activation_block_sums.data(), activation_range,
                accumulation_fits_i32,
                tiled_metadata.data() + row * metadata_row_stride,
                0, channel_scales.data() + row,
                prepared_weight_sums.data() + row,
                activation_zero_point, output_zero_point,
                fractional_constant, 0, 0);
        }
    };
    const auto run_16rows = [&]() {
        for (int row = 0; row < output_size; row += 16) {
            ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_16rows(
                input_size, row16_outputs.data() + row,
                weights.data() + row * weight_row_stride,
                weight_row_stride, activations.data(),
                activation_block_sums.data(), activation_range,
                metadata.data() + row * metadata_row_stride,
                metadata_row_stride, channel_scales.data() + row,
                prepared_weight_sums.data() + row,
                activation_zero_point, output_zero_point,
                fractional_constant, 0, 0);
        }
    };
    const auto run_f32 = [&]() {
        for (int row = 0; row < output_size; ++row) {
            ggml_vec_dot_gptq2_32_f32(
                input_size, f32_outputs.data() + row, sizeof(float),
                weights.data() + row * weight_row_stride, input_size,
                activations_f32.data(), input_size, 1);
        }
    };
    const auto run_q2_k = [&]() {
        for (int row = 0; row < output_size; ++row) {
            ggml_vec_dot_q2_K_q8_K(
                input_size, q2_k_outputs.data() + row, sizeof(float),
                q2_k_weights.data() +
                    static_cast<size_t>(row) * q2_k_blocks_per_row,
                input_size, q8_k_activations.data(), input_size, 1);
        }
    };

    if (multithread_test_threads > 0) {
        run_gs32_8rows();
        const std::vector<uint16_t> reference_outputs = row8_outputs;

        llama_qnn_linear_qparams qparams;
        qparams.projection = "self_attn.q_proj";
        qparams.input.scale = 2.0f;
        qparams.input.zero_point = activation_zero_point;
        qparams.output.scale = 1.0f;
        qparams.output.zero_point = output_zero_point;
        qparams.activation_to_output_q20 = 1;
        qparams.qnn_weight_block_size = 32;
        qparams.qnn_weight_blocks_per_row = blocks;
        qparams.qnn_channel_scale_to_output_q31 = channel_scales;
        qparams.qnn_weight_block_scale_codes = tiled_metadata;
        qparams.qnn_prepared_weight_sums = prepared_weight_sums;
        qparams.qnn_weight_block_codes_prepared = true;
        qparams.weights_gs32_source = true;
        qparams.qnn_weight_block_code_layout =
            LLAMA_QNN_BLOCK_CODES_GS32_TILE8_BLOCK_MAJOR;

        ggml_init_params graph_params {
            /* .mem_size = */ ggml_tensor_overhead() * 16 +
                ggml_graph_overhead_custom(8, false),
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        ggml_context * graph_ctx = ggml_init(graph_params);
        if (graph_ctx == nullptr) {
            return 18;
        }
        ggml_tensor * graph_weights = ggml_new_tensor_2d(
            graph_ctx, GGML_TYPE_GPTQ2_32, input_size, output_size);
        ggml_backend_buffer_t weight_buffer =
            ggml_backend_alloc_ctx_tensors_from_buft(
                graph_ctx, ggml_backend_cpu_buffer_type());
        if (weight_buffer == nullptr) {
            ggml_free(graph_ctx);
            return 19;
        }
        ggml_backend_tensor_set(
            graph_weights, gs32_weights.data(), 0, gs32_weights.size());

        ggml_tensor * graph_input =
            ggml_new_tensor_2d(graph_ctx, GGML_TYPE_U16, input_size, 1);
        ggml_tensor * graph_output = llama_qnn_u16_mul_mat(
            graph_ctx, graph_weights, graph_input, &qparams);
        ggml_backend_buffer_t compute_buffer =
            ggml_backend_alloc_ctx_tensors_from_buft(
                graph_ctx, ggml_backend_cpu_buffer_type());
        if (compute_buffer == nullptr) {
            ggml_backend_buffer_free(weight_buffer);
            ggml_free(graph_ctx);
            return 20;
        }
        ggml_backend_tensor_set(
            graph_input, activations.data(), 0,
            activations.size() * sizeof(activations[0]));

        ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, 8, false);
        ggml_build_forward_expand(graph, graph_output);
        ggml_backend_t backend = ggml_backend_cpu_init();
        ggml_threadpool_params threadpool_params =
            ggml_threadpool_params_default(multithread_test_threads);
        ggml_threadpool * threadpool = ggml_threadpool_new(&threadpool_params);
        ggml_backend_cpu_set_n_threads(backend, multithread_test_threads);
        ggml_backend_cpu_set_threadpool(backend, threadpool);

        const ggml_status warmup_status =
            ggml_backend_graph_compute(backend, graph);
        std::array<double, 4> elapsed_ms {};
        ggml_status status = warmup_status;
        for (int iteration = 0; iteration < 4 && status == GGML_STATUS_SUCCESS;
             ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            status = ggml_backend_graph_compute(backend, graph);
            elapsed_ms[iteration] =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
        }
        std::vector<uint16_t> graph_outputs(output_size);
        ggml_backend_tensor_get(
            graph_output, graph_outputs.data(), 0,
            graph_outputs.size() * sizeof(graph_outputs[0]));
        const bool graph_exact = graph_outputs == reference_outputs;
        uint64_t graph_checksum = 0;
        for (int row = 0; row < output_size; row += 127) {
            graph_checksum += graph_outputs[row];
        }
        const double average_ms =
            (elapsed_ms[0] + elapsed_ms[1] + elapsed_ms[2] + elapsed_ms[3]) /
            elapsed_ms.size();
        std::printf(
            "qnn-gptq2-u16-gemv-mt-test: input=%d output=%d threads=%d "
            "iterations=4 times_ms=%.6f,%.6f,%.6f,%.6f average_ms=%.6f "
            "exact=%d checksum=%llu\n",
            input_size, output_size, multithread_test_threads,
            elapsed_ms[0], elapsed_ms[1], elapsed_ms[2], elapsed_ms[3],
            average_ms, graph_exact ? 1 : 0,
            static_cast<unsigned long long>(graph_checksum));

        ggml_backend_free(backend);
        ggml_threadpool_free(threadpool);
        ggml_backend_buffer_free(compute_buffer);
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(graph_ctx);
        return status == GGML_STATUS_SUCCESS && graph_exact ? 0 : 21;
    }

    run_scalar();
    run_4rows();
    run_8rows();
    run_gs32_8rows();
    run_gs32_16rows();
    run_16rows();
    run_f32();
    run_q2_k();
    const bool fast_exact =
        scalar_outputs == row4_outputs &&
        scalar_outputs == row8_outputs &&
        scalar_outputs == row16_outputs &&
        scalar_outputs == gs32_row16_outputs;
    for (int index = 0; index < input_size; ++index) {
        const int32_t centered = (index * 73 + 19) % 255 - 127;
        activations[index] =
            static_cast<uint16_t>(activation_zero_point + centered);
    }
    run_scalar();
    run_gs32_8rows();
    run_gs32_16rows();
    const bool i8_exact =
        activation_range == GGML_GPTQ2_U16_ACTIVATION_I8 &&
        scalar_outputs == row8_outputs &&
        scalar_outputs == gs32_row16_outputs;
    for (int index = 0; index < input_size; ++index) {
        const int32_t centered = (index * 1229 + 811) % 8191 - 4095;
        activations[index] =
            static_cast<uint16_t>(activation_zero_point + centered);
    }
    run_gs32_8rows();
    constexpr int warmup = 1;
    constexpr int iterations = 4;
    for (int iteration = 0; iteration < warmup; ++iteration) {
        run_scalar();
        run_4rows();
        run_8rows();
        run_gs32_8rows();
        run_gs32_16rows();
        run_16rows();
        run_f32();
        run_q2_k();
    }
    uint64_t checksum = 0;
    constexpr int prepare_iterations = 10000;
    int32_t prepare_max_abs = 0;
    const auto prepare_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < prepare_iterations; ++iteration) {
        activation_range = ggml_gptq2_32_prepare_u16_activation(
            input_size, activations.data(), activation_zero_point,
            activation_block_sums.data(), activation_low.data(),
            activation_high.data(), &prepare_max_abs);
        checksum += static_cast<uint64_t>(
            activation_block_sums[iteration % blocks] + 65536);
    }
    const auto prepare_end = std::chrono::steady_clock::now();
    const auto scalar_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        run_scalar();
        checksum += scalar_outputs[iteration % output_size];
    }
    const auto scalar_end = std::chrono::steady_clock::now();
    const auto row4_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        run_4rows();
        checksum += row4_outputs[iteration % output_size];
    }
    const auto row4_end = std::chrono::steady_clock::now();
    const auto row8_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        run_8rows();
        checksum += row8_outputs[iteration % output_size];
    }
    const auto row8_end = std::chrono::steady_clock::now();
    const auto gs32_row8_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        run_gs32_8rows();
        checksum += row8_outputs[iteration % output_size];
    }
    const auto gs32_row8_end = std::chrono::steady_clock::now();
    const auto gs32_row16_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        run_gs32_16rows();
        checksum += gs32_row16_outputs[iteration % output_size];
    }
    const auto gs32_row16_end = std::chrono::steady_clock::now();
    const auto row16_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        run_16rows();
        checksum += row16_outputs[iteration % output_size];
    }
    const auto row16_end = std::chrono::steady_clock::now();
    const auto f32_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        run_f32();
        checksum += static_cast<uint64_t>(
            std::llround(f32_outputs[iteration % output_size]));
    }
    const auto f32_end = std::chrono::steady_clock::now();
    const auto q2_k_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        run_q2_k();
        checksum += static_cast<uint64_t>(
            std::llround(q2_k_outputs[iteration % output_size]));
    }
    const auto q2_k_end = std::chrono::steady_clock::now();
    constexpr int q8_k_quant_iterations = 1000;
    const auto q8_k_quant_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < q8_k_quant_iterations; ++iteration) {
        quantize_row_q8_K(
            activations_f32.data(), q8_k_activations.data(), input_size);
        checksum += static_cast<uint64_t>(
            static_cast<uint8_t>(q8_k_activations[0].qs[
                iteration % QK_K]));
    }
    const auto q8_k_quant_end = std::chrono::steady_clock::now();
    const uint16_t old_activation0 = activations[0];
    activations[0] = UINT16_MAX;
    activation_block_sums[0] +=
        static_cast<int32_t>(activations[0]) - static_cast<int32_t>(old_activation0);
    activation_range = 0;
    run_scalar();
    run_4rows();
    run_8rows();
    run_gs32_8rows();
    run_gs32_16rows();
    run_16rows();
    std::array<double, iterations> gs32_wide_times_ms {};
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        run_gs32_8rows();
        gs32_wide_times_ms[iteration] =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        checksum += row8_outputs[iteration % output_size];
    }
    const bool wide_exact =
        scalar_outputs == row4_outputs &&
        scalar_outputs == row8_outputs &&
        scalar_outputs == row16_outputs &&
        scalar_outputs == gs32_row16_outputs;
    const bool exact = fast_exact && i8_exact && wide_exact;
    const double scalar_ms = std::chrono::duration<double, std::milli>(
        scalar_end - scalar_start).count() / iterations;
    const double row4_ms = std::chrono::duration<double, std::milli>(
        row4_end - row4_start).count() / iterations;
    const double row8_ms = std::chrono::duration<double, std::milli>(
        row8_end - row8_start).count() / iterations;
    const double gs32_row8_ms = std::chrono::duration<double, std::milli>(
        gs32_row8_end - gs32_row8_start).count() / iterations;
    const double gs32_row16_ms = std::chrono::duration<double, std::milli>(
        gs32_row16_end - gs32_row16_start).count() / iterations;
    const double gs32_wide_ms =
        (gs32_wide_times_ms[0] + gs32_wide_times_ms[1] +
         gs32_wide_times_ms[2] + gs32_wide_times_ms[3]) / iterations;
    const double row16_ms = std::chrono::duration<double, std::milli>(
        row16_end - row16_start).count() / iterations;
    const double f32_ms = std::chrono::duration<double, std::milli>(
        f32_end - f32_start).count() / iterations;
    const double q2_k_ms = std::chrono::duration<double, std::milli>(
        q2_k_end - q2_k_start).count() / iterations;
    const double q8_k_quant_us = std::chrono::duration<double, std::micro>(
        q8_k_quant_end - q8_k_quant_start).count() / q8_k_quant_iterations;
    const double prepare_us = std::chrono::duration<double, std::micro>(
        prepare_end - prepare_start).count() / prepare_iterations;
    std::printf(
        "qnn-gptq2-u16-gemv-4row-test: input=%d output=%d exact=%d "
        "fast_exact=%d i8_exact=%d wide_exact=%d gs32_dotprod=%d "
        "scalar_ms=%.6f row4_ms=%.6f row8_ms=%.6f row16_ms=%.6f "
        "gs32_row8_ms=%.6f gs32_row16_ms=%.6f gs32_16_vs_8_speedup=%.3f "
        "gs32_vs_row_major=%.3f "
        "gs32_wide_times_ms=%.6f,%.6f,%.6f,%.6f gs32_wide_ms=%.6f "
        "row8_vs_row4_speedup=%.3f row16_vs_row8_speedup=%.3f "
        "speedup=%.3f ggml_f32_ms=%.6f "
        "u16_vs_ggml_speedup=%.3f q2_k_q8_k_ms=%.6f "
        "q8_k_quant_us=%.6f prepare_us=%.6f q2_k_vs_u16_speedup=%.3f "
        "gs32_vs_q2_k_ratio=%.3f gs32_vs_q2_k_gap_pct=%.3f "
        "gptq2_bpw=%.3f q2_k_bpw=%.3f checksum=%llu "
        "activation_dequant_buffers=0 packed_int4_buffers=0\n",
        input_size, output_size, exact ? 1 : 0,
        fast_exact ? 1 : 0, i8_exact ? 1 : 0, wide_exact ? 1 : 0,
        ggml_gptq2_32_gs32_dotprod_enabled(),
        scalar_ms, row4_ms, row8_ms, row16_ms,
        gs32_row8_ms, gs32_row16_ms, gs32_row8_ms / gs32_row16_ms,
        row8_ms / gs32_row8_ms,
        gs32_wide_times_ms[0], gs32_wide_times_ms[1],
        gs32_wide_times_ms[2], gs32_wide_times_ms[3], gs32_wide_ms,
        row4_ms / row8_ms, row8_ms / row16_ms,
        scalar_ms / row16_ms,
        f32_ms, f32_ms / row16_ms,
        q2_k_ms, q8_k_quant_us, prepare_us, row16_ms / q2_k_ms,
        gs32_row8_ms / q2_k_ms,
        (gs32_row8_ms / q2_k_ms - 1.0) * 100.0,
        3.0,
        8.0 * sizeof(block_q2_K) / QK_K,
        static_cast<unsigned long long>(checksum));
    return exact ? 0 : 10;
}

int run_self_test(kernel_backend backend) {
    if (!initialize_ggml_cpu_backend()) {
        std::fprintf(stderr, "failed to initialize GGML CPU backend.\n");
        return 6;
    }

    const int gemv_4row_status = run_gptq2_u16_gemv_4row_test();
    if (gemv_4row_status != 0) {
        return gemv_4row_status;
    }

    const int group64_status = run_u16_source_group_kernel_test(64);
    if (group64_status != 0) {
        return group64_status;
    }
    const int group128_status = run_u16_source_group_kernel_test(128);
    if (group128_status != 0) {
        return group128_status;
    }

    constexpr size_t input_size = 96;
    constexpr size_t output_size = 5;
    constexpr size_t group_size = 32;
    constexpr size_t groups = input_size / group_size;

    qnn_a16w2_packed_plan plan {
        input_size,
        output_size,
        group_size,
        { 3.075e-5, 31241 },
        { 1.937e-4, 28719 },
        std::vector<uint8_t>(output_size * groups * (group_size / 4U)),
        std::vector<int8_t>(output_size * groups),
        std::vector<fixed_multiplier>(output_size * groups),
    };

    std::array<uint16_t, input_size> input {};
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<uint16_t>((i * 7919U + 913U) & 0xffffU);
    }
    for (size_t row = 0; row < output_size; ++row) {
        for (size_t group = 0; group < groups; ++group) {
            const size_t group_index = row * groups + group;
            plan.weight_zero_points[group_index] = static_cast<int8_t>((row + group) % 4);
            const double weight_scale = 0.00115 + 0.00017 * static_cast<double>(group_index + 1);
            plan.group_multipliers[group_index] =
                make_multiplier(plan.input.scale * weight_scale / plan.output.scale);
        }
        for (size_t col = 0; col < input_size; ++col) {
            const size_t group = col / group_size;
            const size_t group_index = row * groups + group;
            const size_t packed_index = group_index * (group_size / 4U) + (col % group_size) / 4U;
            const uint8_t code = static_cast<uint8_t>((row * 11 + col * 7) % 4);
            plan.packed_weight_codes[packed_index] |=
                static_cast<uint8_t>(code << ((col & 3U) * 2U));
        }
    }

    // Rebuild the native 12-byte GPTQ2_32 source stream from the same packed
    // INT2 codes. This is test input, not an unpacked runtime weight buffer.
    plan.input.scale = static_cast<double>(static_cast<float>(plan.input.scale));
    plan.output.scale = static_cast<double>(static_cast<float>(plan.output.scale));
    std::vector<uint8_t> gptq2_blocks(output_size * groups * 12U);
    for (size_t row = 0; row < output_size; ++row) {
        for (size_t group = 0; group < groups; ++group) {
            const size_t group_index = row * groups + group;
            const size_t source_offset = group_index * (group_size / 4U);
            uint8_t * const block = gptq2_blocks.data() + group_index * 12U;
            std::memcpy(block, plan.packed_weight_codes.data() + source_offset, group_size / 4U);
            const float source_scale =
                0.00115f + 0.00017f * static_cast<float>(group_index + 1);
            const ggml_fp16_t scale_fp16 = ggml_fp32_to_fp16(source_scale);
            const ggml_fp16_t zero_bias_fp16 = ggml_fp32_to_fp16(
                source_scale * static_cast<float>(plan.weight_zero_points[group_index]));
            std::memcpy(block + 8, &scale_fp16, sizeof(scale_fp16));
            std::memcpy(block + 10, &zero_bias_fp16, sizeof(zero_bias_fp16));
            const double qnn_weight_scale = std::max(
                static_cast<double>(ggml_fp16_to_fp32(scale_fp16)), 1.0e-4);
            plan.group_multipliers[group_index] =
                make_multiplier(plan.input.scale * qnn_weight_scale / plan.output.scale);
        }
    }

    const affine_u16 residual_qparams { 2.742e-4, 30115 };
    const affine_u16 output_qparams { 2.111e-4, 29777 };
    const affine_u16 rms_weight_qparams { static_cast<double>(5.5e-5f), 29101 };
    const affine_u16 rms_output_qparams { static_cast<double>(1.3e-4f), 28771 };
    std::array<uint16_t, output_size> residual {};
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = static_cast<uint16_t>((i * 12347U + 501U) & 0xffffU);
    }

    std::array<uint16_t, input_size> rms_weight {};
    for (size_t i = 0; i < rms_weight.size(); ++i) {
        rms_weight[i] = static_cast<uint16_t>((i * 4099U + 17011U) & 0xffffU);
    }

    std::array<uint16_t, output_size> scalar_gemv {};
    std::array<uint16_t, output_size> scalar_add {};
    qnn_a16w2_packed_gemv_scalar(plan, input.data(), scalar_gemv.data());

    // Exercise the production CPU primitive against the independent scalar
    // reference. The production path consumes the native 12-byte GPTQ2_32
    // stream directly; it never materializes an expanded INT4 weight buffer.
    std::array<uint16_t, output_size> cpu_kernel_gemv {};
    for (size_t row = 0; row < output_size; ++row) {
        ggml_vec_dot_gptq2_32_u16_qnn(
            static_cast<int>(input_size),
            &cpu_kernel_gemv[row],
            gptq2_blocks.data() + row * groups * 12U,
            input.data(),
            static_cast<float>(plan.input.scale),
            plan.input.zero_point,
            static_cast<float>(plan.output.scale),
            plan.output.zero_point);
    }
    const bool cpu_kernel_match = same_values(
        scalar_gemv.data(), cpu_kernel_gemv.data(), scalar_gemv.size());
    if (!cpu_kernel_match) {
        std::fprintf(stderr, "reference and production GPTQ2 U16 kernels differ.\n");
        for (size_t row = 0; row < output_size; ++row) {
            std::fprintf(stderr, "  row=%zu reference=%u cpu=%u\n",
                row,
                static_cast<unsigned>(scalar_gemv[row]),
                static_cast<unsigned>(cpu_kernel_gemv[row]));
        }
        for (size_t group = 0; group < groups; ++group) {
            const uint8_t * const block = gptq2_blocks.data() + group * 12U;
            ggml_fp16_t scale_fp16;
            ggml_fp16_t zero_bias_fp16;
            std::memcpy(&scale_fp16, block + 8, sizeof(scale_fp16));
            std::memcpy(&zero_bias_fp16, block + 10, sizeof(zero_bias_fp16));
            const float scale = ggml_fp16_to_fp32(scale_fp16);
            const float zero_bias = ggml_fp16_to_fp32(zero_bias_fp16);
            std::fprintf(stderr, "  group=%zu scale=%g zero_bias=%g reference_zp=%d source_zp=%ld multiplier=%lld\n",
                group,
                static_cast<double>(scale),
                static_cast<double>(zero_bias),
                static_cast<int>(plan.weight_zero_points[group]),
                std::lround(static_cast<double>(zero_bias) / static_cast<double>(scale)),
                static_cast<long long>(plan.group_multipliers[group].value));
        }
        return 5;
    }

    qnn_u16_add_scalar(
        scalar_gemv.data(),
        plan.output,
        residual.data(),
        residual_qparams,
        output_qparams,
        scalar_add.size(),
        scalar_add.data());

    std::array<uint16_t, output_size> cpu_add {};
    ggml_vec_add_affine_u16_qnn(
        static_cast<int>(cpu_add.size()),
        cpu_add.data(),
        scalar_gemv.data(),
        static_cast<float>(plan.output.scale),
        plan.output.zero_point,
        residual.data(),
        static_cast<float>(residual_qparams.scale),
        residual_qparams.zero_point,
        static_cast<float>(output_qparams.scale),
        output_qparams.zero_point);
    const bool cpu_add_match = same_values(
        scalar_add.data(), cpu_add.data(), scalar_add.size());
    if (!cpu_add_match) {
        std::fprintf(stderr, "reference and production U16 residual kernels differ.\n");
        return 6;
    }

    std::array<uint16_t, input_size> scalar_rmsnorm {};
    std::array<uint16_t, input_size> cpu_rmsnorm {};
    qnn_u16_rms_norm_affine_scalar(
        input.data(),
        plan.input,
        rms_weight.data(),
        rms_weight_qparams,
        1.0e-6,
        rms_output_qparams,
        scalar_rmsnorm.size(),
        scalar_rmsnorm.data());
    ggml_vec_rms_norm_affine_u16_qnn(
        static_cast<int>(cpu_rmsnorm.size()),
        cpu_rmsnorm.data(),
        input.data(),
        static_cast<float>(plan.input.scale),
        plan.input.zero_point,
        rms_weight.data(),
        static_cast<float>(rms_weight_qparams.scale),
        rms_weight_qparams.zero_point,
        1.0e-6f,
        static_cast<float>(rms_output_qparams.scale),
        rms_output_qparams.zero_point);
    const bool cpu_rmsnorm_match = same_values(
        scalar_rmsnorm.data(), cpu_rmsnorm.data(), scalar_rmsnorm.size());
    if (!cpu_rmsnorm_match) {
        std::fprintf(stderr, "reference and production U16 RMSNorm kernels differ.\n");
        return 7;
    }

    std::array<uint16_t, output_size> result_gemv = scalar_gemv;
    std::array<uint16_t, output_size> result_add = scalar_add;
    bool scalar_neon_match = true;

    if (backend != kernel_backend::scalar) {
        std::array<uint16_t, output_size> neon_gemv {};
        std::array<uint16_t, output_size> neon_add {};
        if (!qnn_a16w2_packed_gemv_neon(plan, input.data(), neon_gemv.data()) ||
            !qnn_u16_add_neon(
                neon_gemv.data(),
                plan.output,
                residual.data(),
                residual_qparams,
                output_qparams,
                neon_add.size(),
                neon_add.data())) {
            std::fprintf(stderr, "NEON backend is unavailable in this build.\n");
            return 3;
        }
        scalar_neon_match = same_values(
            scalar_gemv.data(), neon_gemv.data(), scalar_gemv.size()) &&
            same_values(scalar_add.data(), neon_add.data(), scalar_add.size());
        if (!scalar_neon_match) {
            std::fprintf(stderr, "scalar and NEON U16 outputs differ.\n");
            return 4;
        }
        result_gemv = neon_gemv;
        result_add = neon_add;
    }

    uint64_t checksum = fnv1a_u16(result_gemv.data(), result_gemv.size());
    checksum = fnv1a_u16(result_add.data(), result_add.size(), checksum);
    checksum = fnv1a_u16(cpu_rmsnorm.data(), cpu_rmsnorm.size(), checksum);
    std::printf("u16-core-self-test: backend=%s gemv=",
        backend == kernel_backend::scalar ? "scalar" :
        backend == kernel_backend::neon ? "neon" : "compare");
    for (uint16_t value : result_gemv) {
        std::printf(" %u", static_cast<unsigned>(value));
    }
    std::printf(" add=");
    for (uint16_t value : result_add) {
        std::printf(" %u", static_cast<unsigned>(value));
    }
    std::printf(" rmsnorm=");
    for (size_t i = 0; i < output_size; ++i) {
        std::printf(" %u", static_cast<unsigned>(cpu_rmsnorm[i]));
    }
    std::printf(
        " checksum=0x%016llx scalar_neon_match=%d cpu_kernel_match=%d cpu_add_match=%d cpu_rmsnorm_match=%d activation_dequant_buffers=0\n",
        static_cast<unsigned long long>(checksum),
        scalar_neon_match ? 1 : 0,
        cpu_kernel_match ? 1 : 0,
        cpu_add_match ? 1 : 0,
        cpu_rmsnorm_match ? 1 : 0);
    return 0;
}

struct u16_graph_quantize_params { affine_u16 output; };
struct u16_graph_gemv_params { affine_u16 input; affine_u16 output; };
struct u16_graph_add_params { affine_u16 lhs; affine_u16 rhs; affine_u16 output; };
struct u16_graph_rmsnorm_params { affine_u16 input; affine_u16 weight; affine_u16 output; float epsilon; };

void u16_graph_quantize(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) return;
    const auto * params = static_cast<const u16_graph_quantize_params *>(userdata);
    const ggml_tensor * input = dst->src[0];
    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_U16);
    GGML_ASSERT(ggml_nelements(input) == ggml_nelements(dst));
    const float * source = static_cast<const float *>(input->data);
    uint16_t * output = static_cast<uint16_t *>(dst->data);
    for (int64_t index = 0; index < ggml_nelements(dst); ++index) {
        output[index] = saturate_u16(static_cast<int64_t>(std::llround(
            static_cast<double>(source[index]) / params->output.scale)) + params->output.zero_point);
    }
}

void u16_graph_gemv(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) return;
    const auto * params = static_cast<const u16_graph_gemv_params *>(userdata);
    const ggml_tensor * input = dst->src[0];
    const ggml_tensor * weights = dst->src[1];
    GGML_ASSERT(input->type == GGML_TYPE_U16 && weights->type == GGML_TYPE_GPTQ2_32);
    GGML_ASSERT(dst->type == GGML_TYPE_U16 && input->ne[0] == weights->ne[0]);
    GGML_ASSERT(dst->ne[0] == weights->ne[1]);
    const uint16_t * codes = static_cast<const uint16_t *>(input->data);
    const uint8_t * packed = static_cast<const uint8_t *>(weights->data);
    uint16_t * output = static_cast<uint16_t *>(dst->data);
    for (int64_t row = 0; row < weights->ne[1]; ++row) {
        ggml_vec_dot_gptq2_32_u16_qnn(
            static_cast<int>(input->ne[0]), output + row,
            packed + row * weights->nb[1], codes,
            static_cast<float>(params->input.scale), params->input.zero_point,
            static_cast<float>(params->output.scale), params->output.zero_point);
    }
}

void u16_graph_add(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) return;
    const auto * params = static_cast<const u16_graph_add_params *>(userdata);
    const ggml_tensor * lhs = dst->src[0];
    const ggml_tensor * rhs = dst->src[1];
    GGML_ASSERT(lhs->type == GGML_TYPE_U16 && rhs->type == GGML_TYPE_U16);
    GGML_ASSERT(dst->type == GGML_TYPE_U16 && ggml_are_same_shape(lhs, rhs));
    ggml_vec_add_affine_u16_qnn(
        static_cast<int>(ggml_nelements(dst)), static_cast<uint16_t *>(dst->data),
        static_cast<const uint16_t *>(lhs->data), static_cast<float>(params->lhs.scale), params->lhs.zero_point,
        static_cast<const uint16_t *>(rhs->data), static_cast<float>(params->rhs.scale), params->rhs.zero_point,
        static_cast<float>(params->output.scale), params->output.zero_point);
}

void u16_graph_rmsnorm(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) return;
    const auto * params = static_cast<const u16_graph_rmsnorm_params *>(userdata);
    const ggml_tensor * input = dst->src[0];
    const ggml_tensor * weight = dst->src[1];
    GGML_ASSERT(input->type == GGML_TYPE_U16 && weight->type == GGML_TYPE_U16);
    GGML_ASSERT(dst->type == GGML_TYPE_U16 && ggml_are_same_shape(input, weight));
    ggml_vec_rms_norm_affine_u16_qnn(
        static_cast<int>(ggml_nelements(dst)), static_cast<uint16_t *>(dst->data),
        static_cast<const uint16_t *>(input->data), static_cast<float>(params->input.scale), params->input.zero_point,
        static_cast<const uint16_t *>(weight->data), static_cast<float>(params->weight.scale), params->weight.zero_point,
        params->epsilon, static_cast<float>(params->output.scale), params->output.zero_point);
}

int run_u16_graph_test() {
    if (!initialize_ggml_cpu_backend()) {
        std::fprintf(stderr, "failed to initialize GGML CPU backend.\n");
        return 8;
    }
    constexpr size_t input_size = 32;
    constexpr size_t output_size = 2;
    const affine_u16 input_qparams { 3.075e-5, 31241 };
    const affine_u16 gemv_qparams { 1.937e-4, 28719 };
    const affine_u16 residual_qparams { 2.742e-4, 30115 };
    const affine_u16 add_qparams { 2.111e-4, 29777 };
    const affine_u16 rms_weight_qparams { static_cast<double>(5.5e-5f), 29101 };
    const affine_u16 rms_output_qparams { static_cast<double>(1.3e-4f), 28771 };
    std::array<uint16_t, input_size> input_codes {};
    std::array<float, input_size> input_f32 {};
    std::array<uint8_t, output_size * 12U> packed_weights {};
    std::array<uint16_t, output_size> residual {};
    std::array<uint16_t, output_size> rms_weight {};
    for (size_t index = 0; index < input_size; ++index) {
        input_codes[index] = static_cast<uint16_t>((index * 7919U + 913U) & 0xffffU);
        input_f32[index] = static_cast<float>((static_cast<int32_t>(input_codes[index]) - input_qparams.zero_point) * input_qparams.scale);
    }
    for (size_t row = 0; row < output_size; ++row) {
        uint8_t * block = packed_weights.data() + row * 12U;
        for (size_t column = 0; column < input_size; ++column) {
            const uint8_t code = static_cast<uint8_t>((row * 11U + column * 7U) % 4U);
            block[column / 4U] |= static_cast<uint8_t>(code << ((column & 3U) * 2U));
        }
        const float scale = 0.00115f + 0.00017f * static_cast<float>(row + 1U);
        const ggml_fp16_t scale_fp16 = ggml_fp32_to_fp16(scale);
        const ggml_fp16_t zero_fp16 = ggml_fp32_to_fp16(scale * static_cast<float>(row));
        std::memcpy(block + 8U, &scale_fp16, sizeof(scale_fp16));
        std::memcpy(block + 10U, &zero_fp16, sizeof(zero_fp16));
        residual[row] = static_cast<uint16_t>((row * 12347U + 501U) & 0xffffU);
        rms_weight[row] = static_cast<uint16_t>((row * 4099U + 17011U) & 0xffffU);
    }
    std::array<uint16_t, output_size> expected_gemv {};
    std::array<uint16_t, output_size> expected_add {};
    std::array<uint16_t, output_size> expected_rmsnorm {};
    for (size_t row = 0; row < output_size; ++row) {
        ggml_vec_dot_gptq2_32_u16_qnn_fixed(
            static_cast<int>(input_size), expected_gemv.data() + row,
            packed_weights.data() + row * 12U, input_codes.data(),
            static_cast<int64_t>(std::llround(std::ldexp(
                input_qparams.scale / gemv_qparams.scale, 20))),
            input_qparams.zero_point, gemv_qparams.zero_point);
    }
    qnn_u16_add_scalar(expected_gemv.data(), gemv_qparams, residual.data(), residual_qparams,
        add_qparams, expected_add.size(), expected_add.data());
    qnn_u16_rms_norm_affine_scalar(expected_add.data(), add_qparams, rms_weight.data(), rms_weight_qparams,
        1.0e-6, rms_output_qparams, expected_rmsnorm.size(), expected_rmsnorm.data());
    ggml_init_params init_params { 2U * 1024U * 1024U, nullptr, false };
    ggml_context * ctx = ggml_init(init_params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create GGML context for U16 graph test.\n");
        return 9;
    }
    u16_graph_quantize_params quantize_params { input_qparams };
    llama_qnn_linear_qparams gemv_params;
    gemv_params.input.scale = static_cast<float>(input_qparams.scale);
    gemv_params.input.zero_point = input_qparams.zero_point;
    gemv_params.output.scale = static_cast<float>(gemv_qparams.scale);
    gemv_params.output.zero_point = gemv_qparams.zero_point;
    gemv_params.activation_to_output_q20 = static_cast<int64_t>(std::llround(
        std::ldexp(input_qparams.scale / gemv_qparams.scale, 20)));
    u16_graph_add_params add_params { gemv_qparams, residual_qparams, add_qparams };
    u16_graph_rmsnorm_params rmsnorm_params { add_qparams, rms_weight_qparams, rms_output_qparams, 1.0e-6f };
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input_size);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_GPTQ2_32, input_size, output_size);
    ggml_tensor * residual_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_U16, output_size);
    ggml_tensor * rms_weight_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_U16, output_size);
    ggml_tensor * quantize_args[] = { input };
    ggml_tensor * quantized = ggml_custom_4d(ctx, GGML_TYPE_U16, input_size, 1, 1, 1,
        quantize_args, 1, u16_graph_quantize, 1, &quantize_params);
    ggml_tensor * gemv = llama_qnn_u16_mul_mat(ctx, weights, quantized, &gemv_params);
    ggml_tensor * add_args[] = { gemv, residual_tensor };
    ggml_tensor * add = ggml_custom_4d(ctx, GGML_TYPE_U16, output_size, 1, 1, 1,
        add_args, 2, u16_graph_add, 1, &add_params);
    ggml_tensor * rmsnorm_args[] = { add, rms_weight_tensor };
    ggml_tensor * rmsnorm = ggml_custom_4d(ctx, GGML_TYPE_U16, output_size, 1, 1, 1,
        rmsnorm_args, 2, u16_graph_rmsnorm, 1, &rmsnorm_params);
    std::memcpy(input->data, input_f32.data(), sizeof(input_f32));
    std::memcpy(weights->data, packed_weights.data(), sizeof(packed_weights));
    std::memcpy(residual_tensor->data, residual.data(), sizeof(residual));
    std::memcpy(rms_weight_tensor->data, rms_weight.data(), sizeof(rms_weight));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, rmsnorm);
    int f32_intermediates = 0;
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        ggml_tensor * const node = ggml_graph_node(graph, index);
        if (node != input && node->type == GGML_TYPE_F32) ++f32_intermediates;
    }
    const enum ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, 1);
    const bool result_matches = status == GGML_STATUS_SUCCESS &&
        same_values(static_cast<const uint16_t *>(quantized->data), input_codes.data(), input_size) &&
        same_values(static_cast<const uint16_t *>(gemv->data), expected_gemv.data(), output_size) &&
        same_values(static_cast<const uint16_t *>(add->data), expected_add.data(), output_size) &&
        same_values(static_cast<const uint16_t *>(rmsnorm->data), expected_rmsnorm.data(), output_size);
    const uint64_t checksum = fnv1a_u16(static_cast<const uint16_t *>(rmsnorm->data), output_size);
    std::printf("u16-core-graph-test: status=%s input_type=%s intermediate_type=%s output_type=%s f32_activation_boundaries=1 f32_activation_intermediates=%d activation_dequant_buffers=0 packed_int4_buffers=0 result_match=%d checksum=0x%016llx\n",
        ggml_status_to_string(status), ggml_type_name(input->type), ggml_type_name(gemv->type),
        ggml_type_name(rmsnorm->type), f32_intermediates, result_matches ? 1 : 0,
        static_cast<unsigned long long>(checksum));
    ggml_free(ctx);
    return f32_intermediates == 0 && result_matches ? 0 : 10;
}

const llama_qnn_affine_qparams & operation_affine(
        const llama_qnn_quant_profile & profile,
        const llama_qnn_operation & operation,
        const char * role,
        int32_t position) {
    const llama_qnn_u16_tensor * tensor =
        profile.find_u16_operand(operation, role, position);
    if (tensor == nullptr ||
        tensor->qparams.encoding != LLAMA_QNN_QUANTIZATION_SCALE_OFFSET ||
        tensor->qparams.scale_offsets.size() != 1) {
        throw std::runtime_error(
            "operation " + operation.name + " lacks scalar affine U16 qparams");
    }
    return tensor->qparams.scale_offsets.front();
}

int64_t affine_ratio_q15(
        const llama_qnn_affine_qparams & input,
        const llama_qnn_affine_qparams & output) {
    return static_cast<int64_t>(std::llround(std::ldexp(
        static_cast<double>(input.scale) / static_cast<double>(output.scale),
        15)));
}

struct htp_scalefactor_q15 {
    int32_t multiplier;
    int32_t right_shift;
};

htp_scalefactor_q15 make_htp_scalefactor_q15(
        const llama_qnn_affine_qparams & input,
        const llama_qnn_affine_qparams & output) {
    const float ratio = input.scale / output.scale;
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(ratio), "float32 qparams required");
    std::memcpy(&bits, &ratio, sizeof(bits));
    constexpr int significand_bits = 15;
    constexpr int reduction_shift = 24 - significand_bits;
    const uint32_t reduced =
        (bits + (UINT32_C(1) << (reduction_shift - 1))) >> reduction_shift;
    const int32_t multiplier = static_cast<int32_t>(
        (reduced & ((UINT32_C(1) << significand_bits) - 1)) |
        (UINT32_C(1) << (significand_bits - 1)));
    const int32_t exponent = static_cast<int32_t>(
        (reduced >> (significand_bits - 1)) & 0xffU) - 126;
    return { multiplier, significand_bits - exponent };
}

size_t operation_width(
        const llama_qnn_quant_profile & profile,
        const llama_qnn_operation & operation) {
    const llama_qnn_u16_tensor * output =
        profile.find_u16_operand(operation, "output", 0);
    if (output == nullptr || output->dimensions.empty() ||
        output->dimensions.back() <= 0) {
        throw std::runtime_error("operation " + operation.name + " lacks an output width");
    }
    return static_cast<size_t>(output->dimensions.back());
}

void fill_operation_input(
        uint16_t * output,
        size_t count,
        int32_t zero_point,
        uint32_t seed) {
    for (size_t index = 0; index < count; ++index) {
        const int32_t centered = static_cast<int32_t>((index * 37U + seed) % 511U) - 255;
        output[index] = saturate_u16(static_cast<int64_t>(zero_point) + centered);
    }
}

int run_profile_u16_operation_test(const char * profile_path) {
    if (!initialize_ggml_cpu_backend()) {
        std::fprintf(stderr, "failed to initialize GGML CPU backend.\n");
        return 16;
    }
    const auto profile = llama_qnn_quant_profile_load_file(profile_path);
    const llama_qnn_operation * rms =
        profile->find_operation_by_fx(0, "aten_rms_norm_default");
    const llama_qnn_operation * add =
        profile->find_operation_by_fx(0, "aten_add_tensor_3");
    const llama_qnn_operation * sub =
        profile->find_operation_by_fx(0, "aten_sub_tensor_1_h_7");
    const llama_qnn_operation * mul =
        profile->find_operation_by_fx(0, "aten_mul_tensor_9");
    const llama_qnn_operation * sigmoid =
        profile->find_operation_by_fx(0, "aten_sigmoid_default");
    const llama_qnn_operation * rope_mul[4] = {
        profile->find_operation_by_fx(0, "aten_mul_tensor_1_h_0"),
        profile->find_operation_by_fx(0, "aten_mul_tensor_2_h_0"),
        profile->find_operation_by_fx(0, "aten_mul_tensor_3_h_0"),
        profile->find_operation_by_fx(0, "aten_mul_tensor_4_h_0"),
    };
    const llama_qnn_operation * rope_slice[2] = {
        profile->find_operation_by_fx(0, "aten_slice_copy_tensor_h_0"),
        profile->find_operation_by_fx(0, "aten_slice_copy_tensor_1_h_0"),
    };
    const llama_qnn_operation * rope_sub =
        profile->find_operation_by_fx(0, "aten_sub_tensor_h_0");
    const llama_qnn_operation * rope_add =
        profile->find_operation_by_fx(0, "aten_add_tensor_h_0");
    const llama_qnn_operation * qk_rotation =
        profile->find_operation_by_fx(0, "aten_matmul_default_h_0");
    const llama_qnn_operation * k_convert =
        profile->find_operation(0, "aten__to_copy_default_15");
    const llama_qnn_operation * v_convert =
        profile->find_operation(0, "aten__to_copy_default_7");
    const llama_qnn_operation * qk_matmul =
        profile->find_operation(0, "aten_matmul_default_2_h_0");
    const llama_qnn_operation * attention_divide =
        profile->find_operation(0, "aten_div_tensor_h_0");
    const llama_qnn_operation * attention_min =
        profile->find_operation(0, "aten_amin_default_h_0");
    const llama_qnn_operation * attention_floor_add =
        profile->find_operation(0, "aten_add_tensor_2_h_0");
    const llama_qnn_operation * attention_select =
        profile->find_operation(0, "aten_where_self_h_0");
    const llama_qnn_operation * attention_softmax =
        profile->find_operation(0, "aten__softmax_default_h_0");
    const llama_qnn_operation * attention_value_matmul =
        profile->find_operation(0, "aten_matmul_default_3_h_0");
    if (rms == nullptr || add == nullptr || sub == nullptr || mul == nullptr ||
        sigmoid == nullptr || rope_mul[0] == nullptr || rope_mul[1] == nullptr ||
        rope_mul[2] == nullptr || rope_mul[3] == nullptr ||
        rope_slice[0] == nullptr || rope_slice[1] == nullptr ||
        rope_sub == nullptr || rope_add == nullptr || qk_rotation == nullptr ||
        k_convert == nullptr || v_convert == nullptr || qk_matmul == nullptr ||
        attention_divide == nullptr || attention_min == nullptr ||
        attention_floor_add == nullptr || attention_select == nullptr ||
        attention_softmax == nullptr || attention_value_matmul == nullptr) {
        throw std::runtime_error("profile lacks the layer-0 U16 operation representatives");
    }

    const size_t rms_width = operation_width(*profile, *rms);
    const size_t add_width = operation_width(*profile, *add);
    const size_t sub_width = operation_width(*profile, *sub);
    const size_t mul_width = operation_width(*profile, *mul);
    const size_t sigmoid_width = operation_width(*profile, *sigmoid);
    ggml_init_params init_params { 32U * 1024U * 1024U, nullptr, false };
    ggml_context * ctx = ggml_init(init_params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to create GGML context for profile U16 operation test");
    }

    ggml_tensor * rms_input = ggml_new_tensor_1d(ctx, GGML_TYPE_U16, rms_width);
    ggml_tensor * add_lhs = ggml_new_tensor_1d(ctx, GGML_TYPE_U16, add_width);
    ggml_tensor * add_rhs = ggml_new_tensor_1d(ctx, GGML_TYPE_U16, add_width);
    ggml_tensor * sub_lhs = ggml_new_tensor_1d(ctx, GGML_TYPE_U16, sub_width);
    ggml_tensor * sub_rhs = ggml_new_tensor_1d(ctx, GGML_TYPE_U16, sub_width);
    ggml_tensor * mul_lhs = ggml_new_tensor_1d(ctx, GGML_TYPE_U16, mul_width);
    ggml_tensor * mul_rhs = ggml_new_tensor_1d(ctx, GGML_TYPE_U16, mul_width);
    ggml_tensor * sigmoid_input = ggml_new_tensor_1d(ctx, GGML_TYPE_U16, sigmoid_width);
    constexpr int64_t rope_tokens = 4;
    constexpr int64_t rope_dimension = 128;
    constexpr int64_t rope_half_dimension = rope_dimension / 2;
    ggml_tensor * rope_input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_U16, rope_dimension, rope_tokens);
    ggml_tensor * rope_positions = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, rope_tokens);
    ggml_tensor * rms_output = llama_qnn_u16_rms_norm(
        ctx, rms_input, profile.get(), rms);
    ggml_tensor * add_output = llama_qnn_u16_add(
        ctx, add_lhs, add_rhs, profile.get(), add);
    ggml_tensor * sub_output = llama_qnn_u16_sub(
        ctx, sub_lhs, sub_rhs, profile.get(), sub);
    ggml_tensor * mul_output = llama_qnn_u16_mul(
        ctx, mul_lhs, mul_rhs, profile.get(), mul);
    ggml_tensor * sigmoid_output = llama_qnn_u16_sigmoid(ctx, sigmoid_input, sigmoid);
    ggml_tensor * rope_output = llama_qnn_u16_rope(
        ctx, rope_input, rope_positions, profile.get(), 0, 0, false);
    ggml_tensor * rotated_output = llama_qnn_u16_qk_rotate(
        ctx, rope_output, profile.get(), 0, 0, false);
    ggml_tensor * k_convert_input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_U16, rope_dimension, rope_tokens);
    ggml_tensor * v_convert_input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_U16, rope_dimension, rope_tokens);
    ggml_tensor * k_cache_codes = llama_qnn_u16_to_u8(
        ctx, k_convert_input, profile.get(), k_convert);
    ggml_tensor * v_cache_codes = llama_qnn_u16_to_u8(
        ctx, v_convert_input, profile.get(), v_convert);
    constexpr int64_t attention_columns = 19;
    ggml_tensor * qk_query = ggml_new_tensor_2d(
        ctx, GGML_TYPE_U16, rope_dimension, rope_tokens);
    ggml_tensor * qk_key_matrix = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I8, attention_columns, rope_dimension);
    ggml_tensor * qk_logits = llama_qnn_u16_u8_matmul(
        ctx, qk_query, qk_key_matrix, profile.get(), qk_matmul);
    ggml_tensor * scaled_logits = llama_qnn_u16_divide_static(
        ctx, qk_logits, profile.get(), attention_divide);
    ggml_tensor * minimum_logits = llama_qnn_u16_reduce_min(
        ctx, scaled_logits, profile.get(), attention_min);
    ggml_tensor * mask_floor_constant = ggml_new_tensor_2d(
        ctx, GGML_TYPE_U16, 1, rope_tokens);
    ggml_tensor * mask_floor = llama_qnn_u16_add(
        ctx, minimum_logits, mask_floor_constant,
        profile.get(), attention_floor_add);
    ggml_tensor * attention_condition = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I8, attention_columns, rope_tokens);
    ggml_tensor * masked_logits = llama_qnn_u16_select(
        ctx, attention_condition, scaled_logits, mask_floor,
        profile.get(), attention_select);
    ggml_tensor * attention_probabilities = llama_qnn_u16_softmax(
        ctx, masked_logits, profile.get(), attention_softmax);
    ggml_tensor * attention_value_matrix = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I8, rope_dimension, attention_columns);
    ggml_tensor * attention_output = llama_qnn_u16_u8_matmul(
        ctx, attention_probabilities, attention_value_matrix,
        profile.get(), attention_value_matmul);

    const auto & rms_input_q = operation_affine(*profile, *rms, "input", 0);
    const auto & rms_weight_q = operation_affine(*profile, *rms, "input", 1);
    const auto & rms_output_q = operation_affine(*profile, *rms, "output", 0);
    const auto & add_lhs_q = operation_affine(*profile, *add, "input", 0);
    const auto & add_rhs_q = operation_affine(*profile, *add, "input", 1);
    const auto & add_output_q = operation_affine(*profile, *add, "output", 0);
    const auto & sub_lhs_q = operation_affine(*profile, *sub, "input", 0);
    const auto & sub_rhs_q = operation_affine(*profile, *sub, "input", 1);
    const auto & sub_output_q = operation_affine(*profile, *sub, "output", 0);
    const auto & mul_lhs_q = operation_affine(*profile, *mul, "input", 0);
    const auto & mul_rhs_q = operation_affine(*profile, *mul, "input", 1);
    const auto & mul_output_q = operation_affine(*profile, *mul, "output", 0);
    const auto & sigmoid_input_q = operation_affine(*profile, *sigmoid, "input", 0);
    const auto & k_convert_input_q = operation_affine(
        *profile, *k_convert, "input", 0);
    const auto & v_convert_input_q = operation_affine(
        *profile, *v_convert, "input", 0);
    const auto & qk_query_q = operation_affine(
        *profile, *qk_matmul, "input", 0);
    fill_operation_input(
        static_cast<uint16_t *>(rms_input->data), rms_width, rms_input_q.zero_point, 11);
    fill_operation_input(
        static_cast<uint16_t *>(add_lhs->data), add_width, add_lhs_q.zero_point, 23);
    fill_operation_input(
        static_cast<uint16_t *>(add_rhs->data), add_width, add_rhs_q.zero_point, 37);
    fill_operation_input(
        static_cast<uint16_t *>(sub_lhs->data), sub_width, sub_lhs_q.zero_point, 41);
    fill_operation_input(
        static_cast<uint16_t *>(sub_rhs->data), sub_width, sub_rhs_q.zero_point, 53);
    fill_operation_input(
        static_cast<uint16_t *>(mul_lhs->data), mul_width, mul_lhs_q.zero_point, 67);
    fill_operation_input(
        static_cast<uint16_t *>(mul_rhs->data), mul_width, mul_rhs_q.zero_point, 79);
    fill_operation_input(
        static_cast<uint16_t *>(sigmoid_input->data), sigmoid_width,
        sigmoid_input_q.zero_point, 83);
    fill_operation_input(
        static_cast<uint16_t *>(k_convert_input->data),
        rope_dimension * rope_tokens, k_convert_input_q.zero_point, 127);
    fill_operation_input(
        static_cast<uint16_t *>(v_convert_input->data),
        rope_dimension * rope_tokens, v_convert_input_q.zero_point, 139);
    fill_operation_input(
        static_cast<uint16_t *>(qk_query->data),
        rope_dimension * rope_tokens, qk_query_q.zero_point, 151);
    const auto & rope_source_q = operation_affine(
        *profile, *rope_slice[0], "input", 0);
    auto * rope_input_data = static_cast<uint16_t *>(rope_input->data);
    auto * rope_position_data = static_cast<int32_t *>(rope_positions->data);
    for (int64_t token = 0; token < rope_tokens; ++token) {
        fill_operation_input(
            rope_input_data + token * rope_dimension,
            rope_half_dimension, rope_source_q.zero_point,
            static_cast<uint32_t>(97 + token));
        fill_operation_input(
            rope_input_data + token * rope_dimension + rope_half_dimension,
            rope_half_dimension, rope_source_q.zero_point,
            static_cast<uint32_t>(109 + token));
        rope_position_data[token] = static_cast<int32_t>(token * 17);
    }

    std::vector<uint16_t> expected_rms(rms_width);
    std::vector<uint16_t> expected_add(add_width);
    std::vector<uint16_t> expected_sub(sub_width);
    std::vector<uint16_t> expected_mul(mul_width);
    std::vector<uint16_t> expected_sigmoid(sigmoid_width);
    std::vector<uint16_t> expected_rope(rope_dimension * rope_tokens);
    std::vector<uint16_t> expected_rotated(rope_dimension * rope_tokens);
    std::vector<uint8_t> expected_k_cache(rope_dimension * rope_tokens);
    std::vector<uint8_t> expected_v_cache(rope_dimension * rope_tokens);
    std::vector<uint16_t> expected_qk_logits(attention_columns * rope_tokens);
    std::vector<uint16_t> expected_scaled_logits(attention_columns * rope_tokens);
    std::vector<uint16_t> expected_minimum_logits(rope_tokens);
    std::vector<uint16_t> expected_mask_floor(rope_tokens);
    std::vector<uint16_t> expected_masked_logits(attention_columns * rope_tokens);
    std::vector<uint16_t> expected_attention_probabilities(
        attention_columns * rope_tokens);
    std::vector<uint16_t> expected_attention_output(rope_dimension * rope_tokens);
    const llama_qnn_u16_tensor * rms_weight =
        profile->find_u16_operand(*rms, "input", 1);
    if (rms_weight == nullptr || rms_weight->static_data.size() != rms_width) {
        ggml_free(ctx);
        throw std::runtime_error("profile RMSNorm weight payload is incomplete");
    }
    ggml_vec_rms_norm_affine_u16_qnn_fixed(
        static_cast<int>(rms_width), expected_rms.data(),
        static_cast<const uint16_t *>(rms_input->data), rms_input_q.zero_point,
        rms_weight->static_data.data(), rms_weight_q.zero_point,
        rms->rms_epsilon_in_codes_q16,
        static_cast<int64_t>(std::llround(std::ldexp(
            static_cast<double>(rms_weight_q.scale) /
                static_cast<double>(rms_output_q.scale),
            31))),
        rms_output_q.zero_point);
    ggml_vec_add_affine_u16_qnn_q15(
        static_cast<int>(add_width), expected_add.data(),
        static_cast<const uint16_t *>(add_lhs->data),
        affine_ratio_q15(add_lhs_q, add_output_q), add_lhs_q.zero_point,
        static_cast<const uint16_t *>(add_rhs->data),
        affine_ratio_q15(add_rhs_q, add_output_q), add_rhs_q.zero_point,
        add_output_q.zero_point);
    ggml_vec_add_affine_u16_qnn_q15(
        static_cast<int>(sub_width), expected_sub.data(),
        static_cast<const uint16_t *>(sub_lhs->data),
        affine_ratio_q15(sub_lhs_q, sub_output_q), sub_lhs_q.zero_point,
        static_cast<const uint16_t *>(sub_rhs->data),
        -affine_ratio_q15(sub_rhs_q, sub_output_q), sub_rhs_q.zero_point,
        sub_output_q.zero_point);
    ggml_vec_mul_affine_u16_qnn_q31(
        static_cast<int>(mul_width), expected_mul.data(),
        static_cast<const uint16_t *>(mul_lhs->data), mul_lhs_q.zero_point,
        static_cast<const uint16_t *>(mul_rhs->data), mul_rhs_q.zero_point,
        mul->product_to_output_q31, mul->product_requant_nudge_q31,
        mul_output_q.zero_point);
    for (size_t index = 0; index < sigmoid_width; ++index) {
        expected_sigmoid[index] = sigmoid->unary_lut[
            static_cast<const uint16_t *>(sigmoid_input->data)[index]];
    }
    const llama_qnn_u16_tensor * rope_cos =
        profile->find_u16_tensor("b__frozen_param311@0");
    const llama_qnn_u16_tensor * rope_sin =
        profile->find_u16_tensor("b__frozen_param312@0");
    if (rope_cos == nullptr || rope_sin == nullptr ||
        rope_cos->static_data.size() != rope_sin->static_data.size() ||
        rope_cos->dimensions.size() != 2 ||
        rope_cos->dimensions[1] != rope_half_dimension) {
        ggml_free(ctx);
        throw std::runtime_error("profile RoPE source tables are incomplete");
    }
    ggml_u16_rope_qnn_fixed_params rope_params {};
    rope_params.split_input_zero_point = rope_source_q.zero_point;
    for (int operation = 0; operation < 2; ++operation) {
        const auto & split_output_q = operation_affine(
            *profile, *rope_slice[operation], "output", 0);
        rope_params.split_to_output_q31[operation] =
            static_cast<int64_t>(std::llround(std::ldexp(
                rope_source_q.scale / split_output_q.scale, 31)));
        rope_params.split_output_zero_points[operation] =
            split_output_q.zero_point;
    }
    const llama_qnn_u16_tensor * rope_tables[4] = {
        rope_cos, rope_sin, rope_sin, rope_cos,
    };
    for (int operation = 0; operation < 4; ++operation) {
        const auto & lhs_q = operation_affine(
            *profile, *rope_mul[operation], "input", 0);
        const auto & output_q = operation_affine(
            *profile, *rope_mul[operation], "output", 0);
        const auto & table_q = rope_tables[operation]->qparams.scale_offsets.front();
        rope_params.lhs_zero_points[operation] = lhs_q.zero_point;
        rope_params.table_source_zero_points[operation] = table_q.zero_point;
        rope_params.table_source_to_storage_q31[operation] = INT64_C(1) << 31;
        rope_params.table_zero_points[operation] = table_q.zero_point;
        const float product_ratio =
            (static_cast<float>(lhs_q.scale) * static_cast<float>(table_q.scale)) /
            static_cast<float>(output_q.scale);
        rope_params.product_to_output_q31_shift3[operation] =
            static_cast<int64_t>(std::llround(std::ldexp(
                static_cast<double>(product_ratio * 8.0f), 31)));
        rope_params.product_output_zero_points[operation] = output_q.zero_point;
    }
    const auto & rope_sub_output_q = operation_affine(
        *profile, *rope_sub, "output", 0);
    const auto & rope_add_output_q = operation_affine(
        *profile, *rope_add, "output", 0);
    const auto configure_combine = [&](
            const llama_qnn_operation & operation,
            const llama_qnn_affine_qparams & output_q,
            bool subtract,
            int parameter_offset,
            int output_index) {
        const auto & lhs_q = operation_affine(
            *profile, operation, "input", 0);
        const auto & rhs_q = operation_affine(
            *profile, operation, "input", 1);
        const float output_reciprocal =
            1.0f / static_cast<float>(output_q.scale);
        const float ratios[2] = {
            static_cast<float>(lhs_q.scale) * output_reciprocal,
            static_cast<float>(rhs_q.scale) * output_reciprocal,
        };
        const int32_t shift = static_cast<int32_t>(std::floor(std::log2(
            32767.0f / std::max(ratios[0], ratios[1])))) - 1;
        rope_params.combine_shifts[output_index] = shift;
        for (int input = 0; input < 2; ++input) {
            rope_params.combine_multipliers_q15[parameter_offset + input] =
                std::min(INT16_MAX, static_cast<int32_t>(std::llround(
                    std::ldexp(static_cast<double>(ratios[input]), shift + 1))));
        }
        const int64_t zero_product =
            static_cast<int64_t>(lhs_q.zero_point) *
                rope_params.combine_multipliers_q15[parameter_offset] +
            (subtract ? -1 : 1) * static_cast<int64_t>(rhs_q.zero_point) *
                rope_params.combine_multipliers_q15[parameter_offset + 1];
        const int64_t zero_average = zero_product >= 0
            ? zero_product / 2
            : -((-zero_product + 1) / 2);
        rope_params.combine_biases[output_index] =
            (static_cast<int64_t>(output_q.zero_point) << shift) -
            zero_average;
    };
    configure_combine(
        *rope_sub, rope_sub_output_q, true, 0, 0);
    configure_combine(
        *rope_add, rope_add_output_q, false, 2, 1);
    std::printf(
        "qnn-profile-rope-params:"
        " split_zp=%d split_q31=[%lld,%lld] split_out_zp=[%d,%d]"
        " lhs_zp=[%d,%d,%d,%d] table_zp=[%d,%d,%d,%d]"
        " product_q31_shift3=[%lld,%lld,%lld,%lld]"
        " product_out_zp=[%d,%d,%d,%d]"
        " combine_q15=[%d,%d,%d,%d]"
        " combine_shift=[%d,%d] combine_bias=[%lld,%lld]\n",
        rope_params.split_input_zero_point,
        static_cast<long long>(rope_params.split_to_output_q31[0]),
        static_cast<long long>(rope_params.split_to_output_q31[1]),
        rope_params.split_output_zero_points[0],
        rope_params.split_output_zero_points[1],
        rope_params.lhs_zero_points[0],
        rope_params.lhs_zero_points[1],
        rope_params.lhs_zero_points[2],
        rope_params.lhs_zero_points[3],
        rope_params.table_zero_points[0],
        rope_params.table_zero_points[1],
        rope_params.table_zero_points[2],
        rope_params.table_zero_points[3],
        static_cast<long long>(rope_params.product_to_output_q31_shift3[0]),
        static_cast<long long>(rope_params.product_to_output_q31_shift3[1]),
        static_cast<long long>(rope_params.product_to_output_q31_shift3[2]),
        static_cast<long long>(rope_params.product_to_output_q31_shift3[3]),
        rope_params.product_output_zero_points[0],
        rope_params.product_output_zero_points[1],
        rope_params.product_output_zero_points[2],
        rope_params.product_output_zero_points[3],
        rope_params.combine_multipliers_q15[0],
        rope_params.combine_multipliers_q15[1],
        rope_params.combine_multipliers_q15[2],
        rope_params.combine_multipliers_q15[3],
        rope_params.combine_shifts[0],
        rope_params.combine_shifts[1],
        static_cast<long long>(rope_params.combine_biases[0]),
        static_cast<long long>(rope_params.combine_biases[1]));
    for (int64_t token = 0; token < rope_tokens; ++token) {
        const uint16_t * source = rope_input_data + token * rope_dimension;
        const int64_t table_offset =
            static_cast<int64_t>(rope_position_data[token]) * rope_half_dimension;
        uint16_t * destination = expected_rope.data() + token * rope_dimension;
        ggml_vec_rope_affine_u16_qnn_fixed(
            static_cast<int>(rope_half_dimension), destination, source,
            rope_cos->static_data.data() + table_offset,
            rope_sin->static_data.data() + table_offset,
            &rope_params);
    }
    constexpr int rope_benchmark_iterations = 20000;
    std::array<uint16_t, rope_dimension> rope_benchmark_output {};
    uint64_t rope_benchmark_checksum = 0;
    const auto rope_benchmark_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < rope_benchmark_iterations; ++iteration) {
        const int64_t token = iteration % rope_tokens;
        const int64_t table_offset =
            static_cast<int64_t>(rope_position_data[token]) * rope_half_dimension;
        ggml_vec_rope_affine_u16_qnn_fixed(
            static_cast<int>(rope_half_dimension),
            rope_benchmark_output.data(),
            rope_input_data + token * rope_dimension,
            rope_cos->static_data.data() + table_offset,
            rope_sin->static_data.data() + table_offset,
            &rope_params);
        rope_benchmark_checksum +=
            rope_benchmark_output[iteration % rope_dimension];
    }
    const auto rope_benchmark_end = std::chrono::steady_clock::now();
    const double rope_benchmark_us =
        std::chrono::duration<double, std::micro>(
            rope_benchmark_end - rope_benchmark_start).count() /
        rope_benchmark_iterations;
    std::printf(
        "qnn-profile-rope-benchmark: dimension=%lld us=%.6f checksum=%llu\n",
        static_cast<long long>(rope_dimension),
        rope_benchmark_us,
        static_cast<unsigned long long>(rope_benchmark_checksum));
    const auto * rotation_input =
        profile->find_u16_operand(*qk_rotation, "input", 0);
    const auto * rotation_weight =
        profile->find_aux_operand(*qk_rotation, "input", 1);
    const auto * rotation_output =
        profile->find_u16_operand(*qk_rotation, "output", 0);
    if (rotation_input == nullptr || rotation_weight == nullptr ||
        rotation_output == nullptr ||
        rotation_weight->data_type != "QNN_DATATYPE_SFIXED_POINT_16" ||
        rotation_weight->static_data.size() !=
            rope_dimension * rope_dimension * sizeof(int16_t) ||
        qk_rotation->matmul_product_to_output_q31 <= 0) {
        ggml_free(ctx);
        throw std::runtime_error("profile Q/K S16 rotation is incomplete");
    }
    const auto * rotation_weights =
        reinterpret_cast<const int16_t *>(rotation_weight->static_data.data());
    const int32_t rotation_input_zero_point =
        rotation_input->qparams.scale_offsets[0].zero_point;
    const int32_t rotation_output_zero_point =
        rotation_output->qparams.scale_offsets[0].zero_point;
    const double rotation_effective_ratio =
        (
            static_cast<double>(
                rotation_input->qparams.scale_offsets[0].scale)
            * static_cast<double>(
                rotation_weight->qparams.scale_offsets[0].scale)
        ) /
        static_cast<double>(
            rotation_output->qparams.scale_offsets[0].scale);
    const float rotation_reduced_scale =
        static_cast<float>(std::ldexp(rotation_effective_ratio, 16));
    const int64_t rotation_multiplier_q24 = static_cast<int64_t>(
        std::llround(std::ldexp(
            static_cast<double>(rotation_reduced_scale), 24)));
    const auto dense_rotation = [&](const uint16_t * source, uint16_t * destination) {
        int64_t input_sum = 0;
        for (int64_t row = 0; row < rope_dimension; ++row) {
            input_sum +=
                static_cast<int32_t>(source[row]) - rotation_input_zero_point;
        }
        for (int64_t column = 0; column < rope_dimension; ++column) {
            int64_t transformed = 0;
            for (int64_t row = 0; row < rope_dimension; ++row) {
                const int64_t input_value =
                    static_cast<int32_t>(source[row]) - rotation_input_zero_point;
                const int16_t weight =
                    rotation_weights[row * rope_dimension + column];
                transformed += weight > 0 ? input_value : -input_value;
            }
            const int64_t centered =
                INT64_C(32703) * transformed - INT64_C(64) * input_sum;
            const int64_t weight_sum =
                column == 0 ? INT64_C(4177792) : -8192;
            const int64_t correction =
                static_cast<int64_t>(rotation_input_zero_point) * weight_sum;
            const int64_t reduced =
                floor_shift_i64(centered + correction, 16)
                - round_shift_htp(correction, 16);
            destination[column] = saturate_u16(
                round_shift_htp(
                    reduced * rotation_multiplier_q24, 24) +
                rotation_output_zero_point);
        }
    };
    for (int64_t token = 0; token < rope_tokens; ++token) {
        dense_rotation(
            expected_rope.data() + token * rope_dimension,
            expected_rotated.data() + token * rope_dimension);
    }

    constexpr int rotation_benchmark_iterations = 2000;
    std::array<uint16_t, rope_dimension> rotation_benchmark_output {};
    uint64_t rotation_benchmark_checksum = 0;
    const auto optimized_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < rotation_benchmark_iterations; ++iteration) {
        ggml_vec_matmul_u16_s16_qnn_fixed(
            static_cast<int>(rope_dimension),
            static_cast<int>(rope_dimension),
            rotation_benchmark_output.data(),
            expected_rope.data() +
                (iteration % rope_tokens) * rope_dimension,
            rotation_weights,
            rotation_input_zero_point,
            rotation_weight->qparams.scale_offsets[0].zero_point,
            rotation_multiplier_q24,
            rotation_output_zero_point,
            7,
            16);
        rotation_benchmark_checksum +=
            rotation_benchmark_output[iteration % rope_dimension];
    }
    const auto optimized_end = std::chrono::steady_clock::now();
    const auto dense_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < rotation_benchmark_iterations; ++iteration) {
        dense_rotation(
            expected_rope.data() +
                (iteration % rope_tokens) * rope_dimension,
            rotation_benchmark_output.data());
        rotation_benchmark_checksum +=
            rotation_benchmark_output[iteration % rope_dimension];
    }
    const auto dense_end = std::chrono::steady_clock::now();
    const double rotation_optimized_us =
        std::chrono::duration<double, std::micro>(optimized_end - optimized_start).count() /
        rotation_benchmark_iterations;
    const double rotation_dense_us =
        std::chrono::duration<double, std::micro>(dense_end - dense_start).count() /
        rotation_benchmark_iterations;
    const llama_qnn_aux_quantized_tensor * k_convert_output =
        profile->find_aux_operand(*k_convert, "output", 0);
    const llama_qnn_aux_quantized_tensor * v_convert_output =
        profile->find_aux_operand(*v_convert, "output", 0);
    if (k_convert_output == nullptr || v_convert_output == nullptr ||
        k_convert_output->data_type != "QNN_DATATYPE_UFIXED_POINT_8" ||
        v_convert_output->data_type != "QNN_DATATYPE_UFIXED_POINT_8" ||
        k_convert_output->qparams.scale_offsets.size() != 1 ||
        v_convert_output->qparams.scale_offsets.size() != 1 ||
        k_convert->unary_input_to_output_q31 <= 0 ||
        v_convert->unary_input_to_output_q31 <= 0) {
        ggml_free(ctx);
        throw std::runtime_error("profile K/V U16-to-U8 conversion is incomplete");
    }
    const llama_qnn_aux_quantized_tensor * qk_key_qparams =
        profile->find_aux_operand(*qk_matmul, "input", 1);
    const llama_qnn_u16_tensor * qk_output_qparams =
        profile->find_u16_operand(*qk_matmul, "output", 0);
    if (qk_key_qparams == nullptr || qk_output_qparams == nullptr ||
        qk_key_qparams->data_type != "QNN_DATATYPE_UFIXED_POINT_8" ||
        qk_key_qparams->qparams.scale_offsets.size() != 1 ||
        qk_output_qparams->qparams.scale_offsets.size() != 1 ||
        qk_matmul->matmul_product_to_output_q31 <= 0) {
        ggml_free(ctx);
        throw std::runtime_error("profile QK U16-by-U8 MatMul is incomplete");
    }
    const auto * floor_constant_qparams =
        profile->find_u16_operand(*attention_floor_add, "input", 1);
    if (attention_divide->unary_input_to_output_q20 <= 0 ||
        attention_min->unary_input_to_output_q20 <= 0 ||
        attention_floor_add->input_to_output_q20.size() < 2 ||
        attention_select->input_to_output_q20.size() < 3 ||
        attention_softmax->softmax_exp2_lut_q31.size() != 257 ||
        floor_constant_qparams == nullptr ||
        floor_constant_qparams->static_data.size() != 1) {
        ggml_free(ctx);
        throw std::runtime_error("profile attention scale/mask/softmax chain is incomplete");
    }
    const auto * attention_value_qparams =
        profile->find_aux_operand(*attention_value_matmul, "input", 1);
    const auto * attention_output_qparams =
        profile->find_u16_operand(*attention_value_matmul, "output", 0);
    if (attention_value_qparams == nullptr || attention_output_qparams == nullptr ||
        attention_value_qparams->data_type != "QNN_DATATYPE_UFIXED_POINT_8" ||
        attention_value_qparams->qparams.scale_offsets.size() != 1 ||
        attention_output_qparams->qparams.scale_offsets.size() != 1 ||
        attention_value_matmul->matmul_product_to_output_q31 <= 0) {
        ggml_free(ctx);
        throw std::runtime_error("profile attention U16-by-U8 value MatMul is incomplete");
    }
    auto * qk_key_data = static_cast<uint8_t *>(qk_key_matrix->data);
    const int32_t qk_key_zero =
        qk_key_qparams->qparams.scale_offsets[0].zero_point;
    for (int64_t index = 0; index < attention_columns * rope_dimension; ++index) {
        const int32_t centered = static_cast<int32_t>((index * 43 + 163) % 255) - 127;
        qk_key_data[index] = static_cast<uint8_t>(std::clamp<int32_t>(
            qk_key_zero + centered, 0, UINT8_MAX));
    }
    auto * attention_value_data =
        static_cast<uint8_t *>(attention_value_matrix->data);
    const int32_t attention_value_zero =
        attention_value_qparams->qparams.scale_offsets[0].zero_point;
    for (int64_t index = 0; index < attention_columns * rope_dimension; ++index) {
        const int32_t centered = static_cast<int32_t>((index * 59 + 173) % 255) - 127;
        attention_value_data[index] = static_cast<uint8_t>(std::clamp<int32_t>(
            attention_value_zero + centered, 0, UINT8_MAX));
    }
    ggml_vec_convert_u16_u8_qnn_fixed(
        static_cast<int>(expected_k_cache.size()), expected_k_cache.data(),
        static_cast<const uint16_t *>(k_convert_input->data),
        k_convert_input_q.zero_point, k_convert->unary_input_to_output_q31,
        k_convert_output->qparams.scale_offsets[0].zero_point);
    ggml_vec_convert_u16_u8_qnn_fixed(
        static_cast<int>(expected_v_cache.size()), expected_v_cache.data(),
        static_cast<const uint16_t *>(v_convert_input->data),
        v_convert_input_q.zero_point, v_convert->unary_input_to_output_q31,
        v_convert_output->qparams.scale_offsets[0].zero_point);
    const int32_t qk_query_zero = operation_affine(
        *profile, *qk_matmul, "input", 0).zero_point;
    const int32_t qk_output_zero =
        qk_output_qparams->qparams.scale_offsets[0].zero_point;
    const auto reduce_qk_accumulator = [](
            int64_t centered,
            int64_t weight_sum,
            int32_t activation_zero_point) {
        const int64_t correction =
            static_cast<int64_t>(activation_zero_point) * weight_sum;
        return (
            floor_shift_i64(centered + correction, 8)
            - round_shift_htp(correction, 8)
        ) << 8;
    };
    const auto dense_qk_matmul = [&](const uint16_t * query, uint16_t * destination) {
        for (int64_t column = 0; column < attention_columns; ++column) {
            int64_t accumulator = 0;
            int64_t weight_sum = 0;
            for (int64_t row = 0; row < rope_dimension; ++row) {
                const int32_t weight =
                    static_cast<int32_t>(
                        qk_key_data[row * attention_columns + column])
                    - qk_key_zero;
                accumulator +=
                    (static_cast<int32_t>(query[row]) - qk_query_zero) * weight;
                weight_sum += weight;
            }
            accumulator = reduce_qk_accumulator(
                accumulator, weight_sum, qk_query_zero);
            destination[column] = saturate_u16(
                round_shift_away_from_zero(
                    accumulator * qk_matmul->matmul_product_to_output_q31, 31) +
                qk_output_zero);
        }
    };
    const auto * qk_query_data =
        static_cast<const uint16_t *>(qk_query->data);
    for (int64_t token = 0; token < rope_tokens; ++token) {
        dense_qk_matmul(
            qk_query_data + token * rope_dimension,
            expected_qk_logits.data() + token * attention_columns);
    }
    constexpr int64_t qk_benchmark_columns = 4096;
    constexpr int qk_benchmark_iterations = 100;
    std::vector<uint8_t> qk_benchmark_keys(
        rope_dimension * qk_benchmark_columns);
    for (int64_t index = 0; index < rope_dimension * qk_benchmark_columns; ++index) {
        const int32_t centered = static_cast<int32_t>((index * 43 + 163) % 255) - 127;
        qk_benchmark_keys[index] = static_cast<uint8_t>(std::clamp<int32_t>(
            qk_key_zero + centered, 0, UINT8_MAX));
    }
    std::vector<uint16_t> qk_benchmark_output(qk_benchmark_columns);
    const auto dense_qk_benchmark = [&](const uint16_t * query) {
        for (int64_t column = 0; column < qk_benchmark_columns; ++column) {
            int64_t accumulator = 0;
            int64_t weight_sum = 0;
            for (int64_t row = 0; row < rope_dimension; ++row) {
                const int32_t weight =
                    static_cast<int32_t>(
                        qk_benchmark_keys[row * qk_benchmark_columns + column])
                    - qk_key_zero;
                accumulator +=
                    (static_cast<int32_t>(query[row]) - qk_query_zero) * weight;
                weight_sum += weight;
            }
            accumulator = reduce_qk_accumulator(
                accumulator, weight_sum, qk_query_zero);
            qk_benchmark_output[column] = saturate_u16(
                round_shift_away_from_zero(
                    accumulator * qk_matmul->matmul_product_to_output_q31, 31) +
                qk_output_zero);
        }
    };
    uint64_t qk_benchmark_checksum = 0;
    const auto qk_optimized_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < qk_benchmark_iterations; ++iteration) {
        ggml_vec_matmul_u16_u8_qnn_fixed(
            static_cast<int>(rope_dimension),
            static_cast<int>(qk_benchmark_columns),
            qk_benchmark_output.data(),
            qk_query_data + (iteration % rope_tokens) * rope_dimension,
            qk_benchmark_keys.data(), qk_query_zero, qk_key_zero,
            qk_matmul->matmul_product_to_output_q31, qk_output_zero);
        qk_benchmark_checksum +=
            qk_benchmark_output[iteration % qk_benchmark_columns];
    }
    const auto qk_optimized_end = std::chrono::steady_clock::now();
    const auto qk_dense_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < qk_benchmark_iterations; ++iteration) {
        dense_qk_benchmark(
            qk_query_data + (iteration % rope_tokens) * rope_dimension);
        qk_benchmark_checksum +=
            qk_benchmark_output[iteration % qk_benchmark_columns];
    }
    const auto qk_dense_end = std::chrono::steady_clock::now();
    const double qk_optimized_us =
        std::chrono::duration<double, std::micro>(
            qk_optimized_end - qk_optimized_start).count() /
        qk_benchmark_iterations;
    const double qk_dense_us =
        std::chrono::duration<double, std::micro>(
            qk_dense_end - qk_dense_start).count() /
        qk_benchmark_iterations;
    dense_qk_benchmark(qk_query_data);
    const std::vector<uint16_t> qk_decode_expected = qk_benchmark_output;
    constexpr int64_t value_benchmark_rows = 4096;
    constexpr int64_t value_benchmark_columns = 128;
    constexpr int value_benchmark_iterations = 100;
    const auto & value_probability_q = operation_affine(
        *profile, *attention_value_matmul, "input", 0);
    const auto & value_output_q = operation_affine(
        *profile, *attention_value_matmul, "output", 0);
    std::vector<uint16_t> value_benchmark_probabilities(
        value_benchmark_rows, value_probability_q.zero_point);
    for (int index = 0; index < 32; ++index) {
        value_benchmark_probabilities[
            (index * 127 + 19) % value_benchmark_rows] =
            static_cast<uint16_t>(97 + index * 31);
    }
    std::vector<uint8_t> value_benchmark_matrix(
        value_benchmark_rows * value_benchmark_columns);
    for (int64_t index = 0;
         index < value_benchmark_rows * value_benchmark_columns;
         ++index) {
        value_benchmark_matrix[index] = static_cast<uint8_t>(
            (attention_value_zero + index * 29 + index / 17) & UINT8_MAX);
    }
    std::vector<uint16_t> value_benchmark_output(value_benchmark_columns);
    std::vector<uint16_t> value_dense_output(value_benchmark_columns);
    const auto dense_value_benchmark = [&]() {
        for (int64_t column = 0; column < value_benchmark_columns; ++column) {
            int64_t accumulator = 0;
            int64_t weight_sum = 0;
            for (int64_t row = 0; row < value_benchmark_rows; ++row) {
                const int32_t weight =
                    static_cast<int32_t>(
                        value_benchmark_matrix[
                            row * value_benchmark_columns + column]) -
                    attention_value_zero;
                accumulator +=
                    (static_cast<int32_t>(value_benchmark_probabilities[row]) -
                     value_probability_q.zero_point) * weight;
                weight_sum += weight;
            }
            accumulator = reduce_qk_accumulator(
                accumulator, weight_sum, value_probability_q.zero_point);
            value_dense_output[column] = saturate_u16(
                round_shift_away_from_zero(
                    accumulator *
                    attention_value_matmul->matmul_product_to_output_q31,
                    31) +
                value_output_q.zero_point);
        }
    };
    uint64_t value_benchmark_checksum = 0;
    const auto value_sparse_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < value_benchmark_iterations; ++iteration) {
        ggml_vec_matmul_u16_u8_qnn_fixed(
            value_benchmark_rows,
            value_benchmark_columns,
            value_benchmark_output.data(),
            value_benchmark_probabilities.data(),
            value_benchmark_matrix.data(),
            value_probability_q.zero_point,
            attention_value_zero,
            attention_value_matmul->matmul_product_to_output_q31,
            value_output_q.zero_point);
        value_benchmark_checksum +=
            value_benchmark_output[iteration % value_benchmark_columns];
    }
    const auto value_sparse_end = std::chrono::steady_clock::now();
    uint64_t value_dense_checksum = 0;
    const auto value_dense_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < value_benchmark_iterations; ++iteration) {
        dense_value_benchmark();
        value_dense_checksum +=
            value_dense_output[iteration % value_benchmark_columns];
    }
    const auto value_dense_end = std::chrono::steady_clock::now();
    const double value_sparse_us =
        std::chrono::duration<double, std::micro>(
            value_sparse_end - value_sparse_start).count() /
        value_benchmark_iterations;
    const double value_dense_us =
        std::chrono::duration<double, std::micro>(
            value_dense_end - value_dense_start).count() /
        value_benchmark_iterations;
    const bool value_sparse_match =
        value_benchmark_output == value_dense_output;

    // Benchmark the actual CUSTOM scheduling shape used by Decode: one query
    // row, with output columns divided among the GGML workers.
    ggml_tensor * qk_decode_input = ggml_new_tensor_1d(
        ctx, GGML_TYPE_U16, rope_dimension);
    ggml_tensor * qk_decode_matrix = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I8, qk_benchmark_columns, rope_dimension);
    constexpr int decode_matmul_graph_nodes = 64;
    std::vector<ggml_tensor *> qk_decode_outputs;
    qk_decode_outputs.reserve(decode_matmul_graph_nodes);
    for (int node = 0; node < decode_matmul_graph_nodes; ++node) {
        qk_decode_outputs.push_back(llama_qnn_u16_u8_matmul(
            ctx, qk_decode_input, qk_decode_matrix, profile.get(), qk_matmul));
    }
    std::memcpy(
        qk_decode_input->data, qk_query_data,
        rope_dimension * sizeof(uint16_t));
    std::memcpy(
        qk_decode_matrix->data, qk_benchmark_keys.data(),
        qk_benchmark_keys.size() * sizeof(uint8_t));
    ggml_cgraph * qk_decode_graph = ggml_new_graph_custom(ctx, 128, false);
    for (ggml_tensor * output : qk_decode_outputs) {
        ggml_build_forward_expand(qk_decode_graph, output);
    }

    ggml_tensor * value_decode_input = ggml_new_tensor_1d(
        ctx, GGML_TYPE_U16, value_benchmark_rows);
    ggml_tensor * value_decode_matrix = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I8, value_benchmark_columns, value_benchmark_rows);
    std::vector<ggml_tensor *> value_decode_outputs;
    value_decode_outputs.reserve(decode_matmul_graph_nodes);
    for (int node = 0; node < decode_matmul_graph_nodes; ++node) {
        value_decode_outputs.push_back(llama_qnn_u16_u8_matmul(
            ctx, value_decode_input, value_decode_matrix,
            profile.get(), attention_value_matmul));
    }
    std::memcpy(
        value_decode_input->data, value_benchmark_probabilities.data(),
        value_benchmark_probabilities.size() * sizeof(uint16_t));
    std::memcpy(
        value_decode_matrix->data, value_benchmark_matrix.data(),
        value_benchmark_matrix.size() * sizeof(uint8_t));
    ggml_cgraph * value_decode_graph = ggml_new_graph_custom(ctx, 128, false);
    for (ggml_tensor * output : value_decode_outputs) {
        ggml_build_forward_expand(value_decode_graph, output);
    }

    constexpr int decode_matmul_graph_iterations = 3;
    const auto benchmark_decode_graph = [](
            ggml_context * graph_ctx,
            ggml_cgraph * decode_graph,
            int threads) {
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0;
             iteration < decode_matmul_graph_iterations;
             ++iteration) {
            GGML_ASSERT(
                ggml_graph_compute_with_ctx(
                    graph_ctx, decode_graph, threads) == GGML_STATUS_SUCCESS);
        }
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count() /
            (decode_matmul_graph_iterations * decode_matmul_graph_nodes);
    };
    const double qk_decode_graph_1t_us =
        benchmark_decode_graph(ctx, qk_decode_graph, 1);
    const bool qk_decode_graph_match = same_values(
        static_cast<const uint16_t *>(qk_decode_outputs.front()->data),
        qk_decode_expected.data(), qk_decode_expected.size());
    const double qk_decode_graph_4t_us =
        benchmark_decode_graph(ctx, qk_decode_graph, 4);
    const double value_decode_graph_1t_us =
        benchmark_decode_graph(ctx, value_decode_graph, 1);
    const bool value_decode_graph_match = same_values(
        static_cast<const uint16_t *>(value_decode_outputs.front()->data),
        value_benchmark_output.data(), value_benchmark_output.size());
    const double value_decode_graph_4t_us =
        benchmark_decode_graph(ctx, value_decode_graph, 4);
    const auto & divide_input_q = operation_affine(
        *profile, *attention_divide, "input", 0);
    const auto & divide_output_q = operation_affine(
        *profile, *attention_divide, "output", 0);
    const auto & min_input_q = operation_affine(
        *profile, *attention_min, "input", 0);
    const auto & min_output_q = operation_affine(
        *profile, *attention_min, "output", 0);
    const auto & floor_lhs_q = operation_affine(
        *profile, *attention_floor_add, "input", 0);
    const auto & floor_rhs_q = operation_affine(
        *profile, *attention_floor_add, "input", 1);
    const auto & floor_output_q = operation_affine(
        *profile, *attention_floor_add, "output", 0);
    const auto & select_true_q = operation_affine(
        *profile, *attention_select, "input", 1);
    const auto & select_false_q = operation_affine(
        *profile, *attention_select, "input", 2);
    const auto & select_output_q = operation_affine(
        *profile, *attention_select, "output", 0);
    const htp_scalefactor_q15 select_true_scalefactor =
        make_htp_scalefactor_q15(select_true_q, select_output_q);
    const htp_scalefactor_q15 select_false_scalefactor =
        make_htp_scalefactor_q15(select_false_q, select_output_q);
    const auto & softmax_output_q = operation_affine(
        *profile, *attention_softmax, "output", 0);
    auto * floor_constant_data =
        static_cast<uint16_t *>(mask_floor_constant->data);
    auto * condition_data = static_cast<uint8_t *>(attention_condition->data);
    for (int64_t token = 0; token < rope_tokens; ++token) {
        floor_constant_data[token] = floor_constant_qparams->static_data.front();
        for (int64_t column = 0; column < attention_columns; ++column) {
            condition_data[token * attention_columns + column] =
                column <= token + 7 ? 1 : 0;
        }
        const size_t row_offset = static_cast<size_t>(token * attention_columns);
        ggml_vec_requant_u16_qnn_fixed(
            static_cast<int>(attention_columns),
            expected_scaled_logits.data() + row_offset,
            expected_qk_logits.data() + row_offset,
            divide_input_q.zero_point,
            attention_divide->unary_input_to_output_q20,
            divide_output_q.zero_point);
        const uint16_t raw_minimum = *std::min_element(
            expected_scaled_logits.data() + row_offset,
            expected_scaled_logits.data() + row_offset + attention_columns);
        ggml_vec_requant_u16_qnn_fixed(
            1, expected_minimum_logits.data() + token, &raw_minimum,
            min_input_q.zero_point, attention_min->unary_input_to_output_q20,
            min_output_q.zero_point);
        ggml_vec_add_affine_u16_qnn_q15(
            1, expected_mask_floor.data() + token,
            expected_minimum_logits.data() + token,
            affine_ratio_q15(floor_lhs_q, floor_output_q), floor_lhs_q.zero_point,
            floor_constant_data + token,
            affine_ratio_q15(floor_rhs_q, floor_output_q), floor_rhs_q.zero_point,
            floor_output_q.zero_point);
        ggml_vec_select_affine_u16_qnn_fixed(
            static_cast<int>(attention_columns),
            expected_masked_logits.data() + row_offset,
            condition_data + row_offset,
            expected_scaled_logits.data() + row_offset,
            select_true_q.zero_point, select_true_scalefactor.multiplier,
            select_true_scalefactor.right_shift,
            expected_mask_floor.data() + token, 0,
            select_false_q.zero_point, select_false_scalefactor.multiplier,
            select_false_scalefactor.right_shift,
            select_output_q.zero_point);
        ggml_vec_softmax_u16_qnn_fixed(
            static_cast<int>(attention_columns),
            expected_attention_probabilities.data() + row_offset,
            expected_masked_logits.data() + row_offset,
            attention_softmax->softmax_scale_over_ln2_q24,
            attention_softmax->softmax_unit_code,
            softmax_output_q.zero_point,
            attention_softmax->softmax_exp2_lut_q31.data());
        ggml_vec_matmul_u16_u8_qnn_fixed(
            static_cast<int>(attention_columns),
            static_cast<int>(rope_dimension),
            expected_attention_output.data() + token * rope_dimension,
            expected_attention_probabilities.data() + row_offset,
            attention_value_data,
            operation_affine(
                *profile, *attention_value_matmul, "input", 0).zero_point,
            attention_value_zero,
            attention_value_matmul->matmul_product_to_output_q31,
            attention_output_qparams->qparams.scale_offsets[0].zero_point);
    }

    constexpr int select_benchmark_width = 4096;
    constexpr int select_benchmark_iterations = 1000;
    std::vector<uint8_t> select_benchmark_condition(select_benchmark_width);
    std::vector<uint16_t> select_benchmark_true(select_benchmark_width);
    std::array<uint16_t, 1> select_benchmark_false = {
        expected_mask_floor.front(),
    };
    std::vector<uint16_t> select_benchmark_output(select_benchmark_width);
    std::vector<uint16_t> select_reference_output(select_benchmark_width);
    for (int index = 0; index < select_benchmark_width; ++index) {
        select_benchmark_condition[index] = (index % 11) < 7 ? 1 : 0;
        select_benchmark_true[index] =
            expected_scaled_logits[
                (static_cast<size_t>(index) * 17) %
                expected_scaled_logits.size()];
    }
    const auto select_scalar_reference = [&]() {
        for (int index = 0; index < select_benchmark_width; ++index) {
            const bool use_true = select_benchmark_condition[index] != 0;
            const int64_t product = use_true
                ? (static_cast<int32_t>(select_benchmark_true[index]) -
                    select_true_q.zero_point) *
                    select_true_scalefactor.multiplier
                : (static_cast<int32_t>(select_benchmark_false[0]) -
                    select_false_q.zero_point) *
                    select_false_scalefactor.multiplier;
            const int32_t right_shift = use_true
                ? select_true_scalefactor.right_shift
                : select_false_scalefactor.right_shift;
            const int64_t centered = round_shift_htp(product, right_shift);
            select_reference_output[index] =
                saturate_u16(centered + select_output_q.zero_point);
        }
    };
    uint64_t select_benchmark_checksum = 0;
    const auto select_benchmark_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < select_benchmark_iterations; ++iteration) {
        ggml_vec_select_affine_u16_qnn_fixed(
            select_benchmark_width,
            select_benchmark_output.data(),
            select_benchmark_condition.data(),
            select_benchmark_true.data(),
            select_true_q.zero_point,
            select_true_scalefactor.multiplier,
            select_true_scalefactor.right_shift,
            select_benchmark_false.data(),
            0,
            select_false_q.zero_point,
            select_false_scalefactor.multiplier,
            select_false_scalefactor.right_shift,
            select_output_q.zero_point);
        select_benchmark_checksum +=
            select_benchmark_output[iteration % select_benchmark_width];
    }
    const auto select_benchmark_end = std::chrono::steady_clock::now();
    uint64_t select_reference_checksum = 0;
    const auto select_reference_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < select_benchmark_iterations; ++iteration) {
        select_scalar_reference();
        select_reference_checksum +=
            select_reference_output[iteration % select_benchmark_width];
    }
    const auto select_reference_end = std::chrono::steady_clock::now();
    const double select_benchmark_us =
        std::chrono::duration<double, std::micro>(
            select_benchmark_end - select_benchmark_start).count() /
        select_benchmark_iterations;
    const double select_reference_us =
        std::chrono::duration<double, std::micro>(
            select_reference_end - select_reference_start).count() /
        select_benchmark_iterations;
    const bool select_reference_match =
        select_benchmark_output == select_reference_output;

    constexpr int select_stride_width = 4103;
    std::vector<uint8_t> select_stride_condition(select_stride_width);
    std::vector<uint16_t> select_stride_true(select_stride_width);
    std::vector<uint16_t> select_stride_false(select_stride_width);
    std::vector<uint16_t> select_stride_output(select_stride_width);
    std::vector<uint16_t> select_stride_reference(select_stride_width);
    constexpr int32_t select_stride_true_zero = 32768;
    constexpr int32_t select_stride_false_zero = 32767;
    constexpr int32_t select_stride_true_multiplier = 24577;
    constexpr int32_t select_stride_true_shift = 14;
    constexpr int32_t select_stride_false_multiplier = 32763;
    constexpr int32_t select_stride_false_shift = 15;
    constexpr int32_t select_stride_output_zero = 32000;
    for (int index = 0; index < select_stride_width; ++index) {
        static constexpr std::array<uint16_t, 8> edge_codes = {
            0, 1, 32766, 32767, 32768, 32769, 65534, 65535,
        };
        select_stride_condition[index] =
            static_cast<uint8_t>((index % 4) == 0 ? 0 : (index % 255) + 1);
        select_stride_true[index] = edge_codes[index % edge_codes.size()];
        select_stride_false[index] =
            edge_codes[(index * 5 + 3) % edge_codes.size()];
        const bool use_true = select_stride_condition[index] != 0;
        const int64_t product = use_true
            ? (static_cast<int32_t>(select_stride_true[index]) -
                select_stride_true_zero) * select_stride_true_multiplier
            : (static_cast<int32_t>(select_stride_false[index]) -
                select_stride_false_zero) * select_stride_false_multiplier;
        const int64_t centered = round_shift_htp(
            product, use_true ? select_stride_true_shift : select_stride_false_shift);
        select_stride_reference[index] =
            saturate_u16(centered + select_stride_output_zero);
    }
    ggml_vec_select_affine_u16_qnn_fixed(
        select_stride_width,
        select_stride_output.data(),
        select_stride_condition.data(),
        select_stride_true.data(),
        select_stride_true_zero,
        select_stride_true_multiplier,
        select_stride_true_shift,
        select_stride_false.data(),
        1,
        select_stride_false_zero,
        select_stride_false_multiplier,
        select_stride_false_shift,
        select_stride_output_zero);
    const bool select_stride_reference_match =
        select_stride_output == select_stride_reference;

    constexpr int softmax_benchmark_iterations = 200;
    std::vector<uint16_t> softmax_benchmark_input(qk_benchmark_columns);
    std::vector<uint16_t> softmax_benchmark_output(qk_benchmark_columns);
    std::vector<uint16_t> softmax_reference_output(qk_benchmark_columns);
    for (int64_t index = 0; index < qk_benchmark_columns; ++index) {
        softmax_benchmark_input[index] =
            expected_masked_logits[
                (index * 13 + index / attention_columns) %
                expected_masked_logits.size()];
    }
    uint64_t softmax_benchmark_checksum = 0;
    const auto softmax_benchmark_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < softmax_benchmark_iterations; ++iteration) {
        ggml_vec_softmax_u16_qnn_fixed(
            static_cast<int>(qk_benchmark_columns),
            softmax_benchmark_output.data(),
            softmax_benchmark_input.data(),
            attention_softmax->softmax_scale_over_ln2_q24,
            attention_softmax->softmax_unit_code,
            softmax_output_q.zero_point,
            attention_softmax->softmax_exp2_lut_q31.data());
        softmax_benchmark_checksum +=
            softmax_benchmark_output[iteration % qk_benchmark_columns];
    }
    const auto softmax_benchmark_end = std::chrono::steady_clock::now();
    const double softmax_benchmark_us =
        std::chrono::duration<double, std::micro>(
            softmax_benchmark_end - softmax_benchmark_start).count() /
        softmax_benchmark_iterations;
    const auto softmax_exp_reference = [&](uint32_t code_delta) {
        const uint64_t exponent_q24 = static_cast<uint64_t>(code_delta) *
            static_cast<uint64_t>(attention_softmax->softmax_scale_over_ln2_q24);
        const uint32_t integer_part = static_cast<uint32_t>(exponent_q24 >> 24);
        if (integer_part >= 31) {
            return uint32_t { 0 };
        }
        const uint32_t fraction = static_cast<uint32_t>(exponent_q24) & 0x00ffffffU;
        const uint32_t table_index = fraction >> 16;
        const uint32_t remainder = fraction & 0xffffU;
        const uint32_t high =
            attention_softmax->softmax_exp2_lut_q31[table_index];
        const uint32_t low =
            attention_softmax->softmax_exp2_lut_q31[table_index + 1];
        const uint32_t interpolated = high - static_cast<uint32_t>(
            (static_cast<uint64_t>(high - low) * remainder + 32768) >> 16);
        return (interpolated +
            (integer_part == 0 ? 0U : 1U << (integer_part - 1))) >>
            integer_part;
    };
    const auto softmax_division_reference = [&]() {
        const uint16_t maximum = *std::max_element(
            softmax_benchmark_input.begin(), softmax_benchmark_input.end());
        uint64_t sum = 0;
        for (uint16_t code : softmax_benchmark_input) {
            sum += softmax_exp_reference(
                static_cast<uint32_t>(maximum) - code);
        }
        for (int64_t index = 0; index < qk_benchmark_columns; ++index) {
            const uint64_t exponential = softmax_exp_reference(
                static_cast<uint32_t>(maximum) -
                softmax_benchmark_input[index]);
            const int64_t centered = static_cast<int64_t>(
                exponential *
                static_cast<uint64_t>(attention_softmax->softmax_unit_code) /
                sum);
            softmax_reference_output[index] = saturate_u16(
                centered + softmax_output_q.zero_point);
        }
    };
    uint64_t softmax_reference_checksum = 0;
    const auto softmax_reference_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < softmax_benchmark_iterations; ++iteration) {
        softmax_division_reference();
        softmax_reference_checksum +=
            softmax_reference_output[iteration % qk_benchmark_columns];
    }
    const auto softmax_reference_end = std::chrono::steady_clock::now();
    const double softmax_reference_us =
        std::chrono::duration<double, std::micro>(
            softmax_reference_end - softmax_reference_start).count() /
        softmax_benchmark_iterations;
    const bool softmax_reciprocal_match =
        softmax_benchmark_output == softmax_reference_output;

    // A QNN-layout KV view usually has padding between logical matrix rows.
    // Verify that the strided kernel reads the same U8 codes as a compact
    // matrix without introducing a transpose buffer.
    constexpr int stride_test_k = 5;
    constexpr int stride_test_n = 11;
    constexpr size_t stride_test_row_stride = 19;
    std::array<uint16_t, stride_test_k> stride_test_input = {
        32001, 32768, 33017, 31991, 34002,
    };
    std::array<uint8_t, stride_test_k * stride_test_n> stride_test_compact {};
    std::array<uint8_t, stride_test_k * stride_test_row_stride> stride_test_padded {};
    for (int row = 0; row < stride_test_k; ++row) {
        for (int column = 0; column < stride_test_n; ++column) {
            const uint8_t code = static_cast<uint8_t>((row * 37 + column * 13 + 71) & 0xff);
            stride_test_compact[row * stride_test_n + column] = code;
            stride_test_padded[row * stride_test_row_stride + column] = code;
        }
    }
    std::array<uint16_t, stride_test_n> stride_test_expected {};
    std::array<uint16_t, stride_test_n> stride_test_actual {};
    ggml_vec_matmul_u16_u8_qnn_fixed(
        stride_test_k, stride_test_n, stride_test_expected.data(),
        stride_test_input.data(), stride_test_compact.data(),
        32768, 127, 1048576, 32768);
    ggml_vec_matmul_u16_u8_qnn_fixed_strided(
        stride_test_k, stride_test_n, stride_test_row_stride,
        stride_test_actual.data(), stride_test_input.data(),
        stride_test_padded.data(), 32768, 127, 1048576, 32768);
    const bool strided_u8_matmul_match = stride_test_actual == stride_test_expected;

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 104, false);
    ggml_build_forward_expand(graph, rms_output);
    ggml_build_forward_expand(graph, add_output);
    ggml_build_forward_expand(graph, sub_output);
    ggml_build_forward_expand(graph, mul_output);
    ggml_build_forward_expand(graph, sigmoid_output);
    ggml_build_forward_expand(graph, rope_output);
    ggml_build_forward_expand(graph, rotated_output);
    ggml_build_forward_expand(graph, k_cache_codes);
    ggml_build_forward_expand(graph, v_cache_codes);
    ggml_build_forward_expand(graph, qk_logits);
    ggml_build_forward_expand(graph, attention_probabilities);
    ggml_build_forward_expand(graph, attention_output);
    int f32_nodes = 0;
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        f32_nodes += ggml_graph_node(graph, index)->type == GGML_TYPE_F32;
    }
    const enum ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, 4);
    const bool rms_match = status == GGML_STATUS_SUCCESS && same_values(
        static_cast<const uint16_t *>(rms_output->data), expected_rms.data(), rms_width);
    const bool add_match = status == GGML_STATUS_SUCCESS && same_values(
        static_cast<const uint16_t *>(add_output->data), expected_add.data(), add_width);
    const bool sub_match = status == GGML_STATUS_SUCCESS && same_values(
        static_cast<const uint16_t *>(sub_output->data), expected_sub.data(), sub_width);
    const bool mul_match = status == GGML_STATUS_SUCCESS && same_values(
        static_cast<const uint16_t *>(mul_output->data), expected_mul.data(), mul_width);
    const bool sigmoid_match = status == GGML_STATUS_SUCCESS && same_values(
        static_cast<const uint16_t *>(sigmoid_output->data),
        expected_sigmoid.data(), sigmoid_width);
    const bool rope_match = status == GGML_STATUS_SUCCESS && same_values(
        static_cast<const uint16_t *>(rope_output->data),
        expected_rope.data(), expected_rope.size());
    const bool rotation_match = status == GGML_STATUS_SUCCESS && same_values(
        static_cast<const uint16_t *>(rotated_output->data),
        expected_rotated.data(), expected_rotated.size());
    const bool k_convert_match = status == GGML_STATUS_SUCCESS && std::equal(
        static_cast<const uint8_t *>(k_cache_codes->data),
        static_cast<const uint8_t *>(k_cache_codes->data) + expected_k_cache.size(),
        expected_k_cache.data());
    const bool v_convert_match = status == GGML_STATUS_SUCCESS && std::equal(
        static_cast<const uint8_t *>(v_cache_codes->data),
        static_cast<const uint8_t *>(v_cache_codes->data) + expected_v_cache.size(),
        expected_v_cache.data());
    const bool qk_matmul_match = status == GGML_STATUS_SUCCESS && same_values(
        static_cast<const uint16_t *>(qk_logits->data),
        expected_qk_logits.data(), expected_qk_logits.size());
    const bool attention_chain_match = status == GGML_STATUS_SUCCESS &&
        same_values(static_cast<const uint16_t *>(scaled_logits->data),
            expected_scaled_logits.data(), expected_scaled_logits.size()) &&
        same_values(static_cast<const uint16_t *>(minimum_logits->data),
            expected_minimum_logits.data(), expected_minimum_logits.size()) &&
        same_values(static_cast<const uint16_t *>(mask_floor->data),
            expected_mask_floor.data(), expected_mask_floor.size()) &&
        same_values(static_cast<const uint16_t *>(masked_logits->data),
            expected_masked_logits.data(), expected_masked_logits.size()) &&
        same_values(static_cast<const uint16_t *>(attention_probabilities->data),
            expected_attention_probabilities.data(),
            expected_attention_probabilities.size());
    const bool attention_value_match = status == GGML_STATUS_SUCCESS && same_values(
        static_cast<const uint16_t *>(attention_output->data),
        expected_attention_output.data(), expected_attention_output.size());
    uint32_t softmax_max_code_delta = 0;
    uint32_t softmax_max_row_sum_delta = 0;
    const auto & softmax_input_q = operation_affine(
        *profile, *attention_softmax, "input", 0);
    for (int64_t token = 0; token < rope_tokens; ++token) {
        const size_t row_offset = static_cast<size_t>(token * attention_columns);
        double maximum = -std::numeric_limits<double>::infinity();
        for (int64_t column = 0; column < attention_columns; ++column) {
            maximum = std::max(maximum,
                (static_cast<int64_t>(expected_masked_logits[row_offset + column]) -
                    softmax_input_q.zero_point) *
                static_cast<double>(softmax_input_q.scale));
        }
        double sum = 0.0;
        for (int64_t column = 0; column < attention_columns; ++column) {
            const double value =
                (static_cast<int64_t>(expected_masked_logits[row_offset + column]) -
                    softmax_input_q.zero_point) *
                static_cast<double>(softmax_input_q.scale);
            sum += std::exp(value - maximum);
        }
        uint32_t row_code_sum = 0;
        for (int64_t column = 0; column < attention_columns; ++column) {
            const double value =
                (static_cast<int64_t>(expected_masked_logits[row_offset + column]) -
                    softmax_input_q.zero_point) *
                static_cast<double>(softmax_input_q.scale);
            const int64_t reference = static_cast<int64_t>(std::llround(
                std::exp(value - maximum) / sum /
                static_cast<double>(softmax_output_q.scale))) +
                softmax_output_q.zero_point;
            const uint16_t actual = expected_attention_probabilities[row_offset + column];
            softmax_max_code_delta = std::max<uint32_t>(softmax_max_code_delta,
                static_cast<uint32_t>(std::llabs(reference - actual)));
            row_code_sum += actual;
        }
        softmax_max_row_sum_delta = std::max<uint32_t>(softmax_max_row_sum_delta,
            static_cast<uint32_t>(std::llabs(
                static_cast<int64_t>(row_code_sum) -
                attention_softmax->softmax_unit_code)));
    }
    uint64_t checksum = fnv1a_u16(
        static_cast<const uint16_t *>(rms_output->data), rms_width);
    checksum = fnv1a_u16(static_cast<const uint16_t *>(add_output->data), add_width, checksum);
    checksum = fnv1a_u16(static_cast<const uint16_t *>(sub_output->data), sub_width, checksum);
    checksum = fnv1a_u16(static_cast<const uint16_t *>(mul_output->data), mul_width, checksum);
    checksum = fnv1a_u16(
        static_cast<const uint16_t *>(sigmoid_output->data), sigmoid_width, checksum);
    checksum = fnv1a_u16(
        static_cast<const uint16_t *>(rope_output->data), expected_rope.size(), checksum);
    checksum = fnv1a_u16(
        static_cast<const uint16_t *>(rotated_output->data),
        expected_rotated.size(), checksum);
    checksum = fnv1a_bytes(
        static_cast<const uint8_t *>(k_cache_codes->data),
        expected_k_cache.size(), checksum);
    checksum = fnv1a_bytes(
        static_cast<const uint8_t *>(v_cache_codes->data),
        expected_v_cache.size(), checksum);
    checksum = fnv1a_u16(
        static_cast<const uint16_t *>(qk_logits->data),
        expected_qk_logits.size(), checksum);
    checksum = fnv1a_u16(
        static_cast<const uint16_t *>(attention_probabilities->data),
        expected_attention_probabilities.size(), checksum);
    checksum = fnv1a_u16(
        static_cast<const uint16_t *>(attention_output->data),
        expected_attention_output.size(), checksum);
    std::printf(
        "qnn-profile-u16-op-test: rms=%d add=%d sub=%d mul=%d sigmoid=%d "
        "rope=%d qk_rotation=%d k_u8_convert=%d v_u8_convert=%d qk_matmul=%d "
        "attention_scale_mask_softmax=%d "
        "attention_value_matmul=%d strided_u8_matmul=%d softmax_max_code_delta=%u "
        "softmax_max_row_sum_delta=%u "
        "rms_width=%zu add_width=%zu sub_width=%zu mul_width=%zu sigmoid_width=%zu "
        "rope_tokens=%lld rope_dimension=%lld qk_rotation_q31=%lld "
        "qk_rotation_optimized_us=%.3f qk_rotation_dense_us=%.3f "
        "qk_rotation_speedup=%.2f qk_rotation_benchmark_checksum=%llu "
        "qk_matmul_optimized_us=%.3f qk_matmul_dense_us=%.3f "
        "qk_matmul_speedup=%.2f qk_matmul_benchmark_checksum=%llu "
        "value_sparse_us=%.3f value_dense_us=%.3f value_sparse_speedup=%.2f "
        "value_sparse_match=%d value_benchmark_checksum=%llu "
        "value_dense_checksum=%llu "
        "select_benchmark_us=%.3f select_scalar_us=%.3f "
        "select_speedup=%.2f select_reference_match=%d "
        "select_stride_reference_match=%d "
        "select_benchmark_checksum=%llu select_reference_checksum=%llu "
        "softmax_benchmark_us=%.3f softmax_division_reference_us=%.3f "
        "softmax_vs_reference_speedup=%.2f softmax_division_match=%d "
        "softmax_benchmark_checksum=%llu softmax_reference_checksum=%llu "
        "qk_decode_graph_1t_us=%.3f qk_decode_graph_4t_us=%.3f "
        "qk_decode_graph_speedup=%.2f qk_decode_graph_match=%d "
        "value_decode_graph_1t_us=%.3f value_decode_graph_4t_us=%.3f "
        "value_decode_graph_speedup=%.2f value_decode_graph_match=%d "
        "k_convert_q31=%lld v_convert_q31=%lld f32_nodes=%d "
        "activation_dequant_buffers=0 temporary_weight_buffers=0 "
        "checksum=0x%016llx status=%s\n",
        rms_match ? 1 : 0, add_match ? 1 : 0, sub_match ? 1 : 0, mul_match ? 1 : 0,
        sigmoid_match ? 1 : 0, rope_match ? 1 : 0, rotation_match ? 1 : 0,
        k_convert_match ? 1 : 0, v_convert_match ? 1 : 0, qk_matmul_match ? 1 : 0,
        attention_chain_match ? 1 : 0,
        attention_value_match ? 1 : 0, strided_u8_matmul_match ? 1 : 0,
        softmax_max_code_delta,
        softmax_max_row_sum_delta,
        rms_width, add_width, sub_width, mul_width, sigmoid_width,
        static_cast<long long>(rope_tokens), static_cast<long long>(rope_dimension),
        static_cast<long long>(qk_rotation->matmul_product_to_output_q31),
        rotation_optimized_us, rotation_dense_us,
        rotation_dense_us / rotation_optimized_us,
        static_cast<unsigned long long>(rotation_benchmark_checksum),
        qk_optimized_us, qk_dense_us, qk_dense_us / qk_optimized_us,
        static_cast<unsigned long long>(qk_benchmark_checksum),
        value_sparse_us, value_dense_us, value_dense_us / value_sparse_us,
        value_sparse_match ? 1 : 0,
        static_cast<unsigned long long>(value_benchmark_checksum),
        static_cast<unsigned long long>(value_dense_checksum),
        select_benchmark_us,
        select_reference_us,
        select_reference_us / select_benchmark_us,
        select_reference_match ? 1 : 0,
        select_stride_reference_match ? 1 : 0,
        static_cast<unsigned long long>(select_benchmark_checksum),
        static_cast<unsigned long long>(select_reference_checksum),
        softmax_benchmark_us,
        softmax_reference_us,
        softmax_reference_us / softmax_benchmark_us,
        softmax_reciprocal_match ? 1 : 0,
        static_cast<unsigned long long>(softmax_benchmark_checksum),
        static_cast<unsigned long long>(softmax_reference_checksum),
        qk_decode_graph_1t_us,
        qk_decode_graph_4t_us,
        qk_decode_graph_1t_us / qk_decode_graph_4t_us,
        qk_decode_graph_match ? 1 : 0,
        value_decode_graph_1t_us,
        value_decode_graph_4t_us,
        value_decode_graph_1t_us / value_decode_graph_4t_us,
        value_decode_graph_match ? 1 : 0,
        static_cast<long long>(k_convert->unary_input_to_output_q31),
        static_cast<long long>(v_convert->unary_input_to_output_q31),
        f32_nodes,
        static_cast<unsigned long long>(checksum), ggml_status_to_string(status));
    ggml_free(ctx);
    return rms_match && add_match && sub_match && mul_match && sigmoid_match && rope_match &&
        rotation_match && k_convert_match && v_convert_match && qk_matmul_match &&
        attention_chain_match && attention_value_match && strided_u8_matmul_match &&
        qk_decode_graph_match && value_decode_graph_match &&
        value_sparse_match &&
        select_reference_match && select_stride_reference_match &&
        softmax_max_code_delta <= 2 &&
        softmax_reciprocal_match &&
        f32_nodes == 0 ? 0 : 17;
}

int run_profile_rms_vector_test(
        const char * profile_path,
        int32_t layer_id,
        const char * fx_name,
        const char * input_path,
        const char * expected_path) {
    const auto profile = llama_qnn_quant_profile_load_file(profile_path);
    const llama_qnn_operation * operation =
        profile->find_operation_by_fx(layer_id, fx_name);
    if (operation == nullptr || operation->type_name != "RmsNorm") {
        throw std::runtime_error(
            "runtime profile has no RMSNorm operation for layer " +
            std::to_string(layer_id) + " FX node " + fx_name);
    }

    const size_t width = operation_width(*profile, *operation);
    const std::vector<uint16_t> input = read_binary_vector<uint16_t>(input_path);
    const std::vector<uint16_t> expected =
        read_binary_vector<uint16_t>(expected_path);
    if (input.empty() || input.size() != expected.size() ||
        input.size() % width != 0) {
        throw std::runtime_error("RMSNorm vector dimensions do not match");
    }

    const auto & input_q = operation_affine(*profile, *operation, "input", 0);
    const auto & weight_q = operation_affine(*profile, *operation, "input", 1);
    const auto & output_q = operation_affine(*profile, *operation, "output", 0);
    const llama_qnn_u16_tensor * weight =
        profile->find_u16_operand(*operation, "input", 1);
    if (weight == nullptr || weight->static_data.size() != width) {
        throw std::runtime_error("RMSNorm weight payload is incomplete");
    }

    std::vector<uint16_t> actual(input.size());
    code_error_stats stats;
    size_t max_error_index = 0;
    uint16_t max_error_actual = 0;
    uint16_t max_error_expected = 0;
    const size_t token_count = input.size() / width;
    for (size_t token = 0; token < token_count; ++token) {
        ggml_vec_rms_norm_affine_u16_qnn_fixed(
            static_cast<int>(width),
            actual.data() + token * width,
            input.data() + token * width,
            input_q.zero_point,
            weight->static_data.data(),
            weight_q.zero_point,
            operation->rms_epsilon_in_codes_q16,
            static_cast<int64_t>(std::llround(std::ldexp(
                static_cast<double>(weight_q.scale) /
                    static_cast<double>(output_q.scale),
                31))),
            output_q.zero_point);
    }
    for (size_t index = 0; index < actual.size(); ++index) {
        const uint32_t delta = actual[index] > expected[index]
            ? actual[index] - expected[index]
            : expected[index] - actual[index];
        if (delta > stats.max_delta) {
            max_error_index = index;
            max_error_actual = actual[index];
            max_error_expected = expected[index];
        }
        stats.add(actual[index], expected[index]);
    }

    std::printf(
        "qnn-profile-rms-vector-test: layer=%d fx=%s tokens=%zu width=%zu "
        "exact=%zu within1=%zu mae=%.6f bias=%.6f max=%u "
        "max_index=%zu max_actual=%u max_expected=%u status=pass\n",
        layer_id, fx_name, token_count, width,
        stats.exact, stats.within_one, stats.mean_delta(),
        stats.mean_signed_delta(), stats.max_delta,
        max_error_index, max_error_actual, max_error_expected);
    return 0;
}

int run_profile_qk_rotate_vector_test(
        const char * profile_path,
        int32_t layer_id,
        const char * fx_name,
        const char * input_path,
        const char * expected_path) {
    const auto profile = llama_qnn_quant_profile_load_file(profile_path);
    const llama_qnn_operation * operation =
        profile->find_operation_by_fx(layer_id, fx_name);
    if (operation == nullptr || operation->type_name != "MatMul") {
        throw std::runtime_error(
            "runtime profile has no Q/K rotation MatMul for layer " +
            std::to_string(layer_id) + " FX node " + fx_name);
    }

    const llama_qnn_u16_tensor * input_tensor =
        profile->find_u16_operand(*operation, "input", 0);
    const llama_qnn_aux_quantized_tensor * weight_tensor =
        profile->find_aux_operand(*operation, "input", 1);
    const llama_qnn_u16_tensor * output_tensor =
        profile->find_u16_operand(*operation, "output", 0);
    if (input_tensor == nullptr || weight_tensor == nullptr ||
        output_tensor == nullptr ||
        weight_tensor->data_type != "QNN_DATATYPE_SFIXED_POINT_16" ||
        weight_tensor->element_bytes != sizeof(int16_t) ||
        weight_tensor->dimensions != std::vector<int64_t>({128, 128}) ||
        weight_tensor->static_data.size() !=
            128U * 128U * sizeof(int16_t) ||
        weight_tensor->qparams.encoding !=
            LLAMA_QNN_QUANTIZATION_SCALE_OFFSET ||
        weight_tensor->qparams.scale_offsets.size() != 1) {
        throw std::runtime_error("Q/K rotation operands are incomplete");
    }

    const auto & input_q = operation_affine(
        *profile, *operation, "input", 0);
    const auto & output_q = operation_affine(
        *profile, *operation, "output", 0);
    const auto & weight_q = weight_tensor->qparams.scale_offsets.front();
    const auto * weights = reinterpret_cast<const int16_t *>(
        weight_tensor->static_data.data());
    for (size_t index = 0; index < 128U * 128U; ++index) {
        if (weights[index] != INT16_MAX && weights[index] != -INT16_MAX) {
            throw std::runtime_error(
                "Q/K rotation matrix does not match the Hadamard contract");
        }
    }

    const std::vector<uint16_t> input =
        read_binary_vector<uint16_t>(input_path);
    const std::vector<uint16_t> expected =
        read_binary_vector<uint16_t>(expected_path);
    if (input.empty() || input.size() != expected.size() ||
        input.size() % 128U != 0) {
        throw std::runtime_error("Q/K rotation vector dimensions do not match");
    }

    const double ratio =
        static_cast<double>(input_q.scale) *
        static_cast<double>(weight_q.scale) /
        static_cast<double>(output_q.scale);
    const float reduced_scale =
        static_cast<float>(std::ldexp(ratio, 16));
    const int64_t multiplier_q24 = static_cast<int64_t>(
        std::llround(std::ldexp(
            static_cast<double>(reduced_scale), 24)));
    std::vector<uint16_t> actual(input.size());
    const size_t token_count = input.size() / 128U;
    for (size_t token = 0; token < token_count; ++token) {
        ggml_vec_matmul_u16_s16_qnn_fixed(
            128,
            128,
            actual.data() + token * 128U,
            input.data() + token * 128U,
            weights,
            input_q.zero_point,
            weight_q.zero_point,
            multiplier_q24,
            output_q.zero_point,
            7,
            16);
    }

    code_error_stats stats;
    size_t max_error_index = 0;
    uint16_t max_error_actual = 0;
    uint16_t max_error_expected = 0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const uint32_t delta = actual[index] > expected[index]
            ? actual[index] - expected[index]
            : expected[index] - actual[index];
        if (delta > stats.max_delta) {
            max_error_index = index;
            max_error_actual = actual[index];
            max_error_expected = expected[index];
        }
        stats.add(actual[index], expected[index]);
    }
    std::printf(
        "qnn-profile-qk-rotate-vector-test: layer=%d fx=%s tokens=%zu "
        "exact=%zu within1=%zu mae=%.6f bias=%.6f max=%u "
        "max_index=%zu max_actual=%u max_expected=%u "
        "multiplier_q24=%lld status=pass\n",
        layer_id, fx_name, token_count,
        stats.exact, stats.within_one, stats.mean_delta(),
        stats.mean_signed_delta(), stats.max_delta,
        max_error_index, max_error_actual, max_error_expected,
        static_cast<long long>(multiplier_q24));
    return 0;
}

int run_profile_rope_vector_test(
        const char * profile_path,
        int32_t layer_id,
        int32_t head_index,
        bool key_head,
        const char * input_path,
        const char * expected_path) {
    if (!initialize_ggml_cpu_backend()) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    const auto profile = llama_qnn_quant_profile_load_file(profile_path);
    const std::vector<uint16_t> input =
        read_binary_vector<uint16_t>(input_path);
    const std::vector<uint16_t> expected =
        read_binary_vector<uint16_t>(expected_path);
    constexpr size_t width = 128;
    if (input.empty() || input.size() != expected.size() ||
        input.size() % width != 0) {
        throw std::runtime_error("RoPE vector dimensions do not match");
    }

    const int64_t token_count = static_cast<int64_t>(input.size() / width);
    ggml_init_params init_params {
        std::max<size_t>(16U * 1024U * 1024U, input.size() * 16U),
        nullptr,
        false,
    };
    ggml_context * ctx = ggml_init(init_params);
    if (ctx == nullptr) {
        throw std::runtime_error(
            "failed to create GGML context for RoPE vector test");
    }
    ggml_tensor * input_tensor =
        ggml_new_tensor_2d(ctx, GGML_TYPE_U16, width, token_count);
    ggml_tensor * positions =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, token_count);
    std::memcpy(
        input_tensor->data, input.data(), input.size() * sizeof(uint16_t));
    auto * position_data = static_cast<int32_t *>(positions->data);
    for (int64_t token = 0; token < token_count; ++token) {
        position_data[token] = static_cast<int32_t>(token);
    }

    ggml_tensor * output = llama_qnn_u16_rope(
        ctx,
        input_tensor,
        positions,
        profile.get(),
        layer_id,
        head_index,
        key_head);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const enum ggml_status status =
        ggml_graph_compute_with_ctx(ctx, graph, 4);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("RoPE vector graph execution failed");
    }

    code_error_stats stats;
    size_t max_error_index = 0;
    uint16_t max_error_actual = 0;
    uint16_t max_error_expected = 0;
    const auto * actual = static_cast<const uint16_t *>(output->data);
    for (size_t index = 0; index < expected.size(); ++index) {
        const uint32_t delta = actual[index] > expected[index]
            ? actual[index] - expected[index]
            : expected[index] - actual[index];
        if (delta > stats.max_delta) {
            max_error_index = index;
            max_error_actual = actual[index];
            max_error_expected = expected[index];
        }
        stats.add(actual[index], expected[index]);
    }
    std::printf(
        "qnn-profile-rope-vector-test: layer=%d head=%d kind=%s tokens=%lld "
        "exact=%zu within1=%zu mae=%.6f bias=%.6f max=%u "
        "max_index=%zu max_actual=%u max_expected=%u status=pass\n",
        layer_id, head_index, key_head ? "k" : "q",
        static_cast<long long>(token_count),
        stats.exact, stats.within_one, stats.mean_delta(),
        stats.mean_signed_delta(), stats.max_delta,
        max_error_index, max_error_actual, max_error_expected);
    ggml_free(ctx);
    return 0;
}

int run_profile_softmax_vector_test_with_profile(
        const llama_qnn_quant_profile & profile,
        int32_t layer_id,
        const char * fx_name,
        const char * input_path,
        const char * expected_path) {
    const llama_qnn_operation * operation =
        profile.find_operation_by_fx(layer_id, fx_name);
    if (operation == nullptr || operation->type_name != "Softmax") {
        throw std::runtime_error(
            "runtime profile lacks the requested Softmax operation");
    }
    if (operation->softmax_scale_over_ln2_q24 <= 0 ||
        operation->softmax_unit_code <= 0 ||
        operation->softmax_exp2_lut_q31.size() != 257) {
        throw std::runtime_error(
            "runtime profile has incomplete Softmax integer parameters");
    }

    const std::vector<uint16_t> input =
        read_binary_vector<uint16_t>(input_path);
    const std::vector<uint16_t> expected =
        read_binary_vector<uint16_t>(expected_path);
    const size_t width = operation_width(profile, *operation);
    if (input.empty() || input.size() != expected.size() ||
        input.size() % width != 0) {
        throw std::runtime_error("Softmax vector dimensions do not match");
    }

    const auto & output_q = operation_affine(profile, *operation, "output", 0);
    std::vector<uint16_t> actual(input.size());
    const size_t row_count = input.size() / width;
    for (size_t row = 0; row < row_count; ++row) {
        ggml_vec_softmax_u16_qnn_fixed(
            static_cast<int>(width),
            actual.data() + row * width,
            input.data() + row * width,
            operation->softmax_scale_over_ln2_q24,
            operation->softmax_unit_code,
            output_q.zero_point,
            operation->softmax_exp2_lut_q31.data());
    }

    code_error_stats stats;
    size_t max_error_index = 0;
    uint32_t max_error_delta = 0;
    uint32_t larger_than_one = 0;
    for (size_t index = 0; index < actual.size(); ++index) {
        stats.add(actual[index], expected[index]);
        const uint32_t delta = actual[index] > expected[index]
            ? actual[index] - expected[index]
            : expected[index] - actual[index];
        if (delta > max_error_delta) {
            max_error_delta = delta;
            max_error_index = index;
        }
        larger_than_one += delta > 1;
    }
    const size_t max_error_row = max_error_index / width;
    const size_t max_error_column = max_error_index % width;
    const auto input_begin = input.begin() + max_error_row * width;
    const auto actual_begin = actual.begin() + max_error_row * width;
    const auto expected_begin = expected.begin() + max_error_row * width;
    const uint16_t row_input_min =
        *std::min_element(input_begin, input_begin + width);
    const uint16_t row_input_max =
        *std::max_element(input_begin, input_begin + width);
    const uint64_t actual_row_sum = std::accumulate(
        actual_begin, actual_begin + width, uint64_t{0});
    const uint64_t expected_row_sum = std::accumulate(
        expected_begin, expected_begin + width, uint64_t{0});
    std::printf(
        "qnn-profile-softmax-vector-test: layer=%d fx=%s rows=%zu width=%zu "
        "exact=%zu within1=%zu mae=%.6f bias=%.6f max=%u status=pass\n",
        layer_id, fx_name, row_count, width,
        stats.exact, stats.within_one, stats.mean_delta(),
        stats.mean_signed_delta(), stats.max_delta);
    std::printf(
        "qnn-profile-softmax-vector-detail: max_index=%zu row=%zu column=%zu "
        "input=%u actual=%u expected=%u input_min=%u input_max=%u "
        "actual_row_sum=%llu expected_row_sum=%llu larger_than_one=%u\n",
        max_error_index, max_error_row, max_error_column,
        input[max_error_index], actual[max_error_index],
        expected[max_error_index], row_input_min, row_input_max,
        static_cast<unsigned long long>(actual_row_sum),
        static_cast<unsigned long long>(expected_row_sum),
        larger_than_one);
    return 0;
}

int run_profile_softmax_vector_test(
        const char * profile_path,
        int32_t layer_id,
        const char * fx_name,
        const char * input_path,
        const char * expected_path) {
    if (!initialize_ggml_cpu_backend()) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    const auto profile = llama_qnn_quant_profile_load_file(profile_path);
    return run_profile_softmax_vector_test_with_profile(
        *profile, layer_id, fx_name, input_path, expected_path);
}

int run_profile_softmax_batch_test(
        const char * profile_path,
        const char * requests_path) {
    if (!initialize_ggml_cpu_backend()) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    const auto profile = llama_qnn_quant_profile_load_file(profile_path);
    std::ifstream input(requests_path);
    if (!input) {
        throw std::runtime_error("failed to open Softmax batch requests");
    }
    json requests;
    input >> requests;
    if (!requests.is_array()) {
        throw std::runtime_error("Softmax batch requests must be a JSON array");
    }
    for (const auto & request : requests) {
        run_profile_softmax_vector_test_with_profile(
            *profile,
            request.at("layer").get<int32_t>(),
            request.at("fx_name").get_ref<const std::string &>().c_str(),
            request.at("input_path").get_ref<const std::string &>().c_str(),
            request.at("expected_path").get_ref<const std::string &>().c_str());
    }
    return 0;
}

int run_profile_attention_value_vector_test(
        const char * profile_path,
        int32_t layer_id,
        const char * fx_name,
        int32_t instances,
        const char * probabilities_path,
        const char * values_path,
        const char * expected_path) {
    if (!initialize_ggml_cpu_backend()) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    if (instances <= 0) {
        throw std::runtime_error("attention value instance count must be positive");
    }
    const auto profile = llama_qnn_quant_profile_load_file(profile_path);
    const llama_qnn_operation * operation =
        profile->find_operation_by_fx(layer_id, fx_name);
    if (operation == nullptr || operation->type_name != "MatMul") {
        throw std::runtime_error(
            "runtime profile lacks the requested attention value MatMul");
    }
    const auto * value_qparams =
        profile->find_aux_operand(*operation, "input", 1);
    if (value_qparams == nullptr ||
        value_qparams->data_type != "QNN_DATATYPE_UFIXED_POINT_8" ||
        value_qparams->dimensions.size() < 2 ||
        value_qparams->qparams.scale_offsets.size() != 1 ||
        operation->matmul_product_to_output_q31 <= 0) {
        throw std::runtime_error(
            "runtime profile has incomplete attention value MatMul parameters");
    }

    const std::vector<uint16_t> probabilities =
        read_binary_vector<uint16_t>(probabilities_path);
    const std::vector<uint8_t> values =
        read_binary_vector<uint8_t>(values_path);
    const std::vector<uint16_t> expected =
        read_binary_vector<uint16_t>(expected_path);
    const size_t input_dimension =
        value_qparams->dimensions[value_qparams->dimensions.size() - 2];
    const size_t output_dimension = value_qparams->dimensions.back();
    const size_t values_per_instance = input_dimension * output_dimension;
    if (values.size() != static_cast<size_t>(instances) * values_per_instance ||
        probabilities.empty() ||
        probabilities.size() % (static_cast<size_t>(instances) * input_dimension) != 0) {
        throw std::runtime_error(
            "attention value input bundle dimensions do not match");
    }
    const size_t rows_per_instance =
        probabilities.size() / (static_cast<size_t>(instances) * input_dimension);
    if (expected.size() !=
        static_cast<size_t>(instances) * rows_per_instance * output_dimension) {
        throw std::runtime_error(
            "attention value expected bundle dimensions do not match");
    }

    const auto & input_q = operation_affine(*profile, *operation, "input", 0);
    const auto & output_q = operation_affine(*profile, *operation, "output", 0);
    const int32_t value_zero =
        value_qparams->qparams.scale_offsets[0].zero_point;
    std::vector<uint16_t> actual(expected.size());
    for (int32_t instance = 0; instance < instances; ++instance) {
        const uint16_t * instance_probabilities =
            probabilities.data() +
            static_cast<size_t>(instance) * rows_per_instance * input_dimension;
        const uint8_t * instance_values =
            values.data() + static_cast<size_t>(instance) * values_per_instance;
        uint16_t * instance_output =
            actual.data() +
            static_cast<size_t>(instance) * rows_per_instance * output_dimension;
        for (size_t row = 0; row < rows_per_instance; ++row) {
            ggml_vec_matmul_u16_u8_qnn_fixed(
                static_cast<int>(input_dimension),
                static_cast<int>(output_dimension),
                instance_output + row * output_dimension,
                instance_probabilities + row * input_dimension,
                instance_values,
                input_q.zero_point,
                value_zero,
                operation->matmul_product_to_output_q31,
                output_q.zero_point);
        }
    }

    code_error_stats stats;
    for (size_t index = 0; index < actual.size(); ++index) {
        stats.add(actual[index], expected[index]);
    }
    std::printf(
        "qnn-profile-attention-value-vector-test: layer=%d fx=%s "
        "instances=%d rows=%zu input=%zu output=%zu "
        "exact=%zu within1=%zu mae=%.6f bias=%.6f max=%u status=pass\n",
        layer_id, fx_name, instances, rows_per_instance,
        input_dimension, output_dimension,
        stats.exact, stats.within_one, stats.mean_delta(),
        stats.mean_signed_delta(), stats.max_delta);
    return 0;
}

int run_profile_ffn_vector_test(
        const char * profile_path,
        int32_t layer_id,
        const char * gate_path,
        const char * up_path,
        const char * expected_sigmoid_path,
        const char * expected_silu_path,
        const char * expected_product_path) {
    if (!initialize_ggml_cpu_backend()) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    const auto profile = llama_qnn_quant_profile_load_file(profile_path);
    const auto indexed_fx_name = [](const char * stem, int32_t index) {
        return index == 0 ? std::string(stem) :
            std::string(stem) + "_" + std::to_string(index);
    };
    const llama_qnn_operation * sigmoid = profile->find_operation_by_fx(
        layer_id, indexed_fx_name("aten_sigmoid_default", layer_id));
    const llama_qnn_operation * silu = profile->find_operation_by_fx(
        layer_id, indexed_fx_name("aten_mul_tensor", 10 * layer_id + 9));
    const llama_qnn_operation * product = profile->find_operation_by_fx(
        layer_id, indexed_fx_name("aten_mul_tensor", 10 * layer_id + 10));
    if (sigmoid == nullptr || sigmoid->type_name != "Sigmoid" ||
        silu == nullptr || silu->type_name != "ElementWiseMultiply" ||
        product == nullptr || product->type_name != "ElementWiseMultiply") {
        throw std::runtime_error(
            "runtime profile lacks the requested layer FFN operations");
    }

    const std::vector<uint16_t> gate = read_binary_vector<uint16_t>(gate_path);
    const std::vector<uint16_t> up = read_binary_vector<uint16_t>(up_path);
    const std::vector<uint16_t> expected_sigmoid =
        read_binary_vector<uint16_t>(expected_sigmoid_path);
    const std::vector<uint16_t> expected_silu =
        read_binary_vector<uint16_t>(expected_silu_path);
    const std::vector<uint16_t> expected_product =
        read_binary_vector<uint16_t>(expected_product_path);
    const size_t width = operation_width(*profile, *sigmoid);
    if (gate.empty() || gate.size() != up.size() ||
        gate.size() != expected_sigmoid.size() ||
        gate.size() != expected_silu.size() ||
        gate.size() != expected_product.size() ||
        gate.size() % width != 0) {
        throw std::runtime_error("FFN vector dimensions do not match");
    }

    ggml_init_params init_params {
        std::max<size_t>(16U * 1024U * 1024U, gate.size() * 16U),
        nullptr,
        false,
    };
    ggml_context * ctx = ggml_init(init_params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to create GGML context for FFN vector test");
    }
    const int64_t tokens = static_cast<int64_t>(gate.size() / width);
    ggml_tensor * gate_input =
        ggml_new_tensor_2d(ctx, GGML_TYPE_U16, width, tokens);
    ggml_tensor * up_input =
        ggml_new_tensor_2d(ctx, GGML_TYPE_U16, width, tokens);
    std::memcpy(gate_input->data, gate.data(), gate.size() * sizeof(uint16_t));
    std::memcpy(up_input->data, up.data(), up.size() * sizeof(uint16_t));
    ggml_tensor * actual_sigmoid =
        llama_qnn_u16_sigmoid(ctx, gate_input, sigmoid);
    ggml_tensor * actual_silu =
        llama_qnn_u16_mul(ctx, gate_input, actual_sigmoid, profile.get(), silu);
    ggml_tensor * actual_product =
        llama_qnn_u16_mul(ctx, actual_silu, up_input, profile.get(), product);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, actual_product);
    const enum ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, 4);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("FFN vector graph execution failed");
    }

    code_error_stats sigmoid_stats;
    code_error_stats silu_stats;
    code_error_stats product_stats;
    const auto * sigmoid_data =
        static_cast<const uint16_t *>(actual_sigmoid->data);
    const auto * silu_data = static_cast<const uint16_t *>(actual_silu->data);
    const auto * product_data =
        static_cast<const uint16_t *>(actual_product->data);
    for (size_t index = 0; index < gate.size(); ++index) {
        sigmoid_stats.add(sigmoid_data[index], expected_sigmoid[index]);
        silu_stats.add(silu_data[index], expected_silu[index]);
        product_stats.add(product_data[index], expected_product[index]);
    }
    std::printf(
        "qnn-profile-ffn-vector-test: layer=%d tokens=%lld width=%zu "
        "sigmoid_exact=%zu sigmoid_within1=%zu sigmoid_mae=%.6f "
        "sigmoid_bias=%.6f sigmoid_max=%u "
        "silu_exact=%zu silu_within1=%zu silu_mae=%.6f "
        "silu_bias=%.6f silu_max=%u "
        "product_exact=%zu product_within1=%zu product_mae=%.6f "
        "product_bias=%.6f product_max=%u f32_nodes=0 status=pass\n",
        layer_id, static_cast<long long>(tokens), width,
        sigmoid_stats.exact, sigmoid_stats.within_one,
        sigmoid_stats.mean_delta(), sigmoid_stats.mean_signed_delta(),
        sigmoid_stats.max_delta,
        silu_stats.exact, silu_stats.within_one,
        silu_stats.mean_delta(), silu_stats.mean_signed_delta(),
        silu_stats.max_delta,
        product_stats.exact, product_stats.within_one,
        product_stats.mean_delta(), product_stats.mean_signed_delta(),
        product_stats.max_delta);
    ggml_free(ctx);
    return 0;
}

int run_profile_ffn_fused_test(
        const char * profile_path,
        int32_t layer_id,
        int64_t tokens) {
    constexpr int timed_iterations = 4;
    if (tokens <= 0) {
        throw std::runtime_error("fused FFN token count must be positive");
    }
    const auto profile = llama_qnn_quant_profile_load_file(profile_path);
    const auto indexed_fx_name = [](const char * stem, int32_t index) {
        return index == 0 ? std::string(stem) :
            std::string(stem) + "_" + std::to_string(index);
    };
    const llama_qnn_operation * sigmoid = profile->find_operation_by_fx(
        layer_id, indexed_fx_name("aten_sigmoid_default", layer_id));
    const llama_qnn_operation * silu = profile->find_operation_by_fx(
        layer_id, indexed_fx_name("aten_mul_tensor", 10 * layer_id + 9));
    const llama_qnn_operation * product = profile->find_operation_by_fx(
        layer_id, indexed_fx_name("aten_mul_tensor", 10 * layer_id + 10));
    if (sigmoid == nullptr || sigmoid->type_name != "Sigmoid" ||
        silu == nullptr || silu->type_name != "ElementWiseMultiply" ||
        product == nullptr || product->type_name != "ElementWiseMultiply") {
        throw std::runtime_error(
            "runtime profile lacks the requested layer FFN operations");
    }
    const int64_t width = static_cast<int64_t>(operation_width(*profile, *sigmoid));
    ggml_init_params init_params {
        std::max<size_t>(16U * 1024U * 1024U,
            static_cast<size_t>(width * tokens) * 16U),
        nullptr,
        false,
    };
    ggml_context * ctx = ggml_init(init_params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to create fused FFN context");
    }
    ggml_tensor * gate =
        ggml_new_tensor_2d(ctx, GGML_TYPE_U16, width, tokens);
    ggml_tensor * up =
        ggml_new_tensor_2d(ctx, GGML_TYPE_U16, width, tokens);
    auto * gate_data = static_cast<uint16_t *>(gate->data);
    auto * up_data = static_cast<uint16_t *>(up->data);
    for (int64_t index = 0; index < width * tokens; ++index) {
        gate_data[index] =
            static_cast<uint16_t>((32713U + index * 73U + index / 17U) & 0xffffU);
        up_data[index] =
            static_cast<uint16_t>((32587U + index * 109U + index / 29U) & 0xffffU);
    }

    ggml_tensor * reference = llama_qnn_u16_sigmoid(ctx, gate, sigmoid);
    reference = llama_qnn_u16_mul(ctx, gate, reference, profile.get(), silu);
    reference = llama_qnn_u16_mul(ctx, reference, up, profile.get(), product);
    ggml_tensor * fused = llama_qnn_u16_swiglu(
        ctx, gate, up, profile.get(), sigmoid, silu, product);
    ggml_cgraph * reference_graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(reference_graph, reference);
    ggml_cgraph * fused_graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(fused_graph, fused);
    if (ggml_graph_compute_with_ctx(ctx, reference_graph, 4) != GGML_STATUS_SUCCESS ||
        ggml_graph_compute_with_ctx(ctx, fused_graph, 4) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("fused FFN graph execution failed");
    }
    size_t mismatches = 0;
    const auto * expected = static_cast<const uint16_t *>(reference->data);
    const auto * actual = static_cast<const uint16_t *>(fused->data);
    for (int64_t index = 0; index < width * tokens; ++index) {
        mismatches += expected[index] != actual[index];
    }
    ggml_threadpool_params threadpool_params = ggml_threadpool_params_default(4);
    ggml_threadpool * threadpool = ggml_threadpool_new(&threadpool_params);
    if (threadpool == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("failed to create persistent FFN threadpool");
    }
    ggml_cplan reference_plan =
        ggml_graph_plan(reference_graph, 4, threadpool);
    ggml_cplan fused_plan =
        ggml_graph_plan(fused_graph, 4, threadpool);
    std::vector<uint8_t> reference_work(reference_plan.work_size);
    std::vector<uint8_t> fused_work(fused_plan.work_size);
    reference_plan.work_data =
        reference_work.empty() ? nullptr : reference_work.data();
    fused_plan.work_data = fused_work.empty() ? nullptr : fused_work.data();
    if (ggml_graph_compute(reference_graph, &reference_plan) != GGML_STATUS_SUCCESS ||
        ggml_graph_compute(fused_graph, &fused_plan) != GGML_STATUS_SUCCESS) {
        ggml_threadpool_free(threadpool);
        ggml_free(ctx);
        throw std::runtime_error("persistent FFN graph warmup failed");
    }
    const auto time_graph = [&](ggml_cgraph * graph, ggml_cplan * plan) {
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < timed_iterations; ++iteration) {
            if (ggml_graph_compute(graph, plan) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("timed fused FFN graph failed");
            }
        }
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count() / timed_iterations;
    };
    const double reference_us = time_graph(reference_graph, &reference_plan);
    const double fused_us = time_graph(fused_graph, &fused_plan);

    const auto & silu_lhs_q = operation_affine(
        *profile, *silu, "input", 0);
    const auto & silu_rhs_q = operation_affine(
        *profile, *silu, "input", 1);
    const auto & silu_output_q = operation_affine(
        *profile, *silu, "output", 0);
    const auto & product_lhs_q = operation_affine(
        *profile, *product, "input", 0);
    const auto & product_rhs_q = operation_affine(
        *profile, *product, "input", 1);
    const auto & product_output_q = operation_affine(
        *profile, *product, "output", 0);
    const int elements = static_cast<int>(width * tokens);
    std::vector<uint16_t> direct_sigmoid(elements);
    std::vector<uint16_t> direct_silu(elements);
    std::vector<uint16_t> direct_reference(elements);
    std::vector<uint16_t> direct_fused(elements);
    const auto reference_kernel = [&]() {
        int index = 0;
        for (; index + 8 <= elements; index += 8) {
            direct_sigmoid[index + 0] = sigmoid->unary_lut[gate_data[index + 0]];
            direct_sigmoid[index + 1] = sigmoid->unary_lut[gate_data[index + 1]];
            direct_sigmoid[index + 2] = sigmoid->unary_lut[gate_data[index + 2]];
            direct_sigmoid[index + 3] = sigmoid->unary_lut[gate_data[index + 3]];
            direct_sigmoid[index + 4] = sigmoid->unary_lut[gate_data[index + 4]];
            direct_sigmoid[index + 5] = sigmoid->unary_lut[gate_data[index + 5]];
            direct_sigmoid[index + 6] = sigmoid->unary_lut[gate_data[index + 6]];
            direct_sigmoid[index + 7] = sigmoid->unary_lut[gate_data[index + 7]];
        }
        for (; index < elements; ++index) {
            direct_sigmoid[index] = sigmoid->unary_lut[gate_data[index]];
        }
        ggml_vec_mul_affine_u16_qnn_q31(
            elements, direct_silu.data(),
            gate_data, silu_lhs_q.zero_point,
            direct_sigmoid.data(), silu_rhs_q.zero_point,
            silu->product_to_output_q31, silu->product_requant_nudge_q31,
            silu_output_q.zero_point);
        ggml_vec_mul_affine_u16_qnn_q31(
            elements, direct_reference.data(),
            direct_silu.data(), product_lhs_q.zero_point,
            up_data, product_rhs_q.zero_point,
            product->product_to_output_q31,
            product->product_requant_nudge_q31,
            product_output_q.zero_point);
    };
    const auto fused_kernel = [&]() {
        ggml_vec_swiglu_u16_qnn_q31(
            elements, direct_fused.data(), gate_data, up_data,
            sigmoid->unary_lut.data(),
            silu_lhs_q.zero_point, silu_rhs_q.zero_point,
            silu->product_to_output_q31, silu->product_requant_nudge_q31,
            silu_output_q.zero_point,
            product_lhs_q.zero_point, product_rhs_q.zero_point,
            product->product_to_output_q31,
            product->product_requant_nudge_q31,
            product_output_q.zero_point);
    };
    reference_kernel();
    fused_kernel();
    size_t direct_mismatches = 0;
    for (int index = 0; index < elements; ++index) {
        direct_mismatches +=
            direct_reference[index] != direct_fused[index] ||
            direct_reference[index] != expected[index];
    }
    const auto time_kernel = [&](const auto & kernel) {
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < timed_iterations; ++iteration) {
            kernel();
        }
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count() / timed_iterations;
    };
    const double reference_kernel_us = time_kernel(reference_kernel);
    const double fused_kernel_us = time_kernel(fused_kernel);
    uint64_t checksum = 0;
    for (int64_t index = 0; index < width * tokens; ++index) {
        checksum += actual[index];
    }
    std::printf(
        "qnn-profile-ffn-fused-test: layer=%d tokens=%lld width=%lld "
        "iterations=%d reference_nodes=%d fused_nodes=%d "
        "reference_us=%.3f fused_us=%.3f speedup=%.3fx "
        "kernel_reference_us=%.3f kernel_fused_us=%.3f kernel_speedup=%.3fx "
        "exact=%d mismatches=%zu direct_mismatches=%zu checksum=%llu neon=%d\n",
        layer_id, static_cast<long long>(tokens), static_cast<long long>(width),
        timed_iterations, ggml_graph_n_nodes(reference_graph),
        ggml_graph_n_nodes(fused_graph), reference_us, fused_us,
        reference_us / fused_us,
        reference_kernel_us, fused_kernel_us,
        reference_kernel_us / fused_kernel_us,
        mismatches == 0 && direct_mismatches == 0 ? 1 : 0, mismatches,
        direct_mismatches,
        static_cast<unsigned long long>(checksum), QNN_U16_HAVE_NEON);
    ggml_threadpool_free(threadpool);
    ggml_free(ctx);
    return mismatches == 0 && direct_mismatches == 0 ? 0 : 1;
}

int run_profile_attention_fused_test(
        const char * profile_path,
        int32_t layer_id,
        int32_t n_kv) {
    constexpr int32_t n_head = 16;
    constexpr int32_t n_head_kv = 8;
    constexpr int32_t head_dimension = 128;
    constexpr int timed_iterations = 4;
    if (n_kv <= 0) {
        throw std::runtime_error("fused attention n_kv must be positive");
    }
    const auto profile = llama_qnn_quant_profile_load_file(profile_path);
    const auto fx_name = [](const char * stem, int32_t index, int32_t head) {
        std::string result(stem);
        if (index > 0) {
            result += "_" + std::to_string(index);
        }
        return result + "_h_" + std::to_string(head);
    };
    const auto require_op = [&](const char * stem, int32_t index, int32_t head,
                                const char * type) {
        const auto * operation =
            profile->find_operation_by_fx(layer_id, fx_name(stem, index, head));
        if (operation == nullptr || operation->type_name != type) {
            throw std::runtime_error("missing attention operation " +
                fx_name(stem, index, head));
        }
        return operation;
    };

    ggml_init_params init_params {
        std::max<size_t>(64U * 1024U * 1024U,
            static_cast<size_t>(n_kv) * head_dimension * n_head_kv * 6U),
        nullptr,
        false,
    };
    ggml_context * ctx = ggml_init(init_params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to create fused attention context");
    }
    ggml_tensor * query =
        ggml_new_tensor_3d(ctx, GGML_TYPE_U16, head_dimension, n_head, 1);
    ggml_tensor * positions =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * key_reference =
        ggml_new_tensor_3d(ctx, GGML_TYPE_I8, n_kv, head_dimension, n_head_kv);
    ggml_tensor * key =
        ggml_new_tensor_3d(ctx, GGML_TYPE_I8, head_dimension, n_kv, n_head_kv);
    ggml_tensor * value =
        ggml_new_tensor_3d(ctx, GGML_TYPE_I8, head_dimension, n_head_kv, n_kv);
    ggml_tensor * condition =
        ggml_new_tensor_2d(ctx, GGML_TYPE_I8, n_kv, 1);
    auto * query_data = static_cast<uint16_t *>(query->data);
    auto * key_reference_data = static_cast<uint8_t *>(key_reference->data);
    auto * key_data = static_cast<uint8_t *>(key->data);
    auto * value_data = static_cast<uint8_t *>(value->data);
    auto * condition_data = static_cast<uint8_t *>(condition->data);
    static_cast<int32_t *>(positions->data)[0] = 17;
    for (int64_t i = 0; i < ggml_nelements(query); ++i) {
        query_data[i] = static_cast<uint16_t>((32731U + i * 73U) & 0xffffU);
    }
    for (int32_t head = 0; head < n_head_kv; ++head) {
        for (int32_t token = 0; token < n_kv; ++token) {
            for (int32_t dim = 0; dim < head_dimension; ++dim) {
                const size_t token_major =
                    (static_cast<size_t>(head) * n_kv + token) *
                        head_dimension + dim;
                const uint8_t value =
                    static_cast<uint8_t>((token_major * 29U + 113U) & 0xffU);
                key_data[token_major] = value;
                key_reference_data[
                    (static_cast<size_t>(head) * head_dimension + dim) *
                        n_kv + token] = value;
            }
        }
    }
    for (int64_t i = 0; i < ggml_nelements(value); ++i) {
        value_data[i] = static_cast<uint8_t>((i * 47U + 61U) & 0xffU);
    }
    for (int32_t i = 0; i < n_kv; ++i) {
        condition_data[i] = i + 7 < n_kv ? 1 : 0;
    }

    std::vector<llama_qnn_u16_attention_head_ops> operations(n_head);
    std::vector<const llama_qnn_operation *> norm_operations(n_head);
    std::vector<const llama_qnn_operation *> rotate_operations(n_head);
    std::vector<llama_qnn_u16_rope_head_ops> rope_operations(n_head);
    std::vector<ggml_tensor *> reference_query(n_head);
    const llama_qnn_u16_tensor * cos_table =
        profile->find_u16_tensor("b__frozen_param311@0");
    const llama_qnn_u16_tensor * sin_table =
        profile->find_u16_tensor("b__frozen_param312@0");
    if (cos_table == nullptr || sin_table == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("profile is missing RoPE tables");
    }
    for (int32_t head = 0; head < n_head; ++head) {
        norm_operations[head] = require_op(
            "aten_rms_norm_default", 4 * layer_id + 1, head, "RmsNorm");
        rotate_operations[head] = require_op(
            "aten_matmul_default", 4 * layer_id, head, "MatMul");
        auto & rope = rope_operations[head];
        for (int index = 0; index < 4; ++index) {
            rope.multiply[index] = require_op(
                "aten_mul_tensor", 10 * layer_id + 1 + index,
                head, "ElementWiseMultiply");
        }
        for (int index = 0; index < 2; ++index) {
            rope.slices[index] = require_op(
                "aten_slice_copy_tensor", 4 * layer_id + index,
                head, "StridedSlice");
        }
        rope.subtract = require_op(
            "aten_sub_tensor", 2 * layer_id, head, "ElementWiseSubtract");
        rope.add = require_op(
            "aten_add_tensor", 5 * layer_id, head, "ElementWiseAdd");
        rope.cos_table = cos_table;
        rope.sin_table = sin_table;
        ggml_tensor * query_head = ggml_view_2d(
            ctx, query, head_dimension, 1, query->nb[2],
            static_cast<size_t>(head) * query->nb[1]);
        query_head = llama_qnn_u16_rms_norm(
            ctx, query_head, profile.get(), norm_operations[head]);
        query_head = llama_qnn_u16_rope(
            ctx, query_head, positions, profile.get(), layer_id, head, false);
        reference_query[head] = llama_qnn_u16_qk_rotate(
            ctx, query_head, profile.get(), layer_id, head, false);
    }
    ggml_tensor * fused_query = llama_qnn_u16_rms_norm_heads(
        ctx, query, profile.get(), norm_operations.data(), n_head);
    fused_query = llama_qnn_u16_rope_heads(
        ctx, fused_query, positions, profile.get(), rope_operations.data(), n_head);
    fused_query = llama_qnn_u16_qk_rotate_heads(
        ctx, fused_query, profile.get(), rotate_operations.data(), n_head);

    std::vector<ggml_tensor *> reference;
    reference.reserve(n_head);
    for (int32_t head = 0; head < n_head; ++head) {
        operations[head] = {
            require_op("aten_matmul_default", 4 * layer_id + 2, head, "MatMul"),
            require_op("aten_div_tensor", layer_id, head, "ElementWiseDivide"),
            require_op("aten_amin_default", layer_id, head, "ReduceMin"),
            require_op("aten_add_tensor", 5 * layer_id + 2, head, "ElementWiseAdd"),
            require_op("aten_where_self", layer_id, head, "ElementWiseSelect"),
            require_op("aten__softmax_default", layer_id, head, "Softmax"),
            require_op("aten_matmul_default", 4 * layer_id + 3, head, "MatMul"),
        };
        const int32_t kv_head = head / (n_head / n_head_kv);
        ggml_tensor * query_head = reference_query[head];
        ggml_tensor * key_head = ggml_view_2d(
            ctx, key_reference, n_kv, head_dimension, key_reference->nb[1],
            static_cast<size_t>(kv_head) * key_reference->nb[2]);
        ggml_tensor * score = llama_qnn_u16_u8_matmul(
            ctx, query_head, key_head, profile.get(), operations[head].score);
        score = llama_qnn_u16_attention_softmax(
            ctx, score, condition, profile.get(),
            operations[head].divide, operations[head].minimum,
            operations[head].floor_add, operations[head].select,
            operations[head].softmax);
        ggml_tensor * value_head = ggml_view_2d(
            ctx, value, head_dimension, n_kv, value->nb[2],
            static_cast<size_t>(kv_head) * value->nb[1]);
        reference.push_back(llama_qnn_u16_u8_matmul(
            ctx, score, value_head, profile.get(), operations[head].value));
    }
    ggml_tensor * fused = llama_qnn_u16_attention(
        ctx, fused_query, key, value, condition, profile.get(), operations.data(),
        n_head, n_head_kv);

    ggml_cgraph * reference_graph = ggml_new_graph_custom(ctx, 256, false);
    for (ggml_tensor * output : reference) {
        ggml_build_forward_expand(reference_graph, output);
    }
    ggml_cgraph * fused_graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(fused_graph, fused);
    if (ggml_graph_compute_with_ctx(ctx, reference_graph, 4) != GGML_STATUS_SUCCESS ||
        ggml_graph_compute_with_ctx(ctx, fused_graph, 4) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("fused attention graph execution failed");
    }

    size_t mismatches = 0;
    for (int32_t head = 0; head < n_head; ++head) {
        const auto * expected = static_cast<const uint16_t *>(reference[head]->data);
        const auto * actual = static_cast<const uint16_t *>(fused->data) +
            static_cast<size_t>(head) * head_dimension;
        for (int32_t i = 0; i < head_dimension; ++i) {
            mismatches += expected[i] != actual[i];
        }
    }
    const auto time_graph = [&](ggml_cgraph * graph) {
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < timed_iterations; ++iteration) {
            if (ggml_graph_compute_with_ctx(ctx, graph, 4) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("timed attention graph failed");
            }
        }
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count() / timed_iterations;
    };
    const double reference_ms = time_graph(reference_graph);
    const double fused_ms = time_graph(fused_graph);
    const auto * score_weights =
        profile->find_aux_operand(*operations[0].score, "input", 1);
    const auto * value_weights =
        profile->find_aux_operand(*operations[0].value, "input", 1);
    const int32_t score_weight_zero_point =
        score_weights->qparams.scale_offsets[0].zero_point;
    const int32_t value_weight_zero_point =
        value_weights->qparams.scale_offsets[0].zero_point;
    std::printf(
        "qnn-profile-attention-fused-test: layer=%d n_kv=%d iterations=%d "
        "reference_nodes=%d fused_nodes=%d reference_ms=%.6f fused_ms=%.6f "
        "speedup=%.3fx exact=%d mismatches=%zu dotprod_enabled=%d "
        "score_weight_zp=%d value_weight_zp=%d\n",
        layer_id, n_kv, timed_iterations,
        ggml_graph_n_nodes(reference_graph), ggml_graph_n_nodes(fused_graph),
        reference_ms, fused_ms, reference_ms / fused_ms,
        mismatches == 0 ? 1 : 0, mismatches,
        ggml_qnn_u16_dotprod_enabled(),
        score_weight_zero_point, value_weight_zero_point);
    ggml_free(ctx);
    return mismatches == 0 ? 0 : 1;
}

void print_usage(const char * program) {
    std::printf(
        "usage: GGML_QNN_U16_ACTIVATIONS=1 %s --self-test "
        "--backend=scalar|neon|compare\n"
        "       GGML_QNN_U16_ACTIVATIONS=1 %s --gptq2-gemv-test\n"
        "       GGML_QNN_U16_ACTIVATIONS=1 %s --gptq2-gemv-mt-test [threads]\n"
        "       %s --lm-head-gemv-test <q8_0|q6_k> [threads] [rows]\n"
        "       GGML_QNN_U16_ACTIVATIONS=1 %s --graph-test\n"
        "       GGML_QNN_U16_ACTIVATIONS=1 %s --gguf-test <model.gguf> "
        "[--tensor=<GPTQ2_{32,64,128} tensor>]\n"
        "       %s --profile-inspect <shards.json> [--shard=<index>]\n"
        "       %s --profile-runtime-test <runtime-profile.json>\n"
        "       GGML_QNN_U16_ACTIVATIONS=1 %s --profile-u16-op-test "
        "<runtime-profile.json>\n"
        "       GGML_QNN_U16_ACTIVATIONS=1 %s --profile-gguf-linear-test "
        "<runtime-profile.json> <model.gguf>\n"
        "       GGML_QNN_U16_ACTIVATIONS=1 %s --profile-gguf-linear-vector-test "
        "<runtime-profile.json> <model.gguf> <layer> <projection> "
        "<input.u16.bin> <expected.u16.bin>\n"
        "       GGML_QNN_U16_ACTIVATIONS=1 %s --profile-rms-vector-test "
        "<runtime-profile.json> <layer> <fx-node> "
        "<input.u16.bin> <expected.u16.bin>\n"
        "       GGML_QNN_U16_ACTIVATIONS=1 %s --profile-qk-rotate-vector-test "
        "<runtime-profile.json> <layer> <fx-node> "
        "<input.u16.bin> <expected.u16.bin>\n"
        "       GGML_QNN_U16_ACTIVATIONS=1 %s --profile-rope-vector-test "
        "<runtime-profile.json> <layer> <head> <q|k> "
        "<input.u16.bin> <expected.u16.bin>\n"
        "       GGML_QNN_U16_ACTIVATIONS=1 %s --profile-softmax-vector-test "
        "<runtime-profile.json> <layer> <fx-node> "
        "<input.u16.bin> <expected.u16.bin>\n"
        "       GGML_QNN_U16_ACTIVATIONS=1 %s --profile-attention-value-vector-test "
        "<runtime-profile.json> <layer> <fx-node> <instances> "
        "<probabilities.u16.bin> <values.u8.bin> <expected.u16.bin>\n"
        "       GGML_QNN_U16_ACTIVATIONS=1 %s --profile-attention-fused-test "
        "<runtime-profile.bin> <layer> <n-kv>\n"
        "       GGML_QNN_U16_ACTIVATIONS=1 %s --profile-ffn-vector-test "
        "<runtime-profile.json> <layer> <gate.u16.bin> <up.u16.bin> "
        "<sigmoid.u16.bin> <silu.u16.bin> <product.u16.bin>\n"
        "       GGML_QNN_U16_ACTIVATIONS=1 %s --profile-ffn-fused-test "
        "<runtime-profile.bin> <layer> <tokens>\n",
        program,
        program,
        program,
        program,
        program,
        program,
        program,
        program,
        program,
        program,
        program,
        program,
        program,
        program,
        program,
        program,
        program,
        program,
        program);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc >= 2 && argc <= 3 &&
        std::string_view(argv[1]) == "--gptq2-gemv-mt-test") {
        if (!u16_activations_enabled()) {
            std::fprintf(
                stderr,
                "U16 activation core is disabled; set "
                "GGML_QNN_U16_ACTIVATIONS=1 explicitly.\n");
            return 2;
        }
        if (!initialize_ggml_cpu_backend()) {
            std::fprintf(stderr, "failed to initialize GGML CPU backend.\n");
            return 6;
        }
        const int threads = argc == 3 ? std::stoi(argv[2]) : 6;
        return run_gptq2_u16_gemv_4row_test(threads);
    }
    if (argc >= 3 && argc <= 5 &&
        std::string_view(argv[1]) == "--lm-head-gemv-test") {
        const std::string_view type(argv[2]);
        const ggml_type weight_type =
            type == "q8_0" ? GGML_TYPE_Q8_0 :
            type == "q6_k" ? GGML_TYPE_Q6_K : GGML_TYPE_COUNT;
        const int threads = argc >= 4 ? std::stoi(argv[3]) : 1;
        const int64_t rows = argc >= 5 ? std::stoll(argv[4]) : 151936;
        return run_lm_head_gemv_test(weight_type, threads, rows);
    }
    if (argc == 3 && std::string_view(argv[1]) == "--profile-runtime-test") {
        try {
            const auto profile = llama_qnn_quant_profile_load_file(argv[2]);
            size_t u16_operand_count = 0;
            size_t aux_operand_count = 0;
            for (const auto & operation : profile->operations) {
                if (profile->find_operation(operation.shard_index, operation.name) != &operation) {
                    throw std::runtime_error("operation shard/name lookup mismatch");
                }
                if (operation.decoder_binding.layer_ids.size() == 1 &&
                    profile->find_operation_by_fx(
                        operation.decoder_binding.layer_ids.front(),
                        operation.fx_node_name) != &operation) {
                    throw std::runtime_error("operation layer/FX lookup mismatch");
                }
                for (const auto & operand : operation.u16_operands) {
                    const auto * tensor = profile->find_u16_operand(operand);
                    if (tensor == nullptr || tensor->shard_index != operation.shard_index ||
                        tensor->name != operand.name) {
                        throw std::runtime_error("operation U16 operand lookup mismatch");
                    }
                    ++u16_operand_count;
                }
                for (const auto & operand : operation.aux_operands) {
                    const auto * tensor = profile->find_aux_operand(operand);
                    if (tensor == nullptr || tensor->shard_index != operation.shard_index ||
                        tensor->name != operand.name) {
                        throw std::runtime_error("operation auxiliary operand lookup mismatch");
                    }
                    ++aux_operand_count;
                }
            }
            const auto * qk_rotation =
                profile->find_aux_tensor(0, "b__frozen_param313@0");
            const auto * kv_input = profile->find_aux_tensor(0, "args_0_h_0@0");
            const auto * q_rotation_operation =
                profile->find_operation_by_fx(0, "aten_matmul_default_h_0");
            if (qk_rotation == nullptr ||
                qk_rotation->data_type != "QNN_DATATYPE_SFIXED_POINT_16" ||
                qk_rotation->element_bytes != 2 ||
                qk_rotation->dimensions != std::vector<int64_t>({128, 128}) ||
                qk_rotation->static_data.size() != 128U * 128U * sizeof(int16_t) ||
                qk_rotation->qparams.encoding != LLAMA_QNN_QUANTIZATION_SCALE_OFFSET ||
                qk_rotation->qparams.scale_offsets.size() != 1 ||
                qk_rotation->qparams.scale_offsets.front().zero_point != 0) {
                throw std::runtime_error("layer-0 S16 Q/K rotation tensor is incomplete");
            }
            if (kv_input == nullptr ||
                kv_input->data_type != "QNN_DATATYPE_UFIXED_POINT_8" ||
                kv_input->element_bytes != 1 ||
                kv_input->qparams.encoding != LLAMA_QNN_QUANTIZATION_SCALE_OFFSET ||
                kv_input->qparams.scale_offsets.size() != 1) {
                throw std::runtime_error("layer-0 U8 KV tensor qparams are incomplete");
            }
            if (q_rotation_operation == nullptr ||
                profile->find_aux_operand(*q_rotation_operation, "input", 1) !=
                    qk_rotation) {
                throw std::runtime_error("layer-0 Q rotation does not bind the S16 matrix");
            }
            std::printf(
                "qnn-profile-runtime-test: u16_tensors=%zu aux_tensors=%zu "
                "linear_pairs=%zu operations=%zu u16_operands=%zu "
                "aux_operands=%zu static_u16_tensors=%zu static_u16_bytes=%zu "
                "static_aux_tensors=%zu static_aux_bytes=%zu "
                "qk_rotation_bytes=%zu kv_type=%s source_bits=%d group_size=%d "
                "status=pass profile=%s\n",
                profile->u16_tensor_count(),
                profile->aux_quantized_tensor_count(),
                profile->linear_qparams_count(),
                profile->operation_count(),
                u16_operand_count,
                aux_operand_count,
                profile->static_u16_tensor_count(),
                profile->static_u16_bytes(),
                profile->static_aux_tensor_count(),
                profile->static_aux_bytes(),
                qk_rotation->static_data.size(),
                kv_input->data_type.c_str(),
                profile->source_weight_bits,
                profile->source_group_size,
                argv[2]);
            return 0;
        } catch (const std::exception & error) {
            std::fprintf(stderr, "qnn-profile-runtime-test failed: %s\n", error.what());
            return 12;
        }
    }
    if (argc == 4 && std::string_view(argv[1]) == "--profile-gguf-linear-test") {
        if (!u16_activations_enabled()) {
            std::fprintf(stderr, "U16 activation core is disabled; set GGML_QNN_U16_ACTIVATIONS=1 explicitly.\n");
            return 2;
        }
        try {
            return run_profile_gguf_linear_test(argv[2], argv[3]);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "qnn-profile-gguf-linear-test failed: %s\n", error.what());
            return 15;
        }
    }
    if (argc == 8 && std::string_view(argv[1]) == "--profile-gguf-linear-vector-test") {
        if (!u16_activations_enabled()) {
            std::fprintf(stderr, "U16 activation core is disabled; set GGML_QNN_U16_ACTIVATIONS=1 explicitly.\n");
            return 2;
        }
        try {
            return run_profile_gguf_linear_vector_test(
                argv[2], argv[3], std::stoi(argv[4]), argv[5], argv[6], argv[7]);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "qnn-profile-gguf-linear-vector-test failed: %s\n", error.what());
            return 18;
        }
    }
    if (argc == 7 && std::string_view(argv[1]) == "--profile-rms-vector-test") {
        if (!u16_activations_enabled()) {
            std::fprintf(stderr, "U16 activation core is disabled; set GGML_QNN_U16_ACTIVATIONS=1 explicitly.\n");
            return 2;
        }
        try {
            return run_profile_rms_vector_test(
                argv[2], std::stoi(argv[3]), argv[4], argv[5], argv[6]);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "qnn-profile-rms-vector-test failed: %s\n", error.what());
            return 19;
        }
    }
    if (argc == 7 &&
        std::string_view(argv[1]) == "--profile-qk-rotate-vector-test") {
        if (!u16_activations_enabled()) {
            std::fprintf(stderr, "U16 activation core is disabled; set GGML_QNN_U16_ACTIVATIONS=1 explicitly.\n");
            return 2;
        }
        try {
            return run_profile_qk_rotate_vector_test(
                argv[2], std::stoi(argv[3]), argv[4], argv[5], argv[6]);
        } catch (const std::exception & error) {
            std::fprintf(
                stderr,
                "qnn-profile-qk-rotate-vector-test failed: %s\n",
                error.what());
            return 22;
        }
    }
    if (argc == 8 &&
        std::string_view(argv[1]) == "--profile-rope-vector-test") {
        if (!u16_activations_enabled()) {
            std::fprintf(stderr, "U16 activation core is disabled; set GGML_QNN_U16_ACTIVATIONS=1 explicitly.\n");
            return 2;
        }
        try {
            const std::string_view kind(argv[5]);
            if (kind != "q" && kind != "k") {
                throw std::runtime_error("RoPE kind must be q or k");
            }
            return run_profile_rope_vector_test(
                argv[2], std::stoi(argv[3]), std::stoi(argv[4]),
                kind == "k", argv[6], argv[7]);
        } catch (const std::exception & error) {
            std::fprintf(
                stderr,
                "qnn-profile-rope-vector-test failed: %s\n",
                error.what());
            return 23;
        }
    }
    if (argc == 7 && std::string_view(argv[1]) == "--profile-softmax-vector-test") {
        if (!u16_activations_enabled()) {
            std::fprintf(stderr, "U16 activation core is disabled; set GGML_QNN_U16_ACTIVATIONS=1 explicitly.\n");
            return 2;
        }
        try {
            return run_profile_softmax_vector_test(
                argv[2], std::stoi(argv[3]), argv[4], argv[5], argv[6]);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "qnn-profile-softmax-vector-test failed: %s\n", error.what());
            return 20;
        }
    }
    if (argc == 4 && std::string_view(argv[1]) == "--profile-softmax-batch-test") {
        if (!u16_activations_enabled()) {
            std::fprintf(stderr, "U16 activation core is disabled; set GGML_QNN_U16_ACTIVATIONS=1 explicitly.\n");
            return 2;
        }
        try {
            return run_profile_softmax_batch_test(argv[2], argv[3]);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "qnn-profile-softmax-batch-test failed: %s\n", error.what());
            return 20;
        }
    }
    if (argc == 5 &&
        std::string_view(argv[1]) == "--profile-attention-fused-test") {
        if (!u16_activations_enabled()) {
            std::fprintf(stderr, "U16 activation core is disabled; set GGML_QNN_U16_ACTIVATIONS=1 explicitly.\n");
            return 2;
        }
        try {
            return run_profile_attention_fused_test(
                argv[2], std::stoi(argv[3]), std::stoi(argv[4]));
        } catch (const std::exception & error) {
            std::fprintf(stderr, "qnn-profile-attention-fused-test failed: %s\n", error.what());
            return 24;
        }
    }
    if (argc == 9 &&
        std::string_view(argv[1]) == "--profile-attention-value-vector-test") {
        if (!u16_activations_enabled()) {
            std::fprintf(stderr, "U16 activation core is disabled; set GGML_QNN_U16_ACTIVATIONS=1 explicitly.\n");
            return 2;
        }
        try {
            return run_profile_attention_value_vector_test(
                argv[2], std::stoi(argv[3]), argv[4], std::stoi(argv[5]),
                argv[6], argv[7], argv[8]);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "qnn-profile-attention-value-vector-test failed: %s\n", error.what());
            return 21;
        }
    }
    if (argc == 9 && std::string_view(argv[1]) == "--profile-ffn-vector-test") {
        try {
            if (!llama_qnn_u16_activations_enabled()) {
                std::fprintf(stderr, "GGML_QNN_U16_ACTIVATIONS=1 is required.\n");
                return 2;
            }
            return run_profile_ffn_vector_test(
                argv[2], std::stoi(argv[3]), argv[4], argv[5],
                argv[6], argv[7], argv[8]);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "profile FFN vector test failed: %s\n", error.what());
            return 2;
        }
    }
    if (argc == 5 && std::string_view(argv[1]) == "--profile-ffn-fused-test") {
        try {
            if (!llama_qnn_u16_activations_enabled()) {
                std::fprintf(stderr, "GGML_QNN_U16_ACTIVATIONS=1 is required.\n");
                return 2;
            }
            return run_profile_ffn_fused_test(
                argv[2], std::stoi(argv[3]), std::stoll(argv[4]));
        } catch (const std::exception & error) {
            std::fprintf(stderr, "profile fused FFN test failed: %s\n", error.what());
            return 2;
        }
    }
    if (argc == 3 && std::string_view(argv[1]) == "--profile-u16-op-test") {
        if (!u16_activations_enabled()) {
            std::fprintf(stderr, "U16 activation core is disabled; set GGML_QNN_U16_ACTIVATIONS=1 explicitly.\n");
            return 2;
        }
        try {
            return run_profile_u16_operation_test(argv[2]);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "qnn-profile-u16-op-test failed: %s\n", error.what());
            return 17;
        }
    }
    if (argc >= 3 && std::string_view(argv[1]) == "--profile-inspect") {
        std::optional<size_t> shard_index;
        constexpr std::string_view prefix = "--shard=";
        for (int index = 3; index < argc; ++index) {
            const std::string_view option(argv[index]);
            if (option.size() < prefix.size() || option.substr(0, prefix.size()) != prefix) {
                print_usage(argv[0]);
                return 1;
            }
            try {
                shard_index = std::stoull(std::string(option.substr(prefix.size())));
            } catch (const std::exception &) {
                std::fprintf(stderr, "invalid profile shard index: %.*s\n",
                    static_cast<int>(option.size()), option.data());
                return 1;
            }
        }
        return run_profile_inspect(argv[2], shard_index);
    }
    if (argc == 2 && std::string_view(argv[1]) == "--gptq2-gemv-test") {
        if (!u16_activations_enabled()) {
            std::fprintf(
                stderr,
                "U16 activation core is disabled; set GGML_QNN_U16_ACTIVATIONS=1 explicitly.\n");
            return 2;
        }
        if (!initialize_ggml_cpu_backend()) {
            std::fprintf(stderr, "failed to initialize GGML CPU backend.\n");
            return 6;
        }
        return run_gptq2_u16_gemv_4row_test();
    }
    if (argc == 3 && std::string_view(argv[1]) == "--self-test") {
        if (!u16_activations_enabled()) {
            std::fprintf(
                stderr,
                "U16 activation core is disabled; set GGML_QNN_U16_ACTIVATIONS=1 explicitly.\n");
            return 2;
        }
        constexpr std::string_view prefix = "--backend=";
        const std::string_view option(argv[2]);
        if (option.size() < prefix.size() || option.substr(0, prefix.size()) != prefix) {
            print_usage(argv[0]);
            return 1;
        }
        return run_self_test(parse_backend(option.substr(prefix.size())));
    }
    if (argc == 2 && std::string_view(argv[1]) == "--graph-test") {
        if (!u16_activations_enabled()) {
            std::fprintf(stderr, "U16 activation core is disabled; set GGML_QNN_U16_ACTIVATIONS=1 explicitly.\n");
            return 2;
        }
        return run_u16_graph_test();
    }
    if (argc >= 3 && std::string_view(argv[1]) == "--gguf-test") {
        if (!u16_activations_enabled()) {
            std::fprintf(
                stderr,
                "U16 activation core is disabled; set GGML_QNN_U16_ACTIVATIONS=1 explicitly.\n");
            return 2;
        }
        const char * tensor_name = nullptr;
        constexpr std::string_view prefix = "--tensor=";
        for (int index = 3; index < argc; ++index) {
            const std::string_view option(argv[index]);
            if (option.size() >= prefix.size() && option.substr(0, prefix.size()) == prefix) {
                tensor_name = argv[index] + prefix.size();
            } else {
                print_usage(argv[0]);
                return 1;
            }
        }
        return run_gguf_test(argv[2], tensor_name);
    }
    print_usage(argv[0]);
    return 1;
}
