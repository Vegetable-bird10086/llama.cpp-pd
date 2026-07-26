#include "llama-qnn-quant-profile.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include <json.hpp>

namespace {

using json = nlohmann::json;

constexpr const char * kProfileFormat = "llama-qnn-quant-profile-v2";
constexpr const char * kU16DataType = "QNN_DATATYPE_UFIXED_POINT_16";

uint32_t fixed_point_element_bytes(const std::string & data_type) {
    if (data_type == "QNN_DATATYPE_UFIXED_POINT_8" ||
        data_type == "QNN_DATATYPE_SFIXED_POINT_8") {
        return 1;
    }
    if (data_type == kU16DataType || data_type == "QNN_DATATYPE_SFIXED_POINT_16") {
        return 2;
    }
    if (data_type == "QNN_DATATYPE_UFIXED_POINT_32" ||
        data_type == "QNN_DATATYPE_SFIXED_POINT_32") {
        return 4;
    }
    return 0;
}

bool is_affine_quantization_encoding(const std::string & encoding) {
    return encoding == "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET" ||
        encoding == "QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET";
}

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("invalid QNN U16 quantization profile: " + message);
}

uint8_t hex_nibble(const char value, const std::string & location) {
    if (value >= '0' && value <= '9') {
        return static_cast<uint8_t>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
        return static_cast<uint8_t>(10 + value - 'a');
    }
    if (value >= 'A' && value <= 'F') {
        return static_cast<uint8_t>(10 + value - 'A');
    }
    fail(location + " has non-hex scale bytes");
}

int base64_value(const char value) {
    if (value >= 'A' && value <= 'Z') return value - 'A';
    if (value >= 'a' && value <= 'z') return value - 'a' + 26;
    if (value >= '0' && value <= '9') return value - '0' + 52;
    if (value == '+') return 62;
    if (value == '/') return 63;
    return -1;
}

std::vector<uint8_t> decode_base64(const std::string & encoded, const std::string & location) {
    if (encoded.empty() || encoded.size() % 4 != 0) {
        fail(location + " has an invalid base64 length");
    }
    std::vector<uint8_t> decoded;
    decoded.reserve(encoded.size() / 4 * 3);
    for (size_t index = 0; index < encoded.size(); index += 4) {
        const bool last = index + 4 == encoded.size();
        const char c0 = encoded[index + 0];
        const char c1 = encoded[index + 1];
        const char c2 = encoded[index + 2];
        const char c3 = encoded[index + 3];
        const int v0 = base64_value(c0);
        const int v1 = base64_value(c1);
        const int v2 = c2 == '=' ? 0 : base64_value(c2);
        const int v3 = c3 == '=' ? 0 : base64_value(c3);
        if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0 ||
            (!last && (c2 == '=' || c3 == '=')) ||
            (c2 == '=' && c3 != '=')) {
            fail(location + " has invalid base64 data");
        }
        const uint32_t packed = (static_cast<uint32_t>(v0) << 18) |
            (static_cast<uint32_t>(v1) << 12) |
            (static_cast<uint32_t>(v2) << 6) |
            static_cast<uint32_t>(v3);
        decoded.push_back(static_cast<uint8_t>(packed >> 16));
        if (c2 != '=') decoded.push_back(static_cast<uint8_t>(packed >> 8));
        if (c3 != '=') decoded.push_back(static_cast<uint8_t>(packed));
    }
    return decoded;
}

llama_qnn_affine_qparams parse_affine(const json & value, const std::string & location) {
    if (!value.is_object() || !value.contains("scale_f32_le_hex") ||
        !value.contains("offset") || !value.contains("zero_point")) {
        fail(location + " lacks exact scale/offset qparams");
    }
    const std::string raw = value.at("scale_f32_le_hex").get<std::string>();
    if (raw.size() != 8) {
        fail(location + " scale must have exactly four f32 bytes");
    }
    const std::array<uint8_t, 4> bytes = {
        static_cast<uint8_t>((hex_nibble(raw[0], location) << 4) | hex_nibble(raw[1], location)),
        static_cast<uint8_t>((hex_nibble(raw[2], location) << 4) | hex_nibble(raw[3], location)),
        static_cast<uint8_t>((hex_nibble(raw[4], location) << 4) | hex_nibble(raw[5], location)),
        static_cast<uint8_t>((hex_nibble(raw[6], location) << 4) | hex_nibble(raw[7], location)),
    };
    const uint32_t bits = static_cast<uint32_t>(bytes[0]) |
        (static_cast<uint32_t>(bytes[1]) << 8) |
        (static_cast<uint32_t>(bytes[2]) << 16) |
        (static_cast<uint32_t>(bytes[3]) << 24);
    float scale = 0.0f;
    static_assert(sizeof(scale) == sizeof(bits), "f32 size is required by the QNN profile ABI");
    std::memcpy(&scale, &bits, sizeof(scale));
    if (!std::isfinite(scale) || scale <= 0.0f) {
        fail(location + " has a non-positive or non-finite f32 scale");
    }
    const int64_t offset = value.at("offset").get<int64_t>();
    const int64_t zero_point = value.at("zero_point").get<int64_t>();
    if (offset < std::numeric_limits<int32_t>::min() ||
        offset > std::numeric_limits<int32_t>::max() ||
        zero_point < std::numeric_limits<int32_t>::min() ||
        zero_point > std::numeric_limits<int32_t>::max() ||
        zero_point != -offset) {
        fail(location + " has inconsistent QNN offset/zero point");
    }
    return {scale, bits, static_cast<int32_t>(offset), static_cast<int32_t>(zero_point)};
}

struct measured_sigmoid_lut {
    llama_qnn_affine_qparams input;
    llama_qnn_affine_qparams output;
    std::vector<uint16_t> codes;
};

bool same_affine(
        const llama_qnn_affine_qparams & lhs,
        const llama_qnn_affine_qparams & rhs) {
    return lhs.scale_bits == rhs.scale_bits &&
        lhs.offset == rhs.offset &&
        lhs.zero_point == rhs.zero_point;
}

std::unordered_map<int32_t, measured_sigmoid_lut> parse_measured_sigmoid_luts(
        const json & profile,
        int32_t num_decoder_layers) {
    std::unordered_map<int32_t, measured_sigmoid_lut> result;
    const auto entries = profile.find("decoder_sigmoid_u16_luts");
    if (entries == profile.end()) {
        return result;
    }
    if (!entries->is_array()) {
        fail("decoder_sigmoid_u16_luts must be an array");
    }
    for (const auto & entry : *entries) {
        const int64_t layer_id = entry.at("layer_id").get<int64_t>();
        if (layer_id < 0 || layer_id >= num_decoder_layers) {
            fail("decoder Sigmoid LUT has an invalid layer id");
        }
        const std::string location =
            "decoder_sigmoid_u16_luts[" + std::to_string(layer_id) + "]";
        if (entry.value("source", "") != "qnn-htp-exhaustive-u16-sweep" ||
            entry.value("code_count", 0) !=
                static_cast<int64_t>(UINT16_MAX) + 1 ||
            entry.value("code_storage", "") != "uint16_little_endian") {
            fail(location + " has an unsupported source or storage contract");
        }
        const auto bytes = decode_base64(
            entry.at("codes_le_base64").get<std::string>(),
            location + ".codes_le_base64");
        if (bytes.size() !=
            (static_cast<size_t>(UINT16_MAX) + 1) * sizeof(uint16_t)) {
            fail(location + " decoded LUT has the wrong size");
        }
        measured_sigmoid_lut parsed;
        parsed.input = parse_affine(entry.at("input_scale_offset"), location + ".input");
        parsed.output = parse_affine(entry.at("output_scale_offset"), location + ".output");
        parsed.codes.resize(static_cast<size_t>(UINT16_MAX) + 1);
        for (size_t index = 0; index < parsed.codes.size(); ++index) {
            parsed.codes[index] = static_cast<uint16_t>(
                static_cast<uint16_t>(bytes[index * 2]) |
                (static_cast<uint16_t>(bytes[index * 2 + 1]) << 8));
        }
        if (!result.emplace(static_cast<int32_t>(layer_id), std::move(parsed)).second) {
            fail(location + " duplicates a decoder layer");
        }
    }
    return result;
}

llama_qnn_tensor_qparams parse_qparams(const json & tensor, const std::string & location) {
    const std::string encoding = tensor.value("quantization_encoding", "");
    llama_qnn_tensor_qparams result;
    if (encoding == "QNN_QUANTIZATION_ENCODING_UNDEFINED") {
        return result;
    }
    if (encoding == "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET") {
        result.encoding = LLAMA_QNN_QUANTIZATION_SCALE_OFFSET;
        result.scale_offsets.push_back(parse_affine(tensor.at("scale_offset"), location));
        return result;
    }
    const json * payload = nullptr;
    if (encoding == "QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET") {
        result.encoding = LLAMA_QNN_QUANTIZATION_AXIS_SCALE_OFFSET;
        payload = &tensor.at("axis_scale_offset");
    } else if (encoding == "QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION") {
        result.encoding = LLAMA_QNN_QUANTIZATION_BLOCKWISE_EXPANSION;
        payload = &tensor.at("blockwise_expansion");
    } else {
        fail(location + " has unsupported quantization encoding " + encoding);
    }
    const int64_t axis = payload->at("axis").get<int64_t>();
    if (axis < 0 || axis > std::numeric_limits<int32_t>::max()) {
        fail(location + " has an invalid qparam axis");
    }
    result.axis = static_cast<int32_t>(axis);
    const auto & entries = payload->at("scale_offsets");
    if (!entries.is_array() || entries.empty()) {
        fail(location + " lacks scale offsets");
    }
    result.scale_offsets.reserve(entries.size());
    for (size_t index = 0; index < entries.size(); ++index) {
        result.scale_offsets.push_back(parse_affine(
            entries[index], location + ".scale_offsets[" + std::to_string(index) + "]"));
    }
    if (result.encoding == LLAMA_QNN_QUANTIZATION_BLOCKWISE_EXPANSION) {
        const int64_t bitwidth = payload->at("block_scale_bitwidth").get<int64_t>();
        const int64_t element_bytes = payload->at("block_scale_element_bytes").get<int64_t>();
        const int64_t blocks_per_axis = payload->at("num_blocks_per_axis").get<int64_t>();
        if (bitwidth <= 0 || bitwidth > 8 || element_bytes != 1 ||
            blocks_per_axis <= 0 || blocks_per_axis > std::numeric_limits<int32_t>::max() ||
            payload->value("block_scale_layout", "") !=
                "scale_offset_index_major_then_block_index" ||
            payload->value("block_scale_code_storage", "") !=
                "one_unsigned_code_per_storage_element") {
            fail(location + " has an unsupported QNN block-scale layout");
        }
        const uint64_t expected_codes =
            static_cast<uint64_t>(result.scale_offsets.size()) *
            static_cast<uint64_t>(blocks_per_axis);
        if (expected_codes > std::numeric_limits<size_t>::max() ||
            payload->at("block_scales_bytes").get<uint64_t>() != expected_codes) {
            fail(location + " has an invalid QNN block-scale byte count");
        }
        result.block_scale_codes = decode_base64(
            payload->at("block_scales_base64").get<std::string>(),
            location + ".block_scales_base64");
        if (result.block_scale_codes.size() != static_cast<size_t>(expected_codes)) {
            fail(location + " decoded QNN block scales have the wrong size");
        }
        result.block_scale_bitwidth = static_cast<int32_t>(bitwidth);
        result.block_scale_element_bytes = static_cast<int32_t>(element_bytes);
        result.num_blocks_per_axis = static_cast<int32_t>(blocks_per_axis);
    }
    return result;
}

llama_qnn_decoder_binding parse_decoder_binding(const json & value, const std::string & location) {
    if (!value.is_object()) {
        fail(location + " has an invalid decoder binding");
    }
    llama_qnn_decoder_binding result;
    const auto & layer_ids = value.at("layer_ids");
    const auto & module_paths = value.at("module_paths");
    if (!layer_ids.is_array() || !module_paths.is_array()) {
        fail(location + " has malformed decoder binding arrays");
    }
    for (const auto & layer_id : layer_ids) {
        const int64_t parsed = layer_id.get<int64_t>();
        if (parsed < 0 || parsed > std::numeric_limits<int32_t>::max()) {
            fail(location + " has an invalid decoder layer id");
        }
        result.layer_ids.push_back(static_cast<int32_t>(parsed));
    }
    for (const auto & module_path : module_paths) {
        const std::string path = module_path.get<std::string>();
        if (path.empty()) {
            fail(location + " has an empty decoder module path");
        }
        result.module_paths.push_back(path);
    }
    if (!value.at("projection").is_null()) {
        result.projection = value.at("projection").get<std::string>();
        if (result.projection.empty()) {
            fail(location + " has an empty decoder projection");
        }
    }
    return result;
}

void require_capabilities(const json & capabilities) {
    static constexpr std::array<const char *, 8> required = {
        "exact_f32_scale_bits",
        "raw_operator_parameters",
        "operation_tensor_sources",
        "embedded_static_affine_tensor_bytes",
        "blockwise_weight_scale_payload",
        "blockwise_weight_payload_digest",
        "structured_decoder_tensor_bindings",
        "complete_u16_tensor_qparams",
    };
    if (!capabilities.is_object()) {
        fail("profile capabilities are missing");
    }
    for (const char * capability : required) {
        if (!capabilities.value(capability, false)) {
            fail(std::string("profile lacks capability ") + capability);
        }
    }
}

std::string use_key(const llama_qnn_u16_tensor_use & use) {
    return use.operation_name + "\n" + use.role + "\n" + std::to_string(use.position);
}

std::string scoped_tensor_key(const std::string & scope, const std::string & name) {
    return std::to_string(scope.size()) + ":" + scope + name;
}

std::string sharded_tensor_key(const int32_t shard_index, const std::string & name) {
    return std::to_string(shard_index) + ":" + name;
}

std::string linear_qparams_key(const int32_t layer_id, const std::string & projection) {
    return std::to_string(layer_id) + ":" + projection;
}

std::string operation_fx_key(const int32_t layer_id, const std::string & fx_node_name) {
    return std::to_string(layer_id) + ":" + fx_node_name;
}

bool same_affine_qparams(
        const llama_qnn_affine_qparams & lhs,
        const llama_qnn_affine_qparams & rhs) {
    return lhs.scale_bits == rhs.scale_bits && lhs.offset == rhs.offset &&
        lhs.zero_point == rhs.zero_point;
}

int64_t scale_ratio_q20(
        const llama_qnn_affine_qparams & input,
        const llama_qnn_affine_qparams & output,
        const std::string & location) {
    const double scaled = std::ldexp(
        static_cast<double>(input.scale) / static_cast<double>(output.scale), 20);
    if (!std::isfinite(scaled) || scaled < 0.0 ||
        scaled > static_cast<double>(std::numeric_limits<int64_t>::max())) {
        fail(location + " has an unrepresentable input/output scale ratio");
    }
    return static_cast<int64_t>(std::llround(scaled));
}

int64_t scale_ratio_q31(
        const llama_qnn_affine_qparams & input,
        const llama_qnn_affine_qparams & output,
        const std::string & location) {
    const double scaled = std::ldexp(
        static_cast<double>(input.scale) / static_cast<double>(output.scale), 31);
    if (!std::isfinite(scaled) || scaled < 0.0 ||
        scaled > static_cast<double>(std::numeric_limits<int32_t>::max())) {
        fail(location + " has an unrepresentable Q31 input/output scale ratio");
    }
    return static_cast<int64_t>(std::llround(scaled));
}

int64_t scale_product_ratio_q20(
        const llama_qnn_affine_qparams & lhs,
        const llama_qnn_affine_qparams & rhs,
        const llama_qnn_affine_qparams & output,
        const std::string & location) {
    const double scaled = std::ldexp(
        (static_cast<double>(lhs.scale) * static_cast<double>(rhs.scale)) /
            static_cast<double>(output.scale), 20);
    if (!std::isfinite(scaled) || scaled < 0.0 ||
        scaled > static_cast<double>(std::numeric_limits<int64_t>::max())) {
        fail(location + " has an unrepresentable input-product/output scale ratio");
    }
    return static_cast<int64_t>(std::llround(scaled));
}

int64_t scale_product_ratio_q31(
        const llama_qnn_affine_qparams & lhs,
        const llama_qnn_affine_qparams & rhs,
        const llama_qnn_affine_qparams & output,
        const std::string & location) {
    const double scaled = std::ldexp(
        (static_cast<double>(lhs.scale) * static_cast<double>(rhs.scale)) /
            static_cast<double>(output.scale), 31);
    if (!std::isfinite(scaled) || scaled < 0.0 ||
        scaled > static_cast<double>(std::numeric_limits<int64_t>::max())) {
        fail(location + " has an unrepresentable Q31 input-product/output scale ratio");
    }
    return static_cast<int64_t>(std::llround(scaled));
}

int32_t operation_head_index(const std::string & operation_name) {
    const size_t marker = operation_name.rfind("_h_");
    if (marker == std::string::npos || marker + 3 == operation_name.size()) {
        return -1;
    }
    int64_t result = 0;
    for (size_t index = marker + 3; index < operation_name.size(); ++index) {
        const char value = operation_name[index];
        if (value < '0' || value > '9') {
            return -1;
        }
        result = result * 10 + (value - '0');
        if (result > std::numeric_limits<int32_t>::max()) {
            fail("linear operation has an invalid head index: " + operation_name);
        }
    }
    return static_cast<int32_t>(result);
}

void append_linear_blockwise_qparams(
        llama_qnn_linear_qparams & linear,
        const llama_qnn_tensor_qparams & weight,
        const std::string & operation_name,
        const std::string & location) {
    if (weight.encoding != LLAMA_QNN_QUANTIZATION_BLOCKWISE_EXPANSION ||
        weight.scale_offsets.empty() || weight.num_blocks_per_axis <= 0 ||
        weight.block_scale_codes.size() !=
            weight.scale_offsets.size() * static_cast<size_t>(weight.num_blocks_per_axis)) {
        fail(location + " has incomplete blockwise weight qparams");
    }
    const int32_t source_group_size = 32;
    if (linear.qnn_weight_block_size == 0) {
        linear.qnn_weight_block_size = source_group_size;
        linear.qnn_weight_blocks_per_row = weight.num_blocks_per_axis;
    } else if (linear.qnn_weight_block_size != source_group_size ||
               linear.qnn_weight_blocks_per_row != weight.num_blocks_per_axis) {
        fail(location + " disagrees on QNN blockwise weight geometry");
    }

    const int32_t head_index = operation_head_index(operation_name);
    const size_t chunk_rows = weight.scale_offsets.size();
    const size_t first_row = head_index < 0
        ? 0 : static_cast<size_t>(head_index) * chunk_rows;
    const size_t required_rows = first_row + chunk_rows;
    if (linear.qnn_channel_scale_to_output_q31.size() < required_rows) {
        linear.qnn_channel_scale_to_output_q31.resize(required_rows, 0);
    }
    const size_t required_codes =
        required_rows * static_cast<size_t>(linear.qnn_weight_blocks_per_row);
    if (linear.qnn_weight_block_scale_codes.size() < required_codes) {
        linear.qnn_weight_block_scale_codes.resize(required_codes, 0);
    }

    for (size_t row = 0; row < chunk_rows; ++row) {
        const size_t output_row = first_row + row;
        if (linear.qnn_channel_scale_to_output_q31[output_row] != 0) {
            fail(location + " overlaps another QNN weight output row");
        }
        const int64_t multiplier = scale_product_ratio_q31(
            linear.input, weight.scale_offsets[row], linear.output,
            location + ".scale_offsets[" + std::to_string(row) + "]");
        if (multiplier <= 0) {
            fail(location + " has an unrepresentable QNN channel scale");
        }
        linear.qnn_channel_scale_to_output_q31[output_row] = multiplier;
        const size_t source = row * static_cast<size_t>(weight.num_blocks_per_axis);
        const size_t destination =
            output_row * static_cast<size_t>(linear.qnn_weight_blocks_per_row);
        std::copy_n(
            weight.block_scale_codes.begin() + source,
            static_cast<size_t>(weight.num_blocks_per_axis),
            linear.qnn_weight_block_scale_codes.begin() + destination);
    }
}

bool is_gptq_linear_projection(const std::string & projection) {
    static constexpr std::array<std::string_view, 7> projections = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    };
    for (const auto candidate : projections) {
        if (projection == candidate) {
            return true;
        }
    }
    return false;
}

} // namespace

const llama_qnn_u16_tensor * llama_qnn_quant_profile::find_u16_tensor(const std::string & name) const {
    const auto found = u16_tensor_index.find(name);
    return found == u16_tensor_index.end() ? nullptr : &u16_tensors.at(found->second);
}

const llama_qnn_u16_tensor * llama_qnn_quant_profile::find_u16_tensor(
        const int32_t shard_index, const std::string & name) const {
    const auto found = u16_tensor_shard_index.find(sharded_tensor_key(shard_index, name));
    return found == u16_tensor_shard_index.end() ? nullptr : &u16_tensors.at(found->second);
}

const llama_qnn_u16_tensor * llama_qnn_quant_profile::find_u16_tensor(
        const std::string & scope, const std::string & name) const {
    const auto found = u16_tensor_scope_index.find(scoped_tensor_key(scope, name));
    return found == u16_tensor_scope_index.end() ? nullptr : &u16_tensors.at(found->second);
}

const llama_qnn_aux_quantized_tensor * llama_qnn_quant_profile::find_aux_tensor(
        const std::string & name) const {
    const auto found = aux_tensor_index.find(name);
    return found == aux_tensor_index.end() ? nullptr : &aux_quantized_tensors.at(found->second);
}

const llama_qnn_aux_quantized_tensor * llama_qnn_quant_profile::find_aux_tensor(
        const int32_t shard_index, const std::string & name) const {
    const auto found = aux_tensor_shard_index.find(sharded_tensor_key(shard_index, name));
    return found == aux_tensor_shard_index.end() ? nullptr : &aux_quantized_tensors.at(found->second);
}

const llama_qnn_aux_quantized_tensor * llama_qnn_quant_profile::find_aux_tensor(
        const std::string & scope, const std::string & name) const {
    const auto found = aux_tensor_scope_index.find(scoped_tensor_key(scope, name));
    return found == aux_tensor_scope_index.end() ? nullptr : &aux_quantized_tensors.at(found->second);
}

const llama_qnn_linear_qparams * llama_qnn_quant_profile::find_linear_qparams(
        const int32_t layer_id, const std::string & projection) const {
    const auto found = linear_qparams_index.find(linear_qparams_key(layer_id, projection));
    return found == linear_qparams_index.end() ? nullptr : &linear_qparams.at(found->second);
}

const llama_qnn_operation * llama_qnn_quant_profile::find_operation(
        const int32_t shard_index, const std::string & name) const {
    const auto found = operation_shard_index.find(sharded_tensor_key(shard_index, name));
    return found == operation_shard_index.end() ? nullptr : &operations.at(found->second);
}

const llama_qnn_operation * llama_qnn_quant_profile::find_producer(
        const int32_t shard_index, const std::string & tensor_name) const {
    const auto found = operation_output_shard_index.find(
        sharded_tensor_key(shard_index, tensor_name));
    return found == operation_output_shard_index.end()
        ? nullptr : &operations.at(found->second);
}

const llama_qnn_operation * llama_qnn_quant_profile::find_operation_by_fx(
        const int32_t layer_id, const std::string & fx_node_name) const {
    const auto found = operation_fx_index.find(operation_fx_key(layer_id, fx_node_name));
    return found == operation_fx_index.end() ? nullptr : &operations.at(found->second);
}

const llama_qnn_u16_tensor * llama_qnn_quant_profile::find_u16_operand(
        const llama_qnn_operation_u16_operand & operand) const {
    return operand.tensor_index < u16_tensors.size() ? &u16_tensors.at(operand.tensor_index) : nullptr;
}

const llama_qnn_aux_quantized_tensor * llama_qnn_quant_profile::find_aux_operand(
        const llama_qnn_operation_aux_operand & operand) const {
    return operand.tensor_index < aux_quantized_tensors.size()
        ? &aux_quantized_tensors.at(operand.tensor_index) : nullptr;
}

const llama_qnn_aux_quantized_tensor * llama_qnn_quant_profile::find_aux_operand(
        const llama_qnn_operation & operation,
        const std::string & role,
        const int32_t position) const {
    for (const auto & operand : operation.aux_operands) {
        if (operand.role == role && operand.position == position) {
            return find_aux_operand(operand);
        }
    }
    return nullptr;
}


const llama_qnn_u16_tensor * llama_qnn_quant_profile::find_u16_operand(
        const llama_qnn_operation & operation,
        const std::string & role,
        const int32_t position) const {
    for (const auto & operand : operation.u16_operands) {
        if (operand.role == role && operand.position == position) {
            return find_u16_operand(operand);
        }
    }
    return nullptr;
}

size_t llama_qnn_quant_profile::u16_tensor_count() const {
    return u16_tensors.size();
}

size_t llama_qnn_quant_profile::aux_quantized_tensor_count() const {
    return aux_quantized_tensors.size();
}

size_t llama_qnn_quant_profile::linear_qparams_count() const {
    return linear_qparams.size();
}

size_t llama_qnn_quant_profile::operation_count() const {
    return operations.size();
}

size_t llama_qnn_quant_profile::static_u16_tensor_count() const {
    size_t count = 0;
    for (const auto & tensor : u16_tensors) {
        count += !tensor.static_data.empty();
    }
    return count;
}

size_t llama_qnn_quant_profile::static_u16_bytes() const {
    size_t bytes = 0;
    for (const auto & tensor : u16_tensors) {
        bytes += tensor.static_data.size() * sizeof(uint16_t);
    }
    return bytes;
}

size_t llama_qnn_quant_profile::static_aux_tensor_count() const {
    size_t count = 0;
    for (const auto & tensor : aux_quantized_tensors) {
        count += !tensor.static_data.empty();
    }
    return count;
}

size_t llama_qnn_quant_profile::static_aux_bytes() const {
    size_t bytes = 0;
    for (const auto & tensor : aux_quantized_tensors) {
        bytes += tensor.static_data.size();
    }
    return bytes;
}

std::shared_ptr<llama_qnn_quant_profile> llama_qnn_quant_profile_load_file(const std::string & path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open QNN U16 quantization manifest: " + path);
    }
    const json manifest = json::parse(stream);
    const auto & profile = manifest.at("graphs").at("prefill_forward").at("llama_qnn_quant_profile");
    if (profile.value("schema_version", 0) != 2 || profile.value("format", "") != kProfileFormat) {
        fail("manifest has no v2 prefill QNN profile");
    }
    require_capabilities(profile.at("capabilities"));
    const auto & shards = profile.at("shards");
    if (!shards.is_array() || shards.empty()) {
        fail("profile has no shards");
    }

    auto result = std::make_shared<llama_qnn_quant_profile>();
    result->source_path = path;
    result->quantization_formula = profile.value("quantization_formula", "");
    if (result->quantization_formula != "real=(integer_code+offset)*scale") {
        fail("profile uses an unsupported quantization formula");
    }
    const int64_t decoder_layers = manifest.at("num_decoder_layers").get<int64_t>();
    const auto & recipe = profile.at("gptq_source_recipe");
    const auto & contract = recipe.at("qnn_weight_code_contract");
    const int64_t source_weight_bits = recipe.at("source_weight_bits").get<int64_t>();
    const int64_t source_group_size = recipe.at("group_size").get<int64_t>();
    if (decoder_layers <= 0 || decoder_layers > std::numeric_limits<int32_t>::max()) {
        fail("manifest has an invalid decoder layer count");
    }
    if (source_weight_bits != 2 || source_group_size < 32 || source_group_size % 32 != 0 ||
        source_group_size > std::numeric_limits<int32_t>::max() ||
        contract.at("source_group_size").get<int64_t>() != source_group_size ||
        contract.at("source_group_code_bytes").get<int64_t>() != source_group_size / 4 ||
        contract.at("source_code_bits").get<int64_t>() != source_weight_bits ||
        contract.at("qnn_code_bits").get<int64_t>() != 4 ||
        contract.at("source_code_packing").get<std::string>() !=
            "four_int2_codes_per_byte_lsb_first" ||
        contract.at("qnn_code_formula").get<std::string>() !=
            "qnn_signed_code=source_code-source_zero_point") {
        fail("profile has an incompatible GPTQ source contract");
    }
    result->num_decoder_layers = static_cast<int32_t>(decoder_layers);
    result->source_weight_bits = static_cast<int32_t>(source_weight_bits);
    result->source_group_size = static_cast<int32_t>(source_group_size);
    auto measured_sigmoid_luts = parse_measured_sigmoid_luts(
        profile, result->num_decoder_layers);
    std::unordered_set<std::string> ambiguous_names;
    std::unordered_set<std::string> ambiguous_aux_names;
    for (size_t shard_index = 0; shard_index < shards.size(); ++shard_index) {
        const auto & shard = shards[shard_index];
        if (shard_index > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            fail("shard index is outside int32 range");
        }
        const int32_t parsed_shard_index = static_cast<int32_t>(shard_index);
        const std::string scope = shard.at("scope").get<std::string>();
        if (scope.empty()) {
            fail("shard " + std::to_string(shard_index) + " has an empty scope");
        }
        require_capabilities(shard.at("capabilities"));
        const auto & tensors = shard.at("tensors");
        const auto & index = shard.at("u16_tensor_index");
        if (!tensors.is_object() || !index.is_object()) {
            fail("shard " + std::to_string(shard_index) + " lacks U16 tensor metadata");
        }
        std::unordered_set<std::string> expected_names;
        for (auto iterator = tensors.begin(); iterator != tensors.end(); ++iterator) {
            if (iterator.value().value("data_type", "") == kU16DataType) {
                expected_names.insert(iterator.key());
            }
        }
        if (index.size() != expected_names.size()) {
            fail("shard " + std::to_string(shard_index) + " has incomplete U16 tensor coverage");
        }
        for (auto iterator = index.begin(); iterator != index.end(); ++iterator) {
            const std::string name = iterator.key();
            if (!expected_names.erase(name)) {
                fail("shard " + std::to_string(shard_index) + " has an invalid U16 tensor index entry " + name);
            }
            const auto & index_entry = iterator.value();
            const auto tensor_it = tensors.find(name);
            const auto & tensor = tensor_it.value();
            if (index_entry.value("name", "") != name ||
                index_entry.value("data_type", "") != tensor.value("data_type", "") ||
                index_entry.value("quantization_encoding", "") != tensor.value("quantization_encoding", "")) {
                fail("shard " + std::to_string(shard_index) + " has inconsistent U16 tensor identity " + name);
            }
            llama_qnn_u16_tensor parsed;
            parsed.shard_index = parsed_shard_index;
            parsed.scope = scope;
            parsed.name = name;
            parsed.data_type = tensor.at("data_type").get<std::string>();
            const auto & dimensions = tensor.at("dimensions");
            if (!dimensions.is_array() || dimensions.empty()) {
                fail("U16 tensor " + name + " lacks dimensions");
            }
            for (const auto & dimension : dimensions) {
                const int64_t parsed_dimension = dimension.get<int64_t>();
                if (parsed_dimension <= 0) {
                    fail("U16 tensor " + name + " has an invalid dimension");
                }
                parsed.dimensions.push_back(parsed_dimension);
            }
            parsed.qparams = parse_qparams(tensor, "U16 tensor " + name);
            if (parsed.qparams.encoding == LLAMA_QNN_QUANTIZATION_UNDEFINED) {
                fail("U16 tensor " + name + " has no QNN quantization encoding");
            }
            if (tensor.contains("static_payload")) {
                const auto & payload = tensor.at("static_payload");
                if (!payload.is_object() ||
                    payload.value("storage", "") != "embedded_exact_bytes" ||
                    payload.value("element_bytes", 0) != static_cast<int>(sizeof(uint16_t)) ||
                    !payload.contains("data_le_base64") ||
                    !payload.contains("data_bytes") ||
                    !payload.contains("sha256")) {
                    fail("U16 tensor " + name + " has malformed exact static data");
                }
                size_t elements = 1;
                for (const int64_t dimension : parsed.dimensions) {
                    if (static_cast<uint64_t>(dimension) >
                        std::numeric_limits<size_t>::max() / elements) {
                        fail("U16 tensor " + name + " static data dimensions overflow");
                    }
                    elements *= static_cast<size_t>(dimension);
                }
                if (elements > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
                    fail("U16 tensor " + name + " static data byte size overflows");
                }
                const size_t expected_bytes = elements * sizeof(uint16_t);
                const uint64_t declared_bytes = payload.at("data_bytes").get<uint64_t>();
                if (declared_bytes != expected_bytes) {
                    fail("U16 tensor " + name + " static data size disagrees with dimensions");
                }
                const std::vector<uint8_t> decoded = decode_base64(
                    payload.at("data_le_base64").get<std::string>(),
                    "U16 tensor " + name + ".static_payload");
                if (decoded.size() != expected_bytes) {
                    fail("U16 tensor " + name + " decoded static data has the wrong size");
                }
                parsed.static_data.resize(elements);
                for (size_t element = 0; element < elements; ++element) {
                    parsed.static_data[element] =
                        static_cast<uint16_t>(decoded[element * 2]) |
                        static_cast<uint16_t>(decoded[element * 2 + 1] << 8);
                }
                parsed.static_data_sha256 = payload.at("sha256").get<std::string>();
                if (parsed.static_data_sha256.size() != 64) {
                    fail("U16 tensor " + name + " static data has an invalid SHA-256 digest");
                }
            }
            const auto & uses = index_entry.at("operation_uses");
            if (!uses.is_array()) {
                fail("U16 tensor " + name + " has no operation uses");
            }
            std::unordered_set<std::string> seen_uses;
            for (const auto & use : uses) {
                llama_qnn_u16_tensor_use parsed_use;
                parsed_use.operation_name = use.at("operation_name").get<std::string>();
                parsed_use.role = use.at("role").get<std::string>();
                const int64_t position = use.at("position").get<int64_t>();
                if (parsed_use.operation_name.empty() ||
                    (parsed_use.role != "input" && parsed_use.role != "output" && parsed_use.role != "tensor_param") ||
                    position < 0 || position > std::numeric_limits<int32_t>::max()) {
                    fail("U16 tensor " + name + " has an invalid operation use");
                }
                parsed_use.position = static_cast<int32_t>(position);
                if (!seen_uses.insert(use_key(parsed_use)).second) {
                    fail("U16 tensor " + name + " has a duplicate operation use");
                }
                parsed.operation_uses.push_back(std::move(parsed_use));
            }
            const auto & bindings = index_entry.at("decoder_bindings");
            if (!bindings.is_array()) {
                fail("U16 tensor " + name + " lacks decoder bindings");
            }
            for (size_t binding_index = 0; binding_index < bindings.size(); ++binding_index) {
                parsed.decoder_bindings.push_back(parse_decoder_binding(
                    bindings[binding_index],
                    "U16 tensor " + name + ".decoder_bindings[" + std::to_string(binding_index) + "]"));
            }
            const size_t tensor_index = result->u16_tensors.size();
            if (!result->u16_tensor_shard_index.emplace(
                    sharded_tensor_key(parsed.shard_index, parsed.name), tensor_index).second ||
                !result->u16_tensor_scope_index.emplace(
                    scoped_tensor_key(parsed.scope, parsed.name), tensor_index).second) {
                fail("duplicate U16 QNN tensor ABI identity in shard " +
                    std::to_string(parsed.shard_index) + ": " + parsed.name);
            }
            if (ambiguous_names.find(parsed.name) == ambiguous_names.end()) {
                const auto inserted = result->u16_tensor_index.emplace(parsed.name, tensor_index);
                if (!inserted.second) {
                    result->u16_tensor_index.erase(inserted.first);
                    ambiguous_names.insert(parsed.name);
                }
            }
            result->u16_tensors.push_back(std::move(parsed));
        }
        if (!expected_names.empty()) {
            fail("shard " + std::to_string(shard_index) + " U16 index misses a tensor");
        }

        if (shard.contains("quantized_tensor_index")) {
            const auto & quantized_index = shard.at("quantized_tensor_index");
            if (!quantized_index.is_object()) {
                fail("shard " + std::to_string(shard_index) +
                    " has a malformed quantized tensor index");
            }
            std::unordered_set<std::string> expected_quantized_names;
            for (auto iterator = tensors.begin(); iterator != tensors.end(); ++iterator) {
                const auto & tensor = iterator.value();
                if (fixed_point_element_bytes(tensor.value("data_type", "")) != 0 &&
                    is_affine_quantization_encoding(
                        tensor.value("quantization_encoding", ""))) {
                    expected_quantized_names.insert(iterator.key());
                }
            }
            if (quantized_index.size() != expected_quantized_names.size()) {
                fail("shard " + std::to_string(shard_index) +
                    " has incomplete quantized tensor coverage");
            }
            for (auto iterator = quantized_index.begin();
                 iterator != quantized_index.end(); ++iterator) {
                const std::string name = iterator.key();
                if (!expected_quantized_names.erase(name)) {
                    fail("shard " + std::to_string(shard_index) +
                        " has an invalid quantized tensor index entry " + name);
                }
                const auto tensor_it = tensors.find(name);
                const auto & tensor = tensor_it.value();
                const auto & index_entry = iterator.value();
                const std::string data_type = tensor.at("data_type").get<std::string>();
                if (index_entry.value("name", "") != name ||
                    index_entry.value("data_type", "") != data_type ||
                    index_entry.value("quantization_encoding", "") !=
                        tensor.value("quantization_encoding", "")) {
                    fail("shard " + std::to_string(shard_index) +
                        " has inconsistent quantized tensor identity " + name);
                }
                if (data_type == kU16DataType) {
                    if (result->find_u16_tensor(parsed_shard_index, name) == nullptr) {
                        fail("quantized tensor index references an unknown U16 tensor " + name);
                    }
                    continue;
                }

                llama_qnn_aux_quantized_tensor parsed;
                parsed.shard_index = parsed_shard_index;
                parsed.scope = scope;
                parsed.name = name;
                parsed.data_type = data_type;
                parsed.element_bytes = fixed_point_element_bytes(data_type);
                if (parsed.element_bytes == 0) {
                    fail("auxiliary quantized tensor " + name +
                        " has an unsupported fixed-point data type");
                }
                const auto & dimensions = tensor.at("dimensions");
                if (!dimensions.is_array() || dimensions.empty()) {
                    fail("auxiliary quantized tensor " + name + " lacks dimensions");
                }
                size_t elements = 1;
                for (const auto & dimension : dimensions) {
                    const int64_t parsed_dimension = dimension.get<int64_t>();
                    if (parsed_dimension <= 0 ||
                        static_cast<uint64_t>(parsed_dimension) >
                            std::numeric_limits<size_t>::max() / elements) {
                        fail("auxiliary quantized tensor " + name +
                            " has invalid or overflowing dimensions");
                    }
                    parsed.dimensions.push_back(parsed_dimension);
                    elements *= static_cast<size_t>(parsed_dimension);
                }
                parsed.qparams = parse_qparams(
                    tensor, "auxiliary quantized tensor " + name);
                if (parsed.qparams.encoding != LLAMA_QNN_QUANTIZATION_SCALE_OFFSET &&
                    parsed.qparams.encoding != LLAMA_QNN_QUANTIZATION_AXIS_SCALE_OFFSET) {
                    fail("auxiliary quantized tensor " + name +
                        " does not use affine QNN qparams");
                }
                if (tensor.contains("static_payload")) {
                    const auto & payload = tensor.at("static_payload");
                    if (!payload.is_object() ||
                        payload.value("storage", "") != "embedded_exact_bytes" ||
                        payload.value("element_bytes", 0) !=
                            static_cast<int>(parsed.element_bytes) ||
                        !payload.contains("data_le_base64") ||
                        !payload.contains("data_bytes") ||
                        !payload.contains("sha256") ||
                        elements > std::numeric_limits<size_t>::max() /
                            parsed.element_bytes) {
                        fail("auxiliary quantized tensor " + name +
                            " has malformed exact static data");
                    }
                    const size_t expected_bytes = elements * parsed.element_bytes;
                    if (payload.at("data_bytes").get<uint64_t>() != expected_bytes) {
                        fail("auxiliary quantized tensor " + name +
                            " static data size disagrees with dimensions");
                    }
                    parsed.static_data = decode_base64(
                        payload.at("data_le_base64").get<std::string>(),
                        "auxiliary quantized tensor " + name + ".static_payload");
                    if (parsed.static_data.size() != expected_bytes) {
                        fail("auxiliary quantized tensor " + name +
                            " decoded static data has the wrong size");
                    }
                    parsed.static_data_sha256 =
                        payload.at("sha256").get<std::string>();
                    if (parsed.static_data_sha256.size() != 64) {
                        fail("auxiliary quantized tensor " + name +
                            " static data has an invalid SHA-256 digest");
                    }
                }
                const auto & uses = index_entry.at("operation_uses");
                if (!uses.is_array()) {
                    fail("auxiliary quantized tensor " + name +
                        " has no operation uses");
                }
                std::unordered_set<std::string> seen_uses;
                for (const auto & use : uses) {
                    llama_qnn_u16_tensor_use parsed_use;
                    parsed_use.operation_name =
                        use.at("operation_name").get<std::string>();
                    parsed_use.role = use.at("role").get<std::string>();
                    const int64_t position = use.at("position").get<int64_t>();
                    if (parsed_use.operation_name.empty() ||
                        (parsed_use.role != "input" && parsed_use.role != "output" &&
                         parsed_use.role != "tensor_param") ||
                        position < 0 ||
                        position > std::numeric_limits<int32_t>::max()) {
                        fail("auxiliary quantized tensor " + name +
                            " has an invalid operation use");
                    }
                    parsed_use.position = static_cast<int32_t>(position);
                    if (!seen_uses.insert(use_key(parsed_use)).second) {
                        fail("auxiliary quantized tensor " + name +
                            " has a duplicate operation use");
                    }
                    parsed.operation_uses.push_back(std::move(parsed_use));
                }
                const auto & bindings = index_entry.at("decoder_bindings");
                if (!bindings.is_array()) {
                    fail("auxiliary quantized tensor " + name +
                        " lacks decoder bindings");
                }
                for (size_t binding_index = 0;
                     binding_index < bindings.size(); ++binding_index) {
                    parsed.decoder_bindings.push_back(parse_decoder_binding(
                        bindings[binding_index],
                        "auxiliary quantized tensor " + name +
                            ".decoder_bindings[" +
                            std::to_string(binding_index) + "]"));
                }
                const size_t tensor_index = result->aux_quantized_tensors.size();
                if (!result->aux_tensor_shard_index.emplace(
                        sharded_tensor_key(parsed.shard_index, parsed.name),
                        tensor_index).second ||
                    !result->aux_tensor_scope_index.emplace(
                        scoped_tensor_key(parsed.scope, parsed.name),
                        tensor_index).second) {
                    fail("duplicate auxiliary QNN tensor ABI identity in shard " +
                        std::to_string(parsed.shard_index) + ": " + parsed.name);
                }
                if (ambiguous_aux_names.find(parsed.name) ==
                    ambiguous_aux_names.end()) {
                    const auto inserted = result->aux_tensor_index.emplace(
                        parsed.name, tensor_index);
                    if (!inserted.second) {
                        result->aux_tensor_index.erase(inserted.first);
                        ambiguous_aux_names.insert(parsed.name);
                    }
                }
                result->aux_quantized_tensors.push_back(std::move(parsed));
            }
            if (!expected_quantized_names.empty()) {
                fail("shard " + std::to_string(shard_index) +
                    " quantized tensor index misses a tensor");
            }
        }

        const int64_t layer_start = shard.at("layer_start").get<int64_t>();
        const int64_t layer_end = shard.at("layer_end_exclusive").get<int64_t>();
        const auto & operations = shard.at("operations");
        if (layer_start < 0 || layer_start >= layer_end ||
            layer_end > result->num_decoder_layers || !operations.is_array()) {
            fail("shard " + std::to_string(shard_index) + " has invalid layer or operation metadata");
        }
        for (const auto & operation : operations) {
            llama_qnn_operation parsed_operation;
            parsed_operation.shard_index = parsed_shard_index;
            parsed_operation.scope = scope;
            parsed_operation.name = operation.at("name").get<std::string>();
            parsed_operation.type_name = operation.at("type_name").get<std::string>();
            parsed_operation.fx_node_name =
                operation.at("source").at("fx_node_name").get<std::string>();
            parsed_operation.decoder_binding = parse_decoder_binding(
                operation.at("source").at("decoder_binding"),
                "operation " + parsed_operation.name + ".decoder_binding");
            if (parsed_operation.name.empty() || parsed_operation.type_name.empty() ||
                parsed_operation.fx_node_name.empty()) {
                fail("shard " + std::to_string(shard_index) +
                    " has an operation with incomplete identity");
            }
            if (parsed_operation.type_name == "RmsNorm") {
                parsed_operation.rms_epsilon = operation.at("epsilon").get<double>();
                if (!std::isfinite(parsed_operation.rms_epsilon) ||
                    parsed_operation.rms_epsilon < 0.0) {
                    fail("RmsNorm operation " + parsed_operation.name + " has invalid epsilon");
                }
            }
            const auto append_operands = [&](const json & names, const char * role,
                                             std::vector<std::string> & destination) {
                if (!names.is_array()) {
                    fail("operation " + parsed_operation.name +
                        " has a malformed " + role + " list");
                }
                destination.reserve(names.size());
                for (size_t position = 0; position < names.size(); ++position) {
                    const std::string name = names[position].get<std::string>();
                    destination.push_back(name);
                    const auto tensor = result->u16_tensor_shard_index.find(
                        sharded_tensor_key(parsed_shard_index, name));
                    if (tensor != result->u16_tensor_shard_index.end()) {
                        parsed_operation.u16_operands.push_back({
                            name, role, static_cast<int32_t>(position), tensor->second});
                    }
                    const auto aux_tensor = result->aux_tensor_shard_index.find(
                        sharded_tensor_key(parsed_shard_index, name));
                    if (aux_tensor != result->aux_tensor_shard_index.end()) {
                        parsed_operation.aux_operands.push_back({
                            name, role, static_cast<int32_t>(position),
                            aux_tensor->second});
                    }
                }
            };
            append_operands(operation.at("inputs"), "input", parsed_operation.inputs);
            append_operands(operation.at("outputs"), "output", parsed_operation.outputs);
            const auto * operation_output = result->find_u16_tensor(
                parsed_shard_index, parsed_operation.outputs.front());
            if (operation_output != nullptr &&
                operation_output->qparams.encoding == LLAMA_QNN_QUANTIZATION_SCALE_OFFSET &&
                operation_output->qparams.scale_offsets.size() == 1) {
                const auto & output_affine = operation_output->qparams.scale_offsets.front();
                const bool is_add_or_subtract =
                    parsed_operation.type_name == "ElementWiseAdd" ||
                    parsed_operation.type_name == "ElementWiseSubtract";
                const bool is_rms_norm = parsed_operation.type_name == "RmsNorm";
                const bool is_select = parsed_operation.type_name == "ElementWiseSelect";
                if (is_add_or_subtract || is_rms_norm || is_select) {
                    parsed_operation.input_to_output_q20.assign(parsed_operation.inputs.size(), 0);
                }
                for (size_t input_position = 0;
                     input_position < parsed_operation.inputs.size(); ++input_position) {
                    if (!is_add_or_subtract && !(is_rms_norm && input_position == 1) &&
                        !(is_select && input_position >= 1)) {
                        continue;
                    }
                    const auto * operation_input = result->find_u16_tensor(
                        parsed_shard_index, parsed_operation.inputs[input_position]);
                    if (operation_input != nullptr &&
                        operation_input->qparams.encoding == LLAMA_QNN_QUANTIZATION_SCALE_OFFSET &&
                        operation_input->qparams.scale_offsets.size() == 1) {
                        parsed_operation.input_to_output_q20[input_position] = scale_ratio_q20(
                            operation_input->qparams.scale_offsets.front(), output_affine,
                            "operation " + parsed_operation.name + " input " +
                                std::to_string(input_position));
                    }
                }
                if (parsed_operation.type_name == "ElementWiseMultiply" &&
                    parsed_operation.inputs.size() >= 2) {
                    const auto * lhs = result->find_u16_tensor(
                        parsed_shard_index, parsed_operation.inputs[0]);
                    const auto * rhs = result->find_u16_tensor(
                        parsed_shard_index, parsed_operation.inputs[1]);
                    if (lhs != nullptr && rhs != nullptr &&
                        lhs->qparams.encoding == LLAMA_QNN_QUANTIZATION_SCALE_OFFSET &&
                        rhs->qparams.encoding == LLAMA_QNN_QUANTIZATION_SCALE_OFFSET &&
                        lhs->qparams.scale_offsets.size() == 1 &&
                        rhs->qparams.scale_offsets.size() == 1) {
                        parsed_operation.product_to_output_q20 = scale_product_ratio_q20(
                            lhs->qparams.scale_offsets.front(),
                            rhs->qparams.scale_offsets.front(),
                            output_affine,
                            "operation " + parsed_operation.name);
                        parsed_operation.product_to_output_q31 = scale_product_ratio_q31(
                            lhs->qparams.scale_offsets.front(),
                            rhs->qparams.scale_offsets.front(),
                            output_affine,
                            "operation " + parsed_operation.name);
                    }
                }
                if (parsed_operation.type_name == "MatMul" &&
                    parsed_operation.inputs.size() >= 2) {
                    const auto scalar_affine = [&](const std::string & name)
                            -> const llama_qnn_affine_qparams * {
                        const auto * u16 = result->find_u16_tensor(
                            parsed_shard_index, name);
                        if (u16 != nullptr &&
                            u16->qparams.encoding == LLAMA_QNN_QUANTIZATION_SCALE_OFFSET &&
                            u16->qparams.scale_offsets.size() == 1) {
                            return &u16->qparams.scale_offsets.front();
                        }
                        const auto * aux = result->find_aux_tensor(
                            parsed_shard_index, name);
                        if (aux != nullptr &&
                            aux->qparams.encoding == LLAMA_QNN_QUANTIZATION_SCALE_OFFSET &&
                            aux->qparams.scale_offsets.size() == 1) {
                            return &aux->qparams.scale_offsets.front();
                        }
                        return nullptr;
                    };
                    const auto * lhs = scalar_affine(parsed_operation.inputs[0]);
                    const auto * rhs = scalar_affine(parsed_operation.inputs[1]);
                    if (lhs != nullptr && rhs != nullptr) {
                        parsed_operation.matmul_product_to_output_q31 =
                            scale_product_ratio_q31(
                                *lhs, *rhs, output_affine,
                                "operation " + parsed_operation.name);
                    }
                }
                if (parsed_operation.type_name == "Sigmoid") {
                    const auto * sigmoid_input = result->find_u16_tensor(
                        parsed_shard_index, parsed_operation.inputs.front());
                    if (sigmoid_input == nullptr ||
                        sigmoid_input->qparams.encoding != LLAMA_QNN_QUANTIZATION_SCALE_OFFSET ||
                        sigmoid_input->qparams.scale_offsets.size() != 1) {
                        fail("Sigmoid operation " + parsed_operation.name +
                            " lacks scalar input qparams");
                    }
                    const auto & input_affine = sigmoid_input->qparams.scale_offsets.front();
                    if (parsed_operation.decoder_binding.layer_ids.size() != 1) {
                        fail("Sigmoid operation " + parsed_operation.name +
                            " does not bind exactly one decoder layer");
                    }
                    const int32_t layer_id =
                        parsed_operation.decoder_binding.layer_ids.front();
                    const auto measured = measured_sigmoid_luts.find(layer_id);
                    if (measured != measured_sigmoid_luts.end()) {
                        if (!same_affine(measured->second.input, input_affine) ||
                            !same_affine(measured->second.output, output_affine)) {
                            fail("measured Sigmoid LUT qparams do not match layer " +
                                std::to_string(layer_id));
                        }
                        parsed_operation.unary_lut =
                            std::move(measured->second.codes);
                        measured_sigmoid_luts.erase(measured);
                    } else {
                        parsed_operation.unary_lut.resize(
                            static_cast<size_t>(UINT16_MAX) + 1);
                        for (uint32_t code = 0; code <= UINT16_MAX; ++code) {
                            const double input_value =
                                (static_cast<int64_t>(code) - input_affine.zero_point) *
                                static_cast<double>(input_affine.scale);
                            const double sigmoid = input_value >= 0.0
                                ? 1.0 / (1.0 + std::exp(-input_value))
                                : std::exp(input_value) / (1.0 + std::exp(input_value));
                            const double output_range =
                                1.0 / static_cast<double>(output_affine.scale) - 1.0;
                            const int64_t output_code =
                                static_cast<int64_t>(std::floor(sigmoid * output_range)) +
                                output_affine.zero_point;
                            parsed_operation.unary_lut[code] = static_cast<uint16_t>(
                                std::clamp<int64_t>(output_code, 0, UINT16_MAX));
                        }
                    }
                }
                if (parsed_operation.type_name == "RmsNorm") {
                    const auto * rms_input = result->find_u16_tensor(
                        parsed_shard_index, parsed_operation.inputs.front());
                    if (rms_input == nullptr ||
                        rms_input->qparams.encoding != LLAMA_QNN_QUANTIZATION_SCALE_OFFSET ||
                        rms_input->qparams.scale_offsets.size() != 1) {
                        fail("RmsNorm operation " + parsed_operation.name +
                            " lacks scalar input qparams");
                    }
                    const double input_scale = rms_input->qparams.scale_offsets.front().scale;
                    const double epsilon_codes_q16 = std::ldexp(
                        parsed_operation.rms_epsilon /
                            (input_scale * input_scale),
                        16);
                    if (!std::isfinite(epsilon_codes_q16) ||
                        epsilon_codes_q16 < 0.0 ||
                        epsilon_codes_q16 >
                            static_cast<double>(std::numeric_limits<uint64_t>::max())) {
                        fail("RmsNorm operation " + parsed_operation.name +
                            " has unrepresentable code-domain epsilon");
                    }
                    parsed_operation.rms_epsilon_in_codes_q16 =
                        static_cast<uint64_t>(std::llround(epsilon_codes_q16));
                }
                if (parsed_operation.type_name == "ReduceMin") {
                    const auto * input = result->find_u16_tensor(
                        parsed_shard_index, parsed_operation.inputs.front());
                    if (input != nullptr &&
                        input->qparams.encoding == LLAMA_QNN_QUANTIZATION_SCALE_OFFSET &&
                        input->qparams.scale_offsets.size() == 1) {
                        parsed_operation.unary_input_to_output_q20 = scale_ratio_q20(
                            input->qparams.scale_offsets.front(), output_affine,
                            "operation " + parsed_operation.name);
                    }
                }
                if (parsed_operation.type_name == "ElementWiseDivide" &&
                    parsed_operation.inputs.size() == 2) {
                    const auto * input = result->find_u16_tensor(
                        parsed_shard_index, parsed_operation.inputs[0]);
                    const auto * divisor = result->find_u16_tensor(
                        parsed_shard_index, parsed_operation.inputs[1]);
                    if (input != nullptr && divisor != nullptr &&
                        input->qparams.encoding == LLAMA_QNN_QUANTIZATION_SCALE_OFFSET &&
                        divisor->qparams.encoding == LLAMA_QNN_QUANTIZATION_SCALE_OFFSET &&
                        input->qparams.scale_offsets.size() == 1 &&
                        divisor->qparams.scale_offsets.size() == 1 &&
                        divisor->static_data.size() == 1) {
                        const auto & divisor_affine = divisor->qparams.scale_offsets.front();
                        const double divisor_value =
                            (static_cast<int64_t>(divisor->static_data.front()) -
                                divisor_affine.zero_point) *
                            static_cast<double>(divisor_affine.scale);
                        const double scaled = std::ldexp(
                            static_cast<double>(input->qparams.scale_offsets.front().scale) /
                                (divisor_value * static_cast<double>(output_affine.scale)),
                            20);
                        if (!std::isfinite(scaled) || scaled <= 0.0 ||
                            scaled > static_cast<double>(std::numeric_limits<int64_t>::max())) {
                            fail("operation " + parsed_operation.name +
                                " has an unrepresentable static divide ratio");
                        }
                        parsed_operation.unary_input_to_output_q20 =
                            static_cast<int64_t>(std::llround(scaled));
                    }
                }
                if (parsed_operation.type_name == "Softmax") {
                    const auto * input = result->find_u16_tensor(
                        parsed_shard_index, parsed_operation.inputs.front());
                    if (input == nullptr ||
                        input->qparams.encoding != LLAMA_QNN_QUANTIZATION_SCALE_OFFSET ||
                        input->qparams.scale_offsets.size() != 1) {
                        fail("Softmax operation " + parsed_operation.name +
                            " lacks scalar input qparams");
                    }
                    const double input_scale =
                        input->qparams.scale_offsets.front().scale;
                    // HTP consumes a Q22 multiplier widened into its Q24
                    // slot. Preserve the truncation: Q24 rounding changes
                    // piecewise-exponential bucket boundaries.
                    parsed_operation.softmax_scale_over_ln2_q24 =
                        static_cast<int64_t>(std::floor(std::ldexp(
                            input_scale / std::log(2.0), 22))) << 2;
                    parsed_operation.softmax_unit_code =
                        static_cast<int64_t>(std::ceil(
                            1.0 / static_cast<double>(output_affine.scale)));
                    parsed_operation.softmax_exp2_lut_q31.resize(257);
                    for (size_t index = 0;
                         index < parsed_operation.softmax_exp2_lut_q31.size(); ++index) {
                        parsed_operation.softmax_exp2_lut_q31[index] =
                            static_cast<uint32_t>(std::llround(std::ldexp(
                                std::exp2(-static_cast<double>(index) / 256.0), 31)));
                    }
                }
            }
            if (parsed_operation.type_name == "Convert" &&
                parsed_operation.inputs.size() == 1 &&
                parsed_operation.outputs.size() == 1) {
                const auto * input = result->find_u16_tensor(
                    parsed_shard_index, parsed_operation.inputs.front());
                const auto * output = result->find_aux_tensor(
                    parsed_shard_index, parsed_operation.outputs.front());
                if (input != nullptr && output != nullptr &&
                    output->data_type == "QNN_DATATYPE_UFIXED_POINT_8" &&
                    input->qparams.encoding == LLAMA_QNN_QUANTIZATION_SCALE_OFFSET &&
                    output->qparams.encoding == LLAMA_QNN_QUANTIZATION_SCALE_OFFSET &&
                    input->qparams.scale_offsets.size() == 1 &&
                    output->qparams.scale_offsets.size() == 1) {
                    parsed_operation.unary_input_to_output_q31 = scale_ratio_q31(
                        input->qparams.scale_offsets.front(),
                        output->qparams.scale_offsets.front(),
                        "operation " + parsed_operation.name);
                }
            }
            const size_t parsed_operation_index = result->operations.size();
            if (!result->operation_shard_index.emplace(
                    sharded_tensor_key(parsed_shard_index, parsed_operation.name),
                    parsed_operation_index).second) {
                fail("duplicate QNN operation identity in shard " +
                    std::to_string(shard_index) + ": " + parsed_operation.name);
            }
            for (const auto & output_name : parsed_operation.outputs) {
                if (!result->operation_output_shard_index.emplace(
                        sharded_tensor_key(parsed_shard_index, output_name),
                        parsed_operation_index).second) {
                    fail("multiple QNN producers for tensor in shard " +
                        std::to_string(shard_index) + ": " + output_name);
                }
            }
            if (parsed_operation.decoder_binding.layer_ids.size() == 1) {
                const int32_t operation_layer_id =
                    parsed_operation.decoder_binding.layer_ids.front();
                if (operation_layer_id < layer_start || operation_layer_id >= layer_end) {
                    fail("operation " + parsed_operation.name +
                        " is outside its shard layer range");
                }
                if (!result->operation_fx_index.emplace(
                        operation_fx_key(operation_layer_id, parsed_operation.fx_node_name),
                        parsed_operation_index).second) {
                    fail("duplicate decoder FX operation identity for layer " +
                        std::to_string(operation_layer_id) + ": " +
                        parsed_operation.fx_node_name);
                }
            }
            result->operations.push_back(std::move(parsed_operation));

            if (operation.value("type_name", "") != "Conv2d") {
                continue;
            }
            const auto & binding = operation.at("source").at("decoder_binding");
            if (binding.at("projection").is_null()) {
                continue;
            }
            const std::string projection = binding.at("projection").get<std::string>();
            if (!is_gptq_linear_projection(projection)) {
                continue;
            }
            const auto & layer_ids = binding.at("layer_ids");
            if (!layer_ids.is_array() || layer_ids.size() != 1) {
                fail("GPTQ Conv2d operation has an ambiguous decoder layer");
            }
            const int64_t layer_id = layer_ids[0].get<int64_t>();
            if (layer_id < layer_start || layer_id >= layer_end ||
                layer_id > std::numeric_limits<int32_t>::max()) {
                fail("GPTQ Conv2d operation is outside its shard layer range");
            }
            const auto & inputs = operation.at("inputs");
            const auto & outputs = operation.at("outputs");
            if (!inputs.is_array() || inputs.size() < 2 ||
                !outputs.is_array() || outputs.empty()) {
                fail("GPTQ Conv2d operation has malformed tensor lists");
            }
            const std::string input_name = inputs[0].get<std::string>();
            const std::string weight_name = inputs[1].get<std::string>();
            const std::string output_name = outputs[0].get<std::string>();
            const auto * input_tensor = result->find_u16_tensor(parsed_shard_index, input_name);
            const auto * output_tensor = result->find_u16_tensor(parsed_shard_index, output_name);
            const auto weight_tensor = tensors.find(weight_name);
            if (input_tensor == nullptr || output_tensor == nullptr ||
                weight_tensor == tensors.end() ||
                weight_tensor.value().value("quantization_encoding", "") !=
                    "QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION" ||
                input_tensor->qparams.encoding != LLAMA_QNN_QUANTIZATION_SCALE_OFFSET ||
                output_tensor->qparams.encoding != LLAMA_QNN_QUANTIZATION_SCALE_OFFSET ||
                input_tensor->qparams.scale_offsets.size() != 1 ||
                output_tensor->qparams.scale_offsets.size() != 1) {
                fail("GPTQ Conv2d operation lacks exact affine U16 or blockwise weight qparams");
            }
            const std::string operation_name = operation.at("name").get<std::string>();
            const llama_qnn_tensor_qparams weight_qparams = parse_qparams(
                weight_tensor.value(), "GPTQ weight " + weight_name);
            const auto & weight_dimensions = weight_tensor.value().at("dimensions");
            if (!weight_dimensions.is_array() || weight_dimensions.size() != 4 ||
                weight_qparams.axis != 3 ||
                weight_dimensions[0].get<int64_t>() != 1 ||
                weight_dimensions[1].get<int64_t>() != 1 ||
                weight_dimensions[2].get<int64_t>() <= 0 ||
                weight_dimensions[2].get<int64_t>() % 32 != 0 ||
                weight_dimensions[3].get<int64_t>() <= 0 ||
                weight_dimensions[3].get<uint64_t>() != weight_qparams.scale_offsets.size() ||
                weight_dimensions[2].get<int64_t>() / 32 !=
                    weight_qparams.num_blocks_per_axis) {
                fail("GPTQ Conv2d operation has incompatible QNN GS32 weight geometry");
            }
            const int32_t parsed_layer_id = static_cast<int32_t>(layer_id);
            const std::string key = linear_qparams_key(parsed_layer_id, projection);
            const auto existing = result->linear_qparams_index.find(key);
            if (existing == result->linear_qparams_index.end()) {
                llama_qnn_linear_qparams parsed;
                parsed.layer_id = parsed_layer_id;
                parsed.shard_index = parsed_shard_index;
                parsed.scope = scope;
                parsed.projection = projection;
                parsed.input = input_tensor->qparams.scale_offsets[0];
                parsed.output = output_tensor->qparams.scale_offsets[0];
                parsed.activation_to_output_q20 = scale_ratio_q20(
                    parsed.input, parsed.output, key);
                append_linear_blockwise_qparams(
                    parsed, weight_qparams, operation_name, "GPTQ operation " + operation_name);
                parsed.operation_names.push_back(operation_name);
                result->linear_qparams_index.emplace(key, result->linear_qparams.size());
                result->linear_qparams.push_back(std::move(parsed));
            } else {
                auto & parsed = result->linear_qparams.at(existing->second);
                if (parsed.shard_index != parsed_shard_index || parsed.scope != scope ||
                    !same_affine_qparams(parsed.input, input_tensor->qparams.scale_offsets[0]) ||
                    !same_affine_qparams(parsed.output, output_tensor->qparams.scale_offsets[0])) {
                    fail("GPTQ Conv2d heads disagree on exact U16 qparams for " + key);
                }
                append_linear_blockwise_qparams(
                    parsed, weight_qparams, operation_name, "GPTQ operation " + operation_name);
                parsed.operation_names.push_back(operation_name);
            }
        }
    }
    if (result->u16_tensors.empty()) {
        fail("profile contains no QNN U16 tensors");
    }
    const size_t expected_linear_qparams =
        static_cast<size_t>(result->num_decoder_layers) * 7;
    if (result->linear_qparams.size() != expected_linear_qparams) {
        fail("profile does not contain exactly seven GPTQ linear qparam pairs per decoder layer");
    }
    static constexpr std::array<std::string_view, 7> required_projections = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    };
    for (int32_t layer_id = 0; layer_id < result->num_decoder_layers; ++layer_id) {
        for (const auto projection : required_projections) {
            const auto * linear =
                result->find_linear_qparams(layer_id, std::string(projection));
            if (linear == nullptr) {
                fail("profile misses GPTQ linear qparams for layer " +
                    std::to_string(layer_id) + " projection " + std::string(projection));
            }
            if (linear->qnn_weight_block_size != 32 ||
                linear->qnn_weight_blocks_per_row <= 0 ||
                linear->qnn_channel_scale_to_output_q31.empty() ||
                std::find(
                    linear->qnn_channel_scale_to_output_q31.begin(),
                    linear->qnn_channel_scale_to_output_q31.end(), 0) !=
                    linear->qnn_channel_scale_to_output_q31.end() ||
                linear->qnn_weight_block_scale_codes.size() !=
                    linear->qnn_channel_scale_to_output_q31.size() *
                    static_cast<size_t>(linear->qnn_weight_blocks_per_row)) {
                fail("profile has incomplete QNN blockwise scales for layer " +
                    std::to_string(layer_id) + " projection " + std::string(projection));
            }
        }
    }
    if (!measured_sigmoid_luts.empty()) {
        fail("one or more measured Sigmoid LUTs did not match a profile operation");
    }
    const auto bias_entries = profile.find("decoder_linear_output_bias_q7");
    if (bias_entries != profile.end()) {
        if (!bias_entries->is_array()) {
            fail("decoder_linear_output_bias_q7 must be an array");
        }
        for (const auto & entry : *bias_entries) {
            const int32_t layer_id = entry.at("layer_id").get<int32_t>();
            const std::string projection = entry.at("projection").get<std::string>();
            const int32_t bias_q7 = entry.at("bias_q7").get<int32_t>();
            if (layer_id < 0 || layer_id >= result->num_decoder_layers ||
                bias_q7 < -256 || bias_q7 > 256) {
                fail("decoder linear output bias is outside the supported range");
            }
            const auto found = result->linear_qparams_index.find(
                linear_qparams_key(layer_id, projection));
            if (found == result->linear_qparams_index.end()) {
                fail("decoder linear output bias references an unknown projection");
            }
            auto & linear = result->linear_qparams.at(found->second);
            if (linear.output_bias_q7 != 0) {
                fail("duplicate decoder linear output bias");
            }
            linear.output_bias_q7 = bias_q7;
        }
    }
    return result;
}

std::shared_ptr<llama_qnn_quant_profile> llama_qnn_quant_profile_load_from_environment() {
    const char * manifest_path = std::getenv("LLAMA_QNN_U16_QPARAMS_MANIFEST");
    if (manifest_path == nullptr || manifest_path[0] == '\0') {
        return nullptr;
    }
    return llama_qnn_quant_profile_load_file(manifest_path);
}
