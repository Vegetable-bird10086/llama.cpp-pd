#include "llama-qnn-u16.h"

#include "ggml.h"
#include "ggml-cpu/quants.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

namespace {

bool is_enabled_value(const char * value) {
    if (value == nullptr) {
        return false;
    }
    const std::string_view setting(value);
    return setting == "1" || setting == "true" || setting == "TRUE" ||
        setting == "on" || setting == "ON";
}

bool use_profiled_oproj_nearest(const char * value, int32_t layer_id) {
    if (value == nullptr) {
        return false;
    }
    if (std::string_view(value) == "all") {
        return true;
    }
    if (!is_enabled_value(value)) {
        return false;
    }
    // Derived from exact-input comparisons against the matching 14-shard
    // wikitext-limit8 ETDump. Other layers match QNN better with floor.
    static constexpr std::array<int32_t, 14> nearest_layers = {
        0, 1, 2, 5, 6, 9, 10, 11, 12, 13, 16, 17, 20, 26,
    };
    return std::binary_search(
        nearest_layers.begin(), nearest_layers.end(), layer_id);
}

bool use_profiled_ffn_nearest(
        const char * value,
        int32_t layer_id,
        std::string_view projection) {
    if (value == nullptr) {
        return false;
    }
    if (std::string_view(value) == "all") {
        return true;
    }
    if (!is_enabled_value(value)) {
        return false;
    }
    // Exact-input matches against the same wikitext-limit8 ETDump. Every
    // other FFN projection is closer to QNN with the default floor shift.
    return (layer_id == 2 && projection == "mlp.up_proj") ||
        (layer_id == 11 && projection == "mlp.gate_proj") ||
        (layer_id == 27 && projection == "mlp.down_proj");
}

const llama_qnn_u16_tensor * require_u16_operand(
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation,
        const char * role,
        int32_t position) {
    GGML_ASSERT(profile != nullptr && operation != nullptr);
    const llama_qnn_u16_tensor * tensor =
        profile->find_u16_operand(*operation, role, position);
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(tensor->qparams.encoding == LLAMA_QNN_QUANTIZATION_SCALE_OFFSET);
    GGML_ASSERT(tensor->qparams.scale_offsets.size() == 1);
    return tensor;
}

int64_t qnn_u16_scale_ratio_q20(
        const llama_qnn_u16_tensor * input,
        const llama_qnn_u16_tensor * output) {
    GGML_ASSERT(input != nullptr && output != nullptr);
    const double input_scale = input->qparams.scale_offsets[0].scale;
    const double output_scale = output->qparams.scale_offsets[0].scale;
    const double scaled = std::ldexp(input_scale / output_scale, 20);
    GGML_ASSERT(std::isfinite(scaled) && scaled >= 0.0 &&
        scaled <= static_cast<double>(std::numeric_limits<int64_t>::max()));
    return static_cast<int64_t>(std::llround(scaled));
}

int64_t qnn_u16_scale_ratio_q15(
        const llama_qnn_u16_tensor * input,
        const llama_qnn_u16_tensor * output) {
    GGML_ASSERT(input != nullptr && output != nullptr);
    const double input_scale = input->qparams.scale_offsets[0].scale;
    const double output_scale = output->qparams.scale_offsets[0].scale;
    const double scaled = std::ldexp(input_scale / output_scale, 15);
    GGML_ASSERT(std::isfinite(scaled) && scaled >= 0.0 &&
        scaled <= static_cast<double>(std::numeric_limits<int64_t>::max()));
    return static_cast<int64_t>(std::llround(scaled));
}

struct qnn_u16_scalefactor_q15 {
    int32_t multiplier;
    int32_t right_shift;
};

qnn_u16_scalefactor_q15 qnn_u16_htp_scalefactor_q15(
        const llama_qnn_u16_tensor * input,
        const llama_qnn_u16_tensor * output) {
    GGML_ASSERT(input != nullptr && output != nullptr);
    const float input_scale = input->qparams.scale_offsets[0].scale;
    const float output_scale = output->qparams.scale_offsets[0].scale;
    const float ratio = input_scale / output_scale;
    GGML_ASSERT(std::isfinite(ratio) && ratio > 0.0f);

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
    const int32_t right_shift = significand_bits - exponent;
    GGML_ASSERT(multiplier > 0 && multiplier <= INT16_MAX);
    GGML_ASSERT(right_shift > 0 && right_shift < 63);
    return { multiplier, right_shift };
}

int64_t qnn_u16_scale_product_ratio_q20(
        const llama_qnn_u16_tensor * lhs,
        const llama_qnn_u16_tensor * rhs,
        const llama_qnn_u16_tensor * output) {
    GGML_ASSERT(lhs != nullptr && rhs != nullptr && output != nullptr);
    const double lhs_scale = lhs->qparams.scale_offsets[0].scale;
    const double rhs_scale = rhs->qparams.scale_offsets[0].scale;
    const double output_scale = output->qparams.scale_offsets[0].scale;
    const double scaled = std::ldexp((lhs_scale * rhs_scale) / output_scale, 20);
    GGML_ASSERT(std::isfinite(scaled) && scaled >= 0.0 &&
        scaled <= static_cast<double>(std::numeric_limits<int64_t>::max()));
    return static_cast<int64_t>(std::llround(scaled));
}

int64_t qnn_u16_scale_ratio_q31(
        const llama_qnn_u16_tensor * input,
        const llama_qnn_u16_tensor * output) {
    GGML_ASSERT(input != nullptr && output != nullptr);
    const double input_scale = input->qparams.scale_offsets[0].scale;
    const double output_scale = output->qparams.scale_offsets[0].scale;
    const double scaled = std::ldexp(input_scale / output_scale, 31);
    GGML_ASSERT(std::isfinite(scaled) && scaled >= 0.0 &&
        scaled <= static_cast<double>(std::numeric_limits<int64_t>::max()));
    return static_cast<int64_t>(std::llround(scaled));
}

int64_t qnn_u16_scale_product_ratio_q31(
        const llama_qnn_u16_tensor * lhs,
        const llama_qnn_u16_tensor * rhs,
        const llama_qnn_u16_tensor * output) {
    GGML_ASSERT(lhs != nullptr && rhs != nullptr && output != nullptr);
    const double lhs_scale = lhs->qparams.scale_offsets[0].scale;
    const double rhs_scale = rhs->qparams.scale_offsets[0].scale;
    const double output_scale = output->qparams.scale_offsets[0].scale;
    const double scaled = std::ldexp((lhs_scale * rhs_scale) / output_scale, 31);
    GGML_ASSERT(std::isfinite(scaled) && scaled >= 0.0 &&
        scaled <= static_cast<double>(std::numeric_limits<int64_t>::max()));
    return static_cast<int64_t>(std::llround(scaled));
}

int64_t qnn_u16_scale_product_ratio_q31_shift3(
        const llama_qnn_u16_tensor * lhs,
        const llama_qnn_u16_tensor * rhs,
        const llama_qnn_u16_tensor * output) {
    GGML_ASSERT(lhs != nullptr && rhs != nullptr && output != nullptr);
    const float lhs_scale = lhs->qparams.scale_offsets[0].scale;
    const float rhs_scale = rhs->qparams.scale_offsets[0].scale;
    const float output_scale = output->qparams.scale_offsets[0].scale;
    const float ratio = (lhs_scale * rhs_scale) / output_scale;
    const double scaled = std::ldexp(
        static_cast<double>(ratio * 8.0f), 31);
    GGML_ASSERT(std::isfinite(scaled) && scaled >= 0.0 &&
        scaled <= static_cast<double>(std::numeric_limits<int64_t>::max()));
    return static_cast<int64_t>(std::llround(scaled));
}

void qnn_u16_binary_params_q15(
        const llama_qnn_u16_tensor * lhs,
        const llama_qnn_u16_tensor * rhs,
        const llama_qnn_u16_tensor * output,
        bool subtract,
        int32_t * multipliers,
        int32_t * shift,
        int64_t * bias) {
    GGML_ASSERT(lhs != nullptr && rhs != nullptr && output != nullptr);
    const float output_reciprocal =
        1.0f / static_cast<float>(output->qparams.scale_offsets[0].scale);
    const float ratios[2] = {
        static_cast<float>(lhs->qparams.scale_offsets[0].scale) *
            output_reciprocal,
        static_cast<float>(rhs->qparams.scale_offsets[0].scale) *
            output_reciprocal,
    };
    const float max_ratio = std::max(ratios[0], ratios[1]);
    GGML_ASSERT(std::isfinite(max_ratio) && max_ratio > 0.0f);
    *shift = static_cast<int32_t>(
        std::floor(std::log2(32767.0f / max_ratio))) - 1;
    GGML_ASSERT(*shift > 0 && *shift < 31);
    for (int index = 0; index < 2; ++index) {
        const double scaled = std::ldexp(
            static_cast<double>(ratios[index]), *shift + 1);
        multipliers[index] = std::min(
            INT16_MAX, static_cast<int32_t>(std::llround(scaled)));
        GGML_ASSERT(multipliers[index] >= 0);
    }
    const int64_t lhs_zero_product =
        static_cast<int64_t>(lhs->qparams.scale_offsets[0].zero_point) *
        multipliers[0];
    const int64_t rhs_zero_product =
        static_cast<int64_t>(rhs->qparams.scale_offsets[0].zero_point) *
        multipliers[1];
    const int64_t zero_product =
        lhs_zero_product + (subtract ? -rhs_zero_product : rhs_zero_product);
    const int64_t zero_average = zero_product >= 0
        ? zero_product / 2
        : -((-zero_product + 1) / 2);
    *bias =
        (static_cast<int64_t>(output->qparams.scale_offsets[0].zero_point)
            << *shift) -
        zero_average;
}

struct qnn_u16_binary_params {
    int64_t lhs_multiplier;
    int64_t rhs_multiplier;
    int64_t product_requant_nudge_q31;
    int32_t lhs_zero_point;
    int32_t rhs_zero_point;
    int32_t output_zero_point;
    bool multiply;
};

struct qnn_f32_u16_boundary_params {
    float scale;
    int32_t zero_point;
};

struct qnn_u16_static_binary_params {
    qnn_u16_binary_params binary;
    uint16_t rhs_code;
    int32_t folded_multiplier_q15;
    int32_t folded_right_shift;
};

struct qnn_u16_rms_norm_params {
    const uint16_t * weight;
    int64_t weight_elements;
    int32_t input_zero_point;
    int32_t weight_zero_point;
    uint64_t epsilon_in_codes_q16;
    int64_t weight_to_output_q31;
    int32_t output_zero_point;
};

struct qnn_u16_lut_params {
    const uint16_t * lut;
};

struct qnn_u16_swiglu_params {
    const uint16_t * sigmoid_lut;
    qnn_u16_binary_params silu;
    qnn_u16_binary_params product;
};

struct qnn_u16_rope_params {
    ggml_u16_rope_qnn_fixed_params fixed;
    const uint16_t * cos_codes;
    const uint16_t * sin_codes;
    int64_t table_rows;
    int64_t half_dimension;
};

struct qnn_u16_s16_matmul_params {
    const int16_t * weights;
    int64_t input_dimension;
    int64_t output_dimension;
    int32_t input_zero_point;
    int32_t weight_zero_point;
    int64_t product_to_output_q20;
    int32_t output_zero_point;
    int32_t weight_right_shift;
    int32_t accumulator_reduction_shift;
};

struct qnn_u16_to_u8_params {
    int32_t input_zero_point;
    int64_t input_to_output_q31;
    int32_t output_zero_point;
};

struct qnn_u16_u8_matmul_params {
    int32_t input_zero_point;
    int32_t weight_zero_point;
    int64_t product_to_output_q20;
    int32_t output_zero_point;
};

struct qnn_u16_unary_requant_params {
    int32_t input_zero_point;
    int64_t input_to_output_q20;
    int32_t output_zero_point;
};

struct qnn_u16_select_params {
    int32_t true_zero_point;
    int32_t true_multiplier_q15;
    int32_t true_right_shift;
    int32_t false_zero_point;
    int32_t false_multiplier_q15;
    int32_t false_right_shift;
    int32_t output_zero_point;
};

struct qnn_u16_softmax_params {
    int64_t scale_over_ln2_q24;
    int64_t output_unit_code;
    int32_t output_zero_point;
    const uint32_t * exp2_lut_q31;
};

struct qnn_u16_attention_softmax_params {
    qnn_u16_unary_requant_params divide;
    qnn_u16_unary_requant_params minimum;
    qnn_u16_static_binary_params floor_add;
    qnn_u16_select_params select;
    qnn_u16_softmax_params softmax;
};

struct qnn_u16_attention_head_params {
    qnn_u16_u8_matmul_params score;
    qnn_u16_attention_softmax_params softmax;
    qnn_u16_u8_matmul_params value;
};

struct qnn_u16_attention_params {
    const qnn_u16_attention_head_params * heads;
    int32_t n_head;
    int32_t n_head_kv;
};

template <typename T>
struct qnn_u16_head_params {
    const T * values;
    int32_t n_head;
};

template <typename T>
T * new_qnn_u16_params(ggml_context * ctx, const T & value) {
    auto * params = static_cast<T *>(ggml_new_buffer(ctx, sizeof(T)));
    *params = value;
    return params;
}

int64_t qnn_u16_rows(const ggml_tensor * tensor) {
    return tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

const uint16_t * qnn_u16_row(const ggml_tensor * tensor, int64_t row) {
    const int64_t i1 = row % tensor->ne[1];
    const int64_t i2 = (row / tensor->ne[1]) % tensor->ne[2];
    const int64_t i3 = row / (tensor->ne[1] * tensor->ne[2]);
    return reinterpret_cast<const uint16_t *>(
        reinterpret_cast<const char *>(tensor->data) +
        i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3]);
}

uint16_t * qnn_u16_row(ggml_tensor * tensor, int64_t row) {
    const int64_t i1 = row % tensor->ne[1];
    const int64_t i2 = (row / tensor->ne[1]) % tensor->ne[2];
    const int64_t i3 = row / (tensor->ne[1] * tensor->ne[2]);
    return reinterpret_cast<uint16_t *>(
        reinterpret_cast<char *>(tensor->data) +
        i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3]);
}

const uint8_t * qnn_u8_row(const ggml_tensor * tensor, int64_t row) {
    const int64_t i1 = row % tensor->ne[1];
    const int64_t i2 = (row / tensor->ne[1]) % tensor->ne[2];
    const int64_t i3 = row / (tensor->ne[1] * tensor->ne[2]);
    return reinterpret_cast<const uint8_t *>(
        reinterpret_cast<const char *>(tensor->data) +
        i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3]);
}

void qnn_u16_binary_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params = static_cast<const qnn_u16_binary_params *>(userdata);
    const ggml_tensor * lhs = dst->src[0];
    const ggml_tensor * rhs = dst->src[1];
    GGML_ASSERT(params != nullptr && lhs != nullptr && rhs != nullptr);
    GGML_ASSERT(lhs->type == GGML_TYPE_U16 && rhs->type == GGML_TYPE_U16);
    GGML_ASSERT(dst->type == GGML_TYPE_U16 && ggml_are_same_shape(lhs, rhs));
    GGML_ASSERT(ggml_are_same_shape(lhs, dst));
    GGML_ASSERT(lhs->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(rhs->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(dst->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(lhs->ne[0] <= std::numeric_limits<int>::max());

    const int n = static_cast<int>(lhs->ne[0]);
    const int64_t rows = qnn_u16_rows(lhs);
    for (int64_t row = ith; row < rows; row += nth) {
        const uint16_t * lhs_data = qnn_u16_row(lhs, row);
        const uint16_t * rhs_data = qnn_u16_row(rhs, row);
        uint16_t * output = qnn_u16_row(dst, row);
        if (params->multiply) {
            ggml_vec_mul_affine_u16_qnn_q31(
                n, output,
                lhs_data, params->lhs_zero_point,
                rhs_data, params->rhs_zero_point,
                params->lhs_multiplier, params->product_requant_nudge_q31,
                params->output_zero_point);
        } else {
            ggml_vec_add_affine_u16_qnn_q15(
                n, output,
                lhs_data, params->lhs_multiplier, params->lhs_zero_point,
                rhs_data, params->rhs_multiplier, params->rhs_zero_point,
                params->output_zero_point);
        }
    }
}

void qnn_quantize_f32_u16_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params = static_cast<const qnn_f32_u16_boundary_params *>(userdata);
    const ggml_tensor * input = dst->src[0];
    GGML_ASSERT(params != nullptr && input != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_U16);
    GGML_ASSERT(ggml_are_same_shape(input, dst));
    GGML_ASSERT(params->scale > 0.0f);
    const int64_t rows = qnn_u16_rows(dst);
    for (int64_t row = ith; row < rows; row += nth) {
        const int64_t i1 = row % input->ne[1];
        const int64_t i2 = (row / input->ne[1]) % input->ne[2];
        const int64_t i3 = row / (input->ne[1] * input->ne[2]);
        const auto * source = reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(input->data) +
            i1*input->nb[1] + i2*input->nb[2] + i3*input->nb[3]);
        uint16_t * output = qnn_u16_row(dst, row);
        for (int64_t index = 0; index < input->ne[0]; ++index) {
            const int64_t code = static_cast<int64_t>(std::llround(
                static_cast<double>(source[index]) / params->scale)) +
                params->zero_point;
            output[index] = static_cast<uint16_t>(
                std::clamp<int64_t>(code, 0, UINT16_MAX));
        }
    }
}

void qnn_dequantize_u16_f32_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params = static_cast<const qnn_f32_u16_boundary_params *>(userdata);
    const ggml_tensor * input = dst->src[0];
    GGML_ASSERT(params != nullptr && input != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(input, dst));
    const int64_t rows = qnn_u16_rows(input);
    for (int64_t row = ith; row < rows; row += nth) {
        const int64_t i1 = row % dst->ne[1];
        const int64_t i2 = (row / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = row / (dst->ne[1] * dst->ne[2]);
        auto * output = reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst->data) +
            i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
        const uint16_t * source = qnn_u16_row(input, row);
        for (int64_t index = 0; index < input->ne[0]; ++index) {
            output[index] = (static_cast<int32_t>(source[index]) -
                params->zero_point) * params->scale;
        }
    }
}

void qnn_u16_static_binary_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params = static_cast<const qnn_u16_static_binary_params *>(userdata);
    const ggml_tensor * lhs = dst->src[0];
    GGML_ASSERT(params != nullptr && lhs != nullptr);
    GGML_ASSERT(lhs->type == GGML_TYPE_U16 && dst->type == GGML_TYPE_U16);
    GGML_ASSERT(ggml_are_same_shape(lhs, dst));
    const int n = static_cast<int>(lhs->ne[0]);
    const int64_t rows = qnn_u16_rows(lhs);
    for (int64_t row = ith; row < rows; row += nth) {
        if (params->binary.multiply) {
            ggml_vec_mul_static_affine_u16_qnn_q15(
                n, qnn_u16_row(dst, row), qnn_u16_row(lhs, row),
                params->binary.lhs_zero_point,
                params->folded_multiplier_q15,
                params->folded_right_shift,
                params->binary.output_zero_point);
        } else {
            ggml_vec_add_affine_u16_qnn_fixed_scalar(
                n, qnn_u16_row(dst, row), qnn_u16_row(lhs, row),
                params->binary.lhs_multiplier,
                params->binary.lhs_zero_point, params->rhs_code,
                params->binary.rhs_multiplier,
                params->binary.rhs_zero_point,
                params->binary.output_zero_point);
        }
    }
}

void qnn_attention_mask_condition_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void *) {
    const ggml_tensor * mask = dst->src[0];
    GGML_ASSERT(mask != nullptr && dst->type == GGML_TYPE_I8);
    GGML_ASSERT(mask->type == GGML_TYPE_F32 || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_are_same_shape(mask, dst));
    const int64_t rows = qnn_u16_rows(dst);
    for (int64_t row = ith; row < rows; row += nth) {
        const int64_t i1 = row % mask->ne[1];
        const int64_t i2 = (row / mask->ne[1]) % mask->ne[2];
        const int64_t i3 = row / (mask->ne[1] * mask->ne[2]);
        const char * source = reinterpret_cast<const char *>(mask->data) +
            i1*mask->nb[1] + i2*mask->nb[2] + i3*mask->nb[3];
        uint8_t * output = reinterpret_cast<uint8_t *>(dst->data) +
            i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];
        for (int64_t index = 0; index < mask->ne[0]; ++index) {
            const float value = mask->type == GGML_TYPE_F32
                ? reinterpret_cast<const float *>(source)[index]
                : ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(source)[index]);
            output[index] = value >= 0.0f ? 1 : 0;
        }
    }
}

void qnn_u16_rms_norm_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params = static_cast<const qnn_u16_rms_norm_params *>(userdata);
    const ggml_tensor * input = dst->src[0];
    GGML_ASSERT(params != nullptr && input != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16 && dst->type == GGML_TYPE_U16);
    GGML_ASSERT(ggml_are_same_shape(input, dst));
    GGML_ASSERT(input->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(dst->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(input->ne[0] == params->weight_elements);
    GGML_ASSERT(input->ne[0] <= std::numeric_limits<int>::max());

    const int n = static_cast<int>(input->ne[0]);
    const int64_t rows = qnn_u16_rows(input);
    for (int64_t row = ith; row < rows; row += nth) {
        ggml_vec_rms_norm_affine_u16_qnn_fixed(
            n, qnn_u16_row(dst, row), qnn_u16_row(input, row),
            params->input_zero_point,
            params->weight, params->weight_zero_point,
            params->epsilon_in_codes_q16,
            params->weight_to_output_q31,
            params->output_zero_point);
    }
}

void qnn_u16_rms_norm_heads_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * head_params =
        static_cast<const qnn_u16_head_params<qnn_u16_rms_norm_params> *>(userdata);
    const ggml_tensor * input = dst->src[0];
    GGML_ASSERT(head_params != nullptr && head_params->values != nullptr);
    GGML_ASSERT(input != nullptr && input->type == GGML_TYPE_U16);
    GGML_ASSERT(dst->type == GGML_TYPE_U16 && ggml_are_same_shape(input, dst));
    GGML_ASSERT(input->ne[1] == head_params->n_head);
    const int n = static_cast<int>(input->ne[0]);
    const int64_t rows = qnn_u16_rows(input);
    for (int64_t row = ith; row < rows; row += nth) {
        const auto & params = head_params->values[row % head_params->n_head];
        GGML_ASSERT(input->ne[0] == params.weight_elements);
        ggml_vec_rms_norm_affine_u16_qnn_fixed(
            n, qnn_u16_row(dst, row), qnn_u16_row(input, row),
            params.input_zero_point,
            params.weight, params.weight_zero_point,
            params.epsilon_in_codes_q16,
            params.weight_to_output_q31,
            params.output_zero_point);
    }
}

void qnn_u16_lut_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params = static_cast<const qnn_u16_lut_params *>(userdata);
    const ggml_tensor * input = dst->src[0];
    GGML_ASSERT(params != nullptr && params->lut != nullptr && input != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16 && dst->type == GGML_TYPE_U16);
    GGML_ASSERT(ggml_are_same_shape(input, dst));
    GGML_ASSERT(input->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(dst->nb[0] == sizeof(uint16_t));

    const int64_t rows = qnn_u16_rows(input);
    for (int64_t row = ith; row < rows; row += nth) {
        const uint16_t * source = qnn_u16_row(input, row);
        uint16_t * output = qnn_u16_row(dst, row);
        int64_t index = 0;
        for (; index + 8 <= input->ne[0]; index += 8) {
            output[index + 0] = params->lut[source[index + 0]];
            output[index + 1] = params->lut[source[index + 1]];
            output[index + 2] = params->lut[source[index + 2]];
            output[index + 3] = params->lut[source[index + 3]];
            output[index + 4] = params->lut[source[index + 4]];
            output[index + 5] = params->lut[source[index + 5]];
            output[index + 6] = params->lut[source[index + 6]];
            output[index + 7] = params->lut[source[index + 7]];
        }
        for (; index < input->ne[0]; ++index) {
            output[index] = params->lut[source[index]];
        }
    }
}

void qnn_u16_swiglu_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params = static_cast<const qnn_u16_swiglu_params *>(userdata);
    const ggml_tensor * gate = dst->src[0];
    const ggml_tensor * up = dst->src[1];
    GGML_ASSERT(params != nullptr && params->sigmoid_lut != nullptr);
    GGML_ASSERT(gate != nullptr && up != nullptr);
    GGML_ASSERT(gate->type == GGML_TYPE_U16 && up->type == GGML_TYPE_U16);
    GGML_ASSERT(dst->type == GGML_TYPE_U16 && ggml_are_same_shape(gate, up));
    GGML_ASSERT(ggml_are_same_shape(gate, dst));
    GGML_ASSERT(gate->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(up->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(dst->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(gate->ne[0] <= std::numeric_limits<int>::max());
    GGML_ASSERT(params->silu.multiply && params->product.multiply);

    const int n = static_cast<int>(gate->ne[0]);
    const int64_t rows = qnn_u16_rows(gate);
    for (int64_t row = ith; row < rows; row += nth) {
        ggml_vec_swiglu_u16_qnn_q31(
            n, qnn_u16_row(dst, row),
            qnn_u16_row(gate, row), qnn_u16_row(up, row),
            params->sigmoid_lut,
            params->silu.lhs_zero_point,
            params->silu.rhs_zero_point,
            params->silu.lhs_multiplier,
            params->silu.product_requant_nudge_q31,
            params->silu.output_zero_point,
            params->product.lhs_zero_point,
            params->product.rhs_zero_point,
            params->product.lhs_multiplier,
            params->product.product_requant_nudge_q31,
            params->product.output_zero_point);
    }
}

void qnn_u16_rope_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params = static_cast<const qnn_u16_rope_params *>(userdata);
    const ggml_tensor * input = dst->src[0];
    const ggml_tensor * positions = dst->src[1];
    GGML_ASSERT(params != nullptr && input != nullptr && positions != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16 && dst->type == GGML_TYPE_U16);
    GGML_ASSERT(positions->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(input, dst));
    GGML_ASSERT(input->ne[0] == params->half_dimension * 2);
    GGML_ASSERT(input->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(dst->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(positions->nb[0] == sizeof(int32_t));
    GGML_ASSERT(positions->ne[0] >= input->ne[1]);

    const int64_t rows = qnn_u16_rows(input);
    const auto * position_data = static_cast<const int32_t *>(positions->data);
    for (int64_t row = ith; row < rows; row += nth) {
        const int64_t token = row % input->ne[1];
        const int32_t position = position_data[token];
        GGML_ASSERT(position >= 0 && position < params->table_rows);
        const uint16_t * cos_codes =
            params->cos_codes + (int64_t) position * params->half_dimension;
        const uint16_t * sin_codes =
            params->sin_codes + (int64_t) position * params->half_dimension;
        ggml_vec_rope_affine_u16_qnn_fixed(
            static_cast<int>(params->half_dimension),
            qnn_u16_row(dst, row),
            qnn_u16_row(input, row),
            cos_codes,
            sin_codes,
            &params->fixed);
    }
}

void qnn_u16_rope_heads_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * head_params =
        static_cast<const qnn_u16_head_params<qnn_u16_rope_params> *>(userdata);
    const ggml_tensor * input = dst->src[0];
    const ggml_tensor * positions = dst->src[1];
    GGML_ASSERT(head_params != nullptr && head_params->values != nullptr);
    GGML_ASSERT(input != nullptr && positions != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16 && dst->type == GGML_TYPE_U16);
    GGML_ASSERT(positions->type == GGML_TYPE_I32);
    GGML_ASSERT(input->ne[1] == head_params->n_head);
    GGML_ASSERT(positions->ne[0] >= input->ne[2]);
    const int64_t rows = qnn_u16_rows(input);
    const auto * position_data = static_cast<const int32_t *>(positions->data);
    for (int64_t row = ith; row < rows; row += nth) {
        const int32_t head = row % head_params->n_head;
        const int64_t token = (row / head_params->n_head) % input->ne[2];
        const auto & params = head_params->values[head];
        const int32_t position = position_data[token];
        GGML_ASSERT(input->ne[0] == params.half_dimension * 2);
        GGML_ASSERT(position >= 0 && position < params.table_rows);
        ggml_vec_rope_affine_u16_qnn_fixed(
            static_cast<int>(params.half_dimension),
            qnn_u16_row(dst, row), qnn_u16_row(input, row),
            params.cos_codes + (int64_t) position * params.half_dimension,
            params.sin_codes + (int64_t) position * params.half_dimension,
            &params.fixed);
    }
}

void qnn_u16_s16_matmul_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params =
        static_cast<const qnn_u16_s16_matmul_params *>(userdata);
    const ggml_tensor * input = dst->src[0];
    GGML_ASSERT(params != nullptr && params->weights != nullptr && input != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16 && dst->type == GGML_TYPE_U16);
    GGML_ASSERT(input->ne[0] == params->input_dimension);
    GGML_ASSERT(dst->ne[0] == params->output_dimension);
    GGML_ASSERT(input->ne[1] == dst->ne[1] &&
        input->ne[2] == dst->ne[2] && input->ne[3] == dst->ne[3]);
    GGML_ASSERT(input->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(dst->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(params->input_dimension <= std::numeric_limits<int>::max());
    GGML_ASSERT(params->output_dimension <= std::numeric_limits<int>::max());

    const int64_t rows = qnn_u16_rows(input);
    for (int64_t row = ith; row < rows; row += nth) {
        ggml_vec_matmul_u16_s16_qnn_fixed(
            static_cast<int>(params->input_dimension),
            static_cast<int>(params->output_dimension),
            qnn_u16_row(dst, row),
            qnn_u16_row(input, row),
            params->weights,
            params->input_zero_point,
            params->weight_zero_point,
            params->product_to_output_q20,
            params->output_zero_point,
            params->weight_right_shift,
            params->accumulator_reduction_shift);
    }
}

void qnn_u16_s16_matmul_heads_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * head_params =
        static_cast<const qnn_u16_head_params<qnn_u16_s16_matmul_params> *>(userdata);
    const ggml_tensor * input = dst->src[0];
    GGML_ASSERT(head_params != nullptr && head_params->values != nullptr);
    GGML_ASSERT(input != nullptr && input->type == GGML_TYPE_U16);
    GGML_ASSERT(dst->type == GGML_TYPE_U16 && input->ne[1] == head_params->n_head);
    const int64_t rows = qnn_u16_rows(input);
    for (int64_t row = ith; row < rows; row += nth) {
        const auto & params = head_params->values[row % head_params->n_head];
        ggml_vec_matmul_u16_s16_qnn_fixed(
            static_cast<int>(params.input_dimension),
            static_cast<int>(params.output_dimension),
            qnn_u16_row(dst, row), qnn_u16_row(input, row),
            params.weights, params.input_zero_point, params.weight_zero_point,
            params.product_to_output_q20, params.output_zero_point,
            params.weight_right_shift, params.accumulator_reduction_shift);
    }
}

void qnn_u16_to_u8_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params = static_cast<const qnn_u16_to_u8_params *>(userdata);
    const ggml_tensor * input = dst->src[0];
    GGML_ASSERT(params != nullptr && input != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16 && dst->type == GGML_TYPE_I8);
    GGML_ASSERT(input->ne[0] == dst->ne[0] &&
        input->ne[1] == dst->ne[1] && input->ne[2] == dst->ne[2] &&
        input->ne[3] == dst->ne[3]);
    GGML_ASSERT(input->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(dst->nb[0] == sizeof(uint8_t));
    GGML_ASSERT(input->ne[0] <= std::numeric_limits<int>::max());

    const int n = static_cast<int>(input->ne[0]);
    const int64_t rows = qnn_u16_rows(input);
    for (int64_t row = ith; row < rows; row += nth) {
        const int64_t i1 = row % dst->ne[1];
        const int64_t i2 = (row / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = row / (dst->ne[1] * dst->ne[2]);
        auto * output = reinterpret_cast<uint8_t *>(
            reinterpret_cast<char *>(dst->data) +
            i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
        ggml_vec_convert_u16_u8_qnn_fixed(
            n, output, qnn_u16_row(input, row),
            params->input_zero_point,
            params->input_to_output_q31,
            params->output_zero_point);
    }
}

void qnn_u16_to_u8_heads_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * head_params =
        static_cast<const qnn_u16_head_params<qnn_u16_to_u8_params> *>(userdata);
    const ggml_tensor * input = dst->src[0];
    GGML_ASSERT(head_params != nullptr && head_params->values != nullptr);
    GGML_ASSERT(input != nullptr && input->type == GGML_TYPE_U16);
    GGML_ASSERT(dst->type == GGML_TYPE_I8 && input->ne[1] == head_params->n_head);
    const int n = static_cast<int>(input->ne[0]);
    const int64_t rows = qnn_u16_rows(input);
    for (int64_t row = ith; row < rows; row += nth) {
        const auto & params = head_params->values[row % head_params->n_head];
        const int64_t i1 = row % dst->ne[1];
        const int64_t i2 = (row / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = row / (dst->ne[1] * dst->ne[2]);
        auto * output = reinterpret_cast<uint8_t *>(
            reinterpret_cast<char *>(dst->data) +
            i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
        ggml_vec_convert_u16_u8_qnn_fixed(
            n, output, qnn_u16_row(input, row),
            params.input_zero_point, params.input_to_output_q31,
            params.output_zero_point);
    }
}

void qnn_u16_u8_matmul_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params = static_cast<const qnn_u16_u8_matmul_params *>(userdata);
    const ggml_tensor * input = dst->src[0];
    const ggml_tensor * matrix = dst->src[1];
    GGML_ASSERT(params != nullptr && input != nullptr && matrix != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16 && matrix->type == GGML_TYPE_I8);
    GGML_ASSERT(dst->type == GGML_TYPE_U16);
    GGML_ASSERT(input->ne[0] == matrix->ne[1]);
    GGML_ASSERT(dst->ne[0] == matrix->ne[0]);
    GGML_ASSERT(dst->ne[1] == input->ne[1] &&
        dst->ne[2] == input->ne[2] && dst->ne[3] == input->ne[3]);
    GGML_ASSERT(matrix->ne[2] == 1 || matrix->ne[2] == input->ne[2]);
    GGML_ASSERT(matrix->ne[3] == 1 || matrix->ne[3] == input->ne[3]);
    GGML_ASSERT(input->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(matrix->nb[0] == sizeof(uint8_t));
    GGML_ASSERT(matrix->nb[1] >= static_cast<size_t>(matrix->ne[0]));
    GGML_ASSERT(dst->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(input->ne[0] <= std::numeric_limits<int>::max());
    GGML_ASSERT(matrix->ne[0] <= std::numeric_limits<int>::max());

    const int64_t rows = qnn_u16_rows(input);
    for (int64_t row = ith; row < rows; row += nth) {
        const int64_t i2 = (row / input->ne[1]) % input->ne[2];
        const int64_t i3 = row / (input->ne[1] * input->ne[2]);
        const int64_t matrix_i2 = matrix->ne[2] == 1 ? 0 : i2;
        const int64_t matrix_i3 = matrix->ne[3] == 1 ? 0 : i3;
        const auto * weights = reinterpret_cast<const uint8_t *>(
            reinterpret_cast<const char *>(matrix->data) +
            matrix_i2 * matrix->nb[2] + matrix_i3 * matrix->nb[3]);
        ggml_vec_matmul_u16_u8_qnn_fixed_strided(
            static_cast<int>(input->ne[0]),
            static_cast<int>(matrix->ne[0]),
            matrix->nb[1],
            qnn_u16_row(dst, row), qnn_u16_row(input, row), weights,
            params->input_zero_point, params->weight_zero_point,
            params->product_to_output_q20, params->output_zero_point);
    }
}

void qnn_u16_unary_requant_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params = static_cast<const qnn_u16_unary_requant_params *>(userdata);
    const ggml_tensor * input = dst->src[0];
    GGML_ASSERT(params != nullptr && input != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16 && dst->type == GGML_TYPE_U16);
    GGML_ASSERT(ggml_are_same_shape(input, dst));
    const int64_t rows = qnn_u16_rows(input);
    for (int64_t row = ith; row < rows; row += nth) {
        ggml_vec_requant_u16_qnn_fixed(
            static_cast<int>(input->ne[0]), qnn_u16_row(dst, row),
            qnn_u16_row(input, row), params->input_zero_point,
            params->input_to_output_q20, params->output_zero_point);
    }
}

void qnn_u16_reduce_min_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params = static_cast<const qnn_u16_unary_requant_params *>(userdata);
    const ggml_tensor * input = dst->src[0];
    GGML_ASSERT(params != nullptr && input != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16 && dst->type == GGML_TYPE_U16);
    GGML_ASSERT(dst->ne[0] == 1 && dst->ne[1] == input->ne[1] &&
        dst->ne[2] == input->ne[2] && dst->ne[3] == input->ne[3]);
    const int64_t rows = qnn_u16_rows(input);
    for (int64_t row = ith; row < rows; row += nth) {
        const uint16_t * source = qnn_u16_row(input, row);
        uint16_t minimum = source[0];
        for (int64_t index = 1; index < input->ne[0]; ++index) {
            minimum = std::min(minimum, source[index]);
        }
        ggml_vec_requant_u16_qnn_fixed(
            1, qnn_u16_row(dst, row), &minimum,
            params->input_zero_point, params->input_to_output_q20,
            params->output_zero_point);
    }
}

void qnn_u16_select_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params = static_cast<const qnn_u16_select_params *>(userdata);
    const ggml_tensor * condition = dst->src[0];
    const ggml_tensor * when_true = dst->src[1];
    const ggml_tensor * when_false = dst->src[2];
    GGML_ASSERT(params != nullptr && condition != nullptr);
    GGML_ASSERT(when_true != nullptr && when_false != nullptr);
    GGML_ASSERT(condition->type == GGML_TYPE_I8);
    GGML_ASSERT(when_true->type == GGML_TYPE_U16 &&
        when_false->type == GGML_TYPE_U16 && dst->type == GGML_TYPE_U16);
    GGML_ASSERT(ggml_are_same_shape(condition, when_true));
    GGML_ASSERT(ggml_are_same_shape(when_true, dst));
    GGML_ASSERT(when_false->ne[0] == 1 || when_false->ne[0] == when_true->ne[0]);
    GGML_ASSERT(when_false->ne[1] == when_true->ne[1] &&
        when_false->ne[2] == when_true->ne[2] &&
        when_false->ne[3] == when_true->ne[3]);
    const int64_t rows = qnn_u16_rows(when_true);
    for (int64_t row = ith; row < rows; row += nth) {
        ggml_vec_select_affine_u16_qnn_fixed(
            static_cast<int>(when_true->ne[0]), qnn_u16_row(dst, row),
            qnn_u8_row(condition, row), qnn_u16_row(when_true, row),
            params->true_zero_point, params->true_multiplier_q15,
            params->true_right_shift,
            qnn_u16_row(when_false, row), when_false->ne[0] == 1 ? 0 : 1,
            params->false_zero_point, params->false_multiplier_q15,
            params->false_right_shift,
            params->output_zero_point);
    }
}

void qnn_u16_softmax_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params = static_cast<const qnn_u16_softmax_params *>(userdata);
    const ggml_tensor * input = dst->src[0];
    GGML_ASSERT(params != nullptr && params->exp2_lut_q31 != nullptr);
    GGML_ASSERT(input != nullptr && input->type == GGML_TYPE_U16);
    GGML_ASSERT(dst->type == GGML_TYPE_U16 && ggml_are_same_shape(input, dst));
    const int64_t rows = qnn_u16_rows(input);
    for (int64_t row = ith; row < rows; row += nth) {
        ggml_vec_softmax_u16_qnn_fixed(
            static_cast<int>(input->ne[0]), qnn_u16_row(dst, row),
            qnn_u16_row(input, row), params->scale_over_ln2_q24,
            params->output_unit_code, params->output_zero_point,
            params->exp2_lut_q31);
    }
}

void qnn_u16_apply_attention_softmax(
        int n,
        uint16_t * output,
        const uint16_t * score,
        const uint8_t * condition,
        const qnn_u16_attention_softmax_params & params) {
    ggml_vec_requant_u16_qnn_fixed(
        n, output, score,
        params.divide.input_zero_point,
        params.divide.input_to_output_q20,
        params.divide.output_zero_point);
    const uint16_t minimum = ggml_vec_min_u16_qnn(n, output);
    uint16_t requantized_minimum;
    ggml_vec_requant_u16_qnn_fixed(
        1, &requantized_minimum, &minimum,
        params.minimum.input_zero_point,
        params.minimum.input_to_output_q20,
        params.minimum.output_zero_point);
    uint16_t mask_floor;
    ggml_vec_add_affine_u16_qnn_fixed_scalar(
        1, &mask_floor, &requantized_minimum,
        params.floor_add.binary.lhs_multiplier,
        params.floor_add.binary.lhs_zero_point,
        params.floor_add.rhs_code,
        params.floor_add.binary.rhs_multiplier,
        params.floor_add.binary.rhs_zero_point,
        params.floor_add.binary.output_zero_point);
    ggml_vec_select_affine_u16_qnn_fixed(
        n, output, condition, output,
        params.select.true_zero_point,
        params.select.true_multiplier_q15,
        params.select.true_right_shift,
        &mask_floor, 0,
        params.select.false_zero_point,
        params.select.false_multiplier_q15,
        params.select.false_right_shift,
        params.select.output_zero_point);
    ggml_vec_softmax_u16_qnn_fixed(
        n, output, output,
        params.softmax.scale_over_ln2_q24,
        params.softmax.output_unit_code,
        params.softmax.output_zero_point,
        params.softmax.exp2_lut_q31);
}

void qnn_u16_attention_softmax_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params =
        static_cast<const qnn_u16_attention_softmax_params *>(userdata);
    const ggml_tensor * score = dst->src[0];
    const ggml_tensor * condition = dst->src[1];
    GGML_ASSERT(params != nullptr && score != nullptr && condition != nullptr);
    GGML_ASSERT(score->type == GGML_TYPE_U16 &&
        condition->type == GGML_TYPE_I8 && dst->type == GGML_TYPE_U16);
    GGML_ASSERT(ggml_are_same_shape(score, condition));
    GGML_ASSERT(ggml_are_same_shape(score, dst));
    GGML_ASSERT(score->ne[0] <= std::numeric_limits<int>::max());

    const int n = static_cast<int>(score->ne[0]);
    const int64_t rows = qnn_u16_rows(score);
    for (int64_t row = ith; row < rows; row += nth) {
        qnn_u16_apply_attention_softmax(
            n, qnn_u16_row(dst, row), qnn_u16_row(score, row),
            qnn_u8_row(condition, row), *params);
    }
}

void qnn_u16_attention_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * params = static_cast<const qnn_u16_attention_params *>(userdata);
    const ggml_tensor * query = dst->src[0];
    const ggml_tensor * key = dst->src[1];
    const ggml_tensor * value = dst->src[2];
    const ggml_tensor * condition = dst->src[3];
    GGML_ASSERT(params != nullptr && params->heads != nullptr);
    GGML_ASSERT(query != nullptr && key != nullptr && value != nullptr && condition != nullptr);
    GGML_ASSERT(query->type == GGML_TYPE_U16 && key->type == GGML_TYPE_I8);
    GGML_ASSERT(value->type == GGML_TYPE_I8 && condition->type == GGML_TYPE_I8);
    GGML_ASSERT(dst->type == GGML_TYPE_U16);
    GGML_ASSERT(query->ne[0] == dst->ne[0] && query->ne[1] == params->n_head);
    GGML_ASSERT(dst->ne[1] == params->n_head && query->ne[2] == dst->ne[2]);
    GGML_ASSERT(key->ne[0] == query->ne[0] && key->ne[2] == params->n_head_kv);
    GGML_ASSERT(value->ne[0] == query->ne[0] && value->ne[1] == params->n_head_kv);
    GGML_ASSERT(value->ne[2] == key->ne[1]);
    GGML_ASSERT(condition->ne[0] == key->ne[1]);
    GGML_ASSERT(condition->ne[1] == query->ne[2]);
    GGML_ASSERT(params->n_head % params->n_head_kv == 0);
    GGML_ASSERT(query->ne[0] <= std::numeric_limits<int>::max());
    GGML_ASSERT(key->ne[1] <= std::numeric_limits<int>::max());

    const int head_dimension = static_cast<int>(query->ne[0]);
    const int n_kv = static_cast<int>(key->ne[1]);
    const int queries_per_kv = params->n_head / params->n_head_kv;
    const int64_t groups = query->ne[2] * params->n_head_kv;
    thread_local std::vector<uint16_t> score0;
    thread_local std::vector<uint16_t> score1;
    score0.resize(n_kv);
    score1.resize(n_kv);

    for (int64_t group = ith; group < groups; group += nth) {
        const int64_t token = group / params->n_head_kv;
        const int32_t kv_head = group % params->n_head_kv;
        const auto * key_head = reinterpret_cast<const uint8_t *>(
            reinterpret_cast<const char *>(key->data) + kv_head * key->nb[2]);
        const auto * value_head = reinterpret_cast<const uint8_t *>(
            reinterpret_cast<const char *>(value->data) + kv_head * value->nb[1]);
        const auto * condition_row = qnn_u8_row(condition, token);

        if (queries_per_kv == 2) {
            const int32_t head0 = kv_head * 2;
            const int32_t head1 = head0 + 1;
            const int64_t row0 = token * params->n_head + head0;
            const int64_t row1 = row0 + 1;
            const auto & hp0 = params->heads[head0];
            const auto & hp1 = params->heads[head1];
            if (hp0.score.weight_zero_point == hp1.score.weight_zero_point &&
                    hp0.value.weight_zero_point == hp1.value.weight_zero_point) {
                ggml_vec_matmul_u16_u8_qnn_fixed_token_major_pair(
                    head_dimension, n_kv, key->nb[1],
                    score0.data(), score1.data(),
                    qnn_u16_row(query, row0), qnn_u16_row(query, row1),
                    key_head,
                    hp0.score.input_zero_point,
                    hp1.score.input_zero_point,
                    hp0.score.weight_zero_point,
                    hp0.score.product_to_output_q20,
                    hp1.score.product_to_output_q20,
                    hp0.score.output_zero_point,
                    hp1.score.output_zero_point);

                qnn_u16_apply_attention_softmax(
                    n_kv, score0.data(), score0.data(), condition_row,
                    hp0.softmax);
                qnn_u16_apply_attention_softmax(
                    n_kv, score1.data(), score1.data(), condition_row,
                    hp1.softmax);

                ggml_vec_matmul_u16_u8_qnn_fixed_strided_pair(
                    n_kv, head_dimension, value->nb[2],
                    qnn_u16_row(dst, row0), qnn_u16_row(dst, row1),
                    score0.data(), score1.data(), value_head,
                    hp0.value.input_zero_point,
                    hp1.value.input_zero_point,
                    hp0.value.weight_zero_point,
                    hp0.value.product_to_output_q20,
                    hp1.value.product_to_output_q20,
                    hp0.value.output_zero_point,
                    hp1.value.output_zero_point);
                continue;
            }
        }

        for (int32_t offset = 0; offset < queries_per_kv; ++offset) {
            const int32_t head = kv_head * queries_per_kv + offset;
            const int64_t row = token * params->n_head + head;
            const auto & hp = params->heads[head];

            ggml_vec_matmul_u16_u8_qnn_fixed_token_major(
                head_dimension, n_kv, key->nb[1],
                score0.data(), qnn_u16_row(query, row), key_head,
                hp.score.input_zero_point,
                hp.score.weight_zero_point,
                hp.score.product_to_output_q20,
                hp.score.output_zero_point);

            // Preserve the exact exported QNN fixed-point boundaries while
            // keeping the score row thread-local instead of a graph tensor.
            qnn_u16_apply_attention_softmax(
                n_kv, score0.data(), score0.data(), condition_row, hp.softmax);

            ggml_vec_matmul_u16_u8_qnn_fixed_strided(
                n_kv, head_dimension, value->nb[2],
                qnn_u16_row(dst, row), score0.data(), value_head,
                hp.value.input_zero_point,
                hp.value.weight_zero_point,
                hp.value.product_to_output_q20,
                hp.value.output_zero_point);
        }
    }
}

void qnn_u16_mul_mat_compute(
        ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    const auto * qparams = static_cast<const llama_qnn_linear_qparams *>(userdata);
    const ggml_tensor * input = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(qparams != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16 && dst->type == GGML_TYPE_U16);
    GGML_ASSERT(weights->type == GGML_TYPE_GPTQ2_32 ||
                weights->type == GGML_TYPE_GPTQ2_64 ||
                weights->type == GGML_TYPE_GPTQ2_128);
    GGML_ASSERT(weights->ne[0] == input->ne[0]);
    GGML_ASSERT(weights->ne[2] == 1 && weights->ne[3] == 1);
    GGML_ASSERT(input->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(dst->nb[0] == sizeof(uint16_t));
    GGML_ASSERT(qparams->activation_to_output_q20 > 0);

    const int64_t rows = weights->ne[1];
    const bool has_qnn_block_scales =
        !qparams->qnn_weight_block_scale_codes.empty();
    if (has_qnn_block_scales) {
        GGML_ASSERT(qparams->qnn_weight_block_size == 32);
        GGML_ASSERT(qparams->qnn_weight_blocks_per_row == weights->ne[0] / 32);
        GGML_ASSERT(qparams->qnn_channel_scale_to_output_q31.size() ==
            static_cast<size_t>(rows));
        GGML_ASSERT(qparams->qnn_weight_block_scale_codes.size() ==
            static_cast<size_t>(rows * qparams->qnn_weight_blocks_per_row));
        if (qparams->qnn_weight_block_codes_prepared &&
            weights->type == GGML_TYPE_GPTQ2_32) {
            GGML_ASSERT(qparams->qnn_prepared_weight_sums.size() ==
                static_cast<size_t>(rows));
        }
    }
    const bool use_blockwise_requant =
        has_qnn_block_scales &&
        is_enabled_value(std::getenv("GGML_QNN_U16_BLOCKWISE_REQUANT")) &&
        qparams->input.scale / qparams->output.scale >= 1.4142135623730951;
    const int32_t output_bias_q7 =
        is_enabled_value(std::getenv("GGML_QNN_U16_LINEAR_BIAS_CORRECTION"))
        ? qparams->output_bias_q7
        : 0;
    const bool final_round_to_nearest =
        (qparams->projection == "self_attn.o_proj" &&
         use_profiled_oproj_nearest(
             std::getenv("GGML_QNN_U16_OPROJ_NEAREST_REQUANT"),
             qparams->layer_id)) ||
        use_profiled_ffn_nearest(
            std::getenv("GGML_QNN_U16_FFN_NEAREST_REQUANT"),
            qparams->layer_id, qparams->projection);
    const int source_group_size =
        weights->type == GGML_TYPE_GPTQ2_32 ? 32 :
        weights->type == GGML_TYPE_GPTQ2_64 ? 64 : 128;
    const int64_t vectors = input->ne[1] * input->ne[2] * input->ne[3];
    if (has_qnn_block_scales &&
        qparams->qnn_weight_block_codes_prepared &&
        source_group_size == 32) {
        thread_local std::vector<int32_t> activation_block_sums;
        thread_local std::vector<uint8_t> activation_low;
        thread_local std::vector<int8_t> activation_high;
        thread_local std::vector<uint8_t> gs32_row_scratch;
        activation_block_sums.resize(weights->ne[0] / 32);
        if (qparams->weights_gs32_source && rows % 8 != 0) {
            gs32_row_scratch.resize(8 * weights->nb[1]);
        }
        int64_t cached_vector = -1;
        int activation_range = 0;
        int32_t activation_abs_sum = 0;
        const bool tiled_block_codes =
            qparams->qnn_weight_block_code_layout ==
                LLAMA_QNN_BLOCK_CODES_GS32_TILE8_BLOCK_MAJOR;
        const int64_t rows_per_group =
            qparams->weights_gs32_source && rows % 16 == 0 ? 16 : 8;
        const int64_t row_groups = (rows + rows_per_group - 1) / rows_per_group;
        const int64_t grouped_work_items = row_groups * vectors;
        const int64_t work_begin = vectors == 1
            ? grouped_work_items * ith / nth
            : ith;
        const int64_t work_end = vectors == 1
            ? grouped_work_items * (ith + 1) / nth
            : grouped_work_items;
        const int64_t work_stride = vectors == 1 ? 1 : nth;
        for (int64_t work = work_begin; work < work_end; work += work_stride) {
            const int64_t vector = work / row_groups;
            const int64_t row =
                (work - vector * row_groups) * rows_per_group;
            const int64_t i1 = vector % input->ne[1];
            const int64_t i2 = (vector / input->ne[1]) % input->ne[2];
            const int64_t i3 = vector / (input->ne[1] * input->ne[2]);
            const auto * activation = reinterpret_cast<const uint16_t *>(
                reinterpret_cast<const char *>(input->data) +
                i1 * input->nb[1] + i2 * input->nb[2] + i3 * input->nb[3]);
            if (vector != cached_vector) {
                const bool prepare_dotprod =
                    qparams->weights_gs32_source &&
                    ggml_gptq2_32_gs32_dotprod_enabled();
                if (prepare_dotprod) {
                    activation_low.resize(weights->ne[0]);
                    activation_high.resize(weights->ne[0]);
                }
                activation_range =
                    ggml_gptq2_32_prepare_u16_activation(
                        static_cast<int>(weights->ne[0]), activation,
                        qparams->input.zero_point,
                        activation_block_sums.data(),
                        prepare_dotprod ? activation_low.data() : nullptr,
                        prepare_dotprod ? activation_high.data() : nullptr,
                        &activation_abs_sum);
                cached_vector = vector;
            }
            auto * output = reinterpret_cast<uint16_t *>(
                reinterpret_cast<char *>(dst->data) +
                row * dst->nb[0] + i1 * dst->nb[1] +
                i2 * dst->nb[2] + i3 * dst->nb[3]);
            if (qparams->weights_gs32_source && rows_per_group == 16) {
                const bool accumulation_fits_i32 =
                    static_cast<int64_t>(activation_abs_sum) * 7 * 31 <=
                        INT32_MAX;
                ggml_vec_dot_gptq2_32_gs32_u16_qnn_blockwise_affine_16rows(
                    static_cast<int>(weights->ne[0]), output,
                    weights->data, row, activation,
                    activation_range ? activation_low.data() : nullptr,
                    activation_range == GGML_GPTQ2_U16_ACTIVATION_I16
                        ? activation_high.data()
                        : nullptr,
                    activation_block_sums.data(), activation_range,
                    accumulation_fits_i32,
                    qparams->qnn_weight_block_scale_codes.data() +
                        row * qparams->qnn_weight_blocks_per_row,
                    tiled_block_codes ? 0 :
                        qparams->qnn_weight_blocks_per_row,
                    qparams->qnn_channel_scale_to_output_q31.data() + row,
                    qparams->qnn_prepared_weight_sums.data() + row,
                    qparams->input.zero_point, qparams->output.zero_point,
                    use_blockwise_requant ? 29 : 0,
                    final_round_to_nearest, output_bias_q7);
                continue;
            }
            if (qparams->weights_gs32_source && row + 8 <= rows) {
                ggml_vec_dot_gptq2_32_gs32_u16_qnn_blockwise_affine_8rows(
                    static_cast<int>(weights->ne[0]), output,
                    weights->data, row, activation,
                    activation_range ? activation_low.data() : nullptr,
                    activation_range == GGML_GPTQ2_U16_ACTIVATION_I16
                        ? activation_high.data()
                        : nullptr,
                    activation_block_sums.data(), activation_range,
                    qparams->qnn_weight_block_scale_codes.data() +
                        row * qparams->qnn_weight_blocks_per_row,
                    tiled_block_codes ? 0 :
                        qparams->qnn_weight_blocks_per_row,
                    qparams->qnn_channel_scale_to_output_q31.data() + row,
                    qparams->qnn_prepared_weight_sums.data() + row,
                    qparams->input.zero_point, qparams->output.zero_point,
                    use_blockwise_requant ? 29 : 0,
                    final_round_to_nearest, output_bias_q7);
                continue;
            }
            const void * packed_row_group =
                reinterpret_cast<const char *>(weights->data) +
                    row * weights->nb[1];
            if (qparams->weights_gs32_source) {
                const int row_count =
                    static_cast<int>(std::min<int64_t>(8, rows - row));
                ggml_gptq2_32_gs32_restore_rows(
                    static_cast<int>(weights->ne[0]), weights->data,
                    row, row_count, gs32_row_scratch.data(), weights->nb[1]);
                packed_row_group = gs32_row_scratch.data();
            }

            if (row + 8 <= rows) {
                ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_8rows(
                    static_cast<int>(weights->ne[0]), output,
                    packed_row_group,
                    weights->nb[1], activation,
                    activation_block_sums.data(), activation_range,
                    qparams->qnn_weight_block_scale_codes.data() +
                        row * qparams->qnn_weight_blocks_per_row,
                    qparams->qnn_weight_blocks_per_row,
                    qparams->qnn_channel_scale_to_output_q31.data() + row,
                    qparams->qnn_prepared_weight_sums.data() + row,
                    qparams->input.zero_point, qparams->output.zero_point,
                    use_blockwise_requant ? 29 : 0,
                    final_round_to_nearest, output_bias_q7);
                continue;
            }

            int64_t tail_start = row;
            if (tail_start + 4 <= rows) {
                ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_4rows(
                    static_cast<int>(weights->ne[0]),
                    output + (tail_start - row),
                    reinterpret_cast<const char *>(packed_row_group) +
                        (tail_start - row) * weights->nb[1],
                    weights->nb[1], activation,
                    activation_block_sums.data(), activation_range,
                    qparams->qnn_weight_block_scale_codes.data() +
                        tail_start * qparams->qnn_weight_blocks_per_row,
                    qparams->qnn_weight_blocks_per_row,
                    qparams->qnn_channel_scale_to_output_q31.data() + tail_start,
                    qparams->qnn_prepared_weight_sums.data() + tail_start,
                    qparams->input.zero_point, qparams->output.zero_point,
                    use_blockwise_requant ? 29 : 0,
                    final_round_to_nearest, output_bias_q7);
                tail_start += 4;
            }

            for (int64_t tail_row = tail_start; tail_row < rows; ++tail_row) {
                ggml_vec_dot_gptq2_u16_qnn_blockwise_affine(
                    static_cast<int>(weights->ne[0]),
                    output + (tail_row - row),
                    reinterpret_cast<const char *>(packed_row_group) +
                        (tail_row - row) * weights->nb[1],
                    activation,
                    qparams->qnn_weight_block_scale_codes.data() +
                        tail_row * qparams->qnn_weight_blocks_per_row,
                    qparams->qnn_channel_scale_to_output_q31[tail_row],
                    qparams->input.zero_point, qparams->output.zero_point,
                    source_group_size, 1,
                    use_blockwise_requant ? 29 : 0,
                    final_round_to_nearest, output_bias_q7);
            }
        }
        return;
    }

    const int64_t work_items = rows * vectors;
    thread_local std::vector<uint8_t> gs32_single_row_scratch;
    if (qparams->weights_gs32_source) {
        GGML_ASSERT(weights->type == GGML_TYPE_GPTQ2_32);
        gs32_single_row_scratch.resize(weights->nb[1]);
    }
    for (int64_t work = ith; work < work_items; work += nth) {
        const int64_t vector = work / rows;
        const int64_t row = work - vector * rows;
        const int64_t i1 = vector % input->ne[1];
        const int64_t i2 = (vector / input->ne[1]) % input->ne[2];
        const int64_t i3 = vector / (input->ne[1] * input->ne[2]);
        const auto * activation = reinterpret_cast<const uint16_t *>(
            reinterpret_cast<const char *>(input->data) +
            i1 * input->nb[1] + i2 * input->nb[2] + i3 * input->nb[3]);
        const void * packed_weights;
        if (qparams->weights_gs32_source) {
            ggml_gptq2_32_gs32_restore_rows(
                static_cast<int>(weights->ne[0]), weights->data,
                row, 1, gs32_single_row_scratch.data(), weights->nb[1]);
            packed_weights = gs32_single_row_scratch.data();
        } else {
            packed_weights =
                reinterpret_cast<const char *>(weights->data) + row * weights->nb[1];
        }
        auto * output = reinterpret_cast<uint16_t *>(
            reinterpret_cast<char *>(dst->data) +
            row * dst->nb[0] + i1 * dst->nb[1] +
            i2 * dst->nb[2] + i3 * dst->nb[3]);

        const uint8_t * block_scale_codes = has_qnn_block_scales
            ? qparams->qnn_weight_block_scale_codes.data() +
                row * qparams->qnn_weight_blocks_per_row
            : nullptr;
        const int64_t channel_scale_to_output_q31 = has_qnn_block_scales
            ? qparams->qnn_channel_scale_to_output_q31[row]
            : 0;

        if (has_qnn_block_scales) {
            ggml_vec_dot_gptq2_u16_qnn_blockwise_affine(
                static_cast<int>(weights->ne[0]), output, packed_weights,
                activation, block_scale_codes, channel_scale_to_output_q31,
                qparams->input.zero_point, qparams->output.zero_point,
                source_group_size, qparams->qnn_weight_block_codes_prepared,
                use_blockwise_requant ? 29 : 0,
                final_round_to_nearest, output_bias_q7);
            continue;
        }

        switch (weights->type) {
            case GGML_TYPE_GPTQ2_32:
                ggml_vec_dot_gptq2_32_u16_qnn_fixed(
                    static_cast<int>(weights->ne[0]), output, packed_weights,
                    activation, qparams->activation_to_output_q20,
                    qparams->input.zero_point, qparams->output.zero_point);
                break;
            case GGML_TYPE_GPTQ2_64:
                ggml_vec_dot_gptq2_64_u16_qnn_fixed(
                    static_cast<int>(weights->ne[0]), output, packed_weights,
                    activation, qparams->activation_to_output_q20,
                    qparams->input.zero_point, qparams->output.zero_point);
                break;
            case GGML_TYPE_GPTQ2_128:
                ggml_vec_dot_gptq2_128_u16_qnn_fixed(
                    static_cast<int>(weights->ne[0]), output, packed_weights,
                    activation, qparams->activation_to_output_q20,
                    qparams->input.zero_point, qparams->output.zero_point);
                break;
            default:
                GGML_ABORT("unsupported QNN U16 weight type");
        }
    }
}

} // namespace

bool llama_qnn_u16_activations_enabled() {
    return is_enabled_value(std::getenv("GGML_QNN_U16_ACTIVATIONS"));
}

ggml_tensor * llama_qnn_quantize_f32_to_u16(
        ggml_context * ctx,
        ggml_tensor * input,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * quantize_operation) {
    GGML_ASSERT(ctx != nullptr && input != nullptr && profile != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    GGML_ASSERT(quantize_operation != nullptr &&
        quantize_operation->type_name == "Quantize");
    const llama_qnn_u16_tensor * output_tensor =
        require_u16_operand(profile, quantize_operation, "output", 0);
    const auto & qparams = output_tensor->qparams.scale_offsets[0];
    const qnn_f32_u16_boundary_params value { qparams.scale, qparams.zero_point };
    auto * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { input };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        args, 1, qnn_quantize_f32_u16_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_dequantize_u16_to_f32(
        ggml_context * ctx,
        ggml_tensor * input,
        const llama_qnn_u16_tensor * input_qparams) {
    GGML_ASSERT(ctx != nullptr && input != nullptr && input_qparams != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16);
    GGML_ASSERT(input_qparams->qparams.encoding ==
        LLAMA_QNN_QUANTIZATION_SCALE_OFFSET);
    GGML_ASSERT(input_qparams->qparams.scale_offsets.size() == 1);
    const auto & qparams = input_qparams->qparams.scale_offsets[0];
    const qnn_f32_u16_boundary_params value { qparams.scale, qparams.zero_point };
    auto * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { input };
    return ggml_custom_4d(
        ctx, GGML_TYPE_F32,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        args, 1, qnn_dequantize_u16_f32_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_mul_mat(
        ggml_context * ctx,
        ggml_tensor * weights,
        ggml_tensor * input,
        const llama_qnn_linear_qparams * qparams) {
    GGML_ASSERT(ctx != nullptr && weights != nullptr && input != nullptr && qparams != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16);
    GGML_ASSERT(weights->ne[0] == input->ne[0]);
    ggml_tensor * args[] = { input, weights };
    return ggml_custom_4d(
        ctx,
        GGML_TYPE_U16,
        weights->ne[1],
        input->ne[1],
        input->ne[2],
        input->ne[3],
        args,
        2,
        qnn_u16_mul_mat_compute,
        GGML_N_TASKS_MAX,
        const_cast<llama_qnn_linear_qparams *>(qparams));
}

static ggml_tensor * llama_qnn_u16_binary(
        ggml_context * ctx,
        ggml_tensor * lhs,
        ggml_tensor * rhs,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation,
        bool subtract_rhs,
        bool multiply) {
    GGML_ASSERT(ctx != nullptr && lhs != nullptr && rhs != nullptr);
    GGML_ASSERT(profile != nullptr && operation != nullptr);
    GGML_ASSERT(lhs->type == GGML_TYPE_U16 && rhs->type == GGML_TYPE_U16);
    GGML_ASSERT(ggml_are_same_shape(lhs, rhs));
    GGML_ASSERT(operation->inputs.size() >= 2 && !operation->outputs.empty());
    if (multiply) {
        GGML_ASSERT(operation->type_name == "ElementWiseMultiply");
        GGML_ASSERT(operation->product_to_output_q31 >= 0);
    } else if (subtract_rhs) {
        GGML_ASSERT(operation->type_name == "ElementWiseSubtract");
        GGML_ASSERT(operation->input_to_output_q20.size() >= 2);
    } else {
        GGML_ASSERT(operation->type_name == "ElementWiseAdd");
        GGML_ASSERT(operation->input_to_output_q20.size() >= 2);
    }

    const llama_qnn_u16_tensor * lhs_tensor =
        require_u16_operand(profile, operation, "input", 0);
    const llama_qnn_u16_tensor * rhs_tensor =
        require_u16_operand(profile, operation, "input", 1);
    const llama_qnn_u16_tensor * output_tensor =
        require_u16_operand(profile, operation, "output", 0);
    const int64_t lhs_multiplier = multiply
        ? operation->product_to_output_q31
        : qnn_u16_scale_ratio_q15(lhs_tensor, output_tensor);
    const int64_t rhs_multiplier = multiply
        ? 0
        : (subtract_rhs ? -qnn_u16_scale_ratio_q15(rhs_tensor, output_tensor)
                        : qnn_u16_scale_ratio_q15(rhs_tensor, output_tensor));
    const qnn_u16_binary_params value {
        lhs_multiplier,
        rhs_multiplier,
        operation->product_requant_nudge_q31,
        lhs_tensor->qparams.scale_offsets[0].zero_point,
        rhs_tensor->qparams.scale_offsets[0].zero_point,
        output_tensor->qparams.scale_offsets[0].zero_point,
        multiply,
    };
    qnn_u16_binary_params * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { lhs, rhs };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16,
        lhs->ne[0], lhs->ne[1], lhs->ne[2], lhs->ne[3],
        args, 2, qnn_u16_binary_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_add(
        ggml_context * ctx,
        ggml_tensor * lhs,
        ggml_tensor * rhs,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation) {
    return llama_qnn_u16_binary(
        ctx, lhs, rhs, profile, operation, false, false);
}

ggml_tensor * llama_qnn_u16_sub(
        ggml_context * ctx,
        ggml_tensor * lhs,
        ggml_tensor * rhs,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation) {
    return llama_qnn_u16_binary(
        ctx, lhs, rhs, profile, operation, true, false);
}

ggml_tensor * llama_qnn_u16_mul(
        ggml_context * ctx,
        ggml_tensor * lhs,
        ggml_tensor * rhs,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation) {
    return llama_qnn_u16_binary(
        ctx, lhs, rhs, profile, operation, false, true);
}

static ggml_tensor * llama_qnn_u16_binary_static(
        ggml_context * ctx,
        ggml_tensor * lhs,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation,
        bool multiply) {
    GGML_ASSERT(ctx != nullptr && lhs != nullptr && profile != nullptr);
    GGML_ASSERT(operation != nullptr && lhs->type == GGML_TYPE_U16);
    GGML_ASSERT(operation->inputs.size() >= 2 && !operation->outputs.empty());
    GGML_ASSERT(operation->type_name == (multiply
        ? "ElementWiseMultiply" : "ElementWiseAdd"));
    const auto * lhs_tensor = require_u16_operand(profile, operation, "input", 0);
    const auto * rhs_tensor = require_u16_operand(profile, operation, "input", 1);
    const auto * output_tensor = require_u16_operand(profile, operation, "output", 0);
    GGML_ASSERT(rhs_tensor->static_data.size() == 1);
    GGML_ASSERT(multiply || operation->input_to_output_q20.size() >= 2);
    GGML_ASSERT(!multiply || operation->product_to_output_q31 >= 0);
    int32_t folded_multiplier_q15 = 0;
    int32_t folded_right_shift = 0;
    if (multiply) {
        const auto & lhs_qparams = lhs_tensor->qparams.scale_offsets[0];
        const auto & rhs_qparams = rhs_tensor->qparams.scale_offsets[0];
        const auto & output_qparams = output_tensor->qparams.scale_offsets[0];
        const int32_t rhs_centered =
            (int32_t) rhs_tensor->static_data[0] - rhs_qparams.zero_point;
        const double folded_ratio =
            ((double) lhs_qparams.scale * (double) rhs_qparams.scale *
             (double) rhs_centered) /
            (double) output_qparams.scale;
        int exponent = 0;
        const double normalized = std::frexp(folded_ratio, &exponent);
        int64_t multiplier = std::llround(std::ldexp(normalized, 15));
        if (multiplier == (INT64_C(1) << 15)) {
            multiplier >>= 1;
            ++exponent;
        }
        folded_right_shift = 15 - exponent;
        GGML_ASSERT(multiplier >= INT16_MIN && multiplier <= INT16_MAX);
        GGML_ASSERT(folded_right_shift > 0 && folded_right_shift < 31);
        folded_multiplier_q15 = static_cast<int32_t>(multiplier);
    }
    const qnn_u16_static_binary_params value {
        {
            multiply ? operation->product_to_output_q31
                     : operation->input_to_output_q20[0],
            multiply ? 0 : operation->input_to_output_q20[1],
            operation->product_requant_nudge_q31,
            lhs_tensor->qparams.scale_offsets[0].zero_point,
            rhs_tensor->qparams.scale_offsets[0].zero_point,
            output_tensor->qparams.scale_offsets[0].zero_point,
            multiply,
        },
        rhs_tensor->static_data[0],
        folded_multiplier_q15,
        folded_right_shift,
    };
    auto * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { lhs };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16,
        lhs->ne[0], lhs->ne[1], lhs->ne[2], lhs->ne[3],
        args, 1, qnn_u16_static_binary_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_add_static(
        ggml_context * ctx,
        ggml_tensor * lhs,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation) {
    return llama_qnn_u16_binary_static(ctx, lhs, profile, operation, false);
}

ggml_tensor * llama_qnn_u16_mul_static(
        ggml_context * ctx,
        ggml_tensor * lhs,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation) {
    return llama_qnn_u16_binary_static(ctx, lhs, profile, operation, true);
}

ggml_tensor * llama_qnn_attention_mask_condition(
        ggml_context * ctx,
        ggml_tensor * mask) {
    GGML_ASSERT(ctx != nullptr && mask != nullptr);
    GGML_ASSERT(mask->type == GGML_TYPE_F32 || mask->type == GGML_TYPE_F16);
    ggml_tensor * args[] = { mask };
    return ggml_custom_4d(
        ctx, GGML_TYPE_I8,
        mask->ne[0], mask->ne[1], mask->ne[2], mask->ne[3],
        args, 1, qnn_attention_mask_condition_compute, GGML_N_TASKS_MAX, nullptr);
}

namespace {

qnn_u16_rms_norm_params make_rms_norm_params(
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation,
        int64_t dimension) {
    GGML_ASSERT(profile != nullptr && operation != nullptr);
    GGML_ASSERT(operation->type_name == "RmsNorm");
    GGML_ASSERT(operation->inputs.size() >= 3 && !operation->outputs.empty());
    GGML_ASSERT(operation->input_to_output_q20.size() >= 2);
    GGML_ASSERT(operation->input_to_output_q20[1] > 0);
    const llama_qnn_u16_tensor * input_tensor =
        require_u16_operand(profile, operation, "input", 0);
    const llama_qnn_u16_tensor * weight_tensor =
        require_u16_operand(profile, operation, "input", 1);
    const llama_qnn_u16_tensor * bias_tensor =
        require_u16_operand(profile, operation, "input", 2);
    const llama_qnn_u16_tensor * output_tensor =
        require_u16_operand(profile, operation, "output", 0);
    GGML_ASSERT(weight_tensor->static_data.size() == static_cast<size_t>(dimension));
    GGML_ASSERT(!bias_tensor->static_data.empty());
    const int32_t bias_zero_point = bias_tensor->qparams.scale_offsets[0].zero_point;
    for (const uint16_t code : bias_tensor->static_data) {
        GGML_ASSERT(code == bias_zero_point);
    }
    return {
        weight_tensor->static_data.data(),
        static_cast<int64_t>(weight_tensor->static_data.size()),
        input_tensor->qparams.scale_offsets[0].zero_point,
        weight_tensor->qparams.scale_offsets[0].zero_point,
        operation->rms_epsilon_in_codes_q16,
        qnn_u16_scale_ratio_q31(weight_tensor, output_tensor),
        output_tensor->qparams.scale_offsets[0].zero_point,
    };
}

} // namespace

ggml_tensor * llama_qnn_u16_rms_norm(
        ggml_context * ctx,
        ggml_tensor * input,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation) {
    GGML_ASSERT(ctx != nullptr && input != nullptr);
    GGML_ASSERT(profile != nullptr && operation != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16);
    const auto value = make_rms_norm_params(profile, operation, input->ne[0]);
    qnn_u16_rms_norm_params * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { input };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        args, 1, qnn_u16_rms_norm_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_rms_norm_heads(
        ggml_context * ctx,
        ggml_tensor * input,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * const * operations,
        int32_t n_head) {
    GGML_ASSERT(ctx != nullptr && input != nullptr && profile != nullptr);
    GGML_ASSERT(operations != nullptr && n_head > 0);
    GGML_ASSERT(input->type == GGML_TYPE_U16 && input->ne[1] == n_head);
    auto * values = static_cast<qnn_u16_rms_norm_params *>(
        ggml_new_buffer(ctx, sizeof(qnn_u16_rms_norm_params) * n_head));
    for (int32_t head = 0; head < n_head; ++head) {
        values[head] = make_rms_norm_params(profile, operations[head], input->ne[0]);
    }
    const qnn_u16_head_params<qnn_u16_rms_norm_params> value { values, n_head };
    auto * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { input };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        args, 1, qnn_u16_rms_norm_heads_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_sigmoid(
        ggml_context * ctx,
        ggml_tensor * input,
        const llama_qnn_operation * operation) {
    GGML_ASSERT(ctx != nullptr && input != nullptr && operation != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16);
    GGML_ASSERT(operation->type_name == "Sigmoid");
    GGML_ASSERT(operation->unary_lut.size() == static_cast<size_t>(UINT16_MAX) + 1);
    const qnn_u16_lut_params value { operation->unary_lut.data() };
    qnn_u16_lut_params * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { input };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        args, 1, qnn_u16_lut_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_swiglu(
        ggml_context * ctx,
        ggml_tensor * gate,
        ggml_tensor * up,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * sigmoid_operation,
        const llama_qnn_operation * silu_multiply_operation,
        const llama_qnn_operation * product_multiply_operation) {
    GGML_ASSERT(ctx != nullptr && gate != nullptr && up != nullptr);
    GGML_ASSERT(profile != nullptr && sigmoid_operation != nullptr);
    GGML_ASSERT(silu_multiply_operation != nullptr &&
        product_multiply_operation != nullptr);
    GGML_ASSERT(gate->type == GGML_TYPE_U16 && up->type == GGML_TYPE_U16);
    GGML_ASSERT(ggml_are_same_shape(gate, up));
    GGML_ASSERT(sigmoid_operation->type_name == "Sigmoid");
    GGML_ASSERT(sigmoid_operation->unary_lut.size() ==
        static_cast<size_t>(UINT16_MAX) + 1);

    const auto make_multiply_params =
        [profile](const llama_qnn_operation * operation) {
            GGML_ASSERT(operation != nullptr);
            GGML_ASSERT(operation->type_name == "ElementWiseMultiply");
            GGML_ASSERT(operation->inputs.size() >= 2 &&
                !operation->outputs.empty());
            GGML_ASSERT(operation->product_to_output_q31 >= 0);
            const auto * lhs =
                require_u16_operand(profile, operation, "input", 0);
            const auto * rhs =
                require_u16_operand(profile, operation, "input", 1);
            const auto * output =
                require_u16_operand(profile, operation, "output", 0);
            return qnn_u16_binary_params {
                operation->product_to_output_q31,
                0,
                operation->product_requant_nudge_q31,
                lhs->qparams.scale_offsets[0].zero_point,
                rhs->qparams.scale_offsets[0].zero_point,
                output->qparams.scale_offsets[0].zero_point,
                true,
            };
        };
    const qnn_u16_swiglu_params value {
        sigmoid_operation->unary_lut.data(),
        make_multiply_params(silu_multiply_operation),
        make_multiply_params(product_multiply_operation),
    };
    auto * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { gate, up };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16,
        gate->ne[0], gate->ne[1], gate->ne[2], gate->ne[3],
        args, 2, qnn_u16_swiglu_compute, GGML_N_TASKS_MAX, params);
}

namespace {

std::string indexed_head_fx_name(
        const char * stem,
        int32_t operation_index,
        int32_t head_index) {
    std::string result(stem);
    if (operation_index > 0) {
        result += "_" + std::to_string(operation_index);
    }
    result += "_h_" + std::to_string(head_index);
    return result;
}

const llama_qnn_operation * require_fx_operation(
        const llama_qnn_quant_profile * profile,
        int32_t layer_id,
        const std::string & fx_name,
        const char * type_name) {
    const llama_qnn_operation * operation =
        profile->find_operation_by_fx(layer_id, fx_name);
    GGML_ASSERT(operation != nullptr && operation->type_name == type_name);
    return operation;
}

const llama_qnn_u16_tensor * require_storage_qparams_source(
        const llama_qnn_quant_profile * profile,
        const llama_qnn_u16_tensor * tensor) {
    GGML_ASSERT(profile != nullptr && tensor != nullptr);
    const llama_qnn_u16_tensor * current = tensor;
    for (int depth = 0; depth < 16; ++depth) {
        const llama_qnn_operation * producer =
            profile->find_producer(current->shard_index, current->name);
        if (producer == nullptr) {
            return current;
        }
        const bool preserves_codes = producer->type_name == "Reshape" ||
            producer->type_name == "StridedSlice" ||
            producer->type_name == "Gather";
        if (!preserves_codes) {
            return current;
        }
        GGML_ASSERT(!producer->inputs.empty());
        const llama_qnn_u16_tensor * source =
            profile->find_u16_tensor(current->shard_index, producer->inputs.front());
        GGML_ASSERT(source != nullptr);
        GGML_ASSERT(source->qparams.encoding == LLAMA_QNN_QUANTIZATION_SCALE_OFFSET);
        GGML_ASSERT(source->qparams.scale_offsets.size() == 1);
        current = source;
    }
    GGML_ABORT("QNN U16 storage-qparam ancestry exceeds the supported depth");
}

qnn_u16_rope_params make_rope_params(
        const llama_qnn_quant_profile * profile,
        const llama_qnn_u16_rope_head_ops & ops,
        int64_t dimension) {
    GGML_ASSERT(profile != nullptr);
    GGML_ASSERT(ops.subtract != nullptr && ops.add != nullptr);
    GGML_ASSERT(ops.cos_table != nullptr && ops.sin_table != nullptr);
    GGML_ASSERT(ops.cos_table->dimensions.size() == 2);
    GGML_ASSERT(ops.cos_table->dimensions == ops.sin_table->dimensions);
    GGML_ASSERT(!ops.cos_table->static_data.empty() &&
        !ops.sin_table->static_data.empty());
    qnn_u16_rope_params value {};
    value.cos_codes = ops.cos_table->static_data.data();
    value.sin_codes = ops.sin_table->static_data.data();
    value.table_rows = ops.cos_table->dimensions[0];
    value.half_dimension = ops.cos_table->dimensions[1];
    GGML_ASSERT(dimension == value.half_dimension * 2);

    const llama_qnn_u16_tensor * split_input =
        require_u16_operand(profile, ops.slices[0], "input", 0);
    const llama_qnn_u16_tensor * split_input_second =
        require_u16_operand(profile, ops.slices[1], "input", 0);
    GGML_ASSERT(split_input->name == split_input_second->name);
    value.fixed.split_input_zero_point =
        split_input->qparams.scale_offsets[0].zero_point;
    for (int index = 0; index < 2; ++index) {
        const llama_qnn_u16_tensor * split_output =
            require_u16_operand(profile, ops.slices[index], "output", 0);
        value.fixed.split_to_output_q31[index] =
            qnn_u16_scale_ratio_q31(split_input, split_output);
        value.fixed.split_output_zero_points[index] =
            split_output->qparams.scale_offsets[0].zero_point;
    }
    for (int index = 0; index < 4; ++index) {
        const llama_qnn_u16_tensor * lhs =
            require_u16_operand(profile, ops.multiply[index], "input", 0);
        const llama_qnn_u16_tensor * table =
            require_u16_operand(profile, ops.multiply[index], "input", 1);
        const llama_qnn_u16_tensor * table_storage =
            require_storage_qparams_source(profile, table);
        const llama_qnn_u16_tensor * table_source =
            index == 0 || index == 3 ? ops.cos_table : ops.sin_table;
        const llama_qnn_u16_tensor * output =
            require_u16_operand(profile, ops.multiply[index], "output", 0);
        value.fixed.lhs_zero_points[index] =
            lhs->qparams.scale_offsets[0].zero_point;
        value.fixed.table_source_zero_points[index] =
            table_source->qparams.scale_offsets[0].zero_point;
        value.fixed.table_source_to_storage_q31[index] =
            qnn_u16_scale_ratio_q31(table_source, table_storage);
        value.fixed.table_zero_points[index] =
            table_storage->qparams.scale_offsets[0].zero_point;
        value.fixed.product_to_output_q31_shift3[index] =
            qnn_u16_scale_product_ratio_q31_shift3(lhs, table_storage, output);
        value.fixed.product_output_zero_points[index] =
            output->qparams.scale_offsets[0].zero_point;
    }
    const llama_qnn_u16_tensor * subtract_output =
        require_u16_operand(profile, ops.subtract, "output", 0);
    const llama_qnn_u16_tensor * add_output =
        require_u16_operand(profile, ops.add, "output", 0);
    qnn_u16_binary_params_q15(
        require_u16_operand(profile, ops.subtract, "input", 0),
        require_u16_operand(profile, ops.subtract, "input", 1),
        subtract_output, true,
        value.fixed.combine_multipliers_q15,
        &value.fixed.combine_shifts[0],
        &value.fixed.combine_biases[0]);
    qnn_u16_binary_params_q15(
        require_u16_operand(profile, ops.add, "input", 0),
        require_u16_operand(profile, ops.add, "input", 1),
        add_output, false,
        value.fixed.combine_multipliers_q15 + 2,
        &value.fixed.combine_shifts[1],
        &value.fixed.combine_biases[1]);
    return value;
}

} // namespace

ggml_tensor * llama_qnn_u16_rope(
        ggml_context * ctx,
        ggml_tensor * input,
        ggml_tensor * positions,
        const llama_qnn_quant_profile * profile,
        int32_t layer_id,
        int32_t head_index,
        bool key_head) {
    GGML_ASSERT(ctx != nullptr && input != nullptr && positions != nullptr);
    GGML_ASSERT(profile != nullptr && input->type == GGML_TYPE_U16);
    GGML_ASSERT(positions->type == GGML_TYPE_I32);
    GGML_ASSERT(layer_id >= 0 && layer_id < profile->num_decoder_layers);
    GGML_ASSERT(head_index >= 0);

    const int32_t multiply_base = 10 * layer_id + (key_head ? 5 : 1);
    const int32_t slice_base = 4 * layer_id + (key_head ? 2 : 0);
    const int32_t subtract_index = 2 * layer_id + (key_head ? 1 : 0);
    const int32_t add_index = 5 * layer_id + (key_head ? 1 : 0);
    const llama_qnn_operation * multiply[4] = {
        require_fx_operation(profile, layer_id,
            indexed_head_fx_name("aten_mul_tensor", multiply_base, head_index),
            "ElementWiseMultiply"),
        require_fx_operation(profile, layer_id,
            indexed_head_fx_name("aten_mul_tensor", multiply_base + 1, head_index),
            "ElementWiseMultiply"),
        require_fx_operation(profile, layer_id,
            indexed_head_fx_name("aten_mul_tensor", multiply_base + 2, head_index),
            "ElementWiseMultiply"),
        require_fx_operation(profile, layer_id,
            indexed_head_fx_name("aten_mul_tensor", multiply_base + 3, head_index),
            "ElementWiseMultiply"),
    };
    const llama_qnn_operation * slices[2] = {
        require_fx_operation(profile, layer_id,
            indexed_head_fx_name("aten_slice_copy_tensor", slice_base, head_index),
            "StridedSlice"),
        require_fx_operation(profile, layer_id,
            indexed_head_fx_name("aten_slice_copy_tensor", slice_base + 1, head_index),
            "StridedSlice"),
    };
    const llama_qnn_operation * subtract = require_fx_operation(
        profile, layer_id,
        indexed_head_fx_name("aten_sub_tensor", subtract_index, head_index),
        "ElementWiseSubtract");
    const llama_qnn_operation * add = require_fx_operation(
        profile, layer_id,
        indexed_head_fx_name("aten_add_tensor", add_index, head_index),
        "ElementWiseAdd");
    GGML_ASSERT(subtract->input_to_output_q20.size() >= 2);
    GGML_ASSERT(add->input_to_output_q20.size() >= 2);

    // These are the exact U16 source tables embedded by the Qwen3 export.
    // Every later shard receives code-identical views of these two tensors.
    const llama_qnn_u16_tensor * cos_table =
        profile->find_u16_tensor("b__frozen_param311@0");
    const llama_qnn_u16_tensor * sin_table =
        profile->find_u16_tensor("b__frozen_param312@0");
    GGML_ASSERT(cos_table != nullptr && sin_table != nullptr);
    GGML_ASSERT(cos_table->dimensions.size() == 2 && sin_table->dimensions.size() == 2);
    GGML_ASSERT(cos_table->dimensions == sin_table->dimensions);
    GGML_ASSERT(!cos_table->static_data.empty() && !sin_table->static_data.empty());
    const int64_t table_rows = cos_table->dimensions[0];
    const int64_t half_dimension = cos_table->dimensions[1];
    GGML_ASSERT(input->ne[0] == half_dimension * 2);

    qnn_u16_rope_params value {};
    value.cos_codes = cos_table->static_data.data();
    value.sin_codes = sin_table->static_data.data();
    value.table_rows = table_rows;
    value.half_dimension = half_dimension;
    const llama_qnn_u16_tensor * split_input =
        require_u16_operand(profile, slices[0], "input", 0);
    const llama_qnn_u16_tensor * split_input_second =
        require_u16_operand(profile, slices[1], "input", 0);
    GGML_ASSERT(split_input->name == split_input_second->name);
    value.fixed.split_input_zero_point =
        split_input->qparams.scale_offsets[0].zero_point;
    for (int index = 0; index < 2; ++index) {
        const llama_qnn_u16_tensor * split_output =
            require_u16_operand(profile, slices[index], "output", 0);
        value.fixed.split_to_output_q31[index] =
            qnn_u16_scale_ratio_q31(split_input, split_output);
        value.fixed.split_output_zero_points[index] =
            split_output->qparams.scale_offsets[0].zero_point;
    }
    for (int index = 0; index < 4; ++index) {
        const llama_qnn_u16_tensor * lhs =
            require_u16_operand(profile, multiply[index], "input", 0);
        const llama_qnn_u16_tensor * table =
            require_u16_operand(profile, multiply[index], "input", 1);
        const llama_qnn_u16_tensor * table_storage =
            require_storage_qparams_source(profile, table);
        const llama_qnn_u16_tensor * table_source =
            index == 0 || index == 3 ? cos_table : sin_table;
        const llama_qnn_u16_tensor * output =
            require_u16_operand(profile, multiply[index], "output", 0);
        value.fixed.lhs_zero_points[index] =
            lhs->qparams.scale_offsets[0].zero_point;
        value.fixed.table_source_zero_points[index] =
            table_source->qparams.scale_offsets[0].zero_point;
        value.fixed.table_source_to_storage_q31[index] =
            qnn_u16_scale_ratio_q31(table_source, table_storage);
        value.fixed.table_zero_points[index] =
            table_storage->qparams.scale_offsets[0].zero_point;
        value.fixed.product_to_output_q31_shift3[index] =
            qnn_u16_scale_product_ratio_q31_shift3(
                lhs, table_storage, output);
        value.fixed.product_output_zero_points[index] =
            output->qparams.scale_offsets[0].zero_point;
    }
    const llama_qnn_u16_tensor * subtract_output =
        require_u16_operand(profile, subtract, "output", 0);
    const llama_qnn_u16_tensor * add_output =
        require_u16_operand(profile, add, "output", 0);
    qnn_u16_binary_params_q15(
        require_u16_operand(profile, subtract, "input", 0),
        require_u16_operand(profile, subtract, "input", 1),
        subtract_output, true,
        value.fixed.combine_multipliers_q15,
        &value.fixed.combine_shifts[0],
        &value.fixed.combine_biases[0]);
    qnn_u16_binary_params_q15(
        require_u16_operand(profile, add, "input", 0),
        require_u16_operand(profile, add, "input", 1),
        add_output, false,
        value.fixed.combine_multipliers_q15 + 2,
        &value.fixed.combine_shifts[1],
        &value.fixed.combine_biases[1]);

    qnn_u16_rope_params * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { input, positions };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        args, 2, qnn_u16_rope_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_rope_heads(
        ggml_context * ctx,
        ggml_tensor * input,
        ggml_tensor * positions,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_u16_rope_head_ops * head_ops,
        int32_t n_head) {
    GGML_ASSERT(ctx != nullptr && input != nullptr && positions != nullptr);
    GGML_ASSERT(profile != nullptr && head_ops != nullptr && n_head > 0);
    GGML_ASSERT(input->type == GGML_TYPE_U16 && positions->type == GGML_TYPE_I32);
    GGML_ASSERT(input->ne[1] == n_head);
    auto * values = static_cast<qnn_u16_rope_params *>(
        ggml_new_buffer(ctx, sizeof(qnn_u16_rope_params) * n_head));
    for (int32_t head = 0; head < n_head; ++head) {
        values[head] = make_rope_params(profile, head_ops[head], input->ne[0]);
    }
    const qnn_u16_head_params<qnn_u16_rope_params> value { values, n_head };
    auto * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { input, positions };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        args, 2, qnn_u16_rope_heads_compute, GGML_N_TASKS_MAX, params);
}

namespace {

qnn_u16_s16_matmul_params make_qk_rotate_params(
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation,
        int64_t dimension) {
    GGML_ASSERT(profile != nullptr && operation != nullptr);
    GGML_ASSERT(operation->type_name == "MatMul");
    const llama_qnn_u16_tensor * input_tensor =
        require_u16_operand(profile, operation, "input", 0);
    const llama_qnn_aux_quantized_tensor * weight_tensor =
        profile->find_aux_operand(*operation, "input", 1);
    const llama_qnn_u16_tensor * output_tensor =
        require_u16_operand(profile, operation, "output", 0);
    GGML_ASSERT(weight_tensor != nullptr);
    GGML_ASSERT(weight_tensor->data_type == "QNN_DATATYPE_SFIXED_POINT_16");
    GGML_ASSERT(weight_tensor->element_bytes == sizeof(int16_t));
    GGML_ASSERT(weight_tensor->dimensions.size() == 2);
    GGML_ASSERT(weight_tensor->qparams.scale_offsets.size() == 1);
    const int64_t input_dimension = weight_tensor->dimensions[0];
    const int64_t output_dimension = weight_tensor->dimensions[1];
    GGML_ASSERT(dimension == input_dimension);
    GGML_ASSERT(weight_tensor->static_data.size() ==
        static_cast<size_t>(input_dimension * output_dimension) * sizeof(int16_t));
    const auto * static_weights =
        reinterpret_cast<const int16_t *>(weight_tensor->static_data.data());
    bool is_qnn_hadamard = input_dimension == 128 && output_dimension == 128 &&
        weight_tensor->qparams.scale_offsets[0].zero_point == 0;
    for (int64_t index = 0;
         is_qnn_hadamard && index < input_dimension * output_dimension;
         ++index) {
        is_qnn_hadamard = static_weights[index] == INT16_MAX ||
            static_weights[index] == -INT16_MAX;
    }
    GGML_ASSERT(is_qnn_hadamard &&
        "QNN Q/K rotation matrix no longer matches the validated HTP contract");
    const double ratio =
        (static_cast<double>(input_tensor->qparams.scale_offsets[0].scale) *
         static_cast<double>(weight_tensor->qparams.scale_offsets[0].scale)) /
        static_cast<double>(output_tensor->qparams.scale_offsets[0].scale);
    const double q24 = std::ldexp(
        static_cast<double>(static_cast<float>(std::ldexp(ratio, 16))), 24);
    GGML_ASSERT(std::isfinite(q24) && q24 > 0.0 &&
        q24 <= static_cast<double>(std::numeric_limits<int64_t>::max()));
    return {
        static_weights, input_dimension, output_dimension,
        input_tensor->qparams.scale_offsets[0].zero_point,
        weight_tensor->qparams.scale_offsets[0].zero_point,
        static_cast<int64_t>(std::llround(q24)),
        output_tensor->qparams.scale_offsets[0].zero_point,
        7, 16,
    };
}

qnn_u16_to_u8_params make_to_u8_params(
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation) {
    GGML_ASSERT(profile != nullptr && operation != nullptr);
    GGML_ASSERT(operation->type_name == "Convert");
    GGML_ASSERT(operation->unary_input_to_output_q31 > 0);
    const llama_qnn_u16_tensor * input_tensor =
        require_u16_operand(profile, operation, "input", 0);
    const llama_qnn_aux_quantized_tensor * output_tensor =
        profile->find_aux_operand(*operation, "output", 0);
    GGML_ASSERT(output_tensor != nullptr);
    GGML_ASSERT(output_tensor->data_type == "QNN_DATATYPE_UFIXED_POINT_8");
    GGML_ASSERT(output_tensor->element_bytes == sizeof(uint8_t));
    GGML_ASSERT(output_tensor->qparams.scale_offsets.size() == 1);
    return {
        input_tensor->qparams.scale_offsets[0].zero_point,
        operation->unary_input_to_output_q31,
        output_tensor->qparams.scale_offsets[0].zero_point,
    };
}

} // namespace

ggml_tensor * llama_qnn_u16_qk_rotate(
        ggml_context * ctx,
        ggml_tensor * input,
        const llama_qnn_quant_profile * profile,
        int32_t layer_id,
        int32_t head_index,
        bool key_head) {
    GGML_ASSERT(ctx != nullptr && input != nullptr && profile != nullptr);
    GGML_ASSERT(input->type == GGML_TYPE_U16);
    GGML_ASSERT(layer_id >= 0 && layer_id < profile->num_decoder_layers);
    GGML_ASSERT(head_index >= 0);

    const int32_t operation_index = 4 * layer_id + (key_head ? 1 : 0);
    const llama_qnn_operation * operation = require_fx_operation(
        profile, layer_id,
        indexed_head_fx_name(
            "aten_matmul_default", operation_index, head_index),
        "MatMul");
    const llama_qnn_u16_tensor * input_tensor =
        require_u16_operand(profile, operation, "input", 0);
    const llama_qnn_aux_quantized_tensor * weight_tensor =
        profile->find_aux_operand(*operation, "input", 1);
    const llama_qnn_u16_tensor * output_tensor =
        require_u16_operand(profile, operation, "output", 0);
    GGML_ASSERT(weight_tensor != nullptr);
    GGML_ASSERT(weight_tensor->data_type == "QNN_DATATYPE_SFIXED_POINT_16");
    GGML_ASSERT(weight_tensor->element_bytes == sizeof(int16_t));
    GGML_ASSERT(weight_tensor->dimensions.size() == 2);
    GGML_ASSERT(weight_tensor->qparams.encoding ==
        LLAMA_QNN_QUANTIZATION_SCALE_OFFSET);
    GGML_ASSERT(weight_tensor->qparams.scale_offsets.size() == 1);
    GGML_ASSERT(operation->matmul_product_to_output_q31 > 0);

    const int64_t input_dimension = weight_tensor->dimensions[0];
    const int64_t output_dimension = weight_tensor->dimensions[1];
    GGML_ASSERT(input->ne[0] == input_dimension);
    GGML_ASSERT(weight_tensor->static_data.size() ==
        static_cast<size_t>(input_dimension * output_dimension) * sizeof(int16_t));
    const auto * static_weights =
        reinterpret_cast<const int16_t *>(weight_tensor->static_data.data());
    bool is_qnn_hadamard = input_dimension == 128 && output_dimension == 128 &&
        weight_tensor->qparams.scale_offsets[0].zero_point == 0;
    for (int64_t index = 0;
         is_qnn_hadamard && index < input_dimension * output_dimension;
         ++index) {
        is_qnn_hadamard = static_weights[index] == INT16_MAX ||
            static_weights[index] == -INT16_MAX;
    }
    GGML_ASSERT(is_qnn_hadamard &&
        "QNN Q/K rotation matrix no longer matches the validated HTP contract");
    const double hadamard_ratio =
        (
            static_cast<double>(
                input_tensor->qparams.scale_offsets[0].scale) *
            static_cast<double>(
                weight_tensor->qparams.scale_offsets[0].scale)
        ) /
        static_cast<double>(
            output_tensor->qparams.scale_offsets[0].scale);
    const float hadamard_reduced_scale =
        static_cast<float>(std::ldexp(hadamard_ratio, 16));
    const double hadamard_q24 =
        std::ldexp(static_cast<double>(hadamard_reduced_scale), 24);
    GGML_ASSERT(std::isfinite(hadamard_q24) &&
        hadamard_q24 > 0.0 &&
        hadamard_q24 <=
            static_cast<double>(std::numeric_limits<int64_t>::max()));
    const qnn_u16_s16_matmul_params value {
        static_weights,
        input_dimension,
        output_dimension,
        input_tensor->qparams.scale_offsets[0].zero_point,
        weight_tensor->qparams.scale_offsets[0].zero_point,
        static_cast<int64_t>(std::llround(hadamard_q24)),
        output_tensor->qparams.scale_offsets[0].zero_point,
        7,
        16,
    };
    qnn_u16_s16_matmul_params * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { input };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16,
        output_dimension, input->ne[1], input->ne[2], input->ne[3],
        args, 1, qnn_u16_s16_matmul_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_qk_rotate_heads(
        ggml_context * ctx,
        ggml_tensor * input,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * const * operations,
        int32_t n_head) {
    GGML_ASSERT(ctx != nullptr && input != nullptr && profile != nullptr);
    GGML_ASSERT(operations != nullptr && n_head > 0);
    GGML_ASSERT(input->type == GGML_TYPE_U16 && input->ne[1] == n_head);
    auto * values = static_cast<qnn_u16_s16_matmul_params *>(
        ggml_new_buffer(ctx, sizeof(qnn_u16_s16_matmul_params) * n_head));
    for (int32_t head = 0; head < n_head; ++head) {
        values[head] = make_qk_rotate_params(profile, operations[head], input->ne[0]);
    }
    const qnn_u16_head_params<qnn_u16_s16_matmul_params> value { values, n_head };
    auto * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { input };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        args, 1, qnn_u16_s16_matmul_heads_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_to_u8(
        ggml_context * ctx,
        ggml_tensor * input,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation) {
    GGML_ASSERT(ctx != nullptr && input != nullptr && profile != nullptr);
    GGML_ASSERT(operation != nullptr && operation->type_name == "Convert");
    GGML_ASSERT(input->type == GGML_TYPE_U16);
    GGML_ASSERT(operation->unary_input_to_output_q31 > 0);
    const llama_qnn_u16_tensor * input_tensor =
        require_u16_operand(profile, operation, "input", 0);
    const llama_qnn_aux_quantized_tensor * output_tensor =
        profile->find_aux_operand(*operation, "output", 0);
    GGML_ASSERT(output_tensor != nullptr);
    GGML_ASSERT(output_tensor->data_type == "QNN_DATATYPE_UFIXED_POINT_8");
    GGML_ASSERT(output_tensor->element_bytes == sizeof(uint8_t));
    GGML_ASSERT(output_tensor->qparams.encoding ==
        LLAMA_QNN_QUANTIZATION_SCALE_OFFSET);
    GGML_ASSERT(output_tensor->qparams.scale_offsets.size() == 1);

    const qnn_u16_to_u8_params value {
        input_tensor->qparams.scale_offsets[0].zero_point,
        operation->unary_input_to_output_q31,
        output_tensor->qparams.scale_offsets[0].zero_point,
    };
    qnn_u16_to_u8_params * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { input };
    return ggml_custom_4d(
        ctx, GGML_TYPE_I8,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        args, 1, qnn_u16_to_u8_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_to_u8_heads(
        ggml_context * ctx,
        ggml_tensor * input,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * const * operations,
        int32_t n_head) {
    GGML_ASSERT(ctx != nullptr && input != nullptr && profile != nullptr);
    GGML_ASSERT(operations != nullptr && n_head > 0);
    GGML_ASSERT(input->type == GGML_TYPE_U16 && input->ne[1] == n_head);
    auto * values = static_cast<qnn_u16_to_u8_params *>(
        ggml_new_buffer(ctx, sizeof(qnn_u16_to_u8_params) * n_head));
    for (int32_t head = 0; head < n_head; ++head) {
        values[head] = make_to_u8_params(profile, operations[head]);
    }
    const qnn_u16_head_params<qnn_u16_to_u8_params> value { values, n_head };
    auto * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { input };
    return ggml_custom_4d(
        ctx, GGML_TYPE_I8,
        input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        args, 1, qnn_u16_to_u8_heads_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_u8_matmul(
        ggml_context * ctx,
        ggml_tensor * input,
        ggml_tensor * u8_matrix,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation) {
    GGML_ASSERT(ctx != nullptr && input != nullptr && u8_matrix != nullptr);
    GGML_ASSERT(profile != nullptr && operation != nullptr);
    GGML_ASSERT(operation->type_name == "MatMul");
    GGML_ASSERT(input->type == GGML_TYPE_U16 && u8_matrix->type == GGML_TYPE_I8);
    GGML_ASSERT(input->ne[0] == u8_matrix->ne[1]);
    GGML_ASSERT(operation->matmul_product_to_output_q31 > 0);
    const llama_qnn_u16_tensor * input_tensor =
        require_u16_operand(profile, operation, "input", 0);
    const llama_qnn_aux_quantized_tensor * matrix_tensor =
        profile->find_aux_operand(*operation, "input", 1);
    const llama_qnn_u16_tensor * output_tensor =
        require_u16_operand(profile, operation, "output", 0);
    GGML_ASSERT(matrix_tensor != nullptr);
    GGML_ASSERT(matrix_tensor->data_type == "QNN_DATATYPE_UFIXED_POINT_8");
    GGML_ASSERT(matrix_tensor->element_bytes == sizeof(uint8_t));
    GGML_ASSERT(matrix_tensor->qparams.encoding ==
        LLAMA_QNN_QUANTIZATION_SCALE_OFFSET);
    GGML_ASSERT(matrix_tensor->qparams.scale_offsets.size() == 1);

    const qnn_u16_u8_matmul_params value {
        input_tensor->qparams.scale_offsets[0].zero_point,
        matrix_tensor->qparams.scale_offsets[0].zero_point,
        operation->matmul_product_to_output_q31,
        output_tensor->qparams.scale_offsets[0].zero_point,
    };
    qnn_u16_u8_matmul_params * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { input, u8_matrix };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16,
        u8_matrix->ne[0], input->ne[1], input->ne[2], input->ne[3],
        args, 2, qnn_u16_u8_matmul_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_divide_static(
        ggml_context * ctx,
        ggml_tensor * input,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation) {
    GGML_ASSERT(ctx != nullptr && input != nullptr && profile != nullptr);
    GGML_ASSERT(operation != nullptr && operation->type_name == "ElementWiseDivide");
    GGML_ASSERT(input->type == GGML_TYPE_U16);
    GGML_ASSERT(operation->unary_input_to_output_q20 > 0);
    const auto * input_tensor = require_u16_operand(profile, operation, "input", 0);
    const auto * output_tensor = require_u16_operand(profile, operation, "output", 0);
    const qnn_u16_unary_requant_params value {
        input_tensor->qparams.scale_offsets[0].zero_point,
        operation->unary_input_to_output_q20,
        output_tensor->qparams.scale_offsets[0].zero_point,
    };
    auto * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { input };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16, input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        args, 1, qnn_u16_unary_requant_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_reduce_min(
        ggml_context * ctx,
        ggml_tensor * input,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation) {
    GGML_ASSERT(ctx != nullptr && input != nullptr && profile != nullptr);
    GGML_ASSERT(operation != nullptr && operation->type_name == "ReduceMin");
    GGML_ASSERT(input->type == GGML_TYPE_U16);
    GGML_ASSERT(operation->unary_input_to_output_q20 > 0);
    const auto * input_tensor = require_u16_operand(profile, operation, "input", 0);
    const auto * output_tensor = require_u16_operand(profile, operation, "output", 0);
    const qnn_u16_unary_requant_params value {
        input_tensor->qparams.scale_offsets[0].zero_point,
        operation->unary_input_to_output_q20,
        output_tensor->qparams.scale_offsets[0].zero_point,
    };
    auto * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { input };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16, 1, input->ne[1], input->ne[2], input->ne[3],
        args, 1, qnn_u16_reduce_min_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_select(
        ggml_context * ctx,
        ggml_tensor * condition,
        ggml_tensor * when_true,
        ggml_tensor * when_false,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation) {
    GGML_ASSERT(ctx != nullptr && condition != nullptr && when_true != nullptr);
    GGML_ASSERT(when_false != nullptr && profile != nullptr && operation != nullptr);
    GGML_ASSERT(operation->type_name == "ElementWiseSelect");
    GGML_ASSERT(operation->input_to_output_q20.size() >= 3);
    const auto * true_tensor = require_u16_operand(profile, operation, "input", 1);
    const auto * false_tensor = require_u16_operand(profile, operation, "input", 2);
    const auto * output_tensor = require_u16_operand(profile, operation, "output", 0);
    const qnn_u16_scalefactor_q15 true_scalefactor =
        qnn_u16_htp_scalefactor_q15(true_tensor, output_tensor);
    const qnn_u16_scalefactor_q15 false_scalefactor =
        qnn_u16_htp_scalefactor_q15(false_tensor, output_tensor);
    const qnn_u16_select_params value {
        true_tensor->qparams.scale_offsets[0].zero_point,
        true_scalefactor.multiplier,
        true_scalefactor.right_shift,
        false_tensor->qparams.scale_offsets[0].zero_point,
        false_scalefactor.multiplier,
        false_scalefactor.right_shift,
        output_tensor->qparams.scale_offsets[0].zero_point,
    };
    auto * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { condition, when_true, when_false };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16, when_true->ne[0], when_true->ne[1],
        when_true->ne[2], when_true->ne[3], args, 3,
        qnn_u16_select_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_softmax(
        ggml_context * ctx,
        ggml_tensor * input,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation) {
    GGML_ASSERT(ctx != nullptr && input != nullptr && profile != nullptr);
    GGML_ASSERT(operation != nullptr && operation->type_name == "Softmax");
    GGML_ASSERT(input->type == GGML_TYPE_U16);
    GGML_ASSERT(operation->softmax_scale_over_ln2_q24 > 0);
    GGML_ASSERT(operation->softmax_unit_code > 0);
    GGML_ASSERT(operation->softmax_exp2_lut_q31.size() == 257);
    const auto * output_tensor = require_u16_operand(profile, operation, "output", 0);
    const qnn_u16_softmax_params value {
        operation->softmax_scale_over_ln2_q24,
        operation->softmax_unit_code,
        output_tensor->qparams.scale_offsets[0].zero_point,
        operation->softmax_exp2_lut_q31.data(),
    };
    auto * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { input };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16, input->ne[0], input->ne[1], input->ne[2], input->ne[3],
        args, 1, qnn_u16_softmax_compute, GGML_N_TASKS_MAX, params);
}

ggml_tensor * llama_qnn_u16_attention_softmax(
        ggml_context * ctx,
        ggml_tensor * score,
        ggml_tensor * condition,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * divide_operation,
        const llama_qnn_operation * minimum_operation,
        const llama_qnn_operation * floor_add_operation,
        const llama_qnn_operation * select_operation,
        const llama_qnn_operation * softmax_operation) {
    GGML_ASSERT(ctx != nullptr && score != nullptr && condition != nullptr);
    GGML_ASSERT(profile != nullptr && divide_operation != nullptr);
    GGML_ASSERT(minimum_operation != nullptr && floor_add_operation != nullptr);
    GGML_ASSERT(select_operation != nullptr && softmax_operation != nullptr);
    GGML_ASSERT(score->type == GGML_TYPE_U16 && condition->type == GGML_TYPE_I8);
    GGML_ASSERT(ggml_are_same_shape(score, condition));
    GGML_ASSERT(divide_operation->type_name == "ElementWiseDivide");
    GGML_ASSERT(minimum_operation->type_name == "ReduceMin");
    GGML_ASSERT(floor_add_operation->type_name == "ElementWiseAdd");
    GGML_ASSERT(select_operation->type_name == "ElementWiseSelect");
    GGML_ASSERT(softmax_operation->type_name == "Softmax");

    const auto * divide_input =
        require_u16_operand(profile, divide_operation, "input", 0);
    const auto * divide_output =
        require_u16_operand(profile, divide_operation, "output", 0);
    const auto * minimum_input =
        require_u16_operand(profile, minimum_operation, "input", 0);
    const auto * minimum_output =
        require_u16_operand(profile, minimum_operation, "output", 0);
    const auto * floor_lhs =
        require_u16_operand(profile, floor_add_operation, "input", 0);
    const auto * floor_rhs =
        require_u16_operand(profile, floor_add_operation, "input", 1);
    const auto * floor_output =
        require_u16_operand(profile, floor_add_operation, "output", 0);
    const auto * select_true =
        require_u16_operand(profile, select_operation, "input", 1);
    const auto * select_false =
        require_u16_operand(profile, select_operation, "input", 2);
    const auto * select_output =
        require_u16_operand(profile, select_operation, "output", 0);
    const auto * softmax_output =
        require_u16_operand(profile, softmax_operation, "output", 0);
    GGML_ASSERT(divide_operation->unary_input_to_output_q20 > 0);
    GGML_ASSERT(minimum_operation->unary_input_to_output_q20 > 0);
    GGML_ASSERT(floor_add_operation->input_to_output_q20.size() >= 2);
    GGML_ASSERT(floor_rhs->static_data.size() == 1);
    GGML_ASSERT(softmax_operation->softmax_scale_over_ln2_q24 > 0);
    GGML_ASSERT(softmax_operation->softmax_unit_code > 0);
    GGML_ASSERT(softmax_operation->softmax_exp2_lut_q31.size() == 257);

    const qnn_u16_scalefactor_q15 true_scalefactor =
        qnn_u16_htp_scalefactor_q15(select_true, select_output);
    const qnn_u16_scalefactor_q15 false_scalefactor =
        qnn_u16_htp_scalefactor_q15(select_false, select_output);
    const qnn_u16_attention_softmax_params value {
        {
            divide_input->qparams.scale_offsets[0].zero_point,
            divide_operation->unary_input_to_output_q20,
            divide_output->qparams.scale_offsets[0].zero_point,
        },
        {
            minimum_input->qparams.scale_offsets[0].zero_point,
            minimum_operation->unary_input_to_output_q20,
            minimum_output->qparams.scale_offsets[0].zero_point,
        },
        {
            {
                floor_add_operation->input_to_output_q20[0],
                floor_add_operation->input_to_output_q20[1],
                floor_add_operation->product_requant_nudge_q31,
                floor_lhs->qparams.scale_offsets[0].zero_point,
                floor_rhs->qparams.scale_offsets[0].zero_point,
                floor_output->qparams.scale_offsets[0].zero_point,
                false,
            },
            floor_rhs->static_data[0],
            0,
            0,
        },
        {
            select_true->qparams.scale_offsets[0].zero_point,
            true_scalefactor.multiplier,
            true_scalefactor.right_shift,
            select_false->qparams.scale_offsets[0].zero_point,
            false_scalefactor.multiplier,
            false_scalefactor.right_shift,
            select_output->qparams.scale_offsets[0].zero_point,
        },
        {
            softmax_operation->softmax_scale_over_ln2_q24,
            softmax_operation->softmax_unit_code,
            softmax_output->qparams.scale_offsets[0].zero_point,
            softmax_operation->softmax_exp2_lut_q31.data(),
        },
    };
    auto * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { score, condition };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16,
        score->ne[0], score->ne[1], score->ne[2], score->ne[3],
        args, 2, qnn_u16_attention_softmax_compute, GGML_N_TASKS_MAX, params);
}

namespace {

qnn_u16_u8_matmul_params make_u16_u8_matmul_params(
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation) {
    GGML_ASSERT(profile != nullptr && operation != nullptr);
    GGML_ASSERT(operation->type_name == "MatMul");
    GGML_ASSERT(operation->matmul_product_to_output_q31 > 0);
    const llama_qnn_u16_tensor * input_tensor =
        require_u16_operand(profile, operation, "input", 0);
    const llama_qnn_aux_quantized_tensor * matrix_tensor =
        profile->find_aux_operand(*operation, "input", 1);
    const llama_qnn_u16_tensor * output_tensor =
        require_u16_operand(profile, operation, "output", 0);
    GGML_ASSERT(matrix_tensor != nullptr);
    GGML_ASSERT(matrix_tensor->data_type == "QNN_DATATYPE_UFIXED_POINT_8");
    GGML_ASSERT(matrix_tensor->element_bytes == sizeof(uint8_t));
    GGML_ASSERT(matrix_tensor->qparams.encoding ==
        LLAMA_QNN_QUANTIZATION_SCALE_OFFSET);
    GGML_ASSERT(matrix_tensor->qparams.scale_offsets.size() == 1);
    return {
        input_tensor->qparams.scale_offsets[0].zero_point,
        matrix_tensor->qparams.scale_offsets[0].zero_point,
        operation->matmul_product_to_output_q31,
        output_tensor->qparams.scale_offsets[0].zero_point,
    };
}

qnn_u16_attention_softmax_params make_attention_softmax_params(
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * divide_operation,
        const llama_qnn_operation * minimum_operation,
        const llama_qnn_operation * floor_add_operation,
        const llama_qnn_operation * select_operation,
        const llama_qnn_operation * softmax_operation) {
    GGML_ASSERT(profile != nullptr && divide_operation != nullptr);
    GGML_ASSERT(minimum_operation != nullptr && floor_add_operation != nullptr);
    GGML_ASSERT(select_operation != nullptr && softmax_operation != nullptr);
    GGML_ASSERT(divide_operation->type_name == "ElementWiseDivide");
    GGML_ASSERT(minimum_operation->type_name == "ReduceMin");
    GGML_ASSERT(floor_add_operation->type_name == "ElementWiseAdd");
    GGML_ASSERT(select_operation->type_name == "ElementWiseSelect");
    GGML_ASSERT(softmax_operation->type_name == "Softmax");

    const auto * divide_input =
        require_u16_operand(profile, divide_operation, "input", 0);
    const auto * divide_output =
        require_u16_operand(profile, divide_operation, "output", 0);
    const auto * minimum_input =
        require_u16_operand(profile, minimum_operation, "input", 0);
    const auto * minimum_output =
        require_u16_operand(profile, minimum_operation, "output", 0);
    const auto * floor_lhs =
        require_u16_operand(profile, floor_add_operation, "input", 0);
    const auto * floor_rhs =
        require_u16_operand(profile, floor_add_operation, "input", 1);
    const auto * floor_output =
        require_u16_operand(profile, floor_add_operation, "output", 0);
    const auto * select_true =
        require_u16_operand(profile, select_operation, "input", 1);
    const auto * select_false =
        require_u16_operand(profile, select_operation, "input", 2);
    const auto * select_output =
        require_u16_operand(profile, select_operation, "output", 0);
    const auto * softmax_output =
        require_u16_operand(profile, softmax_operation, "output", 0);
    GGML_ASSERT(divide_operation->unary_input_to_output_q20 > 0);
    GGML_ASSERT(minimum_operation->unary_input_to_output_q20 > 0);
    GGML_ASSERT(floor_add_operation->input_to_output_q20.size() >= 2);
    GGML_ASSERT(floor_rhs->static_data.size() == 1);
    GGML_ASSERT(softmax_operation->softmax_scale_over_ln2_q24 > 0);
    GGML_ASSERT(softmax_operation->softmax_unit_code > 0);
    GGML_ASSERT(softmax_operation->softmax_exp2_lut_q31.size() == 257);

    const qnn_u16_scalefactor_q15 true_scalefactor =
        qnn_u16_htp_scalefactor_q15(select_true, select_output);
    const qnn_u16_scalefactor_q15 false_scalefactor =
        qnn_u16_htp_scalefactor_q15(select_false, select_output);
    return {
        {
            divide_input->qparams.scale_offsets[0].zero_point,
            divide_operation->unary_input_to_output_q20,
            divide_output->qparams.scale_offsets[0].zero_point,
        },
        {
            minimum_input->qparams.scale_offsets[0].zero_point,
            minimum_operation->unary_input_to_output_q20,
            minimum_output->qparams.scale_offsets[0].zero_point,
        },
        {
            {
                floor_add_operation->input_to_output_q20[0],
                floor_add_operation->input_to_output_q20[1],
                floor_add_operation->product_requant_nudge_q31,
                floor_lhs->qparams.scale_offsets[0].zero_point,
                floor_rhs->qparams.scale_offsets[0].zero_point,
                floor_output->qparams.scale_offsets[0].zero_point,
                false,
            },
            floor_rhs->static_data[0],
            0,
            0,
        },
        {
            select_true->qparams.scale_offsets[0].zero_point,
            true_scalefactor.multiplier,
            true_scalefactor.right_shift,
            select_false->qparams.scale_offsets[0].zero_point,
            false_scalefactor.multiplier,
            false_scalefactor.right_shift,
            select_output->qparams.scale_offsets[0].zero_point,
        },
        {
            softmax_operation->softmax_scale_over_ln2_q24,
            softmax_operation->softmax_unit_code,
            softmax_output->qparams.scale_offsets[0].zero_point,
            softmax_operation->softmax_exp2_lut_q31.data(),
        },
    };
}

} // namespace

ggml_tensor * llama_qnn_u16_attention(
        ggml_context * ctx,
        ggml_tensor * query,
        ggml_tensor * key_cache,
        ggml_tensor * value_cache,
        ggml_tensor * condition,
        const llama_qnn_quant_profile * profile,
        const llama_qnn_u16_attention_head_ops * head_ops,
        int32_t n_head,
        int32_t n_head_kv) {
    GGML_ASSERT(ctx != nullptr && query != nullptr && key_cache != nullptr);
    GGML_ASSERT(value_cache != nullptr && condition != nullptr && profile != nullptr);
    GGML_ASSERT(head_ops != nullptr);
    GGML_ASSERT(query->type == GGML_TYPE_U16);
    GGML_ASSERT(key_cache->type == GGML_TYPE_I8 && value_cache->type == GGML_TYPE_I8);
    GGML_ASSERT(condition->type == GGML_TYPE_I8);
    GGML_ASSERT(n_head > 0 && n_head_kv > 0 && n_head % n_head_kv == 0);
    GGML_ASSERT(query->ne[1] == n_head);
    GGML_ASSERT(key_cache->ne[0] == query->ne[0] && key_cache->ne[2] == n_head_kv);
    GGML_ASSERT(value_cache->ne[0] == query->ne[0] && value_cache->ne[1] == n_head_kv);
    GGML_ASSERT(value_cache->ne[2] == key_cache->ne[1]);
    GGML_ASSERT(condition->ne[0] == key_cache->ne[1]);
    GGML_ASSERT(condition->ne[1] == query->ne[2]);

    auto * heads = static_cast<qnn_u16_attention_head_params *>(
        ggml_new_buffer(ctx, sizeof(qnn_u16_attention_head_params) * n_head));
    for (int32_t head = 0; head < n_head; ++head) {
        const auto & ops = head_ops[head];
        heads[head] = {
            make_u16_u8_matmul_params(profile, ops.score),
            make_attention_softmax_params(
                profile, ops.divide, ops.minimum, ops.floor_add,
                ops.select, ops.softmax),
            make_u16_u8_matmul_params(profile, ops.value),
        };
    }

    const qnn_u16_attention_params value {
        heads,
        n_head,
        n_head_kv,
    };
    auto * params = new_qnn_u16_params(ctx, value);
    ggml_tensor * args[] = { query, key_cache, value_cache, condition };
    return ggml_custom_4d(
        ctx, GGML_TYPE_U16,
        query->ne[0], query->ne[1], query->ne[2], query->ne[3],
        args, 4, qnn_u16_attention_compute, GGML_N_TASKS_MAX, params);
}
