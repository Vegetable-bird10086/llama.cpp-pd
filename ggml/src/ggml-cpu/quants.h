#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML CPU internal header

#ifdef __cplusplus
extern "C" {
#endif

// Quantization
void quantize_row_q1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void quantize_row_mxfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_nvfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void quantize_row_q2_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q3_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void quantize_row_tq1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_tq2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void quantize_row_iq4_nl (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq4_xs (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

// Dot product
void ggml_vec_dot_q1_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_mxfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_nvfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_tq1_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_tq2_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_gptq2_32_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_gptq2_64_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_gptq2_128_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// QNN-compatible GPTQ2 primitive for the opt-in U16 activation decode path.
// The source weights stay packed: the implementation expands one 32-code INT2
// group only in SIMD registers, then requantizes directly to one U16 code.
void ggml_vec_dot_gptq2_32_u16_qnn(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        float activation_scale,
        int32_t activation_zero_point,
        float output_scale,
        int32_t output_zero_point);

void ggml_vec_dot_gptq2_64_u16_qnn(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        float activation_scale,
        int32_t activation_zero_point,
        float output_scale,
        int32_t output_zero_point);

void ggml_vec_dot_gptq2_128_u16_qnn(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        float activation_scale,
        int32_t activation_zero_point,
        float output_scale,
        int32_t output_zero_point);

// Fixed-point form of the same primitive. `activation_to_output_q20` is
// precomputed at graph construction as round((activation_scale / output_scale)
// * 2^20). The packed FP16 group metadata is decoded directly to Q20 in the
// kernel, so no F32 activation or expanded INT4 weight storage is used.
void ggml_vec_dot_gptq2_32_u16_qnn_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        int64_t activation_to_output_q20,
        int32_t activation_zero_point,
        int32_t output_zero_point);

void ggml_vec_dot_gptq2_64_u16_qnn_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        int64_t activation_to_output_q20,
        int32_t activation_zero_point,
        int32_t output_zero_point);

void ggml_vec_dot_gptq2_128_u16_qnn_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        int64_t activation_to_output_q20,
        int32_t activation_zero_point,
        int32_t output_zero_point);

// Exact QNN blockwise-expansion form.  QNN quantizes each output channel's
// GS32 scales into a per-channel base scale and one U8 block-scale code.  The
// source INT2 codes remain packed in memory and are widened only in registers.
void ggml_vec_dot_gptq2_32_u16_qnn_blockwise_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point);

void ggml_vec_dot_gptq2_64_u16_qnn_blockwise_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point);

void ggml_vec_dot_gptq2_128_u16_qnn_blockwise_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point);

// Benchmark/prototyping entry point for block metadata prepared once at model
// load. Bits [4:0] retain the decoded QNN block scale code (range 1..16) and
// bits [6:5] carry the source INT2 zero point. Weight codes remain packed INT2.
void ggml_gptq2_prepare_qnn_block_codes(
        int n,
        uint8_t * prepared_block_codes,
        const void * GGML_RESTRICT packed_weights,
        const uint8_t * block_scale_codes,
        int group_size);

// Materialize a small row window from the persistent gs32_source_v1 layout
// into the ordinary row-major GPTQ2_32 block layout. This is intended for
// per-thread Decode scratch, never for rewriting or duplicating a tensor.
void ggml_gptq2_32_gs32_restore_rows(
        int n,
        const void * GGML_RESTRICT gs32_weights,
        int64_t first_row,
        int row_count,
        void * GGML_RESTRICT row_major_weights,
        size_t row_major_stride);

// Sum of the prepared, block-scaled centered INT2 weights for one GS32 row.
// This is independent of activations and should be computed once at model load.
int64_t ggml_gptq2_32_qnn_prepared_weight_sum(
        int n,
        const void * GGML_RESTRICT packed_weights,
        const uint8_t * GGML_RESTRICT prepared_block_codes);

void ggml_vec_dot_gptq2_32_u16_qnn_blockwise_prepared(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point);

void ggml_vec_dot_gptq2_64_u16_qnn_blockwise_prepared(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point);

void ggml_vec_dot_gptq2_128_u16_qnn_blockwise_prepared(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point);

// Experimental QNN finite-precision block accumulation. The source INT2
// weights remain packed; block products are shifted directly in accumulators.
void ggml_vec_dot_gptq2_u16_qnn_blockwise_requant(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int group_size,
        int metadata_prepared,
        int fractional_constant);

// Generic blockwise affine path. Intermediate block shifts always floor;
// final_round_to_nearest affects only the final output requant.
void ggml_vec_dot_gptq2_u16_qnn_blockwise_affine(
        int n,
        uint16_t * GGML_RESTRICT output,
        const void * GGML_RESTRICT packed_weights,
        const uint16_t * GGML_RESTRICT activations,
        const uint8_t * GGML_RESTRICT block_scale_codes,
        int64_t channel_scale_to_output_q31,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int group_size,
        int metadata_prepared,
        int fractional_constant,
        int final_round_to_nearest,
        int32_t output_bias_q7);

// Four-row GS32 GEMV for decode. Activations are loaded and centered once per
// block, while source INT2 weights are expanded only in NEON registers.
void ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_4rows(
        int n,
        uint16_t * GGML_RESTRICT outputs,
        const void * GGML_RESTRICT packed_weights,
        size_t weight_row_stride,
        const uint16_t * GGML_RESTRICT activations,
        const int32_t * GGML_RESTRICT activation_block_sums,
        int activations_fit_i16,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        size_t prepared_row_stride,
        const int64_t * GGML_RESTRICT channel_scale_to_output_q31,
        const int64_t * GGML_RESTRICT prepared_weight_sums,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int fractional_constant,
        int final_round_to_nearest,
        int32_t output_bias_q7);

// Eight-row GS32 GEMV for decode. The block activation load and centering are
// shared by two sequential four-row micro-kernels to avoid register spilling.
void ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_8rows(
        int n,
        uint16_t * GGML_RESTRICT outputs,
        const void * GGML_RESTRICT packed_weights,
        size_t weight_row_stride,
        const uint16_t * GGML_RESTRICT activations,
        const int32_t * GGML_RESTRICT activation_block_sums,
        int activations_fit_i16,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        size_t prepared_row_stride,
        const int64_t * GGML_RESTRICT channel_scale_to_output_q31,
        const int64_t * GGML_RESTRICT prepared_weight_sums,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int fractional_constant,
        int final_round_to_nearest,
        int32_t output_bias_q7);

// Eight-row Decode GEMV that consumes gs32_source_v1 in place. Each source
// group exposes four contiguous 16-byte qcode-pair tiles for the eight rows;
// no row-major tensor or row window is materialized.
int ggml_gptq2_32_gs32_dotprod_enabled(void);

void ggml_vec_dot_gptq2_32_gs32_u16_qnn_blockwise_affine_8rows(
        int n,
        uint16_t * GGML_RESTRICT outputs,
        const void * GGML_RESTRICT gs32_weights,
        int64_t first_row,
        const uint16_t * GGML_RESTRICT activations,
        const int32_t * GGML_RESTRICT activation_block_sums,
        int activations_fit_i16,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        size_t prepared_row_stride,
        const int64_t * GGML_RESTRICT channel_scale_to_output_q31,
        const int64_t * GGML_RESTRICT prepared_weight_sums,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int fractional_constant,
        int final_round_to_nearest,
        int32_t output_bias_q7);

// Sixteen-row experiment: four sequential four-row micro-kernels reuse one
// centered activation block. The non-S16 path falls back to two 8-row calls.
void ggml_vec_dot_gptq2_32_u16_qnn_blockwise_affine_16rows(
        int n,
        uint16_t * GGML_RESTRICT outputs,
        const void * GGML_RESTRICT packed_weights,
        size_t weight_row_stride,
        const uint16_t * GGML_RESTRICT activations,
        const int32_t * GGML_RESTRICT activation_block_sums,
        int activations_fit_i16,
        const uint8_t * GGML_RESTRICT prepared_block_codes,
        size_t prepared_row_stride,
        const int64_t * GGML_RESTRICT channel_scale_to_output_q31,
        const int64_t * GGML_RESTRICT prepared_weight_sums,
        int32_t activation_zero_point,
        int32_t output_zero_point,
        int fractional_constant,
        int final_round_to_nearest,
        int32_t output_bias_q7);

// Affine residual add for the same opt-in U16 activation path. Both inputs
// and the result remain U16 codes; conversion between qparam domains uses
// fixed-point arithmetic.
void ggml_vec_add_affine_u16_qnn(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        float lhs_scale,
        int32_t lhs_zero_point,
        const uint16_t * GGML_RESTRICT rhs,
        float rhs_scale,
        int32_t rhs_zero_point,
        float output_scale,
        int32_t output_zero_point);

void ggml_vec_add_affine_u16_qnn_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        int64_t lhs_to_output_q20,
        int32_t lhs_zero_point,
        const uint16_t * GGML_RESTRICT rhs,
        int64_t rhs_to_output_q20,
        int32_t rhs_zero_point,
        int32_t output_zero_point);

void ggml_vec_add_affine_u16_qnn_q15(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        int64_t lhs_to_output_q15,
        int32_t lhs_zero_point,
        const uint16_t * GGML_RESTRICT rhs,
        int64_t rhs_to_output_q15,
        int32_t rhs_zero_point,
        int32_t output_zero_point);

void ggml_vec_mul_affine_u16_qnn_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        int32_t lhs_zero_point,
        const uint16_t * GGML_RESTRICT rhs,
        int32_t rhs_zero_point,
        int64_t product_to_output_q20,
        int32_t output_zero_point);

void ggml_vec_mul_affine_u16_qnn_q31(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        int32_t lhs_zero_point,
        const uint16_t * GGML_RESTRICT rhs,
        int32_t rhs_zero_point,
        int64_t product_to_output_q31,
        int64_t product_requant_nudge_q31,
        int32_t output_zero_point);

// Exact QNN U16 SwiGLU:
//   sigmoid = lut[gate]
//   silu    = requant(gate * sigmoid)
//   output  = requant(silu * up)
// The intermediate codes stay local so the fused graph does not materialize
// the sigmoid and SiLU tensors. ARM builds vectorize the centered products.
void ggml_vec_swiglu_u16_qnn_q31(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT gate,
        const uint16_t * GGML_RESTRICT up,
        const uint16_t * GGML_RESTRICT sigmoid_lut,
        int32_t silu_lhs_zero_point,
        int32_t silu_rhs_zero_point,
        int64_t silu_product_to_output_q31,
        int64_t silu_product_requant_nudge_q31,
        int32_t silu_output_zero_point,
        int32_t product_lhs_zero_point,
        int32_t product_rhs_zero_point,
        int64_t product_to_output_q31,
        int64_t product_requant_nudge_q31,
        int32_t product_output_zero_point);

void ggml_vec_add_affine_u16_qnn_fixed_scalar(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        int64_t lhs_to_output_q20,
        int32_t lhs_zero_point,
        uint16_t rhs_code,
        int64_t rhs_to_output_q20,
        int32_t rhs_zero_point,
        int32_t output_zero_point);

void ggml_vec_mul_affine_u16_qnn_fixed_scalar(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        int32_t lhs_zero_point,
        uint16_t rhs_code,
        int32_t rhs_zero_point,
        int64_t product_to_output_q20,
        int32_t output_zero_point);

void ggml_vec_mul_affine_u16_qnn_q31_scalar(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT lhs,
        int32_t lhs_zero_point,
        uint16_t rhs_code,
        int32_t rhs_zero_point,
        int64_t product_to_output_q31,
        int64_t product_requant_nudge_q31,
        int32_t output_zero_point);

// QNN HTP folds a static scalar multiply into a normalized signed-Q15
// multiplier and an exponent. The runtime path therefore requantizes the
// centered U16 input directly instead of multiplying two quantized codes.
void ggml_vec_mul_static_affine_u16_qnn_q15(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        int32_t input_zero_point,
        int32_t multiplier_q15,
        int32_t right_shift,
        int32_t output_zero_point);

// Fixed-point split-half RoPE matching the decomposed QNN graph. The QNN
// StridedSlice outputs may use qparams different from the RMSNorm source, so
// each half is requantized in registers before the four products.
struct ggml_u16_rope_qnn_fixed_params {
    int32_t split_input_zero_point;
    int64_t split_to_output_q31[2];
    int32_t split_output_zero_points[2];
    int32_t lhs_zero_points[4];
    int32_t table_source_zero_points[4];
    int64_t table_source_to_storage_q31[4];
    int32_t table_zero_points[4];
    // HTP ElementWiseMultiply keeps three guard bits: Q31 requant first,
    // followed by a rounded right shift of three bits.
    int64_t product_to_output_q31_shift3[4];
    int32_t product_output_zero_points[4];
    // HTP Add/Sub uses two unsigned Q15 multipliers with a shared shift.
    // It averages the raw products before applying the affine bias.
    int32_t combine_multipliers_q15[4];
    int32_t combine_shifts[2];
    int64_t combine_biases[2];
};

void ggml_vec_rope_affine_u16_qnn_fixed(
        int half_dimension,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        const uint16_t * GGML_RESTRICT cos_codes,
        const uint16_t * GGML_RESTRICT sin_codes,
        const struct ggml_u16_rope_qnn_fixed_params * params);

// Dense U16 x S16 affine MatMul. The S16 matrix is row-major
// [input_dimension, output_dimension]. The 128x128 Hadamard specialization
// reproduces QNN's A16W16 packing and S16 accumulator reduction.
void ggml_vec_matmul_u16_s16_qnn_fixed(
        int input_dimension,
        int output_dimension,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        const int16_t * GGML_RESTRICT weights,
        int32_t input_zero_point,
        int32_t weight_zero_point,
        int64_t product_to_output_q20,
        int32_t output_zero_point,
        int weight_right_shift,
        int accumulator_reduction_shift);

// Affine U16-to-U8 conversion used when QNN writes K/V projections into its
// 8-bit cache domain. The scale ratio is precomputed as Q31; runtime execution
// contains no floating-point conversion or intermediate activation buffer.
void ggml_vec_convert_u16_u8_qnn_fixed(
        int n,
        uint8_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        int32_t input_zero_point,
        int64_t input_to_output_q31,
        int32_t output_zero_point);

// Dense affine U16 x U8 MatMul shared by QK and softmax x V. The U8 matrix is
// row-major [input_dimension, output_dimension]. NEON widens source bytes only
// in registers, accumulates in S64, and requantizes directly to U16.
void ggml_vec_matmul_u16_u8_qnn_fixed(
        int input_dimension,
        int output_dimension,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        const uint8_t * GGML_RESTRICT weights,
        int32_t input_zero_point,
        int32_t weight_zero_point,
        int64_t product_to_output_q20,
        int32_t output_zero_point);

// Strided variant for QNN-layout K/V cache views. Each logical matrix row
// still stores output_dimension consecutive U8 codes, while adjacent rows may
// be separated by a larger cache stride.
void ggml_vec_matmul_u16_u8_qnn_fixed_strided(
        int input_dimension,
        int output_dimension,
        size_t weight_row_stride,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        const uint8_t * GGML_RESTRICT weights,
        int32_t input_zero_point,
        int32_t weight_zero_point,
        int64_t product_to_output_q20,
        int32_t output_zero_point);

// Two vectors multiplied by the same U8 matrix. The dot-product path shares
// every K/V weight load and weight-sum reduction across a GQA query pair.
void ggml_vec_matmul_u16_u8_qnn_fixed_strided_pair(
        int input_dimension,
        int output_dimension,
        size_t weight_row_stride,
        uint16_t * GGML_RESTRICT output0,
        uint16_t * GGML_RESTRICT output1,
        const uint16_t * GGML_RESTRICT input0,
        const uint16_t * GGML_RESTRICT input1,
        const uint8_t * GGML_RESTRICT weights,
        int32_t input_zero_point0,
        int32_t input_zero_point1,
        int32_t weight_zero_point,
        int64_t product_to_output_q20_0,
        int64_t product_to_output_q20_1,
        int32_t output_zero_point0,
        int32_t output_zero_point1);

// Token-major K cache: weights are [output_token, input_dimension].
void ggml_vec_matmul_u16_u8_qnn_fixed_token_major(
        int input_dimension,
        int output_dimension,
        size_t token_stride,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        const uint8_t * GGML_RESTRICT weights,
        int32_t input_zero_point,
        int32_t weight_zero_point,
        int64_t product_to_output_q20,
        int32_t output_zero_point);

void ggml_vec_matmul_u16_u8_qnn_fixed_token_major_pair(
        int input_dimension,
        int output_dimension,
        size_t token_stride,
        uint16_t * GGML_RESTRICT output0,
        uint16_t * GGML_RESTRICT output1,
        const uint16_t * GGML_RESTRICT input0,
        const uint16_t * GGML_RESTRICT input1,
        const uint8_t * GGML_RESTRICT weights,
        int32_t input_zero_point0,
        int32_t input_zero_point1,
        int32_t weight_zero_point,
        int64_t product_to_output_q20_0,
        int64_t product_to_output_q20_1,
        int32_t output_zero_point0,
        int32_t output_zero_point1);

// Experimental runtime-gated ARM dot-product path for dense U16 x U8
// attention matrices. The unchanged row-major cache layout requires an
// in-register transpose and signed-radix decomposition, so it is opt-in until
// it beats the portable NEON kernel. Set GGML_QNN_U16_DOTPROD=1 to enable.
int ggml_qnn_u16_dotprod_enabled(void);

void ggml_vec_requant_u16_qnn_fixed(
        int n,
        uint16_t * output,
        const uint16_t * input,
        int32_t input_zero_point,
        int64_t input_to_output_q20,
        int32_t output_zero_point);

uint16_t ggml_vec_min_u16_qnn(int n, const uint16_t * input);

void ggml_vec_select_affine_u16_qnn_fixed(
        int n,
        uint16_t * output,
        const uint8_t * GGML_RESTRICT condition,
        const uint16_t * when_true,
        int32_t true_zero_point,
        int32_t true_multiplier_q15,
        int32_t true_right_shift,
        const uint16_t * GGML_RESTRICT when_false,
        int false_stride,
        int32_t false_zero_point,
        int32_t false_multiplier_q15,
        int32_t false_right_shift,
        int32_t output_zero_point);

// Integer softmax over one contiguous U16 row. For the QNN probability
// contract (scale 1/65535, zero point 0), execution mirrors the HTP piecewise
// exponential and reciprocal iteration. Other contracts use the portable LUT
// fallback. Both paths allocate no row buffer.
void ggml_vec_softmax_u16_qnn_fixed(
        int n,
        uint16_t * output,
        const uint16_t * input,
        int64_t scale_over_ln2_q24,
        int64_t output_unit_code,
        int32_t output_zero_point,
        const uint32_t * GGML_RESTRICT exp2_lut_q31);

// RMSNorm with an affine U16 weight. The per-element path operates on U16
// codes and integer products; only the shared inverse RMS is scalar.
void ggml_vec_rms_norm_affine_u16_qnn(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        float input_scale,
        int32_t input_zero_point,
        const uint16_t * GGML_RESTRICT weight,
        float weight_scale,
        int32_t weight_zero_point,
        float epsilon,
        float output_scale,
        int32_t output_zero_point);

// Fully fixed-point RMSNorm execution form.
// `epsilon_in_codes_q16` is precomputed once from the input qparams. Runtime
// execution uses integer square root and U16 code-domain arithmetic only.
void ggml_vec_rms_norm_affine_u16_qnn_fixed(
        int n,
        uint16_t * GGML_RESTRICT output,
        const uint16_t * GGML_RESTRICT input,
        int32_t input_zero_point,
        const uint16_t * GGML_RESTRICT weight,
        int32_t weight_zero_point,
        uint64_t epsilon_in_codes_q16,
        int64_t weight_to_output_q31,
        int32_t output_zero_point);

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_m_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_nl_q8_0 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// Generic implementation
void quantize_row_q8_0_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void quantize_row_q8_1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void quantize_row_q8_K_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void ggml_vec_dot_q1_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_mxfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_nvfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_tq1_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_tq2_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_q2_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_m_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_nl_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
