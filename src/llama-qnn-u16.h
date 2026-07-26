#pragma once

#include "llama-qnn-quant-profile.h"

struct ggml_context;
struct ggml_tensor;

bool llama_qnn_u16_activations_enabled();

// Quantize/dequantize only at the graph boundary. The decode layers between
// these two calls stay in QNN's integer activation domains.
ggml_tensor * llama_qnn_quantize_f32_to_u16(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * quantize_operation);

ggml_tensor * llama_qnn_dequantize_u16_to_f32(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_u16_tensor * input_qparams);

ggml_tensor * llama_qnn_u16_mul_mat(
    ggml_context * ctx,
    ggml_tensor * weights,
    ggml_tensor * input,
    const llama_qnn_linear_qparams * qparams);

ggml_tensor * llama_qnn_u16_add(
    ggml_context * ctx,
    ggml_tensor * lhs,
    ggml_tensor * rhs,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * operation);

ggml_tensor * llama_qnn_u16_sub(
    ggml_context * ctx,
    ggml_tensor * lhs,
    ggml_tensor * rhs,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * operation);

ggml_tensor * llama_qnn_u16_mul(
    ggml_context * ctx,
    ggml_tensor * lhs,
    ggml_tensor * rhs,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * operation);

ggml_tensor * llama_qnn_u16_add_static(
    ggml_context * ctx,
    ggml_tensor * lhs,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * operation);

ggml_tensor * llama_qnn_u16_mul_static(
    ggml_context * ctx,
    ggml_tensor * lhs,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * operation);

// Converts llama.cpp's causal mask metadata into the boolean U8 condition
// consumed by QNN ElementWiseSelect. This is not an activation dequantization.
ggml_tensor * llama_qnn_attention_mask_condition(
    ggml_context * ctx,
    ggml_tensor * mask);

ggml_tensor * llama_qnn_u16_rms_norm(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * operation);

ggml_tensor * llama_qnn_u16_sigmoid(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_operation * operation);

// Fuses the QNN split-half RoPE decomposition for one query or key head. The
// position tensor is I32 and the input/output layout is [head_dim, tokens, ...].
ggml_tensor * llama_qnn_u16_rope(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * positions,
    const llama_qnn_quant_profile * profile,
    int32_t layer_id,
    int32_t head_index,
    bool key_head);

// Applies the exact static S16 Q/K rotation MatMul used immediately after
// QNN RoPE. The matrix stays in the loaded profile; execution is U16 x S16
// integer arithmetic with direct U16 requantization.
ggml_tensor * llama_qnn_u16_qk_rotate(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_quant_profile * profile,
    int32_t layer_id,
    int32_t head_index,
    bool key_head);

// Converts one U16 activation tensor to the exact U8 affine domain declared by
// a QNN Convert operation. GGML_TYPE_I8 is used only as a one-byte container;
// the kernel and cache consumers interpret its bytes as unsigned codes.
ggml_tensor * llama_qnn_u16_to_u8(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * operation);

// Applies a QNN MatMul with U16 lhs activations and a raw U8 rhs matrix. The
// rhs tensor uses GGML_TYPE_I8 as byte storage and has shape [N,K,...].
ggml_tensor * llama_qnn_u16_u8_matmul(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * u8_matrix,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * operation);

ggml_tensor * llama_qnn_u16_divide_static(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * operation);

ggml_tensor * llama_qnn_u16_reduce_min(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * operation);

ggml_tensor * llama_qnn_u16_select(
    ggml_context * ctx,
    ggml_tensor * condition,
    ggml_tensor * when_true,
    ggml_tensor * when_false,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * operation);

ggml_tensor * llama_qnn_u16_softmax(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * operation);
