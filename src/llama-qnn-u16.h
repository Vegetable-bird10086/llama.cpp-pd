#pragma once

#include "llama-qnn-quant-profile.h"

struct ggml_context;
struct ggml_tensor;
struct llama_model;

// The graph builder supplies exported operation metadata; the kernels only
// depend on tensor shape and these descriptors. This keeps multi-head fusion
// usable by any model/export with the same integer attention contract.
struct llama_qnn_u16_rope_head_ops {
    const llama_qnn_operation * multiply[4];
    const llama_qnn_operation * slices[2];
    const llama_qnn_operation * subtract;
    const llama_qnn_operation * add;
    const llama_qnn_u16_tensor * cos_table;
    const llama_qnn_u16_tensor * sin_table;
};

struct llama_qnn_u16_attention_head_ops {
    const llama_qnn_operation * score;
    const llama_qnn_operation * divide;
    const llama_qnn_operation * minimum;
    const llama_qnn_operation * floor_add;
    const llama_qnn_operation * select;
    const llama_qnn_operation * softmax;
    const llama_qnn_operation * value;
};

bool llama_qnn_u16_activations_enabled();

// Loads and attaches the environment-selected kernel-ready profile after the
// GGUF tensors are resident. This is used by PD Decode to defer the large
// metadata mapping until Prefill has released its rebuild working set.
bool llama_qnn_u16_attach_profile_from_environment(llama_model * model);

// Converts canonical FP16 PD KV ([K,V][layer][head][token][dim]) into the
// exact per-layer/per-head U8 cache domains declared by the attached QNN
// profile. This is the bridge used when a non-QNN Prefill backend (for
// example MTK) hands real-valued KV to the QNN-aligned llama.cpp Decode path.
struct llama_qnn_kv_quantize_stats {
    size_t values = 0;
    size_t non_finite = 0;
    size_t code_zero = 0;
    size_t code_255 = 0;
};

bool llama_qnn_u16_quantize_fp16_kv(
    const llama_model * model,
    const uint16_t * fp16_kv,
    size_t fp16_values,
    int32_t num_layers,
    int32_t num_kv_heads,
    int32_t prompt_length,
    int32_t head_dim,
    uint8_t * qnn_u8_kv,
    llama_qnn_kv_quantize_stats * stats,
    std::string * error);

// Quantize/dequantize only at the graph boundary. The decode layers between
// these two calls stay in QNN's integer activation domains.
ggml_tensor * llama_qnn_quantize_f32_to_u16(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * quantize_operation);

ggml_tensor * llama_qnn_quantize_f32_to_u16(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_u16_tensor * output_qparams);

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

ggml_tensor * llama_qnn_u16_rms_norm_heads(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * const * operations,
    int32_t n_head);

ggml_tensor * llama_qnn_u16_sigmoid(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_operation * operation);

// Fuses sigmoid(gate), gate * sigmoid(gate), and SiLU(gate) * up while
// preserving both exported QNN multiply requantization boundaries.
ggml_tensor * llama_qnn_u16_swiglu(
    ggml_context * ctx,
    ggml_tensor * gate,
    ggml_tensor * up,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * sigmoid_operation,
    const llama_qnn_operation * silu_multiply_operation,
    const llama_qnn_operation * product_multiply_operation);

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

ggml_tensor * llama_qnn_u16_rope_heads(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * positions,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_u16_rope_head_ops * head_ops,
    int32_t n_head);

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

ggml_tensor * llama_qnn_u16_qk_rotate_heads(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * const * operations,
    int32_t n_head);

// Converts one U16 activation tensor to the exact U8 affine domain declared by
// a QNN Convert operation. GGML_TYPE_I8 is used only as a one-byte container;
// the kernel and cache consumers interpret its bytes as unsigned codes.
ggml_tensor * llama_qnn_u16_to_u8(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * operation);

ggml_tensor * llama_qnn_u16_to_u8_heads(
    ggml_context * ctx,
    ggml_tensor * input,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * const * operations,
    int32_t n_head);

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

// Fuses the Decode attention score post-processing chain
// Divide -> ReduceMin -> static Add -> Select -> Softmax into one CUSTOM node.
// This preserves the profiled QNN quantization boundaries while avoiding four
// extra graph scheduling barriers for every attention head.
ggml_tensor * llama_qnn_u16_attention_softmax(
    ggml_context * ctx,
    ggml_tensor * score,
    ggml_tensor * condition,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_operation * divide_operation,
    const llama_qnn_operation * minimum_operation,
    const llama_qnn_operation * floor_add_operation,
    const llama_qnn_operation * select_operation,
    const llama_qnn_operation * softmax_operation);

// Executes all query heads for one layer in a single graph node. QK, the
// exported fixed-point mask/softmax chain, and PV remain bit-compatible with
// the per-head path; work is scheduled by GQA group so the two query heads
// sharing one KV head reuse the same K/V cache region.
ggml_tensor * llama_qnn_u16_attention(
    ggml_context * ctx,
    ggml_tensor * query,
    ggml_tensor * key_cache,
    ggml_tensor * value_cache,
    ggml_tensor * condition,
    const llama_qnn_quant_profile * profile,
    const llama_qnn_u16_attention_head_ops * head_ops,
    int32_t n_head,
    int32_t n_head_kv);
