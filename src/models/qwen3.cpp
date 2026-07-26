#include "models.h"

#include "llama-kv-cache.h"
#include "llama-qnn-u16.h"

#include <string>
#include <vector>

namespace {

std::string qnn_indexed_fx_name(const char * stem, int32_t index) {
    std::string result(stem);
    if (index > 0) {
        result += "_" + std::to_string(index);
    }
    return result;
}

std::string qnn_indexed_head_fx_name(
        const char * stem,
        int32_t index,
        int32_t head) {
    return qnn_indexed_fx_name(stem, index) + "_h_" + std::to_string(head);
}

const llama_qnn_operation * qnn_require_fx_operation(
        const llama_qnn_quant_profile * profile,
        int32_t layer,
        const std::string & fx_name,
        const char * type_name) {
    const llama_qnn_operation * operation =
        profile->find_operation_by_fx(layer, fx_name);
    GGML_ASSERT(operation != nullptr && operation->type_name == type_name);
    return operation;
}

const llama_qnn_linear_qparams * qnn_require_linear(
        const llama_qnn_quant_profile * profile,
        int32_t layer,
        const char * projection) {
    const llama_qnn_linear_qparams * qparams =
        profile->find_linear_qparams(layer, projection);
    GGML_ASSERT(qparams != nullptr);
    return qparams;
}

const llama_qnn_operation * qnn_require_kv_convert(
        const llama_qnn_quant_profile * profile,
        int32_t layer,
        int32_t head,
        bool key) {
    const llama_qnn_operation * producer = qnn_require_fx_operation(
        profile, layer,
        qnn_indexed_head_fx_name(
            key ? "aten_matmul_default" : "aten_convolution_default",
            key ? 4 * layer + 1 : 7 * layer + 2,
            head),
        key ? "MatMul" : "Conv2d");
    GGML_ASSERT(producer->outputs.size() == 1);
    const std::string & source_tensor = producer->outputs[0];
    const int32_t shard = layer / 2;
    for (const llama_qnn_operation & operation : profile->operations) {
        if (operation.shard_index == shard &&
                operation.type_name == "Convert" &&
                operation.inputs.size() == 1 &&
                operation.inputs[0] == source_tensor) {
            return &operation;
        }
    }
    GGML_ABORT("missing QNN U8 KV Convert for layer %d head %d (%s)",
        layer, head, key ? "K" : "V");
}

const llama_qnn_u16_tensor * qnn_require_output_qparams(
        const llama_qnn_quant_profile * profile,
        const llama_qnn_operation * operation) {
    GGML_ASSERT(profile != nullptr && operation != nullptr);
    const llama_qnn_u16_tensor * qparams =
        profile->find_u16_operand(*operation, "output", 0);
    GGML_ASSERT(qparams != nullptr);
    return qparams;
}

ggml_tensor * qnn_concat_heads(
        ggml_context * ctx,
        const std::vector<ggml_tensor *> & heads,
        int64_t head_dimension,
        int64_t tokens) {
    GGML_ASSERT(ctx != nullptr && !heads.empty());
    ggml_tensor * result = nullptr;
    for (ggml_tensor * head : heads) {
        GGML_ASSERT(head != nullptr && head->ne[0] == head_dimension);
        GGML_ASSERT(head->ne[1] == tokens && head->ne[2] == 1 && head->ne[3] == 1);
        ggml_tensor * shaped = ggml_reshape_3d(
            ctx, head, head_dimension, 1, tokens);
        result = result == nullptr ? shaped : ggml_concat(ctx, result, shaped, 1);
    }
    return result;
}

} // namespace

void llama_model_qwen3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 28: type = hparams.n_embd == 1024 ? LLM_TYPE_0_6B : LLM_TYPE_1_7B; break;
        case 36: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;
        case 40: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // output rerank head
    cls_out = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen3::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const llama_qnn_quant_profile * qnn_profile = model.get_qnn_u16_profile();
    if (qnn_profile != nullptr && llama_qnn_u16_activations_enabled()) {
        GGML_ASSERT(qnn_profile->num_decoder_layers == n_layer);
        GGML_ASSERT(n_embd_head == 128);
        GGML_ASSERT(n_head == 16 && n_head_kv == 8);
        GGML_ASSERT(loras != nullptr && loras->empty());
        GGML_ASSERT(inp_attn->mctx != nullptr);
        GGML_ASSERT(inp_attn->mctx->type_k() == GGML_TYPE_I8);
        GGML_ASSERT(inp_attn->mctx->type_v() == GGML_TYPE_I8);

        const llama_qnn_operation * input_quantize = qnn_profile->find_operation(
            0, "quantized_decomposed_quantize_per_tensor_default");
        const llama_qnn_operation * input_scale = qnn_profile->find_operation(
            0, "aten_mul_tensor");
        GGML_ASSERT(input_quantize != nullptr && input_quantize->type_name == "Quantize");
        GGML_ASSERT(input_scale != nullptr && input_scale->type_name == "ElementWiseMultiply");

        inpL = llama_qnn_quantize_f32_to_u16(
            ctx0, inpL, qnn_profile, input_quantize);
        inpL = llama_qnn_u16_mul_static(ctx0, inpL, qnn_profile, input_scale);
        cb(inpL, "qnn_u16_input", -1);
        const llama_qnn_u16_tensor * layer_input_qparams =
            qnn_require_output_qparams(qnn_profile, input_scale);

        ggml_tensor * mask_condition = llama_qnn_attention_mask_condition(
            ctx0, inp_attn->get_kq_mask());
        const int64_t n_kv = inp_attn->mctx->get_n_kv();
        GGML_ASSERT(n_kv > 0);

        const auto fx = [qnn_profile](
                int32_t layer,
                const char * stem,
                int32_t index,
                const char * type_name) {
            return qnn_require_fx_operation(
                qnn_profile, layer, qnn_indexed_fx_name(stem, index), type_name);
        };
        const auto fx_head = [qnn_profile](
                int32_t layer,
                const char * stem,
                int32_t index,
                int32_t head,
                const char * type_name) {
            return qnn_require_fx_operation(qnn_profile, layer,
                qnn_indexed_head_fx_name(stem, index, head), type_name);
        };
        const auto cb_head = [this](
                ggml_tensor * tensor,
                const char * stem,
                int32_t layer,
                int32_t head) {
            if (head == 0) {
                cb(tensor, stem, layer);
                return;
            }
            const std::string head_stem =
                std::string(stem) + "_h" + std::to_string(head);
            cb(tensor, head_stem.c_str(), layer);
        };

        for (int32_t il = 0; il < n_layer; ++il) {
            GGML_ASSERT(cvec == nullptr || cvec->tensor_for(il) == nullptr);
            // Layer-input extraction is diagnostic-only. Keep the normal
            // decode graph in U16, but expose a correctly dequantized F32
            // observer when this layer was explicitly requested by a tool.
            res->t_layer_inp[il] = cparams.embeddings_layer_inp[il]
                ? llama_qnn_dequantize_u16_to_f32(
                    ctx0, inpL, layer_input_qparams)
                : inpL;
            if (cparams.embeddings_layer_inp[il]) {
                ggml_build_forward_expand(gf, res->t_layer_inp[il]);
            }
            ggml_tensor * residual = inpL;

            ggml_tensor * attn_norm = llama_qnn_u16_rms_norm(
                ctx0, inpL, qnn_profile,
                fx(il, "aten_rms_norm_default", 4 * il, "RmsNorm"));
            cb(attn_norm, "qnn_attn_norm", il);

            ggml_tensor * q_projection = llama_qnn_u16_mul_mat(
                ctx0, model.layers[il].wq, attn_norm,
                qnn_require_linear(qnn_profile, il, "self_attn.q_proj"));
            ggml_tensor * k_projection = llama_qnn_u16_mul_mat(
                ctx0, model.layers[il].wk, attn_norm,
                qnn_require_linear(qnn_profile, il, "self_attn.k_proj"));
            ggml_tensor * v_projection = llama_qnn_u16_mul_mat(
                ctx0, model.layers[il].wv, attn_norm,
                qnn_require_linear(qnn_profile, il, "self_attn.v_proj"));
            cb(q_projection, "qnn_q_projection", il);
            cb(k_projection, "qnn_k_projection", il);
            cb(v_projection, "qnn_v_projection", il);

            std::vector<ggml_tensor *> queries;
            std::vector<ggml_tensor *> keys;
            std::vector<ggml_tensor *> values;
            queries.reserve(n_head);
            keys.reserve(n_head_kv);
            values.reserve(n_head_kv);

            for (int32_t head = 0; head < n_head; ++head) {
                ggml_tensor * query = ggml_view_2d(
                    ctx0, q_projection, n_embd_head, n_tokens,
                    q_projection->nb[1], head * n_embd_head * q_projection->nb[0]);
                query = llama_qnn_u16_rms_norm(
                    ctx0, query, qnn_profile,
                    fx_head(il, "aten_rms_norm_default", 4 * il + 1,
                        head, "RmsNorm"));
                cb_head(query, "qnn_q_head_norm", il, head);
                query = llama_qnn_u16_rope(
                    ctx0, query, inp_pos, qnn_profile, il, head, false);
                cb_head(query, "qnn_q_head_rope", il, head);
                query = llama_qnn_u16_qk_rotate(
                    ctx0, query, qnn_profile, il, head, false);
                cb_head(query, "qnn_q_head_rotate", il, head);
                queries.push_back(query);
            }

            for (int32_t head = 0; head < n_head_kv; ++head) {
                ggml_tensor * key = ggml_view_2d(
                    ctx0, k_projection, n_embd_head, n_tokens,
                    k_projection->nb[1], head * n_embd_head * k_projection->nb[0]);
                key = llama_qnn_u16_rms_norm(
                    ctx0, key, qnn_profile,
                    fx_head(il, "aten_rms_norm_default", 4 * il + 2,
                        head, "RmsNorm"));
                cb_head(key, "qnn_k_head_norm", il, head);
                key = llama_qnn_u16_rope(
                    ctx0, key, inp_pos, qnn_profile, il, head, true);
                cb_head(key, "qnn_k_head_rope", il, head);
                key = llama_qnn_u16_qk_rotate(
                    ctx0, key, qnn_profile, il, head, true);
                cb_head(key, "qnn_k_head_rotate", il, head);
                key = llama_qnn_u16_to_u8(
                    ctx0, key, qnn_profile,
                    qnn_require_kv_convert(qnn_profile, il, head, true));
                cb_head(key, "qnn_k_head_u8", il, head);
                keys.push_back(key);

                ggml_tensor * value = ggml_view_2d(
                    ctx0, v_projection, n_embd_head, n_tokens,
                    v_projection->nb[1], head * n_embd_head * v_projection->nb[0]);
                value = llama_qnn_u16_to_u8(
                    ctx0, value, qnn_profile,
                    qnn_require_kv_convert(qnn_profile, il, head, false));
                cb_head(value, "qnn_v_head_u8", il, head);
                values.push_back(value);
            }

            ggml_tensor * key_current = qnn_concat_heads(
                ctx0, keys, n_embd_head, n_tokens);
            ggml_tensor * value_current = qnn_concat_heads(
                ctx0, values, n_embd_head, n_tokens);
            ggml_build_forward_expand(gf, key_current);
            ggml_build_forward_expand(gf, value_current);
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(
                ctx0, key_current, inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(
                ctx0, value_current, inp_attn->get_v_idxs(), il));

            ggml_tensor * key_cache = inp_attn->mctx->get_k(ctx0, il);
            ggml_tensor * value_cache = inp_attn->mctx->get_v(ctx0, il);
            GGML_ASSERT(key_cache->ne[0] == n_kv);
            GGML_ASSERT(key_cache->ne[1] == n_embd_head);
            GGML_ASSERT(key_cache->ne[2] == n_head_kv);
            GGML_ASSERT(key_cache->ne[3] == 1);
            GGML_ASSERT(value_cache->ne[0] == n_embd_head);
            GGML_ASSERT(value_cache->ne[1] == n_head_kv);
            GGML_ASSERT(value_cache->ne[2] == n_kv);
            GGML_ASSERT(value_cache->ne[3] == 1);

            std::vector<ggml_tensor *> attention_heads;
            attention_heads.reserve(n_head);
            for (int32_t head = 0; head < n_head; ++head) {
                const int32_t kv_head = head / (n_head / n_head_kv);
                ggml_tensor * key_matrix = ggml_view_2d(
                    ctx0, key_cache, n_kv, n_embd_head,
                    key_cache->nb[1], kv_head * key_cache->nb[2]);
                ggml_tensor * score = llama_qnn_u16_u8_matmul(
                    ctx0, queries[head], key_matrix, qnn_profile,
                    fx_head(il, "aten_matmul_default", 4 * il + 2,
                        head, "MatMul"));
                cb_head(score, "qnn_attention_score", il, head);
                score = llama_qnn_u16_divide_static(
                    ctx0, score, qnn_profile,
                    fx_head(il, "aten_div_tensor", il, head,
                        "ElementWiseDivide"));
                cb_head(score, "qnn_attention_scaled", il, head);
                ggml_tensor * minimum = llama_qnn_u16_reduce_min(
                    ctx0, score, qnn_profile,
                    fx_head(il, "aten_amin_default", il, head, "ReduceMin"));
                cb_head(minimum, "qnn_attention_min", il, head);
                minimum = llama_qnn_u16_add_static(
                    ctx0, minimum, qnn_profile,
                    fx_head(il, "aten_add_tensor", 5 * il + 2,
                        head, "ElementWiseAdd"));
                cb_head(minimum, "qnn_attention_mask_value", il, head);
                score = llama_qnn_u16_select(
                    ctx0, mask_condition, score, minimum, qnn_profile,
                    fx_head(il, "aten_where_self", il, head,
                        "ElementWiseSelect"));
                cb_head(score, "qnn_attention_masked", il, head);
                score = llama_qnn_u16_softmax(
                    ctx0, score, qnn_profile,
                    fx_head(il, "aten__softmax_default", il, head, "Softmax"));
                cb_head(score, "qnn_attention_softmax", il, head);

                ggml_tensor * value_matrix = ggml_view_2d(
                    ctx0, value_cache, n_embd_head, n_kv,
                    value_cache->nb[2], kv_head * value_cache->nb[1]);
                ggml_tensor * attention = llama_qnn_u16_u8_matmul(
                    ctx0, score, value_matrix, qnn_profile,
                    fx_head(il, "aten_matmul_default", 4 * il + 3,
                        head, "MatMul"));
                cb_head(attention, "qnn_attention_value", il, head);
                attention_heads.push_back(attention);
            }

            ggml_tensor * attention = qnn_concat_heads(
                ctx0, attention_heads, n_embd_head, n_tokens);
            attention = ggml_reshape_2d(
                ctx0, attention, n_embd_head * n_head, n_tokens);
            cb(attention, "qnn_attention_concat", il);
            ggml_tensor * attention_output = llama_qnn_u16_mul_mat(
                ctx0, model.layers[il].wo, attention,
                qnn_require_linear(qnn_profile, il, "self_attn.o_proj"));
            cb(attention_output, "qnn_attention_output", il);
            ggml_tensor * ffn_input = llama_qnn_u16_add(
                ctx0, residual, attention_output, qnn_profile,
                fx(il, "aten_add_tensor", 5 * il + 3, "ElementWiseAdd"));
            cb(ffn_input, "qnn_ffn_input", il);

            ggml_tensor * ffn_norm = llama_qnn_u16_rms_norm(
                ctx0, ffn_input, qnn_profile,
                fx(il, "aten_rms_norm_default", 4 * il + 3, "RmsNorm"));
            cb(ffn_norm, "qnn_ffn_norm", il);
            ggml_tensor * gate = llama_qnn_u16_mul_mat(
                ctx0, model.layers[il].ffn_gate, ffn_norm,
                qnn_require_linear(qnn_profile, il, "mlp.gate_proj"));
            cb(gate, "qnn_ffn_gate", il);
            ggml_tensor * up = llama_qnn_u16_mul_mat(
                ctx0, model.layers[il].ffn_up, ffn_norm,
                qnn_require_linear(qnn_profile, il, "mlp.up_proj"));
            cb(up, "qnn_ffn_up", il);
            ggml_tensor * activated = llama_qnn_u16_sigmoid(
                ctx0, gate,
                fx(il, "aten_sigmoid_default", il, "Sigmoid"));
            cb(activated, "qnn_ffn_sigmoid", il);
            activated = llama_qnn_u16_mul(
                ctx0, gate, activated, qnn_profile,
                fx(il, "aten_mul_tensor", 10 * il + 9,
                    "ElementWiseMultiply"));
            cb(activated, "qnn_ffn_silu", il);
            activated = llama_qnn_u16_mul(
                ctx0, activated, up, qnn_profile,
                fx(il, "aten_mul_tensor", 10 * il + 10,
                    "ElementWiseMultiply"));
            cb(activated, "qnn_ffn_product", il);
            ggml_tensor * down = llama_qnn_u16_mul_mat(
                ctx0, model.layers[il].ffn_down, activated,
                qnn_require_linear(qnn_profile, il, "mlp.down_proj"));
            cb(down, "qnn_ffn_down", il);
            const llama_qnn_operation * layer_output_add =
                fx(il, "aten_add_tensor", 5 * il + 4, "ElementWiseAdd");
            inpL = llama_qnn_u16_add(
                ctx0, ffn_input, down, qnn_profile, layer_output_add);
            layer_input_qparams =
                qnn_require_output_qparams(qnn_profile, layer_output_add);
            cb(inpL, "qnn_layer_output", il);
        }

        const llama_qnn_operation * final_add = fx(
            n_layer - 1, "aten_add_tensor", 5 * (n_layer - 1) + 4,
            "ElementWiseAdd");
        cur = llama_qnn_dequantize_u16_to_f32(
            ctx0, inpL, qnn_require_output_qparams(qnn_profile, final_add));
        if (inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);
        res->t_embd = cur;
        cur = build_lora_mm(model.output, cur, model.output_s);
        cb(cur, "result_output", -1);
        res->t_logits = cur;
        ggml_build_forward_expand(gf, cur);
        return;
    }

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            // Keep the projection result distinct from the normalized and
            // RoPE-rotated forms for the QNN numerical-alignment diagnostic.
            cb(Qcur, "Qpre_norm", il);
            cb(Kcur, "Kpre_norm", il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            // The generic build_qkv() callback already labels the raw
            // projections as Qcur/Kcur. Use unique names here so numerical
            // diagnostics capture the RoPE outputs rather than those earlier
            // same-named tensors.
            cb(Qcur, "Qpost_rope", il);
            cb(Kcur, "Kpost_rope", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
