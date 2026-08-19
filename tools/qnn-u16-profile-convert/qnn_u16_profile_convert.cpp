#include "llama-qnn-quant-profile.h"
#include "llama-model.h"
#include "llama.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        std::cerr
            << "usage: qnn-u16-profile-convert INPUT.json MODEL.gguf OUTPUT.meta [SHARDS]\n"
            << "       SHARDS defaults to ceil(num_decoder_layers / 2)\n";
        return 64;
    }
    try {
        unsetenv("LLAMA_QNN_U16_QPARAMS_MANIFEST");
        const auto profile = llama_qnn_quant_profile_load_file(argv[1]);
        if (!profile) {
            std::cerr << "profile loader returned null\n";
            return 1;
        }
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.use_mmap = true;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[2], model_params),
            llama_model_free);
        if (!model) {
            std::cerr << "GGUF model loader returned null\n";
            return 1;
        }
        llama_qnn_quant_profile_prepare_kernel_metadata(*model, *profile);
        size_t shard_count = static_cast<size_t>(
            (profile->num_decoder_layers + 1) / 2);
        if (argc == 5) {
            shard_count = std::stoul(argv[4]);
        }
        if (shard_count == 0) {
            throw std::runtime_error("QNN profile has no decoder shards");
        }
        llama_qnn_quant_profile_save_sharded_binary_file(
            *profile, argv[3], shard_count);
        const auto check = llama_qnn_quant_profile_load_file(argv[3]);
        if (check) {
            llama_qnn_quant_profile_prepare_kernel_metadata(*model, *check);
        }
        size_t streamed_sidecar_bytes = 0;
        if (check) {
            if (!check->binary_stream_prepare(shard_count)) {
                throw std::runtime_error(
                    "cannot release sharded sidecar buffers for verification");
            }
            for (size_t shard = 0; shard < shard_count; ++shard) {
                size_t loaded = 0;
                if (!check->binary_stream_fill(shard, &loaded)) {
                    throw std::runtime_error(
                        "cannot reload sidecar shard " + std::to_string(shard));
                }
                streamed_sidecar_bytes += loaded;
            }
            if (!check->binary_stream_finish()) {
                throw std::runtime_error(
                    "cannot finish sharded sidecar reload verification");
            }
            llama_qnn_quant_profile_prepare_kernel_metadata(*model, *check);
        }
        size_t tiled_linears = 0;
        size_t mapped_linear_buffers = 0;
        size_t mapped_static_buffers = 0;
        size_t mapped_lut_buffers = 0;
        if (check) {
            for (const auto & linear : check->linear_qparams) {
                tiled_linears +=
                    linear.qnn_weight_block_code_layout ==
                    LLAMA_QNN_BLOCK_CODES_GS32_TILE8_BLOCK_MAJOR;
                mapped_linear_buffers +=
                    linear.qnn_channel_scale_to_output_q31.is_mapped() &&
                    linear.qnn_weight_block_scale_codes.is_mapped() &&
                    linear.qnn_prepared_weight_sums.is_mapped();
            }
            for (const auto & tensor : check->u16_tensors) {
                mapped_static_buffers +=
                    tensor.static_data.empty() || tensor.static_data.is_mapped();
            }
            for (const auto & tensor : check->aux_quantized_tensors) {
                mapped_static_buffers +=
                    tensor.static_data.empty() || tensor.static_data.is_mapped();
            }
            for (const auto & operation : check->operations) {
                mapped_lut_buffers +=
                    (operation.unary_lut.empty() || operation.unary_lut.is_mapped()) &&
                    (operation.softmax_exp2_lut_q31.empty() ||
                     operation.softmax_exp2_lut_q31.is_mapped()) &&
                    (operation.input_to_output_q20.empty() ||
                     operation.input_to_output_q20.is_mapped());
            }
        }
        if (!check ||
            check->u16_tensor_count() != profile->u16_tensor_count() ||
            check->aux_quantized_tensor_count() != profile->aux_quantized_tensor_count() ||
            check->linear_qparams_count() != profile->linear_qparams_count() ||
            check->operation_count() != profile->operation_count() ||
            check->static_u16_bytes() != profile->static_u16_bytes() ||
            check->static_aux_bytes() != profile->static_aux_bytes() ||
            check->weight_layout != profile->weight_layout ||
            check->lm_head_type != profile->lm_head_type ||
            check->lm_head_layout != profile->lm_head_layout ||
            tiled_linears != check->linear_qparams_count() ||
            mapped_linear_buffers != check->linear_qparams_count() ||
            mapped_static_buffers !=
                check->u16_tensor_count() + check->aux_quantized_tensor_count() ||
            mapped_lut_buffers != check->operation_count()) {
            std::cerr << "binary profile verification failed\n";
            return 1;
        }
        std::cout
            << "qnn-profile-bin: status=pass"
            << " u16_tensors=" << check->u16_tensor_count()
            << " aux_tensors=" << check->aux_quantized_tensor_count()
            << " linear_pairs=" << check->linear_qparams_count()
            << " operations=" << check->operation_count()
            << " tiled_linears=" << tiled_linears
            << " mapped_linear_buffers=" << mapped_linear_buffers
            << " mapped_static_buffers=" << mapped_static_buffers
            << " mapped_operation_buffers=" << mapped_lut_buffers
            << " sidecar_shards=" << shard_count
            << " streamed_sidecar_bytes=" << streamed_sidecar_bytes
            << " weight_layout=" << check->weight_layout
            << " lm_head_type=" << check->lm_head_type
            << " lm_head_layout=" << check->lm_head_layout
            << " output=" << argv[3] << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "qnn-profile-bin: " << error.what() << '\n';
        return 1;
    }
}
