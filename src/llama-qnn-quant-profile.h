#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

template <typename T>
class llama_qnn_buffer {
public:
    using value_type = T;
    using const_iterator = const T *;
    using iterator = T *;

    llama_qnn_buffer() = default;
    llama_qnn_buffer(std::vector<T> value) : storage(std::move(value)) {}

    llama_qnn_buffer & operator=(std::vector<T> value) {
        storage = std::move(value);
        mapped = nullptr;
        mapped_size = 0;
        return *this;
    }

    bool empty() const { return size() == 0; }
    size_t size() const { return mapped != nullptr ? mapped_size : storage.size(); }
    const T * data() const { return mapped != nullptr ? mapped : storage.data(); }
    T * data() { materialize(); return storage.data(); }
    const T & operator[](size_t index) const { return data()[index]; }
    T & operator[](size_t index) { materialize(); return storage[index]; }
    const T & front() const { return data()[0]; }
    T & front() { materialize(); return storage.front(); }
    const_iterator begin() const { return data(); }
    const_iterator end() const { return data() + size(); }
    iterator begin() { materialize(); return storage.data(); }
    iterator end() { materialize(); return storage.data() + storage.size(); }

    void reserve(size_t size) { materialize(); storage.reserve(size); }
    void clear() { storage.clear(); mapped = nullptr; mapped_size = 0; }
    void resize(size_t size) { materialize(); storage.resize(size); }
    void resize(size_t size, const T & value) { materialize(); storage.resize(size, value); }
    void push_back(const T & value) { materialize(); storage.push_back(value); }
    void push_back(T && value) { materialize(); storage.push_back(std::move(value)); }
    void assign(size_t size, const T & value) { materialize(); storage.assign(size, value); }

    void set_mapped(const T * data, size_t size) {
        storage.clear();
        storage.shrink_to_fit();
        mapped = data;
        mapped_size = size;
    }

    bool is_mapped() const { return mapped != nullptr; }

private:
    void materialize() {
        if (mapped == nullptr) {
            return;
        }
        storage.assign(mapped, mapped + mapped_size);
        mapped = nullptr;
        mapped_size = 0;
    }

    std::vector<T> storage;
    const T * mapped = nullptr;
    size_t mapped_size = 0;
};

// QNN represents affine quantization as real=(code + offset)*scale.  Keep the
// original f32 bit pattern so loading the profile cannot introduce a decimal
// JSON round-trip.
struct llama_qnn_affine_qparams {
    float    scale = 0.0f;
    uint32_t scale_bits = 0;
    int32_t  offset = 0;
    int32_t  zero_point = 0;
};

enum llama_qnn_quantization_encoding {
    LLAMA_QNN_QUANTIZATION_UNDEFINED,
    LLAMA_QNN_QUANTIZATION_SCALE_OFFSET,
    LLAMA_QNN_QUANTIZATION_AXIS_SCALE_OFFSET,
    LLAMA_QNN_QUANTIZATION_BLOCKWISE_EXPANSION,
};

enum llama_qnn_block_code_layout {
    LLAMA_QNN_BLOCK_CODES_ROW_MAJOR = 0,
    LLAMA_QNN_BLOCK_CODES_PREPARED_ROW_MAJOR = 1,
    LLAMA_QNN_BLOCK_CODES_GS32_TILE8_BLOCK_MAJOR = 2,
};

struct llama_qnn_tensor_qparams {
    llama_qnn_quantization_encoding encoding = LLAMA_QNN_QUANTIZATION_UNDEFINED;
    int32_t axis = -1;
    llama_qnn_buffer<llama_qnn_affine_qparams> scale_offsets;
    int32_t block_scale_bitwidth = 0;
    int32_t block_scale_element_bytes = 0;
    int32_t num_blocks_per_axis = 0;
    llama_qnn_buffer<uint8_t> block_scale_codes;
};

struct llama_qnn_u16_tensor_use {
    std::string operation_name;
    std::string role;
    int32_t position = -1;
};

struct llama_qnn_decoder_binding {
    std::vector<int32_t> layer_ids;
    std::vector<std::string> module_paths;
    std::string projection;
};

struct llama_qnn_u16_tensor {
    int32_t shard_index = -1;
    std::string scope;
    std::string name;
    std::string data_type;
    std::vector<int64_t> dimensions;
    llama_qnn_tensor_qparams qparams;
    llama_qnn_buffer<uint16_t> static_data;
    std::string static_data_sha256;
    std::vector<llama_qnn_u16_tensor_use> operation_uses;
    std::vector<llama_qnn_decoder_binding> decoder_bindings;
};

// Non-U16 fixed-point tensors used by the delegated graph. Static payloads
// remain byte-exact so signed S16 matrices and U8 KV tensors retain their QNN
// representation without a host-side floating-point conversion.
struct llama_qnn_aux_quantized_tensor {
    int32_t shard_index = -1;
    std::string scope;
    std::string name;
    std::string data_type;
    uint32_t element_bytes = 0;
    std::vector<int64_t> dimensions;
    llama_qnn_tensor_qparams qparams;
    llama_qnn_buffer<uint8_t> static_data;
    std::string static_data_sha256;
    std::vector<llama_qnn_u16_tensor_use> operation_uses;
    std::vector<llama_qnn_decoder_binding> decoder_bindings;
};

struct llama_qnn_linear_qparams {
    int32_t layer_id = -1;
    int32_t shard_index = -1;
    std::string scope;
    std::string projection;
    llama_qnn_affine_qparams input;
    llama_qnn_affine_qparams output;
    int64_t activation_to_output_q20 = 0;
    int32_t qnn_weight_block_size = 0;
    int32_t qnn_weight_blocks_per_row = 0;
    llama_qnn_buffer<int64_t> qnn_channel_scale_to_output_q31;
    llama_qnn_buffer<uint8_t> qnn_weight_block_scale_codes;
    llama_qnn_buffer<int64_t> qnn_prepared_weight_sums;
    bool qnn_weight_block_codes_prepared = false;
    bool weights_gs32_source = false;
    int32_t qnn_weight_block_code_layout = 0;
    int32_t output_bias_q7 = 0;
    std::vector<std::string> operation_names;
};

struct llama_qnn_operation_u16_operand {
    std::string name;
    std::string role;
    int32_t position = -1;
    size_t tensor_index = 0;
};

struct llama_qnn_operation_aux_operand {
    std::string name;
    std::string role;
    int32_t position = -1;
    size_t tensor_index = 0;
};

struct llama_qnn_operation {
    int32_t shard_index = -1;
    std::string scope;
    std::string name;
    std::string type_name;
    std::string fx_node_name;
    llama_qnn_decoder_binding decoder_binding;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<llama_qnn_operation_u16_operand> u16_operands;
    std::vector<llama_qnn_operation_aux_operand> aux_operands;
    llama_qnn_buffer<int64_t> input_to_output_q20;
    int64_t product_to_output_q20 = 0;
    int64_t product_to_output_q31 = 0;
    int64_t product_requant_nudge_q31 = INT64_C(1) << 30;
    int64_t matmul_product_to_output_q31 = 0;
    int64_t unary_input_to_output_q20 = 0;
    int64_t unary_input_to_output_q31 = 0;
    llama_qnn_buffer<uint16_t> unary_lut;
    int64_t softmax_scale_over_ln2_q24 = 0;
    int64_t softmax_unit_code = 0;
    llama_qnn_buffer<uint32_t> softmax_exp2_lut_q31;
    double rms_epsilon = 0.0;
    uint64_t rms_epsilon_in_codes_q16 = 0;
};

// Immutable profile loaded into the normal llama.cpp model. QNN tensor names
// are local to a delegated context and can repeat in every shard. The decoder
// path therefore queries by shard/scope plus the QNN ABI
// name rather than guessing scales from a checkpoint or parameter ordinal.
struct llama_qnn_quant_profile {
    std::string source_path;
    std::string quantization_formula;
    int32_t num_decoder_layers = 0;
    int32_t source_weight_bits = 0;
    int32_t source_group_size = 0;
    std::string weight_layout;
    std::string lm_head_tensor_name;
    std::string lm_head_type;
    std::string lm_head_layout;
    int64_t lm_head_input_elements = 0;
    int64_t lm_head_output_rows = 0;
    std::vector<llama_qnn_u16_tensor> u16_tensors;
    std::vector<llama_qnn_aux_quantized_tensor> aux_quantized_tensors;
    std::vector<llama_qnn_linear_qparams> linear_qparams;
    std::vector<llama_qnn_operation> operations;

    const llama_qnn_u16_tensor * find_u16_tensor(const std::string & name) const;
    const llama_qnn_u16_tensor * find_u16_tensor(int32_t shard_index, const std::string & name) const;
    const llama_qnn_u16_tensor * find_u16_tensor(const std::string & scope, const std::string & name) const;
    const llama_qnn_aux_quantized_tensor * find_aux_tensor(const std::string & name) const;
    const llama_qnn_aux_quantized_tensor * find_aux_tensor(int32_t shard_index, const std::string & name) const;
    const llama_qnn_aux_quantized_tensor * find_aux_tensor(const std::string & scope, const std::string & name) const;
    const llama_qnn_linear_qparams * find_linear_qparams(int32_t layer_id, const std::string & projection) const;
    const llama_qnn_operation * find_operation(int32_t shard_index, const std::string & name) const;
    const llama_qnn_operation * find_producer(int32_t shard_index, const std::string & tensor_name) const;
    const llama_qnn_operation * find_operation_by_fx(int32_t layer_id, const std::string & fx_node_name) const;
    const llama_qnn_u16_tensor * find_u16_operand(const llama_qnn_operation_u16_operand & operand) const;
    const llama_qnn_u16_tensor * find_u16_operand(
        const llama_qnn_operation & operation,
        const std::string & role,
        int32_t position) const;
    const llama_qnn_aux_quantized_tensor * find_aux_operand(const llama_qnn_operation_aux_operand & operand) const;
    const llama_qnn_aux_quantized_tensor * find_aux_operand(
        const llama_qnn_operation & operation,
        const std::string & role,
        int32_t position) const;
    size_t u16_tensor_count() const;
    size_t aux_quantized_tensor_count() const;
    size_t linear_qparams_count() const;
    size_t operation_count() const;
    size_t static_u16_tensor_count() const;
    size_t static_u16_bytes() const;
    size_t static_aux_tensor_count() const;
    size_t static_aux_bytes() const;

private:
    std::shared_ptr<void> binary_mapping;
    std::unordered_map<std::string, size_t> u16_tensor_index;
    std::unordered_map<std::string, size_t> u16_tensor_shard_index;
    std::unordered_map<std::string, size_t> u16_tensor_scope_index;
    std::unordered_map<std::string, size_t> aux_tensor_index;
    std::unordered_map<std::string, size_t> aux_tensor_shard_index;
    std::unordered_map<std::string, size_t> aux_tensor_scope_index;
    std::unordered_map<std::string, size_t> linear_qparams_index;
    std::unordered_map<std::string, size_t> operation_shard_index;
    std::unordered_map<std::string, size_t> operation_output_shard_index;
    std::unordered_map<std::string, size_t> operation_fx_index;

    friend std::shared_ptr<llama_qnn_quant_profile>
    llama_qnn_quant_profile_load_file(const std::string & path);
    friend std::shared_ptr<llama_qnn_quant_profile>
    llama_qnn_quant_profile_load_binary_file(const std::string & path);
};

// Returns nullptr when LLAMA_QNN_U16_QPARAMS_MANIFEST is unset.  If it is set,
// parsing and validation failures are fatal to model loading so decode never
// silently falls back to unrelated qparams.
std::shared_ptr<llama_qnn_quant_profile> llama_qnn_quant_profile_load_from_environment();
std::shared_ptr<llama_qnn_quant_profile> llama_qnn_quant_profile_load_file(const std::string & path);

// The binary profile is the already validated and derived runtime object.  It
// is loaded from a read-only mmap without constructing a JSON DOM or decoding
// Base64 payloads.
bool llama_qnn_quant_profile_is_binary_file(const std::string & path);
std::shared_ptr<llama_qnn_quant_profile> llama_qnn_quant_profile_load_binary_file(const std::string & path);
void llama_qnn_quant_profile_save_binary_file(
    const llama_qnn_quant_profile & profile,
    const std::string & path);

struct llama_model;
void llama_qnn_quant_profile_prepare_kernel_metadata(
    llama_model & model,
    llama_qnn_quant_profile & profile);
