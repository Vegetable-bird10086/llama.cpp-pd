#pragma once

#include <cstddef>
#include <cstdint>

struct llama_pd_inprocess_request {
    const void * model_data;
    size_t model_size;
    const void * handoff_data;
    size_t handoff_size;
    int32_t prompt_length;
    int32_t num_layers;
    int32_t num_kv_heads;
    int32_t head_dim;
    int32_t first_token;
    bool first_token_is_prompt_tail;
    struct llama_pd_inprocess_result * result;
    // Optional split, zero-copy FP16 handoff. When these fields are set,
    // handoff_data/handoff_size are ignored. This keeps the original packed
    // UINT8 QNN ABI working while allowing MTK's canonical FP16 KV buffer to
    // be imported without first concatenating prompt tokens and KV data.
    const uint64_t * prompt_tokens = nullptr;
    const uint16_t * kv_fp16 = nullptr;
    size_t kv_fp16_values = 0;
    // Optional in-process lifecycle probe. The joint PD runner uses this to
    // synchronously sample model residency and process faults at exact Decode
    // call boundaries without inferring them from a periodic sampler.
    void (*decode_event_callback)(
        void * opaque,
        const char * event,
        int32_t decode_call_index,
        int32_t token) = nullptr;
    void * decode_event_opaque = nullptr;
    int32_t decode_event_call_limit = 0;
};

struct llama_pd_inprocess_result {
    // Process-lifetime work which a persistent multi-turn runner retains.
    double initialization_ms;
    // Per-turn work after Prefill completes and before autoregressive decode.
    double boundary_ms;
    // Autoregressive generation, including sampling and token output.
    double generation_ms;
    int32_t generated_tokens;
};

struct llama_pd_inprocess_runtime;

// Loads model-owned state without creating the Decode context, KV cache or
// scheduler compute buffers.
llama_pd_inprocess_runtime * llama_pd_inprocess_runtime_create_model_only(
    int argc,
    char ** argv,
    const void * model_data,
    size_t model_size,
    double * initialization_ms);

// Creates the fixed Decode context and reserves its compute buffers.
bool llama_pd_inprocess_runtime_prepare_context(
    llama_pd_inprocess_runtime * runtime,
    double * preparation_ms);

// Executes the one-token initialization pass after the context is ready.
bool llama_pd_inprocess_runtime_warmup(
    llama_pd_inprocess_runtime * runtime,
    double * warmup_ms);

// Returns the zero-copy binary QNN profile backing owned by the resident
// Decode model. The pointer remains valid until runtime destruction.
bool llama_pd_inprocess_runtime_profile_backing(
    llama_pd_inprocess_runtime * runtime,
    const void ** data,
    size_t * size,
    bool * anonymous);

bool llama_pd_inprocess_runtime_profile_stream_prepare(
    llama_pd_inprocess_runtime * runtime,
    size_t shard_count);
bool llama_pd_inprocess_runtime_profile_stream_fill(
    llama_pd_inprocess_runtime * runtime,
    size_t shard_index,
    size_t * bytes_loaded);
bool llama_pd_inprocess_runtime_profile_stream_finish(
    llama_pd_inprocess_runtime * runtime);

// Creates the model/context/KV/scheduler state once. The returned runtime can
// serve multiple handoffs; worker threads are attached lazily by the first run.
llama_pd_inprocess_runtime * llama_pd_inprocess_runtime_create(
    int argc,
    char ** argv,
    const void * model_data,
    size_t model_size,
    double * initialization_ms);

int llama_pd_inprocess_runtime_run(
    llama_pd_inprocess_runtime * runtime,
    const llama_pd_inprocess_request * request);

void llama_pd_inprocess_runtime_destroy(
    llama_pd_inprocess_runtime * runtime);

// Runs the normal llama-pd-cli Decode implementation in the calling process.
// Both pointed-to buffers remain owned by the caller and must stay alive until
// this function returns.
int llama_pd_cli_run_inprocess(
    int argc,
    char ** argv,
    const llama_pd_inprocess_request * request);
