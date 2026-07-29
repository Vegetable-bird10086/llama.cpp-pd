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
