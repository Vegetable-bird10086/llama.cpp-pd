#include "arg.h"
#include "common.h"
#include "json.hpp"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include <ggml.h>

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

struct pd_args {
    std::string import_dir;
    bool import_ro = false;
    bool roundtrip_check = false;
    bool native_compare = false;
};

struct pd_handoff {
    json manifest;
    std::vector<llama_token> prompt_tokens;
    llama_token first_token = -1;
    std::vector<uint16_t> kv_fp16;
    int32_t prompt_len = 0;
    int32_t num_layers = 0;
    int32_t num_kv_heads = 0;
    int32_t head_dim = 0;
};

template <typename T>
void append_pod(std::vector<uint8_t> & out, const T & value) {
    const size_t old_size = out.size();
    out.resize(old_size + sizeof(T));
    std::memcpy(out.data() + old_size, &value, sizeof(T));
}

void append_bytes(std::vector<uint8_t> & out, const void * data, size_t size) {
    const size_t old_size = out.size();
    out.resize(old_size + size);
    std::memcpy(out.data() + old_size, data, size);
}

std::vector<uint8_t> read_binary_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("unable to read file: " + path);
    }
    return std::vector<uint8_t>(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

json read_json_file(const std::string & path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("unable to read json file: " + path);
    }
    return json::parse(input);
}

bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::optional<std::string> find_model_meta_value_by_suffix(
        const llama_model * model,
        const std::string & suffix) {
    const int32_t count = llama_model_meta_count(model);
    for (int32_t i = 0; i < count; ++i) {
        char key_buf[256];
        if (llama_model_meta_key_by_index(model, i, key_buf, sizeof(key_buf)) < 0) {
            continue;
        }
        const std::string key(key_buf);
        if (!ends_with(key, suffix)) {
            continue;
        }
        char val_buf[256];
        if (llama_model_meta_val_str_by_index(model, i, val_buf, sizeof(val_buf)) < 0) {
            continue;
        }
        return std::string(val_buf);
    }
    return std::nullopt;
}

std::optional<int32_t> parse_i32(const std::optional<std::string> & value) {
    if (!value.has_value() || value->empty()) {
        return std::nullopt;
    }
    try {
        return static_cast<int32_t>(std::stol(*value));
    } catch (...) {
        return std::nullopt;
    }
}

pd_args parse_pd_args(int argc, char ** argv, std::vector<char *> * forwarded) {
    pd_args out;
    forwarded->clear();
    forwarded->push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--pd-import") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--pd-import requires a directory");
            }
            out.import_dir = argv[++i];
            continue;
        }
        if (arg == "--pd-import-ro") {
            out.import_ro = true;
            continue;
        }
        if (arg == "--pd-roundtrip-check") {
            out.roundtrip_check = true;
            continue;
        }
        if (arg == "--pd-native-compare") {
            out.native_compare = true;
            continue;
        }
        forwarded->push_back(argv[i]);
    }
    if (out.import_dir.empty()) {
        throw std::runtime_error("--pd-import DIR is required");
    }
    return out;
}

std::optional<size_t> first_mismatch_offset(
        const std::vector<uint8_t> & lhs,
        const std::vector<uint8_t> & rhs) {
    const size_t shared = std::min(lhs.size(), rhs.size());
    for (size_t i = 0; i < shared; ++i) {
        if (lhs[i] != rhs[i]) {
            return i;
        }
    }
    if (lhs.size() != rhs.size()) {
        return shared;
    }
    return std::nullopt;
}

std::vector<llama_token> load_prompt_tokens(const std::string & path) {
    const std::vector<uint8_t> bytes = read_binary_file(path);
    if (bytes.size() % sizeof(uint64_t) != 0) {
        throw std::runtime_error("prompt_tokens.bin size is not aligned to uint64_t");
    }
    std::vector<llama_token> tokens;
    tokens.reserve(bytes.size() / sizeof(uint64_t));
    for (size_t offset = 0; offset < bytes.size(); offset += sizeof(uint64_t)) {
        uint64_t token_u64 = 0;
        std::memcpy(&token_u64, bytes.data() + offset, sizeof(token_u64));
        if (token_u64 > static_cast<uint64_t>(std::numeric_limits<llama_token>::max())) {
            throw std::runtime_error("prompt token out of llama_token range");
        }
        tokens.push_back(static_cast<llama_token>(token_u64));
    }
    return tokens;
}

llama_token load_first_token(const json & manifest, const std::string & import_dir) {
    if (manifest.contains("first_token_id")) {
        return static_cast<llama_token>(manifest.at("first_token_id").get<int64_t>());
    }
    const std::vector<uint8_t> bytes = read_binary_file(import_dir + "/first_token.bin");
    if (bytes.size() != sizeof(uint64_t)) {
        throw std::runtime_error("first_token.bin has unexpected size");
    }
    uint64_t token = 0;
    std::memcpy(&token, bytes.data(), sizeof(token));
    return static_cast<llama_token>(token);
}

pd_handoff load_pd_handoff(const std::string & import_dir) {
    pd_handoff out;
    out.manifest = read_json_file(import_dir + "/manifest.json");
    out.prompt_tokens = load_prompt_tokens(import_dir + "/prompt_tokens.bin");
    out.first_token = load_first_token(out.manifest, import_dir);

    const std::vector<uint8_t> kv_bytes = read_binary_file(import_dir + "/kv.bin");
    if (kv_bytes.size() % sizeof(uint16_t) != 0) {
        throw std::runtime_error("kv.bin size is not aligned to fp16");
    }
    out.kv_fp16.resize(kv_bytes.size() / sizeof(uint16_t));
    std::memcpy(out.kv_fp16.data(), kv_bytes.data(), kv_bytes.size());

    out.prompt_len = out.manifest.at("prompt_length").get<int32_t>();
    out.num_layers = out.manifest.at("num_layers").get<int32_t>();
    out.num_kv_heads = out.manifest.at("num_kv_heads").get<int32_t>();
    out.head_dim = out.manifest.at("head_dim").get<int32_t>();

    if (out.prompt_len != static_cast<int32_t>(out.prompt_tokens.size())) {
        throw std::runtime_error("manifest prompt_length does not match prompt_tokens.bin");
    }
    if (out.manifest.value("canonical_kv_dtype", "") != "fp16") {
        throw std::runtime_error("only fp16 canonical KV is supported");
    }

    const size_t expected_values =
        static_cast<size_t>(out.num_layers) *
        out.num_kv_heads *
        out.prompt_len *
        out.head_dim *
        2;
    if (out.kv_fp16.size() != expected_values) {
        std::ostringstream oss;
        oss << "kv.bin element count mismatch: got=" << out.kv_fp16.size()
            << " expected=" << expected_values;
        throw std::runtime_error(oss.str());
    }

    return out;
}

void validate_pd_handoff(
        const pd_handoff & handoff,
        const llama_model * model,
        const llama_context * ctx) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    if (handoff.num_layers != llama_model_n_layer(model)) {
        std::ostringstream oss;
        oss << "PD handoff layer count mismatch: handoff=" << handoff.num_layers
            << " gguf=" << llama_model_n_layer(model);
        throw std::runtime_error(oss.str());
    }
    if (handoff.first_token < 0 || handoff.first_token >= n_vocab) {
        throw std::runtime_error("first_token is out of vocabulary range");
    }
    for (llama_token token : handoff.prompt_tokens) {
        if (token < 0 || token >= n_vocab) {
            throw std::runtime_error("prompt token is out of vocabulary range");
        }
    }

    if (const auto meta_kv_heads =
            parse_i32(find_model_meta_value_by_suffix(model, "attention.head_count_kv"));
            meta_kv_heads.has_value() && *meta_kv_heads != handoff.num_kv_heads) {
        std::ostringstream oss;
        oss << "PD handoff num_kv_heads mismatch: handoff=" << handoff.num_kv_heads
            << " gguf=" << *meta_kv_heads;
        throw std::runtime_error(oss.str());
    }
    if (const auto meta_head_dim =
            parse_i32(find_model_meta_value_by_suffix(model, "attention.key_length"));
            meta_head_dim.has_value() && *meta_head_dim != handoff.head_dim) {
        std::ostringstream oss;
        oss << "PD handoff head_dim mismatch: handoff=" << handoff.head_dim
            << " gguf=" << *meta_head_dim;
        throw std::runtime_error(oss.str());
    }

    if (handoff.prompt_len > llama_n_ctx(ctx)) {
        std::ostringstream oss;
        oss << "PD prompt length exceeds decode context: prompt_len=" << handoff.prompt_len
            << " n_ctx=" << llama_n_ctx(ctx);
        throw std::runtime_error(oss.str());
    }
}

void validate_imported_kv_state(const pd_handoff & handoff, llama_context * ctx) {
    const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    if (pos_max != handoff.prompt_len - 1) {
        std::ostringstream oss;
        oss << "KV import max position mismatch: imported=" << pos_max
            << " expected=" << (handoff.prompt_len - 1);
        throw std::runtime_error(oss.str());
    }
}

std::vector<uint8_t> build_seq_state_blob(const pd_handoff & handoff, bool v_trans) {
    static constexpr uint32_t k_state_seq_magic = 0xaf143cd8U;
    static constexpr llama_seq_id k_seq_id = 0;
    static constexpr uint32_t k_n_stream = 1;

    const uint32_t cell_count = static_cast<uint32_t>(handoff.prompt_len);
    const uint32_t n_layer = static_cast<uint32_t>(handoff.num_layers);
    const uint32_t n_embd_k_gqa =
        static_cast<uint32_t>(handoff.num_kv_heads * handoff.head_dim);
    const uint32_t n_embd_v_gqa = n_embd_k_gqa;
    const uint64_t k_row_size =
        static_cast<uint64_t>(n_embd_k_gqa) * sizeof(uint16_t);
    const uint64_t v_row_size = k_row_size;
    const size_t per_kind_values =
        static_cast<size_t>(handoff.num_layers) *
        handoff.num_kv_heads *
        handoff.prompt_len *
        handoff.head_dim;
    const uint16_t * k_base = handoff.kv_fp16.data();
    const uint16_t * v_base = handoff.kv_fp16.data() + per_kind_values;

    std::vector<uint8_t> blob;
    blob.reserve(
        sizeof(uint32_t) +
        sizeof(llama_seq_id) +
        sizeof(uint32_t) +
        sizeof(uint32_t) +
        cell_count * (sizeof(llama_pos) + sizeof(uint32_t) + sizeof(llama_seq_id)) +
        sizeof(uint32_t) * 2 +
        n_layer * (sizeof(int32_t) + sizeof(uint64_t) + cell_count * k_row_size) +
        n_layer * (sizeof(int32_t) + sizeof(uint32_t) * 2 + cell_count * v_row_size));

    append_pod(blob, k_state_seq_magic);
    append_pod(blob, k_seq_id);
    append_pod(blob, k_n_stream);
    append_pod(blob, cell_count);
    for (uint32_t pos = 0; pos < cell_count; ++pos) {
        const llama_pos llama_position = static_cast<llama_pos>(pos);
        const uint32_t n_seq_id = 1;
        append_pod(blob, llama_position);
        append_pod(blob, n_seq_id);
        append_pod(blob, k_seq_id);
    }

    const uint32_t v_trans_u32 = v_trans ? 1u : 0u;
    append_pod(blob, v_trans_u32);
    append_pod(blob, n_layer);

    for (int32_t layer = 0; layer < handoff.num_layers; ++layer) {
        const int32_t type = GGML_TYPE_F16;
        append_pod(blob, type);
        append_pod(blob, k_row_size);
        for (int32_t pos = 0; pos < handoff.prompt_len; ++pos) {
            for (int32_t head = 0; head < handoff.num_kv_heads; ++head) {
                const size_t src = (((static_cast<size_t>(layer) * handoff.num_kv_heads + head) *
                                    handoff.prompt_len + pos) *
                                    handoff.head_dim);
                append_bytes(
                    blob,
                    k_base + src,
                    static_cast<size_t>(handoff.head_dim) * sizeof(uint16_t));
            }
        }
    }

    for (int32_t layer = 0; layer < handoff.num_layers; ++layer) {
        const int32_t type = GGML_TYPE_F16;
        append_pod(blob, type);
        if (!v_trans) {
            append_pod(blob, v_row_size);
            for (int32_t pos = 0; pos < handoff.prompt_len; ++pos) {
                for (int32_t head = 0; head < handoff.num_kv_heads; ++head) {
                    const size_t src = (((static_cast<size_t>(layer) * handoff.num_kv_heads + head) *
                                        handoff.prompt_len + pos) *
                                        handoff.head_dim);
                    append_bytes(
                        blob,
                        v_base + src,
                        static_cast<size_t>(handoff.head_dim) * sizeof(uint16_t));
                }
            }
        } else {
            const uint32_t v_size_el = sizeof(uint16_t);
            append_pod(blob, v_size_el);
            append_pod(blob, n_embd_v_gqa);
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                const int32_t head = static_cast<int32_t>(j / handoff.head_dim);
                const int32_t dim = static_cast<int32_t>(j % handoff.head_dim);
                for (int32_t pos = 0; pos < handoff.prompt_len; ++pos) {
                    const size_t src =
                        (((static_cast<size_t>(layer) * handoff.num_kv_heads + head) *
                          handoff.prompt_len + pos) *
                         handoff.head_dim) + dim;
                    append_bytes(blob, v_base + src, sizeof(uint16_t));
                }
            }
        }
    }

    return blob;
}

void print_usage(int argc, char ** argv) {
    (void) argc;
    LOG("\nexample usage:\n");
    LOG("  %s --pd-import handoff_dir -m model.gguf -n 128 -c 2048 -t 4 -ngl 0\n", argv[0]);
    LOG("\n");
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    pd_args pd;
    std::vector<char *> forwarded;
    try {
        pd = parse_pd_args(argc, argv, &forwarded);
    } catch (const std::exception & err) {
        LOG_ERR("%s\n", err.what());
        return 1;
    }

    common_params params;
    common_init();

    const int forwarded_argc = static_cast<int>(forwarded.size());
    if (!common_params_parse(
            forwarded_argc,
            forwarded.data(),
            params,
            LLAMA_EXAMPLE_COMPLETION,
            print_usage)) {
        return 1;
    }

    if (!params.prompt.empty()) {
        LOG_ERR("pd-cli does not accept a prompt; prompt tokens come from --pd-import\n");
        return 1;
    }
    if (pd.native_compare) {
        LOG_ERR("--pd-native-compare is not implemented in this llama.cpp port yet\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_context * ctx = llama_init->context();
    llama_model * model = llama_init->model();
    common_sampler * smpl = llama_init->sampler(0);
    if (model == nullptr || ctx == nullptr || smpl == nullptr) {
        LOG_ERR("failed to initialize model/context/sampler\n");
        return 1;
    }

    pd_handoff handoff;
    try {
        handoff = load_pd_handoff(pd.import_dir);
        validate_pd_handoff(handoff, model, ctx);
    } catch (const std::exception & err) {
        LOG_ERR("failed to load PD handoff: %s\n", err.what());
        return 1;
    }

    for (llama_token token : handoff.prompt_tokens) {
        common_sampler_accept(smpl, token, false);
    }

    // The KV cache layout is fixed when the context memory module is created.
    // At that point llama_context has only applied the coarse flash-attn gate:
    // AUTO and ENABLED both build a non-transposed V cache, while DISABLED
    // builds the legacy transposed layout. Later auto_fa resolution only
    // affects graph selection, not the already-allocated KV tensor layout.
    const bool v_trans = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED;
    const std::vector<uint8_t> seq_blob = build_seq_state_blob(handoff, v_trans);
    llama_memory_clear(llama_get_memory(ctx), true);
    const size_t nset =
        llama_state_seq_set_data(ctx, seq_blob.data(), seq_blob.size(), 0);
    if (nset != seq_blob.size()) {
        LOG_ERR(
            "failed to import PD KV state: written=%zu expected=%zu\n",
            nset,
            seq_blob.size());
        return 1;
    }

    try {
        validate_imported_kv_state(handoff, ctx);
    } catch (const std::exception & err) {
        LOG_ERR("PD handoff imported but KV validation failed: %s\n", err.what());
        return 1;
    }

    if (pd.roundtrip_check) {
        const size_t roundtrip_size = llama_state_seq_get_size(ctx, 0);
        std::vector<uint8_t> roundtrip_blob(roundtrip_size);
        const size_t nread =
            llama_state_seq_get_data(ctx, roundtrip_blob.data(), roundtrip_blob.size(), 0);
        if (nread != roundtrip_blob.size()) {
            LOG_ERR(
                "PD roundtrip check failed to read sequence state: got=%zu expected=%zu\n",
                nread,
                roundtrip_blob.size());
            return 1;
        }
        if (seq_blob.size() != roundtrip_blob.size()) {
            LOG_ERR(
                "PD roundtrip size mismatch: imported=%zu roundtrip=%zu\n",
                seq_blob.size(),
                roundtrip_blob.size());
            return 1;
        }
        if (const auto mismatch = first_mismatch_offset(seq_blob, roundtrip_blob); mismatch.has_value()) {
            LOG_ERR(
                "PD roundtrip mismatch at byte offset %zu: imported=0x%02x roundtrip=0x%02x\n",
                *mismatch,
                seq_blob[*mismatch],
                roundtrip_blob[*mismatch]);
            return 1;
        }
        LOG_INF("PD roundtrip check passed\n");
    }

    LOG_INF(
        "PD handoff imported: prompt_len=%d first_token=%d layers=%d kv_heads=%d head_dim=%d\n",
        handoff.prompt_len,
        handoff.first_token,
        handoff.num_layers,
        handoff.num_kv_heads,
        handoff.head_dim);

    if (pd.import_ro) {
        LOG_INF("PD import validation completed in read-only mode\n");
        return 0;
    }

    int n_remain = params.n_predict;
    const bool infinite = n_remain < 0;
    llama_token cur = handoff.first_token;

    if (!infinite) {
        if (n_remain == 0) {
            return 0;
        }
        --n_remain;
    }

    common_sampler_accept(smpl, cur, true);
    LOG("%s", common_token_to_piece(ctx, cur, params.special).c_str());

    if (llama_decode(ctx, llama_batch_get_one(&cur, 1)) != 0) {
        LOG_ERR("llama_decode failed after importing PD state\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    while (infinite || n_remain > 0) {
        const llama_token next = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, next, true);
        LOG("%s", common_token_to_piece(ctx, next, params.special).c_str());

        if (llama_vocab_is_eog(vocab, next)) {
            break;
        }

        cur = next;
        if (!infinite) {
            --n_remain;
            if (n_remain == 0) {
                break;
            }
        }

        if (llama_decode(ctx, llama_batch_get_one(&cur, 1)) != 0) {
            LOG_ERR("llama_decode failed while continuing decode\n");
            return 1;
        }
    }

    LOG("\n");
    common_perf_print(ctx, smpl);
    return 0;
}
