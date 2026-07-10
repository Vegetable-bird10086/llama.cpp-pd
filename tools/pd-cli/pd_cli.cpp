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
#include <memory>
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
    bool native_first_token = false;
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
        if (arg == "--pd-native-first-token") {
            out.native_first_token = true;
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


struct token_logit {
    llama_token token = -1;
    float logit = 0.0f;
};

struct native_compare_result {
    std::vector<float> prompt_logits;
    std::vector<float> resume_logits;
    std::vector<uint8_t> prompt_seq_blob;
};

std::vector<token_logit> top_k_logits(const float * logits, int32_t n_vocab, int32_t k) {
    std::vector<token_logit> out;
    out.reserve(n_vocab);
    for (llama_token token = 0; token < n_vocab; ++token) {
        out.push_back({token, logits[token]});
    }
    if (k < n_vocab) {
        std::partial_sort(
            out.begin(),
            out.begin() + k,
            out.end(),
            [](const token_logit & a, const token_logit & b) {
                return a.logit > b.logit;
            });
        out.resize(k);
    } else {
        std::sort(
            out.begin(),
            out.end(),
            [](const token_logit & a, const token_logit & b) {
                return a.logit > b.logit;
            });
    }
    return out;
}

float fp16_bits_to_float(uint16_t bits) {
    ggml_fp16_t fp16 = bits;
    return ggml_fp16_to_fp32(fp16);
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


std::string describe_top_k(
        llama_context * ctx,
        const std::vector<token_logit> & top,
        bool special) {
    std::ostringstream oss;
    for (size_t i = 0; i < top.size(); ++i) {
        if (i != 0) {
            oss << " ";
        }
        oss << "[" << i
            << ":tok=" << top[i].token
            << ",logit=" << top[i].logit
            << ",piece=" << common_token_to_piece(ctx, top[i].token, special)
            << "]";
    }
    return oss.str();
}

native_compare_result run_native_compare(
        const pd_handoff & handoff,
        llama_model * model,
        const common_params & params) {
    native_compare_result out;
    llama_context_params ctx_params = common_context_params_to_llama(params);
    std::unique_ptr<llama_context, decltype(&llama_free)> native_ctx(
        llama_init_from_model(model, ctx_params),
        llama_free);
    if (!native_ctx) {
        throw std::runtime_error("failed to create native comparison context");
    }

    const int32_t batch_cap = std::max<int32_t>(1, llama_n_batch(native_ctx.get()));
    for (int32_t start = 0; start < handoff.prompt_len; start += batch_cap) {
        const int32_t chunk = std::min(batch_cap, handoff.prompt_len - start);
        if (llama_decode(
                native_ctx.get(),
                llama_batch_get_one(
                    const_cast<llama_token *>(handoff.prompt_tokens.data() + start),
                    chunk)) != 0) {
            throw std::runtime_error("native comparison prompt decode failed");
        }
    }

    {
        const float * logits = llama_get_logits_ith(native_ctx.get(), -1);
        if (logits == nullptr) {
            throw std::runtime_error("native comparison prompt logits are unavailable");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        out.prompt_logits.assign(logits, logits + n_vocab);
    }

    {
        const size_t seq_size = llama_state_seq_get_size(native_ctx.get(), 0);
        out.prompt_seq_blob.resize(seq_size);
        const size_t nread =
            llama_state_seq_get_data(native_ctx.get(), out.prompt_seq_blob.data(), seq_size, 0);
        if (nread != seq_size) {
            throw std::runtime_error("native comparison prompt seq save failed");
        }
    }

    llama_token first = handoff.first_token;
    if (llama_decode(native_ctx.get(), llama_batch_get_one(&first, 1)) != 0) {
        throw std::runtime_error("native comparison first-token decode failed");
    }

    const float * logits = llama_get_logits_ith(native_ctx.get(), -1);
    if (logits == nullptr) {
        throw std::runtime_error("native comparison logits are unavailable");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    out.resume_logits.assign(logits, logits + n_vocab);
    return out;
}

void log_prompt_seq_blob_comparison(
        const std::vector<uint8_t> & imported_blob,
        const std::vector<uint8_t> & native_blob,
        const pd_handoff & handoff,
        bool v_trans) {
    if (imported_blob.size() != native_blob.size()) {
        LOG_INF(
            "PD native compare prompt-KV blob size mismatch: imported=%zu native=%zu\n",
            imported_blob.size(),
            native_blob.size());
        return;
    }

    if (const auto mismatch = first_mismatch_offset(imported_blob, native_blob); !mismatch.has_value()) {
        LOG_INF("PD native compare prompt-KV blob matches native prefill exactly\n");
        return;
    }

    const auto mismatch = first_mismatch_offset(imported_blob, native_blob);
    LOG_INF(
        "PD native compare prompt-KV first mismatch at byte offset %zu: imported=0x%02x native=0x%02x\n",
        *mismatch,
        imported_blob[*mismatch],
        native_blob[*mismatch]);

    const size_t meta_prefix =
        sizeof(uint32_t) +
        sizeof(llama_seq_id) +
        sizeof(uint32_t) +
        sizeof(uint32_t) +
        static_cast<size_t>(handoff.prompt_len) *
            (sizeof(llama_pos) + sizeof(uint32_t) + sizeof(llama_seq_id)) +
        sizeof(uint32_t) * 2;
    size_t offset = meta_prefix;
    const uint32_t n_embd_gqa = static_cast<uint32_t>(handoff.num_kv_heads * handoff.head_dim);
    const size_t row_size = static_cast<size_t>(n_embd_gqa) * sizeof(uint16_t);
    const size_t rows = static_cast<size_t>(handoff.prompt_len);
    const size_t sample_values = std::min<size_t>(8, n_embd_gqa);

    auto log_row_sample = [&](const char * kind, int32_t layer, size_t block_offset) {
        if (rows == 0 || sample_values == 0) {
            return;
        }
        std::ostringstream oss;
        oss << "PD native compare " << kind << " layer=" << layer << " row0 imported/native:";
        for (size_t elem = 0; elem < sample_values; ++elem) {
            uint16_t imported_bits = 0;
            uint16_t native_bits = 0;
            std::memcpy(&imported_bits, imported_blob.data() + block_offset + elem*sizeof(uint16_t), sizeof(uint16_t));
            std::memcpy(&native_bits, native_blob.data() + block_offset + elem*sizeof(uint16_t), sizeof(uint16_t));
            oss << " [" << elem
                << ":i=" << fp16_bits_to_float(imported_bits)
                << ",n=" << fp16_bits_to_float(native_bits)
                << "]";
        }
        LOG_INF("%s\n", oss.str().c_str());
    };

    auto log_maxdiff_sample = [&](const char * kind, int32_t layer, size_t block_offset, size_t elem_index) {
        const size_t start_elem = elem_index > 3 ? elem_index - 3 : 0;
        const size_t end_elem = std::min(elem_index + 4, rows * static_cast<size_t>(n_embd_gqa));
        std::ostringstream oss;
        oss << "PD native compare " << kind << " layer=" << layer
            << " around maxdiff elem=" << elem_index << ":";
        for (size_t elem = start_elem; elem < end_elem; ++elem) {
            uint16_t imported_bits = 0;
            uint16_t native_bits = 0;
            std::memcpy(&imported_bits, imported_blob.data() + block_offset + elem*sizeof(uint16_t), sizeof(uint16_t));
            std::memcpy(&native_bits, native_blob.data() + block_offset + elem*sizeof(uint16_t), sizeof(uint16_t));
            oss << " [" << elem
                << ":i=" << fp16_bits_to_float(imported_bits)
                << ",n=" << fp16_bits_to_float(native_bits)
                << "]";
        }
        LOG_INF("%s\n", oss.str().c_str());
    };

    auto compare_layer_block = [&](const char * kind, int32_t layer, size_t block_offset, size_t data_size) {
        float max_abs_diff = 0.0f;
        size_t max_idx = 0;
        for (size_t i = 0; i < data_size; i += sizeof(uint16_t)) {
            uint16_t imported_bits = 0;
            uint16_t native_bits = 0;
            std::memcpy(&imported_bits, imported_blob.data() + block_offset + i, sizeof(uint16_t));
            std::memcpy(&native_bits, native_blob.data() + block_offset + i, sizeof(uint16_t));
            const float diff = std::fabs(
                fp16_bits_to_float(imported_bits) - fp16_bits_to_float(native_bits));
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_idx = i / sizeof(uint16_t);
            }
        }
        LOG_INF(
            "PD native compare %s layer=%d max_abs_diff=%f elem_index=%zu\n",
            kind,
            layer,
            max_abs_diff,
            max_idx);
        if (layer == 0) {
            log_row_sample(kind, layer, block_offset);
        }
        if (max_abs_diff > 8.0f || layer == 0) {
            log_maxdiff_sample(kind, layer, block_offset, max_idx);
        }
    };

    for (int32_t layer = 0; layer < handoff.num_layers; ++layer) {
        offset += sizeof(int32_t) + sizeof(uint64_t);
        compare_layer_block("K", layer, offset, rows * row_size);
        offset += rows * row_size;
    }

    for (int32_t layer = 0; layer < handoff.num_layers; ++layer) {
        if (!v_trans) {
            offset += sizeof(int32_t) + sizeof(uint64_t);
            compare_layer_block("V", layer, offset, rows * row_size);
            offset += rows * row_size;
        } else {
            offset += sizeof(int32_t) + sizeof(uint32_t) + sizeof(uint32_t);
            compare_layer_block("V", layer, offset, rows * row_size);
            offset += rows * row_size;
        }
    }
}

void log_native_comparison(
        llama_context * imported_ctx,
        llama_model * model,
        const common_params & params,
        const pd_handoff & handoff,
        const std::vector<uint8_t> & imported_prompt_seq_blob,
        bool v_trans) {
    const float * imported_logits = llama_get_logits_ith(imported_ctx, -1);
    if (imported_logits == nullptr) {
        throw std::runtime_error("imported logits are unavailable for comparison");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const native_compare_result native = run_native_compare(handoff, model, params);

    float max_abs_diff = 0.0f;
    double mean_abs_diff = 0.0;
    llama_token max_diff_token = -1;
    for (llama_token token = 0; token < n_vocab; ++token) {
        const float diff = std::fabs(imported_logits[token] - native.resume_logits[token]);
        mean_abs_diff += diff;
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_diff_token = token;
        }
    }
    mean_abs_diff /= static_cast<double>(n_vocab);

    const auto imported_top = top_k_logits(imported_logits, n_vocab, 5);
    const auto native_top = top_k_logits(native.resume_logits.data(), n_vocab, 5);
    const auto native_prompt_top = top_k_logits(native.prompt_logits.data(), n_vocab, 5);

    LOG_INF(
        "PD native compare prompt-only top1=%d handoff_first_token=%d\n",
        native_prompt_top.front().token,
        handoff.first_token);
    LOG_INF(
        "PD native compare native prompt top5: %s\n",
        describe_top_k(imported_ctx, native_prompt_top, false).c_str());

    LOG_INF(
        "PD native compare: imported_top1=%d native_top1=%d max_abs_diff=%f mean_abs_diff=%f max_diff_token=%d\n",
        imported_top.front().token,
        native_top.front().token,
        max_abs_diff,
        mean_abs_diff,
        max_diff_token);
    LOG_INF(
        "PD native compare imported top5: %s\n",
        describe_top_k(imported_ctx, imported_top, false).c_str());
    LOG_INF(
        "PD native compare native   top5: %s\n",
        describe_top_k(imported_ctx, native_top, false).c_str());

    log_prompt_seq_blob_comparison(
        imported_prompt_seq_blob,
        native.prompt_seq_blob,
        handoff,
        v_trans);
}

void print_usage(int argc, char ** argv) {
    (void) argc;
    LOG("\nexample usage:\n");
    LOG("  %s --pd-import handoff_dir -m model.gguf -n 128 -c 2048 -t 4 -ngl 0\n", argv[0]);
    LOG("  diagnostics: add --pd-roundtrip-check to compare imported KV against llama.cpp sequence serialization\n");
    LOG("  diagnostics: add --pd-native-compare to compare imported KV resume logits against native GGUF prefill\n");
    LOG("  quality fallback: add --pd-native-first-token to select the first continuation token with GGUF prompt prefill\n");
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

    if (pd.native_first_token) {
        const llama_token handoff_first_token = handoff.first_token;
        try {
            const native_compare_result native =
                run_native_compare(handoff, model, params);
            const auto native_prompt_top = top_k_logits(
                native.prompt_logits.data(), llama_vocab_n_tokens(llama_model_get_vocab(model)), 1);
            if (native_prompt_top.empty()) {
                throw std::runtime_error("native prompt logits are empty");
            }
            handoff.first_token = native_prompt_top.front().token;
            LOG_INF(
                "PD native first-token override: handoff=%d native_prompt_top1=%d logit=%f\n",
                handoff_first_token,
                handoff.first_token,
                native_prompt_top.front().logit);
        } catch (const std::exception & err) {
            LOG_ERR("PD native first-token selection failed: %s\n", err.what());
            return 1;
        }
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

    if (pd.native_compare) {
        try {
            log_native_comparison(ctx, model, params, handoff, seq_blob, v_trans);
        } catch (const std::exception & err) {
            LOG_ERR("PD native comparison failed: %s\n", err.what());
            return 1;
        }
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
