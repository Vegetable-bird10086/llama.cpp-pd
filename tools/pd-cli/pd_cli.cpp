#include "arg.h"
#include "common.h"
#include "json.hpp"
#include "llama.h"
#include "llama-qnn-u16.h"
#include "log.h"
#include "sampling.h"

#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

struct pd_args {
    std::string import_dir;
    std::string ppl_tokens_path;
    std::string ppl_output_path;
    int32_t ppl_max_tokens = 0;
    bool import_ro = false;
    bool roundtrip_check = false;
    bool native_compare = false;
    bool native_first_token = false;
};

struct pd_capture_state {
    std::filesystem::path output_dir;
    std::unordered_set<std::string> requested;
    std::unordered_set<std::string> captured;
    std::string error;
    bool active = false;
    bool completed = false;
};

static const char * const pd_default_capture_nodes[] = {
    "qnn_u16_input",
    "qnn_attn_norm-0",
    "qnn_q_projection-0",
    "qnn_k_projection-0",
    "qnn_v_projection-0",
    "qnn_q_head_norm-0",
    "qnn_q_head_rope-0",
    "qnn_q_head_rotate-0",
    "qnn_k_head_norm-0",
    "qnn_k_head_rope-0",
    "qnn_k_head_rotate-0",
    "qnn_k_head_u8-0",
    "qnn_v_head_u8-0",
    "qnn_attention_score-0",
    "qnn_attention_scaled-0",
    "qnn_attention_min-0",
    "qnn_attention_mask_value-0",
    "qnn_attention_masked-0",
    "qnn_attention_softmax-0",
    "qnn_attention_value-0",
    "qnn_attention_concat-0",
    "qnn_attention_output-0",
    "qnn_ffn_input-0",
    "qnn_ffn_norm-0",
    "qnn_ffn_gate-0",
    "qnn_ffn_up-0",
    "qnn_ffn_sigmoid-0",
    "qnn_ffn_silu-0",
    "qnn_ffn_product-0",
    "qnn_ffn_down-0",
    "qnn_layer_output-0",
    "qnn_attn_norm-1",
    "qnn_q_projection-1",
    "qnn_k_projection-1",
    "qnn_v_projection-1",
    "qnn_q_head_norm-1",
    "qnn_q_head_rope-1",
    "qnn_q_head_rotate-1",
    "qnn_k_head_norm-1",
    "qnn_k_head_rope-1",
    "qnn_k_head_rotate-1",
    "qnn_k_head_u8-1",
    "qnn_v_head_u8-1",
    "qnn_attention_score-1",
    "qnn_attention_scaled-1",
    "qnn_attention_min-1",
    "qnn_attention_mask_value-1",
    "qnn_attention_masked-1",
    "qnn_attention_softmax-1",
    "qnn_attention_value-1",
    "qnn_attention_concat-1",
    "qnn_attention_output-1",
    "qnn_ffn_input-1",
    "qnn_ffn_norm-1",
    "qnn_ffn_gate-1",
    "qnn_ffn_up-1",
    "qnn_ffn_sigmoid-1",
    "qnn_ffn_silu-1",
    "qnn_ffn_product-1",
    "qnn_ffn_down-1",
    "qnn_layer_output-1",
};

static void configure_pd_capture(pd_capture_state & state, const char * output_dir) {
    state.output_dir = output_dir;
    std::filesystem::create_directories(state.output_dir);
    state.requested.insert(
        std::begin(pd_default_capture_nodes),
        std::end(pd_default_capture_nodes));
}

static bool write_pd_capture(
        ggml_tensor * tensor,
        const std::string & name,
        pd_capture_state & state) {
    if (tensor->data == nullptr) {
        state.error = "target tensor has no host-accessible data: " + name;
        return false;
    }
    const char * suffix = nullptr;
    switch (tensor->type) {
        case GGML_TYPE_U16: suffix = ".u16.bin"; break;
        case GGML_TYPE_I8:  suffix = ".u8.bin";  break;
        default:
            state.error = "unsupported target tensor type: " + name +
                " type=" + ggml_type_name(tensor->type);
            return false;
    }
    if (!ggml_is_contiguous(tensor)) {
        state.error = "target tensor is not contiguous: " + name;
        return false;
    }

    const size_t nbytes = ggml_nbytes(tensor);
    const std::filesystem::path binary_path = state.output_dir / (name + suffix);
    std::ofstream binary(binary_path, std::ios::binary | std::ios::trunc);
    if (!binary.is_open()) {
        state.error = "unable to write " + binary_path.string();
        return false;
    }
    binary.write(reinterpret_cast<const char *>(tensor->data), static_cast<std::streamsize>(nbytes));
    binary.close();
    if (!binary) {
        state.error = "short write for " + binary_path.string();
        return false;
    }

    const std::filesystem::path metadata_path = state.output_dir / (name + ".json");
    std::ofstream metadata(metadata_path, std::ios::trunc);
    if (!metadata.is_open()) {
        state.error = "unable to write " + metadata_path.string();
        return false;
    }
    metadata << "{\n"
             << "  \"tensor_name\": \"" << name << "\",\n"
             << "  \"ggml_type\": \"" << ggml_type_name(tensor->type) << "\",\n"
             << "  \"nbytes\": " << nbytes << ",\n"
             << "  \"ne\": ["
             << tensor->ne[0] << ", " << tensor->ne[1] << ", "
             << tensor->ne[2] << ", " << tensor->ne[3] << "],\n"
             << "  \"nb\": ["
             << tensor->nb[0] << ", " << tensor->nb[1] << ", "
             << tensor->nb[2] << ", " << tensor->nb[3] << "]\n"
             << "}\n";
    return true;
}

static bool pd_capture_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & state = *static_cast<pd_capture_state *>(user_data);
    if (!state.active || !state.error.empty()) {
        return false;
    }
    const char * raw_name = ggml_get_name(tensor);
    const std::string name = raw_name == nullptr ? "" : raw_name;
    if (state.requested.count(name) == 0 || state.captured.count(name) != 0) {
        return false;
    }
    if (ask) {
        return true;
    }
    if (write_pd_capture(tensor, name, state)) {
        state.captured.insert(name);
    }
    return true;
}

static void finish_pd_capture(pd_capture_state & state, llama_token token) {
    state.completed = true;
    std::ofstream summary(state.output_dir / "capture.json", std::ios::trunc);
    summary << "{\n"
            << "  \"decode_input_token\": " << token << ",\n"
            << "  \"requested_nodes\": " << state.requested.size() << ",\n"
            << "  \"captured_nodes\": " << state.captured.size() << ",\n"
            << "  \"error\": \"" << state.error << "\",\n"
            << "  \"missing_nodes\": [";
    bool first = true;
    for (const std::string & name : state.requested) {
        if (state.captured.count(name) != 0) {
            continue;
        }
        summary << (first ? "\n" : ",\n") << "    \"" << name << "\"";
        first = false;
    }
    if (!first) {
        summary << '\n';
    }
    summary << "  ]\n}\n";
}

using steady_clock = std::chrono::steady_clock;

static double elapsed_ms(
        steady_clock::time_point start,
        steady_clock::time_point end = steady_clock::now()) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct process_memory_snapshot {
    uint64_t rss_bytes = 0;
    uint64_t hwm_bytes = 0;
};

static uint64_t read_proc_status_bytes(const char * field) {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind(field, 0) != 0) {
            continue;
        }
        std::istringstream value(line.substr(std::strlen(field)));
        uint64_t kib = 0;
        value >> kib;
        return kib * 1024;
    }
    return 0;
}

static process_memory_snapshot process_memory() {
    return {
        read_proc_status_bytes("VmRSS:"),
        read_proc_status_bytes("VmHWM:"),
    };
}

static double bytes_to_mib(uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

struct pd_handoff {
    json manifest;
    std::vector<llama_token> prompt_tokens;
    llama_token first_token = -1;
    bool first_token_is_prompt_tail = false;
    std::vector<uint16_t> kv_fp16;
    std::vector<uint8_t> kv_qnn_u8;
    int32_t prompt_len = 0;
    int32_t num_layers = 0;
    int32_t num_kv_heads = 0;
    int32_t head_dim = 0;
    double metadata_read_ms = 0.0;
    double kv_read_ms = 0.0;
};

class blob_writer {
public:
    explicit blob_writer(size_t size) : data_(size) {}

    template <typename T>
    void write_pod(const T & value) {
        write_bytes(&value, sizeof(T));
    }

    void write_bytes(const void * data, size_t size) {
        if (size > data_.size() - offset_) {
            throw std::runtime_error("sequence state blob size calculation overflow");
        }
        std::memcpy(data_.data() + offset_, data, size);
        offset_ += size;
    }

    std::vector<uint8_t> finish() {
        if (offset_ != data_.size()) {
            throw std::runtime_error("sequence state blob size calculation mismatch");
        }
        return std::move(data_);
    }

private:
    std::vector<uint8_t> data_;
    size_t offset_ = 0;
};

template <typename T>
std::vector<T> read_binary_vector(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        throw std::runtime_error("unable to read file: " + path);
    }
    const std::streampos end = input.tellg();
    if (end < 0) {
        throw std::runtime_error("unable to determine file size: " + path);
    }
    const size_t size_bytes = static_cast<size_t>(end);
    if (size_bytes % sizeof(T) != 0) {
        throw std::runtime_error("binary file element alignment mismatch: " + path);
    }
    input.seekg(0, std::ios::beg);
    std::vector<T> out(size_bytes / sizeof(T));
    if (size_bytes != 0) {
        input.read(reinterpret_cast<char *>(out.data()), size_bytes);
        if (!input || static_cast<size_t>(input.gcount()) != size_bytes) {
            throw std::runtime_error("short read from file: " + path);
        }
    }
    return out;
}

std::vector<uint8_t> read_binary_file(const std::string & path) {
    return read_binary_vector<uint8_t>(path);
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
        if (arg == "--pd-ppl-tokens") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--pd-ppl-tokens requires a raw uint64 token file");
            }
            out.ppl_tokens_path = argv[++i];
            continue;
        }
        if (arg == "--pd-ppl-output") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--pd-ppl-output requires a path");
            }
            out.ppl_output_path = argv[++i];
            continue;
        }
        if (arg == "--pd-ppl-max-tokens") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--pd-ppl-max-tokens requires a count");
            }
            out.ppl_max_tokens = std::stoi(argv[++i]);
            if (out.ppl_max_tokens < 0) {
                throw std::runtime_error("--pd-ppl-max-tokens cannot be negative");
            }
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

double token_nll(const float * logits, int32_t n_vocab, llama_token target) {
    if (logits == nullptr || target < 0 || target >= n_vocab) {
        throw std::runtime_error("invalid logits or target token for PPL");
    }
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < n_vocab; ++token) {
        max_logit = std::max(max_logit, logits[token]);
    }
    double exp_sum = 0.0;
    for (int32_t token = 0; token < n_vocab; ++token) {
        exp_sum += std::exp(static_cast<double>(logits[token] - max_logit));
    }
    return std::log(exp_sum) -
        static_cast<double>(logits[target] - max_logit);
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
    const auto metadata_read_start = steady_clock::now();
    out.manifest = read_json_file(import_dir + "/manifest.json");
    out.prompt_tokens = load_prompt_tokens(import_dir + "/prompt_tokens.bin");
    out.first_token = load_first_token(out.manifest, import_dir);
    out.first_token_is_prompt_tail =
        out.manifest.value("first_token_is_prompt_tail", false);
    out.metadata_read_ms = elapsed_ms(metadata_read_start);

    const auto kv_read_start = steady_clock::now();
    out.kv_fp16 = read_binary_vector<uint16_t>(import_dir + "/kv.bin");
    const std::string qnn_u8_kv_file = out.manifest.value("qnn_u8_kv_file", "");
    if (!qnn_u8_kv_file.empty()) {
        out.kv_qnn_u8 = read_binary_vector<uint8_t>(import_dir + "/" + qnn_u8_kv_file);
    }
    out.kv_read_ms = elapsed_ms(kv_read_start);

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
    if (!out.kv_qnn_u8.empty() && out.kv_qnn_u8.size() != expected_values) {
        std::ostringstream oss;
        oss << "QNN U8 kv file element count mismatch: got=" << out.kv_qnn_u8.size()
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

    const int32_t required_ctx = handoff.prompt_len +
        (handoff.first_token_is_prompt_tail ? 1 : 0);
    if (required_ctx > static_cast<int32_t>(llama_n_ctx(ctx))) {
        std::ostringstream oss;
        oss << "PD prompt length exceeds decode context: required=" << required_ctx
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

std::vector<uint8_t> build_seq_state_blob(
        const pd_handoff & handoff,
        bool v_trans,
        bool qnn_u8_layout) {
    static constexpr uint32_t k_state_seq_magic = 0xaf143cd8U;
    static constexpr llama_seq_id k_seq_id = 0;
    static constexpr uint32_t k_n_stream = 1;

    const uint32_t cell_count = static_cast<uint32_t>(handoff.prompt_len);
    const uint32_t n_layer = static_cast<uint32_t>(handoff.num_layers);
    const uint32_t n_embd_k_gqa =
        static_cast<uint32_t>(handoff.num_kv_heads * handoff.head_dim);
    const uint32_t n_embd_v_gqa = n_embd_k_gqa;
    if (qnn_u8_layout && v_trans) {
        throw std::runtime_error("QNN U8 KV requires non-transposed V layout");
    }
    if (qnn_u8_layout && handoff.kv_qnn_u8.empty()) {
        throw std::runtime_error("QNN U8 KV handoff is missing kv_qnn_u8.bin");
    }
    const size_t kv_element_size = qnn_u8_layout ? sizeof(uint8_t) : sizeof(uint16_t);
    const uint64_t k_row_size =
        static_cast<uint64_t>(n_embd_k_gqa) * kv_element_size;
    const uint64_t v_row_size = k_row_size;
    const size_t per_kind_values =
        static_cast<size_t>(handoff.num_layers) *
        handoff.num_kv_heads *
        handoff.prompt_len *
        handoff.head_dim;
    const uint16_t * k_base = handoff.kv_fp16.data();
    const uint16_t * v_base = handoff.kv_fp16.data() + per_kind_values;
    const uint8_t * k_u8_base = handoff.kv_qnn_u8.data();
    const uint8_t * v_u8_base = handoff.kv_qnn_u8.data() +
        (qnn_u8_layout ? per_kind_values : 0);

    const size_t state_header_size =
        sizeof(uint32_t) + sizeof(llama_seq_id) +
        sizeof(uint32_t) + sizeof(uint32_t);
    const size_t cell_metadata_size =
        static_cast<size_t>(cell_count) *
        (sizeof(llama_pos) + sizeof(uint32_t) + sizeof(llama_seq_id));
    const size_t kv_header_size = sizeof(uint32_t) * 2;
    const size_t k_layer_size =
        sizeof(int32_t) + sizeof(uint64_t) +
        static_cast<size_t>(cell_count) * k_row_size;
    const size_t v_layer_header_size =
        sizeof(int32_t) +
        (v_trans ? sizeof(uint32_t) * 2 : sizeof(uint64_t));
    const size_t v_layer_size =
        v_layer_header_size + static_cast<size_t>(cell_count) * v_row_size;
    const size_t blob_size =
        state_header_size + cell_metadata_size + kv_header_size +
        static_cast<size_t>(n_layer) * (k_layer_size + v_layer_size);
    blob_writer writer(blob_size);

    writer.write_pod(k_state_seq_magic);
    writer.write_pod(k_seq_id);
    writer.write_pod(k_n_stream);
    writer.write_pod(cell_count);
    for (uint32_t pos = 0; pos < cell_count; ++pos) {
        const llama_pos llama_position = static_cast<llama_pos>(pos);
        const uint32_t n_seq_id = 1;
        writer.write_pod(llama_position);
        writer.write_pod(n_seq_id);
        writer.write_pod(k_seq_id);
    }

    const uint32_t v_trans_u32 = v_trans ? 1u : 0u;
    writer.write_pod(v_trans_u32);
    writer.write_pod(n_layer);

    for (int32_t layer = 0; layer < handoff.num_layers; ++layer) {
        const int32_t type = qnn_u8_layout ? GGML_TYPE_I8 : GGML_TYPE_F16;
        writer.write_pod(type);
        writer.write_pod(k_row_size);
        for (int32_t pos = 0; pos < handoff.prompt_len; ++pos) {
            for (int32_t head = 0; head < handoff.num_kv_heads; ++head) {
                const size_t src = (((static_cast<size_t>(layer) * handoff.num_kv_heads + head) *
                                    handoff.prompt_len + pos) *
                                    handoff.head_dim);
                if (qnn_u8_layout) {
                    writer.write_bytes(
                        k_u8_base + src,
                        static_cast<size_t>(handoff.head_dim));
                } else {
                    writer.write_bytes(
                        k_base + src,
                        static_cast<size_t>(handoff.head_dim) * sizeof(uint16_t));
                }
            }
        }
    }

    for (int32_t layer = 0; layer < handoff.num_layers; ++layer) {
        const int32_t type = qnn_u8_layout ? GGML_TYPE_I8 : GGML_TYPE_F16;
        writer.write_pod(type);
        if (!v_trans) {
            writer.write_pod(v_row_size);
            for (int32_t pos = 0; pos < handoff.prompt_len; ++pos) {
                for (int32_t head = 0; head < handoff.num_kv_heads; ++head) {
                    const size_t src = (((static_cast<size_t>(layer) * handoff.num_kv_heads + head) *
                                        handoff.prompt_len + pos) *
                                        handoff.head_dim);
                    if (qnn_u8_layout) {
                        writer.write_bytes(
                            v_u8_base + src,
                            static_cast<size_t>(handoff.head_dim));
                    } else {
                        writer.write_bytes(
                            v_base + src,
                            static_cast<size_t>(handoff.head_dim) * sizeof(uint16_t));
                    }
                }
            }
        } else {
            const uint32_t v_size_el = sizeof(uint16_t);
            writer.write_pod(v_size_el);
            writer.write_pod(n_embd_v_gqa);
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                const int32_t head = static_cast<int32_t>(j / handoff.head_dim);
                const int32_t dim = static_cast<int32_t>(j % handoff.head_dim);
                for (int32_t pos = 0; pos < handoff.prompt_len; ++pos) {
                    const size_t src =
                        (((static_cast<size_t>(layer) * handoff.num_kv_heads + head) *
                          handoff.prompt_len + pos) *
                         handoff.head_dim) + dim;
                    writer.write_bytes(v_base + src, sizeof(uint16_t));
                }
            }
        }
    }

    return writer.finish();
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
    LOG("  PPL: add --pd-ppl-tokens continuation.u64 --pd-ppl-output result.txt for teacher-forced scoring\n");
    LOG("\n");
}

} // namespace

int main(int argc, char ** argv) {
    const auto process_start = steady_clock::now();
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

    pd_capture_state pd_capture;
    if (const char * dump_dir = std::getenv("LLAMA_QNN_PD_DUMP_DIR");
            dump_dir != nullptr && dump_dir[0] != '\0') {
        try {
            configure_pd_capture(pd_capture, dump_dir);
        } catch (const std::exception & err) {
            LOG_ERR("failed to configure PD intermediate capture: %s\n", err.what());
            return 1;
        }
        params.cb_eval = pd_capture_callback;
        params.cb_eval_user_data = &pd_capture;
        LOG_INF(
            "PD intermediate capture armed: dir=%s nodes=%zu\n",
            dump_dir,
            pd_capture.requested.size());
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

    const process_memory_snapshot memory_after_model_load = process_memory();

    pd_handoff handoff;
    try {
        handoff = load_pd_handoff(pd.import_dir);
        validate_pd_handoff(handoff, model, ctx);
    } catch (const std::exception & err) {
        LOG_ERR("failed to load PD handoff: %s\n", err.what());
        return 1;
    }

    if (pd.native_first_token) {
        if (handoff.first_token_is_prompt_tail) {
            LOG_ERR("--pd-native-first-token is incompatible with a prompt-tail bridge handoff\n");
            return 1;
        }
        const llama_token handoff_first_token = handoff.first_token;
        const bool capture_native_prompt =
            !pd_capture.output_dir.empty() && !pd_capture.completed;
        try {
            pd_capture.active = capture_native_prompt;
            const native_compare_result native =
                run_native_compare(handoff, model, params);
            pd_capture.active = false;
            if (capture_native_prompt) {
                finish_pd_capture(pd_capture, handoff.prompt_tokens.back());
                LOG_INF(
                    "PD native prompt capture finished: tokens=%d captured=%zu requested=%zu error=%s\n",
                    handoff.prompt_len,
                    pd_capture.captured.size(),
                    pd_capture.requested.size(),
                    pd_capture.error.empty() ? "none" : pd_capture.error.c_str());
            }
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
            pd_capture.active = false;
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
    const bool qnn_u8_layout = llama_qnn_u16_activations_enabled();
    const bool v_trans = qnn_u8_layout
        ? false
        : params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED;
    const auto kv_layout_start = steady_clock::now();
    const std::vector<uint8_t> seq_blob =
        build_seq_state_blob(handoff, v_trans, qnn_u8_layout);
    const double kv_layout_ms = elapsed_ms(kv_layout_start);

    const auto kv_import_start = steady_clock::now();
    const size_t nset =
        llama_state_seq_set_data(ctx, seq_blob.data(), seq_blob.size(), 0);
    const double kv_import_ms = elapsed_ms(kv_import_start);
    if (nset != seq_blob.size()) {
        LOG_ERR(
            "failed to import PD KV state: written=%zu expected=%zu\n",
            nset,
            seq_blob.size());
        return 1;
    }

    const auto kv_validation_start = steady_clock::now();
    try {
        validate_imported_kv_state(handoff, ctx);
    } catch (const std::exception & err) {
        LOG_ERR("PD handoff imported but KV validation failed: %s\n", err.what());
        return 1;
    }
    const double kv_validation_ms = elapsed_ms(kv_validation_start);
    const double handoff_total_ms =
        handoff.metadata_read_ms + handoff.kv_read_ms +
        kv_layout_ms + kv_import_ms + kv_validation_ms;
    LOG_INF(
        "PD handoff timing: metadata_read_ms=%.3f kv_read_ms=%.3f "
        "kv_layout_ms=%.3f kv_import_ms=%.3f validation_ms=%.3f "
        "total_ms=%.3f kv_bytes=%zu seq_blob_bytes=%zu\n",
        handoff.metadata_read_ms,
        handoff.kv_read_ms,
        kv_layout_ms,
        kv_import_ms,
        kv_validation_ms,
        handoff_total_ms,
        handoff.kv_fp16.size() * sizeof(uint16_t),
        seq_blob.size());
    const process_memory_snapshot memory_after_import = process_memory();

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
        "PD handoff imported: prompt_len=%d first_token=%d bridge_prompt_tail=%d layers=%d kv_heads=%d head_dim=%d\n",
        handoff.prompt_len,
        handoff.first_token,
        handoff.first_token_is_prompt_tail,
        handoff.num_layers,
        handoff.num_kv_heads,
        handoff.head_dim);

    if (pd.import_ro) {
        LOG_INF("PD import validation completed in read-only mode\n");
        return 0;
    }

    int n_remain = params.n_predict;
    const bool infinite = n_remain < 0;

    int32_t generated_tokens = 0;
    const auto decode_start = steady_clock::now();
    const auto decode_one = [&](llama_token token) {
        const bool capture_this_decode =
            !pd_capture.output_dir.empty() && !pd_capture.completed;
        pd_capture.active = capture_this_decode;
        const int result = llama_decode(ctx, llama_batch_get_one(&token, 1));
        pd_capture.active = false;
        if (capture_this_decode) {
            finish_pd_capture(pd_capture, token);
            LOG_INF(
                "PD intermediate capture finished: token=%d captured=%zu requested=%zu error=%s\n",
                token,
                pd_capture.captured.size(),
                pd_capture.requested.size(),
                pd_capture.error.empty() ? "none" : pd_capture.error.c_str());
        }
        return result;
    };

    if (!pd.ppl_tokens_path.empty()) {
        if (handoff.first_token_is_prompt_tail) {
            LOG_ERR("PD PPL requires a logits-producing prefill handoff\n");
            return 1;
        }
        std::vector<llama_token> continuation;
        try {
            continuation = load_prompt_tokens(pd.ppl_tokens_path);
        } catch (const std::exception & err) {
            LOG_ERR("failed to load PD PPL tokens: %s\n", err.what());
            return 1;
        }
        if (continuation.size() < 2) {
            LOG_ERR("PD PPL requires at least two continuation tokens\n");
            return 1;
        }
        if (pd.ppl_max_tokens > 0 &&
            continuation.size() > static_cast<size_t>(pd.ppl_max_tokens) + 1) {
            continuation.resize(static_cast<size_t>(pd.ppl_max_tokens) + 1);
        }
        if (static_cast<int64_t>(handoff.prompt_len) +
                static_cast<int64_t>(continuation.size()) >
            llama_n_ctx(ctx)) {
            LOG_ERR(
                "PD PPL sequence exceeds context: prompt=%d continuation=%zu context=%u\n",
                handoff.prompt_len,
                continuation.size(),
                llama_n_ctx(ctx));
            return 1;
        }

        const int32_t n_vocab =
            llama_vocab_n_tokens(llama_model_get_vocab(model));
        double total_nll = 0.0;
        int64_t scored_tokens = 0;
        const auto ppl_start = steady_clock::now();
        for (size_t index = 0; index + 1 < continuation.size(); ++index) {
            if (decode_one(continuation[index]) != 0) {
                LOG_ERR("llama_decode failed during PD PPL at continuation index %zu\n", index);
                return 1;
            }
            const float * logits = llama_get_logits_ith(ctx, -1);
            try {
                total_nll += token_nll(
                    logits, n_vocab, continuation[index + 1]);
            } catch (const std::exception & err) {
                LOG_ERR("PD PPL scoring failed: %s\n", err.what());
                return 1;
            }
            ++scored_tokens;
        }
        const double ppl = std::exp(total_nll / static_cast<double>(scored_tokens));
        const double ppl_ms = elapsed_ms(ppl_start);
        LOG_INF(
            "PD WikiPPL: ppl=%.9f nll=%.9f scored_tokens=%" PRId64
            " prompt_tokens=%d continuation_tokens=%zu eval_ms=%.3f tokens_per_second=%.3f\n",
            ppl,
            total_nll,
            scored_tokens,
            handoff.prompt_len,
            continuation.size(),
            ppl_ms,
            1000.0 * static_cast<double>(scored_tokens) / ppl_ms);
        if (!pd.ppl_output_path.empty()) {
            std::ofstream output(pd.ppl_output_path, std::ios::trunc);
            if (!output.is_open()) {
                LOG_ERR("failed to open PD PPL output: %s\n", pd.ppl_output_path.c_str());
                return 1;
            }
            output << "wiki_ppl=" << ppl << "\n"
                   << "total_nll=" << total_nll << "\n"
                   << "scored_tokens=" << scored_tokens << "\n"
                   << "prompt_tokens=" << handoff.prompt_len << "\n"
                   << "continuation_tokens=" << continuation.size() << "\n"
                   << "eval_ms=" << ppl_ms << "\n";
        }
        common_perf_print(ctx, smpl);
        return 0;
    }

    if (!infinite && n_remain == 0) {
        return 0;
    }

    if (handoff.first_token_is_prompt_tail) {
        common_sampler_accept(smpl, handoff.first_token, false);
        if (decode_one(handoff.first_token) != 0) {
            LOG_ERR("llama_decode failed for PD prompt-tail bridge token\n");
            return 1;
        }
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
    llama_token cur = handoff.first_token_is_prompt_tail
        ? common_sampler_sample(smpl, ctx, -1)
        : handoff.first_token;
    while (infinite || n_remain > 0) {
        common_sampler_accept(smpl, cur, true);
        ++generated_tokens;
        LOG("%s", common_token_to_piece(ctx, cur, params.special).c_str());

        if (llama_vocab_is_eog(vocab, cur)) {
            break;
        }
        if (!infinite && --n_remain == 0) {
            break;
        }

        if (decode_one(cur) != 0) {
            LOG_ERR("llama_decode failed while continuing decode\n");
            return 1;
        }
        cur = common_sampler_sample(smpl, ctx, -1);
    }

    LOG("\n");
    const double decode_ms = elapsed_ms(decode_start);
    const process_memory_snapshot memory_after_decode = process_memory();
    LOG_INF(
        "PD decode runtime summary: generated_tokens=%d handoff_ms=%.3f "
        "decode_ms=%.3f process_total_ms=%.3f\n",
        generated_tokens,
        handoff_total_ms,
        decode_ms,
        elapsed_ms(process_start));
    LOG_INF(
        "PD decode memory MiB: after_model_load_rss=%.2f after_import_rss=%.2f "
        "after_decode_rss=%.2f hwm=%.2f\n",
        bytes_to_mib(memory_after_model_load.rss_bytes),
        bytes_to_mib(memory_after_import.rss_bytes),
        bytes_to_mib(memory_after_decode.rss_bytes),
        bytes_to_mib(memory_after_decode.hwm_bytes));
    common_perf_print(ctx, smpl);
    return 0;
}
