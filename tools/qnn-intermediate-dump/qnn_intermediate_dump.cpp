#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

struct args {
    std::string model_path;
    std::string prompt;
    std::string tokens_path;
    std::string output_dir;
    std::vector<std::string> nodes;
    int32_t context_size = 4096;
    int32_t threads = 4;
    bool add_special = true;
    bool all_qnn_stages = false;
};

struct capture_state {
    std::filesystem::path output_dir;
    std::unordered_set<std::string> requested;
    std::unordered_set<std::string> captured;
    std::string error;
    bool active = false;
};

void usage(const char * program) {
    std::cerr << "usage: " << program
              << " --model MODEL.gguf (--prompt TEXT | --tokens FILE) --out DIRECTORY"
              << " [--ctx N] [--threads N] [--nodes NAME,NAME]"
              << " [--all-qnn-stages] [--no-special]\n";
}

std::vector<std::string> all_qnn_stage_names() {
    std::vector<std::string> nodes = {"qnn_u16_input"};
    const std::vector<std::string> layer_stages = {
        "qnn_attn_norm",
        "qnn_q_projection",
        "qnn_k_projection",
        "qnn_v_projection",
        "qnn_attention_concat",
        "qnn_attention_output",
        "qnn_ffn_input",
        "qnn_ffn_norm",
        "qnn_ffn_gate",
        "qnn_ffn_up",
        "qnn_ffn_sigmoid",
        "qnn_ffn_silu",
        "qnn_ffn_product",
        "qnn_ffn_down",
        "qnn_layer_output",
    };
    const std::vector<std::string> q_head_stages = {
        "qnn_q_head_norm",
        "qnn_q_head_rope",
        "qnn_q_head_rotate",
    };
    const std::vector<std::string> kv_head_stages = {
        "qnn_k_head_norm",
        "qnn_k_head_rope",
        "qnn_k_head_rotate",
        "qnn_k_head_u8",
        "qnn_v_head_u8",
    };
    const std::vector<std::string> attention_head_stages = {
        "qnn_attention_score",
        "qnn_attention_scaled",
        "qnn_attention_min",
        "qnn_attention_mask_value",
        "qnn_attention_masked",
        "qnn_attention_softmax",
        "qnn_attention_value",
    };
    for (int32_t layer = 0; layer < 28; ++layer) {
        for (const std::string & stem : layer_stages) {
            nodes.push_back(stem + "-" + std::to_string(layer));
        }
        for (const std::string & stem : q_head_stages) {
            for (int32_t head = 0; head < 16; ++head) {
                const std::string suffix =
                    head == 0 ? "" : "_h" + std::to_string(head);
                nodes.push_back(stem + suffix + "-" + std::to_string(layer));
            }
        }
        for (const std::string & stem : kv_head_stages) {
            for (int32_t head = 0; head < 8; ++head) {
                const std::string suffix =
                    head == 0 ? "" : "_h" + std::to_string(head);
                nodes.push_back(stem + suffix + "-" + std::to_string(layer));
            }
        }
        for (const std::string & stem : attention_head_stages) {
            for (int32_t head = 0; head < 16; ++head) {
                const std::string suffix =
                    head == 0 ? "" : "_h" + std::to_string(head);
                nodes.push_back(stem + suffix + "-" + std::to_string(layer));
            }
        }
    }
    return nodes;
}

int32_t parse_positive_i32(const std::string & value, const char * name) {
    size_t used = 0;
    const long parsed = std::stol(value, &used, 10);
    if (used != value.size() || parsed <= 0 ||
        parsed > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return static_cast<int32_t>(parsed);
}

std::vector<llama_token> read_tokens(const std::string & path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("unable to read token file");
    }
    std::vector<llama_token> tokens;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        size_t used = 0;
        const long parsed = std::stol(line, &used, 10);
        if (used != line.size() ||
            parsed < std::numeric_limits<llama_token>::min() ||
            parsed > std::numeric_limits<llama_token>::max()) {
            throw std::runtime_error("invalid token in token file");
        }
        tokens.push_back(static_cast<llama_token>(parsed));
    }
    if (tokens.empty()) {
        throw std::runtime_error("token file is empty");
    }
    return tokens;
}

std::vector<std::string> parse_nodes(const std::string & value) {
    std::vector<std::string> nodes;
    std::stringstream stream(value);
    std::string name;
    while (std::getline(stream, name, ',')) {
        if (name.empty()) {
            throw std::runtime_error("empty node name in --nodes");
        }
        nodes.push_back(name);
    }
    return nodes;
}

args parse_args(int argc, char ** argv) {
    args result;
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--no-special") {
            result.add_special = false;
            continue;
        }
        if (flag == "--all-qnn-stages") {
            result.all_qnn_stages = true;
            continue;
        }
        if (flag == "--help" || flag == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        if (flag != "--model" && flag != "--prompt" && flag != "--tokens" &&
            flag != "--out" && flag != "--ctx" && flag != "--threads" &&
            flag != "--nodes") {
            throw std::runtime_error("unknown argument: " + flag);
        }
        if (++i >= argc) {
            throw std::runtime_error("missing value for " + flag);
        }
        const std::string value(argv[i]);
        if (flag == "--model") {
            result.model_path = value;
        } else if (flag == "--prompt") {
            result.prompt = value;
        } else if (flag == "--tokens") {
            result.tokens_path = value;
        } else if (flag == "--out") {
            result.output_dir = value;
        } else if (flag == "--ctx") {
            result.context_size = parse_positive_i32(value, "--ctx");
        } else if (flag == "--threads") {
            result.threads = parse_positive_i32(value, "--threads");
        } else {
            result.nodes = parse_nodes(value);
        }
    }
    if (result.model_path.empty() || result.output_dir.empty() ||
        (result.prompt.empty() == result.tokens_path.empty())) {
        throw std::runtime_error("--model, exactly one of --prompt/--tokens, and --out are required");
    }
    if (result.nodes.empty()) {
        result.nodes = result.all_qnn_stages ? all_qnn_stage_names() :
            std::vector<std::string>{
            "attn_norm-0",
            "Qpre_norm-0",
            "Kpre_norm-0",
            "Qcur_normed-0",
            "Kcur_normed-0",
            "Qpost_rope-0",
            "Kpost_rope-0",
            "Vcur-0",
            "kq-0",
            "kq_soft_max-0",
            "kqv-0",
            "kqv_out-0",
            "attn_out-0",
            "ffn_inp-0",
            "ffn_norm-0",
            "ffn_out-0",
            "l_out-0",
        };
    } else if (result.all_qnn_stages) {
        throw std::runtime_error("--nodes and --all-qnn-stages are mutually exclusive");
    }
    return result;
}

bool write_capture(ggml_tensor * tensor, const std::string & name, capture_state & state) {
    if (tensor->data == nullptr) {
        state.error = "target tensor has no host-accessible data: " + name;
        return false;
    }
    const char * suffix = nullptr;
    switch (tensor->type) {
        case GGML_TYPE_F32: suffix = ".f32.bin"; break;
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

bool eval_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & state = *static_cast<capture_state *>(user_data);
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
    if (write_capture(tensor, name, state)) {
        state.captured.insert(name);
    }
    // Returning false here would cancel the graph. Preserve normal execution
    // even if the diagnostic write failed, then report the error after decode.
    return true;
}

void write_tokens(const std::filesystem::path & output_dir, const std::vector<llama_token> & tokens) {
    std::ofstream output(output_dir / "llama_tokens.txt", std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("unable to write llama_tokens.txt");
    }
    for (const llama_token token : tokens) {
        output << token << '\n';
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const args options = parse_args(argc, argv);
        std::filesystem::create_directories(options.output_dir);

        capture_state capture;
        capture.output_dir = options.output_dir;
        capture.requested.insert(options.nodes.begin(), options.nodes.end());

        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(options.model_path.c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("unable to load model");
        }

        std::vector<llama_token> tokens;
        if (!options.tokens_path.empty()) {
            tokens = read_tokens(options.tokens_path);
        } else {
            const llama_vocab * vocab = llama_model_get_vocab(model);
            const int32_t size_query = llama_tokenize(
                vocab,
                options.prompt.c_str(),
                static_cast<int32_t>(options.prompt.size()),
                nullptr,
                0,
                options.add_special,
                true);
            if (size_query == std::numeric_limits<int32_t>::min() || size_query == 0) {
                llama_model_free(model);
                throw std::runtime_error("tokenization size query failed");
            }
            tokens.resize(static_cast<size_t>(size_query < 0 ? -size_query : size_query));
            const int32_t tokenized = llama_tokenize(
                vocab,
                options.prompt.c_str(),
                static_cast<int32_t>(options.prompt.size()),
                tokens.data(),
                static_cast<int32_t>(tokens.size()),
                options.add_special,
                true);
            if (tokenized < 0) {
                llama_model_free(model);
                throw std::runtime_error("tokenization failed");
            }
            tokens.resize(static_cast<size_t>(tokenized));
        }
        if (tokens.size() > static_cast<size_t>(options.context_size)) {
            llama_model_free(model);
            throw std::runtime_error("token count exceeds --ctx");
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(options.context_size);
        context_params.n_batch = static_cast<uint32_t>(tokens.size());
        context_params.n_ubatch = static_cast<uint32_t>(tokens.size());
        context_params.n_seq_max = 1;
        context_params.n_threads = options.threads;
        context_params.n_threads_batch = options.threads;
        context_params.no_perf = true;
        // The QNN PTE exposes QK, softmax, and AV as separate delegate
        // tensors. Disable the fused CPU path only for this diagnostic so
        // those corresponding llama graph nodes remain observable.
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context_params.cb_eval = eval_callback;
        context_params.cb_eval_user_data = &capture;
        llama_context * context = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            llama_model_free(model);
            throw std::runtime_error("unable to create context");
        }
        llama_set_warmup(context, false);

        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        for (size_t index = 0; index < tokens.size(); ++index) {
            batch.token[index] = tokens[index];
            batch.pos[index] = static_cast<llama_pos>(index);
            batch.n_seq_id[index] = 1;
            batch.seq_id[index][0] = 0;
            batch.logits[index] = 0;
        }

        capture.active = true;
        const int32_t decode_status = llama_decode(context, batch);
        capture.active = false;
        llama_batch_free(batch);

        write_tokens(capture.output_dir, tokens);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();

        if (decode_status != 0) {
            throw std::runtime_error("llama_decode failed: " + std::to_string(decode_status));
        }
        if (!capture.error.empty()) {
            throw std::runtime_error(capture.error);
        }
        if (capture.captured.size() != capture.requested.size()) {
            std::ostringstream missing;
            for (const std::string & name : capture.requested) {
                if (capture.captured.count(name) == 0) {
                    if (missing.tellp() > 0) {
                        missing << ',';
                    }
                    missing << name;
                }
            }
            throw std::runtime_error("requested graph nodes were not captured: " + missing.str());
        }
        std::cout << "LLAMA_QNN_INTERMEDIATE_DUMP tokens=" << tokens.size()
                  << " nodes=" << capture.captured.size()
                  << " out=" << capture.output_dir.string() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "llama-qnn-intermediate-dump: " << error.what() << '\n';
        return 1;
    }
}
