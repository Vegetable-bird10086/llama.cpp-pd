#include "llama.h"
#include "llama-ext.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct args {
    std::string model_path;
    std::string prompt;
    std::string tokens_path;
    std::string output_dir;
    std::vector<uint32_t> layers;
    int32_t context_size = 4096;
    int32_t threads = 4;
    bool add_special = true;
};

void usage(const char * program) {
    std::cerr << "usage: " << program
              << " --model MODEL.gguf (--prompt TEXT | --tokens FILE) --out DIRECTORY"
              << " [--ctx N] [--threads N] [--layers 0,2,4] [--no-special]\n";
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

std::vector<uint32_t> parse_layers(const std::string & value) {
    std::vector<uint32_t> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, static_cast<char>(44))) {
        size_t used = 0;
        const unsigned long parsed = std::stoul(item, &used, 10);
        if (item.empty() || used != item.size() ||
            parsed > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("invalid --layers");
        }
        result.push_back(static_cast<uint32_t>(parsed));
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

args parse_args(int argc, char ** argv) {
    args result;
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--no-special") {
            result.add_special = false;
            continue;
        }
        if (flag == "--help" || flag == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        if (flag != "--model" && flag != "--prompt" && flag != "--tokens" && flag != "--out" &&
            flag != "--ctx" && flag != "--threads" && flag != "--layers") {
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
            result.layers = parse_layers(value);
        }
    }
    if (result.model_path.empty() || result.output_dir.empty() ||
        (result.prompt.empty() == result.tokens_path.empty())) {
        throw std::runtime_error("--model, exactly one of --prompt/--tokens, and --out are required");
    }
    return result;
}

void write_tokens(const std::string & output_dir, const std::vector<llama_token> & tokens) {
    std::ofstream output(output_dir + "/llama_tokens.txt", std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("unable to write llama_tokens.txt");
    }
    for (llama_token token : tokens) {
        output << token << static_cast<char>(10);
    }
}

void write_layer(
        const std::string & output_dir,
        uint32_t layer,
        const float * values,
        size_t count,
        int32_t n_tokens,
        int32_t n_embd) {
    const std::string stem = output_dir + "/llama_layer_inp_" + std::to_string(layer);
    std::ofstream binary(stem + ".f32.bin", std::ios::binary | std::ios::trunc);
    if (!binary.is_open()) {
        throw std::runtime_error("unable to write layer binary");
    }
    binary.write(
        reinterpret_cast<const char *>(values),
        static_cast<std::streamsize>(count * sizeof(float)));
    binary.close();
    if (!binary) {
        throw std::runtime_error("short layer binary write");
    }
    std::ofstream metadata(stem + ".json", std::ios::trunc);
    if (!metadata.is_open()) {
        throw std::runtime_error("unable to write layer metadata");
    }
    metadata << "{\n"
             << "  \"layer_input\": " << layer << ",\n"
             << "  \"scalar_type\": \"Float\",\n"
             << "  \"element_size\": 4,\n"
             << "  \"nbytes\": " << count * sizeof(float) << ",\n"
             << "  \"sizes\": [" << n_tokens << ", " << n_embd << "]\n"
             << "}\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const args options = parse_args(argc, argv);
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
            const int32_t token_count = size_query < 0 ? -size_query : size_query;
            tokens.resize(static_cast<size_t>(token_count));
            const int32_t tokenized = llama_tokenize(
                vocab,
                options.prompt.c_str(),
                static_cast<int32_t>(options.prompt.size()),
                tokens.data(),
                token_count,
                options.add_special,
                true);
            if (tokenized != token_count) {
                llama_model_free(model);
                throw std::runtime_error("tokenization output size mismatch");
            }
        }
        const int32_t n_tokens = static_cast<int32_t>(tokens.size());
        if (n_tokens > options.context_size) {
            llama_model_free(model);
            throw std::runtime_error("prompt exceeds context size");
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(options.context_size);
        context_params.n_batch = static_cast<uint32_t>(n_tokens);
        context_params.n_ubatch = static_cast<uint32_t>(n_tokens);
        context_params.n_seq_max = 1;
        context_params.n_threads = options.threads;
        context_params.n_threads_batch = options.threads;
        context_params.no_perf = true;
        llama_context * context = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            llama_model_free(model);
            throw std::runtime_error("unable to initialize context");
        }

        const int32_t n_layers = llama_model_n_layer(model);
        const int32_t n_embd = llama_model_n_embd(model);
        std::vector<uint32_t> layers = options.layers;
        if (layers.empty()) {
            for (int32_t layer = 0; layer < n_layers; layer += 2) {
                layers.push_back(static_cast<uint32_t>(layer));
            }
        }
        for (uint32_t layer : layers) {
            if (layer >= static_cast<uint32_t>(n_layers)) {
                llama_free(context);
                llama_model_free(model);
                throw std::runtime_error("requested layer is outside the model");
            }
            llama_set_embeddings_layer_inp(context, layer, true);
        }

        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        batch.n_tokens = n_tokens;
        for (int32_t i = 0; i < n_tokens; ++i) {
            batch.token[i] = tokens[static_cast<size_t>(i)];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 0;
        }
        const int32_t decode_status = llama_decode(context, batch);
        llama_batch_free(batch);
        if (decode_status != 0) {
            llama_free(context);
            llama_model_free(model);
            throw std::runtime_error("llama_decode failed: " + std::to_string(decode_status));
        }

        const size_t values_per_layer = static_cast<size_t>(n_tokens) * n_embd;
        write_tokens(options.output_dir, tokens);
        for (uint32_t layer : layers) {
            const float * values = llama_get_embeddings_layer_inp(context, layer);
            if (values == nullptr) {
                llama_free(context);
                llama_model_free(model);
                throw std::runtime_error("requested layer input is unavailable");
            }
            write_layer(
                options.output_dir,
                layer,
                values,
                values_per_layer,
                n_tokens,
                n_embd);
        }
        std::cout << "LLAMA_LAYER_DUMP tokens=" << n_tokens
                  << " embd=" << n_embd
                  << " layers=" << layers.size()
                  << " out=" << options.output_dir << static_cast<char>(10);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "llama-qnn-layer-dump: " << error.what() << static_cast<char>(10);
        return 1;
    }
}
