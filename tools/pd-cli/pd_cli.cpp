#include "arg.h"
#include "common.h"
#include "pd_cli_inprocess.h"
#include "json.hpp"
#include "llama.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "llama-qnn-u16.h"
#include "log.h"
#include "sampling.h"

#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cctype>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>
#include <unordered_set>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

struct pd_args {
    std::string import_dir;
    int control_fd = -1;
    int memory_fd = -1;
    const uint8_t * memory_ptr = nullptr;
    size_t memory_size = 0;
    const uint64_t * prompt_tokens_ptr = nullptr;
    const uint16_t * kv_fp16_ptr = nullptr;
    size_t kv_fp16_values = 0;
    int32_t prompt_length = 0;
    int32_t num_layers = 0;
    int32_t num_kv_heads = 0;
    int32_t head_dim = 0;
    llama_token first_token = -1;
    bool first_token_is_prompt_tail = false;
    std::string ppl_tokens_path;
    std::string ppl_output_path;
    std::string op_profile_path;
    std::string disk_embedding_path;
    int32_t ppl_max_tokens = 0;
    bool import_ro = false;
    bool roundtrip_check = false;
    bool native_compare = false;
    bool native_first_token = false;
};

thread_local const llama_pd_inprocess_request * g_inprocess_request = nullptr;
thread_local llama_pd_inprocess_runtime * g_inprocess_runtime = nullptr;
thread_local bool g_inprocess_preparing = false;

constexpr uint32_t PD_RESIDENT_READY_MAGIC = 0x50445259U; // "PDRY"
constexpr uint32_t PD_RESIDENT_PREPARE_MAGIC = 0x50445052U; // "PDPR"
constexpr uint32_t PD_RESIDENT_REQUEST_MAGIC = 0x50444b56U; // "PDKV"
// v5 stores both K and V as [layer, head, token, dim].
constexpr uint32_t PD_RESIDENT_PROTOCOL_VERSION = 5;

struct pd_resident_ready {
    uint32_t magic = PD_RESIDENT_READY_MAGIC;
    uint32_t version = PD_RESIDENT_PROTOCOL_VERSION;
};

struct pd_resident_prepare {
    uint32_t magic = PD_RESIDENT_PREPARE_MAGIC;
    uint32_t version = PD_RESIDENT_PROTOCOL_VERSION;
};

struct pd_resident_request {
    uint32_t magic = PD_RESIDENT_REQUEST_MAGIC;
    uint32_t version = PD_RESIDENT_PROTOCOL_VERSION;
    uint64_t memory_size = 0;
    int32_t prompt_length = 0;
    int32_t num_layers = 0;
    int32_t num_kv_heads = 0;
    int32_t head_dim = 0;
    int32_t first_token = -1;
    uint32_t first_token_is_prompt_tail = 0;
};

void send_resident_ready(int fd) {
    const pd_resident_ready ready;
    ssize_t sent;
    do {
        sent = send(fd, &ready, sizeof(ready), MSG_NOSIGNAL);
    } while (sent < 0 && errno == EINTR);
    if (sent != static_cast<ssize_t>(sizeof(ready))) {
        throw std::runtime_error("unable to signal resident Decode readiness");
    }
}

void receive_resident_prepare(int fd) {
    pd_resident_prepare request;
    ssize_t received;
    do {
        received = recv(fd, &request, sizeof(request), 0);
    } while (received < 0 && errno == EINTR);
    if (received != static_cast<ssize_t>(sizeof(request)) ||
        request.magic != PD_RESIDENT_PREPARE_MAGIC ||
        request.version != PD_RESIDENT_PROTOCOL_VERSION) {
        throw std::runtime_error("invalid resident Decode prepare request");
    }
}

void receive_resident_handoff(pd_args & args) {
    pd_resident_request request;
    iovec iov {};
    iov.iov_base = &request;
    iov.iov_len = sizeof(request);
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))] = {};
    msghdr message {};
    message.msg_iov = &iov;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);

    ssize_t received;
    do {
        received = recvmsg(args.control_fd, &message, 0);
    } while (received < 0 && errno == EINTR);
    if (received != static_cast<ssize_t>(sizeof(request)) ||
        request.magic != PD_RESIDENT_REQUEST_MAGIC ||
        request.version != PD_RESIDENT_PROTOCOL_VERSION) {
        throw std::runtime_error("invalid resident Decode handoff request");
    }

    int memory_fd = -1;
    for (cmsghdr * cmsg = CMSG_FIRSTHDR(&message);
            cmsg != nullptr;
            cmsg = CMSG_NXTHDR(&message, cmsg)) {
        if (cmsg->cmsg_level == SOL_SOCKET &&
            cmsg->cmsg_type == SCM_RIGHTS &&
            cmsg->cmsg_len >= CMSG_LEN(sizeof(int))) {
            std::memcpy(&memory_fd, CMSG_DATA(cmsg), sizeof(memory_fd));
            break;
        }
    }
    if (memory_fd < 0) {
        throw std::runtime_error("resident Decode request did not contain a KV descriptor");
    }

    args.memory_fd = memory_fd;
    args.memory_size = static_cast<size_t>(request.memory_size);
    args.prompt_length = request.prompt_length;
    args.num_layers = request.num_layers;
    args.num_kv_heads = request.num_kv_heads;
    args.head_dim = request.head_dim;
    args.first_token = static_cast<llama_token>(request.first_token);
    args.first_token_is_prompt_tail = request.first_token_is_prompt_tail != 0;
    close(args.control_fd);
    args.control_fd = -1;
}

class pd_disk_embedding {
public:
    pd_disk_embedding() = default;
    pd_disk_embedding(const pd_disk_embedding &) = delete;
    pd_disk_embedding & operator=(const pd_disk_embedding &) = delete;

    ~pd_disk_embedding() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    void open_file(const std::string & path) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::runtime_error("unable to open disk embedding: " + path);
        }

        uint8_t header[20] = {};
        read_exact(0, header, sizeof(header));
        if (std::memcmp(header, "SEMB", 4) != 0 ||
            read_u32(header + 4) != 1 ||
            read_u32(header + 8) != 0) {
            throw std::runtime_error(
                "disk embedding must be an unquantized SEMB v1 file");
        }

        dtype_ = read_u32(header + 12);
        const uint32_t ndim = read_u32(header + 16);
        if ((dtype_ != 1 && dtype_ != 2) || ndim != 2) {
            throw std::runtime_error(
                "disk embedding must be a rank-2 FP16 or FP32 tensor");
        }

        uint8_t descriptor[16] = {};
        read_exact(sizeof(header), descriptor, sizeof(descriptor));
        payload_bytes_ = read_u64(descriptor);
        vocab_size_ = read_u32(descriptor + 8);
        embedding_dim_ = read_u32(descriptor + 12);
        data_offset_ = sizeof(header) + sizeof(descriptor);
        const size_t element_bytes = dtype_ == 2 ? sizeof(ggml_fp16_t) : sizeof(float);
        row_bytes_ = static_cast<size_t>(embedding_dim_) * element_bytes;
        if (vocab_size_ == 0 || embedding_dim_ == 0 ||
            payload_bytes_ != static_cast<uint64_t>(vocab_size_) * row_bytes_) {
            throw std::runtime_error("disk embedding shape/payload mismatch");
        }

        struct stat info {};
        if (fstat(fd_, &info) != 0 || info.st_size < 0 ||
            static_cast<uint64_t>(info.st_size) < data_offset_ + payload_bytes_) {
            throw std::runtime_error("disk embedding file is truncated");
        }
        fp32_row_.resize(embedding_dim_);
        if (dtype_ == 2) {
            fp16_row_.resize(embedding_dim_);
        }
        LOG_INF(
            "PD disk embedding opened: path=%s vocab=%u dim=%u dtype=%s "
            "table_bytes=%" PRIu64 " mode=pread-row\n",
            path.c_str(),
            vocab_size_,
            embedding_dim_,
            dtype_ == 2 ? "fp16" : "fp32",
            payload_bytes_);
    }

    bool is_open() const {
        return fd_ >= 0;
    }

    uint32_t vocab_size() const {
        return vocab_size_;
    }

    uint32_t embedding_dim() const {
        return embedding_dim_;
    }

    float * read_row(llama_token token) {
        if (token < 0 || static_cast<uint32_t>(token) >= vocab_size_) {
            throw std::runtime_error("disk embedding token is out of range");
        }
        const uint64_t offset =
            data_offset_ + static_cast<uint64_t>(token) * row_bytes_;
        if (dtype_ == 2) {
            read_exact(offset, fp16_row_.data(), row_bytes_);
            ggml_fp16_to_fp32_row(
                fp16_row_.data(), fp32_row_.data(), embedding_dim_);
        } else {
            read_exact(offset, fp32_row_.data(), row_bytes_);
        }
        return fp32_row_.data();
    }

private:
    static uint32_t read_u32(const uint8_t * data) {
        uint32_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

    static uint64_t read_u64(const uint8_t * data) {
        uint64_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

    void read_exact(uint64_t offset, void * destination, size_t bytes) const {
        auto * output = static_cast<uint8_t *>(destination);
        size_t done = 0;
        while (done < bytes) {
            const ssize_t result = pread(
                fd_,
                output + done,
                bytes - done,
                static_cast<off_t>(offset + done));
            if (result <= 0) {
                throw std::runtime_error("disk embedding pread failed");
            }
            done += static_cast<size_t>(result);
        }
    }

    int fd_ = -1;
    uint32_t dtype_ = 0;
    uint32_t vocab_size_ = 0;
    uint32_t embedding_dim_ = 0;
    uint64_t data_offset_ = 0;
    uint64_t payload_bytes_ = 0;
    size_t row_bytes_ = 0;
    std::vector<ggml_fp16_t> fp16_row_;
    std::vector<float> fp32_row_;
};

class pd_persistent_threadpools {
public:
    pd_persistent_threadpools() = default;
    pd_persistent_threadpools(const pd_persistent_threadpools &) = delete;
    pd_persistent_threadpools & operator=(const pd_persistent_threadpools &) = delete;

    ~pd_persistent_threadpools() {
        if (ctx_ != nullptr) {
            llama_detach_threadpool(ctx_);
        }
        if (free_fn_ != nullptr) {
            if (threadpool_ != nullptr) {
                free_fn_(threadpool_);
            }
            if (threadpool_batch_ != nullptr) {
                free_fn_(threadpool_batch_);
            }
        }
    }

    void attach(llama_context * ctx, const common_params & params) {
        ggml_backend_dev_t cpu_dev =
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error(
                "unable to find CPU backend for persistent threadpool");
        }
        ggml_backend_reg_t cpu_reg =
            ggml_backend_dev_backend_reg(cpu_dev);
        new_fn_ = reinterpret_cast<decltype(new_fn_)>(
            ggml_backend_reg_get_proc_address(
                cpu_reg, "ggml_threadpool_new"));
        free_fn_ = reinterpret_cast<decltype(free_fn_)>(
            ggml_backend_reg_get_proc_address(
                cpu_reg, "ggml_threadpool_free"));
        if (new_fn_ == nullptr || free_fn_ == nullptr) {
            throw std::runtime_error(
                "CPU backend does not expose threadpool functions");
        }

        ggml_threadpool_params tpp =
            ggml_threadpool_params_from_cpu_params(params.cpuparams);
        ggml_threadpool_params tpp_batch =
            ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
        if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
            threadpool_batch_ = new_fn_(&tpp_batch);
            if (threadpool_batch_ == nullptr) {
                throw std::runtime_error(
                    "unable to create persistent batch threadpool");
            }
            // Only one of the pools is active at a time. Match the common
            // completion CLI and keep the generation pool paused until use.
            tpp.paused = true;
        }

        threadpool_ = new_fn_(&tpp);
        if (threadpool_ == nullptr) {
            throw std::runtime_error(
                "unable to create persistent generation threadpool");
        }

        llama_attach_threadpool(ctx, threadpool_, threadpool_batch_);
        ctx_ = ctx;
        LOG_INF(
            "PD persistent CPU threadpool attached: generation_threads=%d "
            "batch_threads=%d separate_batch_pool=%d\n",
            tpp.n_threads,
            tpp_batch.n_threads,
            threadpool_batch_ != nullptr ? 1 : 0);
    }

private:
    using new_fn_type = decltype(ggml_threadpool_new) *;
    using free_fn_type = decltype(ggml_threadpool_free) *;

    llama_context * ctx_ = nullptr;
    ggml_threadpool * threadpool_ = nullptr;
    ggml_threadpool * threadpool_batch_ = nullptr;
    new_fn_type new_fn_ = nullptr;
    free_fn_type free_fn_ = nullptr;
};

struct pd_capture_state {
    std::filesystem::path output_dir;
    std::unordered_set<std::string> requested;
    std::unordered_set<std::string> captured;
    std::string error;
    bool active = false;
    bool completed = false;
};

static thread_local pd_capture_state * g_pd_capture_state = nullptr;

static const char * const pd_default_capture_nodes[] = {
    "qnn_u16_input",
    "qnn_attn_norm-0",
    "qnn_q_projection-0",
    "qnn_k_projection-0",
    "qnn_v_projection-0",
    "qnn_q_heads_norm-0",
    "qnn_q_heads_rope-0",
    "qnn_q_heads_rotate-0",
    "qnn_k_heads_norm-0",
    "qnn_k_heads_rope-0",
    "qnn_k_heads_rotate-0",
    "qnn_k_heads_u8-0",
    "qnn_v_heads_u8-0",
    "qnn_attention-0",
    "qnn_attention_concat-0",
    "qnn_attention_output-0",
    "qnn_ffn_input-0",
    "qnn_ffn_norm-0",
    "qnn_ffn_gate-0",
    "qnn_ffn_up-0",
    "qnn_ffn_swiglu-0",
    "qnn_ffn_down-0",
    "qnn_layer_output-0",
    "qnn_attn_norm-1",
    "qnn_q_projection-1",
    "qnn_k_projection-1",
    "qnn_v_projection-1",
    "qnn_q_heads_norm-1",
    "qnn_q_heads_rope-1",
    "qnn_q_heads_rotate-1",
    "qnn_k_heads_norm-1",
    "qnn_k_heads_rope-1",
    "qnn_k_heads_rotate-1",
    "qnn_k_heads_u8-1",
    "qnn_v_heads_u8-1",
    "qnn_attention-1",
    "qnn_attention_concat-1",
    "qnn_attention_output-1",
    "qnn_ffn_input-1",
    "qnn_ffn_norm-1",
    "qnn_ffn_gate-1",
    "qnn_ffn_up-1",
    "qnn_ffn_swiglu-1",
    "qnn_ffn_down-1",
    "qnn_layer_output-1",
};

static void configure_pd_capture(pd_capture_state & state, const char * output_dir) {
    state.output_dir = output_dir;
    std::filesystem::create_directories(state.output_dir);
    state.requested.insert(
        std::begin(pd_default_capture_nodes),
        std::end(pd_default_capture_nodes));
    for (int32_t layer = 0; layer < 36; ++layer) {
        const std::string suffix = "-" + std::to_string(layer);
        state.requested.insert("qnn_attn_norm" + suffix);
        state.requested.insert("qnn_attention_output" + suffix);
        state.requested.insert("qnn_ffn_down" + suffix);
        state.requested.insert("qnn_layer_output" + suffix);
    }
    const std::vector<std::string> detailed_stages = {
        "qnn_q_projection",
        "qnn_k_projection",
        "qnn_v_projection",
        "qnn_q_heads_norm",
        "qnn_q_heads_rope",
        "qnn_q_heads_rotate",
        "qnn_k_heads_norm",
        "qnn_k_heads_rope",
        "qnn_k_heads_rotate",
        "qnn_k_heads_u8",
        "qnn_v_heads_u8",
        "qnn_attention",
        "qnn_attention_concat",
        "qnn_ffn_input",
        "qnn_ffn_norm",
        "qnn_ffn_gate",
        "qnn_ffn_up",
        "qnn_ffn_swiglu",
    };
    for (int32_t layer : {5, 6}) {
        const std::string suffix = "-" + std::to_string(layer);
        for (const std::string & stage : detailed_stages) {
            state.requested.insert(stage + suffix);
        }
    }
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
        case GGML_TYPE_U8:  suffix = ".u8.bin";  break;
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
    pd_capture_state * state_ptr = user_data != nullptr
        ? static_cast<pd_capture_state *>(user_data)
        : g_pd_capture_state;
    if (state_ptr == nullptr) {
        return false;
    }
    auto & state = *state_ptr;
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

struct pd_op_profile_event {
    int32_t decode_index = -1;
    llama_token token = -1;
    int64_t node_index = -1;
    int32_t layer = -1;
    std::string name;
    std::string semantic_name;
    std::string op;
    double duration_us = 0.0;
    int64_t ne[GGML_MAX_DIMS] = {};
};

static std::optional<int32_t> layer_from_node_name(const std::string & name) {
    const size_t separator = name.rfind('-');
    if (separator == std::string::npos || separator + 1 >= name.size()) {
        return std::nullopt;
    }
    size_t end = separator + 1;
    if (name[end] == '-') {
        ++end;
    }
    if (end >= name.size() ||
        !std::all_of(name.begin() + end, name.end(), [](unsigned char value) {
            return std::isdigit(value);
        })) {
        return std::nullopt;
    }
    return std::stoi(name.substr(separator + 1));
}

static std::string semantic_node_name(const std::string & name) {
    if (name.empty()) {
        return "<unnamed>";
    }
    std::string result = name;
    if (layer_from_node_name(result).has_value()) {
        result.resize(result.rfind('-'));
    }
    const size_t head = result.rfind("_h");
    if (head != std::string::npos && head + 2 < result.size() &&
        std::all_of(result.begin() + head + 2, result.end(), [](unsigned char value) {
            return std::isdigit(value);
        })) {
        result.resize(head);
    }
    return result;
}

static std::string csv_field(const std::string & value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        return value;
    }
    std::string result = "\"";
    for (char character : value) {
        if (character == '"') {
            result += '"';
        }
        result += character;
    }
    result += '"';
    return result;
}

struct pd_op_profile_state {
    std::filesystem::path output_path;
    std::vector<pd_op_profile_event> events;
    steady_clock::time_point node_start;
    ggml_tensor * pending = nullptr;
    int32_t decode_index = -1;
    llama_token token = -1;
    int64_t node_index = 0;
    int32_t inferred_layer = -1;
    bool layer_granularity = false;
    bool active = false;

    ~pd_op_profile_state() {
        if (output_path.empty()) {
            return;
        }
        try {
            write();
        } catch (const std::exception & error) {
            LOG_ERR("failed to write PD operator profile: %s\n", error.what());
        }
    }

    void begin_decode(llama_token input_token) {
        ++decode_index;
        token = input_token;
        node_index = 0;
        inferred_layer = -1;
        active = true;
    }

    void end_decode() {
        active = false;
        pending = nullptr;
    }

    void write() const {
        if (!output_path.parent_path().empty()) {
            std::filesystem::create_directories(output_path.parent_path());
        }
        std::ofstream raw(output_path, std::ios::trunc);
        if (!raw.is_open()) {
            throw std::runtime_error("unable to open " + output_path.string());
        }
        raw << "decode_index,token,node_index,layer,name,semantic_name,op,"
               "duration_us,ne0,ne1,ne2,ne3\n";
        for (const auto & event : events) {
            raw << event.decode_index << ',' << event.token << ','
                << event.node_index << ',' << event.layer << ','
                << csv_field(event.name) << ','
                << csv_field(event.semantic_name) << ','
                << csv_field(event.op) << ',' << event.duration_us;
            for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
                raw << ',' << event.ne[dimension];
            }
            raw << '\n';
        }

        struct aggregate {
            int64_t count = 0;
            double total_us = 0.0;
            double min_us = std::numeric_limits<double>::infinity();
            double max_us = 0.0;
        };
        using key = std::tuple<int32_t, std::string, std::string>;
        std::map<key, aggregate> aggregates;
        for (const auto & event : events) {
            auto & value = aggregates[{event.layer, event.semantic_name, event.op}];
            ++value.count;
            value.total_us += event.duration_us;
            value.min_us = std::min(value.min_us, event.duration_us);
            value.max_us = std::max(value.max_us, event.duration_us);
        }

        std::filesystem::path summary_path = output_path;
        summary_path += ".summary.csv";
        std::ofstream summary(summary_path, std::ios::trunc);
        if (!summary.is_open()) {
            throw std::runtime_error("unable to open " + summary_path.string());
        }
        summary << "layer,semantic_name,op,count,total_us,mean_us,min_us,max_us\n";
        for (const auto & [aggregate_key, value] : aggregates) {
            const auto & [layer, name, op] = aggregate_key;
            summary << layer << ',' << csv_field(name) << ',' << csv_field(op)
                    << ',' << value.count << ',' << value.total_us << ','
                    << value.total_us / value.count << ',' << value.min_us
                    << ',' << value.max_us << '\n';
        }
    }
};

static thread_local pd_op_profile_state * g_pd_op_profile_state = nullptr;

static bool pd_op_profile_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & state = *static_cast<pd_op_profile_state *>(user_data);
    if (!state.active) {
        return false;
    }
    if (ask) {
        if (state.layer_granularity) {
            const char * raw_name = ggml_get_name(tensor);
            const std::string name = raw_name == nullptr ? "" : raw_name;
            const bool is_layer_boundary =
                name.rfind("qnn_layer_output-", 0) == 0;
            if (name != "qnn_u16_input" &&
                name != "result_output" &&
                !is_layer_boundary) {
                return false;
            }
        }
        state.pending = tensor;
        state.node_start = steady_clock::now();
        return true;
    }
    if (state.pending != tensor) {
        return false;
    }

    pd_op_profile_event event;
    event.decode_index = state.decode_index;
    event.token = state.token;
    event.node_index = state.node_index++;
    const char * raw_name = ggml_get_name(tensor);
    event.name = raw_name == nullptr ? "" : raw_name;
    event.semantic_name = semantic_node_name(event.name);
    if (state.layer_granularity) {
        if (event.name == "qnn_u16_input") {
            event.semantic_name = "input";
        } else if (event.name == "result_output") {
            event.layer = -1;
            event.semantic_name = "output_head";
        } else {
            event.layer = layer_from_node_name(event.name).value_or(-1);
            event.semantic_name = "layer_total";
        }
        event.op = "GRAPH_SEGMENT";
    } else {
        if (const auto explicit_layer = layer_from_node_name(event.name);
                explicit_layer.has_value()) {
            event.layer = *explicit_layer;
            if (event.name.rfind("qnn_", 0) == 0) {
                state.inferred_layer = *explicit_layer;
            }
        } else {
            event.layer = state.inferred_layer;
        }
        event.op = ggml_op_name(tensor->op);
    }
    event.duration_us =
        std::chrono::duration<double, std::micro>(
            steady_clock::now() - state.node_start).count();
    std::copy(std::begin(tensor->ne), std::end(tensor->ne), event.ne);
    state.events.push_back(std::move(event));
    state.pending = nullptr;
    return true;
}

// A resident in-process context is created before an individual PD request has
// its capture/profile state. Keep one callback installed on that context and
// dispatch to the request-local diagnostic state selected around llama_decode.
// Both pointers are null in the normal production path.
static bool pd_resident_eval_callback(ggml_tensor * tensor, bool ask, void *) {
    if (g_pd_capture_state != nullptr) {
        return pd_capture_callback(tensor, ask, nullptr);
    }
    if (g_pd_op_profile_state != nullptr) {
        return pd_op_profile_callback(tensor, ask, g_pd_op_profile_state);
    }
    return false;
}

struct process_memory_snapshot {
    uint64_t rss_bytes = 0;
    uint64_t hwm_bytes = 0;
    uint64_t rss_anon_bytes = 0;
    uint64_t rss_file_bytes = 0;
    uint64_t rss_shmem_bytes = 0;
    uint64_t data_bytes = 0;
    uint64_t swap_bytes = 0;
};

struct mapping_memory_snapshot {
    uint64_t rss_bytes = 0;
    uint64_t pss_bytes = 0;
    uint64_t private_clean_bytes = 0;
    uint64_t private_dirty_bytes = 0;
    uint64_t shared_clean_bytes = 0;
    uint64_t shared_dirty_bytes = 0;
    uint64_t anonymous_bytes = 0;
    uint64_t swap_bytes = 0;
};

using process_mapping_snapshot =
    std::map<std::string, mapping_memory_snapshot>;

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
        read_proc_status_bytes("RssAnon:"),
        read_proc_status_bytes("RssFile:"),
        read_proc_status_bytes("RssShmem:"),
        read_proc_status_bytes("VmData:"),
        read_proc_status_bytes("VmSwap:"),
    };
}

static double bytes_to_mib(uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static bool warmup_memory_profile_enabled() {
    const char * value = std::getenv("LLAMA_PD_WARMUP_MEMORY_PROFILE");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

static std::string trim_left(std::string value) {
    const size_t first = value.find_first_not_of(" \t");
    return first == std::string::npos ? std::string() : value.substr(first);
}

static process_mapping_snapshot process_mappings() {
    std::ifstream smaps("/proc/self/smaps");
    process_mapping_snapshot result;
    std::string current_mapping;
    std::string line;
    while (std::getline(smaps, line)) {
        const size_t dash = line.find('-');
        const bool mapping_header =
            dash != std::string::npos && dash > 0 && dash < 32 &&
            std::all_of(line.begin(), line.begin() + dash, [](unsigned char c) {
                return std::isxdigit(c) != 0;
            });
        if (mapping_header) {
            std::istringstream header(line);
            std::string address;
            std::string permissions;
            std::string offset;
            std::string device;
            std::string inode;
            header >> address >> permissions >> offset >> device >> inode;
            std::getline(header, current_mapping);
            current_mapping = trim_left(std::move(current_mapping));
            if (current_mapping.empty()) {
                current_mapping = "[anonymous]";
            }
            continue;
        }
        if (current_mapping.empty()) {
            continue;
        }
        const auto read_kib = [&](const char * field, uint64_t * destination) {
            if (line.rfind(field, 0) != 0) {
                return false;
            }
            std::istringstream value(line.substr(std::strlen(field)));
            uint64_t kib = 0;
            value >> kib;
            *destination += kib * 1024;
            return true;
        };
        mapping_memory_snapshot & memory = result[current_mapping];
        if (read_kib("Rss:", &memory.rss_bytes) ||
                read_kib("Pss:", &memory.pss_bytes) ||
                read_kib("Private_Clean:", &memory.private_clean_bytes) ||
                read_kib("Private_Dirty:", &memory.private_dirty_bytes) ||
                read_kib("Shared_Clean:", &memory.shared_clean_bytes) ||
                read_kib("Shared_Dirty:", &memory.shared_dirty_bytes) ||
                read_kib("Anonymous:", &memory.anonymous_bytes) ||
                read_kib("Swap:", &memory.swap_bytes)) {
            continue;
        }
    }
    return result;
}

static int64_t mapping_delta(
        uint64_t after,
        uint64_t before) {
    return after >= before
        ? static_cast<int64_t>(after - before)
        : -static_cast<int64_t>(before - after);
}

static double signed_bytes_to_mib(int64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static void log_mapping_delta(
        const char * phase,
        const process_mapping_snapshot & before,
        const process_mapping_snapshot & after) {
    struct mapping_delta_entry {
        std::string name;
        int64_t rss_bytes = 0;
        int64_t pss_bytes = 0;
        int64_t private_dirty_bytes = 0;
        int64_t anonymous_bytes = 0;
        int64_t swap_bytes = 0;
    };
    std::vector<mapping_delta_entry> deltas;
    deltas.reserve(before.size() + after.size());
    std::unordered_set<std::string> names;
    for (const auto & entry : before) {
        names.insert(entry.first);
    }
    for (const auto & entry : after) {
        names.insert(entry.first);
    }
    const mapping_memory_snapshot empty;
    for (const std::string & name : names) {
        const auto before_it = before.find(name);
        const auto after_it = after.find(name);
        const mapping_memory_snapshot & lhs =
            before_it == before.end() ? empty : before_it->second;
        const mapping_memory_snapshot & rhs =
            after_it == after.end() ? empty : after_it->second;
        mapping_delta_entry delta {
            name,
            mapping_delta(rhs.rss_bytes, lhs.rss_bytes),
            mapping_delta(rhs.pss_bytes, lhs.pss_bytes),
            mapping_delta(rhs.private_dirty_bytes, lhs.private_dirty_bytes),
            mapping_delta(rhs.anonymous_bytes, lhs.anonymous_bytes),
            mapping_delta(rhs.swap_bytes, lhs.swap_bytes),
        };
        if (delta.rss_bytes != 0 || delta.swap_bytes != 0) {
            deltas.push_back(std::move(delta));
        }
    }
    std::sort(
        deltas.begin(), deltas.end(),
        [](const mapping_delta_entry & lhs, const mapping_delta_entry & rhs) {
            return std::llabs(lhs.rss_bytes) > std::llabs(rhs.rss_bytes);
        });
    const size_t count = std::min<size_t>(deltas.size(), 16);
    for (size_t i = 0; i < count; ++i) {
        const auto & delta = deltas[i];
        LOG_INF(
            "PD %s mapping delta: rank=%zu rss_mib=%+.2f pss_mib=%+.2f "
            "private_dirty_mib=%+.2f anonymous_mib=%+.2f swap_mib=%+.2f "
            "mapping=%s\n",
            phase,
            i,
            signed_bytes_to_mib(delta.rss_bytes),
            signed_bytes_to_mib(delta.pss_bytes),
            signed_bytes_to_mib(delta.private_dirty_bytes),
            signed_bytes_to_mib(delta.anonymous_bytes),
            signed_bytes_to_mib(delta.swap_bytes),
            delta.name.c_str());
    }
}

static void log_warmup_memory_stage(
        const char * stage,
        const process_memory_snapshot & memory) {
    LOG_INF(
        "PD warmup memory: stage=%s rss_mib=%.2f hwm_mib=%.2f "
        "rss_anon_mib=%.2f rss_file_mib=%.2f rss_shmem_mib=%.2f "
        "vm_data_mib=%.2f swap_mib=%.2f\n",
        stage,
        bytes_to_mib(memory.rss_bytes),
        bytes_to_mib(memory.hwm_bytes),
        bytes_to_mib(memory.rss_anon_bytes),
        bytes_to_mib(memory.rss_file_bytes),
        bytes_to_mib(memory.rss_shmem_bytes),
        bytes_to_mib(memory.data_bytes),
        bytes_to_mib(memory.swap_bytes));
}

struct pd_handoff {
    json manifest;
    std::vector<llama_token> prompt_tokens;
    llama_token first_token = -1;
    bool first_token_is_prompt_tail = false;
    std::vector<uint16_t> kv_fp16;
    const uint16_t * kv_fp16_view = nullptr;
    size_t kv_fp16_view_size = 0;
    std::vector<uint8_t> kv_qnn_u8;
    std::shared_ptr<void> memory_mapping;
    const uint8_t * kv_qnn_u8_view = nullptr;
    size_t kv_qnn_u8_view_size = 0;
    int32_t prompt_len = 0;
    int32_t num_layers = 0;
    int32_t num_kv_heads = 0;
    int32_t head_dim = 0;
    double metadata_read_ms = 0.0;
    double kv_read_ms = 0.0;

    const uint8_t * qnn_u8_data() const {
        return kv_qnn_u8_view != nullptr ? kv_qnn_u8_view : kv_qnn_u8.data();
    }

    size_t qnn_u8_size() const {
        return kv_qnn_u8_view != nullptr ? kv_qnn_u8_view_size : kv_qnn_u8.size();
    }

    const uint16_t * fp16_data() const {
        return kv_fp16_view != nullptr ? kv_fp16_view : kv_fp16.data();
    }

    size_t fp16_size() const {
        return kv_fp16_view != nullptr ? kv_fp16_view_size : kv_fp16.size();
    }
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
        if (arg == "--pd-control-fd") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--pd-control-fd requires a descriptor");
            }
            out.control_fd = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--pd-memory-fd") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--pd-memory-fd requires a descriptor");
            }
            out.memory_fd = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--pd-memory-size") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--pd-memory-size requires a byte count");
            }
            out.memory_size = static_cast<size_t>(std::stoull(argv[++i]));
            continue;
        }
        if (arg == "--pd-prompt-length") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--pd-prompt-length requires a count");
            }
            out.prompt_length = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--pd-num-layers") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--pd-num-layers requires a count");
            }
            out.num_layers = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--pd-num-kv-heads") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--pd-num-kv-heads requires a count");
            }
            out.num_kv_heads = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--pd-head-dim") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--pd-head-dim requires a count");
            }
            out.head_dim = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--pd-first-token") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--pd-first-token requires a token");
            }
            out.first_token = static_cast<llama_token>(std::stoi(argv[++i]));
            continue;
        }
        if (arg == "--pd-first-token-is-prompt-tail") {
            if (i + 1 >= argc) {
                throw std::runtime_error(
                    "--pd-first-token-is-prompt-tail requires 0 or 1");
            }
            out.first_token_is_prompt_tail = std::stoi(argv[++i]) != 0;
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
        if (arg == "--pd-op-profile") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--pd-op-profile requires a CSV path");
            }
            out.op_profile_path = argv[++i];
            continue;
        }
        if (arg == "--pd-disk-embedding") {
            if (i + 1 >= argc) {
                throw std::runtime_error(
                    "--pd-disk-embedding requires a SEMB file");
            }
            out.disk_embedding_path = argv[++i];
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
    const int handoff_sources =
        static_cast<int>(!out.import_dir.empty()) +
        static_cast<int>(out.memory_fd >= 0) +
        static_cast<int>(out.control_fd >= 0) +
        static_cast<int>(
            g_inprocess_request != nullptr || g_inprocess_preparing);
    if (handoff_sources != 1) {
        throw std::runtime_error(
            "provide exactly one of --pd-import DIR, --pd-memory-fd FD, "
            "--pd-control-fd FD, or an in-process handoff");
    }
    if (out.memory_fd >= 0 &&
        (out.memory_size == 0 || out.prompt_length <= 0 ||
         out.num_layers <= 0 || out.num_kv_heads <= 0 || out.head_dim <= 0 ||
         out.first_token < 0)) {
        throw std::runtime_error("incomplete PD memory handoff metadata");
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

pd_handoff load_pd_handoff(
        const std::string & import_dir,
        bool load_fp16_kv,
        bool load_qnn_u8_kv) {
    pd_handoff out;
    const auto metadata_read_start = steady_clock::now();
    out.manifest = read_json_file(import_dir + "/manifest.json");
    out.prompt_tokens = load_prompt_tokens(import_dir + "/prompt_tokens.bin");
    out.first_token = load_first_token(out.manifest, import_dir);
    out.first_token_is_prompt_tail =
        out.manifest.value("first_token_is_prompt_tail", false);
    out.metadata_read_ms = elapsed_ms(metadata_read_start);

    const auto kv_read_start = steady_clock::now();
    const std::string qnn_u8_kv_file = out.manifest.value("qnn_u8_kv_file", "");
    if (load_qnn_u8_kv && !qnn_u8_kv_file.empty()) {
        out.kv_qnn_u8 = read_binary_vector<uint8_t>(import_dir + "/" + qnn_u8_kv_file);
    }
    // Prefer the backend-native QNN U8 handoff when present. Otherwise retain
    // canonical FP16 so a non-QNN Prefill backend can be bridged into the same
    // profiled Decode domains without carrying both copies at peak memory.
    if (load_fp16_kv && (!load_qnn_u8_kv || out.kv_qnn_u8.empty())) {
        out.kv_fp16 = read_binary_vector<uint16_t>(import_dir + "/kv.bin");
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
    if (load_fp16_kv && out.kv_qnn_u8.empty() &&
            out.kv_fp16.size() != expected_values) {
        std::ostringstream oss;
        oss << "kv.bin element count mismatch: got=" << out.kv_fp16.size()
            << " expected=" << expected_values;
        throw std::runtime_error(oss.str());
    }
    if (load_qnn_u8_kv && !out.kv_qnn_u8.empty() &&
            out.kv_qnn_u8.size() != expected_values) {
        std::ostringstream oss;
        oss << "QNN U8 kv file element count mismatch: got=" << out.kv_qnn_u8.size()
            << " expected=" << expected_values;
        throw std::runtime_error(oss.str());
    }

    return out;
}

pd_handoff load_pd_memory_handoff(const pd_args & args) {
    const auto metadata_start = steady_clock::now();
    const bool split_fp16 = args.prompt_tokens_ptr != nullptr ||
        args.kv_fp16_ptr != nullptr || args.kv_fp16_values != 0;
    if (split_fp16) {
        if (args.prompt_tokens_ptr == nullptr || args.kv_fp16_ptr == nullptr) {
            throw std::runtime_error("incomplete split FP16 PD memory handoff");
        }
        const size_t expected_values =
            static_cast<size_t>(args.num_layers) * args.num_kv_heads *
            args.prompt_length * args.head_dim * 2;
        if (args.kv_fp16_values != expected_values) {
            throw std::runtime_error("split FP16 PD memory handoff size mismatch");
        }
        pd_handoff out;
        out.prompt_tokens.reserve(static_cast<size_t>(args.prompt_length));
        for (int32_t i = 0; i < args.prompt_length; ++i) {
            out.prompt_tokens.push_back(static_cast<llama_token>(
                args.prompt_tokens_ptr[static_cast<size_t>(i)]));
        }
        out.kv_fp16_view = args.kv_fp16_ptr;
        out.kv_fp16_view_size = args.kv_fp16_values;
        out.prompt_len = args.prompt_length;
        out.num_layers = args.num_layers;
        out.num_kv_heads = args.num_kv_heads;
        out.head_dim = args.head_dim;
        out.first_token = args.first_token;
        out.first_token_is_prompt_tail = args.first_token_is_prompt_tail;
        out.manifest["original_prompt_length"] =
            out.prompt_len + (out.first_token_is_prompt_tail ? 1 : 0);
        out.metadata_read_ms = elapsed_ms(metadata_start);
        return out;
    }
    if (args.memory_ptr == nullptr) {
        struct stat info {};
        if (fstat(args.memory_fd, &info) != 0) {
            throw std::runtime_error("unable to stat PD memory handoff");
        }
        if (info.st_size < 0 ||
            static_cast<uint64_t>(info.st_size) != args.memory_size) {
            throw std::runtime_error("PD memory handoff descriptor size mismatch");
        }
    }

    const size_t prompt_bytes =
        static_cast<size_t>(args.prompt_length) * sizeof(uint64_t);
    const size_t per_kind_values =
        static_cast<size_t>(args.num_layers) * args.num_kv_heads *
        args.prompt_length * args.head_dim;
    const size_t kv_bytes = per_kind_values * 2;
    if (prompt_bytes > args.memory_size ||
        kv_bytes != args.memory_size - prompt_bytes) {
        throw std::runtime_error("PD memory handoff payload size mismatch");
    }

    const uint8_t * bytes = args.memory_ptr;
    void * mapping = MAP_FAILED;
    if (bytes == nullptr) {
        mapping = mmap(
            nullptr,
            args.memory_size,
            PROT_READ,
            MAP_SHARED,
            args.memory_fd,
            0);
        if (mapping == MAP_FAILED) {
            throw std::runtime_error("unable to map PD memory handoff");
        }
        close(args.memory_fd);
        bytes = static_cast<const uint8_t *>(mapping);
    }

    pd_handoff out;
    if (mapping != MAP_FAILED) {
        out.memory_mapping = std::shared_ptr<void>(
            mapping,
            [size = args.memory_size](void * ptr) {
                munmap(ptr, size);
            });
    }
    const uint64_t * token_data =
        reinterpret_cast<const uint64_t *>(bytes);
    out.prompt_tokens.reserve(static_cast<size_t>(args.prompt_length));
    for (int32_t i = 0; i < args.prompt_length; ++i) {
        out.prompt_tokens.push_back(
            static_cast<llama_token>(token_data[static_cast<size_t>(i)]));
    }
    out.kv_qnn_u8_view =
        bytes + prompt_bytes;
    out.kv_qnn_u8_view_size = kv_bytes;
    out.prompt_len = args.prompt_length;
    out.num_layers = args.num_layers;
    out.num_kv_heads = args.num_kv_heads;
    out.head_dim = args.head_dim;
    out.first_token = args.first_token;
    out.first_token_is_prompt_tail = args.first_token_is_prompt_tail;
    out.manifest["original_prompt_length"] =
        out.prompt_len + (out.first_token_is_prompt_tail ? 1 : 0);
    out.metadata_read_ms = elapsed_ms(metadata_start);
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
    if (qnn_u8_layout && handoff.qnn_u8_size() == 0) {
        throw std::runtime_error("QNN U8 KV handoff is missing");
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
    const uint16_t * k_base = handoff.fp16_data();
    const uint16_t * v_base = handoff.fp16_data() + per_kind_values;
    const uint8_t * k_u8_base = handoff.qnn_u8_data();
    const uint8_t * v_u8_base = handoff.qnn_u8_data() +
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
        const common_params & params,
        pd_disk_embedding * disk_embedding) {
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
        if (disk_embedding != nullptr && disk_embedding->is_open()) {
            for (int32_t index = 0; index < chunk; ++index) {
                const llama_token token = handoff.prompt_tokens[start + index];
                llama_batch batch = {
                    /*.n_tokens =*/ 1,
                    /*.token    =*/ nullptr,
                    /*.embd     =*/ disk_embedding->read_row(token),
                    /*.pos      =*/ nullptr,
                    /*.n_seq_id =*/ nullptr,
                    /*.seq_id   =*/ nullptr,
                    /*.logits   =*/ nullptr,
                };
                if (llama_decode(native_ctx.get(), batch) != 0) {
                    throw std::runtime_error("native comparison embedding prompt decode failed");
                }
            }
        } else if (llama_decode(
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
    int first_result = 0;
    if (disk_embedding != nullptr && disk_embedding->is_open()) {
        llama_batch batch = {
            /*.n_tokens =*/ 1,
            /*.token    =*/ nullptr,
            /*.embd     =*/ disk_embedding->read_row(first),
            /*.pos      =*/ nullptr,
            /*.n_seq_id =*/ nullptr,
            /*.seq_id   =*/ nullptr,
            /*.logits   =*/ nullptr,
        };
        first_result = llama_decode(native_ctx.get(), batch);
    } else {
        first_result = llama_decode(native_ctx.get(), llama_batch_get_one(&first, 1));
    }
    if (first_result != 0) {
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
        bool v_trans,
        bool qnn_u8_layout) {
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
    const size_t element_size = qnn_u8_layout ? sizeof(uint8_t) : sizeof(uint16_t);
    const size_t row_size = static_cast<size_t>(n_embd_gqa) * element_size;
    const size_t rows = static_cast<size_t>(handoff.prompt_len);
    const size_t sample_values = std::min<size_t>(8, n_embd_gqa);

    auto block_in_bounds = [&](size_t block_offset, size_t data_size) {
        return block_offset <= imported_blob.size() &&
            data_size <= imported_blob.size() - block_offset &&
            block_offset <= native_blob.size() &&
            data_size <= native_blob.size() - block_offset;
    };

    auto log_row_sample = [&](const char * kind, int32_t layer, size_t block_offset) {
        if (rows == 0 || sample_values == 0) {
            return;
        }
        std::ostringstream oss;
        oss << "PD native compare " << kind << " layer=" << layer << " row0 imported/native:";
        for (size_t elem = 0; elem < sample_values; ++elem) {
            if (qnn_u8_layout) {
                oss << " [" << elem
                    << ":i=" << static_cast<unsigned>(imported_blob[block_offset + elem])
                    << ",n=" << static_cast<unsigned>(native_blob[block_offset + elem])
                    << "]";
            } else {
                uint16_t imported_bits = 0;
                uint16_t native_bits = 0;
                std::memcpy(&imported_bits, imported_blob.data() + block_offset + elem*sizeof(uint16_t), sizeof(uint16_t));
                std::memcpy(&native_bits, native_blob.data() + block_offset + elem*sizeof(uint16_t), sizeof(uint16_t));
                oss << " [" << elem
                    << ":i=" << fp16_bits_to_float(imported_bits)
                    << ",n=" << fp16_bits_to_float(native_bits)
                    << "]";
            }
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
            if (qnn_u8_layout) {
                oss << " [" << elem
                    << ":i=" << static_cast<unsigned>(imported_blob[block_offset + elem])
                    << ",n=" << static_cast<unsigned>(native_blob[block_offset + elem])
                    << "]";
            } else {
                uint16_t imported_bits = 0;
                uint16_t native_bits = 0;
                std::memcpy(&imported_bits, imported_blob.data() + block_offset + elem*sizeof(uint16_t), sizeof(uint16_t));
                std::memcpy(&native_bits, native_blob.data() + block_offset + elem*sizeof(uint16_t), sizeof(uint16_t));
                oss << " [" << elem
                    << ":i=" << fp16_bits_to_float(imported_bits)
                    << ",n=" << fp16_bits_to_float(native_bits)
                    << "]";
            }
        }
        LOG_INF("%s\n", oss.str().c_str());
    };

    auto compare_layer_block = [&](const char * kind, int32_t layer, size_t block_offset, size_t data_size) {
        if (!block_in_bounds(block_offset, data_size)) {
            LOG_INF(
                "PD native compare %s layer=%d block out of bounds: offset=%zu size=%zu blob=%zu\n",
                kind,
                layer,
                block_offset,
                data_size,
                imported_blob.size());
            return false;
        }

        float max_abs_diff = 0.0f;
        double mean_abs_diff = 0.0;
        size_t exact_values = 0;
        size_t max_idx = 0;
        const size_t value_count = data_size / element_size;
        for (size_t elem = 0; elem < value_count; ++elem) {
            float diff = 0.0f;
            if (qnn_u8_layout) {
                const uint8_t imported_value = imported_blob[block_offset + elem];
                const uint8_t native_value = native_blob[block_offset + elem];
                diff = static_cast<float>(std::abs(
                    static_cast<int>(imported_value) - static_cast<int>(native_value)));
                exact_values += imported_value == native_value;
            } else {
                uint16_t imported_bits = 0;
                uint16_t native_bits = 0;
                const size_t byte_offset = block_offset + elem * sizeof(uint16_t);
                std::memcpy(&imported_bits, imported_blob.data() + byte_offset, sizeof(uint16_t));
                std::memcpy(&native_bits, native_blob.data() + byte_offset, sizeof(uint16_t));
                diff = std::fabs(
                    fp16_bits_to_float(imported_bits) - fp16_bits_to_float(native_bits));
                exact_values += imported_bits == native_bits;
            }
            mean_abs_diff += diff;
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_idx = elem;
            }
        }
        mean_abs_diff /= static_cast<double>(value_count);
        LOG_INF(
            "PD native compare %s layer=%d max_abs_diff=%f mean_abs_diff=%f exact_pct=%.3f elem_index=%zu\n",
            kind,
            layer,
            max_abs_diff,
            mean_abs_diff,
            100.0 * static_cast<double>(exact_values) / static_cast<double>(value_count),
            max_idx);
        if (layer == 0) {
            log_row_sample(kind, layer, block_offset);
        }
        if (max_abs_diff > 8.0f || layer == 0) {
            log_maxdiff_sample(kind, layer, block_offset, max_idx);
        }
        return true;
    };

    for (int32_t layer = 0; layer < handoff.num_layers; ++layer) {
        offset += sizeof(int32_t) + sizeof(uint64_t);
        if (!compare_layer_block("K", layer, offset, rows * row_size)) {
            return;
        }
        offset += rows * row_size;
    }

    for (int32_t layer = 0; layer < handoff.num_layers; ++layer) {
        if (!v_trans) {
            offset += sizeof(int32_t) + sizeof(uint64_t);
            if (!compare_layer_block("V", layer, offset, rows * row_size)) {
                return;
            }
            offset += rows * row_size;
        } else {
            offset += sizeof(int32_t) + sizeof(uint32_t) + sizeof(uint32_t);
            if (!compare_layer_block("V", layer, offset, rows * row_size)) {
                return;
            }
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
        bool v_trans,
        pd_disk_embedding * disk_embedding) {
    const float * imported_logits = llama_get_logits_ith(imported_ctx, -1);
    if (imported_logits == nullptr) {
        throw std::runtime_error("imported logits are unavailable for comparison");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const native_compare_result native = run_native_compare(handoff, model, params, disk_embedding);

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
        v_trans,
        llama_qnn_u16_activations_enabled());
}

void print_usage(int argc, char ** argv) {
    (void) argc;
    LOG("\nexample usage:\n");
    LOG("  %s --pd-import handoff_dir -m model.gguf -n 128 -c 2048 -t 4 -ngl 0\n", argv[0]);
    LOG("  E2E runners may pass an inherited in-memory handoff with --pd-memory-fd\n");
    LOG("  separate embedding: add --pd-disk-embedding separate_embed_matrix.bin\n");
    LOG("  diagnostics: add --pd-roundtrip-check to compare imported KV against llama.cpp sequence serialization\n");
    LOG("  diagnostics: add --pd-native-compare to compare imported KV resume logits against native GGUF prefill\n");
    LOG("  quality fallback: add --pd-native-first-token to select the first continuation token with GGUF prompt prefill\n");
    LOG("  PPL: add --pd-ppl-tokens continuation.u64 --pd-ppl-output result.txt for teacher-forced scoring\n");
    LOG("  profiling: add --pd-op-profile profile.csv for synchronized per-node timing\n");
    LOG("\n");
}

} // namespace

struct llama_pd_inprocess_runtime {
    std::vector<std::string> args;
    common_params params;
    std::optional<std::string> deferred_profile_path;
    std::shared_ptr<llama_qnn_quant_profile> profile_transport;
    common_init_result_ptr llama_init;
    pd_disk_embedding disk_embedding;
    pd_persistent_threadpools threadpools;
    bool threadpools_attached = false;
    bool context_ready = false;
    bool warmup_complete = false;
    bool has_run = false;
    double initialization_ms = 0.0;
};

static int llama_pd_cli_main_impl(int argc, char ** argv) {
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
    if (g_inprocess_request != nullptr) {
        const auto & request = *g_inprocess_request;
        pd.memory_ptr = static_cast<const uint8_t *>(request.handoff_data);
        pd.memory_size = request.handoff_size;
        pd.prompt_tokens_ptr = request.prompt_tokens;
        pd.kv_fp16_ptr = request.kv_fp16;
        pd.kv_fp16_values = request.kv_fp16_values;
        pd.prompt_length = request.prompt_length;
        pd.num_layers = request.num_layers;
        pd.num_kv_heads = request.num_kv_heads;
        pd.head_dim = request.head_dim;
        pd.first_token = static_cast<llama_token>(request.first_token);
        pd.first_token_is_prompt_tail = request.first_token_is_prompt_tail;
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
        LOG_ERR("pd-cli does not accept a prompt; prompt tokens come from the PD handoff\n");
        return 1;
    }

    pd_capture_state pd_capture;
    pd_op_profile_state pd_op_profile;
    pd_op_profile.output_path = pd.op_profile_path;
    if (pd_op_profile.output_path.empty()) {
        if (const char * profile_path = std::getenv("LLAMA_QNN_PD_OP_PROFILE");
                profile_path != nullptr && profile_path[0] != '\0') {
            pd_op_profile.output_path = profile_path;
        }
    }
    if (const char * profile_mode = std::getenv("LLAMA_QNN_PD_OP_PROFILE_MODE");
            profile_mode != nullptr && profile_mode[0] != '\0') {
        if (std::strcmp(profile_mode, "layer") == 0) {
            pd_op_profile.layer_granularity = true;
        } else if (std::strcmp(profile_mode, "node") != 0) {
            LOG_ERR("invalid LLAMA_QNN_PD_OP_PROFILE_MODE=%s (expected node or layer)\n",
                    profile_mode);
            return 1;
        }
    }
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
    if (!pd_op_profile.output_path.empty()) {
        if (!pd_capture.output_dir.empty()) {
            LOG_ERR("--pd-op-profile cannot be combined with LLAMA_QNN_PD_DUMP_DIR\n");
            return 1;
        }
        params.cb_eval = pd_op_profile_callback;
        params.cb_eval_user_data = &pd_op_profile;
        LOG_INF("PD profiling armed: path=%s mode=%s\n",
                pd_op_profile.output_path.c_str(),
                pd_op_profile.layer_granularity ? "layer" : "node");
    }
    if (g_inprocess_runtime == nullptr) {
        llama_backend_init();
        llama_numa_init(params.numa);
    }

    // A resident PD child first loads only the GGUF. The QNN profile mapping,
    // context compute buffers, and full-capacity KV cache are intentionally
    // deferred until Prefill sends PDPR after releasing its rebuild inputs.
    std::optional<std::string> deferred_profile_path;
    if (pd.control_fd >= 0) {
        if (const char * path = std::getenv("LLAMA_QNN_U16_QPARAMS_MANIFEST");
                path != nullptr && path[0] != '\0') {
            deferred_profile_path = path;
            unsetenv("LLAMA_QNN_U16_QPARAMS_MANIFEST");
        }
    }
    common_init_result_ptr local_llama_init;
    common_init_result * llama_init = nullptr;
    if (g_inprocess_runtime != nullptr) {
        llama_init = g_inprocess_runtime->llama_init.get();
    } else {
        local_llama_init = g_inprocess_request != nullptr
            ? common_init_from_params_buffer(
                  params,
                  g_inprocess_request->model_data,
                  g_inprocess_request->model_size,
                  pd.control_fd >= 0)
            : common_init_from_params(params, pd.control_fd >= 0);
        llama_init = local_llama_init.get();
    }
    if (deferred_profile_path.has_value()) {
        setenv(
            "LLAMA_QNN_U16_QPARAMS_MANIFEST",
            deferred_profile_path->c_str(),
            1);
    }
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        LOG_ERR("failed to initialize model\n");
        return 1;
    }

    pd_persistent_threadpools local_threadpools;
    pd_persistent_threadpools & persistent_threadpools =
        g_inprocess_runtime != nullptr
        ? g_inprocess_runtime->threadpools
        : local_threadpools;

    char external_embedding_kind[32] = {};
    const bool model_requires_disk_embedding =
        llama_model_meta_val_str(
            model,
            "general.external_token_embedding",
            external_embedding_kind,
            sizeof(external_embedding_kind)) > 0;
    if (model_requires_disk_embedding && pd.disk_embedding_path.empty()) {
        LOG_ERR(
            "model requires an external token embedding; "
            "pass --pd-disk-embedding PATH\n");
        return 1;
    }

    pd_disk_embedding local_disk_embedding;
    pd_disk_embedding & disk_embedding = g_inprocess_runtime != nullptr
        ? g_inprocess_runtime->disk_embedding
        : local_disk_embedding;
    if (g_inprocess_runtime == nullptr && !pd.disk_embedding_path.empty()) {
        try {
            disk_embedding.open_file(pd.disk_embedding_path);
            if (disk_embedding.vocab_size() !=
                    static_cast<uint32_t>(
                        llama_vocab_n_tokens(llama_model_get_vocab(model))) ||
                disk_embedding.embedding_dim() !=
                    static_cast<uint32_t>(llama_model_n_embd_inp(model))) {
                throw std::runtime_error(
                    "disk embedding dimensions do not match the GGUF model");
            }
        } catch (const std::exception & err) {
            LOG_ERR("failed to initialize PD disk embedding: %s\n", err.what());
            return 1;
        }
    }

    const auto persistent_boundary_start = steady_clock::now();
    std::optional<steady_clock::time_point> lazy_decode_start;
    if (g_inprocess_runtime != nullptr && g_inprocess_request != nullptr &&
            g_inprocess_request->lazy_quant_profile_after_prefill) {
        lazy_decode_start = steady_clock::now();
        const process_memory_snapshot memory_before_lazy = process_memory();
        double preparation_ms = 0.0;
        if (!llama_pd_inprocess_runtime_prepare_context(
                g_inprocess_runtime, &preparation_ms)) {
            LOG_ERR("failed to lazily prepare QNN Decode metadata/context\n");
            return 1;
        }
        const process_memory_snapshot memory_after_lazy = process_memory();
        LOG_INF(
            "PD lazy quant profile prepared after Prefill: ms=%.3f "
            "rss_before_mib=%.2f rss_after_mib=%.2f rss_delta_mib=%.2f\n",
            preparation_ms,
            bytes_to_mib(memory_before_lazy.rss_bytes),
            bytes_to_mib(memory_after_lazy.rss_bytes),
            signed_bytes_to_mib(mapping_delta(
                memory_after_lazy.rss_bytes, memory_before_lazy.rss_bytes)));
    }
    if (pd.control_fd >= 0) {
        try {
            LOG_INF(
                "PD resident Decode ready: control_fd=%d model-only initialized; runtime deferred\n",
                pd.control_fd);
            send_resident_ready(pd.control_fd);
            receive_resident_prepare(pd.control_fd);
            const auto runtime_prepare_start = steady_clock::now();
            LOG_INF(
                "PD resident Decode prepare received: attaching metadata and creating context/KV\n");
            if (!llama_qnn_u16_attach_profile_from_environment(model)) {
                throw std::runtime_error(
                    "unable to attach deferred QNN runtime metadata");
            }
            const double metadata_attach_ms =
                elapsed_ms(runtime_prepare_start);
            const auto context_prepare_start = steady_clock::now();
            if (!llama_init->init_context(params)) {
                throw std::runtime_error(
                    "unable to create deferred Decode context");
            }
            persistent_threadpools.attach(llama_init->context(), params);
            const double context_threadpool_ms =
                elapsed_ms(context_prepare_start);
            const process_memory_snapshot prepared_memory = process_memory();
            LOG_INF(
                "PD resident Decode runtime prepared: metadata_ms=%.3f "
                "context_kv_threadpool_ms=%.3f total_ms=%.3f rss_mib=%.2f hwm_mib=%.2f\n",
                metadata_attach_ms,
                context_threadpool_ms,
                elapsed_ms(runtime_prepare_start),
                prepared_memory.rss_bytes / (1024.0 * 1024.0),
                prepared_memory.hwm_bytes / (1024.0 * 1024.0));
            receive_resident_handoff(pd);
            LOG_INF(
                "PD resident Decode received in-memory handoff: bytes=%zu prompt_len=%d\n",
                pd.memory_size,
                pd.prompt_length);
        } catch (const std::exception & err) {
            LOG_ERR("resident Decode handoff failed: %s\n", err.what());
            return 1;
        }
    } else if (g_inprocess_runtime == nullptr) {
        try {
            persistent_threadpools.attach(llama_init->context(), params);
        } catch (const std::exception & err) {
            LOG_ERR("failed to initialize persistent CPU threadpool: %s\n", err.what());
            return 1;
        }
    } else if (!g_inprocess_runtime->threadpools_attached) {
        try {
            persistent_threadpools.attach(llama_init->context(), params);
            g_inprocess_runtime->threadpools_attached = true;
        } catch (const std::exception & err) {
            LOG_ERR("failed to initialize persistent CPU threadpool: %s\n", err.what());
            return 1;
        }
    }

    llama_context * ctx = llama_init->context();
    common_sampler * smpl = llama_init->sampler(0);
    if (ctx == nullptr || smpl == nullptr) {
        LOG_ERR("failed to initialize context/sampler\n");
        return 1;
    }
    const process_memory_snapshot memory_after_model_load = process_memory();

    const double initialization_ms = g_inprocess_runtime != nullptr
        ? g_inprocess_runtime->initialization_ms
        : elapsed_ms(process_start);
    const auto boundary_start = g_inprocess_runtime != nullptr
        ? persistent_boundary_start
        : steady_clock::now();
    if (g_inprocess_runtime != nullptr) {
        // A reused context keeps model/metadata/KV/scheduler allocation stable.
        // Reset KV metadata only. The imported handoff overwrites every valid
        // prefix entry; clearing the full ctx-sized backing would write about
        // 1.25 GiB for Qwen3-14B ctx4096 and evict the warmed model pages.
        llama_memory_clear(llama_get_memory(ctx), false);
        llama_synchronize(ctx);
        llama_init->reset_samplers();
        llama_perf_context_reset(ctx);
    }
    const bool qnn_u8_layout = llama_qnn_u16_activations_enabled();
    const bool need_fp16_handoff = true;
    pd_handoff handoff;
    try {
        handoff = (pd.memory_fd >= 0 || pd.memory_ptr != nullptr ||
                   pd.prompt_tokens_ptr != nullptr)
            ? load_pd_memory_handoff(pd)
            : load_pd_handoff(
                  pd.import_dir, need_fp16_handoff, qnn_u8_layout);
        validate_pd_handoff(handoff, model, ctx);
        if (qnn_u8_layout && handoff.qnn_u8_size() == 0) {
            if (handoff.fp16_size() == 0) {
                throw std::runtime_error(
                    "QNN-aligned Decode requires QNN U8 or canonical FP16 KV");
            }
            const auto bridge_start = steady_clock::now();
            handoff.kv_qnn_u8.resize(handoff.fp16_size());
            llama_qnn_kv_quantize_stats bridge_stats {};
            std::string bridge_error;
            if (!llama_qnn_u16_quantize_fp16_kv(
                    model,
                    handoff.fp16_data(),
                    handoff.fp16_size(),
                    handoff.num_layers,
                    handoff.num_kv_heads,
                    handoff.prompt_len,
                    handoff.head_dim,
                    handoff.kv_qnn_u8.data(),
                    &bridge_stats,
                    &bridge_error)) {
                throw std::runtime_error(
                    "FP16 to QNN U8 KV bridge failed: " + bridge_error);
            }
            const double zero_pct = bridge_stats.values == 0 ? 0.0 :
                100.0 * bridge_stats.code_zero / bridge_stats.values;
            const double max_pct = bridge_stats.values == 0 ? 0.0 :
                100.0 * bridge_stats.code_255 / bridge_stats.values;
            LOG_INF(
                "PD FP16->QNN-U8 KV bridge: values=%zu ms=%.3f "
                "non_finite=%zu code0=%zu(%.6f%%) code255=%zu(%.6f%%)\n",
                bridge_stats.values,
                elapsed_ms(bridge_start),
                bridge_stats.non_finite,
                bridge_stats.code_zero,
                zero_pct,
                bridge_stats.code_255,
                max_pct);
        }
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
            g_pd_capture_state = capture_native_prompt ? &pd_capture : nullptr;
            pd_capture.active = capture_native_prompt;
            const native_compare_result native =
                run_native_compare(handoff, model, params,
                    disk_embedding.is_open() ? &disk_embedding : nullptr);
            pd_capture.active = false;
            g_pd_capture_state = nullptr;
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
            g_pd_capture_state = nullptr;
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
    const bool v_trans = qnn_u8_layout
        ? false
        : params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED;
    const bool direct_qnn_u8_import =
        qnn_u8_layout && !pd.roundtrip_check && !pd.native_compare;
    std::vector<uint8_t> seq_blob;
    double kv_layout_ms = 0.0;
    if (!direct_qnn_u8_import) {
        const auto kv_layout_start = steady_clock::now();
        seq_blob = build_seq_state_blob(handoff, v_trans, qnn_u8_layout);
        kv_layout_ms = elapsed_ms(kv_layout_start);
    }

    const auto kv_import_start = steady_clock::now();
    bool kv_import_ok = false;
    if (direct_qnn_u8_import) {
        kv_import_ok = llama_state_seq_set_qnn_u8_kv(
            ctx, 0, handoff.qnn_u8_data(),
            static_cast<uint32_t>(handoff.prompt_len),
            static_cast<uint32_t>(handoff.num_layers),
            static_cast<uint32_t>(handoff.num_kv_heads),
            static_cast<uint32_t>(handoff.head_dim));
    } else {
        const size_t nset =
            llama_state_seq_set_data(ctx, seq_blob.data(), seq_blob.size(), 0);
        kv_import_ok = nset == seq_blob.size();
        if (!kv_import_ok) {
            LOG_ERR(
                "failed to import PD KV state: written=%zu expected=%zu\n",
                nset, seq_blob.size());
        }
    }
    const double kv_import_ms = elapsed_ms(kv_import_start);
    if (!kv_import_ok) {
        LOG_ERR("failed to import PD KV state directly=%d\n",
                direct_qnn_u8_import ? 1 : 0);
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
    const size_t handoff_kv_bytes =
        handoff.fp16_size() * sizeof(uint16_t) +
        handoff.qnn_u8_size();
    const size_t imported_seq_blob_size = seq_blob.size();
    LOG_INF(
        "PD handoff timing: metadata_read_ms=%.3f kv_read_ms=%.3f "
        "kv_layout_ms=%.3f kv_import_ms=%.3f validation_ms=%.3f "
        "total_ms=%.3f kv_bytes=%zu seq_blob_bytes=%zu direct_qnn_u8=%d\n",
        handoff.metadata_read_ms,
        handoff.kv_read_ms,
        kv_layout_ms,
        kv_import_ms,
        kv_validation_ms,
        handoff_total_ms,
        handoff_kv_bytes,
        imported_seq_blob_size,
        direct_qnn_u8_import ? 1 : 0);
    if (!pd.roundtrip_check && !pd.native_compare) {
        handoff.kv_qnn_u8.clear();
        handoff.kv_qnn_u8.shrink_to_fit();
        handoff.kv_qnn_u8_view = nullptr;
        handoff.kv_qnn_u8_view_size = 0;
        handoff.memory_mapping.reset();
        seq_blob.clear();
        seq_blob.shrink_to_fit();
    }
    if (!pd.native_compare) {
        handoff.kv_fp16.clear();
        handoff.kv_fp16.shrink_to_fit();
        handoff.kv_fp16_view = nullptr;
        handoff.kv_fp16_view_size = 0;
    }
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
    const double boundary_ms = lazy_decode_start.has_value()
        ? std::chrono::duration<double, std::milli>(
              *lazy_decode_start - boundary_start).count()
        : elapsed_ms(boundary_start);
    const auto decode_start = lazy_decode_start.has_value()
        ? *lazy_decode_start
        : steady_clock::now();
    const bool log_token_timing = std::getenv("LLAMA_PD_TOKEN_TIMING") != nullptr;
    int32_t decode_call_index = 0;
    const auto decode_one = [&](llama_token token) {
        const bool probe_this_decode =
            g_inprocess_request != nullptr &&
            g_inprocess_request->decode_event_callback != nullptr &&
            decode_call_index < g_inprocess_request->decode_event_call_limit;
        if (probe_this_decode) {
            g_inprocess_request->decode_event_callback(
                g_inprocess_request->decode_event_opaque,
                "begin",
                decode_call_index,
                token);
        }
        const bool profile_token_mappings =
            decode_call_index == 0 &&
            std::getenv("LLAMA_PD_TOKEN_MAPPING_PROFILE") != nullptr;
        const process_mapping_snapshot token_mappings_before =
            profile_token_mappings
            ? process_mappings()
            : process_mapping_snapshot();
        const auto token_decode_start =
            decode_call_index == 0 && lazy_decode_start.has_value()
            ? *lazy_decode_start
            : steady_clock::now();
        const bool capture_this_decode =
            !pd_capture.output_dir.empty() && !pd_capture.completed;
        g_pd_capture_state = capture_this_decode ? &pd_capture : nullptr;
        pd_capture.active = capture_this_decode;
        pd_op_profile.begin_decode(token);
        g_pd_op_profile_state = pd_op_profile.output_path.empty()
            ? nullptr
            : &pd_op_profile;
        int result = 0;
        if (disk_embedding.is_open()) {
            try {
                llama_batch batch = {
                    /*.n_tokens =*/ 1,
                    /*.token    =*/ nullptr,
                    /*.embd     =*/ disk_embedding.read_row(token),
                    /*.pos      =*/ nullptr,
                    /*.n_seq_id =*/ nullptr,
                    /*.seq_id   =*/ nullptr,
                    /*.logits   =*/ nullptr,
                };
                result = llama_decode(ctx, batch);
            } catch (const std::exception & err) {
                LOG_ERR("PD disk embedding read failed: %s\n", err.what());
                result = -1;
            }
        } else {
            result = llama_decode(ctx, llama_batch_get_one(&token, 1));
        }
        const double token_decode_ms = elapsed_ms(token_decode_start);
        if (profile_token_mappings) {
            log_mapping_delta(
                "decode-token0",
                token_mappings_before,
                process_mappings());
        }
        if (probe_this_decode) {
            g_inprocess_request->decode_event_callback(
                g_inprocess_request->decode_event_opaque,
                "end",
                decode_call_index,
                token);
        }
        pd_op_profile.end_decode();
        g_pd_op_profile_state = nullptr;
        pd_capture.active = false;
        g_pd_capture_state = nullptr;
        if (capture_this_decode) {
            finish_pd_capture(pd_capture, token);
            LOG_INF(
                "PD intermediate capture finished: token=%d captured=%zu requested=%zu error=%s\n",
                token,
                pd_capture.captured.size(),
                pd_capture.requested.size(),
                pd_capture.error.empty() ? "none" : pd_capture.error.c_str());
        }
        if (log_token_timing) {
            LOG_INF(
                "PD token decode timing: index=%d token=%d ms=%.3f result=%d\n",
                decode_call_index,
                token,
                token_decode_ms,
                result);
        }
        ++decode_call_index;
        return result;
    };

    bool lazy_profile_released = false;
    const auto release_lazy_profile = [&]() {
        if (!lazy_decode_start.has_value() || lazy_profile_released) {
            return;
        }
        llama_synchronize(ctx);
        const auto release_start = steady_clock::now();
        const process_memory_snapshot memory_before_release = process_memory();
        llama_qnn_u16_detach_profile(model);
        g_inprocess_runtime->profile_transport.reset();
        lazy_profile_released = true;
        const process_memory_snapshot memory_after_release = process_memory();
        LOG_INF(
            "PD lazy quant profile released after generation: ms=%.3f "
            "rss_before_mib=%.2f rss_after_mib=%.2f rss_delta_mib=%.2f\n",
            elapsed_ms(release_start),
            bytes_to_mib(memory_before_release.rss_bytes),
            bytes_to_mib(memory_after_release.rss_bytes),
            signed_bytes_to_mib(mapping_delta(
                memory_after_release.rss_bytes, memory_before_release.rss_bytes)));
    };

    if (!pd.ppl_tokens_path.empty()) {
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
        const int32_t ppl_prompt_tokens = handoff.manifest.value(
            "original_prompt_length",
            handoff.prompt_len + (handoff.first_token_is_prompt_tail ? 1 : 0));
        if (static_cast<int64_t>(handoff.prompt_len) +
                (handoff.first_token_is_prompt_tail ? 1 : 0) +
                static_cast<int64_t>(continuation.size()) >
            llama_n_ctx(ctx)) {
            LOG_ERR(
                "PD PPL sequence exceeds context: prompt=%d continuation=%zu context=%u\n",
                ppl_prompt_tokens,
                continuation.size(),
                llama_n_ctx(ctx));
            return 1;
        }

        const int32_t n_vocab =
            llama_vocab_n_tokens(llama_model_get_vocab(model));
        double total_nll = 0.0;
        int64_t scored_tokens = 0;
        const auto ppl_start = steady_clock::now();
        if (handoff.first_token_is_prompt_tail &&
            decode_one(handoff.first_token) != 0) {
            LOG_ERR("llama_decode failed for PD PPL prompt-tail bridge token\n");
            return 1;
        }
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
        release_lazy_profile();
        const double ppl = std::exp(total_nll / static_cast<double>(scored_tokens));
        const double ppl_ms = elapsed_ms(ppl_start);
        LOG_INF(
            "PD WikiPPL: ppl=%.9f nll=%.9f scored_tokens=%" PRId64
            " prompt_tokens=%d continuation_tokens=%zu eval_ms=%.3f tokens_per_second=%.3f\n",
            ppl,
            total_nll,
            scored_tokens,
            ppl_prompt_tokens,
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
                   << "prompt_tokens=" << ppl_prompt_tokens << "\n"
                   << "continuation_tokens=" << continuation.size() << "\n"
                   << "eval_ms=" << ppl_ms << "\n";
        }
        common_perf_print(ctx, smpl);
        return 0;
    }

    if (!infinite && n_remain == 0) {
        release_lazy_profile();
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
            log_native_comparison(ctx, model, params, handoff, seq_blob, v_trans,
                disk_embedding.is_open() ? &disk_embedding : nullptr);
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

    const double decode_ms = elapsed_ms(decode_start);
    release_lazy_profile();
    LOG("\n");
    if (g_inprocess_request != nullptr &&
        g_inprocess_request->result != nullptr) {
        *g_inprocess_request->result = {
            initialization_ms,
            boundary_ms,
            decode_ms,
            generated_tokens,
        };
    }
    if (g_inprocess_runtime != nullptr) {
        g_inprocess_runtime->has_run = true;
    }
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

llama_pd_inprocess_runtime * llama_pd_inprocess_runtime_create_model_only(
        int argc,
        char ** argv,
        const void * model_data,
        size_t model_size,
        double * initialization_ms) {
    if (model_data == nullptr || model_size == 0) {
        return nullptr;
    }

    const auto init_start = steady_clock::now();
    const process_memory_snapshot memory_before = process_memory();
    std::unique_ptr<llama_pd_inprocess_runtime> runtime(
        new llama_pd_inprocess_runtime());
    try {
        runtime->args.reserve(static_cast<size_t>(std::max(argc, 0)));
        for (int i = 0; i < argc; ++i) {
            runtime->args.emplace_back(argv[i] != nullptr ? argv[i] : "");
        }
        std::vector<char *> forwarded;
        g_inprocess_preparing = true;
        pd_args pd;
        try {
            pd = parse_pd_args(argc, argv, &forwarded);
        } catch (...) {
            g_inprocess_preparing = false;
            throw;
        }
        g_inprocess_preparing = false;

        common_init();
        if (!common_params_parse(
                static_cast<int>(forwarded.size()),
                forwarded.data(),
                runtime->params,
                LLAMA_EXAMPLE_COMPLETION,
                print_usage)) {
            throw std::runtime_error(
                "failed to parse persistent Decode parameters");
        }
        if (!runtime->params.prompt.empty()) {
            throw std::runtime_error(
                "persistent Decode does not accept a prompt");
        }

        const char * dump_dir = std::getenv("LLAMA_QNN_PD_DUMP_DIR");
        const char * profile_path = std::getenv("LLAMA_QNN_PD_OP_PROFILE");
        if ((dump_dir != nullptr && dump_dir[0] != 0) ||
                (profile_path != nullptr && profile_path[0] != 0)) {
            runtime->params.cb_eval = pd_resident_eval_callback;
            runtime->params.cb_eval_user_data = nullptr;
        }

        llama_backend_init();
        llama_numa_init(runtime->params.numa);
        if (const char * path =
                std::getenv("LLAMA_QNN_U16_QPARAMS_MANIFEST");
                path != nullptr && path[0] != 0) {
            runtime->deferred_profile_path = path;
            unsetenv("LLAMA_QNN_U16_QPARAMS_MANIFEST");
        }
        try {
            runtime->llama_init = common_init_from_params_buffer(
                runtime->params, model_data, model_size, true);
        } catch (...) {
            if (runtime->deferred_profile_path.has_value()) {
                setenv(
                    "LLAMA_QNN_U16_QPARAMS_MANIFEST",
                    runtime->deferred_profile_path->c_str(),
                    1);
            }
            throw;
        }
        if (runtime->deferred_profile_path.has_value()) {
            setenv(
                "LLAMA_QNN_U16_QPARAMS_MANIFEST",
                runtime->deferred_profile_path->c_str(),
                1);
        }
        llama_model * model = runtime->llama_init
            ? runtime->llama_init->model()
            : nullptr;
        if (model == nullptr) {
            throw std::runtime_error(
                "failed to create persistent Decode model");
        }

        char external_embedding_kind[32] = {};
        const bool model_requires_disk_embedding =
            llama_model_meta_val_str(
                model,
                "general.external_token_embedding",
                external_embedding_kind,
                sizeof(external_embedding_kind)) > 0;
        if (model_requires_disk_embedding && pd.disk_embedding_path.empty()) {
            throw std::runtime_error(
                "persistent Decode model requires --pd-disk-embedding");
        }
        if (!pd.disk_embedding_path.empty()) {
            runtime->disk_embedding.open_file(pd.disk_embedding_path);
            if (runtime->disk_embedding.vocab_size() !=
                    static_cast<uint32_t>(
                        llama_vocab_n_tokens(llama_model_get_vocab(model))) ||
                runtime->disk_embedding.embedding_dim() !=
                    static_cast<uint32_t>(llama_model_n_embd_inp(model))) {
                throw std::runtime_error(
                    "disk embedding dimensions do not match the persistent Decode model");
            }
        }

        runtime->initialization_ms = elapsed_ms(init_start);
        if (initialization_ms != nullptr) {
            *initialization_ms = runtime->initialization_ms;
        }
        const process_memory_snapshot memory_after = process_memory();
        LOG_INF(
            "PD persistent model-only runtime ready: model_ptr=%p "
            "model_bytes=%zu initialization_ms=%.3f rss_before_mib=%.2f "
            "rss_after_mib=%.2f rss_delta_mib=%.2f\n",
            model_data,
            model_size,
            runtime->initialization_ms,
            bytes_to_mib(memory_before.rss_bytes),
            bytes_to_mib(memory_after.rss_bytes),
            bytes_to_mib(memory_after.rss_bytes -
                std::min(memory_after.rss_bytes, memory_before.rss_bytes)));
        return runtime.release();
    } catch (const std::exception & err) {
        g_inprocess_preparing = false;
        LOG_ERR("failed to create model-only in-process Decode runtime: %s\n",
                err.what());
        return nullptr;
    }
}

bool llama_pd_inprocess_runtime_prepare_context(
        llama_pd_inprocess_runtime * runtime,
        double * preparation_ms) {
    if (runtime == nullptr || runtime->llama_init == nullptr ||
            runtime->llama_init->model() == nullptr) {
        return false;
    }
    if (runtime->context_ready) {
        if (preparation_ms != nullptr) {
            *preparation_ms = 0.0;
        }
        return true;
    }

    const auto prepare_start = steady_clock::now();
    const process_memory_snapshot memory_before = process_memory();
    const auto metadata_start = steady_clock::now();
    if (runtime->deferred_profile_path.has_value()) {
        const bool attached = runtime->profile_transport != nullptr
            ? llama_qnn_u16_attach_profile_from_environment(
                  runtime->llama_init->model(), runtime->profile_transport)
            : llama_qnn_u16_attach_profile_from_environment(
                  runtime->llama_init->model());
        if (!attached) {
            LOG_ERR("failed to attach deferred QNN runtime metadata\n");
            return false;
        }
        runtime->profile_transport.reset();
    }
    const double metadata_ms = elapsed_ms(metadata_start);
    const process_memory_snapshot memory_after_metadata = process_memory();
    const char * previous_tg_only = std::getenv("LLAMA_PD_TG_ONLY_RESERVE");
    const bool had_previous_tg_only = previous_tg_only != nullptr;
    const std::string previous_tg_only_value = had_previous_tg_only
        ? previous_tg_only : "";
    const auto restore_tg_only = [&]() {
        if (had_previous_tg_only) {
            setenv("LLAMA_PD_TG_ONLY_RESERVE", previous_tg_only_value.c_str(), 1);
        } else {
            unsetenv("LLAMA_PD_TG_ONLY_RESERVE");
        }
    };
    setenv("LLAMA_PD_TG_ONLY_RESERVE", "1", 1);
    try {
        const bool initialized =
            runtime->llama_init->init_context(runtime->params);
        restore_tg_only();
        if (!initialized ||
                runtime->llama_init->context() == nullptr ||
                runtime->llama_init->sampler(0) == nullptr) {
            throw std::runtime_error(
                "failed to create persistent Decode context");
        }
        runtime->threadpools.attach(
            runtime->llama_init->context(), runtime->params);
        runtime->threadpools_attached = true;
        runtime->context_ready = true;
        const llama_memory_breakdown memory_breakdown =
            llama_get_memory_breakdown(runtime->llama_init->context());
        size_t context_bytes = 0;
        size_t compute_bytes = 0;
        for (const auto & entry : memory_breakdown) {
            context_bytes += entry.second.context;
            compute_bytes += entry.second.compute;
        }
        LOG_INF(
            "PD Decode fixed buffers: context_bytes=%zu compute_bytes=%zu "
            "total_bytes=%zu\n",
            context_bytes,
            compute_bytes,
            context_bytes + compute_bytes);
        const double elapsed = elapsed_ms(prepare_start);
        runtime->initialization_ms += elapsed;
        if (preparation_ms != nullptr) {
            *preparation_ms = elapsed;
        }
        const process_memory_snapshot memory_after = process_memory();
        LOG_INF(
            "PD persistent Decode context ready: preparation_ms=%.3f "
            "metadata_ms=%.3f rss_before_mib=%.2f metadata_rss_mib=%.2f "
            "rss_after_mib=%.2f rss_delta_mib=%.2f\n",
            elapsed,
            metadata_ms,
            bytes_to_mib(memory_before.rss_bytes),
            bytes_to_mib(memory_after_metadata.rss_bytes),
            bytes_to_mib(memory_after.rss_bytes),
            bytes_to_mib(memory_after.rss_bytes -
                std::min(memory_after.rss_bytes, memory_before.rss_bytes)));
        return true;
    } catch (const std::exception & err) {
        restore_tg_only();
        LOG_ERR("failed to prepare persistent in-process Decode context: %s\n",
                err.what());
        return false;
    }
}

bool llama_pd_inprocess_runtime_warmup(
        llama_pd_inprocess_runtime * runtime,
        double * warmup_ms) {
    if (runtime == nullptr || !runtime->context_ready ||
            runtime->llama_init == nullptr ||
            runtime->llama_init->context() == nullptr) {
        return false;
    }
    if (runtime->warmup_complete || !runtime->params.warmup) {
        runtime->warmup_complete = true;
        if (warmup_ms != nullptr) {
            *warmup_ms = 0.0;
        }
        return true;
    }

    const auto warmup_start = steady_clock::now();
    try {
        const bool profile_memory = warmup_memory_profile_enabled();
        const process_memory_snapshot memory_before = process_memory();
        const process_mapping_snapshot mappings_before =
            profile_memory ? process_mappings() : process_mapping_snapshot();
        if (profile_memory) {
            log_warmup_memory_stage("before_decode", memory_before);
        }
        const llama_memory_breakdown breakdown_before =
            llama_get_memory_breakdown(runtime->llama_init->context());
        size_t context_before = 0;
        size_t compute_before = 0;
        for (const auto & entry : breakdown_before) {
            context_before += entry.second.context;
            compute_before += entry.second.compute;
        }
        llama_model * model = runtime->llama_init->model();
        llama_token warmup_token =
            llama_vocab_bos(llama_model_get_vocab(model));
        if (warmup_token == LLAMA_TOKEN_NULL) {
            warmup_token = 0;
        }
        int warmup_result = 0;
        if (runtime->disk_embedding.is_open()) {
            llama_batch warmup_batch = {
                /*.n_tokens =*/ 1,
                /*.token    =*/ nullptr,
                /*.embd     =*/ runtime->disk_embedding.read_row(warmup_token),
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            warmup_result = llama_decode(
                runtime->llama_init->context(), warmup_batch);
        } else {
            warmup_result = llama_decode(
                runtime->llama_init->context(),
                llama_batch_get_one(&warmup_token, 1));
        }
        if (profile_memory) {
            log_warmup_memory_stage("after_decode", process_memory());
        }
        if (warmup_result != 0) {
            throw std::runtime_error(
                "persistent Decode initialization warmup failed");
        }
        llama_synchronize(runtime->llama_init->context());
        if (profile_memory) {
            log_warmup_memory_stage("after_synchronize", process_memory());
        }
        llama_memory_clear(
            llama_get_memory(runtime->llama_init->context()), false);
        llama_perf_context_reset(runtime->llama_init->context());
        runtime->llama_init->reset_samplers();
        const process_memory_snapshot memory_after = process_memory();
        if (profile_memory) {
            log_warmup_memory_stage("after_clear", memory_after);
            log_mapping_delta(
                "warmup", mappings_before, process_mappings());
        }
        const llama_memory_breakdown breakdown_after =
            llama_get_memory_breakdown(runtime->llama_init->context());
        size_t context_after = 0;
        size_t compute_after = 0;
        for (const auto & entry : breakdown_after) {
            context_after += entry.second.context;
            compute_after += entry.second.compute;
        }
        LOG_INF(
            "PD warmup retained memory: rss_delta_mib=%+.2f "
            "rss_anon_delta_mib=%+.2f rss_file_delta_mib=%+.2f "
            "swap_delta_mib=%+.2f context_before=%zu context_after=%zu "
            "compute_before=%zu compute_after=%zu\n",
            signed_bytes_to_mib(mapping_delta(
                memory_after.rss_bytes, memory_before.rss_bytes)),
            signed_bytes_to_mib(mapping_delta(
                memory_after.rss_anon_bytes, memory_before.rss_anon_bytes)),
            signed_bytes_to_mib(mapping_delta(
                memory_after.rss_file_bytes, memory_before.rss_file_bytes)),
            signed_bytes_to_mib(mapping_delta(
                memory_after.swap_bytes, memory_before.swap_bytes)),
            context_before,
            context_after,
            compute_before,
            compute_after);
        runtime->warmup_complete = true;
        const double elapsed = elapsed_ms(warmup_start);
        runtime->initialization_ms += elapsed;
        if (warmup_ms != nullptr) {
            *warmup_ms = elapsed;
        }
        LOG_INF(
            "PD Decode initialization warmup complete: token=%d ms=%.3f\n",
            warmup_token,
            elapsed);
        return true;
    } catch (const std::exception & err) {
        LOG_ERR("failed to warm persistent in-process Decode runtime: %s\n",
                err.what());
        return false;
    }
}

bool llama_pd_inprocess_runtime_profile_backing(
        llama_pd_inprocess_runtime * runtime,
        const void ** data,
        size_t * size,
        bool * anonymous) {
    if (runtime == nullptr || runtime->llama_init == nullptr ||
            runtime->llama_init->model() == nullptr) {
        return false;
    }
    const llama_qnn_quant_profile * profile =
        runtime->llama_init->model()->get_qnn_u16_profile();
    return profile != nullptr &&
        profile->binary_backing_info(data, size, anonymous);
}

bool llama_pd_inprocess_runtime_profile_stream_prepare(
        llama_pd_inprocess_runtime * runtime,
        size_t shard_count) {
    if (runtime == nullptr || runtime->llama_init == nullptr ||
            runtime->llama_init->model() == nullptr) {
        return false;
    }
    const llama_qnn_quant_profile * profile =
        runtime->llama_init->model()->get_qnn_u16_profile();
    if (profile == nullptr && runtime->profile_transport == nullptr) {
        if (!runtime->deferred_profile_path.has_value()) {
            return false;
        }
        try {
            runtime->profile_transport =
                llama_qnn_quant_profile_load_binary_transport_file(
                    *runtime->deferred_profile_path);
        } catch (const std::exception & err) {
            LOG_ERR("failed to prepare deferred QNN sidecar transport: %s\n",
                    err.what());
            return false;
        }
    }
    profile = profile != nullptr ? profile : runtime->profile_transport.get();
    return profile != nullptr && profile->binary_stream_prepare(shard_count);
}

bool llama_pd_inprocess_runtime_profile_stream_fill(
        llama_pd_inprocess_runtime * runtime,
        size_t shard_index,
        size_t * bytes_loaded) {
    if (runtime == nullptr || runtime->llama_init == nullptr ||
            runtime->llama_init->model() == nullptr) {
        return false;
    }
    const llama_qnn_quant_profile * profile =
        runtime->llama_init->model()->get_qnn_u16_profile();
    profile = profile != nullptr ? profile : runtime->profile_transport.get();
    return profile != nullptr &&
        profile->binary_stream_fill(shard_index, bytes_loaded);
}

bool llama_pd_inprocess_runtime_profile_stream_finish(
        llama_pd_inprocess_runtime * runtime) {
    if (runtime == nullptr || runtime->llama_init == nullptr ||
            runtime->llama_init->model() == nullptr) {
        return false;
    }
    const llama_qnn_quant_profile * profile =
        runtime->llama_init->model()->get_qnn_u16_profile();
    profile = profile != nullptr ? profile : runtime->profile_transport.get();
    return profile != nullptr && profile->binary_stream_finish();
}

llama_pd_inprocess_runtime * llama_pd_inprocess_runtime_create(
        int argc,
        char ** argv,
        const void * model_data,
        size_t model_size,
        double * initialization_ms) {
    const auto init_start = steady_clock::now();
    double model_ms = 0.0;
    std::unique_ptr<llama_pd_inprocess_runtime> runtime(
        llama_pd_inprocess_runtime_create_model_only(
            argc, argv, model_data, model_size, &model_ms));
    if (runtime == nullptr) {
        return nullptr;
    }
    double context_ms = 0.0;
    double warmup_ms = 0.0;
    if (!llama_pd_inprocess_runtime_prepare_context(
            runtime.get(), &context_ms) ||
        !llama_pd_inprocess_runtime_warmup(
            runtime.get(), &warmup_ms)) {
        return nullptr;
    }
    runtime->initialization_ms = elapsed_ms(init_start);
    if (initialization_ms != nullptr) {
        *initialization_ms = runtime->initialization_ms;
    }
    LOG_INF(
        "PD persistent in-process runtime ready: initialization_ms=%.3f "
        "model_ms=%.3f context_ms=%.3f warmup_ms=%.3f\n",
        runtime->initialization_ms,
        model_ms,
        context_ms,
        warmup_ms);
    return runtime.release();
}

int llama_pd_inprocess_runtime_run(
        llama_pd_inprocess_runtime * runtime,
        const llama_pd_inprocess_request * request) {
    const bool packed_handoff = request != nullptr &&
        request->handoff_data != nullptr && request->handoff_size != 0;
    const bool split_fp16_handoff = request != nullptr &&
        request->prompt_tokens != nullptr && request->kv_fp16 != nullptr &&
        request->kv_fp16_values != 0;
    if (runtime == nullptr || request == nullptr ||
        (!packed_handoff && !split_fp16_handoff)) {
        return 1;
    }
    if (request->result != nullptr) {
        *request->result = {};
    }
    g_inprocess_runtime = runtime;
    g_inprocess_request = request;
    // The runtime was created with the same argv; parsing it again only
    // constructs per-turn sampling/profiling state and does not reload model,
    // metadata, context, KV allocation, or scheduler buffers.
    std::vector<char *> argv;
    argv.reserve(runtime->args.size() + 1);
    for (std::string & arg : runtime->args) {
        argv.push_back(arg.data());
    }
    argv.push_back(nullptr);
    const int result = llama_pd_cli_main_impl(
        static_cast<int>(runtime->args.size()), argv.data());
    g_inprocess_request = nullptr;
    g_inprocess_runtime = nullptr;
    return result;
}

void llama_pd_inprocess_runtime_destroy(
        llama_pd_inprocess_runtime * runtime) {
    delete runtime;
}

int llama_pd_cli_run_inprocess(
        int argc,
        char ** argv,
        const llama_pd_inprocess_request * request) {
    if (request == nullptr || request->model_data == nullptr ||
        request->model_size == 0) {
        return 1;
    }
    double initialization_ms = 0.0;
    llama_pd_inprocess_runtime * runtime =
        llama_pd_inprocess_runtime_create(
            argc,
            argv,
            request->model_data,
            request->model_size,
            &initialization_ms);
    if (runtime == nullptr) {
        return 1;
    }
    const int result = llama_pd_inprocess_runtime_run(runtime, request);
    llama_pd_inprocess_runtime_destroy(runtime);
    return result;
}

#ifndef LLAMA_PD_CLI_NO_MAIN
int main(int argc, char ** argv) {
    return llama_pd_cli_main_impl(argc, argv);
}
#endif
