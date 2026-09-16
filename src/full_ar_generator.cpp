#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <full_ar_generator.hpp>
#include <fmt/core.h>
#include <sampler.hpp>
static const char* slow_input_names[] = { "codes", "input_pos", "cache_key_0", "cache_value_0", "cache_key_1", "cache_value_1", "cache_key_2", "cache_value_2", "cache_key_3", "cache_value_3", "cache_key_4", "cache_value_4", "cache_key_5", "cache_value_5", "cache_key_6", "cache_value_6", "cache_key_7", "cache_value_7", "cache_key_8", "cache_value_8", "cache_key_9", "cache_value_9", "cache_key_10", "cache_value_10", "cache_key_11", "cache_value_11", "cache_key_12", "cache_value_12", "cache_key_13", "cache_value_13", "cache_key_14", "cache_value_14", "cache_key_15", "cache_value_15", "cache_key_16", "cache_value_16", "cache_key_17", "cache_value_17", "cache_key_18", "cache_value_18", "cache_key_19", "cache_value_19", "cache_key_20", "cache_value_20", "cache_key_21", "cache_value_21", "cache_key_22", "cache_value_22", "cache_key_23", "cache_value_23" };
static const char* slow_output_names[] = { "logits", "slow_hidden", "key_delta_0", "value_delta_0", "key_delta_1", "value_delta_1", "key_delta_2", "value_delta_2", "key_delta_3", "value_delta_3", "key_delta_4", "value_delta_4", "key_delta_5", "value_delta_5", "key_delta_6", "value_delta_6", "key_delta_7", "value_delta_7", "key_delta_8", "value_delta_8", "key_delta_9", "value_delta_9", "key_delta_10", "value_delta_10", "key_delta_11", "value_delta_11", "key_delta_12", "value_delta_12", "key_delta_13", "value_delta_13", "key_delta_14", "value_delta_14", "key_delta_15", "value_delta_15", "key_delta_16", "value_delta_16", "key_delta_17", "value_delta_17", "key_delta_18", "value_delta_18", "key_delta_19", "value_delta_19", "key_delta_20", "value_delta_20", "key_delta_21", "value_delta_21", "key_delta_22", "value_delta_22", "key_delta_23", "value_delta_23" };
static const char* fast_input_names[] = { "slow_hidden", "token_id", "use_slow_hidden", "input_pos", "cache_key_0", "cache_value_0", "cache_key_1", "cache_value_1", "cache_key_2", "cache_value_2", "cache_key_3", "cache_value_3" };
static const char* fast_output_names[] = { "logits", "key_delta_0", "value_delta_0", "key_delta_1", "value_delta_1", "key_delta_2", "value_delta_2", "key_delta_3", "value_delta_3" };

// ============================================================================
// 1. RAII 辅助工具：管理显存指针与 Ort::Value 包装
// ============================================================================
struct CudaDeleter {
    void operator()(void* ptr) const {
        if (ptr) ::cudaFree(ptr);
    }
};
using UniqueCudaPtr = std::unique_ptr<void, CudaDeleter>;

struct GpuBuffer {
    UniqueCudaPtr gpu_ptr;
    size_t element_count{ 0 };
    size_t byte_size{ 0 };
    std::vector<int64_t> shape;
    ONNXTensorElementDataType element_type;

    GpuBuffer() = default;

    static GpuBuffer create(const std::vector<int64_t>& shape,
        ONNXTensorElementDataType type,
        cudaStream_t stream = nullptr) {
        GpuBuffer buf;
        buf.shape = shape;
        buf.element_type = type;
        buf.element_count = 1;
        for (auto dim : shape) buf.element_count *= dim;

        size_t elem_bytes = 0;
        switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: elem_bytes = sizeof(uint16_t); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   elem_bytes = sizeof(float); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   elem_bytes = sizeof(int64_t); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    elem_bytes = sizeof(bool); break;
        default: throw std::runtime_error("Unsupported ONNX data type");
        }
        buf.byte_size = buf.element_count * elem_bytes;

        void* ptr = nullptr;
        cudaError_t err = ::cudaMalloc(&ptr, buf.byte_size);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed: ") + ::cudaGetErrorString(err));
        }
        buf.gpu_ptr.reset(ptr);

        if (stream) {
            ::cudaMemsetAsync(buf.gpu_ptr.get(), 0, buf.byte_size, stream);
        }
        else {
            ::cudaMemset(buf.gpu_ptr.get(), 0, buf.byte_size);
        }
        return buf;
    }

    // 封装为 Ort::Value 视图
    Ort::Value get_ort_value() const {
        Ort::MemoryInfo cuda_mem_info("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
        return Ort::Value::CreateTensor(
            cuda_mem_info,
            gpu_ptr.get(),
            byte_size,
            shape.data(),
            shape.size(),
            element_type
        );
    }

    void reset_async(cudaStream_t stream) {
        if (gpu_ptr) {
            ::cudaMemsetAsync(gpu_ptr.get(), 0, byte_size, stream);
        }
    }
};

class FullARGenerator::Impl {
public:
    static constexpr int SLOW_AR_LAYERS = 24;
    static constexpr int FAST_AR_LAYERS = 4;
    static constexpr int MAX_SEQ_LEN = 2048;
    static constexpr int NUM_HEADS = 2;
    static constexpr int HEAD_DIM = 64;
    static constexpr int HIDDEN_DIM = 896;
    static constexpr int VOCAB_SIZE = 4096;

private:

    cudaStream_t compute_stream_ = nullptr;
    Ort::Env& env_;
    const std::filesystem::path& slow_ar_path_;
    const std::filesystem::path& fast_ar_path_;
    std::atomic_bool cancel_requested_;
    bool initialized_;

    std::unique_ptr<Sampler> sampler_;

    // Sessions
    std::unique_ptr<Ort::Session> slow_session_;
    std::unique_ptr<Ort::Session> fast_session_;

    // 持久化静态 IoBindings
    std::unique_ptr<Ort::IoBinding> slow_io_binding_;
    std::unique_ptr<Ort::IoBinding> fast_io_binding_;

    // ------------------------------------------------------------------------
    // 全局预分配显存池 (GPU VRAM Buffers)
    // ------------------------------------------------------------------------
    // SlowAR 与 FastAR 共享的 slow_hidden 显存块
    GpuBuffer shared_slow_hidden_buf_;

    // SlowAR 显存块
    GpuBuffer slow_input_pos_buf_;
    GpuBuffer slow_codes_buf_;
    GpuBuffer slow_logits_buf_;
    std::vector<GpuBuffer> slow_k_cache_bufs_;
    std::vector<GpuBuffer> slow_v_cache_bufs_;
    std::vector<GpuBuffer> slow_k_delta_bufs_;
    std::vector<GpuBuffer> slow_v_delta_bufs_;

    // FastAR 显存块
    GpuBuffer fast_input_token_id_buf_;
    GpuBuffer fast_input_pos_buf_;
    GpuBuffer fast_input_use_slow_hidden_buf_;
    GpuBuffer fast_logits_buf_;
    std::vector<GpuBuffer> fast_k_cache_bufs_;
    std::vector<GpuBuffer> fast_v_cache_bufs_;
    std::vector<GpuBuffer> fast_k_delta_bufs_;
    std::vector<GpuBuffer> fast_v_delta_bufs_;

public:
    Impl(Ort::Env& env, const std::filesystem::path& slow_ar_path, const std::filesystem::path& fast_ar_path, const RuntimeConfig& config) :
        env_(env), 
        slow_ar_path_(slow_ar_path), 
        fast_ar_path_(fast_ar_path), 
        cancel_requested_(false), 
        initialized_(false){};

    ~Impl() {
        if (compute_stream_) {
            ::cudaStreamSynchronize(compute_stream_);
            ::cudaStreamDestroy(compute_stream_);
        }
    }

    // initialize slow_ar fast_ar session and  GPU vram
    bool initialize() {

        sampler_ = std::make_unique<Sampler>(0.7, 0.9, 50);

        try {
            // 1. 创建共享的 CUDA Stream
            if (::cudaStreamCreate(&compute_stream_) != cudaSuccess) {
                std::cerr << "[Audio8Engine] Failed to create cuda stream." << "\n";
                return false;
            }

            // 2. 配置 ONNX Runtime Sessions 共享此 CUDA Stream
            Ort::SessionOptions session_options;
            session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            cuda_options.has_user_compute_stream = 1;
            cuda_options.user_compute_stream = static_cast<void*>(compute_stream_);
            //session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
            session_options.AppendExecutionProvider_CUDA(cuda_options);

            slow_session_ = std::make_unique<Ort::Session>(env_, slow_ar_path_.c_str(), session_options);
            fast_session_ = std::make_unique<Ort::Session>(env_, fast_ar_path_.c_str(), session_options);

            // 3. 预分配所有 GPU 显存缓冲区
            allocate_vram_pool();

            // 4. 构建并绑定静态 IoBindings
            setup_io_bindings();

            std::cout << "[Audio8Engine] GPU VRAM Memory Pool & Static IoBindings initialized successfully.\n";
            initialized_ = true;
        }
        catch (const std::exception& e) {
            std::cerr << "[Audio8Engine] Initialization failed: " << e.what() << "\n";
        }

        return initialized_;
    }

    void cancel() { cancel_requested_.store(true); }

    // 每次开始生成新音频时快速重置 KV Cache (零显存重新分配)
    void reset_slow_kv_cache() {
        for (int i = 0; i < SLOW_AR_LAYERS; ++i) {
            slow_k_cache_bufs_[i].reset_async(compute_stream_);
            slow_v_cache_bufs_[i].reset_async(compute_stream_);
        }
    }

    void reset_fast_kv_cache() {
        for (int i = 0; i < FAST_AR_LAYERS; ++i) {
            fast_k_cache_bufs_[i].reset_async(compute_stream_);
            fast_v_cache_bufs_[i].reset_async(compute_stream_);
        }
    }

    // ============================================================================
    // Step 0: 处理 Prompt (Prefill 阶段，序列长度为 prompt_len)
    // ============================================================================
    void run_slow_ar_step_0(FullARInput& in) {
        int64_t prompt_len = in.prompt_len;

        // 2. 将数据从 CPU 拷贝到预分配的 GPU VRAM 缓冲区中 (H2D 拷贝)
        cudaMemcpyAsync(slow_input_pos_buf_.gpu_ptr.get(), in.slow_ar.input_pos.ptr(),
            prompt_len * sizeof(int64_t), cudaMemcpyHostToDevice, compute_stream_);

        cudaMemcpyAsync(slow_codes_buf_.gpu_ptr.get(), in.slow_ar.codes.ptr(),
            in.slow_ar.codes.size() * sizeof(int64_t), cudaMemcpyHostToDevice, compute_stream_);

        // 3. 基于 GPU 显存指针，创建匹配 Step 0 实际 Shape 的 Ort::Value 视图 (0 显存分配)
        std::vector<int64_t> step0_pos_shape = { prompt_len };
        std::vector<int64_t> step0_codes_shape = { 1, 11, prompt_len };

        Ort::MemoryInfo cuda_mem_info("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);

        Ort::Value step0_pos_tensor = Ort::Value::CreateTensor<int64_t>(
            cuda_mem_info, static_cast<int64_t*>(slow_input_pos_buf_.gpu_ptr.get()),
            prompt_len, step0_pos_shape.data(), step0_pos_shape.size()
        );

        Ort::Value step0_codes_tensor = Ort::Value::CreateTensor<int64_t>(
            cuda_mem_info, static_cast<int64_t*>(slow_codes_buf_.gpu_ptr.get()),
            11 * prompt_len, step0_codes_shape.data(), step0_codes_shape.size()
        );

        // 4. 重新绑定 Step 0 专用的输入 Shape
        slow_io_binding_->BindInput("input_pos", step0_pos_tensor);
        slow_io_binding_->BindInput("codes", step0_codes_tensor);

        bind_slow_delta_outputs(in.prompt_len);

        // 5. 执行 Step 0 模型的 Prefill 计算
        try
        {
            slow_session_->Run(Ort::RunOptions{ nullptr }, *slow_io_binding_);
        }
        catch (const Ort::Exception& exception) {
            fmt::print("SlowARGenerator step0: {}\n", exception.what());
            return;
        }

        for (int i = 0; i < SLOW_AR_LAYERS; i++)
        {
            update_slow_kv_cache_prefill_batch(
                slow_k_cache_bufs_[i].gpu_ptr.get(),
                slow_k_delta_bufs_[i].gpu_ptr.get(),
                in.prompt_len
            );

            update_slow_kv_cache_prefill_batch(
                slow_v_cache_bufs_[i].gpu_ptr.get(),
                slow_v_delta_bufs_[i].gpu_ptr.get(),
                in.prompt_len
            );
        }
    }

    // ============================================================================
    // Step >= 1: 单 Step 自回归生成 (序列长度为 1)
    // ============================================================================
    void run_slow_ar_step_n(int64_t current_pos, int64_t semantic_token, const code_frame& codebooks) {
        // 1. 构建单 Token 的输入数据 (codes 为，input_pos 为)
        std::array<int64_t, 11> single_step_codes;
        single_step_codes[0] = semantic_token;
        for (int i = 0; i < 10; ++i) {
            single_step_codes[i + 1] = codebooks[i];
        }

        // 2. 将单个 pos 和 11 个 codes H2D 拷贝到 GPU 缓冲区头部
        cudaMemcpyAsync(slow_input_pos_buf_.gpu_ptr.get(), &current_pos,
            sizeof(int64_t), cudaMemcpyHostToDevice, compute_stream_);

        cudaMemcpyAsync(slow_codes_buf_.gpu_ptr.get(), single_step_codes.data(),
            11 * sizeof(int64_t), cudaMemcpyHostToDevice, compute_stream_);

        // 3. 创建单 Step Shape 的 Ort::Value 视图 (Shape 为 和)
        std::vector<int64_t> decode_pos_shape = { 1 };
        std::vector<int64_t> decode_codes_shape = { 1, 11, 1 };

        Ort::MemoryInfo cuda_mem_info("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);

        Ort::Value decode_pos_tensor = Ort::Value::CreateTensor<int64_t>(
            cuda_mem_info, static_cast<int64_t*>(slow_input_pos_buf_.gpu_ptr.get()),
            1, decode_pos_shape.data(), decode_pos_shape.size()
        );

        Ort::Value decode_codes_tensor = Ort::Value::CreateTensor<int64_t>(
            cuda_mem_info, static_cast<int64_t*>(slow_codes_buf_.gpu_ptr.get()),
            11, decode_codes_shape.data(), decode_codes_shape.size()
        );

        // 4. Re-bind 为单 Step 的 Shape
        slow_io_binding_->BindInput("input_pos", decode_pos_tensor);
        slow_io_binding_->BindInput("codes", decode_codes_tensor);

        bind_slow_delta_outputs(1);

        // 5. 执行自回归 Step 推理
        try
        {
            slow_session_->Run(Ort::RunOptions{ nullptr }, *slow_io_binding_);
        }
        catch (const Ort::Exception& exception) {
            fmt::print("SlowARGenerator step{}: {}\n", current_pos, exception.what());
            return;
        }

        for (int i = 0; i < SLOW_AR_LAYERS; i++)
        {
            update_kv_cache_2d(slow_k_cache_bufs_[i].gpu_ptr.get(), slow_k_delta_bufs_[i].gpu_ptr.get(),
                current_pos, 2, 2048, 64, compute_stream_);
            update_kv_cache_2d(slow_v_cache_bufs_[i].gpu_ptr.get(), slow_v_delta_bufs_[i].gpu_ptr.get(),
                current_pos, 2, 2048, 64, compute_stream_);
        }
    }

    void run_fast_ar_step_n(int64_t token_id, int64_t input_pos, int8_t use_slow_hidden) {

        cudaMemcpyAsync(fast_input_token_id_buf_.gpu_ptr.get(), &token_id,
            sizeof(int64_t), cudaMemcpyHostToDevice, compute_stream_);

        cudaMemcpyAsync(fast_input_pos_buf_.gpu_ptr.get(), &input_pos,
            sizeof(int64_t), cudaMemcpyHostToDevice, compute_stream_);

        cudaMemcpyAsync(fast_input_use_slow_hidden_buf_.gpu_ptr.get(), &use_slow_hidden,
            sizeof(int8_t), cudaMemcpyHostToDevice, compute_stream_);

        try
        {
            fast_session_->Run(Ort::RunOptions{ nullptr }, *fast_io_binding_);
        }
        catch (const Ort::Exception& exception) {
            fmt::print("FastARGenerator step{}: {}\n", input_pos, exception.what());
            return;
        }

        //update fast kv cache
        for (int i = 0; i < FAST_AR_LAYERS; i++)
        {
            update_kv_cache_2d(fast_k_cache_bufs_[i].gpu_ptr.get(), fast_k_delta_bufs_[i].gpu_ptr.get(),
                input_pos, 2, 10, 64, compute_stream_);
            update_kv_cache_2d(fast_v_cache_bufs_[i].gpu_ptr.get(), fast_v_delta_bufs_[i].gpu_ptr.get(),
                input_pos, 2, 10, 64, compute_stream_);
        }
    }

    void generate_frame(FullARInput& in, frame_callback callback) {
        if (!initialized_) {
            initialized_ = initialize();
            if (!initialized_) return;
        }

        cancel_requested_.store(false);

        reset_slow_kv_cache();
        run_slow_ar_step_0( in );

        std::list<int> previous;
        code_frame codebooks;
        std::vector<float> slow_logits(4097);
        std::vector<float> fast_logits(4096);

        for (int step = 0; step < in.max_new_tokens; step++) {
            if (cancel_requested_.load()) {
                return;
            }

            cudaMemcpy(
                slow_logits.data(),
                slow_logits_buf_.gpu_ptr.get(),
                sizeof(float) * slow_logits.size(),
                cudaMemcpyDeviceToHost
            );

            int semantic = sampler_->sample_semantic(slow_logits, previous);
            if (semantic == audio8::IM_END_ID) {

                // final frame
                callback(codebooks, step, true, false);
                return;
            }
            previous.push_back(semantic);
            if (previous.size() > 10) previous.pop_front();

            // fast step 0
            reset_fast_kv_cache();
            run_fast_ar_step_n(0, 0, true);

            int token = std::min(std::max(semantic - audio8::SEMATIC_BEGIN_ID, 0), audio8::CODEBOOK_SIZE - 1);
            codebooks[0] = token;

            //fast step 1-9
            for (int fast_pos = 1; fast_pos < audio8::NUM_CODEBOOKS; fast_pos++) {
                run_fast_ar_step_n(token, fast_pos, false);

                cudaMemcpy(
                    fast_logits.data(),
                    fast_logits_buf_.gpu_ptr.get(),
                    sizeof(float) * fast_logits.size(),
                    cudaMemcpyDeviceToHost
                );

                token = sampler_->sample(fast_logits);
                codebooks[fast_pos] = token;
            }

            // new frame 
            callback(codebooks, step, false, false);

            run_slow_ar_step_n(in.prompt_len + step, semantic, codebooks);
        }

        // limite reached
        callback(codebooks, in.max_new_tokens, false, true);
    }

private:

    // ============================================================================
    // 绑定 SlowAR 的 Delta 输出视图 (0 显存分配，仅配置 Shape 元数据)
    // ============================================================================
    void bind_slow_delta_outputs(size_t current_step_seq_len) {
        // Step 0 时 current_step_seq_len = prompt_len (例如 300)
        // Step >= 1 时 current_step_seq_len = 1
        std::vector<int64_t> current_delta_shape = { 1, NUM_HEADS, static_cast<int64_t>(current_step_seq_len), HEAD_DIM };

        Ort::MemoryInfo cuda_mem_info("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);

        for (int i = 0; i < SLOW_AR_LAYERS; ++i) {
            std::string k_out = "key_delta_" + std::to_string(i);
            std::string v_out = "value_delta_" + std::to_string(i);

            // 创建对应当前 Step 实际输出 Shape 的 Ort::Value 视图 (指向同一个底层 GPU 指针)
            Ort::Value k_delta_view = Ort::Value::CreateTensor<Ort::Float16_t>(
                cuda_mem_info,
                reinterpret_cast<Ort::Float16_t*>(slow_k_delta_bufs_[i].gpu_ptr.get()),
                NUM_HEADS * current_step_seq_len * HEAD_DIM,
                current_delta_shape.data(),
                current_delta_shape.size()
            );

            Ort::Value v_delta_view = Ort::Value::CreateTensor<Ort::Float16_t>(
                cuda_mem_info,
                reinterpret_cast<Ort::Float16_t*>(slow_v_delta_bufs_[i].gpu_ptr.get()),
                NUM_HEADS * current_step_seq_len * HEAD_DIM,
                current_delta_shape.data(),
                current_delta_shape.size()
            );

            // 绑定到 IoBinding
            slow_io_binding_->BindOutput(k_out.c_str(), k_delta_view);
            slow_io_binding_->BindOutput(v_out.c_str(), v_delta_view);
        }
    }

    // 分配所有的显存块
    void allocate_vram_pool() {
        std::vector<int64_t> slow_max_pos_shape = { MAX_SEQ_LEN };
        std::vector<int64_t> slow_max_codes_shape = { 1, 11, MAX_SEQ_LEN };
        std::vector<int64_t> slow_max_kv_shape = { 1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM };
        std::vector<int64_t> slow_max_delta_shape = { 1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM };
        std::vector<int64_t> slow_hidden_shape = { 1, 1, HIDDEN_DIM };
        std::vector<int64_t> slow_logits_shape = { 1, 1, VOCAB_SIZE + 1 };
        std::vector<int64_t> fast_kv_shape = { 1, NUM_HEADS, 10, HEAD_DIM };
        std::vector<int64_t> fast_delta_shape = { 1, NUM_HEADS, 1, HEAD_DIM };
        std::vector<int64_t> fast_logits_shape = { 1, 1, VOCAB_SIZE };
        std::vector<int64_t> token_id_shape = { 1, 1 };
        std::vector<int64_t> scalar_shape = { 1 };

        // A. 共享的 slow_hidden [1, 1, 896] FP16
        shared_slow_hidden_buf_ = GpuBuffer::create(slow_hidden_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, compute_stream_);

        // B. SlowAR Buffers
        slow_input_pos_buf_ = GpuBuffer::create(slow_max_pos_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, compute_stream_);
        slow_codes_buf_ = GpuBuffer::create(slow_max_codes_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, compute_stream_);
        slow_logits_buf_ = GpuBuffer::create(slow_logits_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, compute_stream_);

        slow_k_cache_bufs_.resize(SLOW_AR_LAYERS);
        slow_v_cache_bufs_.resize(SLOW_AR_LAYERS);
        slow_k_delta_bufs_.resize(SLOW_AR_LAYERS);
        slow_v_delta_bufs_.resize(SLOW_AR_LAYERS);

        for (int i = 0; i < SLOW_AR_LAYERS; ++i) {
            slow_k_cache_bufs_[i] = GpuBuffer::create(slow_max_kv_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, compute_stream_);
            slow_v_cache_bufs_[i] = GpuBuffer::create(slow_max_kv_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, compute_stream_);
            slow_k_delta_bufs_[i] = GpuBuffer::create(slow_max_delta_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, compute_stream_);
            slow_v_delta_bufs_[i] = GpuBuffer::create(slow_max_delta_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, compute_stream_);
        }

        // C. FastAR Buffers
        fast_input_token_id_buf_ = GpuBuffer::create(token_id_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, compute_stream_);
        fast_input_pos_buf_ = GpuBuffer::create(scalar_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, compute_stream_);
        fast_input_use_slow_hidden_buf_ = GpuBuffer::create(scalar_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, compute_stream_);
        fast_logits_buf_ = GpuBuffer::create(fast_logits_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, compute_stream_);

        fast_k_cache_bufs_.resize(FAST_AR_LAYERS);
        fast_v_cache_bufs_.resize(FAST_AR_LAYERS);
        fast_k_delta_bufs_.resize(FAST_AR_LAYERS);
        fast_v_delta_bufs_.resize(FAST_AR_LAYERS);

        for (int i = 0; i < FAST_AR_LAYERS; ++i) {
            fast_k_cache_bufs_[i] = GpuBuffer::create(fast_kv_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, compute_stream_);
            fast_v_cache_bufs_[i] = GpuBuffer::create(fast_kv_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, compute_stream_);
            fast_k_delta_bufs_[i] = GpuBuffer::create(fast_delta_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, compute_stream_);
            fast_v_delta_bufs_[i] = GpuBuffer::create(fast_delta_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, compute_stream_);
        }
    }

    // 绑定静态 IoBindings (包括输入、输出与 KV Cache)
    void setup_io_bindings() {
        // --------------------------------------------------------------------
        // 1. SlowAR IoBinding
        // --------------------------------------------------------------------
        slow_io_binding_ = std::make_unique<Ort::IoBinding>(*slow_session_);

        slow_io_binding_->BindInput("codes", slow_codes_buf_.get_ort_value());
        slow_io_binding_->BindInput("input_pos", slow_input_pos_buf_.get_ort_value());
        slow_io_binding_->BindOutput("logits", slow_logits_buf_.get_ort_value());

        // 核心：将 SlowAR 的输出 "slow_hidden" 绑定到共享缓冲区
        slow_io_binding_->BindOutput("slow_hidden", shared_slow_hidden_buf_.get_ort_value());

        for (int i = 0; i < SLOW_AR_LAYERS; ++i) {
            std::string k_in = "cache_key_" + std::to_string(i);
            std::string v_in = "cache_value_" + std::to_string(i);
            std::string k_out = "key_delta_" + std::to_string(i);
            std::string v_out = "value_delta_" + std::to_string(i);

            slow_io_binding_->BindInput(k_in.c_str(), slow_k_cache_bufs_[i].get_ort_value());
            slow_io_binding_->BindInput(v_in.c_str(), slow_v_cache_bufs_[i].get_ort_value());
            slow_io_binding_->BindOutput(k_out.c_str(), slow_k_delta_bufs_[i].get_ort_value());
            slow_io_binding_->BindOutput(v_out.c_str(), slow_v_delta_bufs_[i].get_ort_value());
        }

        // --------------------------------------------------------------------
        // 2. FastAR IoBinding
        // --------------------------------------------------------------------
        fast_io_binding_ = std::make_unique<Ort::IoBinding>(*fast_session_);

        fast_io_binding_->BindInput("token_id", fast_input_token_id_buf_.get_ort_value());
        fast_io_binding_->BindInput("input_pos", fast_input_pos_buf_.get_ort_value());
        fast_io_binding_->BindInput("use_slow_hidden", fast_input_use_slow_hidden_buf_.get_ort_value());
        fast_io_binding_->BindOutput("logits", fast_logits_buf_.get_ort_value());

        // 核心：将 FastAR 的输入 "slow_hidden" 直接绑定到同一个共享缓冲区 (Zero-Copy)
        fast_io_binding_->BindInput("slow_hidden", shared_slow_hidden_buf_.get_ort_value());

        for (int i = 0; i < FAST_AR_LAYERS; ++i) {
            std::string k_in = "cache_key_" + std::to_string(i);
            std::string v_in = "cache_value_" + std::to_string(i);
            std::string k_out = "key_delta_" + std::to_string(i);
            std::string v_out = "value_delta_" + std::to_string(i);

            fast_io_binding_->BindInput(k_in.c_str(), fast_k_cache_bufs_[i].get_ort_value());
            fast_io_binding_->BindInput(v_in.c_str(), fast_v_cache_bufs_[i].get_ort_value());
            fast_io_binding_->BindOutput(k_out.c_str(), fast_k_delta_bufs_[i].get_ort_value());
            fast_io_binding_->BindOutput(v_out.c_str(), fast_v_delta_bufs_[i].get_ort_value());
        }
    }

    // Step 0 专用：一次性将 prompt_len 个 Token 的 KV Delta 写入 Cache
    void update_slow_kv_cache_prefill_batch(
        void* gpu_cache_ptr,         // KV Cache 地址 (Shape:)
        const void* gpu_delta_ptr,   // Step 0 输出的 Delta 地址 (Shape: [1, 2, prompt_len, 64])
        size_t prompt_len,           // Prompt 序列长度
        int num_heads = 2,           // 头数 (SlowAR 为 2)
        int max_seq_len = 2048,       // Cache 支持的最大长度
        int head_dim = 64           // 维度
    ) {
        constexpr size_t element_size = sizeof(uint16_t); // FP16 占 2 字节

        // 每行实际需要连续拷贝的字节数：prompt_len 个 token * 64 维 * 2 字节
        size_t width = prompt_len * head_dim * element_size;

        // 源内存 Delta 中，Head 0 开头到 Head 1 开头的字节距离
        size_t spitch = prompt_len * head_dim * element_size;

        // 目标内存 Cache 中，Head 0 开头到 Head 1 开头的字节距离
        size_t dpitch = max_seq_len * head_dim * element_size;

        size_t height = static_cast<size_t>(num_heads); // 拷贝 2 个 Head

        // 一口气在 GPU 内部完成整个 Prompt 序列的 KV 复制
        cudaMemcpy2DAsync(
            gpu_cache_ptr, dpitch,  // 目标地址 (从 Pos 0 开始)
            gpu_delta_ptr, spitch,  // 源地址 (从 Delta 0 开始)
            width, height,
            cudaMemcpyDeviceToDevice,
            compute_stream_
        );
    }


    // 封装 KV Cache 更新函数
    void update_kv_cache_2d(
        void* gpu_cache_ptr,         // 指向 GPU 上该层的 KV Cache Tensor 内存
        const void* gpu_delta_ptr,   // 指向 GPU 上当前 Step 模型输出的 delta Tensor 内存
        int64_t target_position,     // 当前自回归 Step 对应的写入位置 (例如 0, 1, 2...)
        int num_heads,               // 注意力头数 (例如 SlowAR 为 2)
        int max_seq_len,             // KV Cache 支持的最大序列长度 (例如 2048)
        int head_dim,                // 每个 Head 的维度 (例如 64)
        cudaStream_t stream = nullptr // ONNX Runtime 传入或绑定的 CUDA Stream
    ) {
        // 1. 计算基础字节单位
        constexpr size_t element_size = sizeof(uint16_t); // FP16 为 2 字节
        size_t head_bytes = head_dim * element_size;      // 64 * 2 = 128 字节

        // 2. 计算源地址 (Delta) 和 目标地址 (Cache) 的指针与 Step 偏移
        // Cache 中每个 Head 占用: max_seq_len * head_bytes 字节
        // 写入的目标偏移位置: target_position * head_bytes 字节
        uint8_t* dst = static_cast<uint8_t*>(gpu_cache_ptr) + (target_position * head_bytes);
        const uint8_t* src = static_cast<const uint8_t*>(gpu_delta_ptr);

        // 3. 计算二维拷贝参数 (全部转换为字节)
        size_t width = head_bytes;                 // 每行拷贝 128 字节 (1 个 pos 的 1 个 head 数据)
        size_t dpitch = max_seq_len * head_bytes;   // 目标内存中，Header 0 到 Header 1 开头的字节距离
        size_t spitch = 1 * head_bytes;             // 源内存中，Header 0 到 Header 1 开头的字节距离
        size_t height = static_cast<size_t>(num_heads); // 总共拷贝 num_heads 行 (2 行)

        // 4. 执行 GPU 内部 2D 异步内存拷贝
        cudaError_t err = cudaMemcpy2DAsync(
            dst, dpitch,
            src, spitch,
            width, height,
            cudaMemcpyDeviceToDevice, // 完全在 GPU VRAM 内部完成
            stream
        );

        // 5. 错误检查 (开发阶段调试用)
        if (err != cudaSuccess) {
            fmt::print("cudaMemcpy2DAsync failed: {}\n", cudaGetErrorString(err));
        }
    }


};


FullARGenerator::FullARGenerator(
    void* env,
    const Audio8ModelPaths& paths,
    const RuntimeConfig& config) :
    pImpl{ std::make_unique<Impl>(*((Ort::Env*)env), paths.slow_ar, paths.fast_ar, config) } {
}

FullARGenerator::~FullARGenerator() = default;

bool FullARGenerator::init() {
    return pImpl->initialize();
}
void FullARGenerator::generate_frame( FullARInput& input, frame_callback fcb) {
    pImpl->generate_frame(input, fcb);
}

void FullARGenerator::cancle() {
    pImpl->cancel();
}
