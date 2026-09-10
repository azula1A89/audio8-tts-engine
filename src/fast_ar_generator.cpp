#include <fast_ar_generator.hpp>
#include <onnxruntime_cxx_api.h>
#include <fmt/ranges.h>
#include <map>


static const char* input_names[] = {"slow_hidden", "token_id", "use_slow_hidden", "input_pos", "cache_key_0", "cache_value_0", "cache_key_1", "cache_value_1", "cache_key_2", "cache_value_2", "cache_key_3", "cache_value_3"};

static const char* output_names[] = {"logits", "key_delta_0", "value_delta_0", "key_delta_1", "value_delta_1", "key_delta_2", "value_delta_2", "key_delta_3", "value_delta_3"};

class FastARGenerator::Impl
{
public:

    Impl( Ort::Env& env, const std::filesystem::path& path, const RuntimeConfig& config) :
        env_(env), path_(path), session_(nullptr), initialized_(false) {}

    void reset_kvcache() {
        kv_cache_.clear();
        
        for (int i = 0; i < audio8::NUM_FAST_LAYERS; i++) {
            auto ck = fmt::format("cache_key_{}", i);
            auto cv = fmt::format("cache_value_{}", i);

            kv_cache_[ck] = Tensor<Ort::Float16_t>();
            kv_cache_[cv] = Tensor<Ort::Float16_t>();

            kv_cache_[ck].shape = { 1, 2, 10, 64};
            kv_cache_[cv].shape = { 1, 2, 10, 64};

            kv_cache_[ck].data.resize(1280UL, Ort::Float16_t());
            kv_cache_[cv].data.resize(1280UL, Ort::Float16_t());
        }
     }

    bool initialize() {
        reset_kvcache();
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetIntraOpNumThreads(1);

        session_ = std::make_unique<Ort::Session>(env_, path_.c_str(), options);
        initialized_ = session_ != nullptr;
        return initialized_;
    }

    int generate_next( FastARInput& input, FastAROutput& state) {
        if ( !initialized_ ) {
            initialized_ = initialize();
            if ( !initialized_ ) return -1;
        }
        
        try {
            std::vector<Ort::Value> input_values;
            make_input_values( input, input_values);

            std::vector<Ort::Value> output_values = session_->Run(
                Ort::RunOptions{nullptr}, 
                input_names, 
                input_values.data(), 
                12, 
                output_names, 
                9
            );

            update_state(output_values, state);
            update_kv_cache(output_values, input.input_pos.data[0]);
        } catch(const Ort::Exception& exception) {
            fmt::print("FastARGenerator: {}\n", exception.what());
            return -1;
        }

        return 0;
    }

private:
    void make_input_values(FastARInput& input, std::vector<Ort::Value>& input_values) {
        auto memory_info = Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeDefault);
        
        input_values.resize(session_->GetInputCount());
        input_values[0] = Ort::Value::CreateTensor<Ort::Float16_t>(
            memory_info,
            reinterpret_cast<Ort::Float16_t*>(input.slow_hidden.ptr()),
            input.slow_hidden.size(),
            input.slow_hidden.shape.data(),
            input.slow_hidden.shape.size()
        );

        input_values[1] = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            input.token_id.ptr(),
            input.token_id.size(),
            input.token_id.shape.data(),
            input.token_id.shape.size()
        );

        input_values[2] = Ort::Value::CreateTensor<bool>(
            memory_info,
            reinterpret_cast<bool*>(input.use_slow_hidden.ptr()),
            input.use_slow_hidden.size(),
            input.use_slow_hidden.shape.data(),
            input.use_slow_hidden.shape.size()
        );

        input_values[3] = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            input.input_pos.ptr(),
            input.input_pos.size(),
            input.input_pos.shape.data(),
            input.input_pos.shape.size()
        );

        for (int i = 0; i < audio8::NUM_FAST_LAYERS; i++) {
            int ki = (i + 2)*2;
            int vi = (i + 2)*2 + 1;
            auto& key = kv_cache_[fmt::format("cache_key_{}", i)];
            auto& value = kv_cache_[fmt::format("cache_value_{}", i)];
            input_values[ki] = Ort::Value::CreateTensor < Ort::Float16_t > (
                memory_info, 
                key.ptr(), 
                key.size(), 
                key.shape.data(), 
                key.shape.size());

            input_values[vi] = Ort::Value::CreateTensor < Ort::Float16_t > (
                memory_info, 
                value.ptr(), 
                value.size(), 
                value.shape.data(), 
                value.shape.size());
        }
    }

    void update_state( const std::vector<Ort::Value>& output_values, FastAROutput& state) {
        const float* logits = output_values[0].GetTensorData<float>();
        size_t logits_bytes = output_values[0].GetTensorSizeInBytes();
        state.logits.shape = output_values[0].GetTensorTypeAndShapeInfo().GetShape();
        state.logits.data.resize(logits_bytes / sizeof(float));
        memcpy(state.logits.ptr(), logits, logits_bytes);
    }

    void update_kv_cache( const std::vector<Ort::Value>& output_values, const int64_t& position ) {
        for (int i = 0; i < audio8::NUM_FAST_LAYERS; i++) {
            int ki = i*2 + 1;
            int vi = i*2 + 2;
            auto& ck = kv_cache_[fmt::format("cache_key_{}", i)];
            auto& cv = kv_cache_[fmt::format("cache_value_{}", i)];

            const Ort::Float16_t* k_ptr = output_values[ki].GetTensorData<Ort::Float16_t>();
            const Ort::Float16_t* v_ptr = output_values[vi].GetTensorData<Ort::Float16_t>();

            auto kd_shape = output_values[ki].GetTensorTypeAndShapeInfo().GetShape();
            auto vd_shape = output_values[vi].GetTensorTypeAndShapeInfo().GetShape();

            update_kvcache_item(ck, k_ptr, kd_shape, position);
            update_kvcache_item(cv, v_ptr, vd_shape, position);
        }
        // fmt::print("fastar cache-shape: {} update pos: {}\n", kv_cache_["cache_key_0"].shape, position);
        // fmt::print("fastar delta-shape: {} update pos: {}\n", kv_cache_["key_delta_0"].shape, position);
    }

    void update_kvcache_item(Tensor<Ort::Float16_t>& data, 
        const Ort::Float16_t* delta,
        const std::vector<int64_t>& delta_shape,
        const int64_t& position) {
        size_t batch_size = data.shape[0];      // dim 0
        size_t num_heads = data.shape[1];       // dim 1
        size_t max_seq_len = data.shape[2];     // dim 2
        size_t head_dim = data.shape[3];        // dim 3
        size_t delta_batch = delta_shape[0];    // dim 0
        size_t delta_heads = delta_shape[1];    // dim 1
        size_t delta_seq_len = delta_shape[2];  // dim 2
        
        // delta shape: [delta_batch, delta_heads, positions.size(), head_dim]
        size_t delta_stride_b = delta_heads * delta_seq_len * head_dim;
        size_t delta_stride_h = delta_seq_len * head_dim;
        size_t delta_stride_s = head_dim;

        auto get_linear_index = [num_heads, max_seq_len, head_dim](size_t b, size_t h, size_t s, size_t d) -> size_t {
                return b * (num_heads * max_seq_len * head_dim) +
                    h * (max_seq_len * head_dim) +
                    s * head_dim +
                    d;
        };

        // 遍历delta中的每个元素，将其写入缓存的对应位置
        for (size_t b = 0; b < delta_batch; b++) {
            for (size_t h = 0; h < delta_heads; h++) {
                int64_t seq_pos = position;  // 缓存中的实际位置
                // 由于delta的seq_len维度为1，我们可以直接复制整个head_dim的块
                size_t delta_idx = b * delta_stride_b + 
                                    h * delta_stride_h +
                                    0 * delta_stride_s; // seq_len维度为1，所以索引为0
                size_t cache_idx = get_linear_index(b, h, seq_pos, 0); // seq_len维度为1，所以索引为0
                memcpy(&data.data[cache_idx], delta + delta_idx, head_dim * sizeof(Ort::Float16_t));
            }
        }
    }
private:

    Ort::Env& env_;
    const std::filesystem::path& path_;
    std::unique_ptr<Ort::Session> session_;
    std::map<std::string, Tensor<Ort::Float16_t>> kv_cache_;
    bool initialized_;
};


FastARGenerator::FastARGenerator(
        void* env,
        const Audio8ModelPaths& paths,
        const RuntimeConfig& config) : 
        pImpl{ std::make_unique<Impl>(*((Ort::Env*)env), paths.fast_ar, config) } { }

FastARGenerator::~FastARGenerator() = default;

bool FastARGenerator::init() {
    return pImpl->initialize();
}

void FastARGenerator::reset_kvcache() {
    pImpl->reset_kvcache();
};

int FastARGenerator::generate_next( FastARInput& input, FastAROutput& state) {
    return pImpl->generate_next(input, state);
}