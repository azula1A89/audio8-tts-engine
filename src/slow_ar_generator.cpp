#include <slow_ar_generator.hpp>
#include <onnxruntime_cxx_api.h>
#include <fmt/ranges.h>
#include <map>


static const char* input_names[] = {"codes", "input_pos", "cache_key_0", "cache_value_0", "cache_key_1", "cache_value_1", "cache_key_2", "cache_value_2", "cache_key_3", "cache_value_3", "cache_key_4", "cache_value_4", "cache_key_5", "cache_value_5", "cache_key_6", "cache_value_6", "cache_key_7", "cache_value_7", "cache_key_8", "cache_value_8", "cache_key_9", "cache_value_9", "cache_key_10", "cache_value_10", "cache_key_11", "cache_value_11", "cache_key_12", "cache_value_12", "cache_key_13", "cache_value_13", "cache_key_14", "cache_value_14", "cache_key_15", "cache_value_15", "cache_key_16", "cache_value_16", "cache_key_17", "cache_value_17", "cache_key_18", "cache_value_18", "cache_key_19", "cache_value_19", "cache_key_20", "cache_value_20", "cache_key_21", "cache_value_21", "cache_key_22", "cache_value_22", "cache_key_23", "cache_value_23"};

static const char* output_names[] = {"logits", "slow_hidden", "key_delta_0", "value_delta_0", "key_delta_1", "value_delta_1", "key_delta_2", "value_delta_2", "key_delta_3", "value_delta_3", "key_delta_4", "value_delta_4", "key_delta_5", "value_delta_5", "key_delta_6", "value_delta_6", "key_delta_7", "value_delta_7", "key_delta_8", "value_delta_8", "key_delta_9", "value_delta_9", "key_delta_10", "value_delta_10", "key_delta_11", "value_delta_11", "key_delta_12", "value_delta_12", "key_delta_13", "value_delta_13", "key_delta_14", "value_delta_14", "key_delta_15", "value_delta_15", "key_delta_16", "value_delta_16", "key_delta_17", "value_delta_17", "key_delta_18", "value_delta_18", "key_delta_19", "value_delta_19", "key_delta_20", "value_delta_20", "key_delta_21", "value_delta_21", "key_delta_22", "value_delta_22", "key_delta_23", "value_delta_23"};

class SlowARGenerator::Impl
{
public:

    Impl( Ort::Env& env, const std::filesystem::path& path, const RuntimeConfig& config) :
        env_(env), path_(path), session_(nullptr), initialized_(false) {}
    
    void reset_kvcache() {
        kv_cache_.clear();
        
        for (int i = 0; i < audio8::NUM_LAYERS; i++) {
            auto ck = fmt::format("cache_key_{}", i);
            auto cv = fmt::format("cache_value_{}", i);

            kv_cache_[ck] = Tensor<Ort::Float16_t>();
            kv_cache_[cv] = Tensor<Ort::Float16_t>();

            kv_cache_[ck].shape = { 1, 2, 2048, 64};
            kv_cache_[cv].shape = { 1, 2, 2048, 64};

            kv_cache_[ck].data.resize(262144UL, Ort::Float16_t());
            kv_cache_[cv].data.resize(262144UL, Ort::Float16_t());
        }
     }

    bool initialize() {
        reset_kvcache();
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetIntraOpNumThreads(4);

        session_ = std::make_unique<Ort::Session>(env_, path_.c_str(), options);
        initialized_ = session_ != nullptr;
        return initialized_;
    }

    int generate_next( SlowARInput& input, SlowAROutput& state) {
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
                50, 
                output_names, 
                50
            );

            update_state(output_values, state);
            update_kv_cache(output_values, input.input_pos);
        } catch(const Ort::Exception& exception) {
            fmt::print("SlowARGenerator: {}\n", exception.what());
            return -1;
        }

        return 0;
    }

private:
    void make_input_values(SlowARInput& input, std::vector<Ort::Value>& input_values) {
        auto memory_info = Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeDefault);
        input_values.resize(session_->GetInputCount());
        input_values[0] = Ort::Value::CreateTensor<int64_t>(
            memory_info, 
            input.codes.ptr(), 
            input.codes.size(), 
            input.codes.shape.data(), 
            input.codes.shape.size());

        input_values[1] = Ort::Value::CreateTensor<int64_t>(
            memory_info, 
            input.input_pos.ptr(), 
            input.input_pos.size(), 
            input.input_pos.shape.data(), 
            input.input_pos.shape.size());

        for (int i = 0; i < audio8::NUM_LAYERS; i++) {
            int ki = (i + 1)*2;
            int vi = (i + 1)*2 + 1;
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

    void update_state( const std::vector<Ort::Value>& output_values, SlowAROutput& state) {
        const float* l_ptr = output_values[0].GetTensorData<float>();
        size_t l_bytes = output_values[0].GetTensorSizeInBytes();
        
        const Ort::Float16_t* s_ptr = output_values[1].GetTensorData<Ort::Float16_t>();
        size_t s_bytes = output_values[1].GetTensorSizeInBytes();

        state.logits.shape = output_values[0].GetTensorTypeAndShapeInfo().GetShape();
        state.slow_hidden.shape = output_values[1].GetTensorTypeAndShapeInfo().GetShape();

        state.logits.data.clear();
        state.slow_hidden.data.clear();
        
        state.logits.data.resize(l_bytes / sizeof(float));
        state.slow_hidden.data.resize(s_bytes / sizeof(int16_t));

        memcpy(state.logits.ptr(), l_ptr, l_bytes);
        memcpy(state.slow_hidden.ptr(), s_ptr, s_bytes);
    }

    void update_kv_cache( const std::vector<Ort::Value>& output_values, const Tensor<int64_t>& position ) {
        for (int i = 0; i < audio8::NUM_LAYERS; i++) {
            int ki = (i + 1)*2;
            int vi = (i + 1)*2 + 1;
            auto& ck = kv_cache_[fmt::format("cache_key_{}", i)];
            auto& cv = kv_cache_[fmt::format("cache_value_{}", i)];

            const Ort::Float16_t* k_ptr = output_values[ki].GetTensorData<Ort::Float16_t>();
            const Ort::Float16_t* v_ptr = output_values[vi].GetTensorData<Ort::Float16_t>();

            auto kd_shape = output_values[ki].GetTensorTypeAndShapeInfo().GetShape();
            auto vd_shape = output_values[vi].GetTensorTypeAndShapeInfo().GetShape();

            update_kvcache_item(ck, k_ptr, kd_shape, position);
            update_kvcache_item(cv, v_ptr, vd_shape, position);
        }
        // fmt::print("slowar cache-shape: {} update pos: {}\n", kv_cache_["cache_key_0"].shape, position.data);
        // fmt::print("slowar delta-shape: {} update pos: {}\n", kv_cache_["key_delta_0"].shape, position.data);
    }

    void update_kvcache_item(Tensor<Ort::Float16_t>& cache, 
        const Ort::Float16_t* delta,
        const std::vector<int64_t>& delta_shape,
        const Tensor<int64_t>& position_in) {
        size_t batch_size = cache.shape[0];      // dim 0
        size_t num_heads = cache.shape[1];       // dim 1
        size_t max_seq_len = cache.shape[2];     // dim 2
        size_t head_dim = cache.shape[3];        // dim 3
        size_t delta_batch = delta_shape[0];    // dim 0
        size_t delta_heads = delta_shape[1];    // dim 1
        size_t delta_seq_len = delta_shape[2];  // dim 2

        auto& positions = position_in.data;
        
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

        // Positions are normally consecutive; copy a whole sequence in that
        // case, while retaining the indexed path for arbitrary positions.
        for (size_t b = 0; b < delta_batch; b++) {
            for (size_t h = 0; h < delta_heads; h++) {
                const size_t delta_base = b * delta_stride_b + h * delta_stride_h;

                bool contiguous = !positions.empty();
                for (size_t pos_idx = 1; pos_idx < positions.size(); pos_idx++) {
                    if (positions[pos_idx] != positions[0] + static_cast<int64_t>(pos_idx)) {
                        contiguous = false;
                        break;
                    }
                }

                if (contiguous) {
                    const size_t cache_idx = get_linear_index(
                        b, h, static_cast<size_t>(positions[0]), 0);
                    std::memcpy(cache.data.data() + cache_idx,
                                delta + delta_base,
                                positions.size() * head_dim * sizeof(Ort::Float16_t));
                    continue;
                }

                for (size_t pos_idx = 0; pos_idx < positions.size(); pos_idx++) {
                    const size_t cache_idx = get_linear_index(
                        b, h, static_cast<size_t>(positions[pos_idx]), 0);
                    std::memcpy(cache.data.data() + cache_idx,
                                delta + delta_base + pos_idx * delta_stride_s,
                                head_dim * sizeof(Ort::Float16_t));
                }
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


SlowARGenerator::SlowARGenerator(
        void* env,
        const Audio8ModelPaths& paths,
        const RuntimeConfig& config) : 
        pImpl{ std::make_unique<Impl>(*((Ort::Env*)env), paths.slow_ar, config) } { }

SlowARGenerator::~SlowARGenerator() = default;
bool SlowARGenerator::init() {
    return pImpl->initialize();
}

void SlowARGenerator::reset_kvcache() {
    pImpl->reset_kvcache();
};

int SlowARGenerator::generate_next( SlowARInput& input, SlowAROutput& state) {
    return pImpl->generate_next(input, state);
}