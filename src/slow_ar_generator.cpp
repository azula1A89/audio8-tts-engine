/*
MIT License

Copyright (c) [2026] [azula1A89]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <slow_ar_generator.hpp>
#include <onnxruntime_cxx_api.h>
#include <fmt/ranges.h>
#include <map>


static const char* input_names[] = {"codes", "position", "cache_keys", "cache_values", "conv_states", "ssm_states"};

static const char* output_names[] = {"logits", "hidden", "key_delta", "value_delta", "next_conv_states", "next_ssm_states"};

class SlowARGenerator::Impl
{
private:
    Ort::Env& env_;
    const std::filesystem::path& path_;
    std::unique_ptr<Ort::Session> session_;
    std::map<std::string, Tensor<Ort::Float16_t>> kv_cache_;
    Tensor<float> k_cache_; // 24 * 1 * 2 * 2048 * 64 = 6291456
    Tensor<float> v_cache_; // 24 * 1 * 2 * 2048 * 64 = 6291456
    Tensor<float> conv_states_; //24 * 1 * 896 * 4 = 86016
    Tensor<float> ssm_states_; // 24 * 1 * 24 * 32 * 64 = 1179648
    bool initialized_;

public:

    Impl( Ort::Env& env, const std::filesystem::path& path, const RuntimeConfig& config) :
        env_(env), path_(path), session_(nullptr), initialized_(false) {}
    
    void reset_kvcache() {
        k_cache_.data.resize(24 * 1 * 2 * 2048 * 64, 0.0f);
        v_cache_.data.resize(24 * 1 * 2 * 2048 * 64, 0.0f);
        conv_states_.data.resize(24 * 1 * 896 * 4, 0.0f);
        ssm_states_.data.resize(24 * 1 * 24 * 32 * 64, 0.0f);

        k_cache_.shape = { 24, 1, 2, 2048, 64 };
        v_cache_.shape = { 24, 1, 2, 2048, 64 };
        conv_states_.shape = { 24, 1, 896, 4 };
        ssm_states_.shape = { 24, 1, 24, 32, 64 };
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
                6, 
                output_names, 
                6
            );

            update_state(output_values, state);
            update_kv_cache_ssm(output_values, input.input_pos.data[0]);
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

            // "codes" [1, 11, 1]
            input_values[0] = Ort::Value::CreateTensor<int64_t>(
                memory_info, 
                input.codes.ptr(), 
                input.codes.size(), 
                input.codes.shape.data(), 
                input.codes.shape.size());

            // "position" [1]
            input_values[1] = Ort::Value::CreateTensor<int64_t>(
                memory_info, 
                input.input_pos.ptr(), 
                input.input_pos.size(), 
                input.input_pos.shape.data(), 
                input.input_pos.shape.size());

            // "cache_keys"
            input_values[2] = Ort::Value::CreateTensor<float>( 
                memory_info, 
                k_cache_.ptr(), 
                k_cache_.size(), 
                k_cache_.shape.data(), 
                k_cache_.shape.size());

            // "cache_values"
            input_values[3] = Ort::Value::CreateTensor<float>( 
                memory_info, 
                v_cache_.ptr(), 
                v_cache_.size(), 
                v_cache_.shape.data(), 
                v_cache_.shape.size());

            // "conv_states"
            input_values[4] = Ort::Value::CreateTensor<float>( 
                memory_info, 
                conv_states_.ptr(), 
                conv_states_.size(), 
                conv_states_.shape.data(), 
                conv_states_.shape.size());

            // "ssm_states"
            input_values[5] = Ort::Value::CreateTensor<float>( 
                memory_info, 
                ssm_states_.ptr(), 
                ssm_states_.size(), 
                ssm_states_.shape.data(), 
                ssm_states_.shape.size());
    }

    void update_state( const std::vector<Ort::Value>& output_values, SlowAROutput& state) {
        const float* l_ptr = output_values[0].GetTensorData<float>();
        size_t l_bytes = output_values[0].GetTensorSizeInBytes();
        
        const float* s_ptr = output_values[1].GetTensorData<float>();
        size_t s_bytes = output_values[1].GetTensorSizeInBytes();

        state.logits.shape = output_values[0].GetTensorTypeAndShapeInfo().GetShape();
        state.slow_hidden.shape = output_values[1].GetTensorTypeAndShapeInfo().GetShape();

        if ( state.logits.data.size() != audio8::CODEBOOK_SIZE + 1 ) {
            state.logits.data.resize(audio8::CODEBOOK_SIZE + 1);
        }

        if ( state.slow_hidden.size() != audio8::HIDDEN_SIZE ) {
            state.slow_hidden.data.resize(audio8::HIDDEN_SIZE);
        }

        memcpy(state.logits.ptr(), l_ptr, l_bytes);
        memcpy(state.slow_hidden.ptr(), s_ptr, s_bytes);
    }

    void update_kv_cache_ssm( const std::vector<Ort::Value>& output_values, const int64_t& position ) {
        const float* key_delta_ptr = output_values[2].GetTensorData<float>();
        const float* value_delta_ptr = output_values[3].GetTensorData<float>();

        update_kv_cache_slice(k_cache_.data, key_delta_ptr, position);
        update_kv_cache_slice(v_cache_.data, value_delta_ptr, position);

        const float* conv_states_ptr = output_values[4].GetTensorData<float>();
        size_t c_bytes = output_values[4].GetTensorSizeInBytes();
        const float* ssm_states_ptr = output_values[5].GetTensorData<float>();
        size_t s_bytes = output_values[5].GetTensorSizeInBytes();

        std::memcpy(conv_states_.ptr(), conv_states_ptr, c_bytes);
        std::memcpy(ssm_states_.ptr(), ssm_states_ptr, s_bytes);
    }

    /**
    * @brief 将 ONNX 推理输出的 key_delta / value_delta 切片写入全局 cache_keys / cache_values
    * 
    * @param cache        全局缓存张量内存指针 (Shape:)
    * @param delta        单步输出增量内存指针 (Shape:)
    * @param position     当前推导步的位置索引 (0 <= position < 2048)
    * @param num_layers   层数，默认 24
    * @param num_heads    Attention KV Head 数量，默认 2
    * @param max_seq_len  最大序列长度，默认 2048
    * @param head_dim     Head 维度，默认 64
    */
    void update_kv_cache_slice(
        std::vector<float>& cache,
        const float* delta,
        int64_t position,
        size_t num_layers = 24,
        size_t num_heads = 2,
        size_t max_seq_len = 2048,
        size_t head_dim = 64) 
    {
        // 边界安全检查
        if (position < 0 || static_cast<size_t>(position) >= max_seq_len) {
            throw std::out_of_range("Position exceeds max sequence length in KV Cache.");
        }

        // 各种维度的 Stride (元素个数) 计算：
        // cache 维度:
        const size_t cache_layer_stride = 1 * num_heads * max_seq_len * head_dim; // 2 * 2048 * 64 = 262,144
        const size_t cache_head_stride  = max_seq_len * head_dim;                 // 2048 * 64 = 131,072
        const size_t cache_pos_stride   = head_dim;                               // 64

        // delta 维度:
        const size_t delta_layer_stride = 1 * num_heads * head_dim;               // 2 * 64 = 128
        const size_t delta_head_stride  = head_dim;                               // 64

        const size_t copy_bytes = head_dim * sizeof(float);                       // 64 * 4 = 256 字节

        // 遍历每一层 (num_layers=24) 和每个 KV Head (num_heads=2)
        for (size_t l = 0; l < num_layers; ++l) {
            for (size_t h = 0; h < num_heads; ++h) {
                // 计算在 delta 数据区中的源偏移量
                size_t delta_offset = l * delta_layer_stride + h * delta_head_stride;

                // 计算在 cache 数据区中 target position 处的绝对偏移量
                size_t cache_offset = l * cache_layer_stride + h * cache_head_stride + position * cache_pos_stride;

                // 一次性批量复制该 Head 在当前 position 的 64 个 float 元素
                std::memcpy(cache.data() + cache_offset, delta + delta_offset, copy_bytes);
            }
        }
    }
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