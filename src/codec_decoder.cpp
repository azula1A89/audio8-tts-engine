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

#include <atomic>
#include <onnxruntime_cxx_api.h>
#include <fmt/core.h>
#include <codec_decoder.hpp>

static const char* input_names[] = { "codes" };
static const char* output_names[] = {"audio" };

class CodecDecoder::Impl {
public:
    Impl( Ort::Env& env, const std::filesystem::path& path, const RuntimeConfig& config) :
    env_(env), 
    path_(path), 
    session_(nullptr), 
    initialized_(false), 
    t_per_frame_ms_(0.5), 
    run_opts_(), 
    on_pcm_update_(nullptr), 
    is_inferencing_(false) {}
    
    bool initialize() {
        Ort::SessionOptions options;
        try
        {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;

            options.AppendExecutionProvider_CUDA(cuda_options);

            session_ = make_unique_nothrow<Ort::Session>(env_, path_.c_str(), options);
        }
        catch (const Ort::Exception&)
        {
            options.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_ALL);
            options.SetIntraOpNumThreads(4);

            session_ = make_unique_nothrow<Ort::Session>(env_, path_.c_str(), options);
        }
        initialized_ = session_ != nullptr;
        calibrate_performance();
        return initialized_;
    }

    bool is_running() {
        return is_inferencing_.load();
    }

    void terminate() {
        run_opts_.SetTerminate();
        if ( on_pcm_update_ ) {
            std::vector<float> empty;
            on_pcm_update_(empty);
        }
    }

    void decode_audio_batch(const std::vector<code_frame>& frames,  decoder_callback cb) {
        if (!initialized_) {
            initialized_ = initialize();
            if (!initialized_) return;
        }

        if ( cb ) {
            on_pcm_update_ = cb;
        }

        size_t T = frames.size();
        if (T == 0) {
            if ( on_pcm_update_ ) {
                std::vector<float> empty;
                on_pcm_update_(empty);
            }
            return;
        }

        is_inferencing_.store(true);

        // fmt::print("start decode audio. \n\n");
        try {
            Tensor<int64_t> input;
            input.shape = {1, audio8::NUM_CODEBOOKS, static_cast<int64_t>(T)}; // shape [1, 10, T]
            input.data.resize(10 * T);

            for (size_t t = 0; t < T; ++t) {
                for (size_t c = 0; c < audio8::NUM_CODEBOOKS; ++c) {
                    input.data[c * T + t] = frames[t][c];
                }
            }

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_value = Ort::Value::CreateTensor<int64_t>(
                memory_info,
                input.ptr(),
                input.size(),
                input.shape.data(),
                input.shape.size()
            );
            run_opts_.UnsetTerminate();
            std::vector<Ort::Value> output_values = session_->Run(
                run_opts_, 
                input_names, 
                &input_value, 
                1,
                output_names, 
                1
            );

            auto p = output_values[0].GetTensorData<float>();
            auto s = output_values[0].GetTensorSizeInBytes();
            std::vector<float> out(s / sizeof(float));
            memcpy(out.data(), p, s);

            for (float& sample : out) {
                sample = std::max(-1.0f, std::min(1.0f, sample));
            }

            if ( on_pcm_update_ ) {
                on_pcm_update_(out);
            } else {
                miniaudio_impl::wav_write(out.data(), out.size());
            }

        } catch (const Ort::Exception& exception) {
            fmt::print("CodecDecoder Batch Error: {}\n", exception.what());
            return;
        }
        is_inferencing_.store(false);
    }
    
    float estimate_decode_time_ms(size_t T) const {
        return T * t_per_frame_ms_ + 5.0;
    }

    void calibrate_performance() {
        try {
            size_t warm_up_T = 20; 
            Tensor<int64_t> dummy_input;
            dummy_input.shape = {1, audio8::NUM_CODEBOOKS, static_cast<int64_t>(warm_up_T)};
            dummy_input.data.resize(audio8::NUM_CODEBOOKS * warm_up_T, 1); 

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_value = Ort::Value::CreateTensor<int64_t>(
                memory_info, dummy_input.ptr(), dummy_input.size(), 
                dummy_input.shape.data(), dummy_input.shape.size()
            );

            auto start = std::chrono::high_resolution_clock::now();

            std::vector<Ort::Value> output_values = session_->Run(
                Ort::RunOptions{nullptr}, input_names, &input_value, 1, output_names, 1
            );

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;

            t_per_frame_ms_ = elapsed.count() / static_cast<double>(warm_up_T);

            fmt::print("[Decoder] calibrate result: {:.3f} ms/frame\n", t_per_frame_ms_);

        } catch (const std::exception& e) {

            t_per_frame_ms_ = 1.0; 
            fmt::print("[Decoder] calibrate failed: {} ms. error: {}\n", t_per_frame_ms_, e.what());
        }
    }

private:

    Ort::Env& env_;
    const std::filesystem::path& path_;
    std::unique_ptr<Ort::Session> session_;
    bool initialized_;
    double t_per_frame_ms_;
    Ort::RunOptions run_opts_;
    decoder_callback on_pcm_update_;
    std::atomic<bool> is_inferencing_; 
};


CodecDecoder::CodecDecoder(
        void* env,
        const Audio8ModelPaths& paths,
        const RuntimeConfig& config) :
        pImpl{ std::make_unique<Impl>(*((Ort::Env*)env), paths.codec_decoder, config) } {};

CodecDecoder::~CodecDecoder() = default;
bool CodecDecoder::init() {
    return pImpl->initialize();
}

bool CodecDecoder::is_running() {
    return pImpl->is_running();
}

void CodecDecoder::terminate() {
    pImpl->terminate();
}

void CodecDecoder::decode_audio_batch(const std::vector<code_frame>& frames,  decoder_callback cb) {
    pImpl->decode_audio_batch(frames, cb);
}

float CodecDecoder::estimate_decode_time_ms(size_t T) {
    return pImpl->estimate_decode_time_ms(T);
}
