#include <codec_decoder.hpp>
#include <onnxruntime_cxx_api.h>
#include <fmt/core.h>

static const char* input_names[] = { "codes" };
static const char* output_names[] = {"audio" };

class CodecDecoder::Impl {
public:
    Impl( Ort::Env& env, const std::filesystem::path& path, const RuntimeConfig& config) :
    env_(env), path_(path), session_(nullptr), initialized_(false), t_per_frame_ms_(0.5) {}
    
    bool initialize() {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetIntraOpNumThreads(4);

        session_ = std::make_unique<Ort::Session>(env_, path_.c_str(), options);
        initialized_ = session_ != nullptr;
        calibrate_performance();
        return initialized_;
    }

    void decode_audio_batch(const std::vector<code_frame>& frames,  decoder_callback cb) {
        if (!initialized_) {
            initialized_ = initialize();
            if (!initialized_) return;
        }

        size_t T = frames.size();
        if (T == 0) return;

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

            std::vector<Ort::Value> output_values = session_->Run(
                Ort::RunOptions{nullptr}, 
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

            if ( cb ) {
                cb(out.data(), out.size());
            }

            miniaudio_impl::wav_write(out.data(), out.size());

        } catch (const Ort::Exception& exception) {
            fmt::print("CodecDecoder Batch Error: {}\n", exception.what());
            return;
        }
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

void CodecDecoder::decode_audio_batch(const std::vector<code_frame>& frames,  decoder_callback cb) {
    pImpl->decode_audio_batch(frames, cb);
}

float CodecDecoder::estimate_decode_time_ms(size_t T) {
    return pImpl->estimate_decode_time_ms(T);
}
