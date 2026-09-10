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

#include <audio8_engine.hpp>
#include <filesystem>
#include <atomic>
#include <text_processor.hpp>
#include <prompt_builder.hpp>
#include <slow_ar_generator.hpp>
#include <fast_ar_generator.hpp>
#include <codec_decoder.hpp>
#include <voice_manager.hpp>
#include <sampler.hpp>
#include <array>
#include <onnxruntime_cxx_api.h>
#include <fmt/color.h>
#include <miniaudio_impl.hpp>

using CodecFrame = std::array<int64_t, audio8::NUM_CODEBOOKS>;

namespace miniaudio_impl {
    MiniAudio miniaudio;
    void play() { miniaudio.play(); }
    void stop() { miniaudio.stop(); }
    std::vector<float> load_audio( const std::string& path ) { return  miniaudio.load_audio(path); };
    void wav_write(const float *buff, uint64_t count) { miniaudio.wav_write(buff, count); };
}

class Audio8Engine::Impl
{
public:
    Impl() :
        paths_(),
        env_(nullptr),
        voice_manager_(nullptr),
        prompt_builder_(nullptr),
        slow_ar_(nullptr),
        fast_ar_(nullptr),
        codec_decoder_(nullptr),
        sampler_(nullptr),
        cancel_requested_(false),
        initialized_(false) {};

    ~Impl() {};

    bool initialize(const std::filesystem::path& model_dir) {
        RuntimeConfig cfg;
        if( !model_dir.empty() ) {
            paths_ = Audio8ModelPaths(model_dir);
        }
        initialized_ = paths_.check();
        if ( !initialized_ ) {
            fmt::print(fmt::emphasis::bold | fg(fmt::color::orange_red), 
            "[{}] is not a vaild model path.\n", paths_.root.string());
        }

        env_ = std::make_unique<Ort::Env>( ORT_LOGGING_LEVEL_ERROR, "Audio8Engine");
        initialized_ &= (env_ != nullptr);

        voice_manager_ = std::make_unique<VoiceManager>(
            env_.get(),
            paths_,
            cfg
        );
        initialized_ &= (voice_manager_ != nullptr);

        prompt_builder_ = std::make_unique<PromptBuilder>(
            paths_.tokenizer, 
            audio8::SEMATIC_BEGIN_ID, 
            audio8::NUM_CODEBOOKS
        );
        initialized_ &= (prompt_builder_ != nullptr);

        slow_ar_ = std::make_unique<SlowARGenerator>(
            env_.get(),
            paths_,
            cfg
        );
        initialized_ &= (slow_ar_ != nullptr);
        
        fast_ar_ = std::make_unique<FastARGenerator>(
            env_.get(),
            paths_,
            cfg
        );
        initialized_ &= (fast_ar_ != nullptr);

        codec_decoder_ = std::make_unique<CodecDecoder>(
            env_.get(),
            paths_,
            cfg
        );
        initialized_ &= (codec_decoder_ != nullptr);

        sampler_ = std::make_unique<Sampler>(0.7, 0.9, 50);
        initialized_ &= (sampler_ != nullptr);

        return initialized_;
    }

    void preload_model() {
        if ( initialized_ ) {
            slow_ar_->init();
            fast_ar_->init();
            codec_decoder_->init();
        } else {
            fmt::print("Audio8Engine not initialized. \n");
        }
    }

    void uninit() {
        env_.reset();
        slow_ar_.reset();
        fast_ar_.reset();
        codec_decoder_.reset();
        sampler_.reset();
        prompt_builder_.reset();
        voice_manager_.reset();
        initialized_ = false;
    }

    void shutdown() {}

    void cancel() { cancel_requested_.store(true); }

    std::vector<std::string> list_voices() {
        if ( voice_manager_ ) {
            return voice_manager_->list_voices();
        } else {
            return {};
        }
    }

    void registration(std::string voice_name, std::string transcript,
                  std::filesystem::path audio) {
        if ( voice_manager_ ) {
            voice_manager_->registration(voice_name, transcript, audio);
        }
    }

    void synthesize( const TTSRequest& request, progress_callback progress_cb, codebooks_callback codebook_cb) {
        if ( !initialized_ ) {
            fmt::print("Audio8Engine not initialized. \n");
            return;
        }

        auto progress = [&progress_cb](float p){ if (progress_cb) progress_cb(p); };

        VoiceProfile profile;
        bool ret = voice_manager_->load_profile(request.voice_name, profile);
        if( !ret ) return;
        
        progress(0.05f);
        Prompt prompt = prompt_builder_->build( request.text, profile.transcript, profile.codec_codes);

        size_t char_count = prompt.suffix_len - 11;
        double estimated_total_frames = std::max(1.0, char_count * 6.0);

        SlowARInput slow_input;
        make_initial_slow_input(profile, prompt, slow_input);

        SlowAROutput slow_output;
        slow_ar_->reset_kvcache();
        slow_ar_->generate_next(slow_input, slow_output);
        // fmt::print("initial step done.\n\n");

        std::list<int> previous;
        code_frame codebooks;
        std::vector<code_frame> frames;

        for (int step = 0; step < request.max_new_tokens; step++) {
            if ( cancel_requested_.load() ) {
                return;
            }

            int semantic = sampler_->sample_semantic(slow_output.logits.data, previous);
            if ( semantic == audio8::IM_END_ID ) {

                // fmt::print("\n\n STOP SIGN FOUND. \n\n");
                size_t total_frames = frames.size();
                float predicted_decoder_cost_ms = codec_decoder_->estimate_decode_time_ms(total_frames);
                
                fmt::print("\n decoder ETA {:.1f}sec. \n", predicted_decoder_cost_ms * 1e-3);

                progress(0.9f);
                codec_decoder_->decode_audio_batch(frames);
                progress(1.0f);
                return;
            }
            previous.push_back(semantic);
            if( previous.size() > 10 ) previous.pop_front();

            // fast step 0
            FastARInput fast_input(slow_output.slow_hidden, 0, true, 0);
            FastAROutput fast_output;
            fast_ar_->reset_kvcache();
            fast_ar_->generate_next(fast_input, fast_output);

            int token = std::min(std::max(semantic - audio8::SEMATIC_BEGIN_ID, 0), audio8::CODEBOOK_SIZE - 1);

            codebooks[0] = token;

            //fast step 1-9
            for (int fast_pos = 1; fast_pos < audio8::NUM_CODEBOOKS; fast_pos++) {
                FastARInput fast_input(slow_output.slow_hidden, token, false, fast_pos);
                
                fast_ar_->generate_next(fast_input, fast_output);

                token = sampler_->sample(fast_output.logits.data);
                codebooks[fast_pos] = token;
            }
            frames.push_back(codebooks);

            if ( codebook_cb ) {
                codebook_cb(codebooks.data(), audio8::NUM_CODEBOOKS);
            }

            update_slow_input(slow_input, semantic, prompt.prompt_len, step, codebooks);
            slow_ar_->generate_next(slow_input, slow_output);
            // fmt::print("step {} done.\n\n", step);
            progress(0.05f + 0.85f * std::min(0.99f, (float)step / (float)estimated_total_frames));
        }

        // fmt::print("\n\n MAX TOKEN REACHED. \n\n");
        progress(0.9f);
        codec_decoder_->decode_audio_batch(frames);
        progress(1.0f);
    }

private:

    void update_slow_input(SlowARInput& slow_input, const int& semantic, const int64_t& prompt_len, const int64_t& step, const code_frame& codebooks) {
        if ( slow_input.input_pos.size() > 1 || slow_input.codes.size() > 11 ) {
            slow_input.input_pos.data.resize(1);
            slow_input.input_pos.shape.resize(1);
            slow_input.codes.data.resize(11);
            slow_input.codes.shape.resize(3);
        }
        slow_input.input_pos.shape[0] = 1;
        slow_input.input_pos.data[0] = prompt_len + step;
        slow_input.codes.shape[0] = 1;
        slow_input.codes.shape[1] = 11;
        slow_input.codes.shape[2] = 1;
        slow_input.codes.data[0] = semantic;
        for (int i = 0; i < audio8::NUM_CODEBOOKS; i++) {
            slow_input.codes.data[i + 1] = codebooks[i];
        }
    }

    void make_initial_slow_input(const VoiceProfile& profile, const Prompt& prompt, SlowARInput& slow_input) {
        slow_input.input_pos.shape = { prompt.prompt_len };
        slow_input.input_pos.data.resize(prompt.prompt_len);
        memcpy(slow_input.input_pos.ptr(), prompt.position.data(), prompt.prompt_len*sizeof(int64_t));

        slow_input.codes.shape = { 1, 11, prompt.prompt_len };
        size_t slow_input_size = slow_input.codes.shape[0] * slow_input.codes.shape[1] * slow_input.codes.shape[2];

        slow_input.codes.data.resize( slow_input_size , 0);

        memcpy(slow_input.codes.ptr(), prompt.row0.data(), prompt.row0.size() * sizeof(int64_t));
        
        auto row_count = slow_input.codes.shape[1]; //11
        auto row_size = prompt.prompt_len;
        for (int row = 1; row < row_count; row++) {
            auto base = row * row_size;
            memcpy(&slow_input.codes.data[base + prompt.prefix_len], 
                &profile.codec_codes[(row - 1) * profile.frames], 
                profile.frames * sizeof(int64_t));
        }
    }

private:
    Audio8ModelPaths paths_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<VoiceManager> voice_manager_;
    std::unique_ptr<PromptBuilder> prompt_builder_;
    std::unique_ptr<SlowARGenerator> slow_ar_;
    std::unique_ptr<FastARGenerator> fast_ar_;
    std::unique_ptr<CodecDecoder> codec_decoder_;
    std::unique_ptr<Sampler> sampler_;
    std::atomic_bool cancel_requested_;
    bool initialized_;
};

Audio8Engine::Audio8Engine() : pImpl{ std::make_unique<Impl>() } {}
Audio8Engine::~Audio8Engine() = default;

std::vector<std::string> Audio8Engine::list_voices() { return pImpl->list_voices(); }

void Audio8Engine::registers(std::string voice_name, std::string transcript, std::filesystem::path audio) {
    pImpl->registration(voice_name, transcript, audio);
}

bool Audio8Engine::initialize(const std::filesystem::path& model_dir) {
    return pImpl->initialize(model_dir);
};

void Audio8Engine::preload_model() { pImpl->preload_model(); }

void Audio8Engine::uninit() { pImpl->uninit(); }

void Audio8Engine::synthesize( const TTSRequest& request, progress_callback progress, codebooks_callback codebook) {
    pImpl->synthesize(request, progress, codebook);
};

