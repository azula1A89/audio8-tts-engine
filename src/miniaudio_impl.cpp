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

#include <miniaudio_impl.hpp>
#include <fmt/core.h>
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>


class MiniAudio::LoopPlayer
{
public:
    enum class State {
        Audio, // playing output.WAV
        Beep   // playing 1000Hz triangle wave generation
    };

    LoopPlayer(ma_uint32 sample_rate, float duration_in_seconds = 0.2f)
        : sample_rate_(sample_rate)
    {
        beep_frames_total_ = (ma_uint64)(duration_in_seconds * sample_rate_);
        beep_frames_remaining_ = 0;
        state_ = State::Audio;

        ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, 1, sample_rate_);
        if (ma_decoder_init_file("output.WAV", &decoder_config, &decoder_) != MA_SUCCESS) return;

        ma_data_source_set_looping(&decoder_, MA_FALSE);

        ma_device_config device_config = ma_device_config_init(ma_device_type_playback);
        device_config.playback.format   = decoder_.outputFormat;
        device_config.playback.channels = decoder_.outputChannels;
        device_config.sampleRate        = decoder_.outputSampleRate;
        device_config.dataCallback      = data_callback;
        device_config.pUserData         = this;

        if (ma_device_init(NULL, &device_config, &device_) != MA_SUCCESS) {
            ma_decoder_uninit(&decoder_);
            return;
        }

        ma_device_start(&device_);
    }

    void stop() { ma_device_stop(&device_); }

    virtual ~LoopPlayer() {
        ma_device_uninit(&device_);
        ma_decoder_uninit(&decoder_);
    }

private:
    static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
        LoopPlayer* p_player = (LoopPlayer*)pDevice->pUserData;
        if (!p_player) return;

        float* p_output_f32 = (float*)pOutput;
        ma_uint32 frames_read_total = 0;

        while (frames_read_total < frameCount) {
            ma_uint32 frames_remaining = frameCount - frames_read_total;
            float* p_buffer_out = p_output_f32 + (frames_read_total * p_player->decoder_.outputChannels);

            if (p_player->state_ == State::Audio) {
                // read audio from file decoder
                ma_uint64 frames_read = 0;
                ma_data_source_read_pcm_frames(&p_player->decoder_, p_buffer_out, frames_remaining, &frames_read);
                
                frames_read_total += (ma_uint32)frames_read;

                // output.WAV playback finished switch to Beep playing state
                if (frames_read < frames_remaining) {
                    p_player->state_ = State::Beep;
                    p_player->beep_frames_remaining_ = p_player->beep_frames_total_;
                }
            }
            else if (p_player->state_ == State::Beep) {
                // triangle wave generation
                ma_uint32 beep_frames_to_generate = (ma_uint32)ma_min(frames_remaining, (ma_uint32)p_player->beep_frames_remaining_);
                ma_uint32 channels = p_player->decoder_.outputChannels;

                for (ma_uint32 i = 0; i < beep_frames_to_generate; ++i) {
                    // Calculate the phase of the current frame
                    ma_uint64 current_frame = p_player->beep_frames_total_ - p_player->beep_frames_remaining_ + i;

                    // 1000 Hz triangle wave generation
                    double time_in_sec = (double)current_frame / p_player->sample_rate_;
                    double t = std::fmod(time_in_sec * 1000.0, 1.0); // cycle time [0, 1)
                    float sample = (float)(4.0 * std::abs(t - 0.5) - 1.0);
                    double pos = ((double)current_frame / (double)p_player->beep_frames_total_ );// [0, 1]

                    // beep sound volume tweeking
                    sample *= 0.2f;
                    sample *= pow(1-pos, 12);

                    for (ma_uint32 ch = 0; ch < channels; ++ch) {
                        p_buffer_out[i * channels + ch] = sample;
                    }
                }

                frames_read_total += beep_frames_to_generate;
                p_player->beep_frames_remaining_ -= beep_frames_to_generate;

                // Beep play finished switch back to State::Audio
                if (p_player->beep_frames_remaining_ == 0) {
                    ma_data_source_seek_to_pcm_frame(&p_player->decoder_, 0);
                    p_player->state_ = State::Audio;
                }
            }
        }
    }

    ma_uint32 sample_rate_;
    ma_decoder decoder_;
    ma_device device_;

    State state_;
    ma_uint64 beep_frames_total_;     // Beep sound total frames
    ma_uint64 beep_frames_remaining_; // Beep sound remaining frames
};

class MiniAudio::Recoder
{
public:
    explicit Recoder(std::string name, ma_uint32 sample_rate): filename_(name), sample_rate_(sample_rate)  {
        ma_encoder_config  encoder_config = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, sample_rate_);
        if (MA_SUCCESS != ma_encoder_init_file(filename_.c_str(), &encoder_config, &encoder_)) { /* handle error .*/ }
    }
    virtual ~Recoder() {
        ma_encoder_uninit(&encoder_);
    }
    void write(float* buff, ma_uint64 count) {
        ma_uint64 frames_written;
        if (ma_encoder_write_pcm_frames(&encoder_, buff, count, &frames_written) != MA_SUCCESS) { /* handle error .*/ }
    }
    void write(const float* buff, ma_uint64 count) {
        ma_uint64 frames_written;
        if (ma_encoder_write_pcm_frames(&encoder_, buff, count, &frames_written) != MA_SUCCESS) { /* handle error .*/ }
    }
private:
    std::string filename_;
    ma_uint32 sample_rate_;
    ma_encoder encoder_;
};

MiniAudio::MiniAudio() : player_(nullptr), recoder_(nullptr) {};
MiniAudio::~MiniAudio(){};

void MiniAudio::play() {
    if ( !player_ ) {
        player_ = std::make_unique<LoopPlayer>(ma_standard_sample_rate_44100, 2.5f);
    }
};

void MiniAudio::stop() {
    if ( player_ ) {
        player_->stop();
        player_.reset();
    }
};

void MiniAudio::wav_write(const float *buff, uint64_t count) {
    if ( !recoder_ ) {
        recoder_ = std::make_unique<Recoder>("output.WAV", ma_standard_sample_rate_44100);
    }

    if ( recoder_ ) {
        recoder_->write(buff, count);
        recoder_.reset();
        recoder_ = nullptr;
        fmt::print("\n\n audio saved to output.WAV. \n\n");
    }
}

std::vector<float> MiniAudio::load_audio( const std::string& path ) {
    ma_decoder decoder;
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, 44100); 

    if (ma_decoder_init_file(path.c_str(), &config, &decoder) != MA_SUCCESS) {
        fmt::print("ma_decoder_init_file failed.");
    }

    ma_uint64 total_frames;
    ma_decoder_get_length_in_pcm_frames(&decoder, &total_frames);

    std::vector<float> data(total_frames, 0.0f);

    ma_uint64 frames_read;
    ma_result ret = ma_decoder_read_pcm_frames(&decoder, data.data(), total_frames, &frames_read);
    if ( ret != MA_SUCCESS ) {
        fmt::print("ma_decoder_read_pcm_frames failed.");
    }

    ma_decoder_uninit(&decoder);
    return data;
};



