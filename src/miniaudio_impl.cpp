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

#include <utility>
#include <vector>
#include <mutex>
#include <cstring>
#include <filesystem>

#include <fmt/core.h>
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include <miniaudio_impl.hpp>

template <typename T, typename... Args>
std::unique_ptr<T> make_unique_nothrow(Args&&... args) noexcept {
    try {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    } catch (...) {
        return nullptr;
    }
}

class MiniAudio::Tracks {
public:
    Tracks(ma_uint32 channels, ma_uint32 sample_rate)
        : channels_(channels), sample_rate_(sample_rate), 
          cursor_(0), total_frames_(0), 
          decoder_open_(false), current_track_idx_(0)
    {
        static ma_data_source_vtable vtable = {
            read_callback,
            seek_callback,
            get_data_format_callback,
            get_cursor_callback,
            get_length_callback,
            nullptr, // onSetLooping
            0        // flags
        };

        data_source_impl_.parent = this;
        ma_data_source_config config = ma_data_source_config_init();
        config.vtable = &vtable;
        ma_data_source_init(&config, &data_source_impl_.base);
    }

    ~Tracks() {
        close_current_decoder_nolock();
        ma_data_source_uninit(&data_source_impl_.base);
    }

    void track_add(const std::filesystem::path& wav_path) {
        std::lock_guard<std::mutex> lock(mutex_);

        ma_decoder_config config = ma_decoder_config_init(ma_format_f32, channels_, sample_rate_);
        ma_decoder temp_decoder;
        
        if (ma_decoder_init_file(wav_path.string().c_str(), &config, &temp_decoder) != MA_SUCCESS) {
            return;
        }

        ma_uint64 length = 0;
        ma_decoder_get_length_in_pcm_frames(&temp_decoder, &length);
        ma_decoder_uninit(&temp_decoder);

        if (length == 0) return;

        tracks_.push_back(wav_path);
        track_offsets_.push_back(total_frames_);
        total_frames_ += length;
    }

    void track_delete(int index) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= 0 && index < tracks_.size()) {

            if (decoder_open_ && current_track_idx_ == index) {
                close_current_decoder_nolock();
            }

            tracks_.erase(tracks_.begin() + index);
            track_offsets_.erase(track_offsets_.begin() + index);

            total_frames_ = 0;
            for (size_t i = 0; i < tracks_.size(); ++i) {
                track_offsets_[i] = total_frames_;
                
                ma_decoder_config config = ma_decoder_config_init(ma_format_f32, channels_, sample_rate_);
                ma_decoder temp;
                ma_uint64 len = 0;
                if (ma_decoder_init_file(tracks_[i].string().c_str(), &config, &temp) == MA_SUCCESS) {
                    ma_decoder_get_length_in_pcm_frames(&temp, &len);
                    ma_decoder_uninit(&temp);
                }
                total_frames_ += len;
            }

            if (cursor_ > total_frames_) {
                cursor_ = total_frames_;
            }
        }
    }

    void track_clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        close_current_decoder_nolock();
        tracks_.clear();
        track_offsets_.clear();
        total_frames_ = 0;
        cursor_ = 0;
    }

    size_t track_count() {
        std::lock_guard<std::mutex> lock(mutex_);
        return tracks_.size();
    }

    size_t track_begin_index(int index) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= 0 && index < track_offsets_.size()) {
            return track_offsets_[index];
        }
        return 0;
    }

    double track_duration_sec(int index) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= 0 && index < tracks_.size()) {
            ma_uint64 start = track_offsets_[index];
            ma_uint64 end = (index + 1 < tracks_.size()) ? track_offsets_[index + 1] : total_frames_;
            return static_cast<double>(end - start) / sample_rate_;
        }
        return 0.0;
    }

    size_t buffer_length() {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_frames_ * channels_;
    }

    ma_data_source* get_ma_data_source() {
        return (ma_data_source*)&data_source_impl_.base;
    }

private:
    struct DataSourceImpl {
        ma_data_source_base base;
        Tracks* parent;
    } data_source_impl_;

    std::mutex mutex_;
    std::vector<std::filesystem::path> tracks_;
    std::vector<ma_uint64> track_offsets_;
    
    ma_uint64 cursor_;
    ma_uint64 total_frames_;
    ma_uint32 channels_;
    ma_uint32 sample_rate_;

    bool decoder_open_;
    size_t current_track_idx_;
    ma_decoder current_decoder_;

    void close_current_decoder_nolock() {
        if (decoder_open_) {
            ma_decoder_uninit(&current_decoder_);
            decoder_open_ = false;
        }
    }

    bool open_decoder_for_track_nolock(size_t track_idx) {
        close_current_decoder_nolock();
        
        if (track_idx >= tracks_.size()) return false;

        ma_decoder_config config = ma_decoder_config_init(ma_format_f32, channels_, sample_rate_);
        if (ma_decoder_init_file(tracks_[track_idx].string().c_str(), &config, &current_decoder_) != MA_SUCCESS) {
            return false;
        }
        
        decoder_open_ = true;
        current_track_idx_ = track_idx;
        return true;
    }

    size_t get_track_index_from_cursor_nolock(ma_uint64 cursor) {
        for (size_t i = 1; i < track_offsets_.size(); ++i) {
            if (cursor < track_offsets_[i]) {
                return i - 1;
            }
        }
        return track_offsets_.empty() ? 0 : track_offsets_.size() - 1;
    }

    static Tracks* get_parent(ma_data_source* p_data_source) {
        return ((DataSourceImpl*)p_data_source)->parent;
    }

    static ma_result read_callback(ma_data_source* p_data_source, void* p_frames_out, ma_uint64 frame_count, ma_uint64* p_frames_read) {
        Tracks* self = get_parent(p_data_source);
        std::lock_guard<std::mutex> lock(self->mutex_);

        if (self->cursor_ >= self->total_frames_) {
            if (p_frames_read) *p_frames_read = 0;
            return MA_AT_END;
        }

        ma_uint64 total_frames_read = 0;
        float* p_out = static_cast<float*>(p_frames_out);

        while (total_frames_read < frame_count && self->cursor_ < self->total_frames_) {
            size_t expected_track = self->get_track_index_from_cursor_nolock(self->cursor_);

            if (!self->decoder_open_ || self->current_track_idx_ != expected_track) {
                if (!self->open_decoder_for_track_nolock(expected_track)) {
                    break;
                }
                
                ma_uint64 local_offset = self->cursor_ - self->track_offsets_[expected_track];
                ma_decoder_seek_to_pcm_frame(&self->current_decoder_, local_offset);
            }

            ma_uint64 frames_to_read = frame_count - total_frames_read;
            ma_uint64 frames_read_this_iter = 0;

            ma_decoder_read_pcm_frames(&self->current_decoder_, p_out, frames_to_read, &frames_read_this_iter);

            if (frames_read_this_iter == 0) {
                self->cursor_ = (expected_track + 1 < self->tracks_.size()) 
                                 ? self->track_offsets_[expected_track + 1] 
                                 : self->total_frames_;
            } else {
                total_frames_read += frames_read_this_iter;
                p_out += frames_read_this_iter * self->channels_;
                self->cursor_ += frames_read_this_iter;
            }
        }

        if (p_frames_read) *p_frames_read = total_frames_read;
        return (total_frames_read == frame_count) ? MA_SUCCESS : MA_AT_END;
    }

    static ma_result seek_callback(ma_data_source* p_data_source, ma_uint64 frame_index) {
        Tracks* self = get_parent(p_data_source);
        std::lock_guard<std::mutex> lock(self->mutex_);
        
        self->cursor_ = (frame_index > self->total_frames_) ? self->total_frames_ : frame_index;
        self->close_current_decoder_nolock();
        return MA_SUCCESS;
    }

    static ma_result get_data_format_callback(ma_data_source* p_data_source, ma_format* p_format, ma_uint32* p_channels, ma_uint32* p_sample_rate, ma_channel* p_channel_map, size_t channel_map_size) {
        Tracks* self = get_parent(p_data_source);
        if (p_format) *p_format = ma_format_f32;
        if (p_channels) *p_channels = self->channels_;
        if (p_sample_rate) *p_sample_rate = self->sample_rate_;
        return MA_SUCCESS;
    }

    static ma_result get_cursor_callback(ma_data_source* p_data_source, ma_uint64* p_cursor) {
        Tracks* self = get_parent(p_data_source);
        std::lock_guard<std::mutex> lock(self->mutex_);
        if (p_cursor) *p_cursor = self->cursor_;
        return MA_SUCCESS;
    }

    static ma_result get_length_callback(ma_data_source* p_data_source, ma_uint64* p_length) {
        Tracks* self = get_parent(p_data_source);
        std::lock_guard<std::mutex> lock(self->mutex_);
        if (p_length) *p_length = self->total_frames_;
        return MA_SUCCESS;
    }
};

class MiniAudio::TrackPlayer{
    ma_engine engine_;
    ma_sound sound_;
    Tracks& tracks_;
    bool initialized_;
public:
    explicit TrackPlayer(Tracks& tracks) : engine_(), sound_(), tracks_(tracks), initialized_(false) {
        initialized_ = (ma_engine_init(NULL, &engine_) == MA_SUCCESS);
        initialized_ &= (ma_sound_init_from_data_source(&engine_, tracks.get_ma_data_source(), 0, NULL, &sound_) == MA_SUCCESS);
    }
    ~TrackPlayer() {
        ma_sound_uninit(&sound_);
        ma_engine_uninit(&engine_);
    }

    bool is_playing() {
        if ( !initialized_ ) return false;

        return ma_sound_is_playing(&sound_);
    }

    bool play() {
        if ( !initialized_ ) return false;

        return (MA_SUCCESS == ma_sound_start(&sound_));
    }

    bool pause() {
        if ( !initialized_ ) return false;
        return (MA_SUCCESS == ma_sound_stop(&sound_));
    }

    bool stop() {
        if ( !initialized_ ) return false;
        bool ret = (MA_SUCCESS == ma_sound_stop(&sound_));
             ret &= (MA_SUCCESS == ma_sound_seek_to_pcm_frame(&sound_, 0));
        return ret;
    }
};

class MiniAudio::LoopPlayer
{
public:
    enum class State {
        Audio, // playing single audio file
        Beep   // playing 1000Hz triangle wave generation
    };

    LoopPlayer(const char* file, ma_uint32 sample_rate, float duration_in_seconds = 0.2f)
        : initialized_(false), sample_rate_(sample_rate)
    {
        beep_frames_total_ = (ma_uint64)(duration_in_seconds * sample_rate_);
        beep_frames_remaining_ = 0;
        state_ = State::Audio;

        ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, 1, sample_rate_);
        if (ma_decoder_init_file(file, &decoder_config, &decoder_) != MA_SUCCESS) return;

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
        initialized_ = true;
    }

    bool initialized() { return initialized_; }

    void stop() { if( initialized_ ) ma_device_stop(&device_); }

    virtual ~LoopPlayer() {
        if ( initialized_ ) {
            ma_device_uninit(&device_);
            ma_decoder_uninit(&decoder_);
        }
    }

private:
    static void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frame_count) {
        LoopPlayer* p_player = (LoopPlayer*)pDevice->pUserData;
        if (!p_player) return;

        float* p_output_f32 = (float*)pOutput;
        ma_uint32 frames_read_total = 0;

        while (frames_read_total < frame_count) {
            ma_uint32 frames_remaining = frame_count - frames_read_total;
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

    bool initialized_;
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

    void write(const float* buff, ma_uint64 count) {
        ma_uint64 frames_written;
        if (ma_encoder_write_pcm_frames(&encoder_, buff, count, &frames_written) != MA_SUCCESS) { /* handle error .*/ }
    }
private:
    std::string filename_;
    ma_uint32 sample_rate_;
    ma_encoder encoder_;
};

class MiniAudio::Resampler
{
public:
    // 构造函数：建议加入 channels 通道数参数（默认为单声道 1）
    explicit Resampler(ma_uint32 in_sample_rate, 
                       ma_uint32 out_sample_rate, 
                       ma_uint32 channels = 1)
        : in_sample_rate_(in_sample_rate),
          out_sample_rate_(out_sample_rate),
          channels_(channels),
          is_initialized_(false)
    {
        // 1. 初始化重采样配置对象
        ma_resampler_config config = ma_resampler_config_init(
            ma_format_f32,                // 采样格式（根据 float* 接口固定为 32位浮点型）
            channels_,                    // 声道数
            in_sample_rate_,              // 输入采样率 (Hz)
            out_sample_rate_,             // 输出采样率 (Hz)
            ma_resample_algorithm_linear  // 重采样算法（线性插值）
        );

        // 2. 初始化底层 ma_resampler 结构
        ma_result result = ma_resampler_init(&config, nullptr, &resampler_);
        if (result == MA_SUCCESS) {
            is_initialized_ = true;
        }
    }

    virtual ~Resampler() {
        // 3. 析构时安全释放重采样器
        if (is_initialized_) {
            ma_resampler_uninit(&resampler_, nullptr);
        }
    }

    /**
     * @brief 执行 PCM 帧重采样
     * @param in  输入 float PCM 缓存指针
     * @param out 输出 float PCM 缓存指针
     * @param in_frame_count  输入的 PCM 帧数
     * @param max_out_frame_count 输出缓冲区能够容纳的最大 PCM 帧数
     * @return ma_uint64 实际生成的输出 PCM 帧数
     */
    ma_uint64 resample(const float* in, float* out, ma_uint64 in_frame_count, ma_uint64 max_out_frame_count) {
        if (!is_initialized_ || !in || !out) {
            return 0;
        }

        ma_uint64 frame_count_in = in_frame_count;
        ma_uint64 frame_count_out = max_out_frame_count;

        // 4. 调用 miniaudio 原生帧重采样接口
        ma_result result = ma_resampler_process_pcm_frames(
            &resampler_, 
            in, 
            &frame_count_in, 
            out, 
            &frame_count_out
        );

        if (result != MA_SUCCESS) {
            return 0;
        }

        // 返回实际写入 out 缓冲区的帧数
        return frame_count_out;
    }

    // 辅助 API：根据输入的帧数预估所需的输出缓冲区大小（帧数）
    ma_uint64 get_expected_output_frame_count(ma_uint64 input_frame_count) const {
        if (!is_initialized_) return 0;
        ma_uint64 expected_out = 0;
        ma_resampler_get_expected_output_frame_count(
            const_cast<ma_resampler*>(&resampler_), 
            input_frame_count, 
            &expected_out
        );
        return expected_out;
    }

private:
    ma_uint32 in_sample_rate_;
    ma_uint32 out_sample_rate_;
    ma_uint32 channels_;
    ma_resampler resampler_;
    bool is_initialized_;
};


MiniAudio::MiniAudio() :
    tracks_{ std::make_unique<Tracks>(1, 44100) }, 
    loop_player_(nullptr), 
    track_player_(nullptr), 
    recoder_(nullptr) {
        track_player_ = make_unique_nothrow<TrackPlayer>( *tracks_ );
};

MiniAudio::~MiniAudio(){};

void MiniAudio::track_add(const std::string& path) {
    if ( tracks_ ) {
        tracks_->track_add(path);
    }
}

void MiniAudio::track_delete(int index) {
    if ( tracks_ ) {
        tracks_->track_delete(index);
    }
}

void MiniAudio::track_clear() {
    if ( tracks_ ) {
        tracks_->track_clear();
    }
}

size_t MiniAudio::track_count() {
    if ( tracks_ ) {
        return tracks_->track_count();
    }
    return 0;
}

void MiniAudio::export_audio( int max_length_sec, const char* export_path, uint32_t sample_rate ) {
    ma_result result;
    ma_uint64 frames_read;
    ma_uint64 max_frames = max_length_sec * 44100U * 1;
    ma_uint64 len = tracks_->buffer_length();

    Resampler resampler(44100, sample_rate);

    std::filesystem::path folder = export_path;
    if ( len && std::filesystem::exists(folder) && std::filesystem::is_directory(folder) ) {
        int parts = len / max_frames;
        int final_len = len % max_frames;
        std::vector<float> buffer;
        std::vector<float> resampled_buffer;
        
        for (int n = 0; n < parts + 1; n++) {
            int length = ( n >= parts ) ? final_len : max_frames;
            buffer.resize(length);
            int out_length = resampler.get_expected_output_frame_count(length);
            resampled_buffer.resize(out_length);

            result = ma_data_source_seek_to_pcm_frame(tracks_->get_ma_data_source(), n * max_frames);
            if (result != MA_SUCCESS) { break; }

            result = ma_data_source_read_pcm_frames(tracks_->get_ma_data_source(), buffer.data(), length, &frames_read);
            if (result != MA_SUCCESS) { break; }

            resampler.resample(buffer.data(), resampled_buffer.data(), length, out_length);

            auto wav_file = folder / fmt::format("part_{}_{}.wav", n+1, parts+1);
            auto recoder = make_unique_nothrow<Recoder>( wav_file.string().c_str(), sample_rate);
            if ( recoder ) {
                recoder->write(resampled_buffer.data(), resampled_buffer.size());
            }
            recoder.reset();
        }
    }

}

size_t MiniAudio::buffer_length() {
    if ( tracks_ ) {
        return tracks_->buffer_length();
    }
    return 0;
}

bool MiniAudio::play() {
    if ( track_player_ ) {
        return track_player_->play();
    }
    return false;
}

bool MiniAudio::pause() {
    if ( track_player_ ) {
        return track_player_->pause();
    }
    return false;
}

bool MiniAudio::stop() {
    if ( track_player_ ) {
        return track_player_->stop();
    }
    return false;
}

bool MiniAudio::play_file(const char* file) {
    bool ret = false;
    if ( !loop_player_ ) {
        loop_player_ = make_unique_nothrow<LoopPlayer>(file, ma_standard_sample_rate_44100, 2.5f);
        ret = (loop_player_ && loop_player_->initialized());
    }
    return ret;
};

void MiniAudio::stop_file() {
    if ( loop_player_ ) {
        loop_player_->stop();
        loop_player_.reset();
    }
};

bool MiniAudio::is_playing_list() {
    return track_player_->is_playing();
}

bool MiniAudio::is_playing_file() {
    return (loop_player_ && loop_player_->initialized());
}

void MiniAudio::wav_write(const float *buff, uint64_t count, const char* name) {
    if ( !recoder_ ) {
        recoder_ = make_unique_nothrow<Recoder>(name, ma_standard_sample_rate_44100);
    }

    if ( recoder_ ) {
        recoder_->write(buff, count);
        recoder_.reset();
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



