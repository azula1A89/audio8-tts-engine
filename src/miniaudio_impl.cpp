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
#include <algorithm>

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
    Tracks(ma_uint32 channels, ma_uint32 sampleRate)
        : channels_(channels), sample_tate_(sampleRate), cursor_(0), total_frames_(0)
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
        ma_data_source_uninit(&data_source_impl_.base);
    }

    void track_add(const std::vector<float>& track) {
        if (track.empty()) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        tracks_.push_back(track);
        
        track_offsets_.push_back(total_frames_);
        total_frames_ += track.size() / channels_;
    }

    void track_delete(int index) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= 0 && index < tracks_.size()) {
            ma_uint64 deleted_frames = tracks_[index].size() / channels_;
            
            tracks_.erase(tracks_.begin() + index);
            track_offsets_.erase(track_offsets_.begin() + index);
            
            total_frames_ = 0;
            for (size_t i = 0; i < tracks_.size(); ++i) {
                track_offsets_[i] = total_frames_;
                total_frames_ += tracks_[i].size() / channels_;
            }

            if (cursor_ > total_frames_) {
                cursor_ = total_frames_;
            }
        }
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
            size_t frames = tracks_[index].size() / channels_;
            return static_cast<double>(frames) / sample_tate_;
        }
        return 0.0;
    }

    void buffer_reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        tracks_.clear();
        track_offsets_.clear();
        cursor_ = 0;
    }

    size_t buffer_length() {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_frames_ * channels_;
    }

    void copy_to_buffer_offset(float* dest, size_t begin, size_t length) {
        const ma_uint64 total_samples = total_frames_ * channels_;
        if (!dest || length == 0 || begin > total_samples ) return;

        size_t begin_track_idx = 0;
        auto upper_begin = std::upper_bound(track_offsets_.begin(), track_offsets_.end(), begin);
        if ( upper_begin != track_offsets_.end() ) {
            begin_track_idx = std::distance(track_offsets_.begin(), upper_begin) - 1;
        } else {
            begin_track_idx = tracks_.size() - 1;
        }

        size_t end_track_idx = 0;
        auto end = begin + length;
        end = end > total_samples ? total_samples : end;
        auto upper_end = std::upper_bound(track_offsets_.begin(), track_offsets_.end(), end);
        if ( upper_end != track_offsets_.end() ) {
            end_track_idx = std::distance(track_offsets_.begin(), upper_end) - 1;
        } else {
            end_track_idx = tracks_.size() - 1;
        }

        size_t copied = 0;
        for (size_t track_idx = begin_track_idx; track_idx <= end_track_idx; ++track_idx) {
            const auto& track = tracks_[track_idx];
            size_t start_frame = (track_idx == begin_track_idx) ? (begin - track_offsets_[begin_track_idx]) : 0;
            size_t end_frame = (track_idx == end_track_idx) ? (end - track_offsets_[track_idx]) : track.size();
            size_t to_copy = end_frame - start_frame;
            if (to_copy > 0) {
                std::memcpy(dest + copied, track.data() + start_frame, to_copy * sizeof(float));
                copied += to_copy;
            }
            if (copied >= length) break;
        }
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
    std::vector<std::vector<float>> tracks_;
    std::vector<ma_uint64> track_offsets_;
    
    ma_uint64 cursor_;
    ma_uint64 total_frames_;
    ma_uint32 channels_;
    ma_uint32 sample_tate_;

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

        ma_uint64 frames_remaining_to_read = frame_count;
        float* p_out = static_cast<float*>(p_frames_out);
        ma_uint64 total_frames_read = 0;

        size_t track_idx = 0;
        auto upper = std::upper_bound(self->track_offsets_.begin(), self->track_offsets_.end(), self->cursor_);

        if ( upper != self->track_offsets_.end() ) {
            track_idx = std::distance(self->track_offsets_.begin(), upper) - 1;
        } else {
            track_idx = self->tracks_.size() - 1;
        }
        
        while (frames_remaining_to_read > 0 && track_idx < self->tracks_.size()) {
            ma_uint64 track_start_frame = self->track_offsets_[track_idx];
            ma_uint64 track_end_frame = track_start_frame + (self->tracks_[track_idx].size() / self->channels_);
            ma_uint64 offset_in_track = self->cursor_ - track_start_frame;
            ma_uint64 frames_avail_in_track = track_end_frame - self->cursor_;
            ma_uint64 frames_to_copy = std::min(frames_remaining_to_read, frames_avail_in_track);

            const float* p_src = self->tracks_[track_idx].data() + (offset_in_track * self->channels_);
            std::memcpy(p_out, p_src, frames_to_copy * self->channels_ * sizeof(float));

            p_out += frames_to_copy * self->channels_;
            self->cursor_ += frames_to_copy;
            total_frames_read += frames_to_copy;
            frames_remaining_to_read -= frames_to_copy;

            track_idx++;
        }

        if (p_frames_read) *p_frames_read = total_frames_read;
        return (total_frames_read == frame_count) ? MA_SUCCESS : MA_AT_END;
    }

    static ma_result seek_callback(ma_data_source* p_data_source, ma_uint64 frame_index) {
        Tracks* self = get_parent(p_data_source);
        std::lock_guard<std::mutex> lock(self->mutex_);
        self->cursor_ = (frame_index > self->total_frames_) ? self->total_frames_ : frame_index;
        return MA_SUCCESS;
    }

    static ma_result get_data_format_callback(ma_data_source* p_data_source, ma_format* p_format, ma_uint32* p_channels, ma_uint32* p_sample_rate, ma_channel* p_channel_map, size_t channel_map_size) {
        Tracks* self = get_parent(p_data_source);
        if (p_format) *p_format = ma_format_f32;
        if (p_channels) *p_channels = self->channels_;
        if (p_sample_rate) *p_sample_rate = self->sample_tate_;
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
public:
    explicit TrackPlayer(Tracks& tracks) : engine_(), sound_(), tracks_(tracks) {
        ma_engine_init(NULL, &engine_);
        ma_sound_init_from_data_source(&engine_, tracks.get_ma_data_source(), 0, NULL, &sound_);
    }
    ~TrackPlayer() {
        ma_sound_uninit(&sound_);
        ma_engine_uninit(&engine_);
    }
    bool play(int index) {
        // stop current playing
        if ( ma_sound_is_playing(&sound_) ) {
            ma_sound_start(&sound_);
        }

        // setup stop point
        auto t = ma_engine_get_time_in_pcm_frames(&engine_) +
                    (ma_engine_get_sample_rate(&engine_)) * tracks_.track_duration_sec(index);
        ma_sound_set_stop_time_in_pcm_frames(&sound_, t);

        // start playing
        size_t begin = tracks_.track_begin_index(index);
        bool ret = (MA_SUCCESS == ma_sound_seek_to_pcm_frame(&sound_, begin));
             ret |= (MA_SUCCESS == ma_sound_start(&sound_));
        
        return ret;
    }
    bool play() {
        ma_sound_reset_stop_time(&sound_);
        return (MA_SUCCESS == ma_sound_start(&sound_));
    }
    bool pause() {
        ma_sound_reset_stop_time(&sound_);
        return (MA_SUCCESS == ma_sound_stop(&sound_));
    }
    bool stop() {
        ma_sound_reset_stop_time(&sound_);
        bool ret = (MA_SUCCESS == ma_sound_seek_to_pcm_frame(&sound_, 0));
             ret |= (MA_SUCCESS == ma_sound_stop(&sound_));
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

MiniAudio::MiniAudio() :
    tracks_{ std::make_unique<Tracks>(1, 44100) }, 
    loop_player_(nullptr), 
    track_player_(nullptr), 
    recoder_(nullptr) {
        track_player_ = make_unique_nothrow<TrackPlayer>( *tracks_ );
};

MiniAudio::~MiniAudio(){};

void MiniAudio::track_add(const std::vector<float>& item) {
    if ( tracks_ ) {
        tracks_->track_add(const_cast<std::vector<float>&>(item));
    }
}

void MiniAudio::track_delete(int index) {
    if ( tracks_ ) {
        tracks_->track_delete(index);
    }
}

size_t MiniAudio::track_count() {
    if ( tracks_ ) {
        return tracks_->track_count();
    }
    return 0;
}

void MiniAudio::copy_to_buffer(float* dest, size_t begin, size_t length) {
    if ( tracks_ ) {
        tracks_->copy_to_buffer_offset(dest, begin, length);
    }
}

size_t MiniAudio::buffer_length() {
    if ( tracks_ ) {
        return tracks_->buffer_length();
    }
    return 0;
}

void MiniAudio::buffer_reset() {
    if ( tracks_ ) {
        tracks_->buffer_reset();
    }
}

bool MiniAudio::play(int index) {
    if ( track_player_ ) {
        return track_player_->play(index);
    }
    return false;
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

bool MiniAudio::is_playing() {
    return (loop_player_ && loop_player_->initialized());
}

void MiniAudio::wav_write(const float *buff, uint64_t count, const char* name) {
    if ( !recoder_ ) {
        recoder_ = make_unique_nothrow<Recoder>(name, ma_standard_sample_rate_44100);
    }

    if ( recoder_ ) {
        recoder_->write(buff, count);
        recoder_.reset();
        recoder_ = nullptr;
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



