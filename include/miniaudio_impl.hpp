#pragma once
#include <memory>
#include <vector>
#include <string>

class MiniAudio {
    class LoopPlayer;
    class Recoder;
public:
    MiniAudio();
    ~MiniAudio();
    MiniAudio(MiniAudio&&) = default;
    MiniAudio& operator=(MiniAudio&&) = default;
    MiniAudio(const MiniAudio&) = delete;
    MiniAudio& operator=(const MiniAudio&) = delete;
    void play();
    void stop();
    std::vector<float> load_audio( const std::string& path );
    void wav_write(const float *buff, uint64_t count);

private:
    std::unique_ptr<LoopPlayer> player_;
    std::unique_ptr<Recoder> recoder_;
};