#pragma once
#include "audio8_engine.hpp"

struct VoiceProfile
{
    std::string name;

    std::string transcript;

    // frames * audio8::NUM_CODEBOOKS
    std::vector<int64_t> codec_codes;

    int frames = 0;
};

class VoiceManager {
    class Impl;
    std::unique_ptr<Impl> pImpl;
public:
    VoiceManager(
        void* env,
        const Audio8ModelPaths& paths,
        const RuntimeConfig& config);
    ~VoiceManager();
    VoiceManager(VoiceManager&&) = default;
    VoiceManager& operator=(VoiceManager&&) = default;
    VoiceManager(const VoiceManager&) = delete;
    VoiceManager& operator=(const VoiceManager&) = delete;
    bool init();
    std::vector<std::string> list_voices();
    void registration(std::string voice_name, std::string transcript, std::filesystem::path audio);
    bool load_profile(std::string voice_name, VoiceProfile& profile);
    
};