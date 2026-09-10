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