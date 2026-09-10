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