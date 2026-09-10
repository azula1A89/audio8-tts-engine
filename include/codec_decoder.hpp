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
#include <audio8_engine.hpp>
#include <memory>

typedef void (*decoder_callback) (const float* data, size_t size) ;

class CodecDecoder {
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
public:
    CodecDecoder(
        void* env,
        const Audio8ModelPaths& paths,
        const RuntimeConfig& config);
    ~CodecDecoder();
    CodecDecoder(CodecDecoder&&) = default;
    CodecDecoder& operator=(CodecDecoder&&) = default;
    CodecDecoder(const CodecDecoder&) = delete;
    CodecDecoder& operator=(const CodecDecoder&) = delete;
    bool init();
    void decode_audio_batch(const std::vector<code_frame>& frames, decoder_callback cb = nullptr);
    float estimate_decode_time_ms(size_t T);
};