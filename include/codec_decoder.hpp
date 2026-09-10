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