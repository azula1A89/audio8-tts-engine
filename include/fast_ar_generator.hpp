#pragma once
#include "audio8_engine.hpp"

class FastARGenerator
{
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
public:

    FastARGenerator(
        void* env,
        const Audio8ModelPaths& paths,
        const RuntimeConfig& config);
    ~FastARGenerator();
    FastARGenerator(FastARGenerator&&) = default;
    FastARGenerator& operator=(FastARGenerator&&) = default;
    FastARGenerator(FastARGenerator&) = delete;
    FastARGenerator& operator=(const FastARGenerator&) = delete;
    bool init();
    void reset_kvcache();

    int generate_next(
        FastARInput& input,
        FastAROutput& state);
};