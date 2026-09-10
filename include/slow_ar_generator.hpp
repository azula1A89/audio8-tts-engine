#pragma once
#include "audio8_engine.hpp"

class SlowARGenerator
{
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
public:

    SlowARGenerator(
        void* env,
        const Audio8ModelPaths& paths,
        const RuntimeConfig& config);
    ~SlowARGenerator();
    SlowARGenerator(SlowARGenerator&&) = default;
    SlowARGenerator& operator=(SlowARGenerator&&) = default;
    SlowARGenerator(SlowARGenerator&) = delete;
    SlowARGenerator& operator=(const SlowARGenerator&) = delete;
    bool init();
    void reset_kvcache();

    int generate_next(
        SlowARInput& input,
        SlowAROutput& state);
};