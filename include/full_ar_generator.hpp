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
#include <functional>
typedef std::function<void(const code_frame&, const int& step, const bool finished, const bool max_token_reached)> frame_callback;

class FullARGenerator
{
    class Impl;
    std::unique_ptr<Impl> pImpl;

public:

    FullARGenerator(
        void* env,
        const Audio8ModelPaths& paths,
        const RuntimeConfig& config);
    ~FullARGenerator();
    FullARGenerator(FullARGenerator&&) = default;
    FullARGenerator& operator=(FullARGenerator&&) = default;
    FullARGenerator(FullARGenerator&) = delete;
    FullARGenerator& operator=(const FullARGenerator&) = delete;
    bool init();
    void cancle();
    void generate_frame( FullARInput& input, frame_callback fcb);
};