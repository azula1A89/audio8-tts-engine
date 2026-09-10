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
#include <list>

class Sampler
{
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
public:
    Sampler() = delete;
    explicit Sampler(float temperature = 0.7, float top_p = 0.9, int top_k = 50);
    ~Sampler();
    Sampler(Sampler&&) = default;
    Sampler& operator=(Sampler&&) = default;
    Sampler(Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    int sample(const std::vector<float>& logits);
    int sample_semantic(const std::vector<float>& logits,
                        const std::list<int>& previous);
// private:
//     std::vector<float> logits_;
};