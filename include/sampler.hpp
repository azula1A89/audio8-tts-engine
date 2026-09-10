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