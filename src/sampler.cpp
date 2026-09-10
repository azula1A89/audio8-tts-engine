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

#include <sampler.hpp>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include "audio8_engine.hpp"


class Sampler::Impl {
public:
    Impl(float temperature = 0.7, float top_p = 0.9, int top_k = 50):
    temperature_(temperature),
    top_p_(top_p),
    top_k_(top_k),
    semantic_begin_id_(audio8::SEMATIC_BEGIN_ID),
    semantic_end_id_(audio8::SEMATIC_END_ID),
    im_end_id_(audio8::IM_END_ID),
    slow_logits_layout_semantic_then_eos_(true),
    rng_(42)
    {}

    int sample(const std::vector<float>& logits) {
        return _sample(logits, temperature_, top_p_, top_k_, rng_);
    }
    int sample_semantic(const std::vector<float>& logits,
                        const std::list<int>& previous) {
        return _sample_semantic(logits, previous, temperature_, top_p_, top_k_, rng_, semantic_begin_id_, semantic_end_id_, im_end_id_, slow_logits_layout_semantic_then_eos_);
    }

private:

    int _sample(const std::vector<float>& logits, float temperature, float top_p, int top_k, std::mt19937& rng) {
        size_t n = logits.size();
        if (n == 0) throw std::invalid_argument("logits cannot be empty");

        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return logits[a] > logits[b];
        });

        std::vector<float> base(n);
        float max_val = logits[order[0]];
        float sum_base = 0.0;
        
        for (size_t i = 0; i < n; ++i) {
            base[i] = std::exp(logits[order[i]] - max_val);
            sum_base += base[i];
        }

        float cumulative = 0.0;
        std::vector<bool> remove(n, false);
        for (size_t i = 0; i < n; ++i) {
            base[i] /= sum_base;
            cumulative += base[i];
            if ((cumulative > top_p) || (i >= static_cast<size_t>(top_k))) {
                remove[i] = true;
            }
        }
        remove[0] = false;

        std::vector<float> masked = logits;
        for (size_t i = 0; i < n; ++i) {
            if (remove[i]) {
                masked[order[i]] = -std::numeric_limits<float>::infinity();
            }
        }

        float temp_clamped = std::max(temperature, 1e-5f);
        float max_scaled = -std::numeric_limits<float>::infinity();
        std::vector<float> scaled(n);
        
        for (size_t i = 0; i < n; ++i) {
            scaled[i] = masked[i] / temp_clamped;
            if (scaled[i] > max_scaled) {
                max_scaled = scaled[i];
            }
        }

        float sum_probs = 0.0;
        std::vector<float> probs(n);
        for (size_t i = 0; i < n; ++i) {
            probs[i] = std::exp(scaled[i] - max_scaled);
            sum_probs += probs[i];
        }

        std::uniform_real_distribution<float> dist(1e-12, 1.0);
        float max_score = -std::numeric_limits<float>::infinity();
        int best_idx = -1;

        for (size_t i = 0; i < n; ++i) {
            probs[i] /= sum_probs;
            float noise = -std::log(dist(rng));
            float score = probs[i] / noise; 
            
            if (score > max_score) {
                max_score = score;
                best_idx = i;
            }
        }

        return best_idx;
    }


    int _sample_semantic(
        const std::vector<float>& logits, 
        const std::list<int>& previous, 
        float temperature, 
        float top_p, 
        int top_k, 
        std::mt19937& rng,
        int semantic_begin_id,
        int semantic_end_id,
        int im_end_id,
        bool layout_semantic_then_eos
    ) {
        std::vector<int> allowed_ids;
        int num_semantic = semantic_end_id - semantic_begin_id + 1;
        allowed_ids.reserve(num_semantic + 1);
        
        for (int i = semantic_begin_id; i <= semantic_end_id; ++i) {
            allowed_ids.push_back(i);
        }
        allowed_ids.push_back(im_end_id);

        std::vector<float> allowed_logits;
        if (layout_semantic_then_eos) {
            allowed_logits = logits;
        } else {
            allowed_logits.reserve(allowed_ids.size());
            for (int id : allowed_ids) {
                allowed_logits.push_back(logits[id]);
            }
        }

        if (allowed_logits.size() != allowed_ids.size()) {
            throw std::runtime_error("unexpected slow logits size");
        }

        int normal_index = _sample(allowed_logits, temperature, top_p, top_k, rng);
        int normal = allowed_ids[normal_index];

        int high_index = _sample(allowed_logits, 1.0, 0.9, top_k, rng);
        int high = allowed_ids[high_index];

        bool in_previous = (std::find(previous.begin(), previous.end(), normal) != previous.end());
        
        if (normal >= semantic_begin_id && normal <= semantic_end_id && in_previous) {
            return high;
        }

        return normal;
    }

private:

    float temperature_;
    float top_p_;
    int top_k_;
    int semantic_begin_id_;
    int semantic_end_id_;
    int im_end_id_;
    bool slow_logits_layout_semantic_then_eos_;
    std::mt19937 rng_;
};

Sampler::Sampler(float temperature, float top_p, int top_k) : 
pImpl{ std::make_unique<Impl>(temperature, top_p, top_k) }
{};

Sampler::~Sampler() = default;

int Sampler::sample(const std::vector<float>& logits) {
    return pImpl->sample(logits);
};

int Sampler::sample_semantic(const std::vector<float>& logits, 
    const std::list<int>& previous){
    return pImpl->sample_semantic(logits, previous);
};






