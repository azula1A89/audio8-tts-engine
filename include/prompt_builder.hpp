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
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

struct Prompt
{
    int64_t prompt_len;
    int64_t prefix_len;
    int64_t suffix_len;
    
    std::vector<int64_t> row0;

    std::vector<int64_t> position;

    std::vector<int64_t> reference_codes;
};


class PromptBuilder {
    class Impl;
    std::unique_ptr<Impl> pImpl;

public:
    PromptBuilder() = delete;
    PromptBuilder(const std::filesystem::path& tokenizer_dir, int semantic_begin_id, int num_codebooks);
    ~PromptBuilder();
    PromptBuilder(PromptBuilder&&) = default;
    PromptBuilder& operator=(PromptBuilder&&) = default;
    PromptBuilder(const PromptBuilder&) = delete;
    PromptBuilder& operator=(const PromptBuilder&) = delete;
    Prompt build( const std::string& target_text, const std::string& transcript, const std::filesystem::path code_file);
    Prompt build( const std::string& target_text, const std::string& transcript, const std::vector<int64_t>& codes);
};