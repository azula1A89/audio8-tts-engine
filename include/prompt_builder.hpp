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