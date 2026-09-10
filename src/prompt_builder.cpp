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

#include <prompt_builder.hpp>
// #include <tokenizers_cpp/tokenizer.hpp>
#include <tokenizers_cpp.h>
#include <text_processor.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <fmt/color.h>

class PromptBuilder::Impl {
public:
    Impl(const std::filesystem::path& tokenizer_dir, int semantic_begin_id, int num_codebooks) :
    tokenizer_dir_(tokenizer_dir),
    semantic_begin_id_(semantic_begin_id), 
    num_codebooks_(num_codebooks),
    tokenizer_(nullptr),
    text_processor_{ std::make_unique<TextProcessor>() }
    {}

    Prompt build( const std::string& target_text, const std::string& transcript, const std::vector<int64_t>& codes) {

        Prompt prompt;
        
        std::vector<std::string> prefix_parts = {
            "<|im_start|>system\n",
            "convert the provided text to speech reference to the following:\n\nText:\n",
            text_processor_->format_reference_text(transcript),
            "\n\nSpeech:\n",
        };
        std::vector<std::string> suffix_parts = {
            "<|im_end|>\n",
            "<|im_start|>user\n",
            text_processor_->clean_text(target_text),
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        };

        std::vector<uint32_t> prefix_toks;
        for (auto part : prefix_parts) {
            auto toks = encode_text(part);
            for (auto& t : toks) {
                prefix_toks.push_back(t);
            }
        }

        std::vector<uint32_t> suffix_toks;
        for (auto part : suffix_parts) {
            auto toks = encode_text(part);
            for (auto& t : toks) {
                suffix_toks.push_back(t);
            }
        }

        int len = codes.size() / num_codebooks_; //frames
        std::vector<int64_t> semantic_ids(len, 0);

        for (int i = 0; i<len; i++) {
            semantic_ids[i] = semantic_begin_id_ + codes[i];
        }

        prompt.reference_codes = codes;
        prompt.prefix_len = prefix_toks.size();
        prompt.suffix_len = suffix_toks.size();
        for (auto& p : prefix_toks) {
            prompt.row0.push_back(p);
        }
        for (auto& p : semantic_ids) {
            prompt.row0.push_back(p);
        }
        for (auto& p : suffix_toks) {
            prompt.row0.push_back(p);
        }
        
        prompt.prompt_len = prompt.row0.size();
        for (size_t i = 0; i < prompt.prompt_len; i++) {
            prompt.position.push_back(i);
        }
        

        return prompt;
    
    }

    Prompt build( const std::string& target_text, const std::string& transcript, const std::filesystem::path code_file) {
        auto codes = load_codes(code_file);
        return build(target_text, transcript, codes);
    }

private:

    std::vector<int64_t> load_codes(const std::filesystem::path code_file) {

        std::vector<int64_t> ret{};

        std::error_code ec;
        auto size = std::filesystem::file_size(code_file, ec);

        if (ec) {
            fmt::print(fmt::emphasis::bold | fg(fmt::color::red), 
                "Could not get file size: {}\n", ec.message());
            return ret;
        }

        if( size % sizeof(int64_t) ) {
            fmt::print(fmt::emphasis::bold | fg(fmt::color::red), 
                "data corrupted.\n");
            return ret;
        }

        ret.resize(size / sizeof(int64_t), 0);
        std::ifstream in_file(code_file, std::ios::binary);

        if (!in_file) {
            fmt::print(fmt::emphasis::bold | fg(fmt::color::red), 
                "Could not open file for reading\n");
            return ret;
        }

        in_file.read(reinterpret_cast<char*>(ret.data()), size);

        if (!in_file) {
            fmt::print(fmt::emphasis::bold | fg(fmt::color::red), 
                "Could not read the whole data\n");
            return {};
        }

        return ret;
    }

    std::string load_bytes_from_file(const std::string& path) {
        std::ifstream fs(path, std::ios::in | std::ios::binary);
        if (fs.fail()) {
            std::cerr << "Cannot open " << path << std::endl;
            exit(1);
        }
        
        std::string data;
        fs.seekg(0, std::ios::end);
        size_t size = static_cast<size_t>(fs.tellg());
        fs.seekg(0, std::ios::beg);
        data.resize(size);
        fs.read(data.data(), size);
        return data;
    }

    std::vector<int32_t> encode_text( const std::string& text ) {
        if ( !tokenizer_ ) {
            auto blob = load_bytes_from_file(tokenizer_dir_.string());
            tokenizer_ = std::move(tokenizers::Tokenizer::FromBlobJSON(blob));
        }
        return tokenizer_->Encode( text);
    }

private:
    std::filesystem::path tokenizer_dir_;
    int semantic_begin_id_;
    int num_codebooks_;

    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    std::unique_ptr<TextProcessor> text_processor_;
};

PromptBuilder::PromptBuilder(const std::filesystem::path& tokenizer_dir, int semantic_begin_id, int num_codebooks) : pImpl{ std::make_unique<Impl>( tokenizer_dir, semantic_begin_id, num_codebooks ) } {}
PromptBuilder::~PromptBuilder() = default;

Prompt PromptBuilder::build( const std::string& target_text, const std::string& transcript, const std::filesystem::path code_file) {
    return pImpl->build(target_text, transcript, code_file);
}

Prompt PromptBuilder::build( const std::string& target_text, const std::string& transcript, const std::vector<int64_t>& codes) {
    return pImpl->build(target_text, transcript, codes);
}

