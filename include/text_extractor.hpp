/*extract plain text from E-Book file*/
#pragma once
#include <memory>
#include <filesystem>

class TextExtractor {
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
public:
    TextExtractor();
    ~TextExtractor();
    TextExtractor(TextExtractor&&) = default;
    TextExtractor& operator=(TextExtractor&&) = default;
    TextExtractor(const TextExtractor&) = delete;
    TextExtractor& operator=(const TextExtractor&) = delete;

    std::string from_epub(const std::filesystem::path& file);
    std::string from_text(const std::filesystem::path& file);
    std::string from_fb2(const std::filesystem::path& file);
    std::string from_mobi(const std::filesystem::path& file);
};
