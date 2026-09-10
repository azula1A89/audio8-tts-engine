#pragma once
#include <string>
#include <memory>

class TextProcessor
{
    class Impl;
    std::unique_ptr<Impl> pImpl;
public:
    TextProcessor();
    ~TextProcessor();
    TextProcessor(TextProcessor&& ) = default;
    TextProcessor& operator=(TextProcessor&&) = default;
    TextProcessor(const TextProcessor&) = delete;
    TextProcessor& operator=(const TextProcessor&) = delete;


    std::string clean_text( const std::string& text ) const;
    std::string format_reference_text(const std::string& text) const;
};

