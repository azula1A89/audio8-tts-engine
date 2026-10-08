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

#include <regex>
#include <string>
#include <string_view>
#include <cctype> 
#include <fmt/core.h>

#include <uni_algo/norm.h>
#include <uni_algo/prop.h>
#include <uni_algo/ranges.h>
#include <uni_algo/ranges_conv.h>
#include <uni_algo/conv.h>

#include <text_processor.hpp>

class TextProcessor::Impl {
public:
    Impl() = default;
    ~Impl() = default;
    std::string clean_text(std::string_view input) {
        // 1. NFKC normalization (full-width to half-width, handling combining characters, etc.)
        std::string normalized = una::norm::to_nfkc_utf8(input);

        std::string result;
        result.reserve(normalized.size());

        bool last_was_space = false;
        char32_t last_punct = 0;

        // 2. Filter, deduplicate, and convert types in a single pass
        for (char32_t c : normalized | una::views::utf8) {
            // Determine whether to keep the current character
            bool keep = false;
            if (una::codepoint::is_alphabetic(c) ||
                una::codepoint::is_numeric(c)    ||
                una::codepoint::is_whitespace(c) ||
                is_useful_punctuation(c)) {
                keep = true;
            } else {
                keep = !should_remove(c);
            }

            if (!keep) continue;

            // Handle whitespace characters
            if (una::codepoint::is_whitespace(c)) {
                if (!last_was_space && !result.empty()) {
                    result.push_back(' '); // unify to half-width space
                    last_was_space = true;
                }
                last_punct = 0;
                continue;
            }

            last_was_space = false;

            // Handle punctuation characters
            if (is_useful_punctuation(c)) {
                if (c == last_punct)
                    continue; // skip consecutive identical punctuation marks
                last_punct = c;
            } else {
                last_punct = 0;
            }

            // Append the current character to the result string
            append_utf8(result, c);
        }

        // Remove trailing space if present
        if (!result.empty() && result.back() == ' ') {
            result.pop_back();
        }

        return result;
    }

    std::string format_reference_text(const std::string& text) {
        std::string cleaned = clean_text(text);
        std::regex speaker_re(R"(<\|speaker:\d+\|>)");
        
        if (std::regex_search(cleaned, speaker_re)) {
            return cleaned;
        }
        return "<|speaker:0|>" + cleaned;
    }

    std::vector<std::string> split_into_sentences(std::string_view text) {
        std::vector<std::string> sentences;
        std::string sentence;
        std::string cleaned;
        int cut_count = 0;
        int last_cut_count = 0;

        for (char32_t cp : text | una::views::utf8) {
            auto prop = una::codepoint::prop{cp};
                
            if ( is_delimiters(cp) ) {
                append_utf8(sentence, cp);
                cleaned = clean_text(sentence);
                if ( !is_pure_punctuation(cleaned) ) {
                    sentences.push_back(cleaned);
                }
                
                sentence.clear();
                cut_count++;
            } else if ( prop.General_Category_Pe() || prop.General_Category_Pf() ) {
                if ( sentence.empty() ) {
                    if ( last_cut_count != cut_count ) {
                        last_cut_count = cut_count;
                        append_utf8(sentences.back(), cp);
                    }
                }
            } else {
                append_utf8(sentence, cp);
            }
        }

        // last sentence if any.
        if ( !sentence.empty() ) {
            cleaned = clean_text(sentence);
            if ( !is_pure_punctuation(cleaned) ) {
                sentences.push_back(cleaned);
            }
        }

        return sentences;
    }

    // if a string contains any CJK characters, it is considered as CJK text
    bool contains_cjk( std::string_view input ) {
        for (char32_t c : input | una::views::utf8) {
            if (is_cjk(c)) {
                return true;
            }
        }
        return false;
    }

private:
    bool should_remove(char32_t c) {
        using namespace una::codepoint;

        if (is_control(c) || is_private_use(c) || is_noncharacter(c)) return true;

        // ZWSP, ZWNJ, ZWJ BOM / Word Joiner
        if (c == 0x200B || c == 0x200C || c == 0x200D || c == 0xFEFF || c == 0x2060) return true;

        auto p = prop{c};
        if (p.General_Category_So() || p.General_Category_Sm() || p.General_Category_Sc() || p.General_Category_Sk()) {
            // remove all:
            // Other_Symbol -> most emoji
            // Math_Symbol  -> ~ + - ...
            // Currency_Symbol
            // Modifier_Symbol
            return true;
        } else if (p.General_Category_Po()) {
            // remove all
            // Other_Punctuation
            return true;
        }

        // Emoji blocks as a fallback
        if ((c >= 0x1F300 && c <= 0x1F9FF) || (c >= 0x1FA00 && c <= 0x1FAFF) ||
            (c >= 0x2600  && c <= 0x26FF)  || (c >= 0x2700  && c <= 0x27BF)  ||
            (c >= 0xFE00  && c <= 0xFE0F)  || (c >= 0x1F000 && c <= 0x1F02F))
            return true;

        return false;
    }

    bool is_useful_punctuation(char32_t c) {
        switch (c) {
            // english
            case U'.': case U',': case U'!': case U'?':
            case U';': case U':': case U'\'': case U'"':
            case U'-': case U'(': case U')': case U'[': case U']':
            // cjk
            case U'。': case U'，': case U'！': case U'？':
            case U'；': case U'：': case U'、': case U'…':
            case U'—': case U'（': case U'）': case U'【': case U'】':
            case U'《': case U'》': case U'“': case U'”': case U'‘': case U'’':
            case U'『': case U'』': case U'「': case U'」':
                return true;
            default:
                return false;
        }
    }

    bool is_delimiters(char32_t c) {
        switch (c) {
            case U'。': case U'？': case U'！': case U'；': case U'，': case U'\n':
            case U'!': case U'?': case U';': case U',':
                return true;
            default:
                return false;
        }
    }

    bool is_pure_punctuation(std::string_view sentence) {
        for (char32_t cp : sentence | una::views::utf8) {
            auto prop = una::codepoint::prop{cp};
            if ( !prop.General_Category_P() ) {
                    return false; 
            }
        }
        return true;
    }

    inline void append_utf8(std::string& out, char32_t cp) {
        if (cp <= 0x7F) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    bool is_cjk(char32_t c) {
        return (c >= 0x1100 && c <= 0x11FF) || (c >= 0x2E80 && c <= 0x2FDF) ||
            (c >= 0x3000 && c <= 0x303F) || (c >= 0x3040 && c <= 0x30FF) ||
            (c >= 0x3100 && c <= 0x31FF) || (c >= 0x3400 && c <= 0x4DBF) ||
            (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0xA960 && c <= 0xA97F) ||
            (c >= 0xAC00 && c <= 0xD7A3) || (c >= 0xD7B0 && c <= 0xD7FF) ||
            (c >= 0xF900 && c <= 0xFAFF) || (c >= 0xFE30 && c <= 0xFE4F) ||
            (c >= 0xFF01 && c <= 0xFF9F) || (c >= 0x20000 && c <= 0x2FA1F);
    }
};

TextProcessor::TextProcessor() : pImpl{std::make_unique<Impl>()}{};
std::string TextProcessor::clean_text( const std::string& text ) const {
    return pImpl->clean_text(text);
}
TextProcessor::~TextProcessor() = default;

std::vector<std::string> TextProcessor::split_into_sentences(const std::string& text) {
    return pImpl->split_into_sentences(text);
}

std::string TextProcessor::format_reference_text(const std::string& text) const {
    return pImpl->format_reference_text(text);
}

bool TextProcessor::contains_cjk(const std::string& text) {
    return pImpl->contains_cjk(text);
}