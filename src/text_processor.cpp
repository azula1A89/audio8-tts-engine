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

#include <text_processor.hpp>
#include <regex>

class TextProcessor::Impl {
public:
    Impl() = default;
    ~Impl() = default;
    std::string clean_text(const std::string& text) {
        std::u32string u32text = utf8_to_utf32(text);
        std::u32string filtered;
        
        for (char32_t c : u32text) {
            if (is_whitespace(c)) {
                filtered += c;
            } else if (!is_category_C(c)) {
                filtered += c;
            }
        }
        
        std::u32string normalized = _normalize_whitespace(filtered);
        return utf32_to_utf8(normalized);
    }

    std::string format_reference_text(const std::string& text) {
        std::string cleaned = clean_text(text);
        std::regex speaker_re(R"(<\|speaker:\d+\|>)");
        
        if (std::regex_search(cleaned, speaker_re)) {
            return cleaned;
        }
        return "<|speaker:0|>" + cleaned;
    }
    
private:

    std::u32string utf8_to_utf32(const std::string& utf8) {
        std::u32string utf32;
        size_t i = 0;
        while (i < utf8.length()) {
            unsigned char c = utf8[i];
            if (c <= 0x7F) {
                utf32 += c; i += 1;
            } else if ((c & 0xE0) == 0xC0) {
                if (i + 1 < utf8.length()) utf32 += ((c & 0x1F) << 6) | (utf8[i+1] & 0x3F);
                i += 2;
            } else if ((c & 0xF0) == 0xE0) {
                if (i + 2 < utf8.length()) utf32 += ((c & 0x0F) << 12) | ((utf8[i+1] & 0x3F) << 6) | (utf8[i+2] & 0x3F);
                i += 3;
            } else if ((c & 0xF8) == 0xF0) {
                if (i + 3 < utf8.length()) utf32 += ((c & 0x07) << 18) | ((utf8[i+1] & 0x3F) << 12) | ((utf8[i+2] & 0x3F) << 6) | (utf8[i+3] & 0x3F);
                i += 4;
            } else {
                i += 1;
            }
        }
        return utf32;
    }

    std::string utf32_to_utf8(const std::u32string& utf32) {
        std::string utf8;
        for (char32_t c : utf32) {
            if (c <= 0x7F) {
                utf8 += static_cast<char>(c);
            } else if (c <= 0x7FF) {
                utf8 += static_cast<char>(0xC0 | (c >> 6));
                utf8 += static_cast<char>(0x80 | (c & 0x3F));
            } else if (c <= 0xFFFF) {
                utf8 += static_cast<char>(0xE0 | (c >> 12));
                utf8 += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
                utf8 += static_cast<char>(0x80 | (c & 0x3F));
            } else if (c <= 0x10FFFF) {
                utf8 += static_cast<char>(0xF0 | (c >> 18));
                utf8 += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
                utf8 += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
                utf8 += static_cast<char>(0x80 | (c & 0x3F));
            }
        }
        return utf8;
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

    bool is_line_break(char32_t c) {
        return c == U'\r' || c == U'\n' || c == U'\v' || c == U'\f' || 
            (c >= 0x1C && c <= 0x1E) || c == 0x85 || 
            c == 0x2028 || c == 0x2029;
    }

    bool is_whitespace(char32_t c) {
        return c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' || 
            c == U'\v' || c == U'\f' || c == 0x1C || c == 0x1D || 
            c == 0x1E || c == 0x85 || c == 0xA0 || c == 0x1680 || 
            (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || 
            c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
    }

    bool is_category_C(char32_t c) {
        if (c <= 0x1F || (c >= 0x7F && c <= 0x9F)) return true; // Cc (Control)
        if (c >= 0xD800 && c <= 0xDFFF) return true;            // Cs (Surrogate)
        if ((c >= 0xE000 && c <= 0xF8FF) || 
            (c >= 0xF0000 && c <= 0xFFFFD) || 
            (c >= 0x100000 && c <= 0x10FFFD)) return true;      // Co (Private Use)
        if (c == 0xAD || c == 0x200B || c == 0x200C || c == 0x200D || 
            c == 0x200E || c == 0x200F || c == 0xFEFF) return true; // Cf (Format subset)
        return false;
    }

    std::u32string _normalize_whitespace(const std::u32string& text) {
        std::u32string result;
        size_t n = text.length();
        size_t i = 0;
        
        // re.sub(r"\s+", replace, text)
        while (i < n) {
            if (is_whitespace(text[i])) {
                size_t start = i;
                bool has_line_break = false;
                while (i < n && is_whitespace(text[i])) {
                    if (is_line_break(text[i])) has_line_break = true;
                    i++;
                }
                
                char32_t left = (start > 0) ? text[start - 1] : 0;
                char32_t right = (i < n) ? text[i] : 0;
                
                if (has_line_break && left != 0 && right != 0 && is_cjk(left) && is_cjk(right)) {

                } else {
                    result += U' ';
                }
            } else {
                result += text[i];
                i++;
            }
        }
        
        // .strip()
        size_t first = 0;
        while (first < result.length() && result[first] == U' ') first++;
        if (first == result.length()) return U"";
        
        size_t last = result.length() - 1;
        while (last >= 0 && result[last] == U' ') last--;
        
        return result.substr(first, last - first + 1);
    }
};


TextProcessor::TextProcessor() : pImpl{std::make_unique<Impl>()}{};
std::string TextProcessor::clean_text( const std::string& text ) const {
    return pImpl->clean_text(text);
}
TextProcessor::~TextProcessor() = default;

std::string TextProcessor::format_reference_text(const std::string& text) const {
    return pImpl->format_reference_text(text);
}
