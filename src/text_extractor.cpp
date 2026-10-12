
#include <miniz.h>
#include <pugixml.hpp>

#include <text_extractor.hpp>

class TextExtractor::Impl {
public:
    Impl() {};
    ~Impl() {};

    // miniz + pugixml
    std::string from_epub(const std::filesystem::path& file) {

    };

    // direct read out
    std::string from_text(const std::filesystem::path& file) {

    };

    // pugixml
    std::string from_fb2(const std::filesystem::path& file) {

    };

    // nanopdf
    std::string from_pdf(const std::filesystem::path& file) {};

    // libmobi
    std::string from_mobi(const std::filesystem::path& file) {

    };
};

TextExtractor::TextExtractor() : pImpl{std::make_unique<Impl>()} {}
TextExtractor::~TextExtractor() = default;
std::string TextExtractor::from_epub(const std::filesystem::path& file) { return pImpl->from_epub(file); }
std::string TextExtractor::from_text(const std::filesystem::path& file) { return pImpl->from_text(file); }
std::string TextExtractor::from_fb2(const std::filesystem::path& file) { return pImpl->from_fb2(file); }
std::string TextExtractor::from_pdf(const std::filesystem::path& file) { return pImpl->from_pdf(file); }
std::string TextExtractor::from_mobi(const std::filesystem::path& file) { return pImpl->from_mobi(file); }
