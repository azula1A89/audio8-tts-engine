
#include <mobi_wrapper.hpp>
#include <mobi.h>

#include <pugixml_wrapper.hpp>

namespace mobi_wrapper {
using namespace pugixml_wrapper;

struct FileCloser {
    void operator()(FILE* file) const {
        if (file) {
            std::fclose(file);
        }
    }
};

std::string from_mobi(const std::filesystem::path& file) {
    using MobiPtr = std::unique_ptr<MOBIData, decltype(&mobi_free)>;
    using RawmlPtr = std::unique_ptr<MOBIRawml, decltype(&mobi_free_rawml)>;
    using FilePtr = std::unique_ptr<FILE, FileCloser>;

    FilePtr input{std::fopen(file.string().c_str(), "rb")};
    if (!input) {
        throw std::runtime_error("Unable to open MOBI file: " + file.string());
    }

    MobiPtr mobi{mobi_init(), &mobi_free};
    if (!mobi) {
        throw std::runtime_error("Unable to initialize MOBI parser");
    }
    if (mobi_load_file(mobi.get(), input.get()) != MOBI_SUCCESS) {
        throw std::runtime_error("Unable to parse MOBI file: " + file.string());
    }

    RawmlPtr rawml{mobi_init_rawml(mobi.get()), &mobi_free_rawml};
    if (!rawml) {
        throw std::runtime_error("Unable to initialize MOBI content parser");
    }
    if (mobi_parse_rawml_opt(rawml.get(), mobi.get(), false, false, true) != MOBI_SUCCESS) {
        throw std::runtime_error("Unable to extract MOBI content: " + file.string());
    }

    std::string extracted;
    for (auto* part = rawml->markup; part; part = part->next) {
        if (!part->data || part->size == 0) {
            continue;
        }
        pugi::xml_document document;
        const auto result = document.load_buffer(part->data, part->size);
        if (!result) {
            throw std::runtime_error("Unable to parse MOBI markup: " + std::string{result.description()});
        }
        const auto text = extract_document_text(document, false, true);
        if (!text.empty()) {
            if (!extracted.empty()) {
                extracted += "\n\n";
            }
            extracted += text;
        }
    }
    return normalize_text(extracted);
};

};