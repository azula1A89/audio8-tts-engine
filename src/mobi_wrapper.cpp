#include <vector>
#include <fstream>

#include <mobi_wrapper.hpp>
#include <mobi.h>
#include <pugixml.hpp>

namespace mobi_wrapper {
namespace {

struct FileCloser {
    void operator()(FILE* file) const {
        if (file) {
            std::fclose(file);
        }
    }
};

std::string path_string(const std::filesystem::path& file) {
    return file.string();
}

std::string read_file(const std::filesystem::path& file) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Unable to open file: " + path_string(file));
    }

    std::string contents{std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
    if (stream.bad()) {
        throw std::runtime_error("Unable to read file: " + path_string(file));
    }
    return contents;
}

std::string_view local_name(std::string_view name) {
    const auto separator = name.rfind(':');
    return separator == std::string_view::npos ? name : name.substr(separator + 1);
}

pugi::xml_node find_descendant(const pugi::xml_node& node, std::string_view name) {
    for (const auto& child : node.children()) {
        if (child.type() != pugi::node_element) {
            continue;
        }
        if (local_name(child.name()) == name) {
            return child;
        }
        if (auto descendant = find_descendant(child, name)) {
            return descendant;
        }
    }
    return {};
}

void collect_descendants(const pugi::xml_node& node,
                         std::string_view name,
                         std::vector<pugi::xml_node>& matches) {
    for (const auto& child : node.children()) {
        if (child.type() != pugi::node_element) {
            continue;
        }
        if (local_name(child.name()) == name) {
            matches.push_back(child);
        } else {
            collect_descendants(child, name, matches);
        }
    }
}

bool is_block_element(std::string_view name) {
    return name == "address" || name == "article" || name == "blockquote" ||
           name == "body" || name == "div" || name == "dd" || name == "dl" ||
           name == "dt" || name == "h1" || name == "h2" || name == "h3" ||
           name == "h4" || name == "h5" || name == "h6" || name == "li" ||
           name == "ol" || name == "p" || name == "pre" || name == "section" ||
           name == "table" || name == "td" || name == "th" || name == "tr" ||
           name == "title" || name == "ul" || name == "v" || name == "poem" ||
           name == "stanza" || name == "subtitle";
}

void append_xml_text(const pugi::xml_node& node, std::string& text, bool skip_navigation) {
    if (node.type() == pugi::node_pcdata || node.type() == pugi::node_cdata) {
        text += node.value();
        return;
    }
    if (node.type() != pugi::node_element && node.type() != pugi::node_document) {
        return;
    }

    const auto name = local_name(node.name());
    if (node.type() == pugi::node_element &&
        (name == "head" || name == "script" || name == "style" || name == "svg" ||
         name == "noscript" || name == "binary" || (skip_navigation && name == "nav"))) {
        return;
    }

    const bool block = is_block_element(name);
    if (block) {
        text += "\n\n";
    } else if (name == "br" || name == "empty-line") {
        text += '\n';
    }

    for (const auto& child : node.children()) {
        append_xml_text(child, text, skip_navigation);
    }

    if (block) {
        text += "\n\n";
    }
}

};

std::string normalize_text(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    std::size_t index = 0;

    while (index < text.size()) {
        const char current = text[index];
        if (current != ' ' && current != '\t' && current != '\n' &&
            current != '\r' && current != '\f' && current != '\v') {
            result += current;
            ++index;
            continue;
        }

        bool saw_newline = false;
        std::size_t newline_count = 0;
        while (index < text.size() &&
               (text[index] == ' ' || text[index] == '\t' || text[index] == '\n' ||
                text[index] == '\r' || text[index] == '\f' || text[index] == '\v')) {
            if (text[index] == '\n' || text[index] == '\r') {
                saw_newline = true;
                ++newline_count;
            }
            ++index;
        }

        if (saw_newline) {
            while (!result.empty() && result.back() == ' ') {
                result.pop_back();
            }
            const std::size_t breaks = newline_count > 1 ? 2 : 1;
            std::size_t existing = 0;
            while (existing < result.size() && result[result.size() - existing - 1] == '\n') {
                ++existing;
            }
            while (existing < breaks) {
                result += '\n';
                ++existing;
            }
        } else if (!result.empty() && result.back() != ' ' && result.back() != '\n') {
            result += ' ';
        }
    }

    while (!result.empty() && (result.back() == ' ' || result.back() == '\n')) {
        result.pop_back();
    }
    return result;
}

std::string extract_document_text(const pugi::xml_document& document,
                                  bool body_only,
                                  bool skip_navigation) {
    std::string text;
    if (body_only) {
        std::vector<pugi::xml_node> bodies;
        collect_descendants(document, "body", bodies);
        for (const auto& body : bodies) {
            append_xml_text(body, text, skip_navigation);
        }
    } else {
        append_xml_text(document, text, skip_navigation);
    }
    return normalize_text(text);
}

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