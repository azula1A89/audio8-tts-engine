#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <miniz.h>
#include <miniz_zip.h>
#include <mobi_wrapper.hpp>
#include <pugixml.hpp>

#include <text_extractor.hpp>

namespace {

struct FileCloser {
    void operator()(FILE* file) const {
        if (file) {
            std::fclose(file);
        }
    }
};

struct MinizHeapDeleter {
    void operator()(void* data) const {
        mz_free(data);
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

std::string decode_uri_path(std::string_view path) {
    const auto suffix = path.find_first_of("?#");
    if (suffix != std::string_view::npos) {
        path = path.substr(0, suffix);
    }

    std::string decoded;
    decoded.reserve(path.size());
    for (std::size_t i = 0; i < path.size(); ++i) {
        if (path[i] == '%' && i + 2 < path.size()) {
            const auto hex_value = [](char value) -> int {
                if (value >= '0' && value <= '9') return value - '0';
                if (value >= 'a' && value <= 'f') return value - 'a' + 10;
                if (value >= 'A' && value <= 'F') return value - 'A' + 10;
                return -1;
            };
            const int high = hex_value(path[i + 1]);
            const int low = hex_value(path[i + 2]);
            if (high >= 0 && low >= 0) {
                decoded += static_cast<char>((high << 4) | low);
                i += 2;
                continue;
            }
        }
        decoded += path[i];
    }
    return decoded;
}

class ZipReader {
public:
    explicit ZipReader(const std::filesystem::path& file) {
        if (mz_zip_reader_init_file(&archive_, path_string(file).c_str(), 0) != MZ_TRUE) {
            throw std::runtime_error("Unable to open EPUB archive: " + path_string(file));
        }
        initialized_ = true;
    }

    ~ZipReader() {
        if (initialized_) {
            mz_zip_reader_end(&archive_);
        }
    }

    std::string read(std::string_view name) const {
        const std::string entry_name{name};
        const int index = mz_zip_reader_locate_file(&archive_, entry_name.c_str(), nullptr, 0);
        if (index < 0) {
            throw std::runtime_error("Missing EPUB archive entry: " + entry_name);
        }

        size_t size = 0;
        std::unique_ptr<void, MinizHeapDeleter> data{
            mz_zip_reader_extract_to_heap(&archive_, static_cast<mz_uint>(index), &size, 0)};
        if (!data) {
            throw std::runtime_error("Unable to read EPUB archive entry: " + entry_name);
        }
        return {static_cast<const char*>(data.get()), size};
    }

private:
    mutable mz_zip_archive archive_{};
    bool initialized_ = false;
};

void load_xml(pugi::xml_document& document,
              std::string_view data,
              const std::string& description) {
    const auto result = document.load_buffer(data.data(), data.size());
    if (!result) {
        throw std::runtime_error("Unable to parse " + description + ": " + result.description());
    }
}

} // namespace

class TextExtractor::Impl {
public:
    Impl() {};
    ~Impl() {};

    // miniz + pugixml
    std::string from_epub(const std::filesystem::path& file) {
        ZipReader archive{file};
        const auto container_data = archive.read("META-INF/container.xml");
        pugi::xml_document container;
        load_xml(container, container_data, "EPUB container");
        const auto rootfile = find_descendant(container, "rootfile");
        if (!rootfile || !rootfile.attribute("full-path")) {
            throw std::runtime_error("EPUB container does not identify a package document");
        }

        const std::string package_path = decode_uri_path(rootfile.attribute("full-path").value());
        const auto package_data = archive.read(package_path);
        pugi::xml_document package;
        load_xml(package, package_data, "EPUB package document");
        const auto manifest = find_descendant(package, "manifest");
        const auto spine = find_descendant(package, "spine");
        if (!manifest || !spine) {
            throw std::runtime_error("EPUB package document is missing its manifest or spine");
        }

        std::unordered_map<std::string, std::string> resources;
        for (const auto& item : manifest.children()) {
            if (local_name(item.name()) != "item") {
                continue;
            }
            const auto id = item.attribute("id");
            const auto href = item.attribute("href");
            if (id && href) {
                resources.emplace(id.value(), href.value());
            }
        }

        std::string extracted;
        for (const auto& itemref : spine.children()) {
            if (local_name(itemref.name()) != "itemref") {
                continue;
            }
            const auto idref = itemref.attribute("idref");
            if (!idref) {
                continue;
            }
            const auto resource = resources.find(idref.value());
            if (resource == resources.end()) {
                throw std::runtime_error("EPUB spine references an unknown manifest item");
            }

            const auto href = decode_uri_path(resource->second);
            auto content_path = (std::filesystem::path(package_path).parent_path() / href)
                                    .lexically_normal()
                                    .generic_string();
            if (!content_path.empty() && content_path.front() == '/') {
                content_path.erase(content_path.begin());
            }
            const auto content_data = archive.read(content_path);
            pugi::xml_document content;
            load_xml(content, content_data, "EPUB content document");
            const auto text = extract_document_text(content, false, true);
            if (!text.empty()) {
                if (!extracted.empty()) {
                    extracted += "\n\n";
                }
                extracted += text;
            }
        }
        return normalize_text(extracted);
    };

    // direct read out
    std::string from_text(const std::filesystem::path& file) {
        return read_file(file);
    };

    // pugixml
    std::string from_fb2(const std::filesystem::path& file) {
        const auto contents = read_file(file);
        pugi::xml_document document;
        load_xml(document, contents, "FB2 document");
        return extract_document_text(document, true, false);
    };

    // nanopdf
    std::string from_pdf(const std::filesystem::path& file) { return ""; };

    // libmobi
    std::string from_mobi(const std::filesystem::path& file) {
        return mobi_wrapper::from_mobi(file);
    };
};

TextExtractor::TextExtractor() : pImpl{std::make_unique<Impl>()} {}
TextExtractor::~TextExtractor() = default;
std::string TextExtractor::from_epub(const std::filesystem::path& file) { return pImpl->from_epub(file); }
std::string TextExtractor::from_text(const std::filesystem::path& file) { return pImpl->from_text(file); }
std::string TextExtractor::from_fb2(const std::filesystem::path& file) { return pImpl->from_fb2(file); }
std::string TextExtractor::from_pdf(const std::filesystem::path& file) { return pImpl->from_pdf(file); }
std::string TextExtractor::from_mobi(const std::filesystem::path& file) { return pImpl->from_mobi(file); }
