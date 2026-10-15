#include <cstdio>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include <miniz.h>
#include <miniz_zip.h>
#include <pugixml_wrapper.hpp>
#include <text_extractor.hpp>

using namespace pugixml_wrapper;

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
};

TextExtractor::TextExtractor() : pImpl{std::make_unique<Impl>()} {}
TextExtractor::~TextExtractor() = default;
std::string TextExtractor::from_epub(const std::filesystem::path& file) { return pImpl->from_epub(file); }
std::string TextExtractor::from_text(const std::filesystem::path& file) { return pImpl->from_text(file); }
std::string TextExtractor::from_fb2(const std::filesystem::path& file) { return pImpl->from_fb2(file); }
