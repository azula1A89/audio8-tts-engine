#pragma once
#include <string>
#include <filesystem>
#include <pugixml.hpp>

namespace pugixml_wrapper {
    std::string path_string(const std::filesystem::path& file);
    std::string read_file(const std::filesystem::path& file);
    std::string_view local_name(std::string_view name);
    bool is_block_element(std::string_view name);
    std::string normalize_text(std::string_view text);
    pugi::xml_node find_descendant(const pugi::xml_node& node, std::string_view name);
    std::string extract_document_text(const pugi::xml_document& document, bool body_only, bool skip_navigation);
    std::string decode_uri_path(std::string_view path);
};

