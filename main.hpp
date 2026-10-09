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


#pragma once

#include <imgui.h>
#include <utility>
#include <unordered_map>
#include <string>
#include <fmt/ranges.h>
#include <fstream>
#include <optional>
#include <iostream>
#include <nlohmann/json.hpp>

class Localization {
const char* model_info_ = "This project using the following model:";
const char* model_structure_info_ = "The expected model directory is:";
const char* model_folder_structure_ = R"(models/
├── slow_ar_int4.onnx
├── slow_ar_int4.onnx.data
│
├── fast_ar_int4.onnx
├── fast_ar_int4.onnx.data
│
├── codec_decoder_fp16.onnx
├── codec_decoder_fp16.onnx.data
│
├── runtime_manifest.json
├── tokenizer/
│   └── tokenizer.json
│
└── registration/
    ├── codec_encoder_fp16.onnx
    └── codec_encoder_fp16.onnx.data
)";

const std::string help_text_en_ = R"(
tips:
___
  * step 1: start a new session or open an existing one.
  * step 2: paste the text you want to convert to speech.
  * step 3: click the table header "segment [20 token limit]" to adjust the segment length. then click "update segmention" to split the text into segments.
  * step 4: optional, you can edit the text or voice settings for each segment as needed.
  * step 5: click "run" to start the text-to-speech conversion process.
  * step 6: once the conversion is complete, you can export the audio files for further use.

segment table:
___
  * click the table header "segment [20 token limit]" to adjust the segment length.
  * click "update segmention" to apply the changes.(this will override the whole segmention table)
  * after updating the segment length, you can manually adjust individual segments if needed.
  * you can also delete segments by selecting them and press the delete key.
  * to select multiple segments, hold the left-shift key while clicking on the segments you want to select.
)";

    std::optional<std::unordered_map<std::string, std::string>> from_json(const std::string& json_file) {
        std::ifstream ifs(json_file, std::ios::binary);
        if (!ifs.is_open()) {
            return std::nullopt;
        }

        nlohmann::json j;
        ifs >> j;
        ifs.close();

        try {
            std::optional<std::unordered_map<std::string, std::string>> ret = std::nullopt;
            for (const auto& i : j) {
                language_list_[i["code"]] = i["name"];
                if ( i["code"] == current_lang_ ) {
                    ret = i["translation"];
                }
            }
            return ret;
        } catch (const std::exception& e) {}
        return std::nullopt;
    }

public:
    static Localization& get() {
        static Localization instance;
        return instance;
    }

    void set_language(const std::string& lang) {
        current_lang_ = lang;
        auto data = from_json("languages.json");
        if ( data.has_value() ) {
            translations_ = data.value();
        } else {
            translations_.clear();
            translations_["help_text"] = help_text_en_;
        }
    }

    const char* tr(const char* key) {
        auto it = translations_.find(key);
        return (it != translations_.end()) ? it->second.c_str() : key;
    }

    std::unordered_map<std::string, std::string> language_list() const {
       return language_list_;
    }

    const char* model_folder_structure() {
        return model_folder_structure_;
    }

    const char* model_info() {
        return model_info_;
    }

    const char* model_structure_info() {
        return model_structure_info_;
    }

private:
    std::string current_lang_ = "en";
    std::unordered_map<std::string, std::string> translations_;
    std::unordered_map<std::string, std::string> language_list_{{"en", "English"}};
};

#define TR(key) Localization::get().tr(key)

namespace imgui_scoped {

    static void TableTextCentered(const char* text) {
        float cell_width = ImGui::GetContentRegionAvail().x;
        float text_width = ImGui::CalcTextSize(text).x;
        
        // Prevent a negative offset if the text is wider than the column
        float offset_x = (cell_width - text_width) * 0.5f;
        if (offset_x > 0.0f) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset_x);
        }
        
        ImGui::Text("%s", text);
    }

    class NonCopyable {
    protected:
        NonCopyable() = default;
        ~NonCopyable() = default;
        NonCopyable(const NonCopyable&) = delete;
        NonCopyable& operator=(const NonCopyable&) = delete;
        NonCopyable(NonCopyable&&) = default;
        NonCopyable& operator=(NonCopyable&&) = default;
    };

    // ==========================================
    // 1. Style Color & Style Var 作用域
    // ==========================================

    /// Push/PopStyleColor
    class StyleColor : private NonCopyable {
    public:
        StyleColor(ImGuiCol idx, ImU32 col, bool condition = true)
            : m_Count(condition ? 1 : 0) {
            if (condition) ImGui::PushStyleColor(idx, col);
        }

        StyleColor(ImGuiCol idx, const ImVec4& col, bool condition = true)
            : m_Count(condition ? 1 : 0) {
            if (condition) ImGui::PushStyleColor(idx, col);
        }

        StyleColor(std::initializer_list<std::pair<ImGuiCol, ImVec4>> styles)
            : m_Count(static_cast<int>(styles.size())) {
            for (const auto& style : styles) {
                ImGui::PushStyleColor(style.first, style.second);
            }
        }

        ~StyleColor() {
            if (m_Count > 0) ImGui::PopStyleColor(m_Count);
        }

    private:
        int m_Count = 0;
    };

    /// Push/PopStyleVar
    class StyleVar : private NonCopyable {
    public:
        StyleVar(ImGuiStyleVar idx, float val, bool condition = true)
            : m_Count(condition ? 1 : 0) {
            if (condition) ImGui::PushStyleVar(idx, val);
        }

        StyleVar(ImGuiStyleVar idx, const ImVec2& val, bool condition = true)
            : m_Count(condition ? 1 : 0) {
            if (condition) ImGui::PushStyleVar(idx, val);
        }

        ~StyleVar() {
            if (m_Count > 0) ImGui::PopStyleVar(m_Count);
        }

    private:
        int m_Count = 0;
    };


    /// Push/PopID
    class ID : private NonCopyable {
    public:
        explicit ID(const char* str_id) { ImGui::PushID(str_id); }
        explicit ID(const char* str_id_begin, const char* str_id_end) { ImGui::PushID(str_id_begin, str_id_end); }
        explicit ID(const void* ptr_id) { ImGui::PushID(ptr_id); }
        explicit ID(int int_id) { ImGui::PushID(int_id); }

        ~ID() { ImGui::PopID(); }
    };


    /// Push/PopFont
    class Font : private NonCopyable {
    public:
        explicit Font(ImFont* font) : m_Active(font != nullptr) {
            if (m_Active) ImGui::PushFont(font);
        }

        ~Font() {
            if (m_Active) ImGui::PopFont();
        }

    private:
        bool m_Active = false;
    };

    class FontSize : private NonCopyable {
    public:
        explicit FontSize(float size) : m_Active(size > 0) {
            if (m_Active) ImGui::PushFont(NULL, size);
        }

        ~FontSize() {
            if (m_Active) ImGui::PopFont();
        }

    private:
        bool m_Active = false;
    };

    /// SetNextItemWidth / Push/PopItemWidth
    class ItemWidth : private NonCopyable {
    public:
        explicit ItemWidth(float item_width) { ImGui::PushItemWidth(item_width); }
        ~ItemWidth() { ImGui::PopItemWidth(); }
    };

    /// Push/PopTextWrapPos
    class TextWrapPos : private NonCopyable {
    public:
        explicit TextWrapPos(float wrap_local_pos_x = 0.0f) { ImGui::PushTextWrapPos(wrap_local_pos_x); }
        ~TextWrapPos() { ImGui::PopTextWrapPos(); }
    };

    /// Begin/EndChild
    class Child : private NonCopyable {
    public:
        Child(const char* str_id, const ImVec2& size = ImVec2(0, 0), bool border = false, ImGuiWindowFlags flags = 0) {
            m_Open = ImGui::BeginChild(str_id, size, border, flags);
        }

        Child(ImGuiID id, const ImVec2& size = ImVec2(0, 0), bool border = false, ImGuiWindowFlags flags = 0) {
            m_Open = ImGui::BeginChild(id, size, border, flags);
        }

        ~Child() {
            ImGui::EndChild();
        }

        explicit operator bool() const { return m_Open; }

    private:
        bool m_Open = false;
    };

    /// BeginTable/EndTable
    class Table : private NonCopyable {
    public:
        Table(const char* str_id, int columns, ImGuiTableFlags flags = 0, const ImVec2& outer_size = ImVec2(0.0f, 0.0f), float inner_width = 0.0f) {
            m_Open = ImGui::BeginTable(str_id, columns, flags, outer_size, inner_width);
        }

        ~Table() {
            if (m_Open) ImGui::EndTable();
        }

        explicit operator bool() const { return m_Open; }

    private:
        bool m_Open = false;
    };

    /// Indent/Unindent
    class Indent : private NonCopyable {
    public:
        explicit Indent(float indent_w = 0.0f) : m_Width(indent_w) { ImGui::Indent(m_Width); }
        ~Indent() { ImGui::Unindent(m_Width); }

    private:
        float m_Width;
    };

    class Group : private NonCopyable {
    public:
        Group() { ImGui::BeginGroup(); }
        ~Group() { ImGui::EndGroup(); }
    };

#if defined(IMGUI_VERSION_NUM) && IMGUI_VERSION_NUM >= 18400
    class Disabled : private NonCopyable {
    public:
        explicit Disabled(bool disabled = true) : m_Disabled(disabled) {
            if (m_Disabled) ImGui::BeginDisabled(true);
        }

        ~Disabled() {
            if (m_Disabled) ImGui::EndDisabled();
        }

    private:
        bool m_Disabled = false;
    };
#endif

} // namespace imgui_scoped