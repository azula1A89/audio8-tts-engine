#pragma once

#include <imgui.h>
#include <utility>

namespace imgui_scoped {

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