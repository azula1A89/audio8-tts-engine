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

class Localization {
public:
    static Localization& get() {
        static Localization instance;
        return instance;
    }

    void set_language(const std::string& lang) {
        current_lang = lang;

        if (lang == "zh") {
            translations = {
                    {"decoder thread", "解码线程"},
                    {"fast ar thread", "快速AR线程"},
                    {"font scale", "字体缩放"},
                    {"theme", "主题"},
                    {"encoder thread", "编码线程"},
                    {"user interface", "用户界面"},
                    {"cancel", "取消"},
                    {"ok", "确定"},
                    {"choose", "选择"},
                    {"audio file path", "音频文件路径"},
                    {"slow ar thread", "慢速AR线程"},
                    {"voice name", "声音名称"},
                    {"CPU", "CPU"},
                    {"Generating..", "生成中..."},
                    {"choose models path", "选择模型路径"},
                    {"resume", "继续"},
                    {"recent", "最近"},
                    {"run", "运行"},
                    {"export", "导出"},
                    {"choose folder", "选择文件夹"},
                    {"segment: [ {} token limit ]", "分段: [每段最多{}个词元]"},
                    {"sample rate", "采样率"},
                    {"settings", "设置"},
                    {"GPU", "GPU"},
                    {"model", "模型"},
                    {"Loading..", "加载中..."},
                    {"update segmentation", "更新分段"},
                    {"max audio length(second)", "最大音频长度(秒)"},
                    {"transcript", "转录文本"},
                    {"session", "会话"},
                    {"play", "播放"},
                    {"registration", "注册"},
                    {"edit trigger segmention", "文本改变触发分段"},
                    {"done", "已完成"},
                    {"todo", "待处理"},
                    {"voices", "音色"},
                    {"seq", "序号"},
                    {"update segmention", "更新分段"},
                    {"status", "状态"},
                    {"new", "新建"},
                    {"text", "文本"},
                    {"You can download model from:", "你可以从这里下载模型文件:"},
                    {"export to folder", "导出到目录"},
                    {"help", "帮助"},
                    {"language", "语言"},
                    {"stop", "停止"}
            };
        } else if (lang == "ja") {
            translations = {
                    {"decoder thread", "デコーダースレッド"},
                    {"fast ar thread", "高速ARスレッド"},
                    {"font scale", "フォントスケール"},
                    {"theme", "テーマ"},
                    {"encoder thread", "エンコーダースレッド"},
                    {"user interface", "ユーザーインターフェース"},
                    {"cancel", "キャンセル"},
                    {"ok", "OK"},
                    {"choose", "選択"},
                    {"audio file path", "オーディオファイルパス"},
                    {"slow ar thread", "低速ARスレッド"},
                    {"voice name", "音声名"},
                    {"CPU", "CPU"},
                    {"Generating..", "生成中..."},
                    {"choose models path", "モデルパスを選択"},
                    {"resume", "再開"},
                    {"recent", "最近の"},
                    {"run", "実行"},
                    {"export", "エクスポート"},
                    {"choose folder", "フォルダを選択"},
                    {"segment: [ {} token limit ]", "セグメント: [ {} トークン制限 ]"},
                    {"sample rate", "サンプルレート"},
                    {"settings", "設定"},
                    {"GPU", "GPU"},
                    {"model", "モデル"},
                    {"Loading..", "読み込み中..."},
                    {"update segmentation", "セグメンテーションを更新する"},
                    {"max audio length(second)", "最大オーディオ長(秒)"},
                    {"transcript", "トランスクリプト"},
                    {"session", "セッション"},
                    {"play", "再生する"},
                    {"registration", "登録する"},
                    {"edit trigger segmention", "テキスト変更でセグメンテーションをトリガーする"},
                    {"done", "完了"},
                    {"todo", "未処理"},
                    {"voices", "音声"},
                    {"seq", "シーケンス"},
                    {"update segmention", "セグメンテーションを更新する"},
                    {"status", "ステータス"},
                    {"new", "新規"},
                    {"text", "テキスト"},
                    {"You can download model from:", "モデルをダウンロードできます:"},
                    {"export to folder", "フォルダにエクスポート"},
                    {"help", "ヘルプ"},
                    {"language", "言語"},
                    {"stop", "停止"}
            };
        } else if (lang == "ko") {
            translations = {
                    {"decoder thread", "디코더 스레드"},
                    {"fast ar thread", "빠른 AR 스레드"},
                    {"font scale", "글꼴 크기"},
                    {"theme", "테마"},
                    {"encoder thread", "인코더 스레드"},
                    {"user interface", "사용자 인터페이스"},
                    {"cancel", "취소"},
                    {"ok", "확인"},
                    {"choose", "선택"},
                    {"audio file path", "오디오 파일 경로"},
                    {"slow ar thread", "느린 AR 스레드"},
                    {"voice name", "음성 이름"},
                    {"CPU", "CPU"},
                    {"Generating..", "생성 중..."},
                    {"choose models path", "모델 경로 선택"},
                    {"resume", "재개"},
                    {"recent", "최근"},
                    {"run", "실행"},
                    {"export", "내보내기"},
                    {"choose folder", "폴더 선택"},
                    {"segment: [ {} token limit ]", "세그먼트: [ {} 토큰 제한 ]"},
                    {"sample rate", "샘플링 속도"},
                    {"settings", "설정"},
                    {"GPU", "GPU"},
                    {"model", "모델"},
                    {"Loading..", "로딩 중..."},
                    {"update segmentation", "세그먼트 업데이트"},
                    {"max audio length(second)", "최대 오디오 길이(초)"},
                    {"transcript", "전사"},
                    {"session", "세션"},
                    {"play", "재생"},
                    {"registration", "등록"},
                    {"edit trigger segmention", "텍스트 변경 시 세그먼트 트리거"},
                    {"done", "완료"},
                    {"todo", "할 일"},
                    {"voices", "음성"},
                    {"seq", "순서"},
                    {"update segmention", "세그먼트 업데이트"},
                    {"status", "상태"},
                    {"new", "새로 만들기"},
                    {"text", "텍스트"},
                    {"You can download model from:", "모델을 다운로드할 수 있습니다:"},
                    {"export to folder", "폴더로 내보내기"},
                    {"help", "도움말"},
                    {"language", "언어"},
                    {"stop", "중지"}
            };

        } else {
            translations.clear();
        }
    }

    const char* tr(const char* key) {
        auto it = translations.find(key);
        translations[key] = (it != translations.end()) ? it->second.c_str() : key;
        return translations[key].c_str();
    }

    std::unordered_map<std::string, std::string> language_list() const {
       return {
            {"en", "English"},
            {"zh", "中文"},
            {"ja", "日本語"},
            {"ko", "한국어"}
        };
    }

private:
    std::string current_lang = "en";
    std::unordered_map<std::string, std::string> translations;
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