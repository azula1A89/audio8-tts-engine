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


#include "main.hpp"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <future>
#include <stdio.h>
#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <nfd.hpp>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui_markdown.h>
#include <imgui_stdlib.h>
#include <imgui_theme.h>
#include <imgui_internal.h>
#include <imspinner_compat.h>
#include <imspinner_text.h>

#include <miniaudio_impl.hpp>

#include <audio8_engine.hpp>

using namespace std::chrono_literals;
using json = nlohmann::json;

const char* model_url = "https://huggingface.co/Edge0/Audio8-TTS-Preview-0.6B-ONNX-INT4/tree/main";
const char* backup_url = "https://modelscope.ai/models/Edge0/Audio8-TTS-Preview-0.6B-ONNX-INT4/files";

class UserSettings {
private:
    struct settings_s {
        std::string model_folder = "models";
        std::string session_folder = "sessions";
        std::string export_folder = "sessions";
        std::string default_voice = "anthony";
        std::string font = "fonts/LXGWWenKai-Regular.ttf";
        std::string language = "en";
        float font_scale_main = 1.0f;
        int theme = 0;
        int export_length = 1200;
        int export_sample_rate = 44100;
        int segment_max_token = 25;
        bool enable_automatic_segmentation_trigger = false;
        int window_width = 1500;
        int window_height = 610;
        RuntimeConfig runtime_config;
    } settings_;

public:
    UserSettings() {}

    static void* read_open_func(ImGuiContext* ctx, ImGuiSettingsHandler* handler, const char* name) {
        // ImGui expects [TypeName][Name]. We use "Data" as the sub-name.
        if (strcmp(name, "Data") == 0) {
            return (void*)handler->UserData;
        }
        return nullptr;
    }

    static void read_line_func(ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
        settings_s* setting = static_cast<settings_s*>(entry);
        
        const char* eq_pos = strchr(line, '=');
        if (!eq_pos) return;

        std::string key(line, eq_pos - line);
        std::string value(eq_pos + 1);

        try {
            if (key == "model_folder") setting->model_folder = value;
            else if (key == "session_folder") setting->session_folder = value;
            else if (key == "export_folder") setting->export_folder = value;
            else if (key == "default_voice") setting->default_voice = value;
            else if (key == "font") setting->font = value;
            else if (key == "font_scale_main") setting->font_scale_main = std::stof(value);
            else if (key == "language") setting->language = value;
            else if (key == "theme") setting->theme = std::stoi(value);
            else if (key == "export_length") setting->export_length = std::stoi(value);
            else if (key == "export_sample_rate") setting->export_sample_rate = std::stoi(value);
            else if (key == "segment_max_token") setting->segment_max_token = std::stoi(value);
            else if (key == "enable_automatic_segmentation_trigger") setting->enable_automatic_segmentation_trigger = std::stoi(value) != 0;
            else if (key == "window_width") setting->window_width = std::stoi(value);
            else if (key == "window_height") setting->window_height = std::stoi(value);
            else if (key == "slow_ar_thread_num") setting->runtime_config.slow_ar_thread_num = std::stoi(value);
            else if (key == "fast_ar_thread_num") setting->runtime_config.fast_ar_thread_num = std::stoi(value);
            else if (key == "codec_encoder_thread_num") setting->runtime_config.codec_encoder_thread_num = std::stoi(value);
            else if (key == "codec_decoder_thread_num") setting->runtime_config.codec_decoder_thread_num = std::stoi(value);
            else if (key == "execution_provider") setting->runtime_config.execution_provider = static_cast<ExecutionProvider>(std::stoi(value));
        } catch (const std::exception&) {
            fmt::print("Failed to parse setting: {}\n", key);
        }
    }

    static void write_all_func(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
        settings_s* setting = static_cast<settings_s*>(handler->UserData);
        
        // ImGui expects [TypeName][Name]. We use "Data" as the sub-name.
        buf->appendf("[%s][Data]\n", handler->TypeName);
        
        buf->appendf("model_folder=%s\n", setting->model_folder.c_str());
        buf->appendf("session_folder=%s\n", setting->session_folder.c_str());
        buf->appendf("export_folder=%s\n", setting->export_folder.c_str());
        buf->appendf("default_voice=%s\n", setting->default_voice.c_str());
        buf->appendf("font=%s\n", setting->font.c_str());
        buf->appendf("font_scale_main=%.2f\n", setting->font_scale_main);
        buf->appendf("language=%s\n", setting->language.c_str());
        buf->appendf("theme=%d\n", setting->theme);
        buf->appendf("export_length=%d\n", setting->export_length);
        buf->appendf("export_sample_rate=%d\n", setting->export_sample_rate);
        buf->appendf("segment_max_token=%d\n", setting->segment_max_token);
        buf->appendf("enable_automatic_segmentation_trigger=%d\n", setting->enable_automatic_segmentation_trigger);
        buf->appendf("window_width=%d\n", setting->window_width);
        buf->appendf("window_height=%d\n", setting->window_height);
        buf->appendf("slow_ar_thread_num=%d\n", setting->runtime_config.slow_ar_thread_num);
        buf->appendf("fast_ar_thread_num=%d\n", setting->runtime_config.fast_ar_thread_num);
        buf->appendf("codec_encoder_thread_num=%d\n", setting->runtime_config.codec_encoder_thread_num);
        buf->appendf("codec_decoder_thread_num=%d\n", setting->runtime_config.codec_decoder_thread_num);
        buf->appendf("execution_provider=%d\n", static_cast<int>(setting->runtime_config.execution_provider));
        buf->append("\n"); 
    }

    void initialize() {
        ImGuiSettingsHandler ini_handler;
        ini_handler.TypeName = "UserSettings";
        ini_handler.TypeHash = ImHashStr("UserSettings");
        ini_handler.ReadOpenFn = read_open_func;
        ini_handler.ReadLineFn = read_line_func;
        ini_handler.WriteAllFn = write_all_func;
        ini_handler.UserData = &settings_;
        ImGui::AddSettingsHandler(&ini_handler);
        ImGui::LoadIniSettingsFromDisk(ImGui::GetIO().IniFilename);
    }

    settings_s& get() {
        return settings_;
    }

    // synchronize the settings with ImGui's INI file
    void sync() {
        ImGuiContext* ctx = ImGui::GetCurrentContext();
        if (ctx) {
            ImGui::MarkIniSettingsDirty();
        }
    }
};

class Session {
public:
    struct config_item_s {
        int id; 
        std::string text;
        std::string voice;
        bool selected;
        bool done;
        size_t hash() const {
            return std::hash<std::string>{}(text) ^ std::hash<std::string>{}(voice);
        }
    };
    
private:
    std::filesystem::path root_;
    std::string cache_;
    size_t track_count_;
    size_t request_count_;
    std::vector<config_item_s> configs_;
    std::string origin_text_ = "";
    std::atomic<int> done_id_ = -1;
    
public:
    Session(std::filesystem::path root = "sessions", 
        std::string cache = fmt::format("{:%F_%H-%M-%S}", fmt::localtime(std::time(nullptr))), 
        size_t track_count = 0) :
        root_(root), cache_(cache), track_count_(track_count), request_count_(0) {
        auto cache_path = root_ / cache_;
        if ( !std::filesystem::is_directory(cache_path) ) {
            std::filesystem::create_directories(cache_path);
        }
        rebuild_playlist();
    };

    ~Session() {
        stop_file();
        save();
    }

    static std::unique_ptr<Session> from_json(std::filesystem::path json_file) {
        std::ifstream ifs(json_file, std::ios::binary);
        if (!ifs.is_open()) {
            return nullptr;
        }
        json j;
        ifs >> j;
        ifs.close();

        auto session = std::make_unique<Session>(j["root"], j["cache"], j["track_count"]);
        session->origin_text_ = j["origin_text"];
        for (const auto& cfg : j["configs"]) {
            Session::config_item_s config;
            config.id = cfg["id"];
            config.text = cfg["text"];
            config.voice = cfg["voice"];
            config.selected = cfg["selected"];
            config.done = cfg["done"];
            session->configs().push_back(config);
        }
        session->rebuild_playlist();
        return session;
    }

    void export_audio( const int& max_length_sec, const char* path, uint32_t sample_rate = 44100U) {
        miniaudio_impl::export_audio(max_length_sec, path, sample_rate);
    }

    size_t& request_count() {
        return request_count_;
    }
    
    std::vector<config_item_s>& configs() {
        return configs_;
    }

    std::string& text() {
        return origin_text_;
    }

    void rebuild_playlist() {
        auto cache_path = root_ / cache_;
        if (!std::filesystem::is_directory(cache_path)) {
            return;
        }

        miniaudio_impl::track_clear();
        for ( const auto& cfg  : configs_) {
            auto wav_file = root_ / cache_ / fmt::format("{}.wav", cfg.hash());
            if (std::filesystem::exists(wav_file)) {
                miniaudio_impl::track_add(wav_file.string());
            }
        }
    }

    bool start_playlist() {
        miniaudio_impl::play();
        return true;
    }

    void pause_playlist() {
        miniaudio_impl::pause();
    }

    void stop_playlist() {
        miniaudio_impl::stop();
    }

    bool is_playing_list() {
        return miniaudio_impl::is_playing_list();
    }

    bool play_id(size_t hash) {
        size_t count = track_count();
        if ( count == 0 ) return false;

        auto wav_file = root_ / cache_ / fmt::format("{}.wav", hash);
        if (!std::filesystem::exists(wav_file)) {
            return false;
        }

        miniaudio_impl::stop_file();
        return miniaudio_impl::play_file(wav_file.string().c_str());
    }

    void stop_file() {
        miniaudio_impl::stop_file();
    }

    bool is_playing_file() {
        return miniaudio_impl::is_playing_file();
    }

    // save PCM data to a wav file
    void save_to_wav(const std::vector<float> &track, const int& id) {
        // find the configs_ item whose ID matches the given ID.
        auto cfg = std::find_if(configs_.begin(), configs_.end(), [&id](const config_item_s& item){ return item.id == id; });
        if ( cfg == configs_.end() ) return;

        auto wav_file = root_ / cache_ / fmt::format("{}.wav", cfg->hash());
        miniaudio_impl::wav_write(track.data(), track.size(), wav_file.string().c_str());
        track_count_ = track_count();
        done_id_.store(id);
    };
    
    size_t track_count() {
        auto cache_path = root_ / cache_;
        if (!std::filesystem::is_directory(cache_path)) {
            return 0;
        }

        std::size_t count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(cache_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".wav") {
                count++;
            }
        }
        return count;
    };

    void refresh_status() {

        // random done status update
        int done_id = done_id_.load();
            done_id_.store(-1);
        if (done_id >= 0 ) {
            set_is_done(done_id);
            rebuild_playlist();
        }
    }

    void set_is_done(const int& id) {
        auto search = std::find_if(configs_.begin(), configs_.end(), [&id](const config_item_s& item){ return item.id == id; });
        if ( search != configs_.end() ) {
            search->done = true;
        }
    }

    void set_done_value(bool v) {
        for (auto& i : configs_) { i.done = i.selected ? v : i.done; }
    }

    void set_default_voice (const std::string& default_voice){ 
        for (auto& i : configs_) { i.voice = default_voice; }
    };

    void selected_all() {
        for (auto& i : configs_) { i.selected = true; }
    };

    void unselected_all() { 
        for (auto& i : configs_) { i.selected = false; }
    };

    void toggle_selected(int id) {
        auto search = std::find_if(configs_.begin(), configs_.end(), [&id](const config_item_s& item){ return item.id == id; });
        if ( search != configs_.end() ) {
            search->selected = !search->selected;
        }
    }

    void delete_selected() {
        std::erase_if(configs_, [](const auto& item){ return item.selected; });
    }

    float progress() {
        float x = 0.0f;
        float request_count = configs_.size();
        if ( request_count > 0 ) {
            x = track_count_;
            x /= request_count;
        }
        return x;
    }

    // save configs to session root folder, as {runtime_str_}.json
    void save() {
        json save;
        save["root"] = root_;
        save["cache"] = cache_;
        save["track_count"] = track_count();
        save["origin_text"] = origin_text_;
        save["configs"] = json::array();
        for (size_t i = 0; i < configs_.size(); ++i) {
            const auto& config = configs_[i];
            json cfg;
            cfg["id"] = config.id;
            cfg["text"] = config.text;
            cfg["voice"] = config.voice;
            cfg["selected"] = config.selected;
            cfg["done"] = config.done;
            save["configs"].push_back(cfg);
        }

        if (!save.empty()) {
            try{
                auto json_file = root_ / fmt::format("{}.json", cache_);
                std::ofstream ostrm(json_file, std::ios::binary);
                ostrm.write(save.dump(4).c_str(), save.dump(4).size());
            }catch(const std::exception& e){
                fmt::print("while save:{}\n", e.what());
            }
        }
    }

};

class GuiContext {
public:
    UserSettings ini;
    std::unique_ptr<Audio8Engine> engine;
    std::unique_ptr<Session> session;
    Audio8ModelPaths paths;
    std::filesystem::path session_root_path;
    std::vector<std::string> voices;

    std::future<void> engine_status = {};
    std::future<void> registration_status = {};
    std::future<std::optional<std::vector<std::string>>> split_text_status = {};
    std::future<void> cancel_status = {};
    std::future<std::unique_ptr<Session>> import_status = {};
    std::future<void> export_status = {};

    bool is_initialized = false;
    bool is_loading = false;
    bool is_segmenting = false;
    bool is_encoding = false;
    bool is_cancelling = false;
    bool is_importing = false;
    bool is_exporting = false;

    float generate_progress = 0.0f;
    int generate_id = -1;
    int decode_id = -1;
    float decode_eta = -1.0f;
    std::atomic<int> request_done = 0;

public:
    GuiContext() {};

    ~GuiContext() = default;

    void initialize() {
        ini.initialize();
        Localization::get().set_language(ini.get().language);
        engine = std::make_unique<Audio8Engine>(ini.get().runtime_config);
        session = nullptr;
        paths = Audio8ModelPaths(ini.get().model_folder);
        session_root_path = ini.get().session_folder;
    }

    bool is_busy() const {
        return is_loading || is_segmenting || is_encoding || is_cancelling || is_importing || is_exporting;
    }
};

std::string choose_folder();
std::string choose_audio_path();
std::string choose_font_path();
void imgui_parent_window();
void render_menubar( GuiContext& ctx );
void render_progressbar( GuiContext& ctx );
void render_segmention_table( GuiContext& ctx );
void render_choose_model_popup( GuiContext& ctx );
void render_registration_popup( GuiContext& ctx );
void render_main_window( GuiContext& ctx );
void render_frame(GLFWwindow* window) {
    auto ctx = static_cast<GuiContext*>(glfwGetWindowUserPointer(window));

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    //root window
    imgui_parent_window();

    // default model path invalid
    if ( !ctx->paths.check() ) {
        // ask for models path
        ImGui::OpenPopup(TR("choose models path"));
        render_choose_model_popup(*ctx);
    }else if ( !ctx->is_initialized ) {
        if( ctx->engine->initialize(ctx->paths.root) )// engine initialize
            ctx->is_initialized = true;

        ctx->engine->set_generate_callback([ctx](float progress_in, int id_in){
            ctx->generate_progress = progress_in;
            ctx->generate_id = id_in;
        });

        ctx->engine->set_decoder_callback([ctx](std::vector<float> pcm_in, const int id_in){
            if( ctx->session ) ctx->session->save_to_wav(pcm_in, id_in);
            ctx->request_done.fetch_add(1);
        });

        ctx->engine->set_decoder_eta_callback([ctx](float eta_in, const int id_in){
            ctx->decode_eta = eta_in;
            ctx->decode_id = id_in;
        });

        // auto preload
        if ( ctx->is_initialized ) {
            ctx->is_loading = true;
            ctx->engine_status = std::async(std::launch::async,[ctx](){
                ctx->engine->preload_model();
            });
        }
    }

    if ( ctx->voices.empty() ) {
        ctx->voices = ctx->engine->list_voices();
    }

    if ( ctx->session ) {
        ctx->session->refresh_status();
    }

    // main window
    render_main_window(*ctx);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
}

int main(int argc, char** argv)
{
    const char* glsl_version = "#version 330";
    if (glfwInit() == GL_FALSE){
        return GL_FALSE;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    GLFWwindow*main_window = glfwCreateWindow(900, 300, "audio8-tts-engine", NULL, NULL);
    if (main_window == NULL) {
        glfwTerminate();
        return GL_FALSE;
    }

    glfwMakeContextCurrent(main_window);
    glfwSwapInterval(1);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    GuiContext ctx;
    ctx.initialize();
    
    glfwSetWindowUserPointer(main_window, &ctx);
    glfwSetWindowSize(main_window, ctx.ini.get().window_width, ctx.ini.get().window_height);

    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    
    ImGuiTheme::ApplyTweakedTheme(static_cast<ImGuiTheme::ImGuiTheme_>(ctx.ini.get().theme));

    std::string font_path = ctx.ini.get().font;
    if ( std::filesystem::is_regular_file( font_path ) ) {
        ImGui::GetIO().FontDefault = io.Fonts->AddFontFromFileTTF(font_path.c_str());
    }

    float xscale, yscale;
    glfwGetWindowContentScale((GLFWwindow *) main_window, &xscale, &yscale);

    ImGuiStyle& style = ImGui::GetStyle();
    style.FontScaleDpi = std::max(xscale, yscale);
    style.FontScaleMain = ctx.ini.get().font_scale_main;

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(main_window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    glfwSetWindowRefreshCallback(main_window, [](GLFWwindow* window){
        render_frame(window);
    });

    glfwSetFramebufferSizeCallback(main_window, [](GLFWwindow* window, int width, int height){
        auto ctx = static_cast<GuiContext*>(glfwGetWindowUserPointer(window));
        ctx->ini.get().window_width = width;
        ctx->ini.get().window_height = height;
        ctx->ini.sync();
        glViewport(0, 0, width, height);
        render_frame(window);
    });

    // Main loop
    while ( glfwWindowShouldClose(main_window) == GL_FALSE || ctx.is_busy() )
    {
        // Poll for and process events
        glfwPollEvents();
        render_frame(main_window);
    }

    if ( ctx.session ) {
        ctx.session.reset();
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(main_window);
    glfwTerminate();
    return 0;
}

std::string choose_folder()
{
    std::string path="";
    NFD::Guard nfd_guard;
    NFD::UniquePath out_path;
    nfdresult_t result = NFD::PickFolder(out_path);
    if (result == NFD_OKAY){
        path = out_path.get();
    }
    return path;
}

std::string choose_audio_path()
{
    std::string path="";
    NFD::Guard nfd_guard;
    NFD::UniquePath out_path;
    nfdfilteritem_t filter_item[3] = {
        {"*", "wav,mp3,flac"}};
    std::string default_path = std::filesystem::current_path().string();
    nfdresult_t result = NFD::OpenDialog(out_path, filter_item, 1, default_path.c_str());
    if (result == NFD_OKAY){
        path = out_path.get();
    }
    return path;
}

std::string choose_font_path()
{
    std::string path="";
    NFD::Guard nfd_guard;
    NFD::UniquePath out_path;
    nfdfilteritem_t filter_item[1] = {
        {"TrueType Font", "ttf"}};
    std::string default_path = std::filesystem::current_path().string();
    nfdresult_t result = NFD::OpenDialog(out_path, filter_item, 1, default_path.c_str());
    if (result == NFD_OKAY){
        path = out_path.get();
    }
    return path;
}

void imgui_parent_window()
{
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |ImGuiWindowFlags_NoBackground;

    bool p_open = true;
    ImGui::Begin("MyDockSpace", &p_open, window_flags);

    ImGui::PopStyleVar(3);

    // DockSpace
    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_DockingEnable)
    {
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_AutoHideTabBar);
    }
    ImGui::End();
}

void render_menubar( GuiContext& ctx ) {
    ImGuiIO& io = ImGui::GetIO();

    // Menubar
    if( ctx.is_initialized ) {
        imgui_scoped::StyleVar popup_rounding(ImGuiStyleVar_PopupRounding, 6.0f);
        imgui_scoped::StyleVar item_speacing(ImGuiStyleVar_ItemSpacing, {20.0f, 5.0f});
        imgui_scoped::Disabled disable(ctx.is_loading);
        if (ImGui::BeginMenuBar()) {

            // session, export
            {
                imgui_scoped::Disabled disable(ctx.engine->is_busy());
                if (ImGui::BeginMenu(TR("session"))) {
                    if ( ImGui::MenuItem(TR("new")) ) {
                        if ( ctx.session ) {
                            ctx.session.reset();
                        }
                        ctx.session = make_unique_nothrow<Session>();
                    }

                    if ( ImGui::BeginMenu(TR("recent")) ) {
                        static std::string session_recent = "";
                        if (std::filesystem::is_directory(ctx.session_root_path)) {
                            std::vector<std::string> recent_sessions;
                            for (const auto& entry : std::filesystem::directory_iterator(ctx.session_root_path)) {
                                if (entry.is_regular_file() && entry.path().extension() == ".json") {
                                    recent_sessions.push_back(entry.path().filename().string());
                                }
                            }
                            std::sort(recent_sessions.begin(), recent_sessions.end());
                            for (const auto& filename : recent_sessions) {
                                if ( ImGui::MenuItem(filename.c_str(), nullptr, session_recent == filename) ) {
                                    if ( !ctx.is_importing ) {
                                        ctx.is_importing = true;
                                        session_recent = filename;
                                        ctx.import_status = std::async(std::launch::async, [&ctx, filename]() -> std::unique_ptr<Session> {
                                            return Session::from_json((ctx.session_root_path / filename).string().c_str());
                                        });
                                    }
                                }
                            }
                        }
                        ImGui::EndMenu();
                    }

                    {
                        int& export_length = ctx.ini.get().export_length;
                        imgui_scoped::Disabled disable(ctx.session == nullptr);
                        if ( ImGui::BeginMenu(TR("export")) ) {
                            if (ImGui::DragInt(TR("max audio length(second)"), &export_length, 1.0f, 1)) {
                                ctx.ini.sync();
                            }

                            int& sample_rate = ctx.ini.get().export_sample_rate;
                            if (ImGui::DragInt(TR("sample rate"), &sample_rate, 100.0f, 8000, 96000)) {
                                ctx.ini.sync();
                            }

                            std::string folder = ctx.ini.get().export_folder;
                            if (ImGui::InputTextWithHint("##export to folder", TR("export to folder"), &folder)) {
                                if (!folder.empty() && std::filesystem::is_directory(folder)) {
                                    ctx.ini.get().export_folder = folder;
                                    ctx.ini.sync();
                                }
                            }
                            ImGui::SameLine();
                            if (ImGui::Button(TR("choose folder"))) {
                                folder = choose_folder();
                                if (!folder.empty() && std::filesystem::is_directory(folder)) {
                                    ctx.ini.get().export_folder = folder;
                                    ctx.ini.sync();
                                }
                            }
                            if ( ImGui::Button(TR("export")) ) {
                                if ( ctx.session->is_playing_list() ) {
                                    ctx.session->stop_playlist();
                                }
                                if ( !ctx.is_exporting ) {
                                    ctx.is_exporting = true;
                                    ctx.export_status = std::async(std::launch::async, [&ctx]() {
                                        ctx.session->rebuild_playlist();
                                        ctx.session->export_audio(ctx.ini.get().export_length, ctx.ini.get().export_folder.c_str(), ctx.ini.get().export_sample_rate);
                                    });
                                }
                            }
                            ImGui::EndMenu();
                        }
                    }

                    ImGui::EndMenu();
                }
            }

            // run, cancel
            {
                imgui_scoped::Disabled disable(ctx.session == nullptr);
                std::string status_text = TR(ctx.engine->is_busy() ? "cancel" : "run");
                if ( ImGui::MenuItem(status_text.c_str()) ) {
                    if ( !ctx.engine->is_busy() ) {
                        ctx.request_done.store(0);
                        ctx.session->request_count() = 0;

                        TTSRequest request{};
                        for (const auto& item : ctx.session->configs()) {
                            if ( item.done ) { continue; } //skip processed item
                            request.id = item.id;
                            request.text = item.text;
                            request.voice_name = item.voice;
                            request.max_new_tokens = 1024;
                            ctx.engine->push(request);
                            ctx.session->request_count()++;
                        }
                    } else {
                        if ( !ctx.is_cancelling ) {
                            ctx.is_cancelling = true;
                            ctx.cancel_status = std::async(std::launch::async,[&](){
                                ctx.engine->cancel();
                            });
                        }
                    }
                }

                // play tracks
                {
                    // imgui_scoped::Disabled disable( session->is_playing_file() );
                    std::string status_txt = TR(ctx.session->is_playing_list() ? "stop":"play");
                    if ( ImGui::MenuItem(status_txt.c_str()) ) {
                        if( !ctx.session->is_playing_list() ) {
                            ctx.session->start_playlist();
                        } else {
                            ctx.session->stop_playlist();
                        }
                    }
                }

            }

            // voice registration (voice clone)
            {
                imgui_scoped::Disabled disable( ctx.is_loading );
                if (ImGui::MenuItem(TR("registration"))) {

                    ImGui::OpenPopup(TR("registration"));
                }
                render_registration_popup(ctx);
            }

            // settings
            if (ImGui::BeginMenu(TR("settings"))) {

                if(ImGui::BeginMenu(TR("user interface"))) {
                    auto i = ctx.ini.get().language;
                    const auto current_language_name = Localization::get().language_list()[i];
                    if ( ImGui::BeginCombo(TR("language"), current_language_name.c_str()) ) {
                        for (const auto& [key, val] : Localization::get().language_list()) {
                            if ( ImGui::Selectable(val.c_str(), key == i) ) {
                                if ( ctx.ini.get().language != key ) {
                                    ctx.ini.get().language = key;
                                    ctx.ini.sync();
                                    Localization::get().set_language(key);
                                }
                            }
                        }
                        ImGui::EndCombo();
                    }

                    int& theme = ctx.ini.get().theme;
                    if( ImGui::BeginCombo(TR("theme"), ImGuiTheme::ImGuiTheme_Name(static_cast<ImGuiTheme::ImGuiTheme_>(theme))) ) {
                        for (int i = 0; i< ImGuiTheme::ImGuiTheme_Count; i++) {
                            imgui_scoped::ID id(i);
                            
                            if(ImGui::Selectable(ImGuiTheme::ImGuiTheme_Name((ImGuiTheme::ImGuiTheme_)i), theme == i)) {
                                theme = (ImGuiTheme::ImGuiTheme_)i;
                                ctx.ini.sync();
                                float size1 = ImGui::GetStyle().FontSizeBase;
                                float size2 = ImGui::GetStyle().FontScaleDpi;
                                float size3 = ImGui::GetStyle().FontScaleMain;
                                ImGuiTheme::ApplyTweakedTheme(static_cast<ImGuiTheme::ImGuiTheme_>(theme));
                                ImGui::GetStyle().FontSizeBase = size1;
                                ImGui::GetStyle().FontScaleDpi = size2;
                                ImGui::GetStyle().FontScaleMain = size3;
                            }
                        }
                        ImGui::EndCombo();
                    }

                    float& font_scale = ctx.ini.get().font_scale_main;
                    if (ImGui::DragFloat(TR("font scale"), &font_scale, 0.01f, 0.2f, 3.0f)) {
                        ImGui::GetStyle().FontScaleMain = font_scale;
                        ctx.ini.sync();
                    }

                    imgui_scoped::StyleVar var(ImGuiStyleVar_FrameRounding, 8);
                    static std::filesystem::path font = ctx.ini.get().font;
                    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
                    if ( ImGui::Button(font.filename().string().c_str()) ) {
                        font = choose_font_path();
                        font = font.empty()? ctx.ini.get().font : font.string();
                        if ( std::filesystem::is_regular_file(font) && 
                            font.extension().string() == ".ttf") {
                            ctx.ini.get().font = font.string();
                            ctx.ini.sync();
                            ImGui::GetIO().FontDefault = io.Fonts->AddFontFromFileTTF(font.string().c_str(), 16.0f);
                        }
                    }

                    ImGui::EndMenu();
                }

                if ( ImGui::BeginMenu(TR("onnxruntime")) ) {
                    uint32_t step = 1;
                    const ImGuiDataType_ type = ImGuiDataType_U32;
                    static bool value_changed = false;
                    static uint32_t n = std::thread::hardware_concurrency();
                    uint32_t& slowar_t_num = ctx.ini.get().runtime_config.slow_ar_thread_num;
                    uint32_t& fastar_t_num = ctx.ini.get().runtime_config.fast_ar_thread_num;
                    uint32_t& encoder_t_num = ctx.ini.get().runtime_config.codec_encoder_thread_num;
                    uint32_t& decoder_t_num = ctx.ini.get().runtime_config.codec_decoder_thread_num;
                    ExecutionProvider& ep = ctx.ini.get().runtime_config.execution_provider;

                    if (ImGui::RadioButton( TR("CPU"), ep == ExecutionProvider::CPU)) {
                        ep = ExecutionProvider::CPU;
                        value_changed = true;
                        ctx.ini.sync();
                    }
                    ImGui::SameLine();
                    if (ImGui::RadioButton( TR("GPU"), ep == ExecutionProvider::GPU)) {
                        ep = ExecutionProvider::GPU;
                        value_changed = true;
                        ctx.ini.sync();
                    }

                    if ( ImGui::InputScalar(TR("slow ar thread"), type, &slowar_t_num, &step) ) {
                        slowar_t_num = std::clamp(slowar_t_num, 1u, n);
                        value_changed = true;
                        ctx.ini.sync();
                    }
                    if ( ImGui::InputScalar(TR("fast ar thread"), type, &fastar_t_num, &step) ) {
                        fastar_t_num = std::clamp(fastar_t_num, 1u, n);
                        value_changed = true;
                        ctx.ini.sync();
                    }

                    if ( ImGui::InputScalar(TR("encoder thread"), type, &encoder_t_num, &step) ) {
                        encoder_t_num = std::clamp(encoder_t_num, 1u, n);
                        value_changed = true;
                        ctx.ini.sync();
                    }

                    if ( ImGui::InputScalar(TR("decoder thread"), type, &decoder_t_num, &step) ) {
                        decoder_t_num = std::clamp(decoder_t_num, 1u, n);
                        value_changed = true;
                        ctx.ini.sync();
                    }
                    if (value_changed) {
                        imgui_scoped::StyleVar var(ImGuiStyleVar_FrameRounding, 8);
                        if ( ImGui::Button(TR("apply"), ImVec2(-1.0f, 0.0f)) ) {
                            value_changed = false;
                            ctx.engine->cancel();
                            ctx.engine = std::make_unique<Audio8Engine>(ctx.ini.get().runtime_config);
                            ctx.is_initialized = false;
                        }
                    }
                    ImGui::EndMenu();
                }
                
                ImGui::EndMenu();
            }

            // help, about
            if (ImGui::BeginMenu(TR("help"))) 
            {
                static ImVec2 md_size = {1500, 500};
                ImGui::MarkdownConfig md_config{ 
                    NULL, NULL, NULL, NULL, 
                    { { io.FontDefault, true }, 
                    { io.FontDefault, true }, 
                    { io.FontDefault, false } }, 
                    NULL };
                if ( imgui_scoped::Child help = imgui_scoped::Child(TR("help"), md_size) ) {
                    ImGui::Markdown(TR("help_text"), std::string(TR("help_text")).length(), md_config);
                }
                ImGui::EndMenu();
            }

            ImGui::EndMenuBar();
        }
    }
}

void render_progressbar( GuiContext& ctx ) {
    // Progress bar
    {
        imgui_scoped::StyleVar frame_padding(ImGuiStyleVar_FramePadding, {5.0f, 0.0f});
        imgui_scoped::StyleVar frame_rounding(ImGuiStyleVar_FrameRounding, 6.0f);

        // "Loading" progress bar
        if ( ctx.is_loading ) {
            ImGui::ProgressBar(-1.0f * (float)ImGui::GetTime(), ImVec2(-1.0f, 0.0f), TR("Loading.."));
        }

        // "Encoding" progress bar
        if ( ctx.is_encoding ) {
            ImGui::ProgressBar(-1.0f * (float)ImGui::GetTime(), ImVec2(-1.0f, 0.0f), TR("Encoding.."));
        }

        // "Generating" "Total" progress bar
        if( ctx.is_initialized && ctx.engine->is_busy() && ctx.session ){
            ImGui::ProgressBar(ctx.generate_progress, ImVec2(-1.0f, 0.0f), TR("Generating.."));
            ImGui::ProgressBar(ctx.session->progress(), ImVec2(-1.0f, 0.0f), fmt::format("{}/{}", ctx.request_done.load(), ctx.session->request_count()).c_str());
        }

        {
            bool open = ctx.is_segmenting || ctx.is_cancelling || ctx.is_exporting || ctx.is_importing;
            if( open ) {
                if ( !ImGui::IsPopupOpen("my_spinner_popup") ) {
                    ImGui::OpenPopup("my_spinner_popup");
                }
            }

            const int popup_spinner_flag = ImGuiWindowFlags_NoDecoration 
                    | ImGuiWindowFlags_NoMove 
                    | ImGuiWindowFlags_NoBackground;

            // Always at center
            ImVec2 center = ImGui::GetMainViewport()->GetCenter();
            ImVec2 size = ImGui::GetMainViewport()->WorkSize;
            ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
            if (ImGui::BeginPopupModal("my_spinner_popup", &open, popup_spinner_flag)) {
                
                float r = (size.x > size.y ? size.x:size.y) * 0.5f * 0.1f;
                if ( r < 10 ) { ImGui::CloseCurrentPopup(); }
                auto color = ImGui::GetStyle().Colors[ImGuiCol_Text];                        
                ImSpinner::SpinnerRainbow("rainbow", r, 2.f, color, 8.f);
                
                ImGui::EndPopup();
            }
        }
    }
}

void render_segmention_table( GuiContext& ctx ) {
    // Text input + chunk info table
    if( ctx.is_initialized  && ctx.session) {
        std::string& origin_text = ctx.session->text();
        int& max_token = ctx.ini.get().segment_max_token;
        static bool enable_edit_trigger = false;
        static int last_max_token = -1;
        static int last_edit_count = -1;
        static int edit_count = 0;
        static bool btn_update_seg = false;
        bool disable_edit = ctx.engine->is_busy() || ctx.is_segmenting;

        imgui_scoped::Disabled disable(disable_edit);
        if ( ImGui::CollapsingHeader(TR("text")) ) {
            auto sz = ImGui::GetContentRegionAvail();
            edit_count += ImGui::InputTextMultiline("##text to speach", &origin_text,
                ImVec2(-FLT_MIN, sz.y * 0.25f), 
                0);
        }

        bool should_update = (last_edit_count != edit_count);
                should_update |= (last_max_token != max_token);
                should_update &= !ctx.is_loading;
                should_update &= !ctx.is_segmenting;
                should_update &= enable_edit_trigger;
                should_update |= btn_update_seg;
                
        if( should_update ) {
            btn_update_seg = false;
            ctx.is_segmenting = true;
            last_edit_count = edit_count;
            last_max_token = max_token;

            ctx.split_text_status = std::async(std::launch::async, [&origin_text, &ctx, &max_token](){
                return ctx.engine->split_text_by_tokens(origin_text, max_token);
            });
        }
        
        if( imgui_scoped::Child chunk_info = imgui_scoped::Child("##chunk info") ) {
            ImGuiTableColumnFlags table_flags = 0//ImGuiTableFlags_Borders
                            | ImGuiTableFlags_ScrollY
                            | ImGuiTableFlags_RowBg;
            ImGuiSelectableFlags select_flags = ImGuiSelectableFlags_SpanAllColumns 
                            | ImGuiSelectableFlags_AllowDoubleClick
                            | ImGuiSelectableFlags_AllowOverlap;
            ImGuiTableColumnFlags column_flags = ImGuiTableColumnFlags_WidthFixed;

            imgui_scoped::StyleVar f_padding(ImGuiStyleVar_FramePadding, {0.0f, 0.0f});
            imgui_scoped::StyleVar s_txt_align(ImGuiStyleVar_SelectableTextAlign, {0.5f, 0.5f});

            if(imgui_scoped::Table table = imgui_scoped::Table("##chunk table", 4, table_flags)){
                float ax = ImGui::GetContentRegionAvail().x;
                ImGui::TableSetupColumn(TR("seq"), column_flags, 0.04f * ax);
                ImGui::TableSetupColumn(TR("text"), column_flags, 0.8f * ax );
                ImGui::TableSetupColumn(TR("status"), column_flags, 0.06f * ax);
                ImGui::TableSetupColumn(TR("voices"), column_flags, 0.1f * ax);
                ImGui::TableSetupScrollFreeze(0, 1);
                ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

                // column 0
                ImGui::TableSetColumnIndex(0);
                imgui_scoped::TableTextCentered(TR("seq"));

                // column 1
                ImGui::TableSetColumnIndex(1);
                auto str = fmt::format(fmt::runtime(TR("segment: [ {} token limit ]")), max_token);
                if(ImGui::Selectable(str.c_str())) {
                    ImGui::OpenPopup("my_segment_popup");
                }

                if (ImGui::BeginPopup("my_segment_popup")) {
                    uint32_t step = 1;
                    if (ImGui::InputScalar("##segment_max_token", ImGuiDataType_U32, &max_token, &step)) {
                        max_token = std::max(5, max_token);
                        ctx.ini.sync();
                    }
                    
                    ImGui::Checkbox(TR("edit trigger segmention"), &enable_edit_trigger);
                    if ( ImGui::Button(TR("update segmention"), {-1,0}) ) {
                        btn_update_seg = true;
                    }
                    ImGui::EndPopup();
                }

                // column 2
                ImGui::TableSetColumnIndex(2);
                if(ImGui::Selectable(TR("status"))) {
                    ImGui::OpenPopup("my_status_popup");
                }

                if (ImGui::BeginPopup("my_status_popup")) {
                    if ( ImGui::MenuItem(TR("done")) ) {
                        ctx.session->set_done_value(true);
                    }
                    if ( ImGui::MenuItem(TR("todo")) ) {
                        ctx.session->set_done_value(false);
                    }
                    ImGui::EndPopup();
                }

                // column 3
                ImGui::TableSetColumnIndex(3);
                if(ImGui::Selectable(TR("voices"))) {
                    ImGui::OpenPopup("my_voices_popup");
                }

                if (ImGui::BeginPopup("my_voices_popup")) {
                    for (const auto& voice : ctx.voices ) {
                        if ( ImGui::MenuItem( voice.c_str(), NULL, ctx.ini.get().default_voice == voice) ) {
                            ctx.ini.get().default_voice = voice;
                            ctx.session->set_default_voice(ctx.ini.get().default_voice);
                            ctx.ini.sync();
                        }
                    }
                    ImGui::EndPopup();
                }

                imgui_scoped::StyleVar s_var(ImGuiStyleVar_SelectableRounding, 12.0f);

                ImGuiListClipper clipper;
                clipper.Begin(ctx.session->configs().size());

                static size_t last_playing_hash = 0;
                static int last_selected_seq = -1;
                static int edit_select_seq = -1;

                bool is_ctrl_a_down = ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_A);
                bool is_ctrl_down = ImGui::IsKeyDown(ImGuiKey_LeftCtrl);
                bool is_shift_down = ImGui::IsKeyDown(ImGuiKey_LeftShift);
                bool is_delete_down = ImGui::IsKeyDown(ImGuiKey_Delete);
                bool is_enter_down = ImGui::IsKeyDown(ImGuiKey_Enter) || ImGui::IsKeyDown(ImGuiKey_KeypadEnter);
                bool is_escape_down = ImGui::IsKeyDown(ImGuiKey_Escape);

                if ( disable_edit ) {
                    ctx.session->unselected_all();
                }

                while (clipper.Step()) {
                    for (int seq = clipper.DisplayStart; seq < clipper.DisplayEnd; seq++) {
                        auto& cfg = ctx.session->configs()[seq];
                        bool generate_ongoing = ( cfg.id == ctx.generate_id );
                            generate_ongoing &= ctx.engine->is_busy();
                        bool decode_ongoing = ( cfg.id == ctx.decode_id );
                            decode_ongoing &= ctx.engine->is_decoding();
                        bool selected = generate_ongoing;
                            selected |= ( cfg.selected );

                        if ( generate_ongoing ) {
                            ImGui::SetScrollHereY(0.5f);
                        }

                        imgui_scoped::ID id(seq);
                        ImGui::TableNextRow();

                        ImGui::TableSetColumnIndex(0); 
                        if(ImGui::Selectable(std::to_string(cfg.id).c_str(), selected, select_flags)) {

                            //toggle play
                            if ( !ctx.session->is_playing_file() ) {

                                if ( ctx.session->play_id(cfg.hash()) ) {
                                    last_playing_hash = cfg.hash();
                                }
                            } else if(last_playing_hash == cfg.hash()) {
                                ctx.session->stop_file();
                                last_playing_hash = 0;
                            } else {
                                ctx.session->stop_file();
                                if ( ctx.session->play_id(cfg.hash()) ) {
                                    last_playing_hash = cfg.hash();
                                }
                            }

                            if ( is_shift_down ) { // range select
                                if( last_selected_seq >= 0 ) {
                                    int start = std::min(last_selected_seq, (int)seq);
                                    int end = std::max(last_selected_seq, (int)seq);

                                    ctx.session->unselected_all();
                                    for (size_t j = start; j <= end; j++) {
                                        ctx.session->configs()[j].selected = true;
                                    }
                                }
                            } else if ( is_ctrl_down ) {// multiple select
                                ctx.session->toggle_selected(cfg.id);
                            } else {
                                last_selected_seq = seq;
                                ctx.session->unselected_all();
                                ctx.session->configs()[seq].selected = true;
                            }
                        }
                        ImGui::SetItemTooltip("Hash: %zu", cfg.hash());

                        if (ImGui::IsItemFocused()) {
                            if (ImGui::IsMouseDoubleClicked(0)) {
                                edit_select_seq = last_selected_seq;
                            }
                        }

                        if ( edit_select_seq != last_selected_seq ) {
                            edit_select_seq = -1;
                        }

                        ImGui::TableSetColumnIndex(1);
                        if ( edit_select_seq == seq ) {
                            if ( is_enter_down || is_escape_down ) {
                                edit_select_seq = -1;
                            }
                            ImGui::SetNextItemWidth(0.77f * ax);
                            ImGui::InputText("##text", &cfg.text);
                        } else {
                            imgui_scoped::TableTextCentered(cfg.text.c_str());
                        }

                        ImGui::TableSetColumnIndex(2);

                        if ( decode_ongoing || generate_ongoing ) {
                            ImGui::SameLine();
                            float r = ImGui::GetFrameHeight() * 0.5f;
                            auto color = ImGui::GetStyle().Colors[ImGuiCol_Text];
                            float cell_width = ImGui::GetContentRegionAvail().x;
                            float offset_x = cell_width* 0.5f - r;
                            if (offset_x > 0.0f) {
                                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset_x);
                            }
                            int arcs = decode_ongoing?2:1;
                            
                            ImSpinner::SpinnerRainbow("ongoing", r, 2.f, color, 8.f, 0.0f, ImSpinner::PI_2, arcs);
                        } else if(  ctx.session->is_playing_file() && last_playing_hash == cfg.hash() ) {
                            ImGui::SameLine();
                            float r = ImGui::GetFrameHeight() * 0.5f;
                            auto color = ImGui::GetStyle().Colors[ImGuiCol_Text];
                            float cell_width = ImGui::GetContentRegionAvail().x;
                            float offset_x = cell_width* 0.5f - r;
                            if (offset_x > 0.0f) {
                                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset_x);
                            }
                            ImSpinner::SpinnerBarChartRainbow("playing", r, 2.0f, color, 4.0f);

                        } else {
                            auto text = TR(cfg.done?"done":"todo");
                            if ( ImGui::Selectable(text) ) {
                                ImGui::OpenPopup("my_done_popup");
                            }

                            if (ImGui::BeginPopup("my_done_popup")) {
                                if ( ImGui::MenuItem(TR("done")) ) {
                                    cfg.done = true;
                                }
                                if ( ImGui::MenuItem(TR("todo")) ) {
                                    cfg.done = false;
                                }
                                ImGui::EndPopup();
                            }
                        }

                        ImGui::TableSetColumnIndex(3);
                        if ( ImGui::Selectable(cfg.voice.c_str()) ) {
                            ImGui::OpenPopup("my_voices_popup");
                        }

                        if (ImGui::BeginPopup("my_voices_popup")) {
                            for (int m = 0; m < ctx.voices.size(); m++) {
                                imgui_scoped::ID id(m);
                                if ( ImGui::MenuItem( ctx.voices[m].c_str(), NULL, cfg.voice == ctx.voices[m]) ) {
                                    cfg.voice = ctx.voices[m];
                                }
                            }
                            ImGui::EndPopup();
                        }

                    }
                }
                if ( is_ctrl_a_down ) {
                    ctx.session->selected_all();
                }

                if ( is_delete_down ) {
                    last_selected_seq = -1;
                    ctx.session->delete_selected();
                }
                
            }
        }

    }
}
void render_choose_model_popup( GuiContext& ctx ) {

    imgui_scoped::StyleVar frame_padding(ImGuiStyleVar_FramePadding, {5.0f, 0.0f});
    imgui_scoped::StyleVar frame_rounding(ImGuiStyleVar_FrameRounding, 4.0f);
    imgui_scoped::StyleVar item_speacing(ImGuiStyleVar_ItemSpacing, {10.0f, 1.0f});
    if (ImGui::BeginPopupModal(TR("choose models path"), NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::TextColored(ImColor(200,0,0,255), 
        "current models path: %s invalid.", ctx.paths.root.c_str());
        
        ImGui::SameLine();
        if (ImGui::Button(TR("choose folder"))) {
            std::string folder = choose_folder();
            ctx.ini.get().model_folder = folder.empty() ? ctx.ini.get().model_folder : folder;
            ctx.ini.sync();
            ctx.paths = Audio8ModelPaths(ctx.ini.get().model_folder);
            ImGui::CloseCurrentPopup();
        }

        {
            ImGui::Text("%s", TR(Localization::get().model_info()));
            ImGui::TextLinkOpenURL(model_url);
            ImGui::TextLinkOpenURL(backup_url);
            ImGui::Text("");
            ImGui::Text("%s", TR(Localization::get().model_structure_info()));
            ImGui::Text("%s", Localization::get().model_folder_structure());
        }
        ImGui::EndPopup();
    }
}

void render_registration_popup( GuiContext& ctx ) {
    if (ImGui::BeginPopupModal(TR("registration"), NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
        static std::string new_voice_name;
        static std::string transcript;
        static std::string ref_audio_path;
        ImGui::InputText(TR("voice name"), &new_voice_name);
        ImGui::InputText(TR("transcript"), &transcript);
        ImGui::InputTextWithHint("##ref audio", TR("audio file path"), &ref_audio_path);
        ImGui::SameLine();
        if (ImGui::Button(TR("choose"))) {
            ref_audio_path = choose_audio_path();
        }

        if (ImGui::Button(TR("ok"), ImVec2(120, 0))) {

            if ( new_voice_name.empty() ) {
            } else if ( transcript.empty() ) {
            } else if ( ref_audio_path.empty() ) {
            } else if ( !std::filesystem::exists(ref_audio_path) ) {
            } else {
                ctx.registration_status = std::async(std::launch::async, [&](){
                    ctx.engine->registers(new_voice_name, transcript, ref_audio_path);
                });
                    ImGui::CloseCurrentPopup();
            }
        }

        ImGui::SetItemDefaultFocus();
        ImGui::SameLine();
        if (ImGui::Button(TR("cancel"), ImVec2(120, 0))) { ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }
}

void render_main_window( GuiContext& ctx ) {

    ImGui::Begin("main", NULL, ImGuiWindowFlags_MenuBar);
    
    render_menubar(ctx);

    render_progressbar(ctx);

    render_segmention_table(ctx);

    // Check if the engine status has changed
    if ( ctx.engine_status.valid() ) {
        if ( ctx.engine_status.wait_for(10ms) == std::future_status::ready ) {
            ctx.engine_status.get();
            ctx.engine_status = {};
            ctx.is_loading = false;
        }
    }
    // Check if the import operation has completed
    if ( ctx.import_status.valid() ) {
        if ( ctx.import_status.wait_for(10ms) == std::future_status::ready ) {
            ctx.session = std::move(ctx.import_status.get());
            ctx.import_status = {};
            ctx.is_importing = false;
        }
    }

    // Check if the export operation has completed
    if ( ctx.export_status.valid() ) {
        if ( ctx.export_status.wait_for(10ms) == std::future_status::ready ) {
            ctx.export_status.get();
            ctx.export_status = {};
            ctx.is_exporting = false;
        }
    }

    // Check if the cancellation operation has completed
    if ( ctx.cancel_status.valid() ) {
        if ( ctx.cancel_status.wait_for(10ms) == std::future_status::ready ) {
            ctx.cancel_status.get();
            ctx.cancel_status = {};
            ctx.session->request_count() = 0;
            ctx.is_cancelling = false;
        }
    }

    // Check if the text splitting operation has completed
    if ( ctx.split_text_status.valid() ) {

        if ( ctx.split_text_status.wait_for(10ms) == std::future_status::ready ) {
            const auto& chunks = ctx.split_text_status.get();
            if ( chunks.has_value() ) {

                if ( !ctx.session ) { ctx.session = make_unique_nothrow<Session>(); }

                ctx.session->configs().clear();
                Session::config_item_s item;
                for (int i = 0; i < chunks->size(); i++) {
                    item.id = i;
                    item.text = chunks.value()[i];
                    item.voice = ctx.ini.get().default_voice;
                    item.selected = false;
                    item.done = false;
                    ctx.session->configs().push_back(item);
                }
            }
            ctx.split_text_status = {};
            ctx.is_segmenting = false;
        }
    }

    // Check if the registration operation has completed
    if ( ctx.registration_status.valid() ) {
        ctx.is_encoding = true;

        if ( std::future_status::ready == ctx.registration_status.wait_for(10ms) ) {
            ctx.registration_status.get();
            ctx.registration_status = {};
            ctx.is_encoding = false;
            ctx.voices = ctx.engine->list_voices();
        }
    }

    ImGui::End();
}


// Windows specific entry point
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#ifdef __cpp_lib_byte
#define byte win_byte_override
#include <windows.h>
#undef byte
#else
#include <windows.h>
#endif
INT WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
    PSTR lpCmdLine, INT nCmdShow)
{
    const char* argv[] = {"ChoreoGraph"};
    return main(1, (char**)argv);
}
#endif

