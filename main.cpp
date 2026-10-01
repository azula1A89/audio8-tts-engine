#include "main.hpp"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <future>
#include <mutex>
#include <queue>
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
#include <text_processor.hpp>

using namespace std::chrono_literals;
using json = nlohmann::json;
const char* model_info = "This project using the following model:";
const char* model_url = "https://huggingface.co/Edge0/Audio8-TTS-Preview-0.6B-ONNX-INT4/tree/main";
const char* model_folder = R"(
The expected model directory is:

models/
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

const std::string help_text = R"(
tips:
___
  * step 1: start a new session or open an existing one.
  * step 2: paste the text you want to convert to speech.
  * step 3: click the table header "segment [20 token limit]" to adjust the segment length. then click "update segmention" to split the text into segments.
  * step 4: optional, you can edit the text or voice settings for each segment as needed.
  * step 5: click "run" to start the text-to-speech conversion process.
  * step 6: once the conversion is complete, you can export the audio files for further use.
___
segment table:
  * click the table header "segment [20 token limit]" to adjust the segment length.
  * click "update segmention" to apply the changes.(this will override the whole segmention table)
  * after updating the segment length, you can manually adjust individual segments if needed.
  * you can also delete segments by selecting them and press the delete key.
  * to select multiple segments, hold the left-shift key while clicking on the segments you want to select.
___
about:
  * [source] (https://github.com/azula1A89/audio8-tts-engine.git)
  * [model 1] (https://huggingface.co/Edge0/Audio8-TTS-Preview-0.6B-ONNX-INT4/tree/main)
  * [model 2] (https://modelscope.ai/models/Edge0/Audio8-TTS-Preview-0.6B-ONNX-INT4/files)
)";

class UserSettings {
private:
    struct settings_s {
        std::string model_folder = "models";
        std::string session_folder = "sessions";
        std::string export_folder = "sessions";
        std::string default_voice = "anthony";
        float font_scale = 1.0f;
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
            else if (key == "font_scale") setting->font_scale = std::stof(value);
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
        buf->appendf("font_scale=%.2f\n", setting->font_scale);
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
        buf->appendf("execution_provider=%d\n", setting->runtime_config.execution_provider);
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
    };
    
private:
    std::filesystem::path root_;
    std::string cache_;
    size_t track_count_;
    size_t request_count_;
    std::vector<config_item_s> configs_;
    std::queue<int> done_;
    std::mutex done_mutex_;
    
public:
    Session(std::filesystem::path root = "sessions", 
        std::string cache = fmt::format("{:%F_%H-%M-%S}", fmt::localtime(std::time(nullptr))), 
        size_t track_count = 0) :
        root_(root), cache_(cache), track_count_(track_count), request_count_(0) {
        auto cache_path = root_ / cache_;
        if ( !std::filesystem::exists(cache_path) ) {
            std::filesystem::create_directories(cache_path);
        }
        init_playlist();
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
        for (const auto& cfg : j["configs"]) {
            Session::config_item_s config;
            config.id = cfg["id"];
            config.text = cfg["text"];
            config.voice = cfg["voice"];
            config.selected = cfg["selected"];
            config.done = cfg["done"];
            session->configs().push_back(config);
        }
        session->init_playlist();
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

    void init_playlist() {
        auto cache_path = root_ / cache_;
        if (!std::filesystem::exists(cache_path) || !std::filesystem::is_directory(cache_path)) {
            return;
        }

        miniaudio_impl::track_clear();
        std::vector<std::filesystem::path> paths;
        for (const auto& entry : std::filesystem::directory_iterator(cache_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".wav") {
                paths.push_back(entry.path());
            }
        }

        std::sort(paths.begin(), paths.end());
        for (const auto& p : paths) {
            miniaudio_impl::track_add(p.string());
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

    bool play_id(int idx) {
        size_t count = track_count();
        if ( count == 0 || idx < 0 || idx > count ) return false;

        auto wav_file = root_ / cache_ / fmt::format("{:05}.wav", idx);
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
        auto wav_file = root_ / cache_ / fmt::format("{:05}.wav", id);
        miniaudio_impl::wav_write(track.data(), track.size(), wav_file.string().c_str());
        track_count_ = track_count();
        miniaudio_impl::track_add(wav_file.string());
        std::lock_guard<std::mutex> guard(done_mutex_);
        done_.push(id);
    };
    
    size_t track_count() {
        auto cache_path = root_ / cache_;
        if (!std::filesystem::exists(cache_path) || !std::filesystem::is_directory(cache_path)) {
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
        int id = -1;
        {
            std::lock_guard<std::mutex> guard(done_mutex_);
            if ( !done_.empty() ) {
                id = done_.front();
                done_.pop();
            }
        }

        if ( id >= 0 ) {
            set_is_done(id);
        }
    }

    void set_is_done(const int& id) {
        auto search = std::find_if(configs_.begin(), configs_.end(), [&id](const config_item_s& item){ return item.id == id; });
        if ( search != configs_.end() ) {
            search->done = true;
        }
    }

    void set_done_value(bool v) {
        for (auto& i : configs_) { i.done = v; }
    }

    void set_default_voice (const std::string& default_voice){ 
        for (auto& i : configs_) { i.voice = default_voice; }
    };

    void unselected_all() { 
        for (auto& i : configs_) { i.selected = false; }
    };

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

std::string choose_folder();
std::string choose_audio_path();
void imgui_parent_window();
std::function<void(GLFWwindow*)> render_frame;
UserSettings settings;

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
    settings.initialize();

    glfwSetWindowSize(main_window, settings.get().window_width, settings.get().window_height);

    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    
    ImGuiTheme::ApplyTweakedTheme(static_cast<ImGuiTheme::ImGuiTheme_>(settings.get().theme));

    auto cjk = io.Fonts->AddFontFromFileTTF("fonts/NotoSansSC-Regular.ttf");
    auto english = io.Fonts->AddFontFromFileTTF("fonts/Cousine-Regular.ttf");

    float xscale, yscale;
    glfwGetWindowContentScale((GLFWwindow *) main_window, &xscale, &yscale);

    ImGuiStyle& style = ImGui::GetStyle();
    style.FontScaleDpi = std::max(xscale, yscale);

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(main_window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImGui::MarkdownConfig md_config{ NULL, NULL, NULL, NULL, { { english, true }, { english, true }, { english, false } }, NULL };

    const int popup_spinner_flag = ImGuiWindowFlags_NoDecoration 
                             | ImGuiWindowFlags_NoMove 
                             | ImGuiWindowFlags_NoBackground;

    Audio8ModelPaths paths{ settings.get().model_folder  };
    std::filesystem::path session_root_path = settings.get().session_folder;
    std::string txt = "大家好，我是anthony。";
    std::vector<std::string> voices;
    std::string& default_voice = settings.get().default_voice;
    std::string new_voice_name;
    std::string transcript;
    std::string ref_audio_path;

    std::unique_ptr<Audio8Engine> engine = std::make_unique<Audio8Engine>(settings.get().runtime_config);
    std::unique_ptr<Session> session = nullptr;
    std::future<void> engine_status = {};
    std::future<void> registration_status = {};
    std::future<std::optional<std::vector<std::string>>> split_text_status = {};
    std::future<void> cancle_status = {};

    bool is_initialized = false;
    bool is_loading = false;
    bool is_segmenting = false;
    bool is_encoding = false;
    bool is_cancelling = false;

    float generate_progress = 0.0f;
    int generate_id = -1;
    int decode_id = -1;
    float decode_eta = -1.0f;
    std::atomic<int> request_done = 0;

    render_frame = [&](GLFWwindow* window){
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        //root window
        imgui_parent_window();

        // default modelspath invalid
        if ( !paths.check() ) {

            // ask for models path
            ImGui::OpenPopup("choose models path");

            imgui_scoped::StyleVar frame_padding(ImGuiStyleVar_FramePadding, {5.0f, 0.0f});
            imgui_scoped::StyleVar frame_rounding(ImGuiStyleVar_FrameRounding, 4.0f);
            imgui_scoped::StyleVar item_speacing(ImGuiStyleVar_ItemSpacing, {10.0f, 1.0f});
            if (ImGui::BeginPopupModal("choose models path", NULL, ImGuiWindowFlags_AlwaysAutoResize))
            {
                ImGui::TextColored(ImColor(200,0,0,255), 
                "current models path: %s invalid.", paths.root.c_str());
                
                ImGui::SameLine();
                if (ImGui::Button("choose")) {
                    settings.get().model_folder = choose_folder();
                    settings.sync();
                    paths = Audio8ModelPaths(settings.get().model_folder);
                    ImGui::CloseCurrentPopup();
                }

                {
                    imgui_scoped::FontSize font(12.0f);
                    ImGui::Text("%s", model_info);
                    ImGui::TextLinkOpenURL(model_url);
                    ImGui::Text("%s", model_folder);
                }
                ImGui::EndPopup();
            }
        
        }else if ( !is_initialized ) {
            if( engine->initialize(paths.root) )// engine initialize
                is_initialized = true;

            engine->set_generate_callback([&](float progress_in, int id_in){
                generate_progress = progress_in;
                generate_id = id_in;
            });

            engine->set_decoder_callback([&](std::vector<float> pcm_in, const int id_in){
                if( session ) session->save_to_wav(pcm_in, id_in);
                request_done.fetch_add(1);
            });

            engine->set_decoder_eta_callback([&](float eta_in, const int id_in){
                decode_eta = eta_in;
                decode_id = id_in;
            });

            // auto preload
            if ( is_initialized ) {
                is_loading = true;
                engine_status = std::async(std::launch::async,[&](){
                    engine->preload_model();
                    is_loading = false;
                });
            }
        }

        if ( voices.empty() ) {
            voices = engine->list_voices();
        }

        if ( session ) {
            session->refresh_status();
        }

        // main window
        {
            imgui_scoped::Font font(english);
            ImGui::Begin("main", NULL, ImGuiWindowFlags_MenuBar);
            
            // Menubar
            if( is_initialized ) {
                imgui_scoped::StyleVar popup_rounding(ImGuiStyleVar_PopupRounding, 6.0f);
                imgui_scoped::StyleVar item_speacing(ImGuiStyleVar_ItemSpacing, {20.0f, 5.0f});
                imgui_scoped::Disabled disable(is_loading);
                if (ImGui::BeginMenuBar()) {

                    // session, export
                   {
                        imgui_scoped::Disabled disable(engine->is_busy());
                        if (ImGui::BeginMenu("session")) {
                            if ( ImGui::MenuItem("new") ) {
                                if ( session ) {
                                    session.reset();
                                }
                                session = make_unique_nothrow<Session>();
                            }

                            if ( ImGui::BeginMenu("recent") ) {
                                static std::string session_recent = "";
                                if (std::filesystem::exists(session_root_path) && std::filesystem::is_directory(session_root_path)) {
                                    for (const auto& entry : std::filesystem::directory_iterator(session_root_path)) {
                                        if (entry.is_regular_file() && entry.path().extension() == ".json") {
                                            std::string filename = entry.path().filename().string();
                                            if ( ImGui::MenuItem(filename.c_str(), nullptr, session_recent == filename) ) {
                                                session_recent = filename;
                                                session = Session::from_json(entry.path().string().c_str());
                                            }
                                        }
                                    }
                                }
                                ImGui::EndMenu();
                            }

                            {
                                int& export_length = settings.get().export_length;
                                imgui_scoped::Disabled disable(session == nullptr);
                                if ( ImGui::BeginMenu("export") ) {
                                    if (ImGui::DragInt("max audio length(second)", &export_length, 1.0f, 1)) {
                                        settings.sync();
                                    }

                                    int& sample_rate = settings.get().export_sample_rate;
                                    if (ImGui::DragInt("sample rate", &sample_rate, 100.0f, 8000, 96000)) {
                                        settings.sync();
                                    }

                                    std::string folder = settings.get().export_folder;
                                    if (ImGui::InputTextWithHint("##export to folder", "export to folder", &folder)) {
                                        if (!folder.empty() && std::filesystem::exists(folder) && std::filesystem::is_directory(folder)) {
                                            settings.get().export_folder = folder;
                                            settings.sync();
                                        }
                                    }
                                    ImGui::SameLine();
                                    if (ImGui::Button("choose folder")) {
                                        folder = choose_folder();
                                        if (!folder.empty() && std::filesystem::exists(folder) && std::filesystem::is_directory(folder)) {
                                            settings.get().export_folder = folder;
                                            settings.sync();
                                        }
                                    }
                                    if ( ImGui::Button("export") ) {
                                        if ( session->is_playing_list() ) {
                                            session->stop_playlist();
                                        }
                                        session->export_audio(export_length, settings.get().export_folder.c_str(), sample_rate);
                                    }
                                    ImGui::EndMenu();
                                }
                            }

                            ImGui::EndMenu();
                        }
                    }

                    // run, cancle
                    {
                        imgui_scoped::Disabled disable(session == nullptr);
                        std::string status_text = engine->is_busy() ? "cancle" : "run";
                        if ( ImGui::MenuItem(status_text.c_str()) ) {
                            if ( !engine->is_busy() ) {
                                request_done.store(0);
                                session->request_count() = 0;

                                TTSRequest request{};
                                for (const auto& item : session->configs()) {
                                    if ( item.done ) { continue; } //skip processed item
                                    request.id = item.id;
                                    request.text = item.text;
                                    request.voice_name = item.voice;
                                    request.max_new_tokens = 1024;
                                    engine->push(request);
                                    session->request_count()++;
                                }
                            } else {
                                if ( !is_cancelling ) {
                                    is_cancelling = true;
                                    cancle_status = std::async(std::launch::async,[&](){
                                        engine->cancel();
                                    });
                                }
                            }
                        }

                        // play tracks
                        {
                            // imgui_scoped::Disabled disable( session->is_playing_file() );
                            std::string status_txt = session->is_playing_list() ? "stop":"play";
                            if ( ImGui::MenuItem(status_txt.c_str()) ) {
                                if( !session->is_playing_list() ) {
                                    session->start_playlist();
                                } else {
                                    session->stop_playlist();
                                }
                            }
                        }

                    }

                    // voice registration (voice clone)
                    {
                        imgui_scoped::Disabled disable( is_loading );
                        if (ImGui::MenuItem("registration")) {

                            ImGui::OpenPopup("registration");
                        }

                        if (ImGui::BeginPopupModal("registration", NULL, ImGuiWindowFlags_AlwaysAutoResize))
                        {
                            ImGui::InputText("voice name", &new_voice_name);
                            ImGui::InputText("transcript", &transcript);
                            ImGui::InputTextWithHint("##ref audio", "audio file path", &ref_audio_path);
                            ImGui::SameLine();
                            if (ImGui::Button("choose")) {
                                ref_audio_path = choose_audio_path();
                            }

                            if (ImGui::Button("OK", ImVec2(120, 0))) {
                                registration_status = std::async(std::launch::async, [&](){
                                    
                                    engine->registers(new_voice_name, transcript, ref_audio_path);
                                });
                                
                                ImGui::CloseCurrentPopup();
                            }

                            ImGui::SetItemDefaultFocus();
                            ImGui::SameLine();
                            if (ImGui::Button("Cancel", ImVec2(120, 0))) { ImGui::CloseCurrentPopup(); }
                            ImGui::EndPopup();
                        }
                    }

                    // settings
                    if (ImGui::BeginMenu("settings")) {

                        if(ImGui::BeginMenu("user interface")) {
                            float& font_scale = settings.get().font_scale;
                            if (ImGui::DragFloat("font scale", &font_scale, 0.01f, 0.2f, 3.0f)) {
                                ImGui::GetStyle().FontScaleMain = font_scale;
                                settings.sync();
                            }

                            // ImGuiIO& io = ImGui::GetIO();
                            // ImFontAtlas* atlas = io.Fonts;
                            // ImGui::ShowFontAtlas(atlas);
                            int& theme = settings.get().theme;
                            if( ImGui::BeginCombo("theme", ImGuiTheme::ImGuiTheme_Name(static_cast<ImGuiTheme::ImGuiTheme_>(theme))) ) {
                                for (int i = 0; i< ImGuiTheme::ImGuiTheme_Count; i++) {
                                    imgui_scoped::ID id(i);
                                    
                                    if(ImGui::Selectable(ImGuiTheme::ImGuiTheme_Name((ImGuiTheme::ImGuiTheme_)i), theme == i)) {
                                        theme = (ImGuiTheme::ImGuiTheme_)i;
                                        settings.sync();
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

                            ImGui::EndMenu();
                        }

                        if ( ImGui::BeginMenu("onnxruntime") ) {
                            uint32_t step = 1;
                            const ImGuiDataType_ type = ImGuiDataType_U32;
                            static bool value_changed = false;
                            static uint32_t n = std::thread::hardware_concurrency();
                            uint32_t& slowar_t_num = settings.get().runtime_config.slow_ar_thread_num;
                            uint32_t& fastar_t_num = settings.get().runtime_config.fast_ar_thread_num;
                            uint32_t& encoder_t_num = settings.get().runtime_config.codec_encoder_thread_num;
                            uint32_t& decoder_t_num = settings.get().runtime_config.codec_decoder_thread_num;
                            ExecutionProvider& ep = settings.get().runtime_config.execution_provider;

                            if (ImGui::RadioButton( "CPU", ep == ExecutionProvider::CPU)) {
                                ep = ExecutionProvider::CPU;
                                value_changed = true;
                                settings.sync();
                            }
                            ImGui::SameLine();
                            if (ImGui::RadioButton( "GPU", ep == ExecutionProvider::GPU)) {
                                ep = ExecutionProvider::GPU;
                                value_changed = true;
                                settings.sync();
                            }

                            if ( ImGui::InputScalar("slow ar thread", type, &slowar_t_num, &step) ) {
                                slowar_t_num = std::clamp(slowar_t_num, 1u, n);
                                value_changed = true;
                                settings.sync();
                            }
                            if ( ImGui::InputScalar("fast ar thread", type, &fastar_t_num, &step) ) {
                                fastar_t_num = std::clamp(fastar_t_num, 1u, n);
                                value_changed = true;
                                settings.sync();
                            }

                            if ( ImGui::InputScalar("encoder thread", type, &encoder_t_num, &step) ) {
                                encoder_t_num = std::clamp(encoder_t_num, 1u, n);
                                value_changed = true;
                                settings.sync();
                            }

                            if ( ImGui::InputScalar("decoder thread", type, &decoder_t_num, &step) ) {
                                decoder_t_num = std::clamp(decoder_t_num, 1u, n);
                                value_changed = true;
                                settings.sync();
                            }
                            if (value_changed) {
                                imgui_scoped::StyleVar var(ImGuiStyleVar_FrameRounding, 8);
                                if ( ImGui::Button("apply", ImVec2(-1.0f, 0.0f)) ) {
                                    value_changed = false;
                                    engine->cancel();
                                    engine = std::make_unique<Audio8Engine>(settings.get().runtime_config);
                                    is_initialized = false;
                                }
                            }
                            ImGui::EndMenu();
                        }
                        
                        ImGui::EndMenu();
                    }

                    // help, about
                    if (ImGui::BeginMenu("help")) 
                    {
                        static ImVec2 md_size = {900, 500};

                        if ( imgui_scoped::Child help = imgui_scoped::Child("help", md_size) ) {
                            ImGui::Markdown(help_text.c_str(), help_text.length(), md_config);
                        }
                        ImGui::EndMenu();
                    }

                    ImGui::EndMenuBar();
                }
            }

            // Progress bar
            {
                imgui_scoped::StyleVar frame_padding(ImGuiStyleVar_FramePadding, {5.0f, 0.0f});
                imgui_scoped::StyleVar frame_rounding(ImGuiStyleVar_FrameRounding, 6.0f);

                // "Loading" progress bar
                if ( is_loading ) {
                    ImGui::ProgressBar(-1.0f * (float)ImGui::GetTime(), ImVec2(-1.0f, 0.0f), "Loading..");
                }

                // "Encoding" progress bar
                if ( is_encoding ) {
                    ImGui::ProgressBar(-1.0f * (float)ImGui::GetTime(), ImVec2(-1.0f, 0.0f), "Encoding..");
                }

                // "Generating" "Total" progress bar
                if( is_initialized && engine->is_busy() && session ){
                    ImGui::ProgressBar(generate_progress, ImVec2(-1.0f, 0.0f), "Generating..");
                    ImGui::ProgressBar(session->progress(), ImVec2(-1.0f, 0.0f), fmt::format("{}/{}", request_done.load(), session->request_count()).c_str());
                }

                {
                    bool open = is_segmenting || is_cancelling;
                    if( open ) {
                        if ( !ImGui::IsPopupOpen("my_spinner_popup") ) {
                            ImGui::OpenPopup("my_spinner_popup");
                        }
                    }

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

            // Text input + chunk info table
            if( is_initialized  && session) {
                int& segment_max_token = settings.get().segment_max_token;
                static bool enable_edit_trigger = false;
                static int last_segment_max_token = -1;
                static int last_edit_count = -1;
                static int edit_count = 0;
                static bool btn_update_segmention = false;
                static ImFont* editor_font = cjk;
                bool disable_edit = engine->is_busy() || is_segmenting;

                imgui_scoped::Font font(editor_font);
                imgui_scoped::Disabled disable(disable_edit);
                auto sz = ImGui::GetContentRegionAvail();
                int edited = ImGui::InputTextMultiline("##text to speach", &txt,
                    ImVec2(-FLT_MIN, sz.y * 0.25f), 
                    0);
                edit_count += edited;

                bool should_update = (last_edit_count != edit_count);
                     should_update |= (last_segment_max_token != segment_max_token);
                     should_update &= !is_loading;
                     should_update &= !is_segmenting;
                     should_update &= enable_edit_trigger;
                     should_update |= btn_update_segmention;
                     
                if( should_update ) {
                    btn_update_segmention = false;
                    is_segmenting = true;
                    last_edit_count = edit_count;
                    last_segment_max_token = segment_max_token;
                    editor_font = engine->contains_cjk(txt) ? cjk : english;

                    split_text_status = std::async(std::launch::async, [&txt, &engine, &default_voice, &segment_max_token](){
                        return engine->split_text_by_tokens(txt, segment_max_token);
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
                        ImGui::TableSetupColumn("seq", column_flags, 0.04f * ax);
                        ImGui::TableSetupColumn("text", column_flags, 0.8f * ax );
                        ImGui::TableSetupColumn("status", column_flags, 0.06f * ax);
                        ImGui::TableSetupColumn("options", column_flags, 0.1f * ax);
                        ImGui::TableSetupScrollFreeze(0, 1);
                        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

                        // column 0
                        ImGui::TableSetColumnIndex(0);
                        imgui_scoped::TableTextCentered("seq");

                        // column 1
                        ImGui::TableSetColumnIndex(1);
                        auto str = fmt::format("segment: [ {} token limit ]", segment_max_token);
                        if(ImGui::Selectable(str.c_str())) {
                            ImGui::OpenPopup("my_segment_popup");
                        }

                        if (ImGui::BeginPopup("my_segment_popup")) {
                            uint32_t step = 1;
                            if (ImGui::InputScalar("##segment_max_token", ImGuiDataType_U32, &segment_max_token, &step)) {
                                segment_max_token = std::max(5, segment_max_token);
                                settings.sync();
                            }
                            
                            ImGui::Checkbox("edit trigger segmention", &enable_edit_trigger);
                            if ( ImGui::Button("update segmention", {-1,0}) ) {
                                btn_update_segmention = true;
                            }
                            ImGui::EndPopup();
                        }

                        // column 2
                        ImGui::TableSetColumnIndex(2);
                        if(ImGui::Selectable("status")) {
                            ImGui::OpenPopup("my_status_popup");
                        }

                        if (ImGui::BeginPopup("my_status_popup")) {
                            if ( ImGui::MenuItem("done") ) {
                                session->set_done_value(true);
                            }
                            if ( ImGui::MenuItem("todo") ) {
                                session->set_done_value(false);
                            }
                            ImGui::EndPopup();
                        }

                        // column 3
                        ImGui::TableSetColumnIndex(3);
                        if(ImGui::Selectable("option")) {
                            ImGui::OpenPopup("my_option_popup");
                        }

                        if (ImGui::BeginPopup("my_option_popup")) {
                            for (const auto& voice : voices ) {
                                if ( ImGui::MenuItem( voice.c_str(), NULL, default_voice == voice) ) {
                                    default_voice = voice;
                                    session->set_default_voice(default_voice);
                                    settings.sync();
                                }
                            }
                            ImGui::EndPopup();
                        }

                        

                        imgui_scoped::StyleVar s_var(ImGuiStyleVar_SelectableRounding, 12.0f);

                        ImGuiListClipper clipper;
                        clipper.Begin(session->configs().size());

                        static int last_playing_id = -1;
                        static int last_selected_seq = -1;
                        static int edit_select_id = -1;
                        size_t num = session->track_count();

                        bool is_shift_down = ImGui::IsKeyDown(ImGuiKey_LeftShift);
                        bool is_delete_down = ImGui::IsKeyDown(ImGuiKey_Delete);
                        if ( disable_edit ) {
                            session->unselected_all();
                        }

                        while (clipper.Step()) {
                            for (int seq = clipper.DisplayStart; seq < clipper.DisplayEnd; seq++) {
                                auto& cfg = session->configs()[seq];
                                bool generate_ongoing = ( cfg.id == generate_id );
                                    generate_ongoing &= engine->is_busy();
                                bool decode_ongoing = ( cfg.id == decode_id );
                                    decode_ongoing &= engine->is_decoding();
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
                                    if ( !session->is_playing_file() ) {

                                        if ( session->play_id(cfg.id) ) {
                                            last_playing_id = cfg.id;
                                        }
                                    } else if(last_playing_id == cfg.id) {
                                        session->stop_file();
                                        last_playing_id = -1;
                                    } else {
                                        session->stop_file();
                                        if ( session->play_id(cfg.id) ) {
                                            last_playing_id = cfg.id;
                                        }
                                    }

                                    if ( is_shift_down ) { // range select
                                        if( last_selected_seq >= 0 ) {
                                            int start = std::min(last_selected_seq, (int)seq);
                                            int end = std::max(last_selected_seq, (int)seq);

                                            session->unselected_all();
                                            for (size_t j = start; j <= end; j++) {
                                                session->configs()[j].selected = true;
                                            }
                                        }
                                    } else {
                                        last_selected_seq = seq;
                                        session->unselected_all();
                                        session->configs()[seq].selected = true;
                                    }
                                }

                                if (ImGui::IsItemFocused()) {
                                    if (ImGui::IsMouseDoubleClicked(0)) {
                                        edit_select_id = last_selected_seq;
                                    }
                                }

                                if ( edit_select_id != last_selected_seq ) {
                                    edit_select_id = -1;
                                }

                                ImGui::TableSetColumnIndex(1);
                                if ( edit_select_id == cfg.id ) {
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
                                } else if(  last_playing_id == cfg.id ) {
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
                                    auto text = cfg.done?"done":"todo";
                                    if ( ImGui::Selectable(text) ) {
                                        ImGui::OpenPopup("my_done_popup");
                                    }

                                    if (ImGui::BeginPopup("my_done_popup")) {
                                        if ( ImGui::MenuItem("done") ) {
                                            cfg.done = true;
                                        }
                                        if ( ImGui::MenuItem("todo") ) {
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
                                    for (int m = 0; m < voices.size(); m++) {
                                        imgui_scoped::ID id(m);
                                        if ( ImGui::MenuItem( voices[m].c_str(), NULL, cfg.voice == voices[m]) ) {
                                            cfg.voice = voices[m];
                                        }
                                    }
                                    ImGui::EndPopup();
                                }

                            }
                        }

                        if ( is_delete_down ) {
                            last_selected_seq = -1;
                            session->delete_selected();
                        }
                        
                    }
                }

            }

            // Check if the cancellation operation has completed
            if ( cancle_status.valid() ) {
                if ( cancle_status.wait_for(10ms) == std::future_status::ready ) {
                    cancle_status.get();
                    cancle_status = {};
                    session->request_count() = 0;
                    is_cancelling = false;
                }
            }

            // Check if the text splitting operation has completed
            if ( split_text_status.valid() ) {

                if ( split_text_status.wait_for(10ms) == std::future_status::ready ) {
                    const auto& chunks = split_text_status.get();
                    if ( chunks.has_value() ) {
                        TextProcessor processor;

                        if ( !session ) { session = make_unique_nothrow<Session>(); }

                        session->configs().clear();
                        Session::config_item_s item;
                        for (int i = 0; i < chunks->size(); i++) {
                            item.id = i;
                            item.text = processor.clean_text(chunks.value()[i]);
                            item.voice = default_voice;
                            item.selected = false;
                            item.done = false;
                            session->configs().push_back(item);
                        }
                    }
                    split_text_status = {};
                    is_segmenting = false;
                }
            }

            // Check if the registration operation has completed
            if ( registration_status.valid() ) {
                is_encoding = true;

                if ( std::future_status::ready == registration_status.wait_for(10ms) ) {
                    registration_status.get();
                    registration_status = {};
                    is_encoding = false;
                    voices = engine->list_voices();
                }
            }

            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(main_window);
    };

    glfwSetWindowRefreshCallback(main_window, [](GLFWwindow* window){
        render_frame(window);
    });

    glfwSetFramebufferSizeCallback(main_window, [](GLFWwindow* window, int width, int height){
        settings.get().window_width = width;
        settings.get().window_height = height;
        settings.sync();
        glViewport(0, 0, width, height);
        render_frame(window);
    });

    // Main loop
    while ( glfwWindowShouldClose(main_window) == GL_FALSE )
    {
        // Poll for and process events
        glfwPollEvents();
        render_frame(main_window);
    }

    if ( session ) {
        session.reset();
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

