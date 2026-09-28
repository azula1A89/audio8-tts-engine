#include "main.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui_theme.h>
#include <imgui_stdlib.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <fmt/chrono.h>
#include <nfd.hpp>
#include <audio8_engine.hpp>
#include <text_processor.hpp>
#include <future>
#include <imspinner_compat.h>
#include <imspinner_text.h>
#include <ctime>
#include <nlohmann/json.hpp>
#include <fstream>
#include <queue>
#include <mutex>

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
    };

    ~Session() {
        stop();
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
        return session;
    }

    size_t& request_count() {
        return request_count_;
    }
    
    std::vector<config_item_s>& configs() {
        return configs_;
    }

    bool play() {

        return true;
    }

    bool play(int idx) {
        size_t count = track_count();
        if ( count == 0 || idx < 0 || idx > count ) return false;

        auto wav_file = root_ / cache_ / fmt::format("{:05}.wav", idx);
        miniaudio_impl::stop_file();
        return miniaudio_impl::play_file(wav_file.string().c_str());
    }

    void stop() {
        miniaudio_impl::stop_file();
    }

    // save PCM data to a wav file
    void save_to_wav(const std::vector<float> &track, const int& id) {
        auto wav_file = root_ / cache_ / fmt::format("{:05}.wav", id);
        miniaudio_impl::wav_write(track.data(), track.size(), wav_file.string().c_str());
        track_count_ = track_count();
        
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
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       // Enable Multi-Viewport / 
    
    ImGuiTheme::ImGuiTheme_ theme = ImGuiTheme::ImGuiTheme_ImGuiColorsClassic;
    ImGuiTheme::ApplyTweakedTheme(theme);

    auto cjk = io.Fonts->AddFontFromFileTTF("fonts/NotoSansSC-Regular.ttf");
    auto english = io.Fonts->AddFontFromFileTTF("fonts/Cousine-Regular.ttf");

    float xscale, yscale;
    glfwGetWindowContentScale((GLFWwindow *) main_window, &xscale, &yscale);

    ImGuiStyle& style = ImGui::GetStyle();
    style.FontScaleDpi = std::max(xscale, yscale);
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(main_window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    const int popup_spinner_flag = ImGuiWindowFlags_NoDecoration 
                             | ImGuiWindowFlags_NoMove 
                             | ImGuiWindowFlags_NoBackground;

    Audio8ModelPaths paths{"models"};
    std::filesystem::path session_root_path = "sessions";
    std::string txt = "大家好，我是anthony。";
    std::vector<std::string> voices;
    std::string default_voice = "anthony";
    std::string new_voice_name;
    std::string transcript;
    std::string ref_audio_path;

    std::unique_ptr<Audio8Engine> engine = std::make_unique<Audio8Engine>();
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

    uint32_t channels = 1;
    uint32_t max_audio_duration_sec = 1200;
    uint32_t sample_rate = 44100;
    uint32_t max_audio_frames = max_audio_duration_sec * sample_rate;
    uint32_t segment_max_token = 20;
    float generate_progress = 0.0f;
    int generate_id = -1;
    int decode_id = -1;
    float decode_eta = -1.0f;
    int request_session = 0;

    // Main loop
    while ( glfwWindowShouldClose(main_window) == GL_FALSE )
    {
        // Poll for and process events
        glfwPollEvents();
        
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
                    paths = Audio8ModelPaths(choose_folder());
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
                imgui_scoped::StyleVar item_speacing(ImGuiStyleVar_ItemSpacing, {20.0f, 5.0f});
                imgui_scoped::Disabled disable(is_loading);
                if (ImGui::BeginMenuBar()) {

                   {
                        imgui_scoped::Disabled disable(engine->is_busy());
                        if (ImGui::BeginMenu("session")) {
                            if ( ImGui::MenuItem("new") ) {
                                if ( session ) {
                                    session.reset(nullptr);
                                }
                                session = make_unique_nothrow<Session>();
                            }

                            if ( ImGui::BeginMenu("recent") ) {
                                if (std::filesystem::exists(session_root_path) && std::filesystem::is_directory(session_root_path)) {
                                    for (const auto& entry : std::filesystem::directory_iterator(session_root_path)) {
                                        if (entry.is_regular_file() && entry.path().extension() == ".json") {
                                            if ( ImGui::MenuItem(entry.path().string().c_str()) ) {
                                                session = Session::from_json(entry.path().string().c_str());
                                            }
                                        }
                                    }
                                }
                                ImGui::EndMenu();
                            }

                            ImGui::EndMenu();
                        }
                    }

                    {
                        imgui_scoped::Disabled disable(session == nullptr);
                        // generate speech
                        if ( ImGui::MenuItem("run") ) {
                            if( !session ) session = make_unique_nothrow<Session>();

                            if ( session && !engine->is_busy() ) {
                                request_session++;
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
                            }
                        }

                        // cancle generation
                        if ( ImGui::MenuItem("cancle") ) {

                            if ( !is_cancelling ) {
                                is_cancelling = true;
                                cancle_status = std::async(std::launch::async,[&](){
                                    engine->cancel();
                                });
                            }
                        }

                        // play tracks
                        {
                            if ( ImGui::MenuItem("play") ) {
                                if( session && !session->play() ) {
                                    ImGui::OpenPopup("my_play_popup");
                                }
                            }

                            imgui_scoped::StyleVar popup_rounding(ImGuiStyleVar_PopupRounding, 6.0f);
                            if (ImGui::BeginPopup("my_play_popup")) {
                                ImGui::TextColored(ImColor(200,0,0,255), "play failed.");
                                ImGui::EndPopup();
                            }

                            if ( ImGui::MenuItem("stop") ) {
                                if ( session ) {
                                    session->stop();
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

                            ImGui::DragFloat("font scale", &ImGui::GetStyle().FontScaleMain, 0.01f, 0.2f, 3.0f);

                            ImGuiIO& io = ImGui::GetIO();
                            ImFontAtlas* atlas = io.Fonts;
                            ImGui::ShowFontAtlas(atlas);

                            if( ImGui::BeginCombo("theme", ImGuiTheme::ImGuiTheme_Name(theme)) ) {
                                for (int i = 0; i< ImGuiTheme::ImGuiTheme_Count; i++) {
                                    imgui_scoped::ID id(i);
                                    
                                    if(ImGui::Selectable(ImGuiTheme::ImGuiTheme_Name((ImGuiTheme::ImGuiTheme_)i), theme == i)) {
                                        theme = (ImGuiTheme::ImGuiTheme_)i;
                                        float size1 = ImGui::GetStyle().FontSizeBase;
                                        float size2 = ImGui::GetStyle().FontScaleDpi;
                                        float size3 = ImGui::GetStyle().FontScaleMain;
                                        ImGuiTheme::ApplyTweakedTheme(theme);
                                        ImGui::GetStyle().FontSizeBase = size1;
                                        ImGui::GetStyle().FontScaleDpi = size2;
                                        ImGui::GetStyle().FontScaleMain = size3;
                                    }
                                }
                                ImGui::EndCombo();
                            }

                            ImGui::EndMenu();
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
                    ImGui::ProgressBar(session->progress(), ImVec2(-1.0f, 0.0f), "Total..");
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

            // Text input
            if( is_initialized  && session) {
                static bool enable_text_segmentation = false;
                static int last_segment_max_token = -1;
                static int last_edit_count = -1;
                static int edit_count = 0;
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
                     should_update &= enable_text_segmentation;
                     
                if( should_update ) {
                    is_segmenting = true;
                    last_edit_count = edit_count;
                    last_segment_max_token = segment_max_token;
                    editor_font = engine->contains_cjk(txt) ? cjk : english;

                    split_text_status = std::async(std::launch::async, [&txt, &engine, &default_voice, &segment_max_token](){
                        return engine->split_text_by_tokens(txt, segment_max_token);
                    });
                }

                ImGui::BeginChild("##chunk info");
                {
                    ImGuiTableColumnFlags table_flags = 0//ImGuiTableFlags_Borders
                                    | ImGuiTableFlags_ScrollY
                                    | ImGuiTableFlags_RowBg;
                    ImGuiSelectableFlags select_flags = ImGuiSelectableFlags_SpanAllColumns 
                                    | ImGuiSelectableFlags_AllowDoubleClick
                                    | ImGuiSelectableFlags_AllowOverlap;
                    ImGuiTableColumnFlags column_flags = ImGuiTableColumnFlags_WidthFixed;

                    imgui_scoped::Table table("##chunk table", 4, table_flags);
                    imgui_scoped::StyleVar f_padding(ImGuiStyleVar_FramePadding, {0.0f, 0.0f});
                    imgui_scoped::StyleVar s_txt_align(ImGuiStyleVar_SelectableTextAlign, {0.5f, 0.5f});

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
                        ImGui::InputScalar("##segment_max_token", ImGuiDataType_U32, &segment_max_token, &step);
                        segment_max_token = std::max(10U, segment_max_token);
                        ImGui::Checkbox("edit trigger segmention", &enable_text_segmentation);
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
                            }
                        }
                        ImGui::EndPopup();
                    }

                     

                    imgui_scoped::StyleVar s_var(ImGuiStyleVar_SelectableRounding, 12.0f);

                    ImGuiListClipper clipper;
                    clipper.Begin(session->configs().size());

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
                                session->play(cfg.id);
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
                            }else {
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
                ImGui::EndChild();
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
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(main_window);

    }
    if ( session ) {
        session.reset(nullptr);
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

