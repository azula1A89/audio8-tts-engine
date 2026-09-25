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

using namespace std::chrono_literals;

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
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
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
    std::string run_time_str;

    std::unique_ptr<Audio8Engine> engine = std::make_unique<Audio8Engine>();

    std::future<void> engine_status = {};
    std::future<void> registration_status = {};
    std::future<std::optional<std::vector<std::string>>> split_text_status = {};
    std::future<void> cancle_status = {};

    bool is_initialized = false;
    bool is_loading = false;
    bool is_segmenting = false;
    bool is_encoding = false;
    bool is_cancelling = false;

    uint32_t segment_max_token = 20;
    float progress = 0.0f;
    float eta = -1.0f;

    struct config_item_s {
        int id;
        std::string text;
        std::string voice;
        bool selected;
        bool done;
    };
    using config_t = std::vector<config_item_s>;
    config_t configs;

    int request_session = 0;
    int request_count = 0;

    auto progress_total = [&](){
        float x = 0.0f;
        if ( request_count > 0 ) {
            x = miniaudio_impl::track_count();
            x /= request_count;
        }
        return x;
    };

    auto set_default_voice = [&configs, &default_voice](){ 
        for (auto& i : configs) { i.voice = default_voice; }
    };

    auto unselected_all = [&configs](){ 
        for (auto& i : configs) { i.selected = false; }
    };

    auto datetime_str = [](){
        std::time_t t = std::time(nullptr);
        return fmt::format("{:%F_%H-%M-%S}", fmt::localtime(t));
    };

    auto save = [&](){
        if ( miniaudio_impl::buffer_length() && !run_time_str.empty()) {

            auto wav_file = session_root_path / fmt::format("{}.wav", run_time_str);
            if ( !std::filesystem::exists(wav_file) ) {

                miniaudio_impl::wav_write(
                    miniaudio_impl::buffer_ptr(), 
                    miniaudio_impl::buffer_length(), 
                    wav_file.string().c_str());
            }
        }
    };

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

            engine->set_progress_callback([&](float progress_in, float eta_in){
                progress = progress_in;
            });

            engine->set_decoder_callback([&](std::vector<float> pcm_in){
                auto count = miniaudio_impl::track_count();
                auto wav_file = session_root_path / run_time_str / fmt::format("{}.wav", count);
                miniaudio_impl::wav_write(pcm_in.data(), pcm_in.size(), wav_file.string().c_str());
                miniaudio_impl::track_add(pcm_in);
            });

            engine->set_decoder_eta_callback([&](float eta_in){
                eta = eta_in;
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

        // main window
        {
            imgui_scoped::Font font(english);
            ImGui::Begin("main", NULL, ImGuiWindowFlags_MenuBar);
            
            // Menubar
            if( is_initialized ) {
                imgui_scoped::StyleVar item_speacing(ImGuiStyleVar_ItemSpacing, {20.0f, 5.0f});
                imgui_scoped::Disabled disable(is_loading);
                if (ImGui::BeginMenuBar()) {

                    // generate speech
                    if ( ImGui::MenuItem("run") ) {
                        if ( !configs.empty() && !engine->is_busy() ) {
                            run_time_str = datetime_str();
                            auto session_path = session_root_path / run_time_str;
                            if ( !std::filesystem::exists(session_path) ) {
                                std::filesystem::create_directories(session_path);
                            }
                            request_session++;
                            request_count = 0;
                            miniaudio_impl::buffer_reset();

                            TTSRequest request{};
                            for (const auto& r : configs) {

                                request.text = r.text;
                                request.voice_name = r.voice;
                                request.max_new_tokens = 1024;
                                engine->push(request);
                                request_count++;
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
                            if( !miniaudio_impl::play() ) {
                                ImGui::OpenPopup("my_play_popup");
                            }
                        }

                        imgui_scoped::StyleVar popup_rounding(ImGuiStyleVar_PopupRounding, 6.0f);
                        if (ImGui::BeginPopup("my_play_popup")) {
                            ImGui::TextColored(ImColor(200,0,0,255), "play failed.");
                            ImGui::EndPopup();
                        }

                        if ( ImGui::MenuItem("stop") ) {
                            miniaudio_impl::stop();
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
                if( is_initialized && engine->is_busy() && request_count ){
                    ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), "Generating..");
                    ImGui::ProgressBar(progress_total(), ImVec2(-1.0f, 0.0f), "Total..");
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
            if( is_initialized ) {
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
                        ImGui::EndPopup();
                    }

                    // column 2
                    ImGui::TableSetColumnIndex(2);
                    imgui_scoped::TableTextCentered("status");

                    // column 3
                    ImGui::TableSetColumnIndex(3);
                    if(ImGui::Selectable("option")) {
                        ImGui::OpenPopup("my_option_popup");
                    }

                    if (ImGui::BeginPopup("my_option_popup")) {
                        for (const auto& voice : voices ) {
                            if ( ImGui::MenuItem( voice.c_str(), NULL, default_voice == voice) ) {
                                default_voice = voice;
                                set_default_voice();
                            }
                        }
                        ImGui::EndPopup();
                    }

                    imgui_scoped::StyleVar s_var(ImGuiStyleVar_SelectableRounding, 12.0f);
                    ImGuiListClipper clipper;
                    clipper.Begin(configs.size());

                    static int last_selected_id = -1;
                    static int edit_select_id = -1;
                    size_t num = miniaudio_impl::track_count();

                    bool is_shift_down = ImGui::IsKeyDown(ImGuiKey_LeftShift);
                    bool is_delete_down = ImGui::IsKeyDown(ImGuiKey_Delete);
                    if ( disable_edit ) {
                        unselected_all();
                    }

                    while (clipper.Step()) {
                        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++) {
                            auto& cfg = configs[i];
                            bool idle = ( request_count == 0 );
                            bool done = ( i < num );
                            bool ongoing = ( i == num );
                            bool todo = ( i > num );
                            bool active = ( request_count > 0 );
                            bool selected = ( ongoing && active );
                                selected |= ( cfg.selected );

                            if ( ongoing && active ) {
                                ImGui::SetScrollHereY(0.5f);
                            }

                            imgui_scoped::ID id(i);
                            ImGui::TableNextRow();

                            ImGui::TableSetColumnIndex(0); 
                            if(ImGui::Selectable(std::to_string(i).c_str(), selected, select_flags)) {
                               miniaudio_impl::play(i);
                                if ( is_shift_down ) { // range select
                                    if( last_selected_id >= 0 ) {
                                        int start = std::min(last_selected_id, (int)i);
                                        int end = std::max(last_selected_id, (int)i);

                                        unselected_all();
                                        for (size_t j = start; j <= end; j++) {
                                            configs[j].selected = true;
                                        }
                                    }
                                } else {
                                    last_selected_id = i;
                                    unselected_all();
                                    configs[i].selected = true;
                                }
                            }

                            if (ImGui::IsItemFocused()) {
                                if (ImGui::IsMouseDoubleClicked(0)) {
                                    edit_select_id = last_selected_id;
                                }
                            }

                            if ( edit_select_id != last_selected_id ) {
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
                            if ( !idle && ongoing ) {
                                ImGui::SameLine();
                                float r = ImGui::GetFrameHeight() * 0.5f;
                                auto color = ImGui::GetStyle().Colors[ImGuiCol_Text];
                                int arcs = engine->is_decoding()?2:1;
                                float cell_width = ImGui::GetContentRegionAvail().x;
                                float offset_x = cell_width* 0.5f - r;
                                if (offset_x > 0.0f) {
                                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset_x);
                                }
                                
                                ImSpinner::SpinnerRainbow("ongoing", r, 2.f, color, 8.f, 0.0f, ImSpinner::PI_2, arcs);
                            } else {
                                imgui_scoped::TableTextCentered(done?"done":ongoing?"-":todo?"todo":"...");
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
                        last_selected_id = -1;
                        std::erase_if(configs, [](const auto& item){ return item.selected; });
                    }
                }
                ImGui::EndChild();
            }

            // Check if the cancellation operation has completed
            if ( cancle_status.valid() ) {
                if ( cancle_status.wait_for(10ms) == std::future_status::ready ) {
                    cancle_status.get();
                    cancle_status = {};
                    request_count = 0;
                    save();
                    is_cancelling = false;
                }
            }

            // Check if the text splitting operation has completed
            if ( split_text_status.valid() ) {

                if ( split_text_status.wait_for(10ms) == std::future_status::ready ) {
                    const auto& chunks = split_text_status.get();
                    if ( chunks.has_value() ) {
                        TextProcessor processor;
                        configs.clear();
                        config_item_s item;
                        for (int i = 0; i < chunks->size(); i++) {
                            item.id = i;
                            item.text = processor.clean_text(chunks.value()[i]);
                            item.voice = default_voice;
                            item.selected = false;
                            item.done = false;
                            configs.push_back(item);
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

    save();
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

