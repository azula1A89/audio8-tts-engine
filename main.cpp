#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <audio8_engine.hpp>
#include <imgui_theme.h>
#include <imgui_stdlib.h>
#include <future>
#include "main.hpp"
#include <nfd.hpp>
#include <text_processor.hpp>

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

    Audio8ModelPaths paths{"models"};
    std::string txt = "大家好，我是anthony。";
    std::vector<std::string> voices;
    std::string default_voice = "anthony";
    std::string new_voice_name;
    std::string transcript;
    std::string ref_audio_path;

    std::unique_ptr<Audio8Engine> engine = std::make_unique<Audio8Engine>();

    std::future<void> engine_status = {};
    std::future<void> registration_status = {};
    std::future<std::optional<std::vector<std::string>>> split_text_status = {};

    bool is_initialized = false;
    bool is_loading = false;
    bool is_segmenting = false;
    bool is_encoding = false;

    float progress = 0.0f;
    float eta = -1.0f;

    struct config_item_s {
        int id;
        std::string text;
        std::string voice;
    };
    using config_t = std::vector<config_item_s>;
    config_t configs;

    int request_count = 0;
    std::vector<std::vector<float>> pcm_data;
    // std::mutex pcm_data_mutex;

    auto progress_total = [&request_count, &pcm_data, &progress](){
        float x = 0.0f;
        if ( request_count > 0 ) {
            x = pcm_data.size();
            x /= request_count;
        }
        return x;
    };

    auto is_finished = [&request_count, &pcm_data](){
        return (request_count == pcm_data.size());
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

            engine->set_progress_callback([&progress, &eta](float progress_in, float eta_in){
                progress = progress_in;
            });

            engine->set_decoder_callback([&pcm_data](std::vector<float> pcm_in){
                miniaudio_impl::wav_write(pcm_in.data(), pcm_in.size(), 
                        fmt::format("output_{}.wav", pcm_data.size()+1).c_str());
                pcm_data.push_back(std::move(pcm_in));
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
            
            {
                imgui_scoped::StyleVar item_speacing(ImGuiStyleVar_ItemSpacing, {20.0f, 5.0f});
                imgui_scoped::Disabled disable(is_loading);
                if (ImGui::BeginMenuBar()) {

                    {   // select voice profile
                        // imgui_scoped::Disabled disable(!is_finished());
                        if (ImGui::BeginMenu("voices")) {

                            for (const auto& voice : voices ) {
                                if ( ImGui::MenuItem( voice.c_str(), NULL, default_voice == voice) ) {
                                    default_voice = voice;
                                }
                            }

                            ImGui::EndMenu();
                        }
                    }

                    // generate speech
                    if ( ImGui::MenuItem("run") ) {
                        if ( !configs.empty() && is_finished() ) {
                            request_count = 0;
                            pcm_data.clear();

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
                        engine->cancel();
                        request_count = 0;
                        pcm_data.clear();
                    }

                    {
                        // imgui_scoped::Disabled disable(!is_finished());
                        if ( ImGui::MenuItem("play") ) {
                            if( !miniaudio_impl::play() ) {
                                miniaudio_impl::stop();
                                ImGui::OpenPopup("my_play_popup");
                            }
                        }

                        imgui_scoped::StyleVar popup_rounding(ImGuiStyleVar_PopupRounding, 6.0f);
                        if (ImGui::BeginPopup("my_play_popup")) {
                            ImGui::TextColored(ImColor(200,0,0,255), "play failed. check output.WAV");
                            ImGui::EndPopup();
                        }

                        if ( ImGui::MenuItem("stop") ) {
                            miniaudio_impl::stop();
                        }
                    }

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

                    if(ImGui::BeginMenu("theme")) {
                        for (int i = 0; i< ImGuiTheme::ImGuiTheme_Count; i++) {
                            imgui_scoped::ID id(i);
                            auto theme_name = ImGuiTheme::ImGuiTheme_Name((ImGuiTheme::ImGuiTheme_)i);
                            if ( ImGui::MenuItem(theme_name, NULL, theme == i) ) {
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
                        ImGui::EndMenu();
                    }

                    ImGui::EndMenuBar();
                }
            }


            if ( is_loading ) {
                imgui_scoped::StyleVar frame_padding(ImGuiStyleVar_FramePadding, {5.0f, 0.0f});
                imgui_scoped::StyleVar frame_rounding(ImGuiStyleVar_FrameRounding, 6.0f);
                ImGui::ProgressBar(-1.0f * (float)ImGui::GetTime(), ImVec2(-1.0f, 0.0f), "Loading..");
            }

            if ( split_text_status.valid() ) {
                imgui_scoped::StyleVar frame_padding(ImGuiStyleVar_FramePadding, {5.0f, 0.0f});
                imgui_scoped::StyleVar frame_rounding(ImGuiStyleVar_FrameRounding, 6.0f);
                ImGui::ProgressBar(-1.0f * (float)ImGui::GetTime(), ImVec2(-1.0f, 0.0f), "Segmenting text...");

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
                            configs.push_back(item);
                        }
                    }
                    split_text_status = {};
                    is_segmenting = false;
                }
            }

            if ( registration_status.valid() ) {
                imgui_scoped::StyleVar frame_padding(ImGuiStyleVar_FramePadding, {5.0f, 0.0f});
                imgui_scoped::StyleVar frame_rounding(ImGuiStyleVar_FrameRounding, 6.0f);
                is_encoding = true;
                ImGui::ProgressBar(-1.0f * (float)ImGui::GetTime(), ImVec2(-1.0f, 0.0f), "Encoding..");

                if ( std::future_status::ready == registration_status.wait_for(std::chrono::milliseconds(1)) ) {
                    registration_status.get();
                    registration_status = {};
                    is_encoding = false;
                    voices = engine->list_voices();
                }
            }

            if( !is_finished() ){
                imgui_scoped::StyleVar frame_padding(ImGuiStyleVar_FramePadding, {5.0f, 0.0f});
                imgui_scoped::StyleVar frame_rounding(ImGuiStyleVar_FrameRounding, 6.0f);
                ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), "Generating..");
                ImGui::ProgressBar(progress_total(), ImVec2(-1.0f, 0.0f), "Total..");
            }

            // Text input
            {
                imgui_scoped::Font font(cjk);
                imgui_scoped::Disabled disable(!is_finished() || is_segmenting);

                auto sz = ImGui::GetContentRegionAvail();
                bool changed = ImGui::InputTextMultiline("##text to speak", &txt,
                    ImVec2(-FLT_MIN, sz.y * 0.25f), 
                    0);

                if( changed ) {
                    is_segmenting = true;
                    configs.clear();
                    split_text_status = std::async(std::launch::async, [&txt, &engine, &default_voice](){
                        return engine->split_text_by_tokens(txt, 40);
                    });
                }
                
                ImGui::BeginChild("##chunk info");
                {
                    ImGuiTableColumnFlags table_flags = ImGuiTableFlags_BordersOuter 
                                    | ImGuiTableFlags_BordersInner 
                                    | ImGuiTableFlags_RowBg;
                    ImGuiSelectableFlags select_flags = ImGuiSelectableFlags_SpanAllColumns 
                                    | ImGuiSelectableFlags_AllowDoubleClick
                                    | ImGuiSelectableFlags_AllowOverlap;
                    ImGuiTableColumnFlags column_flags = ImGuiTableColumnFlags_WidthFixed;
                    // seq, text, status, options
                    imgui_scoped::Table table("##chunk table", 4, table_flags);
                    imgui_scoped::StyleVar frame_padding(ImGuiStyleVar_FramePadding, {0.0f, 0.0f});
                    imgui_scoped::StyleVar s_txt_align(ImGuiStyleVar_SelectableTextAlign, {0.5f, 0.5f});
                    imgui_scoped::StyleVar selectable_var(ImGuiStyleVar_SelectableRounding, 12.0f);

                    float ax = ImGui::GetContentRegionAvail().x;
                    ImGui::TableSetupColumn("seq", column_flags, 0.05f * ax);
                    ImGui::TableSetupColumn("text", column_flags, 0.77f * ax );
                    ImGui::TableSetupColumn("status", column_flags, 0.08f * ax);
                    ImGui::TableSetupColumn("options", column_flags, 0.1f * ax);
                    ImGui::TableHeadersRow();

                    
                    ImGuiListClipper clipper;
                    clipper.Begin(configs.size());

                    static int select_id = -1;
                    static int edit_select_id = -1;
                    while (clipper.Step()) {
                        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++) {
                            auto& cfg = configs[i];

                            bool done = ( i < pcm_data.size() );
                            bool ongoing = ( i == pcm_data.size() );
                            bool todo = ( i > pcm_data.size() );
                            bool active = ( request_count > 0 );
                            bool selected = ( ongoing && active );
                                selected |= ( select_id == cfg.id );

                            if ( ongoing && active ) {
                                ImGui::SetScrollHereY(0.5f);
                            }

                            imgui_scoped::ID id(i);
                            ImGui::TableNextRow();

                            ImGui::TableSetColumnIndex(0); 
                            if(ImGui::Selectable(fmt::format("{}", i).c_str(), selected, select_flags)) {
                                select_id = cfg.id;
                            }
                            if (ImGui::IsItemFocused()) {
                                if (ImGui::IsMouseDoubleClicked(0)) {
                                    edit_select_id = select_id;
                                }
                            }

                            if ( edit_select_id != select_id ) {
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
                            imgui_scoped::TableTextCentered(done?"done":ongoing?"ongoing":todo?"todo":"...");

                            ImGui::TableSetColumnIndex(3);
                            if (ImGui::BeginMenu(cfg.voice.c_str())) {

                                for (const auto& voice : voices ) {
                                    if ( ImGui::MenuItem( voice.c_str(), NULL, cfg.voice == voice) ) {
                                        cfg.voice = voice;
                                    }
                                }
                                ImGui::EndMenu();
                            }
                        }
                    }

                    if ( ImGui::IsKeyDown(ImGuiKey_Delete) ) {
                        if ( select_id >= 0 ) {
                            for(config_t::iterator it = configs.begin(); it != configs.end();) {
                                if ( it->id == select_id ) {
                                    it = configs.erase(it);
                                    break;
                                } else {
                                    ++it;
                                }
                            }
                            select_id = -1;
                        }
                    }
                }
                ImGui::EndChild();
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

