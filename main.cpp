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

std::string choose_data_path();
void imgui_parent_window();
float progress = 0.0f;
float eta = -1.0f;

int main(int argc, char** argv)
{
    const char* glsl_version = "#version 330";

    if (glfwInit() == GL_FALSE){
        return GL_FALSE;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    GLFWwindow*main_window = glfwCreateWindow(1200, 466, "audio8-tts-engine", NULL, NULL);
    if (main_window == NULL) {
        glfwTerminate();
        return GL_FALSE;
    }

    glfwMakeContextCurrent(main_window);
    glfwSwapInterval(1);
    if (glewInit() != GLEW_OK) {
        glfwTerminate();
        return GL_FALSE;
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    int width, height; 
    glfwGetWindowSize(main_window, &width, &height);
    glViewport(0, 0, width, height);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       // Enable Multi-Viewport / 
    
    ImGuiTheme::ApplyTweakedTheme(ImGuiTheme::ImGuiTheme_Darcula);

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
    style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.5f);

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(main_window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    std::string txt = "大家好，我是anthony。";
    bool is_loading = true;
    bool is_running = false;
    bool is_encoding = false;

    std::vector<std::string> voices;
    std::string new_voice_name;
    std::string transcript;
    std::string ref_audio_path;
    TTSRequest request{};

    std::unique_ptr<Audio8Engine> engine = std::make_unique<Audio8Engine>();
    if( !engine->initialize() ) return -1;
    
    std::future<void> engine_status = std::async(std::launch::async,[&](){
        engine->preload_model();
        is_loading = false;
    });

    std::future<void> synthesize_status = {};
    std::future<void> registration_status = {};

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

        if ( voices.empty() ) {
            voices = engine->list_voices();
        }

        // main window
        {
            imgui_scoped::Font font(english);
            ImGui::Begin("main", NULL, ImGuiWindowFlags_MenuBar);
            
            {
                imgui_scoped::Disabled disable(is_loading);
                if (ImGui::BeginMenuBar()) {

                    // select voice profile
                    if (ImGui::BeginMenu("voices")) {

                        for (const auto& voice : voices ) {
                            if ( ImGui::MenuItem( voice.c_str() ) ) {
                                request.voice_name = voice;
                            }
                        }

                        ImGui::EndMenu();
                    }

                    // generate speech
                    if ( ImGui::MenuItem("run") ) {
                        if ( !is_running && !txt.empty() ) {
                            synthesize_status = std::async(std::launch::async, [&](){
                                request.text = txt;
                                request.voice_name = request.voice_name.empty()?"anthony":request.voice_name;
                                request.max_new_tokens = 1024;
                                engine->synthesize( request, [](float progress_in, float eta_in){
                                    progress = progress_in;
                                    eta = eta_in;
                                });
                            });
                        }
                    }

                    // cancle generation
                    if ( ImGui::MenuItem("cancle") ) {
                        engine->cancel();
                    }

                    {
                        imgui_scoped::Disabled disable(is_running);
                        if ( ImGui::MenuItem("play") ) {
                            miniaudio_impl::play();
                        }

                        if ( ImGui::MenuItem("stop") ) {
                            miniaudio_impl::stop();
                        }
                    }

                    {
                        imgui_scoped::Disabled disable(is_running || is_loading);
                        if (ImGui::MenuItem("registration")) {

                            ImGui::OpenPopup("registration");
                        }

                        if (ImGui::BeginPopupModal("registration", NULL, ImGuiWindowFlags_AlwaysAutoResize))
                        {
                            ImGui::InputText("voice name", &new_voice_name);
                            ImGui::InputText("transcript", &transcript);
                            ImGui::InputTextWithHint("ref audio", "audio file path", &ref_audio_path);
                            ImGui::SameLine();
                            if (ImGui::Button("choose")) {
                                ref_audio_path = choose_data_path();
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

                    ImGui::EndMenuBar();
                }
            }


            if ( is_loading ) {

                ImGui::ProgressBar(-1.0f * (float)ImGui::GetTime(), ImVec2(-1.0f, 0.0f), "Loading..");
            }

            if ( registration_status.valid() ) {
                is_encoding = true;
                ImGui::ProgressBar(-1.0f * (float)ImGui::GetTime(), ImVec2(-1.0f, 0.0f), "Encoding..");

                if ( std::future_status::ready == registration_status.wait_for(std::chrono::milliseconds(1)) ) {
                    registration_status.get();
                    registration_status = {};
                    is_encoding = false;
                    voices = engine->list_voices();
                }
            }

            if ( synthesize_status.valid() ) {
                is_running = true;
                if ( eta < 0 ) {

                    ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), "Generating..");
                } else {

                    ImGui::ProgressBar(-1.0f * (float)ImGui::GetTime(), ImVec2(-1.0f, 0.0f), "Decoding..");
                }
                
                if ( std::future_status::ready == synthesize_status.wait_for(std::chrono::milliseconds(1)) ) {
                    synthesize_status.get();
                    synthesize_status = {};
                    is_running = false;
                }
            }

            // Text input
            {
                imgui_scoped::Font font(cjk);
                imgui_scoped::Disabled disable(is_running);
                ImGui::InputTextMultiline("##text to speak", &txt,
                    ImVec2(-FLT_MIN, -FLT_MIN), 
                    0);
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

std::string choose_data_path()
{
    std::string path="";
    NFD::Guard nfd_guard;
    NFD::UniquePath out_path;
    nfdfilteritem_t filter_item[1] = {{"*", "wav"}};
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



