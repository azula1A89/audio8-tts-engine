# MIT License

# Copyright (c) [2026] [azula1A89]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.26)

project(audio8 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT WIN32)
  option(GLFW_BUILD_WAYLAND "Build support for Wayland" OFF)
  option(GLFW_BUILD_X11 "Build support for X11" ON)
  option(NFD_PORTAL "Use xdg-desktop-portal instead of GTK" ON)
else()
  option(GLFW_BUILD_WAYLAND "Build support for Wayland" OFF)
  option(GLFW_BUILD_X11 "Build support for X11" OFF)
  option(NFD_PORTAL "Use xdg-desktop-portal instead of GTK" OFF)
endif()
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
option(GLFW_DOCUMENT_INTERNALS "Include internals in documentation" OFF)

include(FetchContent)

FetchContent_Declare(
    glew
    URL https://github.com/nigels-com/glew/releases/download/glew-2.3.1/glew-2.3.1.tgz
)
FetchContent_MakeAvailable(glew)

FetchContent_Declare(
    glfw
    URL https://github.com/glfw/glfw/releases/download/3.5.1/glfw-3.5.1.zip
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.92.9b-docking.zip
)
FetchContent_MakeAvailable(imgui)

if(NOT TARGET imgui)
    
    add_subdirectory(${glew_SOURCE_DIR}/build/cmake)
    add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/theme/imgui_theme.cpp
    )
    target_include_directories(imgui PUBLIC "${imgui_SOURCE_DIR}")
    target_include_directories(imgui PUBLIC "${imgui_SOURCE_DIR}/misc")
    target_include_directories(imgui PUBLIC "${imgui_SOURCE_DIR}/misc/cpp")
    target_include_directories(imgui PUBLIC ${glew_SOURCE_DIR}/include)
    target_include_directories(imgui PUBLIC ${glfw_SOURCE_DIR}/include)
    target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/theme)
    target_link_libraries(imgui PUBLIC glew glfw ${OPENGL_LIBRARIES})
endif()

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/include.zip
)
FetchContent_MakeAvailable(json)
if(NOT TARGET json)
    add_library(json INTERFACE)
    target_include_directories(json INTERFACE "${json_SOURCE_DIR}/include")
endif()

FetchContent_Declare(
    miniaudio
    URL https://github.com/mackron/miniaudio/archive/refs/tags/0.11.25.zip
)
FetchContent_MakeAvailable(miniaudio)

if(NOT TARGET miniaudio)
    add_library(miniaudio INTERFACE)
    target_include_directories(miniaudio INTERFACE "${miniaudio_SOURCE_DIR}")
endif()


FetchContent_Declare(
    fmt
    URL https://github.com/fmtlib/fmt/archive/refs/tags/11.1.0.zip
)
FetchContent_MakeAvailable(fmt)

if(WIN32)
    set(ONNXURL https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-win-x64-1.29.0.zip)
    set(ONNXHASH SHA256=c9b4b7086b529ad814f428c1bad028e20a25d7dc0699836775faace4ab5b78b2)
    set(ONNX_DLL_NAME "onnxruntime.dll")
    set(ONNX_LIB_NAME "onnxruntime.lib")
elseif(APPLE)
    set(ONNXURL https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-osx-arm64-1.29.0.tgz)
    set(ONNXHASH SHA256=d0706fc34f315d8c88639d0a8c81f2e09e815f282cabed3493c06a054352cf92)
    set(ONNX_DLL_NAME "libonnxruntime.dylib")
    set(ONNX_LIB_NAME "libonnxruntime.dylib")
else()
    set(ONNXURL https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-linux-x64-1.29.0.tgz)
    set(ONNXHASH SHA256=c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba)
    set(ONNX_DLL_NAME "libonnxruntime.so")
    set(ONNX_LIB_NAME "libonnxruntime.so")
endif()

FetchContent_Declare(
    onnxruntime
    URL ${ONNXURL}
    URL_HASH ${ONNXHASH}
)
FetchContent_MakeAvailable(onnxruntime)

add_library(onnxruntime_imported SHARED IMPORTED)
set_target_properties(onnxruntime_imported PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
)

FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG        main
)

FetchContent_MakeAvailable(tokenizers_cpp)

if(WIN32)
    set_target_properties(onnxruntime_imported PROPERTIES
        IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/${ONNX_LIB_NAME}"
        IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/${ONNX_DLL_NAME}"
    )
else()
    set_target_properties(onnxruntime_imported PROPERTIES
        IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/${ONNX_LIB_NAME}"
    )
endif()

file(GLOB_RECURSE _HDRS "include/*.hpp")
file(GLOB_RECURSE _SRCS "src/*.cpp" "src/*.hpp")

add_library(audio8 ${_SRCS} ${_HDRS})

target_include_directories(audio8 PUBLIC include)

target_link_libraries(audio8 
    PUBLIC 
        fmt::fmt
    PRIVATE 
        json
        miniaudio
        onnxruntime_imported
        tokenizers_cpp
)

set(app_name audio8-tts-engine)

add_executable(${app_name} main.cpp)
target_link_libraries(${app_name} PRIVATE audio8 imgui)

if(MSVC)
    target_compile_options(${app_name} PRIVATE /utf-8)

    set_target_properties(${app_name} PROPERTIES
        VS_GLOBAL_EnableUnitySupport OFF
    )
    
    set(MANIFEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/utf8.manifest")
    file(WRITE "${MANIFEST_FILE}"
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
        "<assembly xmlns=\"urn:schemas-microsoft-com:asm.v1\" manifestVersion=\"1.0\">\n"
        "  <application xmlns=\"urn:schemas-microsoft-com:asm.v3\">\n"
        "    <windowsSettings>\n"
        "      <activeCodePage xmlns=\"http://schemas.microsoft.com/SMI/2019/WindowsSettings\">UTF-8</activeCodePage>\n"
        "    </windowsSettings>\n"
        "  </application>\n"
        "</assembly>\n"
    )
    
    target_sources(${app_name} PRIVATE "${MANIFEST_FILE}")
endif()

add_custom_command(TARGET ${app_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${onnxruntime_SOURCE_DIR}/lib/${ONNX_DLL_NAME}"
    $<TARGET_FILE_DIR:${app_name}>
)

add_custom_command(
  TARGET ${app_name}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/models
          $<TARGET_FILE_DIR:${app_name}>/models

  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/voices
          $<TARGET_FILE_DIR:${app_name}>/voices

  COMMAND ${CMAKE_COMMAND} -E copy_directory 
          ${CMAKE_CURRENT_SOURCE_DIR}/fonts
          $<TARGET_FILE_DIR:${app_name}>/fonts

  COMMAND ${CMAKE_COMMAND} -E copy_directory 
          ${imgui_SOURCE_DIR}/misc/fonts
          $<TARGET_FILE_DIR:${app_name}>/fonts

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/imgui.ini"
    $<TARGET_FILE_DIR:${app_name}>
)

