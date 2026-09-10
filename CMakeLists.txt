cmake_minimum_required(VERSION 3.26)

project(audio8 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

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

add_executable(audio8-tts-engine main.cpp)
target_link_libraries(audio8-tts-engine PRIVATE audio8)

if(MSVC)
    target_compile_options(audio8-tts-engine PRIVATE /utf-8)

    set_target_properties(audio8-tts-engine PROPERTIES
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
    
    target_sources(audio8-tts-engine PRIVATE "${MANIFEST_FILE}")
endif()

add_custom_command(TARGET audio8-tts-engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${onnxruntime_SOURCE_DIR}/lib/${ONNX_DLL_NAME}"
    $<TARGET_FILE_DIR:audio8-tts-engine>
)
