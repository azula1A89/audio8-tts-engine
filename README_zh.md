# Audio8 TTS Engine

[![Build and Release audio8-tts-engine](https://github.com/azula1A89/audio8-tts-engine/actions/workflows/build.yml/badge.svg)](https://github.com/azula1A89/audio8-tts-engine/actions/workflows/build.yml)

一个基于 **ONNX Runtime C++ API** 实现的轻量级 **Audio8-TTS 本地 C++ 推理引擎**。

本项目旨在将 Audio8-TTS 的推理流程原生实现为 C++，无需依赖原始 Python 运行环境。项目包含文本预处理、Prompt 构建、语义/自回归生成、Codec 解码、音色/参考音频管理、采样以及音频播放等功能。

**预编译版本可在 Releases 页面获取：**

- [Windows x64](https://github.com/azula1A89/audio8-tts-engine/releases/download/v0.0.1/audio8-tts-engine-windows.zip)
- [Linux x64](https://github.com/azula1A89/audio8-tts-engine/releases/download/v0.0.1/audio8-tts-engine-linux.tar.gz)

> **项目状态：** 实验性 / 早期阶段的开源实现  
> 当前推理流程已经可以正常工作，但性能、音频质量、平台兼容性以及 API 稳定性仍有进一步改进空间。

## 功能特性

- 原生 C++20 实现
- 基于 ONNX Runtime 的模型推理
- Audio8-TTS 两阶段自回归生成
  - Slow AR：生成语义 Token
  - Fast AR：生成音频 Codebook
- 自回归推理 KV Cache 管理
- 10 个 Codebook 的音频表示
- 基于 ONNX Codec Decoder 的波形重建
- 基于参考音频的音色克隆 / 语音合成
- 本地音色注册
- Temperature / Top-P / Top-K 采样
- UTF-8 文本清理与参考文本格式化
- WAV 音频输出及基于 miniaudio 的本地播放
- 基于 CMake 的构建系统
- 通过 CMake `FetchContent` 自动获取主要 C++ 依赖

## 工作流程

整体推理流程大致如下：

```text
                 ┌──────────────────┐
                 │     输入文本     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ TextProcessor    │
                 │    文本处理      │
                 └────────┬─────────┘
                          │
                          ▼
参考音色 ────────► ┌──────────────────┐
                    │ PromptBuilder    │
                    │   Prompt 构建    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Slow AR      │
                    │   语义 Token     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Fast AR      │
                    │  10 个音频 Codes │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Codec Decoder   │
                    │    Codec 解码    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    PCM / WAV     │
                    └──────────────────┘
```

首先由 Slow AR 模型生成语义 Token，然后 Fast AR 模型根据语义 Token 进一步生成对应的音频 Codebook Token。

所有音频帧生成完成后，Codec Decoder 将音频 Code 转换为 PCM 波形。

在自回归推理过程中，C++ 实现负责维护 KV Cache，并在每次模型调用后更新 Cache。

## 项目结构

```text
.
├── CMakeLists.txt
├── main.cpp
│
├── include/
│   ├── audio8_engine.hpp
│   ├── codec_decoder.hpp
│   ├── fast_ar_generator.hpp
│   ├── miniaudio_impl.hpp
│   ├── prompt_builder.hpp
│   ├── sampler.hpp
│   ├── slow_ar_generator.hpp
│   ├── text_processor.hpp
│   └── voice_manager.hpp
│
├── src/
│   ├── audio8_engine.cpp
│   ├── codec_decoder.cpp
│   ├── fast_ar_generator.cpp
│   ├── miniaudio_impl.cpp
│   ├── prompt_builder.cpp
│   ├── sampler.cpp
│   ├── slow_ar_generator.cpp
│   ├── text_processor.cpp
│   └── voice_manager.cpp
│
├── models/
│   └── readme.txt
│
└── voices/
    ├── anthony/
    └── tom/
```

### 主要组件

| 组件 | 作用 |
|---|---|
| `Audio8Engine` | TTS 引擎高层接口及整体推理流程 |
| `TextProcessor` | UTF-8 文本清理与标准化 |
| `PromptBuilder` | 根据文本和参考音色构建模型输入 |
| `SlowARGenerator` | Slow AR 语义 Token 自回归生成 |
| `FastARGenerator` | Fast AR 音频 Code 自回归生成 |
| `Sampler` | Temperature / Top-P / Top-K 采样 |
| `CodecDecoder` | 将音频 Code 转换为 PCM 波形 |
| `VoiceManager` | 音色注册及参考 Code 管理 |
| `MiniAudio` | 音频加载、WAV 写入及音频播放 |

## 模型

本项目使用：

**Audio8-TTS-Preview-0.6B-ONNX-INT4**

模型地址：

[https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4)

模型目录结构应如下：

```text
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
│
├── tokenizer/
│   └── tokenizer.json
│
└── registration/
    ├── codec_encoder_fp16.onnx
    └── codec_encoder_fp16.onnx.data
```

模型文件**不包含在本仓库中**。

请从模型仓库下载对应模型文件，并放置到项目的 `models/` 目录中。

## 环境要求

### 软件

- CMake 3.26 或更高版本
- 支持 C++20 的编译器
- Git
- 首次配置/构建时需要互联网连接，用于自动下载依赖

当前 CMake 配置支持：

- Windows x64
- Linux x64
- macOS ARM64（**未测试**）

其他平台可能需要根据实际环境修改 ONNX Runtime 的配置。

### 第三方依赖

项目目前使用：

- ONNX Runtime 1.29.0
- fmt 11.1.0
- nlohmann/json 3.12.0
- miniaudio 0.11.25
- tokenizers-cpp

主要依赖会通过 CMake 自动下载。

## 编译

克隆仓库：

```bash
git clone https://github.com/azula1A89/audio8-tts-engine
cd audio8-tts-engine
```

配置：

```bash
cmake -S . -B build
```

编译：

```bash
cmake --build build --config Release
```

编译完成后生成：

```text
audio8-tts-engine
```

Windows 下对应：

```text
audio8-tts-engine.exe
```

构建完成后，CMake 会自动将 `models/` 和 `voices/` 目录复制到可执行文件所在目录。

## 使用方法

### 1. 使用默认音色

```bash
audio8-tts-engine "Hello, this is Audio8 TTS."
```

当前示例程序默认使用 `anthony` 音色。

### 2. 使用指定音色

```bash
audio8-tts-engine "朋友们大家好，我是anthony。" anthony
```

音色名称对应 `voices/` 下的目录名称。

例如：

```text
voices/
├── anthony/
└── tom/
```

可以使用：

```bash
audio8-tts-engine "Hello, nice to meet you." tom
```

### 3. 注册新的音色

注册音色需要提供：

1. 音色名称
2. 与参考音频对应的文本
3. 参考音频文件

例如：

```bash
audio8-tts-engine my_voice "This is the transcript of the reference audio." reference.wav
```

注册过程：

```text
reference.wav
     │
     ▼
Audio Encoder
     │
     ▼
Audio Codec Codes
     │
     ├── my_voice.bin
     │
     └── my_voice.json
```

生成的文件会保存到：

```text
voices/
└── my_voice/
    ├── my_voice.bin
    └── my_voice.json
```

## 音色文件格式

每个已注册音色包含一个 JSON 配置文件和一个二进制 Codec Code 文件。

例如：

```json
{
    "name": "anthony",
    "transcript": "Reference transcript",
    "codes": "anthony.bin"
}
```

`.bin` 文件中保存由 Codec Encoder 根据参考音频生成的 Codec Token 序列。

当前实现使用：

```text
10 codebooks / frame
```

## C++ API

主要公共接口为 `Audio8Engine`。

基本使用方式：

```cpp
#include <audio8_engine.hpp>

Audio8Engine engine;

if (!engine.initialize()) {
    return -1;
}

engine.preload_model();

TTSRequest request;
request.text = "Hello from Audio8 TTS.";
request.voice_name = "tom";
request.max_new_tokens = 1024;

engine.synthesize(
    request,
    [](float progress) {
        // progress: 0.0 ~ 1.0
    }
);
```

还可以通过可选的 Callback 获取生成的音频 Code：

```cpp
engine.synthesize(
    request,
    progress_callback,
    [](const int64_t* codes, size_t size) {
        // codes 包含一个生成的音频帧
    }
);
```

因此，可以在 C++ 应用中进一步对生成的音频 Code 进行实时或增量处理，而不仅限于命令行示例程序。

## 推理架构

整个引擎被划分为多个相对独立的模块。

### Prompt 构建

`PromptBuilder` 将以下内容组合起来：

- 目标文本
- 参考文本
- 参考音频 Codec Codes
- Tokenizer 信息

并构建 Audio8-TTS 模型所需要的 Tensor 输入。

### Slow AR

`SlowARGenerator` 执行 Slow AR ONNX 模型，并生成：

- Semantic Token Logits
- Fast AR 阶段所需要的 Hidden Representation

实现中在 C++ 侧维护 Slow AR 的 KV Cache，并根据当前生成位置更新 Cache。

### Semantic Token 采样

`Sampler::sample_semantic()` 将采样范围限制在 Semantic Token 区间以及结束 Token。

支持：

- Temperature
- Top-P
- Top-K

同时维护最近生成的部分 Semantic Token 历史，用于降低短距离重复生成的概率。

### Fast AR

对于每一个 Semantic Token，`FastARGenerator` 生成对应的 10 通道 Audio Code。

第一个 Codebook 根据 Semantic Token 得到，其余 Codebook 采用自回归方式逐个生成。

Fast AR 同样拥有独立的 KV Cache。

### Codec 解码

所有音频 Code 生成完成后：

```text
[10 codebooks × T frames]
```

通过 `CodecDecoder` 转换为 PCM 波形。

最终的 PCM 数据可以写入 WAV 文件，并通过 miniaudio 播放。

## 项目目标

本项目主要用于：

- 提供 Audio8-TTS 推理流程的原生 C++ 实现；
- 方便将本地 Audio8-TTS 推理集成到 C++ 应用程序；
- 推理阶段无需依赖 Python 运行环境；
- 提供对中间 Audio Code 生成阶段的访问；
- 为未来的实时 TTS 或嵌入式/本地 TTS 应用提供基础。

项目在设计上尽可能保持各个推理组件之间的独立性，使后续可以分别替换或优化各个阶段。

## 当前限制

目前项目仍处于早期阶段，后续计划或需要改进的方向包括：

- 推理性能优化；
- GPU Execution Provider 支持与配置；
- 流式音频解码与播放；
- 更完善的取消和异步 API；
- 更完善的错误处理；
- 更广泛的平台测试；
- API 和配置系统进一步整理；
- 音频质量优化；
- 更完善的模型及 ONNX Runtime 兼容性检查。

目前示例程序是在整个自回归生成完成后，再统一进行 Codec 解码。

虽然 `synthesize()` 已经可以通过 Callback 输出生成的 Audio Code，但当前 Demo 程序还没有实现完整的 **PCM 流式生成与播放**。

## 与 Audio8-TTS 的关系

本项目是基于 Audio8-TTS 导出的 ONNX 模型实现的 C++ 推理引擎。

它并不是 Audio8-TTS 原项目或模型的替代品。

有关模型结构、模型训练、许可证以及原始 Python 实现，请参考上游 Audio8-TTS 项目及模型仓库。

## 第三方软件

本项目使用以下开源项目：

- [fmt](https://github.com/fmtlib/fmt)
- [nlohmann/json](https://github.com/nlohmann/json)
- [miniaudio](https://github.com/mackron/miniaudio)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)

请分别查看各项目的许可证及仓库，以了解对应的使用和分发条款。

## 许可证

本项目采用：

**MIT License**

详细内容请参见仓库中的 `LICENSE` 文件。

## 致谢

感谢以下项目的作者和维护者：

- Audio8-TTS
- ONNX Runtime
- tokenizers-cpp
- fmt
- nlohmann/json
- miniaudio