# Audio8 TTS Engine
[![Build and Release audio8-tts-engine](https://github.com/azula1A89/audio8-tts-engine/actions/workflows/build.yml/badge.svg)](https://github.com/azula1A89/audio8-tts-engine/actions/workflows/build.yml)


中文文档: [README_zh.md](README_zh.md)


A lightweight C++20 local inference engine for **Audio8-TTS**, implemented with **ONNX Runtime C++ API**.

This project focuses on running the Audio8-TTS inference pipeline natively in C++, without depending on the original Python runtime. It provides text preprocessing, prompt construction, semantic/autoregressive generation, codec decoding, voice/reference management, sampling, and audio playback.

Pre-built binaries are available on the releases page: 

- [Windows x64](https://github.com/azula1A89/audio8-tts-engine/releases/download/v0.0.1/audio8-tts-engine-windows.zip)
- [Linux x64](https://github.com/azula1A89/audio8-tts-engine/releases/download/v0.0.1/audio8-tts-engine-linux.tar.gz)


> **Project status:** Experimental / early-stage open-source implementation.  
> The inference pipeline is functional, but performance, audio quality, portability, and API stability may still be improved.

## Features

- Native C++20 implementation
- ONNX Runtime based inference
- Audio8-TTS two-stage autoregressive generation
  - Slow AR for semantic token generation
  - Fast AR for audio codebook generation
- KV-cache management for autoregressive inference
- 10-codebook audio representation
- ONNX codec decoder for waveform reconstruction
- Reference-voice based speech synthesis
- Local voice registration
- Temperature / Top-P / Top-K sampling
- UTF-8 text cleaning and reference-text formatting
- WAV output and local audio playback through miniaudio
- CMake-based build
- Automatic fetching of major C++ dependencies through CMake `FetchContent`

## How it works

The overall pipeline is approximately:

```text
                 ┌──────────────────┐
                 │   Input Text     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ TextProcessor    │
                 └────────┬─────────┘
                          │
                          ▼
Reference Voice ──► ┌──────────────────┐
                    │ PromptBuilder    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Slow AR       │
                    │ semantic tokens  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Fast AR       │
                    │ 10 audio codes   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Codec Decoder    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   PCM / WAV      │
                    └──────────────────┘
```

The generated semantic token is first produced by the Slow AR model. The Fast AR model then expands it into the corresponding audio codebook tokens. After all frames have been generated, the codec decoder converts the code sequence into PCM audio.

For autoregressive inference, the implementation maintains KV caches in C++ and updates them after every model invocation.

## Project structure

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

### Main components

| Component | Responsibility |
|---|---|
| `Audio8Engine` | High-level TTS engine and inference pipeline |
| `TextProcessor` | UTF-8 text cleaning and normalization |
| `PromptBuilder` | Builds model input from text and reference voice |
| `SlowARGenerator` | Slow autoregressive semantic-token generation |
| `FastARGenerator` | Fast autoregressive audio-code generation |
| `Sampler` | Temperature / Top-P / Top-K sampling |
| `CodecDecoder` | Converts audio codes into PCM waveform |
| `VoiceManager` | Voice registration and reference-code management |
| `MiniAudio` | Audio loading, WAV writing and playback |

## Model

This project uses:

**Audio8-TTS-Preview-0.6B-ONNX-INT4**

Model repository:

```text
https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4
```

The expected model directory is:

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

The model files are not included in this repository. Download them from the model repository and place them under `models/`.

## Requirements

### Software

- CMake 3.26 or newer
- C++20 compatible compiler
- Git
- Internet access during the first CMake configuration/build, because dependencies are fetched automatically

The current CMake configuration supports:

- Windows x64
- macOS ARM64 (Not tested)
- Linux x64

Other platforms may require changes to the ONNX Runtime package configuration.

### Dependencies

The project currently uses:

- ONNX Runtime 1.29.0
- fmt 11.1.0
- nlohmann/json 3.12.0
- miniaudio 0.11.25
- tokenizers-cpp

All dependencies are downloaded automatically by CMake.

## Build

Clone the repository:

```bash
git clone https://github.com/azula1A89/audio8-tts-engine
cd audio8-tts-engine
```

Configure:

```bash
cmake -S . -B build
```

Build:

```bash
cmake --build build --config Release
```

The generated executable is:

```text
audio8-tts-engine
```

After building, CMake automatically copies the `models/` and `voices/` directories next to the executable.

## Usage

### 1. Use the default voice

```bash
./audio8-tts-engine "Hello, this is Audio8 TTS."
```

The current example application uses `anthony(Female, Chinese)` as the default voice.

### 2. Use a specific reference voice

```bash
./audio8-tts-engine "Hello, nice to meet you." tom
```

The voice name corresponds to a directory under `voices/`.

### 3. Register a new voice

Provide:

1. Voice name
2. Transcript corresponding to the reference audio
3. Reference audio file

Example:

```bash
./audio8-tts-engine my_voice "This is the transcript of the reference audio." reference.wav
```

The registration process:

```text
reference.wav
     │
     ▼
Audio Encoder
     │
     ▼
audio codec codes
     │
     ├── my_voice.bin
     │
     └── my_voice.json
```

The generated files are stored under:

```text
voices/
└── my_voice/
    ├── my_voice.bin
    └── my_voice.json
```

## Voice format

Each registered voice contains a JSON profile and binary codec codes.

Example:

```json
{
    "name": "anthony",
    "transcript": "Reference transcript",
    "codes": "anthony.bin"
}
```

The `.bin` file contains the codec token sequence generated by the codec encoder.

The current implementation uses:

```text
10 codebooks / frame
```

## C++ API

The main public API is `Audio8Engine`.

Basic usage:

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

Generated codebooks can also be received through the optional callback:

```cpp
engine.synthesize(
    request,
    progress_callback,
    [](const int64_t* codes, size_t size) {
        // codes contains one generated audio frame
    }
);
```

This makes it possible to build applications that process generated audio/code frames incrementally instead of using only the command-line example.

## Inference architecture

The engine is divided into several relatively independent stages:

### Prompt construction

`PromptBuilder` combines:

- target text
- reference transcript
- reference codec codes
- tokenizer information

into the tensor layout expected by the Audio8-TTS model.

### Slow AR

`SlowARGenerator` executes the Slow AR ONNX model and produces semantic-token logits together with the hidden representation required by the Fast AR stage.

The implementation maintains the Slow AR KV cache in host memory and updates the cache according to the generated position.

### Semantic sampling

`Sampler::sample_semantic()` restricts sampling to the semantic-token range and the end-of-sequence token.

Sampling uses:

- temperature
- top-p
- top-k

The implementation also keeps a short history of previously generated semantic tokens to reduce immediate repetition.

### Fast AR

For every semantic token, `FastARGenerator` generates the ten-channel audio code representation.

The first codebook is derived from the semantic token, while the remaining codebooks are generated autoregressively.

The Fast AR model has its own KV cache.

### Codec decoding

After generation finishes, `CodecDecoder` converts:

```text
[10 codebooks × T frames]
```

into PCM waveform data.

The resulting waveform can be written as WAV and played through miniaudio.

## Design goals

This project is primarily intended to:

- provide a native C++ implementation of the Audio8-TTS inference pipeline;
- make local Audio8-TTS inference easier to integrate into C++ applications;
- avoid requiring a Python runtime for the inference side;
- expose the intermediate audio-code generation stage;
- provide a foundation for future real-time or embedded/local TTS applications.

The implementation intentionally keeps the inference components separated so that individual stages can be replaced or optimized independently.

## Current limitations

This is an early-stage implementation. Known areas for further work include:

- inference performance optimization;
- GPU execution provider support/configuration;
- streaming audio decoding/playback;
- more robust cancellation and asynchronous APIs;
- improved error reporting;
- broader platform testing;
- API cleanup and configuration support;
- audio quality tuning;
- more comprehensive model/runtime compatibility checks.

In particular, the current example performs codec decoding after the autoregressive generation stage completes. Although generated codebooks can be received through a callback, the demo application itself does not yet provide fully streamed PCM playback.

## Relationship to [Audio8-TTS](https://github.com/Edge0-AI/Audio8_TTS)

This repository is a C++ inference implementation built around the exported ONNX models from Audio8-TTS.

It is not a replacement for the original Audio8-TTS project or model.

Please refer to the upstream Audio8-TTS project and model repository for model details, licensing, training information, and the original reference implementation.

## Third-party software

This project uses the following open-source components:

- [fmt](https://github.com/fmtlib/fmt)
- [nlohmann/json](https://github.com/nlohmann/json)
- [miniaudio](https://github.com/mackron/miniaudio)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp.git)

Please refer to each project's own license and repository for the applicable terms.

## License

MIT License

## Acknowledgements

Thanks to the authors and maintainers of:

- Audio8-TTS
- ONNX Runtime
- tokenizers-cpp
- fmt
- nlohmann/json
- miniaudio

