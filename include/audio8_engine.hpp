#pragma once
#include <memory>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <array>

namespace miniaudio_impl {
    void play();
    void stop();
    std::vector<float> load_audio( const std::string& path );
    void wav_write(const float *buff, uint64_t count);
};

namespace audio8
{
constexpr int NUM_LAYERS = 24;
constexpr int NUM_FAST_LAYERS = 4;
constexpr int SEMATIC_BEGIN_ID = 151678;
constexpr int SEMATIC_END_ID = 155773;
constexpr int IM_END_ID = 151645;

constexpr int NUM_CODEBOOKS = 10;
constexpr int CODEBOOK_SIZE = 4096;

constexpr int HIDDEN_SIZE = 896;

constexpr int SAMPLE_RATE = 44100;
constexpr int FRAME_SAMPLES = 2048;

}

using code_frame = std::array<int64_t, audio8::NUM_CODEBOOKS>;
struct RuntimeConfig { };
struct SamplingConfig { };

template<class T>
class Tensor
{
public:
    std::vector<int64_t> shape;
    std::vector<T> data;

    T* ptr()
    {
        return data.data();
    }

    size_t size() const
    {
        return data.size();
    }
};

struct CacheTensor
{
    std::vector<int64_t> shape;

    std::vector<uint16_t> data;
};

struct SlowARInput {
    //INT64 [-1, 11, -1]
    Tensor<int64_t> codes;

    //INT64 [-1]
    Tensor<int64_t> input_pos;
};

struct SlowAROutput {
    //FLOAT32 [-1, -1, 4097]
    Tensor<float> logits;

    //FLOAT16 [-1, -1, 896]
    Tensor<int16_t> slow_hidden;//FLOAT16
};

struct FastARInput {
    Tensor<int16_t>& slow_hidden; 
    Tensor<int64_t> token_id;
    Tensor<int8_t> use_slow_hidden;
    Tensor<int64_t> input_pos;

    FastARInput(Tensor<int16_t>& slow_hidden_in, int64_t token_id_in, bool use_slow_hidden_in, int64_t input_pos_in) 
        : slow_hidden(slow_hidden_in) {
        token_id.data = {token_id_in};
        token_id.shape = {1,1};
        use_slow_hidden.data = {use_slow_hidden_in};
        use_slow_hidden.shape = {1};
        input_pos.data = {input_pos_in};
        input_pos.shape = {1};
    }
};

struct FastAROutput {
    //FLOAT32 [-1, 1, 4096]
    Tensor<float> logits;
};


struct Audio8ModelPaths
{
    std::filesystem::path root;

    std::filesystem::path manifest;
    std::filesystem::path tokenizer;

    std::filesystem::path slow_ar;
    std::filesystem::path fast_ar;
    std::filesystem::path codec_decoder;

    std::filesystem::path codec_encoder;

    Audio8ModelPaths() {
        Audio8ModelPaths(std::filesystem::current_path() / "models");
    }

    Audio8ModelPaths(std::filesystem::path model_dir) {
        root = model_dir;
        manifest = model_dir / "runtime_manifest.json";
        tokenizer = model_dir / "tokenizer" / "tokenizer.json";
        slow_ar = model_dir / "slow_ar_int4.onnx";
        fast_ar = model_dir / "fast_ar_int4.onnx";
        codec_decoder = model_dir / "codec_decoder_fp16.onnx";
        codec_encoder = model_dir / "registration" / "codec_encoder_fp16.onnx";
    }

    bool check() {
        bool result = std::filesystem::exists(manifest);
            result &= std::filesystem::exists(tokenizer);
            result &= std::filesystem::exists(slow_ar);
            result &= std::filesystem::exists(fast_ar);
            result &= std::filesystem::exists(codec_decoder);
            result &= std::filesystem::exists(codec_encoder);
        return result;
    }
};

struct TTSRequest
{
    std::string text;

    std::string voice_name;

    SamplingConfig sampling;

    int max_new_tokens = 256;
};

typedef void (*codebooks_callback) (const int64_t* v, size_t size);
typedef void (*progress_callback) (float progress);

class Audio8Engine 
{
    class Impl;
    std::unique_ptr<Impl> pImpl;

public:
    Audio8Engine();
    ~Audio8Engine();
    Audio8Engine(Audio8Engine&&);
    Audio8Engine& operator=(Audio8Engine&&);

    Audio8Engine(const Audio8Engine&) = delete;
    Audio8Engine& operator=(const Audio8Engine&) = delete;
    std::vector<std::string> list_voices();
    void registers(std::string voice_name, std::string transcript,
                  std::filesystem::path audio);
    bool initialize(const std::filesystem::path& model_dir = std::filesystem::current_path() / "models");
    void preload_model();
    void uninit();
    void synthesize( const TTSRequest& request, progress_callback progress = nullptr, codebooks_callback codebook = nullptr);
};