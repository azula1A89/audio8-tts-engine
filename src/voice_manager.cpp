/*
MIT License

Copyright (c) [2026] [azula1A89]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <voice_manager.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <fmt/ranges.h>
#include <onnxruntime_cxx_api.h>

using json = nlohmann::json;

class VoiceManager::Impl {
public:
    Impl( Ort::Env& env, const std::filesystem::path& path, const RuntimeConfig& config) : env_(env), path_(path), session_(nullptr), initialized_(false) { }

    bool initialize() {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetIntraOpNumThreads(4);

        session_ = std::make_unique<Ort::Session>(env_, path_.c_str(), options);
        initialized_ = session_ != nullptr;
        return initialized_;
    }

    void registration(std::string voice_name, std::string transcript, std::filesystem::path audio) {
        if ( !initialized_ ) {
            initialized_ = initialize();
            if ( !initialized_ ) return;
        }

        std::filesystem::path root = "voices";
        std::filesystem::path voice_folder = root / voice_name;
        std::filesystem::path json_path = voice_folder / (voice_name + ".json");
        std::filesystem::path codes_path = voice_folder / (voice_name + ".bin");

        if ( !std::filesystem::exists( root ) ) {
            std::filesystem::create_directories( root );
        }

        if ( !std::filesystem::exists( voice_folder ) ) {
            std::filesystem::create_directories( voice_folder );
        }

        json profile;
        profile["name"] = voice_name;
        profile["transcript"] = transcript;
        profile["codes"] = codes_path.filename().string();

        if (!profile.empty()) {
            try{
                std::ofstream ostrm(json_path, std::ios::binary);
                ostrm.write(profile.dump(4).c_str(), profile.dump(4).size());
                ostrm.close();
            }catch(const std::exception& e){
                fmt::print("while saving:{} exception:{}\n", json_path.string(), e.what());
            }
        }

        std::vector<float> pcm = miniaudio_impl::load_audio(audio.string());
        std::vector<int64_t> codes;
        encode(pcm, codes);

        // save to file.
        std::ofstream out_file(codes_path, std::ios::out | std::ios::binary);
        out_file.write(reinterpret_cast<const char*>(codes.data()), codes.size() * sizeof(int64_t));
        out_file.close();
    }

    std::vector<std::string> list_voices() {
        std::vector<std::string> voices;
        std::filesystem::path root = "voices";
        if (std::filesystem::exists(root) && std::filesystem::is_directory(root)) {
            for (const auto& entry : std::filesystem::directory_iterator(root)) {
                if (entry.is_directory()) {
                    voices.push_back(entry.path().filename().string());
                }
            }
        }
        return voices;
    }

    bool load_profile(std::string voice_name, VoiceProfile& result){
        bool ret = false;
        std::filesystem::path root = "voices";
        std::filesystem::path json_path = root / voice_name / (voice_name + ".json");
        std::filesystem::path code_path = root / voice_name / (voice_name + ".bin");
        if ( std::filesystem::exists(json_path) ) {
            if ( std::filesystem::exists(code_path) ) {
                std::vector<int64_t> codes;
                if ( load_codes(code_path, codes) ) {

                    try {
                        std::ifstream f(json_path);
                        json data = json::parse(f);
                        if ( !data.empty() ) {
                            result.name = voice_name;
                            result.transcript = data["transcript"];
                            result.frames = codes.size() / audio8::NUM_CODEBOOKS;
                            result.codec_codes.resize(codes.size());
                            memcpy(result.codec_codes.data(), codes.data(), codes.size()*sizeof(int64_t));
                        }
                        f.close();
                        
                        ret = true;
                    } catch (const std::exception& e) {
                        fmt::print("exception occured while load {}, exception: {}", json_path.string(), e.what());
                    }
                } else {
                    fmt::print("failed to load codes from {}.\n", code_path.string());
                }
            } else {
                fmt::print("{} not found.\n", (voice_name + ".bin"));
            }
        } else {
            fmt::print("{} not found.\n", (voice_name + ".json"));
        }

        return ret;
    }

private:

    bool load_codes(const std::filesystem::path code_file, std::vector<int64_t>& codes) {

        bool ret = false;

        std::error_code ec;
        auto size = std::filesystem::file_size(code_file, ec);

        if (ec) {
            fmt::print("Could not get file size: {}\n", ec.message());
            return ret;
        }

        if( size % sizeof(int64_t) && ( (size / sizeof(int64_t)) % audio8::NUM_CODEBOOKS == 0 ) ) {
            fmt::print("data corrupted.\n");
            return ret;
        }

        codes.resize(size / sizeof(int64_t), 0);
        std::ifstream in_file(code_file, std::ios::binary);

        if (!in_file) {
            fmt::print("Could not open file for reading\n");
            return ret;
        }

        in_file.read(reinterpret_cast<char*>(codes.data()), size);

        if (!in_file) {
            fmt::print("Could not read the whole data\n");
            return ret;
        }

        return true;
    }

    bool encode(const std::vector<float>& pcm, std::vector<int64_t>& audio_codes) {
        bool ret = false;
        std::vector<Ort::Float16_t> buffer;
        buffer.resize(pcm.size(), Ort::Float16_t());
        for (size_t i = 0; i < pcm.size(); i++) {
            buffer[i] = Ort::Float16_t(pcm[i]);
        }
        
        std::vector<int64_t> input_shape = {1, 1, (int64_t)buffer.size() };
        auto memory_info = Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_values = Ort::Value::CreateTensor<Ort::Float16_t>(
            memory_info, 
            buffer.data(), 
            buffer.size(), 
            input_shape.data(), 
            input_shape.size());

        try {
            const char* input_names[] = {"audio"};
            const char* output_names[] = {"codes"};
            std::vector<Ort::Value> output_values = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_values, 1, output_names, 1);

            auto codes = output_values[0].GetTensorTypeAndShapeInfo().GetShape()[2];
            auto data = output_values[0].GetTensorMutableData<int64_t>();
            auto size = output_values[0].GetTensorSizeInBytes();
            audio_codes.resize(size / sizeof(int64_t));
            memcpy(audio_codes.data(), data, size);

            fmt::print("output_values: size={}, shape={}, bytes={}\n", 
                output_values.size(), 
                output_values[0].GetTensorTypeAndShapeInfo().GetShape(), 
            size);
            ret = true;
        } catch(const Ort::Exception& exception) {
            fmt::print("{}\n", exception.what());
        }
        return ret;
    }

private:
    // Ort::SessionOptions options_;
    Ort::Env& env_;
    const std::filesystem::path& path_;
    std::unique_ptr<Ort::Session> session_;
    bool initialized_;
};


VoiceManager::VoiceManager(
        void* env,
        const Audio8ModelPaths& paths,
        const RuntimeConfig& config) : 
        pImpl{ std::make_unique<Impl>(*((Ort::Env*)env), paths.codec_encoder, config) } { }
VoiceManager::~VoiceManager() = default;

bool VoiceManager::init() {
    return pImpl->initialize();
}

std::vector<std::string> VoiceManager::list_voices() { return pImpl->list_voices(); };

void VoiceManager::registration(
    std::string voice_name, 
    std::string transcript, 
    std::filesystem::path audio){

    pImpl->registration(voice_name, transcript, audio);
};

bool VoiceManager::load_profile(std::string voice_name, VoiceProfile& profile) { return pImpl->load_profile(voice_name, profile); }