#include <fmt/color.h>
#include <fmt/ranges.h>
#include <audio8_engine.hpp>
#include <iostream>

using namespace std::literals;

void progress(float progress, float eta) {
    fmt::print( fmt::emphasis::bold | fg(fmt::color::pale_green), 
    "\r progress: {:.1f}%", progress * 100);
};

int main(int argc, char** argv) {
    std::unique_ptr<Audio8Engine> engine = std::make_unique<Audio8Engine>();
    if( !engine->initialize() ) return -1;

    if (argc < 2) {
        fmt::print(fmt::emphasis::bold, "\n\nUsage:\n");
        fmt::print(fmt::emphasis::bold, "  1. TTS using default voice:\n");
        fmt::print("     {} <text>\n\n", argv[0]);
        fmt::print(fmt::emphasis::bold, "  2. TTS using specific voice:\n");
        fmt::print("     {} <text> <voice name>\n\n", argv[0]);
        fmt::print(fmt::emphasis::bold, "  3. Voice registration:\n");
        fmt::print("     {} <voice name> <transcript> <audio>\n\n", argv[0]);
        fmt::print(fmt::emphasis::bold, "all available voices: {}\n", engine->list_voices());
        return -1;
    }
    
    engine->preload_model();

    if ( argc == 2 ) {

        TTSRequest request;
        request.text = argv[1];
        request.voice_name = "anthony";
        request.max_new_tokens = 1024;
        fmt::print("\n\n default voice \n\n");
        engine->synthesize( request, progress);
        miniaudio_impl::play();
    } else if ( argc == 3 ) {

        TTSRequest request;
        request.text = argv[1];
        request.voice_name = argv[2];
        request.max_new_tokens = 1024;
        fmt::print("\n\n try using {} as reference voice.\n\n", argv[2]);
        engine->synthesize( request, progress);
        miniaudio_impl::play();
    } else if ( argc == 4 ) {

        engine->registers(argv[1], argv[2], argv[3]);
        fmt::print("\n\n voice registration done. \n\n");
    }

    engine.reset();

    // press enter to exit
    fmt::print("\n\n Press Enter to exit...\n");
    std::cin.get();
    miniaudio_impl::stop();

    return 0;
}
