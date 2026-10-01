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


#include <fmt/color.h>
#include <fmt/ranges.h>
#include <audio8_engine.hpp>
#include <iostream>

using namespace std::literals;

void progress(float progress, float eta) {
    fmt::print( fmt::emphasis::bold | fg(fmt::color::pale_green), 
    "\r progress: {:.1f}%", progress * 100);
    fflush(stdout);
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
