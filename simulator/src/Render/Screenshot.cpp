#include "CarSim/Render/Screenshot.hpp"

#include "CnaEngine/Graphics/ScreenCapture.hpp"

#include "Microsoft/Xna/Framework/Graphics/GraphicsDevice.hpp"
#include "Microsoft/Xna/Framework/Graphics/Texture2D.hpp"
#include "System/IO/FileMode.hpp"
#include "System/IO/FileStream.hpp"

#include <exception>
#include <iostream>

namespace CarSim::Render
{
    using namespace Microsoft::Xna::Framework;
    using namespace Microsoft::Xna::Framework::Graphics;

    bool SaveBackBufferPng(GraphicsDevice& device, const std::string& path)
    {
        // cna-engine's capture reads the back buffer (GetBackBufferData) and writes the PNG;
        // a refusal (Reach device, no back buffer, unwritable file) comes back as a value.
        CnaEngine::Graphics::CaptureOptions options;
        options.path = path;
        const auto result = CnaEngine::Graphics::CaptureBackBuffer(device, options);
        if (!result.success) {
            std::cerr << "screenshot: failed to save '" << path << "': " << result.error << "\n";
        }
        return result.success;
    }

    bool SaveTexturePng(Texture2D& texture, const std::string& path)
    {
        try {
            System::IO::FileStream stream(path, System::IO::FileMode::Create);
            texture.SaveAsPng(&stream, texture.getWidthProperty(), texture.getHeightProperty());
            return true;
        } catch (const std::exception& ex) {
            std::cerr << "texture screenshot failed: " << ex.what() << "\n";
            return false;
        }
    }
}
