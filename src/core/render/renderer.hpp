#pragma once

#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <filesystem>

class Textures;
class Framework;
class Buffers;
class World;

struct Options {
    uint32_t maxFps = 1e6;
    uint32_t inactivityFpsLimit = 1e6;
    bool vsync = true;
    uint32_t upscalerMode = 2;       // Quality preset: 0=Performance, 1=Balanced, 2=Quality, 3=Native/DLAA, 4=Custom
    uint32_t upscalerResOverride = 100; // Resolution override percentage (33-100%)
    uint32_t upscalerType = 1;       // 0=Off, 1=FSR3, 2=DLSS SR
    uint32_t upscalerQuality = 0;
    uint32_t denoiserMode = 1;
    uint32_t rayBounces = 4;
    uint32_t debugMode = 0;
    bool needRecreate = false;

    uint32_t chunkBuildingBatchSize = 2;
    uint32_t chunkBuildingTotalBatches = 4;
    uint32_t tonemappingMode = 1; // 0 = PBR Neutral, 1 = Reinhard Extended
    float minExposure = 0.0001f;       // Minimum exposure clamp
    float maxExposure = 2.0f;
    float exposureCompensation = 0.0f; // EV offset (-3 to +3)
    float middleGrey = 0.18f;          // Middle grey point (0.01 to 0.50)
    float Lwhite = 4.0f;               // White point for Reinhard Extended
    float saturation = 1.3f;           // Saturation/Vibrance boost (0.0 to 2.0)
    uint32_t upscalerPreset = 5; // DLSS: Preset E (latest transformer). Generic for future upscalers.

    // HDR10 output settings (default: disabled, pure SDR)
    bool hdrEnabled = false;
    float hdrPeakNits = 1000.0f;          // Display peak brightness (400–10000 nits)
    float hdrPaperWhiteNits = 203.0f;     // ITU-R BT.2408 reference white
    float hdrUiBrightnessNits = 100.0f;   // UI brightness in HDR mode (50–300 nits)
};

class Renderer : public Singleton<Renderer> {
    friend class Singleton<Renderer>;

  public:
    static std::filesystem::path folderPath;
    static Options options;

    ~Renderer();

    std::shared_ptr<Framework> framework();
    std::shared_ptr<Textures> textures();
    std::shared_ptr<Buffers> buffers();
    std::shared_ptr<World> world();

    void close();

  private:
    Renderer(GLFWwindow *window);

    std::shared_ptr<Framework> framework_;
    std::shared_ptr<Textures> textures_;
    std::shared_ptr<Buffers> buffers_;
    std::shared_ptr<World> world_;
    bool closed_ = false;
};
