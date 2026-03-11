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
    uint32_t rayBounces = 16;
    bool ommEnabled = false; // Opacity Micro Maps (disabled by default until Phase 1 validated)
    uint32_t ommBakerLevel = 4; // OMM baker max subdivision level (1-8)
    bool simplifiedIndirect = false; // Skip detail textures on indirect bounces + simplify shadow AHS
    bool outputScale2x = false;     // Render world at 2x display resolution, Lanczos 3 downscale
    bool reflexEnabled = false;     // NVIDIA Reflex low-latency mode (VK_NV_low_latency2)
    bool reflexBoost = false;       // Reflex Boost — raise GPU clocks during latency-sensitive work
    bool vrrMode = false;           // VRR frame cap: 3600*Hz/(Hz+3600) via Reflex frameLimitUs
    bool needRecreate = false;

    uint32_t chunkBuildingBatchSize = 6;
    uint32_t chunkBuildingTotalBatches = 6;
    uint32_t tonemappingMode = 1; // 0 = PBR Neutral, 1 = Reinhard Extended
    float minExposure = 0.0001f;       // Minimum exposure clamp
    float maxExposure = 8.0f;          // ~3 EV boost headroom (was 2.0, too restrictive for dark scenes)
    float exposureCompensation = 0.0f; // EV offset (-3 to +3)
    bool manualExposureEnabled = true;
    float manualExposure = 1.0f;
    bool casEnabled = false;
    float casSharpness = 0.5f;
    float middleGrey = 0.18f;          // Middle grey point (0.01 to 0.50)
    float Lwhite = 4.0f;               // White point for Reinhard Extended
    bool legacyExposure = false;       // Use legacy exposure algorithm (keeps legacy failure modes)
    float exposureUpSpeed = 1.0f;      // Max EV increase rate (EV/s)
    float exposureDownSpeed = 1.0f;    // Max EV decrease rate (EV/s)
    float exposureBrightAdaptBoost = 1.0f; // Multiplier applied when stopping down (improved mode)
    float exposureHighlightProtection = 1.0f; // 0..1, improved mode only
    float exposureHighlightPercentile = 0.95f; // 0..1, improved mode only (was 0.985, now caps at 95th percentile)
    float exposureHighlightSmoothingSpeed = 10.0f; // 0..30, 0 disables smoothing
    float exposureLog2MaxImproved = 14.0f; // Histogram max log2(luminance) for improved mode
    float saturation = 1.3f;           // Saturation/Vibrance boost (0.0 to 2.0)
    uint32_t upscalerPreset = 4; // DLSS: Preset D (default). Generic for future upscalers.

    // SDR output transfer function
    // 0 = Gamma 2.2, 1 = sRGB
    uint32_t sdrTransferFunction = 0;

    // PsychoV tonemapper (RenoDX psycho_test11)
    bool psychoEnabled = true;
    float psychoHighlights = 1.0f;       // 0.0-3.0, default 1.0
    float psychoShadows = 1.0f;          // 0.0-3.0, default 1.0
    float psychoContrast = 1.0f;         // 0.0-3.0, default 1.0
    float psychoPurity = 1.0f;           // 0.0-3.0, default 1.0 (perceptual saturation)
    float psychoBleaching = 0.0f;        // 0.0-1.0, default 0.0 (photopigment bleaching)
    float psychoClipPoint = 100.0f;      // 1.0-500.0, default 100.0 (Neutwo clip)
    float psychoHueRestore = 0.5f;       // 0.0-1.0, default 0.5 (hue correction)
    float psychoAdaptContrast = 1.0f;    // 0.0-3.0, default 1.0 (Weber-Fechner adaptation)
    uint32_t psychoWhiteCurve = 1;       // 0 = Neutwo, 1 = Naka-Rushton
    float psychoConeExponent = 1.0f;     // 0.1-3.0, default 1.0 (Naka-Rushton exponent)

    // HDR10 output settings (default: disabled, pure SDR)
    bool hdrEnabled = false;
    float hdrPeakNits = 1000.0f;          // Display peak brightness (400–10000 nits)
    float hdrPaperWhiteNits = 203.0f;     // ITU-R BT.2408 reference white
    float hdrUiBrightnessNits = 100.0f;   // UI brightness in HDR mode (50–300 nits)

    // Area lights
    bool areaLightsEnabled = true;
    bool restirEnabled = true;            // ReSTIR DI temporal reuse for area lights
    float areaLightIntensity = 1.0f;      // Global multiplier [0.0 - 5.0]
    float areaLightRange = 48.0f;         // Max cull distance [8 - 512]
    float shadowSoftness = 1.0f;          // Shadow softness multiplier [0.0 - 2.0]

    // ReSTIR DI tuning
    int restirCandidates = 32;            // Total RIS candidates per pixel [8 - 64]
    int restirTemporalMClamp = 20;        // Temporal reservoir M clamp [5 - 50]
    int restirWClamp = 50;                // Importance weight W clamp [10 - 200]
    int restirSpatialTaps = 5;            // Spatial neighbor taps [1 - 10]
    int restirSpatialRadius = 30;         // Spatial search radius in pixels [5 - 60]

    // ReSTIR DI performance
    bool restirSimplifiedBRDF = false;    // Lambertian instead of Disney for area lights
    bool restirSpatialEnabled = false;    // Enable spatial reuse compute pass
    bool restirBounceEnabled = false;     // Enable ReSTIR on indirect bounces (1-3)

    float perBlockIntensity[50] = {       // Per-block intensity multiplier, indexed by LightTypeId
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    };
    float perBlockScale[50] = {          // Per-block halfExtent scale multiplier (1.0 = 100%)
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    };
    float perBlockYOffset[50] = {        // Per-block additive Y offset in blocks (all baked into LIGHT_DEFS)
    };
    float perBlockColorR[50] = {         // Per-block color R (0-1), sentinel -1 = use LIGHT_DEFS
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    };
    float perBlockColorG[50] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    };
    float perBlockColorB[50] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    };
    int blockLightMode[50] = {};         // Per-block light mode: 0=Auto, 1=ForceAreaLight, 2=ForceEmissive
};

class Renderer : public Singleton<Renderer> {
    friend class Singleton<Renderer>;

  public:
    static std::filesystem::path folderPath;
    static Options options;
    static float preExposure;  // Set by tone mapping, read by RT + DLSS (1-frame delay)

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
