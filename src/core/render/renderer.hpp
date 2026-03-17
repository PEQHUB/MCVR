#pragma once

#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <filesystem>

#include "core/render/gpu_profiler.hpp"

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
    bool outputScale2x = false;     // Render world at 2x display resolution, FSR1 EASU downscale
    bool reflexEnabled = false;     // NVIDIA Reflex low-latency mode (VK_NV_low_latency2)
    bool reflexBoost = false;       // Reflex Boost — raise GPU clocks during latency-sensitive work
    bool vrrMode = false;           // VRR frame cap: 3600*Hz/(Hz+3600) via Reflex frameLimitUs
    bool needRecreate = false;

    uint32_t chunkBuildingBatchSize = 6;
    uint32_t chunkBuildingTotalBatches = 6;
    uint32_t tonemappingMode = 1; // 0 = PBR Neutral, 1 = Reinhard Extended
    float minExposure = 1e-7f;         // Minimum exposure clamp (lowered for physical sun ~100k lux)
    float maxExposure = 2.0f;           // Moderate dark adaptation — caves dark but shadow detail visible
    float exposureCompensation = 0.0f; // EV offset (-3 to +3)
    bool manualExposureEnabled = false;  // Auto-exposure on by default (required for physical luminance range)
    float manualExposure = 2.54e-5f;  // EV100=15 (sunny day): 1/(1.2 * 2^15)
    uint32_t sharpenerMode = 0;  // 0=None, 1=CAS, 2=RCAS
    float casSharpness = 0.5f;   // Sharpness for both CAS and RCAS (0.0-1.0)
    float middleGrey = 0.18f;          // Middle grey point (0.01 to 0.50)
    float Lwhite = 4.0f;               // White point for Reinhard Extended
    bool legacyExposure = false;       // Use legacy exposure algorithm (keeps legacy failure modes)
    float exposureUpSpeed = 0.8f;      // Max EV increase rate (EV/s) — dark adaptation: gradual
    float exposureDownSpeed = 1.5f;    // Max EV decrease rate (EV/s) — bright adaptation: moderate
    float exposureBrightAdaptBoost = 1.0f; // Multiplier on downSpeed entering bright (1.0 = no boost, eliminates 4:1 asymmetry)
    float exposureHighlightProtection = 0.3f; // 0..1, soft nudge only; tonemapper handles clipping
    float exposureHighlightPercentile = 0.98f; // 0..1, less sensitive to emissive surface outliers
    float exposureHighlightSmoothingSpeed = 2.0f; // 0..30, ~0.35s half-life filters percentile noise
    float exposureLog2MaxImproved = 18.0f; // Histogram max log2(luminance) for improved mode (physical sun ~100k lux needs ~17)
    float saturation = 1.3f;           // Saturation/Vibrance boost (0.0 to 2.0)
    bool saturationAdaptive = false;   // Adaptive saturation: brightness+chroma-dependent (Special K style)
    bool noiseLOD = true;              // Noise quality LOD: reduce octaves with distance, skip gradient far away
    bool multiScatterGGX = true;       // Kulla-Conty multi-scatter GGX energy compensation (flag bit 7)
    bool eonDiffuse = true;            // EON energy-preserving diffuse BRDF, replaces Disney diffuse (flag bit 8)

    // Per-tonemapper configurable parameters (8 generic slots)
    float tonemapParams[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t upscalerPreset = 4; // DLSS: Preset D (default). Generic for future upscalers.

    // SDR output transfer function
    // 0 = Gamma 2.2, 1 = sRGB
    uint32_t sdrTransferFunction = 1;

    // PsychoV tonemapper (RenoDX psycho_test11)
    bool psychoEnabled = true;
    float psychoHighlights = 1.0f;       // 0.0-3.0, default 1.0
    float psychoShadows = 1.0f;          // 0.0-3.0, default 1.0
    float psychoContrast = 1.0f;         // 0.0-3.0, default 1.0
    float psychoPurity = 1.05f;          // 0.0-3.0, default 1.05 (perceptual saturation)
    float psychoBleaching = 0.0f;        // 0.0-1.0, default 0.0 (photopigment bleaching)
    float psychoClipPoint = 100.0f;      // 1.0-500.0, default 100.0 (Neutwo clip)
    float psychoHueRestore = 0.0f;       // 0.0-1.0, default 0.0 (hue correction off)
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
    float perBlockTemperatureK[50] = {}; // Per-type temperature override in Kelvin. 0 = use LIGHT_DEFS default.
    float areaLightIntensity = 1.0f;      // Global multiplier [0.0 - 5.0]
    float areaLightRange = 128.0f;        // Max cull distance [8 - 512]
    float shadowSoftness = 1.0f;          // Shadow softness multiplier [0.0 - 2.0]
    float colorExpansion = 1.0f;          // Per-block vivid color chroma boost [0.0 - 2.0] (1.0 = neutral)

    // ReSTIR DI tuning
    int restirCandidates = 32;            // Total RIS candidates per pixel [8 - 64]
    int restirTemporalMClamp = 20;        // Temporal reservoir M clamp [5 - 50]
    int restirWClamp = 30;                // Importance weight W clamp [10 - 200]
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

    // SHARC radiance cache
    bool sharcEnabled = true;
    float sharcSceneScale = 4.0f;           // Grid voxel size (1.0-20.0)
    float sharcRoughnessThreshold = 0.25f;  // Min roughness for cache query (0.0-1.0)
    int sharcAccumulationFrames = 32;       // Temporal accumulation window (4-256)
    int sharcStaleFrames = 16;              // Stale entry eviction threshold (4-128)
    int sharcDownscale = 1;                 // Update pass resolution divisor (1-8)
    int sharcUpdateBlockSize = 5;            // Sparse update NxN block size (2-8)
    int sharcUpdateBounces = 4;              // Max bounces in update pass (2-8)
    int sharcCapacityExponent = 21;          // Cache capacity = 2^N entries (20-24)

    // Parallax Occlusion Mapping
    bool  pomEnabled      = false;
    float pomHeightScale  = 0.05f;  // Tile-local depth scale (0.01–0.50)
    int   pomSteps        = 64;     // Linear search steps (8–512)
    int   pomRefinement   = 4;      // Binary refinement iterations (0–8)
    float pomFadeDistance = 64.0f;  // Distance in blocks to fade POM out (8–256)

    // Offline accumulation mode
    uint32_t offlineState = 0;       // 0=NORMAL, 1=FREE, 2=ACCUMULATING
    bool offlineGroundTruth = false; // unbiased path tracing mode (all hacks off)
    // Individual shader quality toggles
    bool beerLawShadows = false;
    bool noEmissionClamp = false;
    bool physicalSunDisk = true;
    bool noHandAmbient = false;
    uint32_t offlineBounces = 16;    // ray bounces during accumulation (1-128)
    bool offlineDisableRR = false;
    bool offlineDisableClamp = false;
    float offlineAperture = 0.0f;    // thin lens aperture (0=pinhole, 0.001-2.0)
    float dofStrength = 1.0f;        // artistic DOF multiplier (1.0=physical, up to 20.0 for cinematic)
    float offlineFocalDistance = 10.0f; // focal distance in blocks (1-256)
    bool offlineNativeRes = false;        // force render-res = display-res
    uint32_t offlineDenoised = 0;         // 0=raw, 1=DLSS+Welford, 2=DLSS temporal
    uint32_t savedUpscalerMode = 0;       // saved for restore on exit
    bool offlineNativeResActive = false;  // tracks if resolution was overridden

    // Diagnostics
    bool loggingEnabled = false;
};

class Renderer : public Singleton<Renderer> {
    friend class Singleton<Renderer>;

  public:
    static std::filesystem::path folderPath;
    static Options options;
    static float preExposure;  // Set by tone mapping, read by RT + DLSS (1-frame delay)
    static bool resetExposureAdaptation;  // Set by JNI on world load, consumed by tone mapping
    static uint32_t accumFrameCount;
    static std::shared_ptr<vk::DeviceLocalImage> accumOutputImage;

    // Offline accumulation pipeline (shared between RT and DLSS modules)
    static VkPipeline accumPipeline;
    static VkPipelineLayout accumPipelineLayout;
    static std::shared_ptr<vk::DeviceLocalImage> accumBufferImage;
    static VkDescriptorPool accumDescPool;
    static VkDescriptorSetLayout accumDescSetLayout;
    static std::vector<VkDescriptorSet> accumDescSets;
    static bool accumPipelineReady;

    static std::vector<std::shared_ptr<vk::DeviceLocalImage>> emissionImages;  // RT emission, read by tone mapping
    static std::vector<std::shared_ptr<vk::DeviceLocalImage>> renderResHdrImages;  // DLSS input (render-res HDR), read by tone mapping histogram
    static GpuProfiler gpuProfiler;

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
