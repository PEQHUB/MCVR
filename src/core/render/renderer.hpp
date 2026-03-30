#pragma once

#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <filesystem>

#include "core/render/block_model_table.hpp"
#include "core/render/gpu_profiler.hpp"
#include "core/render/texture_system.hpp"
#include "core/render/thread_pool.hpp"

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
    bool greedyMeshingEnabled = true; // Merge coplanar block faces (50-70% triangle reduction)
    bool simplifiedIndirect = false; // Skip detail textures on indirect bounces + simplify shadow AHS
    bool outputScale2x = false;     // Render world at 2x display resolution, FSR1 EASU downscale
    bool reflexEnabled = false;     // NVIDIA Reflex low-latency mode (VK_NV_low_latency2)
    bool reflexBoost = false;       // Reflex Boost — raise GPU clocks during latency-sensitive work
    bool vrrMode = false;           // VRR frame cap: 3600*Hz/(Hz+3600) via Reflex frameLimitUs
    bool needRecreate = false;
    bool reflexDirty = false;       // Deferred Reflex settings apply (avoids slider spam)

    uint32_t chunkBuildingBatchSize = 6;
    uint32_t chunkBuildingTotalBatches = 6;
    float chunkCullDistance = 384.0f;  // Max chunk distance in blocks (64-1024), chunks beyond are excluded from TLAS
    float chunkLodDistance = 160.0f;  // LOD boundary in blocks (64-512): ≤ = lossless 64B vertex, > = compact 32B
    float megaMergeDistance = 0.0f;  // Beyond this distance (blocks), chunks are merged into mega-BLASes (0=disabled)
    static constexpr uint32_t ommBatchCap = 2; // Max chunks per GPU batch when OMM active (prevents TDR)
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
    float exposureDownSpeed = 1.0f;    // Max EV decrease rate (EV/s) — moderate: less pulsing than 1.5, better low-light than 0.7
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
    bool psychoEnabled = true;   // kept for JNI compat — used by SDR tonemapMode==8 visibility
    int hdrTonemapMode = 0;      // 0 = PsychoVisual, 1 = BT.2390 EETF
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

    // HDR output settings (default: disabled, pure SDR)
    bool hdrEnabled = false;
    bool hdrScrgbMode = false;            // false = HDR10 (default, DLSS-FG compatible), true = scRGB
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

    // SER: Shader Execution Reordering
    // Disabled by default: Minecraft's material uniformity means SER overhead exceeds coherence gain.
    // Profiled 2026-03-25: SER ON harms perf vs OFF. Keep option for future re-evaluation.
    bool serEnabled = false;
    bool serHintsEnabled = true;  // explicit geometry-based coherence hints (on top of driver reorder)

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

    // Geometric displacement tessellation (replaces DDA intersection system)
    uint32_t displacementQuality = 0;  // 0=Off, 1+=Tessellation enabled
    uint32_t tessMaxLevel = 16;        // Max tessellation grid resolution (2–32)
    float tessNearDist  = 32.0f;       // Full tessellation distance (blocks)
    float tessMidDist   = 96.0f;       // Half tessellation distance
    float tessFarDist   = 192.0f;      // Quarter tessellation distance

    // Offline accumulation mode
    uint32_t offlineState = 0;       // 0=NORMAL, 1=FREE, 2=ACCUMULATING
    bool offlineGroundTruth = false; // unbiased path tracing mode (all hacks off)
    // Individual shader quality toggles
    bool beerLawShadows = false;
    bool noEmissionClamp = false;
    bool physicalSunDisk = true;
    bool noHandAmbient = false;
    bool entityNormalsEnabled = true;
    uint32_t offlineBounces = 16;    // ray bounces during accumulation (1-128)
    bool offlineDisableRR = false;
    bool offlineDisableClamp = false;
    float offlineAperture = 0.0f;    // thin lens aperture (0=pinhole, 0.001-2.0)
    float dofStrength = 1.0f;        // artistic DOF multiplier (1.0=physical, up to 20.0 for cinematic)
    float offlineFocalDistance = 10.0f; // focal distance in blocks (1-256)
    bool offlineNativeRes = false;        // N key: force render-res = display-res in FREE mode
    uint32_t offlineDenoised = 0;         // 0=Raw Fast (RR on), 1=Raw Accurate (RR off), 2=Denoised (epoch-based DLSS-RR)
    uint32_t dlssEpochLength = 16;        // frames per Denoised epoch (user-configurable, 4-64)
    uint32_t savedUpscalerMode = 0;       // saved for restore on exit
    bool offlineNativeResActive = false;  // tracks if resolution was overridden

    // Frame Generation (DLSS-G / FSR3-FG)
    bool frameGenEnabled = false;        // Master toggle
    uint32_t frameGenMode = 0;           // 0=Off, 1=On, 2=Auto (adaptive MFG)
    uint32_t frameGenMultiplier = 1;     // 1=2x, 2=3x, 3=4x, 4=5x, 5=6x (clamped to hardware max)

    // Volumetric clouds
    uint32_t cloudQuality = 3;        // 0=Off, 1=Low, 2=Medium, 3=High, 4=Ultra, 5=Extreme
    float cloudDensity = 1.0f;        // Density multiplier [0.1 - 3.0]
    float cloudCoverage = 0.5f;       // Base coverage [0.0 - 1.0]
    float cloudType = 0.67f;          // 0=Stratus, 0.33=Stratocumulus, 0.67=Cumulus, 1.0=Cumulonimbus
    float cloudSpeed = 1.0f;          // Wind speed multiplier [0.0 - 5.0]
    float cloudAltitude = 192.0f;     // Cloud base height in blocks [128 - 320]
    float cloudThickness = 64.0f;     // Cloud layer thickness [32 - 128]
    float cloudDetailStrength = 1.0f; // Detail erosion multiplier [0.0 - 2.0]
    uint32_t cloudScatterOctaves = 3; // Multi-scatter octaves [1 - 8]
    float cloudAmbientStrength = 1.0f;// Ambient occlusion [0.0 - 2.0]
    float cloudTemporalBlend = -1.0f; // Temporal blend override (-1 = use quality default)
    float cloudNoiseScale = 1000.0f;  // Noise texture period in blocks [64 - 4096]
    float cloudCellFrequency = 4.0f;  // Voronoi cell count across weather map [1 - 16]
    float cloudAtmosphereFadeDist = 800.0f; // Cloud atmospheric fade distance [200 - 2000]
    uint32_t cloudDebugMode = 0;      // 0=normal, 1-8=debug views (see cloud_raymarch.comp)
    float cloudWindAngle = 0.0f;      // Wind direction in radians [0, 2π]
    uint32_t cloudMarchStepsOverride = 0;  // 0=use quality preset, 32-256
    uint32_t cloudLightStepsOverride = 0;  // 0=use quality preset, 1-12
    uint32_t cloudResDivisorOverride = 0;  // 0=use quality preset, 1-4
    uint32_t cloudNoiseRes = 128;         // 3D noise texture resolution: 128 (8MB), 256 (64MB), 512 (512MB)
    float wetSurfaceStrength = 1.0f;  // Wet surface effect strength [0.0 - 2.0]

    // Diagnostics
    bool loggingEnabled = false;
    bool gpuDiagnostics = false;   // GPU checkpoints for DEVICE_LOST debugging (zero cost when false)
    bool validationLayers = false;
};

class Renderer : public Singleton<Renderer> {
    friend class Singleton<Renderer>;

  public:
    static std::filesystem::path folderPath;
    static Options options;
    static float preExposure;  // Set by tone mapping, read by RT + DLSS (1-frame delay)
    static bool resetExposureAdaptation;  // Set by JNI on world load, consumed by tone mapping
    static uint32_t accumFrameCount;
    static uint32_t dlssEpochFrame;   // current frame within Denoised epoch (0..epochLength-1)
    static uint32_t dlssEpochCount;   // completed Denoised epochs (= Welford N)
    static std::shared_ptr<vk::DeviceLocalImage> accumOutputImage;

    // Offline accumulation pipeline (shared between RT and DLSS modules)
    static VkPipeline accumPipeline;
    static VkPipelineLayout accumPipelineLayout;
    static std::shared_ptr<vk::DeviceLocalImage> accumBufferImage;
    static VkDescriptorPool accumDescPool;
    static VkDescriptorSetLayout accumDescSetLayout;
    static std::vector<VkDescriptorSet> accumDescSets;
    static bool accumPipelineReady;

    // Emission compose pipeline (DLSS-RR: subtract emission before, add after)
    static VkPipeline emissionComposePipeline;
    static VkPipelineLayout emissionComposePipelineLayout;
    static VkDescriptorPool emissionComposeDescPool;
    static VkDescriptorSetLayout emissionComposeDescSetLayout;
    static std::vector<VkDescriptorSet> emissionComposeDescSets;
    static bool emissionComposePipelineReady;

    static std::vector<std::shared_ptr<vk::DeviceLocalImage>> emissionImages;  // RT emission, read by tone mapping
    static std::vector<std::shared_ptr<vk::DeviceLocalImage>> renderResHdrImages;  // DLSS input (render-res HDR), read by tone mapping histogram
    static GpuProfiler gpuProfiler;
    static ThreadPool threadPool;
    static BlockModelTable blockModelTable;
    static TextureSystem textureSystem;

    // Frame Generation: images set by pipeline modules, read by render_framework for SL tagging
    static std::vector<std::shared_ptr<vk::DeviceLocalImage>> frameGenDepthImages;        // Linear depth (render res)
    static std::vector<std::shared_ptr<vk::DeviceLocalImage>> frameGenMotionVectorImages;  // Motion vectors (render res)

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
