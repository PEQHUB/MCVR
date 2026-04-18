#pragma once

// V2 Cloud Adapter
//
// Ports the V1 CloudModule volumetric cloud system into the V2 adapter/render-graph
// pattern.  The adapter owns 6 compute pipelines:
//   1. noise_gen   — one-shot 3D noise generation (first frame only)
//   2. weather     — 512x512 RGBA8 weather-map update
//   3. raymarch    — low-res (cloud-res) cloud ray march
//   4. temporal    — temporal reprojection / accumulation at cloud-res
//   5. composite   — upscale + blend clouds onto full-res radiance
//   6. shadow      — project cloud shadow map
//
// Graph integration:
//   registerPass() is called AFTER the DLSS/denoiser pass but BEFORE tone mapping.
//   It takes the upscaled/denoised radiance handle and the linear-depth handle from RT,
//   declares an output resource (cloud_radiance) and registers a pass that calls
//   execute() each frame.
//
// When cloudQuality == 0, the adapter is still registered but execute() is a near
// no-op: it blits the radiance input to the output with a pipeline image-copy barrier.
//
// Invariant: Adapters never read global state. All inputs come from EngineServices.
// NOTE: FeatureAdapter::init() returns void; cloud-specific init failure is signalled
//       via isInitialized().

#include "feature_adapter.hpp"
#include "rendergraph/graph_builder.hpp"
#include "platform/vulkan/vk2_shader.hpp"
#include "platform/vulkan/vk2_pipeline.hpp"
#include "platform/vulkan/vk2_descriptor.hpp"
#include "platform/vulkan/vk2_image.hpp"

#include <array>
#include <chrono>
#include <cstdint>

namespace engine {

// ---------------------------------------------------------------------------
// Push constant — must match cloud_common.glsl layout exactly.
// Identical to CloudPushConstant in cloud_module.hpp (104 bytes, alignas(16)).
// ---------------------------------------------------------------------------

struct CloudPushConstant {
    uint32_t renderWidth;        // Full render resolution width
    uint32_t renderHeight;       // Full render resolution height
    uint32_t cloudWidth;         // Cloud render resolution width  (renderWidth / divisor)
    uint32_t cloudHeight;        // Cloud render resolution height (renderHeight / divisor)
    float cloudBase;             // Cloud layer base altitude in blocks
    float cloudThickness;        // Cloud layer thickness in blocks
    float coverage;              // Base cloud coverage [0-1]
    float cloudType;             // Cloud type [0-1]: 0=stratus, 0.33=sc, 0.67=cu, 1.0=cb
    float densityMultiplier;     // Density scale
    float windTime;              // Accumulated wind time (seconds), wrapped at 86 400 s
    uint32_t frameIndex;         // Monotonic counter (temporal jitter + blue-noise index)
    uint32_t marchSteps;         // Ray-march step count
    uint32_t lightSteps;         // Light-march step count
    float temporalBlend;         // History blend factor (0.90–0.97)
    uint32_t shadowMapSize;      // Cloud shadow-map resolution (pixels, square)
    float eyePosX;               // Camera world position X
    float eyePosY;               // Camera world position Y
    float eyePosZ;               // Camera world position Z
    float detailStrength;        // Detail-erosion multiplier [0-3]
    uint32_t scatterOctaves;     // Multi-scatter octave count [1-8]
    float ambientStrength;       // Density-based ambient occlusion [0-3]
    float noiseScale;            // Noise texture period in blocks [16-4096]
    float cellFrequency;         // Voronoi cell count across weather map [1-32]
    float atmosphereFadeDist;    // Cloud atmospheric fade distance in blocks
    float windAngle;             // Wind direction in radians
    uint32_t debugMode;          // 0=normal, 1–8=debug visualisations
    // --- 104 bytes total ---
};

static_assert(sizeof(CloudPushConstant) == 104,
              "CloudPushConstant must be exactly 104 bytes to match cloud_common.glsl");

// ---------------------------------------------------------------------------
// CloudAdapter
// ---------------------------------------------------------------------------

class CloudAdapter : public FeatureAdapter {
public:
    CloudAdapter()  = default;
    ~CloudAdapter() = default;

    CloudAdapter(const CloudAdapter&) = delete;
    CloudAdapter& operator=(const CloudAdapter&) = delete;

    // FeatureAdapter interface ---------------------------------------------------

    // Load shaders, create pipelines, create internal images.
    // On failure (shaders not found, device error) logs a warning and returns;
    // isInitialized() will remain false and execute() will be a safe no-op.
    void init(EngineServices& services) override;

    // Read cloud-quality config and adjust internal quality parameters.
    // Handles quality changes that require internal image recreation.
    void configure(const ConfigSnapshot& config) override;

    // Execute the full cloud pipeline for one frame.
    // If cloudQuality == 0: copies radiance input → output unchanged (pass-through).
    void execute(const FrameContext& ctx, const PassResources& resources) override;

    // Release all Vulkan resources.
    void shutdown() override;

    const char* name() const override { return "Cloud"; }

    // Graph integration ---------------------------------------------------------

    // Register the cloud pass in the render graph.
    // radianceHandle  — output of DLSS-RR or temporal denoiser
    // depthHandle     — linear depth from RayTracingAdapter
    // Returns the cloud_radiance handle that replaces radianceHandle downstream
    // (tone mapping, post-process, etc. should use the returned handle instead).
    ResourceHandle registerPass(GraphBuilder& builder,
                                ResourceHandle radianceHandle,
                                ResourceHandle depthHandle);

    // Post-registration accessors (valid after registerPass()).
    ResourceHandle cloudRadianceHandle() const { return cloudRadianceHandle_; }
    ResourceHandle cloudShadowHandle()   const { return cloudShadowHandle_; }

    bool isInitialized() const { return initialized_; }

private:
    // Internal helpers ----------------------------------------------------------

    // (Re-)create the resolution-dependent images (cloudColor per-frame, history).
    // Called from execute() on first frame and when render resolution changes.
    void createInternalImages(uint32_t renderW, uint32_t renderH);
    void destroyInternalImages();

    // Allocate and write descriptor sets 0 (images) and 1 (UBOs) for one frame.
    VkDescriptorSet allocAndWriteDescSet0(uint32_t frameIndex,
                                          VkImageView radianceView,
                                          VkImageView depthView,
                                          VkImageView cloudRadianceView);
    VkDescriptorSet allocAndWriteDescSet1(uint32_t frameIndex);

    // Map cloudQuality preset to per-frame parameters.
    void applyQualityPreset(uint32_t quality);

    // Full-pipeline image-memory barrier helper (mirrors atmosphere_adapter.cpp).
    static void imageBarrier(VkCommandBuffer cmd, VkImage image,
                             VkImageLayout oldLayout, VkImageLayout newLayout,
                             VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                             VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                             uint32_t layerCount = 1, uint32_t depth = 1);

    // ---------------------------------------------------------------------------
    // Quality/config parameters (refreshed by configure())
    // ---------------------------------------------------------------------------

    uint32_t cloudQuality_       = 0;       // 0 = off
    uint32_t resolutionDivisor_  = 2;       // 1/2/4
    uint32_t marchSteps_         = 64;
    uint32_t lightSteps_         = 4;
    float    temporalBlend_      = 0.95f;
    uint32_t shadowMapSize_      = 256;

    float cloudBase_             = 192.0f;
    float cloudThickness_        = 64.0f;
    float coverage_              = 0.35f;
    float cloudType_             = 0.67f;
    float densityMultiplier_     = 1.0f;
    float windSpeed_             = 1.0f;
    float detailStrength_        = 1.0f;
    uint32_t scatterOctaves_     = 3;
    float ambientStrength_       = 1.0f;
    float noiseScale_            = 256.0f;
    float cellFrequency_         = 4.0f;
    float atmosphereFadeDist_    = 1000.0f;
    float windAngle_             = 0.0f;
    uint32_t debugMode_          = 0;

    // ---------------------------------------------------------------------------
    // Render / cloud resolution (set on first execute() or configure())
    // ---------------------------------------------------------------------------

    uint32_t renderW_ = 0, renderH_ = 0;
    uint32_t cloudW_ = 0, cloudH_ = 0;

    // ---------------------------------------------------------------------------
    // Services (non-owning, set during init())
    // ---------------------------------------------------------------------------

    EngineServices* services_ = nullptr;
    VkDevice        device_   = VK_NULL_HANDLE;

    // ---------------------------------------------------------------------------
    // Shader modules (6 compute shaders)
    // ---------------------------------------------------------------------------

    vk2::ShaderModule noiseGenShader_;
    vk2::ShaderModule weatherShader_;
    vk2::ShaderModule raymarchShader_;
    vk2::ShaderModule temporalShader_;
    vk2::ShaderModule compositeShader_;
    vk2::ShaderModule shadowShader_;

    // ---------------------------------------------------------------------------
    // Descriptor layouts (two sets per frame)
    //
    //  set 0 — images (10 bindings, per-frame)
    //    b0  storage image    — radiance input  (read by composite)
    //    b1  storage image    — linear depth    (read by raymarch / composite)
    //    b2  storage image    — cloud radiance output
    //    b3  storage image    — cloudColor (cloud-res HDR, written by raymarch/temporal)
    //    b4  storage image    — cloudHistory (previous frame cloud, read by temporal)
    //    b5  storage image    — weatherMap  (written by weather pass)
    //    b6  CIS              — noiseTexture3D (sampled with repeat sampler)
    //    b7  storage image    — cloudShadowMap output
    //    b8  CIS              — weatherMap (sampled by raymarch)
    //    b9  storage image    — noiseTexture3D (written by noise_gen)
    //
    //  set 1 — UBOs (3 bindings, per-frame)
    //    b0  uniform buffer   — WorldUBO
    //    b1  uniform buffer   — SkyUBO
    //    b2  uniform buffer   — LastWorldUBO (previous frame camera)
    // ---------------------------------------------------------------------------

    VkDescriptorSetLayout descLayout0_ = VK_NULL_HANDLE;  // images
    VkDescriptorSetLayout descLayout1_ = VK_NULL_HANDLE;  // UBOs

    vk2::DescriptorAllocator descAllocator0_;  // one set per frame (images)
    vk2::DescriptorAllocator descAllocator1_;  // one set per frame (UBOs)

    // ---------------------------------------------------------------------------
    // Pipeline layout (shared by all 6 passes)
    // ---------------------------------------------------------------------------

    vk2::PipelineLayout pipelineLayout_;

    // ---------------------------------------------------------------------------
    // Compute pipelines
    // ---------------------------------------------------------------------------

    vk2::ComputePipeline noiseGenPipeline_;
    vk2::ComputePipeline weatherPipeline_;
    vk2::ComputePipeline raymarchPipeline_;
    vk2::ComputePipeline temporalPipeline_;
    vk2::ComputePipeline compositePipeline_;
    vk2::ComputePipeline shadowPipeline_;

    // ---------------------------------------------------------------------------
    // Internal images (NOT in the render graph resource pool — owned here)
    // ---------------------------------------------------------------------------

    // Per-frame cloud color (cloud-res RGBA16F, written by raymarch, read by temporal)
    std::vector<vk2::Image> cloudColorImages_;   // size = framesInFlight

    // Single shared history image (RGBA16F cloud-res, copied from cloudColor each frame)
    vk2::Image cloudHistoryImage_;

    // Single shared weather map (RGBA8 512×512)
    vk2::Image weatherMapImage_;

    // Single shared 3D noise texture (RGBA8 128³)
    static constexpr uint32_t NOISE_RES = 128;
    vk2::Image noiseTexture3D_;

    // Single shared shadow map (R16F, shadowMapSize × shadowMapSize)
    vk2::Image cloudShadowImage_;

    // Samplers
    VkSampler noiseSampler_   = VK_NULL_HANDLE;   // REPEAT linear
    VkSampler weatherSampler_ = VK_NULL_HANDLE;   // CLAMP_TO_EDGE linear

    // Current layout tracking for internal images (not managed by graph pool)
    VkImageLayout noiseLayout_   = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout weatherLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout historyLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout shadowLayout_  = VK_IMAGE_LAYOUT_UNDEFINED;

    // ---------------------------------------------------------------------------
    // Graph handles (set by registerPass())
    // ---------------------------------------------------------------------------

    ResourceHandle inputRadianceHandle_;
    ResourceHandle inputDepthHandle_;
    ResourceHandle cloudRadianceHandle_;
    ResourceHandle cloudShadowHandle_;

    // ---------------------------------------------------------------------------
    // Frame state
    // ---------------------------------------------------------------------------

    uint32_t frameCounter_ = 0;
    float    windTime_     = 0.0f;
    std::chrono::steady_clock::time_point lastFrameTime_{};
    bool noiseGenerated_  = false;
    bool initialized_     = false;
    bool imagesReady_     = false;
};

} // namespace engine
