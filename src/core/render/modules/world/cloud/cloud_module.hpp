#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/modules/world/world_module.hpp"

#include <chrono>

class Framework;
class FrameworkContext;
class WorldPipeline;
struct WorldModuleContext;

struct CloudModuleContext;

// Push constant shared by all cloud compute passes.
// Must match PushConstants layout in cloud_common.glsl exactly.
// 104 bytes of data (within Vulkan guaranteed minimum of 128 bytes).
struct alignas(16) CloudPushConstant {
    uint32_t renderWidth;        // Render resolution width
    uint32_t renderHeight;       // Render resolution height
    uint32_t cloudWidth;         // Cloud render resolution width
    uint32_t cloudHeight;        // Cloud render resolution height
    float cloudBase;             // Cloud layer base altitude (blocks)
    float cloudThickness;        // Cloud layer thickness (blocks)
    float coverage;              // Base coverage [0-1]
    float cloudType;             // [0-1] maps to 4 types: 0=stratus, 0.33=sc, 0.67=cu, 1.0=cb
    float densityMultiplier;     // Density scale
    float windTime;              // Accumulated wind time (seconds), wrapped at 86400s
    uint32_t frameIndex;         // For temporal jitter + blue noise
    uint32_t marchSteps;         // Ray march step count
    uint32_t lightSteps;         // Light march step count
    float temporalBlend;         // History blend factor (0.93-0.97)
    uint32_t shadowMapSize;      // Cloud shadow map resolution
    float eyePosX;               // Camera world position X
    float eyePosY;               // Camera world position Y
    float eyePosZ;               // Camera world position Z
    float detailStrength;        // Detail erosion multiplier [0-2]
    uint32_t scatterOctaves;     // Multi-scatter octave count [1-4] [Wrenninge13]
    float ambientStrength;       // Density-based ambient occlusion [0-2]
    float noiseScale;            // Noise texture period in blocks [128-512] — smaller = more detail
    float cellFrequency;         // Voronoi cell count across weather map [2-16]
    float atmosphereFadeDist;    // Cloud atmospheric fade distance in blocks [200-2000]
    float windAngle;             // Wind direction in radians [0, 2π]
    uint32_t debugMode;          // 0=normal, 1=weather coverage, 2=weather type, 3-5=noise R/G/A, 6=heightProfile, 7=raw density, 8=final density
};

class CloudModule : public WorldModule, public SharedObject<CloudModule> {
    friend CloudModuleContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.cloud.name";
    constexpr static uint32_t inputImageNum = 2;   // radiance, linear_depth
    constexpr static uint32_t outputImageNum = 2;   // cloud_radiance (composited), cloud_shadow_map

    CloudModule();

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline);

    bool setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                std::vector<VkFormat> &formats,
                                uint32_t frameIndex) override;
    bool setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                 std::vector<VkFormat> &formats,
                                 uint32_t frameIndex) override;

    void setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) override;

    void build() override;

    std::vector<std::shared_ptr<WorldModuleContext>> &contexts() override;

    void
    bindTexture(std::shared_ptr<vk::Sampler> sampler, std::shared_ptr<vk::DeviceLocalImage> image, int index) override;

    void preClose() override;

  private:
    void initDescriptorTables();
    void initImages();
    void initPipeline();

  private:
    // Input images (per swapchain frame)
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> radianceImages_;        // Input 0: HDR radiance
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> linearDepthImages_;     // Input 1: linear depth

    // Internal images
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> cloudColorImages_;      // RGBA16F: cloud color + transmittance (cloud res)
    std::shared_ptr<vk::DeviceLocalImage> cloudHistoryImage_;                  // RGBA16F: previous frame cloud (temporal)
    std::shared_ptr<vk::DeviceLocalImage> weatherMapImage_;                    // RGBA8: weather map (256x256)
    std::shared_ptr<vk::DeviceLocalImage> noiseTexture3D_;                     // RGBA8: 128^3 tiled noise
    std::shared_ptr<vk::DeviceLocalImage> cloudShadowImage_;                   // R16F: cloud shadow map
    std::shared_ptr<vk::Sampler> noiseSampler_;                                // Repeat sampler for 3D noise
    std::shared_ptr<vk::Sampler> weatherSampler_;                               // Repeat sampler for weather map

    // Output images (per frame)
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> cloudRadianceImages_;   // Output 0: cloud-composited radiance
    // Output 1 (cloud_shadow_map) uses cloudShadowImage_ (shared, not per-frame)

    // Descriptor tables (per frame)
    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables_;

    // Compute pipelines (6 passes: noise gen [init] + weather + raymarch + temporal + composite + shadow)
    std::shared_ptr<vk::Shader> noiseGenShader_;
    std::shared_ptr<vk::ComputePipeline> noiseGenPipeline_;

    std::shared_ptr<vk::Shader> weatherShader_;
    std::shared_ptr<vk::ComputePipeline> weatherPipeline_;

    std::shared_ptr<vk::Shader> raymarchShader_;
    std::shared_ptr<vk::ComputePipeline> raymarchPipeline_;

    std::shared_ptr<vk::Shader> temporalShader_;
    std::shared_ptr<vk::ComputePipeline> temporalPipeline_;

    std::shared_ptr<vk::Shader> compositeShader_;
    std::shared_ptr<vk::ComputePipeline> compositePipeline_;

    std::shared_ptr<vk::Shader> shadowShader_;
    std::shared_ptr<vk::ComputePipeline> shadowPipeline_;

    // Module contexts
    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;

    // Render resolution (set from output images)
    uint32_t width_ = 0, height_ = 0;

    // Cloud render resolution (quarter/half/full based on quality)
    uint32_t cloudWidth_ = 0, cloudHeight_ = 0;
    uint32_t resolutionDivisor_ = 2;  // 1=full, 2=half, 4=quarter

    // Configurable attributes (from YAML / Java Options)
    float cloudBase_ = 192.0f;
    float cloudThickness_ = 64.0f;
    float coverage_ = 0.35f;
    float cloudType_ = 0.67f;   // Default: cumulus (0=stratus, 0.33=sc, 0.67=cu, 1.0=cb)
    float densityMultiplier_ = 1.0f;
    float windSpeed_ = 1.0f;
    float detailStrength_ = 1.0f;
    uint32_t scatterOctaves_ = 3;
    uint32_t marchSteps_ = 64;
    uint32_t lightSteps_ = 4;
    float temporalBlend_ = 0.95f;
    uint32_t shadowMapSize_ = 256;
    bool enabled_ = true;

    uint32_t frameCounter_ = 0;
    float windTime_ = 0.0f;
    std::chrono::steady_clock::time_point lastFrameTime_{};
    bool noiseGenerated_ = false;
};

struct CloudModuleContext : public WorldModuleContext, SharedObject<CloudModuleContext> {
    std::weak_ptr<CloudModule> cloudModule;

    // Input references
    std::shared_ptr<vk::DeviceLocalImage> radianceImage;
    std::shared_ptr<vk::DeviceLocalImage> linearDepthImage;

    // Internal textures
    std::shared_ptr<vk::DeviceLocalImage> cloudColorImage;
    std::shared_ptr<vk::DeviceLocalImage> cloudHistoryImage;
    std::shared_ptr<vk::DeviceLocalImage> weatherMapImage;
    std::shared_ptr<vk::DeviceLocalImage> noiseTexture3D;
    std::shared_ptr<vk::DeviceLocalImage> cloudShadowImage;

    // Output references
    std::shared_ptr<vk::DeviceLocalImage> cloudRadianceImage;

    // Descriptor table for this frame
    std::shared_ptr<vk::DescriptorTable> descriptorTable;

    CloudModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                       std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                       std::shared_ptr<CloudModule> cloudModule);

    void render() override;
};
