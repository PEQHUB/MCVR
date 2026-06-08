#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/modules/world/dlss/dlss_wrapper.hpp"
#include "core/render/modules/world/world_module.hpp"

#include <chrono>

class Framework;
class FrameworkContext;
class WorldPipeline;
struct WorldModuleContext;

struct DLSSModuleContext;

class DLSSModule : public WorldModule, public SharedObject<DLSSModule> {
    friend DLSSModuleContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.dlss.name";
    constexpr static uint32_t inputImageNum = 24;
    constexpr static uint32_t outputImageNum = 2;

    static bool initNGXContext();
    static void deinitNGXContext();

    DLSSModule();

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
    static std::shared_ptr<NgxContext> ngxContext_;

    // input
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> hdrImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> diffuseAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> normalRoughnessImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> motionVectorImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> linearDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularHitDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> diffuseRayDirHitDistImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularRayDirHitDistImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> reflectionMvImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> animatedTexMaskImages_;
    // Extended optional guide buffers
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> particleMaskImages_;      // [12] pInIsParticleMask
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitBaseEmissionImages_; // [13] GBuffer emissive
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> biasMaskImages_;           // [14] pInBiasCurrentColorMask
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> rtHitDistImages_;          // [15] pInRayTracingHitDistance
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> motionVectors3DImages_;    // [16] pInMotionVectors3D
    // GBuffer + additional inputs
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> gbufferMetallicImages_;       // [17] GBuffer metallic
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> gbufferShadingModelIdImages_; // [18] GBuffer shading model ID
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> gbufferMaterialIdImages_;     // [19] GBuffer material ID
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> positionViewSpaceImages_;     // [20] view-space hit position
    // DLSS-RR transparency layer (stable planes)
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> transparencyLayerImages_;         // [21] glass foreground color
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> transparencyLayerOpacityImages_;  // [22] glass per-channel opacity
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> transparencyLayerMvecsImages_;    // [23] glass surface MVs

    // dlss
    std::shared_ptr<DlssRR> dlss_;
    NgxContext::SupportedSizes supportedSizes_{};
    NVSDK_NGX_PerfQuality_Value mode_ = NVSDK_NGX_PerfQuality_Value_Balanced;

    // output
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> processedImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> upscaledFirstHitDepthImages_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;

    uint32_t inputWidth_, inputHeight_;
    uint32_t outputWidth_, outputHeight_;

    // Output Scale 2x: DLSS targets 2x, FSR1 EASU downscales to 1x output
    uint32_t dlssOutputWidth_ = 0;
    uint32_t dlssOutputHeight_ = 0;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> upscaled2xImages_;
    std::shared_ptr<vk::Shader> easuShader_;
    std::shared_ptr<vk::Sampler> easuSampler_;
    std::vector<std::shared_ptr<vk::DescriptorTable>> easuDescriptorTables_;
    std::shared_ptr<vk::ComputePipeline> easuPipeline_;
    void initEasuResources();
};

struct DLSSModuleContext : public WorldModuleContext, SharedObject<DLSSModuleContext> {
    std::weak_ptr<DLSSModule> dLSSModule;

    // input
    std::shared_ptr<vk::DeviceLocalImage> hdrImage;
    std::shared_ptr<vk::DeviceLocalImage> diffuseAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> specularAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> normalRoughnessImage;
    std::shared_ptr<vk::DeviceLocalImage> motionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> linearDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> specularHitDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> diffuseRayDirHitDistImage;
    std::shared_ptr<vk::DeviceLocalImage> specularRayDirHitDistImage;
    std::shared_ptr<vk::DeviceLocalImage> reflectionMvImage;
    std::shared_ptr<vk::DeviceLocalImage> animatedTexMaskImage;
    // Extended optional guide buffer images
    std::shared_ptr<vk::DeviceLocalImage> particleMaskImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitBaseEmissionImage;
    std::shared_ptr<vk::DeviceLocalImage> biasMaskImage;
    std::shared_ptr<vk::DeviceLocalImage> rtHitDistImage;
    std::shared_ptr<vk::DeviceLocalImage> motionVectors3DImage;
    // GBuffer + additional input images
    std::shared_ptr<vk::DeviceLocalImage> gbufferMetallicImage;
    std::shared_ptr<vk::DeviceLocalImage> gbufferShadingModelIdImage;
    std::shared_ptr<vk::DeviceLocalImage> gbufferMaterialIdImage;
    std::shared_ptr<vk::DeviceLocalImage> positionViewSpaceImage;
    // Transparency layer
    std::shared_ptr<vk::DeviceLocalImage> transparencyLayerImage;
    std::shared_ptr<vk::DeviceLocalImage> transparencyLayerOpacityImage;
    std::shared_ptr<vk::DeviceLocalImage> transparencyLayerMvecsImage;

    // output
    std::shared_ptr<vk::DeviceLocalImage> processedImage;          // DLSS writes here (2x when outputScale2x, else 1x)
    std::shared_ptr<vk::DeviceLocalImage> finalOutputImage;        // EASU writes here (shared 1x, only when outputScale2x)
    std::shared_ptr<vk::DeviceLocalImage> upscaledFirstHitDepthImage;

    // Per-context frame timer for InFrameTimeDeltaInMsec
    std::chrono::steady_clock::time_point lastRenderTime_ = std::chrono::steady_clock::now();

    DLSSModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                      std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                      std::shared_ptr<DLSSModule> dLSSModule);

    void render() override;
};
