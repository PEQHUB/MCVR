#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/modules/world/dlss/dlss_wrapper.hpp"
#include "core/render/modules/world/world_module.hpp"

class Framework;
class FrameworkContext;
class WorldPipeline;
struct WorldModuleContext;

struct DLSSModuleContext;

class DLSSModule : public WorldModule, public SharedObject<DLSSModule> {
    friend DLSSModuleContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.dlss.name";
    constexpr static uint32_t inputImageNum = 10;
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

    // Output Scale 2x: DLSS targets 2x, Lanczos downscales to 1x output
    uint32_t dlssOutputWidth_ = 0;
    uint32_t dlssOutputHeight_ = 0;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> upscaled2xImages_;
    std::shared_ptr<vk::Shader> lanczosShader_;
    std::shared_ptr<vk::Sampler> lanczosSampler_;
    std::vector<std::shared_ptr<vk::DescriptorTable>> lanczosDescriptorTables_;
    std::shared_ptr<vk::ComputePipeline> lanczosPipeline_;
    void initLanczosResources();
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

    // output
    std::shared_ptr<vk::DeviceLocalImage> processedImage;          // DLSS writes here (2x when outputScale2x, else 1x)
    std::shared_ptr<vk::DeviceLocalImage> finalOutputImage;        // Lanczos writes here (shared 1x, only when outputScale2x)
    std::shared_ptr<vk::DeviceLocalImage> upscaledFirstHitDepthImage;

    DLSSModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                      std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                      std::shared_ptr<DLSSModule> dLSSModule);

    void render() override;
};