#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/modules/world/world_module.hpp"

class Framework;
class FrameworkContext;
class WorldPipeline;
struct WorldModuleContext;

struct MotionBlurModuleContext;

struct MotionBlurModulePushConstant {
    float strength;  // velocity scale multiplier
    float samples;   // sample count along velocity vector (4–16)
    float enabled;   // 0 = pass-through, 1 = active
};

class MotionBlurModule : public WorldModule, public SharedObject<MotionBlurModule> {
    friend MotionBlurModuleContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.motion_blur.name";
    constexpr static uint32_t inputImageNum  = 2;  // [0] HDR colour, [1] motion vectors (RG16F)
    constexpr static uint32_t outputImageNum = 1;  // HDR colour output

    MotionBlurModule();

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

    void bindTexture(std::shared_ptr<vk::Sampler> sampler,
                     std::shared_ptr<vk::DeviceLocalImage> image,
                     int index) override;

    void preClose() override;

  private:
    void initDescriptorTables();
    void initImages();
    void initPipeline();

  private:
    // input[0]: HDR colour
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> hdrImages_;
    // input[1]: motion vectors
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> motionVectorImages_;

    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables_;
    std::vector<std::shared_ptr<vk::Sampler>>          hdrSamplers_;
    std::vector<std::shared_ptr<vk::Sampler>>          mvSamplers_;

    std::shared_ptr<vk::Shader>          motionBlurShader_;
    std::shared_ptr<vk::ComputePipeline> motionBlurPipeline_;

    // output
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> outputImages_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;

    uint32_t width_  = 0;
    uint32_t height_ = 0;
};

struct MotionBlurModuleContext : public WorldModuleContext, SharedObject<MotionBlurModuleContext> {
    std::weak_ptr<MotionBlurModule> motionBlurModule;

    std::shared_ptr<vk::DeviceLocalImage> hdrImage;
    std::shared_ptr<vk::DeviceLocalImage> motionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> outputImage;
    std::shared_ptr<vk::DescriptorTable>  descriptorTable;

    MotionBlurModuleContext(std::shared_ptr<FrameworkContext>     frameworkContext,
                            std::shared_ptr<WorldPipelineContext>  worldPipelineContext,
                            std::shared_ptr<MotionBlurModule>      motionBlurModule);

    void render() override;
};
