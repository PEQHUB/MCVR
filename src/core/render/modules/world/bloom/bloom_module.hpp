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

struct BloomModuleContext;

struct BloomModulePushConstant {
    float threshold;  // luminance threshold for bright-pass
    float strength;   // bloom intensity multiplier
    float radius;     // blur radius in pixels
    float enabled;    // 0 = pass-through, 1 = active
};

class BloomModule : public WorldModule, public SharedObject<BloomModule> {
    friend BloomModuleContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.bloom.name";
    constexpr static uint32_t inputImageNum  = 1;  // HDR colour input
    constexpr static uint32_t outputImageNum = 1;  // HDR colour output (with bloom added)

    BloomModule();

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
    // input
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> inputImages_;

    // internal
    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables_;
    std::vector<std::shared_ptr<vk::Sampler>>          samplers_;

    std::shared_ptr<vk::Shader>          bloomShader_;
    std::shared_ptr<vk::ComputePipeline> bloomPipeline_;

    // output (same image — bloom is added in-place into an output buffer)
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> outputImages_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;

    uint32_t width_  = 0;
    uint32_t height_ = 0;
};

struct BloomModuleContext : public WorldModuleContext, SharedObject<BloomModuleContext> {
    std::weak_ptr<BloomModule> bloomModule;

    std::shared_ptr<vk::DeviceLocalImage>  inputImage;
    std::shared_ptr<vk::DeviceLocalImage>  outputImage;
    std::shared_ptr<vk::DescriptorTable>   descriptorTable;

    BloomModuleContext(std::shared_ptr<FrameworkContext>      frameworkContext,
                       std::shared_ptr<WorldPipelineContext>  worldPipelineContext,
                       std::shared_ptr<BloomModule>           bloomModule);

    void render() override;
};
