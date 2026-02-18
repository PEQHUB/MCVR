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

struct DofModuleContext;

struct DofModulePushConstant {
    float focalDistance;  // world units to focal plane
    float aperture;       // circle-of-confusion scale
    float maxRadius;      // maximum blur radius in pixels
    float enabled;        // 0 = pass-through, 1 = active
};

class DofModule : public WorldModule, public SharedObject<DofModule> {
    friend DofModuleContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.dof.name";
    constexpr static uint32_t inputImageNum  = 2;  // [0] HDR colour, [1] linear depth (R32F)
    constexpr static uint32_t outputImageNum = 1;  // HDR colour output

    DofModule();

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
    // input[1]: linear depth
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> depthImages_;

    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables_;
    std::vector<std::shared_ptr<vk::Sampler>>          hdrSamplers_;
    std::vector<std::shared_ptr<vk::Sampler>>          depthSamplers_;

    std::shared_ptr<vk::Shader>          dofShader_;
    std::shared_ptr<vk::ComputePipeline> dofPipeline_;

    // output
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> outputImages_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;

    uint32_t width_  = 0;
    uint32_t height_ = 0;
};

struct DofModuleContext : public WorldModuleContext, SharedObject<DofModuleContext> {
    std::weak_ptr<DofModule> dofModule;

    std::shared_ptr<vk::DeviceLocalImage> hdrImage;
    std::shared_ptr<vk::DeviceLocalImage> depthImage;
    std::shared_ptr<vk::DeviceLocalImage> outputImage;
    std::shared_ptr<vk::DescriptorTable>  descriptorTable;

    DofModuleContext(std::shared_ptr<FrameworkContext>     frameworkContext,
                     std::shared_ptr<WorldPipelineContext>  worldPipelineContext,
                     std::shared_ptr<DofModule>             dofModule);

    void render() override;
};
