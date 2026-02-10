#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <array>

class Framework;
class FrameworkContext;
class RayTracingModule;
struct RayTracingModuleContext;

struct AtmosphereContext;

class Atmosphere : public SharedObject<Atmosphere> {
    friend RayTracingModule;
    friend RayTracingModuleContext;
    friend AtmosphereContext;

  public:
    Atmosphere();

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<RayTracingModule> rayTracingModule);

    void build();

  private:
    void initDescriptorTables();
    void initImages();
    void initAtmLUTRenderPass();
    void initAtmCubeMapRenderPass();
    void initFrameBuffers();
    void initAtmLUTPipeline();
    void initAtmCubeMapPipeline();

  private:
    bool lutRendered_ = false;

    std::weak_ptr<Framework> framework_;
    std::weak_ptr<RayTracingModule> rayTracingModule_;

    std::vector<std::shared_ptr<vk::DescriptorTable>> atmDescriptorTables_;

    std::shared_ptr<vk::Shader> atmLUTVertShader_;
    std::shared_ptr<vk::Shader> atmLUTFragShader_;
    std::shared_ptr<vk::DeviceLocalImage> atmLUTImage_;
    std::shared_ptr<vk::Sampler> atmLUTImageSampler_;
    std::shared_ptr<vk::RenderPass> atmLUTRenderPass_;
    std::shared_ptr<vk::Framebuffer> atmLUTFramebuffer_;
    std::shared_ptr<vk::GraphicsPipeline> atmLUTPipeline_;

    std::shared_ptr<vk::Shader> atmCubeMapVertShader_;
    std::shared_ptr<vk::Shader> atmCubeMapFragShader_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> atmCubeMapImages_;
    std::vector<std::shared_ptr<vk::Sampler>> atmCubeMapImageSamplers_;
    std::shared_ptr<vk::RenderPass> atmCubeMapRenderPass_;
    std::vector<std::array<std::shared_ptr<vk::Framebuffer>, 6>> atmCubeMapFramebuffers_;
    std::shared_ptr<vk::GraphicsPipeline> atmCubeMapPipeline_;

    std::vector<std::shared_ptr<AtmosphereContext>> contexts_;
};

struct AtmosphereContext : public SharedObject<AtmosphereContext> {
    std::weak_ptr<FrameworkContext> frameworkContext;
    std::weak_ptr<RayTracingModuleContext> rayTracingModuleContext;
    std::weak_ptr<Atmosphere> atmosphere;

    std::shared_ptr<vk::DescriptorTable> atmDescriptorTable;

    std::shared_ptr<vk::DeviceLocalImage> atmLUTImage;
    std::shared_ptr<vk::Framebuffer> atmLUTFramebuffer;

    std::shared_ptr<vk::DeviceLocalImage> atmCubeMapImage;
    std::array<std::shared_ptr<vk::Framebuffer>, 6> atmCubeMapFramebuffer;

    AtmosphereContext(std::shared_ptr<FrameworkContext> frameworkContext, std::shared_ptr<Atmosphere> atmosphere);

    void render();
};