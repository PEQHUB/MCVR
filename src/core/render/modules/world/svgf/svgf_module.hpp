#pragma once

#include "core/render/modules/world/world_module.hpp"
#include "core/render/renderer.hpp"
#include "svgf_denoiser.hpp"
#include <memory>
#include <vector>

struct SvgfModuleContext;

class SvgfModule : public WorldModule, public SharedObject<SvgfModule> {
  public:
    static constexpr auto NAME = "SVGF";
    static constexpr uint32_t inputImageNum = 10;
    static constexpr uint32_t outputImageNum = 1;

    SvgfModule();
    virtual ~SvgfModule() = default;

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline);
    bool setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                std::vector<VkFormat> &formats,
                                uint32_t frameIndex) override;
    bool setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                 std::vector<VkFormat> &formats,
                                 uint32_t frameIndex) override;
    void build() override;
    void setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) override;
    std::vector<std::shared_ptr<WorldModuleContext>> &contexts() override;
    void
    bindTexture(std::shared_ptr<vk::Sampler> sampler, std::shared_ptr<vk::DeviceLocalImage> image, int index) override;
    void preClose() override;

    std::shared_ptr<SvgfDenoiser> denoiser() {
        return m_denoiser;
    }

  private:
    friend class SvgfModuleContext;

    uint32_t width_ = 0;
    uint32_t height_ = 0;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;
    std::shared_ptr<SvgfDenoiser> m_denoiser;

    // Image Resources
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> diffuseRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> directRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> diffuseAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> normalRoughnessImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> motionVectorImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> linearDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> clearRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> baseEmissionImages_;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> denoisedRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> denoisedDiffuseRadianceImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> denoisedSpecularRadianceImages_;
};

class SvgfModuleContext : public WorldModuleContext, public SharedObject<SvgfModuleContext> {
  public:
    SvgfModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                      std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                      std::shared_ptr<SvgfModule> svgfModule);

    void render() override;

  private:
    std::weak_ptr<SvgfModule> svgfModule;

    // Inputs
    std::shared_ptr<vk::DeviceLocalImage> diffuseRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> specularRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> directRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> diffuseAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> specularAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> normalRoughnessImage;
    std::shared_ptr<vk::DeviceLocalImage> motionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> linearDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> clearRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> baseEmissionImage;

    // Outputs
    std::shared_ptr<vk::DeviceLocalImage> denoisedRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> denoisedDiffuseRadianceImage;
    std::shared_ptr<vk::DeviceLocalImage> denoisedSpecularRadianceImage;
};