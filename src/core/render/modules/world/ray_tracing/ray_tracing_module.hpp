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

struct RayTracingModuleContext;

class Atmosphere;
class AtmosphereContext;
class WorldPrepare;
class WorldPrepareContext;

struct RayTracingPushConstant {
    int numRayBounces;
    int flags;           // bit 0: simplified indirect, bit 1: area lights enabled
                         // bit 2: restir, bit 3: simplified BRDF, bit 4: restir bounce
    int areaLightCount;  // number of active area lights this frame
    float shadowSoftness;
    int risCandidates;   // total RIS candidates per pixel
    int temporalMClamp;  // temporal reservoir M clamp (used as float in shader)
    int wClamp;          // importance weight W clamp (used as float in shader)
    float preExposure;   // pre-exposure multiplier for DLSS-RR normalization
};

class RayTracingModule : public WorldModule, public SharedObject<RayTracingModule> {
    friend RayTracingModuleContext;
    friend Atmosphere;
    friend AtmosphereContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.ray_tracing.name";
    constexpr static uint32_t inputImageNum = 0;
    constexpr static uint32_t outputImageNum = 16;

    RayTracingModule();

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
    void initSBT();
    void initSpatialPipeline();
    void initClusterPipeline();

  private:
    // input
    // none

    // ray tracing
    std::shared_ptr<vk::Shader> worldRayGenShader_;

    std::shared_ptr<vk::Shader> worldRayMissShader_;
    std::shared_ptr<vk::Shader> handRayMissShader_;
    std::shared_ptr<vk::Shader> shadowRayMissShader_;
    std::shared_ptr<vk::Shader> pointLightShadowMissShader_;

    std::shared_ptr<vk::Shader> shadowRayClosestHitShader_;
    std::shared_ptr<vk::Shader> shadowAnyHitShader_;

    std::shared_ptr<vk::Shader> worldSolidTransparentClosestHitShader_;
    std::shared_ptr<vk::Shader> worldTransparentAnyHitShader_;

    std::shared_ptr<vk::Shader> worldNoReflectClosestHitShader_;
    std::shared_ptr<vk::Shader> worldNoReflectAnyHitShader_;

    std::shared_ptr<vk::Shader> worldCloudClosestHitShader_;
    std::shared_ptr<vk::Shader> worldCloudAnyHitShader_;

    std::shared_ptr<vk::Shader> boatWaterMaskClosestHitShader_;
    std::shared_ptr<vk::Shader> boatWaterMaskAnyHitShader_;

    std::shared_ptr<vk::Shader> endPortalClosestHitShader_;
    std::shared_ptr<vk::Shader> endPortalAnyHitShader_;

    std::shared_ptr<vk::Shader> endGatewayClosestHitShader_;
    std::shared_ptr<vk::Shader> endGatewayAnyHitShader_;

    std::shared_ptr<vk::Shader> worldPostColorToDepthVertShader_;
    std::shared_ptr<vk::Shader> worldPostColorToDepthFragShader_;
    std::shared_ptr<vk::Shader> worldPostVertShader_;
    std::shared_ptr<vk::Shader> worldPostFragShader_;
    std::shared_ptr<vk::Shader> worldToneMappingVertShader_;
    std::shared_ptr<vk::Shader> worldToneMappingFragShader_;
    std::shared_ptr<vk::Shader> radianceHistCompShader_;
    std::shared_ptr<vk::Shader> worldLightMapVertShader_;
    std::shared_ptr<vk::Shader> worldLightMapFragShader_;

    std::vector<std::shared_ptr<vk::DescriptorTable>> rayTracingDescriptorTables_;
    std::shared_ptr<vk::RayTracingPipeline> rayTracingPipeline_;
    std::vector<std::shared_ptr<vk::SBT>> sbts_;

    uint32_t numRayBounces_ = 4;
    bool useJitter_ = true;

    // output
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> hdrNoisyOutputImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> diffuseAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularAlbedoImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> normalRoughnessImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> motionVectorImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> linearDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularHitDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitDiffuseDirectLightImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitDiffuseIndirectLightImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitSpecularImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitClearImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> firstHitBaseEmissionImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> directLightDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> diffuseRayDirHitDistImages_;   // DLSS-RR guide: xyz=dir, w=hitDist
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> specularRayDirHitDistImages_;  // DLSS-RR guide: xyz=dir, w=hitDist

    // ReSTIR DI reservoir images (fixed roles)
    // [0] = temporal output (CHS writes), [1] = spatial output (compute writes)
    std::shared_ptr<vk::DeviceLocalImage> reservoirImages_[2];

    // Bounce ReSTIR DI reservoir images (per-bounce temporal reuse)
    // [0] = bounce 1, [1] = bounce 2, [2] = bounce 3
    std::shared_ptr<vk::DeviceLocalImage> bounceReservoirImages_[3];

    // Spatial reuse compute pipeline
    VkPipeline spatialPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout spatialPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout spatialDescSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool spatialDescPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> spatialDescSets_;
    std::shared_ptr<vk::Shader> spatialShader_;

    // Light clustering compute pipeline
    VkPipeline clusterPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout clusterPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout clusterDescSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool clusterDescPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> clusterDescSets_;
    std::shared_ptr<vk::Shader> clusterShader_;
    std::shared_ptr<vk::DeviceLocalBuffer> tileLightBuffer_;
    static constexpr int TILE_SIZE = 16;
    static constexpr int MAX_LIGHTS_PER_TILE = 512;

    // submodules
    std::shared_ptr<Atmosphere> atmosphere_;
    std::shared_ptr<WorldPrepare> worldPrepare_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;
};

struct RayTracingModuleContext : public WorldModuleContext, SharedObject<RayTracingModuleContext> {
    std::weak_ptr<RayTracingModule> rayTracingModule;

    // input
    // none

    // ray tracing
    std::shared_ptr<vk::DescriptorTable> rayTracingDescriptorTable;
    std::shared_ptr<vk::SBT> sbt;

    // output
    std::shared_ptr<vk::DeviceLocalImage> hdrNoisyOutputImage;
    std::shared_ptr<vk::DeviceLocalImage> diffuseAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> specularAlbedoImage;
    std::shared_ptr<vk::DeviceLocalImage> normalRoughnessImage;
    std::shared_ptr<vk::DeviceLocalImage> motionVectorImage;
    std::shared_ptr<vk::DeviceLocalImage> linearDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> specularHitDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitDiffuseDirectLightImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitDiffuseIndirectLightImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitSpecularImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitClearImage;
    std::shared_ptr<vk::DeviceLocalImage> firstHitBaseEmissionImage;
    std::shared_ptr<vk::DeviceLocalImage> directLightDepthImage;
    std::shared_ptr<vk::DeviceLocalImage> diffuseRayDirHitDistImage;
    std::shared_ptr<vk::DeviceLocalImage> specularRayDirHitDistImage;

    // submodule
    std::shared_ptr<AtmosphereContext> atmosphereContext;
    std::shared_ptr<WorldPrepareContext> worldPrepareContext;

    RayTracingModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                            std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                            std::shared_ptr<RayTracingModule> rayTracingModule);

    void render() override;
};