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
    int flags;
    int reservedLighting0;
    float reservedLighting1;
    int reservedLighting2;
    int reservedLighting3;
    int reservedLighting4;
    float preExposure;
    float displacementDepthScale;
    int displacementPrimarySteps;
    int displacementRefinementSteps;
    float displacementFadeDistanceBlocks;
    uint32_t blueNoiseFrame;
    uint32_t rtDebugFlags;
    uint32_t handInstanceCount;
    uint32_t reservedPc0;
    uint64_t sharcHashEntries;
    uint64_t sharcAccumulation;
    uint64_t sharcResolved;
    float sharcCameraX;
    float sharcCameraY;
    float sharcCameraZ;
    float sharcSceneScale;
    uint32_t sharcCapacity;
    float sharcRadianceScale;
    uint32_t sharcFrameIndex;
    float sharcRoughnessThreshold;
    int sharcUpdateBlockSize;
    int sharcUpdateBounces;
    int sharcQueryMode;
    int sharcQueryReserved;
    int offlineFlags;
    int accumFrameCount;
    float aperture;
    float focalDistance;
    uint64_t reservedAddr;
};

class RayTracingModule : public WorldModule, public SharedObject<RayTracingModule> {
    friend RayTracingModuleContext;
    friend Atmosphere;
    friend AtmosphereContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.ray_tracing.name";
    constexpr static uint32_t inputImageNum = 0;
    constexpr static uint32_t outputImageNum = 29;

    constexpr static std::string_view TARGET_RADIANCE = "out:radiance";
    constexpr static std::string_view TARGET_DIFFUSE_ALBEDO_METALLIC = "out:diffuse_albedo_metallic";
    constexpr static std::string_view TARGET_SPECULAR_ALBEDO = "out:specular_albedo";
    constexpr static std::string_view TARGET_NORMAL_ROUGHNESS = "out:normal_roughness";
    constexpr static std::string_view TARGET_MOTION_VECTOR = "out:motion_vector";
    constexpr static std::string_view TARGET_LINEAR_DEPTH = "out:linear_depth";
    constexpr static std::string_view TARGET_SPECULAR_HIT_DEPTH = "out:specular_hit_depth";
    constexpr static std::string_view TARGET_FIRST_HIT_DEPTH = "out:first_hit_depth";
    constexpr static std::string_view TARGET_FIRST_HIT_DIFFUSE_DIRECT_LIGHT = "out:first_hit_diffuse_direct_light";
    constexpr static std::string_view TARGET_FIRST_HIT_DIFFUSE_INDIRECT_LIGHT = "out:first_hit_diffuse_indirect_light";
    constexpr static std::string_view TARGET_FIRST_HIT_SPECULAR = "out:first_hit_specular";
    constexpr static std::string_view TARGET_FIRST_HIT_CLEAR = "out:first_hit_clear";
    constexpr static std::string_view TARGET_FIRST_HIT_BASE_EMISSION = "out:first_hit_base_emission";
    constexpr static std::string_view TARGET_FOG_IMAGE = "out:fog_image";
    constexpr static std::string_view TARGET_FIRST_HIT_REFRACTION = "out:first_hit_refraction";

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

    std::string diagnosticFeatureTruth() const;

    void preClose() override;

  private:
    void initDescriptorTables();
    void initImages();
    void initPipeline();
    void initSBT();
    void initSharcBuffers();
    void initSharcUpdatePipeline();
    void initSharcResolvePipeline();
    void initSharcQueryPipeline();

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
    std::shared_ptr<vk::Shader> worldSolidTransparentNoDisplacementClosestHitShader_;
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
    uint64_t lastTextureDescriptorRefreshGeneration_ = UINT64_MAX;
    VkImageView lastAlbedoTextureView_ = VK_NULL_HANDLE;
    VkImageView lastSpecularTextureView_ = VK_NULL_HANDLE;
    VkImageView lastNormalTextureView_ = VK_NULL_HANDLE;
    VkImageView lastFlagTextureView_ = VK_NULL_HANDLE;
    VkBuffer lastSpriteRegistryBuffer_ = VK_NULL_HANDLE;
    VkBuffer lastTextureRuleBuffer_ = VK_NULL_HANDLE;

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
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> reflectionMvImages_;            // DLSS-RR guide: specular reflection MVs
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> animatedTexMaskImages_;         // DLSS-RR guide: animated texture mask
    // Extended DLSS-RR guide buffers
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> particleMaskImages_;            // [18] transparent/refractive surface mask
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> biasMaskImages_;               // [19] history bias for animated/emissive surfaces
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> rtHitDistImages_;              // [20] per-pixel noise level hint
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> motionVectors3DImages_;        // [21] world-space 3D velocity
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> gbufferMetallicImages_;       // [22] GBuffer metallic
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> gbufferShadingModelIdImages_; // [23] GBuffer shading model ID
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> gbufferMaterialIdImages_;     // [24] GBuffer material ID
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> positionViewSpaceImages_;     // [25] view-space hit position
    // DLSS-RR transparency layer (stable planes)
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> transparencyLayerImages_;         // [26] glass/water foreground color
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> transparencyLayerOpacityImages_;  // [27] glass/water per-channel opacity
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> transparencyLayerMvecsImages_;    // [28] glass/water surface MVs
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> upstreamFogImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> upstreamFirstHitRefractionImages_;

    // Private SHARC query-pass candidate images. These are internal to RayTracingModule.
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> sharcCandidatePosHitTImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> sharcCandidateNormalRoughnessImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> sharcCandidateThroughputImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> sharcCandidatePrefixRadianceFlagsImages_;


    // SHARC radiance cache
    uint32_t sharcCapacity_ = 1u << 21; // dynamic: 2^exponent entries
    std::shared_ptr<vk::DeviceLocalBuffer> sharcHashEntries_;
    std::shared_ptr<vk::DeviceLocalBuffer> sharcAccumulation_;
    std::shared_ptr<vk::DeviceLocalBuffer> sharcResolved_;
    bool sharcBuffersInitialized_ = false;
    bool sharcResizePending_ = false;
    bool sharcDispatchReady_ = false;
    uint32_t sharcStreamStableFrames_ = 0;
    uint32_t sharcFrameIndex_ = 0;
    float sharcPrevCameraX_ = 0.0f, sharcPrevCameraY_ = 0.0f, sharcPrevCameraZ_ = 0.0f;

    // SHARC update RT pipeline (sparse tracing to populate cache)
    std::shared_ptr<vk::Shader> sharcUpdateRayGenShader_;
    std::shared_ptr<vk::RayTracingPipeline> sharcUpdatePipeline_;
    std::vector<std::shared_ptr<vk::SBT>> sharcUpdateSbts_;

    // SHARC resolve compute pipeline (temporal blend + stale eviction)
    VkPipeline sharcResolvePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout sharcResolvePipelineLayout_ = VK_NULL_HANDLE;
    std::shared_ptr<vk::Shader> sharcResolveShader_;

    // SHARC isolated query compute pipeline (keeps hash/BDA traversal out of world.rgen)
    VkPipeline sharcQueryPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout sharcQueryPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout sharcQueryDescSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool sharcQueryDescPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sharcQueryDescSets_;
    std::shared_ptr<vk::Shader> sharcQueryShader_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> sharcQueryCounterBuffers_;
    uint32_t sharcQueryLastCounters_[8] = {};

    // Offline accumulation compute pipeline (resources stored in Renderer statics)
    std::shared_ptr<vk::Shader> accumShader_;
    void initAccumulationPipeline();

    // Blue noise (Owen-scrambled Sobol + spatial scrambling tile)
    std::shared_ptr<class BlueNoise> blueNoise_;
    bool blueNoiseUploaded_ = false;

    // Energy compensation LUT (64x64 RGBA16F)
    // R = GGX E(NdotV, alpha), G = FON E(NdotV, r), B = GGX E_avg(alpha), A = FON E_avg(r)
    std::shared_ptr<vk::DeviceLocalImage> energyLUT_;
    std::shared_ptr<vk::Sampler> energyLUTSampler_;
    void initEnergyLUT();

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
    std::shared_ptr<vk::SBT> sharcUpdateSbt;  // SHARC update pipeline SBT

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
