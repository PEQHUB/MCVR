#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/modules/world/world_module.hpp"
#include "core/render/modules/world/shader_pack/shader_pack.hpp"

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

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

struct ShaderPackVisualSettings {
    int32_t cloudMode = 1;                    // 0=off, 1=vanilla geometry clouds, 2=volumetric
    int32_t captureVolumetricCloudIndirect = 0;
    int32_t volumetricCloudTemporalAccumulation = 0;
    int32_t volumetricCloudCastShadow = 0;
    int32_t waterSurfaceMode = 0;             // 0=vanilla, 1=realistic FFT
    int32_t waterCausticsEnabled = 0;
    int32_t volumetricLightMode = 0;          // 0=vanilla fog, 1=volumetric
    int32_t volumetricCloudClearAmount = 1;   // 0=few, 1=medium, 2=many

    int32_t indirectVolumetricCloudViewSteps = 16;
    int32_t indirectVolumetricCloudLightSteps = 4;
    int32_t indirectVolumetricCloudAmbientSteps = 2;
    int32_t volumetricCloudViewSteps = 24;
    int32_t volumetricCloudLightSteps = 6;
    int32_t volumetricCloudAmbientSteps = 4;
    int32_t volumetricLightSamples = 8;
    int32_t reservedVisual0 = 0;

    float indirectVolumetricCloudReflectionMaxRoughness = 0.12f;
    float volumetricCloudBottomHeight = 3000.0f;
    float volumetricCloudTopHeight = 11000.0f;
    float volumetricCloudBaseScale = 0.30f;
    float volumetricCloudDetailScale = 0.60f;
    float volumetricCloudCoverage = 0.50f;
    float volumetricCloudDensity = 1.00f;
    float volumetricCloudShadowSoftness = 0.50f;

    float volumetricCloudAmbientStrength = 1.0f;
    float volumetricCloudPowderStrength = 1.0f;
    float volumetricCloudWeatherScale = 0.020f;
    float volumetricLightScatteringStrength = 1.0f;
    float volumetricLightMaxDistance = 128.0f;
    float volumetricLightNearStepSize = 1.75f;
    float volumetricLightFarStepSize = 5.0f;
    float volumetricLightLuminanceLimit = 4.0f;
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
    void validateShaderPackRuntime();
    void initShaderPackRuntimeResources();
    void initShaderPackRayTracingExecutor();
    void disableShaderPackBackend(const std::string &status, const std::string &reason);
    bool recordShaderPackRayTracingGraph(uint32_t width, uint32_t height, bool &worldPassRequested);
    bool recordShaderPackRayTracingExecutor(const std::shared_ptr<vk::CommandBuffer> &commandBuffer,
                                            const std::shared_ptr<vk::DescriptorTable> &descriptorTable,
                                            const std::shared_ptr<WorldPrepareContext> &worldPrepareContext,
                                            const RayTracingPushConstant &pushConstant,
                                            uint32_t frameIndex,
                                            uint32_t width,
                                            uint32_t height);
    void bindLegacyTexturesForSlot(const std::shared_ptr<vk::DescriptorTable>& descriptorTable, uint32_t frameIndex);

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
    std::shared_ptr<vk::Shader> worldSolidAnyHitShader_;
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
    struct TextureDescriptorSlotState {
        uint64_t generation = UINT64_MAX;
        uint64_t materialTexturePageRevision = UINT64_MAX;
        VkImageView albedoTextureView = VK_NULL_HANDLE;
        VkImageView specularTextureView = VK_NULL_HANDLE;
        VkImageView normalTextureView = VK_NULL_HANDLE;
        VkImageView flagTextureView = VK_NULL_HANDLE;
        VkBuffer spriteRegistryBuffer = VK_NULL_HANDLE;
        VkBuffer materialRegistryBuffer = VK_NULL_HANDLE;
        VkBuffer textureRuleBuffer = VK_NULL_HANDLE;
    };
    std::vector<TextureDescriptorSlotState> textureDescriptorSlotStates_;
    struct LegacyTextureBinding {
        std::shared_ptr<vk::Sampler> sampler;
        std::shared_ptr<vk::DeviceLocalImage> image;
    };
    std::mutex legacyTextureBindingsMutex_;
    std::unordered_map<int, LegacyTextureBinding> legacyTextureBindings_;
    std::vector<uint64_t> legacyTextureBindingRevisions_;
    std::atomic<uint64_t> legacyTextureBindingRevision_{1};

    uint32_t numRayBounces_ = 2;
    bool useJitter_ = true;
    int shaderPackCloudMode_ = 1;          // 0=off, 1=vanilla, 2=volumetric
    int shaderPackWaterSurfaceMode_ = 0;   // 0=vanilla, 1=realistic FFT
    int shaderPackVolumetricLightMode_ = 0; // 0=vanilla, 1=volumetric pack mode
    bool shaderPackCloudShadowsEnabled_ = false;
    bool shaderPackWaterCausticsEnabled_ = false;
    ShaderPackVisualSettings shaderPackVisualSettings_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> shaderPackVisualSettingBuffers_;
    std::string shaderPackRequestedPath_ = "shaders/world/ray_tracing/vanilla-pt.zip";
    std::string shaderPackPath_ = "shaders/world/ray_tracing/vanilla-pt.zip";
    bool shaderPackRuntimeValid_ = false;
    std::string shaderPackRuntimeStatus_ = "not_initialized";
    std::vector<std::string> shaderPackAttributeKVs_;
    std::shared_ptr<ShaderPack> shaderPack_;
    bool shaderPackBackendEnabled_ = false;
    bool shaderPackRuntimeResourcesReady_ = false;
    bool shaderPackFirstFrameAttempted_ = false;
    std::string shaderPackBackendStatus_ = "not_initialized";
    std::string shaderPackFallbackReason_;
    uint64_t shaderPackGraphPassesExecuted_ = 0;
    uint64_t shaderPackGraphUnsupportedPasses_ = 0;
    uint64_t shaderPackGraphWorldPasses_ = 0;
    bool shaderPackExecutorReady_ = false;
    uint32_t shaderPackRuntimeResourceSetIndex_ = 5;
    uint32_t shaderPackExecutionSetIndex_ = 6;
    uint64_t shaderPackIntermediatesGenerated_ = 0;
    uint64_t shaderPackFullScreenPassDispatches_ = 0;
    uint64_t shaderPackComputePassDispatches_ = 0;
    uint64_t shaderPackRayTracingPassDispatches_ = 0;
    std::unordered_map<std::string, uint64_t> shaderPackPassDispatchCounts_;
    std::unordered_map<std::string, FullScreenPass> shaderPackFullScreenPasses_;
    std::unordered_map<std::string, ComputePass> shaderPackComputePasses_;
    std::unordered_map<std::string, RayTracingPass> shaderPackRayTracingPasses_;
    std::shared_ptr<vk::Shader> shaderPackFullScreenVertexShader_;
    ShaderPack::ExecutionVariables shaderPackRayTracingVariables_;
    std::unordered_set<std::string> shaderPackUnsupportedPassesLogged_;

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
