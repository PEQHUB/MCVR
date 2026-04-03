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
                         // bit 5: SHARC enabled, bit 6: noise LOD
                         // bit 7: multi-scatter GGX, bit 8: EON diffuse
    int areaLightCount;  // number of active area lights this frame
    float shadowSoftness;
    int risCandidates;   // total RIS candidates per pixel
    int temporalMClamp;  // temporal reservoir M clamp (used as float in shader)
    int wClamp;          // importance weight W clamp (used as float in shader)
    float preExposure;   // pre-exposure multiplier for DLSS-RR normalization
    // POM fields (fields 8-11, 16 bytes)
    float pomHeightScale;        // 0 = disabled, else depth scale (0.01-0.50)
    int   pomSteps;              // linear search steps (8-512)
    int   pomRefinement;         // binary refinement iterations (0-8)
    float pomFadeDistance;       // distance in blocks to fade POM out (8-256)
    // Color expansion (offset 48)
    float colorExpansion;        // per-block vivid color chroma boost (0.0-2.0, 1.0=neutral)
    uint32_t blueNoiseFrame;     // monotonic frame counter for blue noise temporal offset
    // SHARC fields (offset 56, 52 bytes) — buffer device addresses + grid params
    uint64_t sharcHashEntries;   // BDA of hash entry buffer
    uint64_t sharcAccumulation;  // BDA of accumulation buffer
    uint64_t sharcResolved;      // BDA of resolved radiance buffer
    float sharcCameraX;          // camera world position for LOD grid
    float sharcCameraY;
    float sharcCameraZ;
    float sharcSceneScale;       // scene scale for voxel sizing (default 4.0)
    uint32_t sharcCapacity;      // hash map capacity (2^21 = 2M entries)
    float sharcRadianceScale;    // quantization scale for accumulation atomics
    uint32_t sharcFrameIndex;    // frame counter for resolve
    float sharcRoughnessThreshold; // min roughness for cache query (0=all, 1=diffuse only)
    int sharcUpdateBlockSize;      // sparse update NxN block size (2-8)
    int sharcUpdateBounces;        // max bounces in SHARC update pass (2-8)
    // Offline accumulation fields (offset 112, 16 bytes)
    int offlineFlags;              // bit 0: accumulating, bit 1: disable RR, bit 2: disable clamp
    int accumFrameCount;           // frame index for jitter sequence during accumulation
    float aperture;                // thin lens aperture radius (0 = pinhole)
    float focalDistance;           // focal distance in blocks
    // Material SSBO BDA — avoids descriptor lookup for material reads
    uint64_t materialClassAddr;    // BDA of MaterialClassMapping buffer
    // DDA displacement BDA
    uint64_t displacedFaceDataAddr; // BDA of merged DisplacedFaceData buffer (0 = no displaced faces)
    T_INT displacedFaceCount;       // Total displaced faces across all chunks
    T_INT displacementQuality;      // 0=Off, 1=DDA, 2=Tess, 3=Hybrid, 4=CLAS
};

class RayTracingModule : public WorldModule, public SharedObject<RayTracingModule> {
    friend RayTracingModuleContext;
    friend Atmosphere;
    friend AtmosphereContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.ray_tracing.name";
    constexpr static uint32_t inputImageNum = 0;
    constexpr static uint32_t outputImageNum = 29;

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
    void initSharcBuffers();
    void initSharcUpdatePipeline();
    void initSharcResolvePipeline();

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

    // DDA displacement procedural hit group
    std::shared_ptr<vk::Shader> displacedIntersectionShader_;  // .rint
    std::shared_ptr<vk::Shader> displacedClosestHitShader_;    // .rchit
    std::shared_ptr<vk::Shader> displacedShadowClosestHitShader_; // .rchit (shadow)

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

    // SHARC radiance cache
    uint32_t sharcCapacity_ = 1u << 21; // dynamic: 2^exponent entries
    std::shared_ptr<vk::DeviceLocalBuffer> sharcHashEntries_;
    std::shared_ptr<vk::DeviceLocalBuffer> sharcAccumulation_;
    std::shared_ptr<vk::DeviceLocalBuffer> sharcResolved_;
    bool sharcBuffersInitialized_ = false;
    bool sharcResizePending_ = false;
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