#pragma once

// Ownership: EngineServices (1:1).
// Thread: Main thread only.
// Dependencies: DeviceService.
//
// Owns the per-frame uniforms (WorldUBO, SkyUBO) and global RT resources
// (blue noise, energy LUT, material/light SSBOs) needed by the real RT shaders.
// Mirrors V1's Renderer/Buffers globals but as a service.

#include "platform/vulkan/vk2_buffer.hpp"
#include "platform/vulkan/vk2_image.hpp"
#include "platform/vulkan/vk2_result.hpp"
#include "common/shared.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace engine {

namespace vk2 { class DeviceService; }
class ResourceGC;

struct CameraData;
struct ConfigSnapshot;

// Mirror of vk::Data::WorldUBO from common/shared.hpp:304.
// std430 layout — must match GLSL declaration in shader/util/world_ubo.glsl.
struct WorldUBOData {
    float cameraViewMat[16];
    float cameraEffectedViewMat[16];
    float cameraProjMat[16];
    float cameraViewMatInv[16];
    float cameraEffectedViewMatInv[16];
    float cameraProjMatInv[16];
    float cameraJitter[2];
    float gameTime;
    uint32_t seed;
    float textureMat[16];
    uint32_t overlayTextureID;
    uint32_t isFirstPerson;
    float fogStart;
    float fogEnd;
    float fogColor[4];
    uint32_t fogType;
    uint32_t skyType;
    uint32_t rayBounces;
    float cameraTmin;
    double cameraPos[4]; // dvec4
    uint32_t endSkyTextureID;
    uint32_t endPortalTextureID;
    uint32_t animTick;
    uint32_t pad5;
    float emissionData[50 * 4];
    float emissiveGamut[13 * 4];
};

// Mirror of vk::Data::SkyUBO. Simplified — fill in as needed.
struct SkyUBOData {
    float baseColor[3];
    uint32_t skyType;
    float horizonColor[4];
    float sunDirection[3];
    uint32_t isSunRisingOrSetting;
    float moonDirection[3];
    float moonDirPad;
    uint32_t isSkyDark;
    uint32_t hasBlindnessOrDarkness;
    uint32_t cameraSubmersionType;
    uint32_t moonPhase;
    float rainGradient;
    float hdrRadianceScale;
    float thunderGradient;
    float wetSurfaceStrength;
    // Atmosphere params
    float Rg, Rt, Hr, Hm;
    float betaR[3]; float mieG;
    float betaM[3]; float minViewCos;
    float sunRadiance[3]; uint32_t sunTextureID;
    float moonRadiance[3]; uint32_t moonTextureID;
    float envCelestial[4];
    float envWaterTintFog[4];
    float envSky[4];
    float envCloud[4];
    int32_t cloudTile[4];
    float cloudWrap[4];
    float cloudShape[4];
    float cloudLighting[4];
};

class SceneResourceService {
public:
    SceneResourceService() = default;
    ~SceneResourceService() = default;

    SceneResourceService(const SceneResourceService&) = delete;
    SceneResourceService& operator=(const SceneResourceService&) = delete;

    vk2::Result<void> init(vk2::DeviceService& device, uint32_t framesInFlight,
                            ResourceGC* gc = nullptr);

    // The number of frame slots used for per-frame SSBOs (set during init).
    uint32_t framesInFlight() const { return framesInFlight_; }
    void shutdown();

    bool isInitialized() const { return initialized_; }

    // Record any one-shot GPU init commands into cmd. Idempotent — after the first
    // successful call, subsequent calls are no-ops. Currently clears the energy LUT
    // to 1.0 and transitions it to SHADER_READ_ONLY_OPTIMAL.
    //
    // Call from processScene() once the first real command buffer is available.
    // Returns true iff init commands were recorded this call.
    bool runDeferredInit(VkCommandBuffer cmd);
    bool isDeferredInitComplete() const { return !firstFrameInitPending_; }

    // Set sub-pixel jitter (pixel-space, centered [-0.5, 0.5]).
    // Call each frame BEFORE updateWorldUBO() so the jitter is applied
    // to cameraProjMat and stored in WorldUBO::cameraJitter.
    void setJitter(float jitterX, float jitterY) { jitterX_ = jitterX; jitterY_ = jitterY; }

    // Update the WorldUBO for the current frame from camera + config.
    // Also snapshots current → last for temporal reprojection.
    void updateWorldUBO(uint32_t frameIndex, const CameraData& camera,
                        const ConfigSnapshot& config, uint64_t frameNumber);

    // Update SkyUBO from bridge data (sun direction, weather, etc.).
    void updateSkyUBO(const SkyUBOData& sky);

    // Read-only access to the latest sky data (CPU-side copy, updated by updateSkyUBO).
    // Used by StarFieldAdapter to read sunDirection, skyType, rainGradient for push constants.
    const SkyUBOData& latestSkyData() const { return latestSky_; }

    // Replace texture mapping SSBO contents with raw bytes.
    // Reallocates the SSBO if the new size differs from the existing one.
    // Caller's data is copied — safe to free immediately after the call.
    void uploadTextureMapping(const void* data, size_t size);

    // Replace area light SSBO contents with pre-gathered AreaLight structs.
    // Each entry is 48 bytes (vk::Data::AreaLight from common/shared.hpp).
    // Reallocates the SSBO if the new size differs. Stores the light count
    // for push constant propagation via areaLightCount().
    void uploadAreaLights(const void* data, size_t size, uint32_t lightCount);

    // Upload per-emissive-block emission data and gamut boosts.
    // Stored CPU-side and written into the WorldUBO each frame by updateWorldUBO().
    // Called once at startup and whenever the user changes emission settings.
    void uploadEmissionData(const float* emissionData, const float* gamutData);

    // Number of area lights currently uploaded (for push constants).
    uint32_t areaLightCount() const { return areaLightCount_; }

    // Per-frame buffer accessors (for descriptor writes).
    VkBuffer worldUBO(uint32_t frameIndex) const { return worldUBOs_[frameIndex].handle(); }
    VkBuffer lastWorldUBO(uint32_t frameIndex) const { return lastWorldUBOs_[frameIndex].handle(); }
    VkBuffer skyUBO(uint32_t frameIndex) const { return skyUBOs_[frameIndex].handle(); }

    VkDeviceSize worldUBOSize() const { return sizeof(WorldUBOData); }
    VkDeviceSize skyUBOSize() const { return sizeof(SkyUBOData); }

    // Blue noise (Owen-scrambled Sobol). Initialized to zeros for now —
    // proper data should be uploaded via bridge command later.
    VkBuffer blueNoiseSobol() const { return blueNoiseSobol_.handle(); }
    VkBuffer blueNoiseScrambling() const { return blueNoiseScrambling_.handle(); }
    VkDeviceSize blueNoiseSobolSize() const { return BLUE_NOISE_SOBOL_SIZE; }
    VkDeviceSize blueNoiseScramblingSize() const { return BLUE_NOISE_SCRAMBLING_SIZE; }

    // Energy compensation LUT (Kulla-Conty GGX+EON). 64x64 RGBA16F.
    // Initialized as a 1x1 white image until real LUT is loaded.
    VkImageView energyLUTView() const { return energyLUT_.defaultView(); }
    VkSampler energyLUTSampler() const { return linearSampler_; }

    // BLAS offsets SSBO — format-encoded upper 2 bits + byte offset per BLAS instance.
    // Stub until real data is populated by TlasService or bridge.
    VkBuffer blasOffsetsSSBO() const { return blasOffsetsSSBO_.handle(); }

    // Per-frame-slot SSBO accessors for double-buffered scene data.
    // Pass ctx.frameIndex so that each frame reads its own copy, avoiding WAR hazards
    // when bridge().flush() writes to the next slot while the GPU reads the previous.
    VkBuffer textureMappingSSBO(uint32_t frameIndex) const { return textureMappingSSBOs_[frameIndex % framesInFlight_].handle(); }
    VkBuffer materialClassSSBO() const { return materialClassSSBO_.handle(); }
    VkBuffer areaLightSSBO(uint32_t frameIndex) const { return areaLightSSBOs_[frameIndex % framesInFlight_].handle(); }
    VkBuffer tileLightSSBO(uint32_t frameIndex) const { return tileLightSSBOs_[frameIndex % framesInFlight_].handle(); }
    VkBuffer spriteRegistrySSBO(uint32_t frameIndex) const { return spriteRegistrySSBOs_[frameIndex % framesInFlight_].handle(); }
    VkBuffer blenderPbrSSBO(uint32_t frameIndex) const { return blenderPbrSSBOs_[frameIndex % framesInFlight_].handle(); }
    VkBuffer displacedFaceSSBO(uint32_t frameIndex) const { return displacedFaceSSBOs_[frameIndex % framesInFlight_].handle(); }
    // Per-instance biome tint colors (grass/foliage/water, uvec4 per instance).
    // Populated by the bridge with packed 0x00RRGGBB values; stub is zeroed (no tint).
    VkBuffer biomeColorSSBO(uint32_t frameIndex) const {
        if (biomeColorSSBOs_.size() < framesInFlight_) return VK_NULL_HANDLE;
        return biomeColorSSBOs_[frameIndex % framesInFlight_].handle();
    }
    VkDeviceSize biomeColorSSBOSize(uint32_t frameIndex) const {
        if (biomeColorSSBOs_.size() < framesInFlight_) return 0;
        return biomeColorSSBOs_[frameIndex % framesInFlight_].size();
    }

    // Grow (or keep) the biome color SSBO so it has room for at least minInstances
    // uvec4 entries (16 bytes each). The buffer is zero-filled (no tint). Safe to call
    // every frame — only reallocates when the current size is insufficient.
    // Must be called before the RT dispatch that reads biomeColors.data[instanceID].
    void resizeBiomeColors(uint32_t minInstances);

    // BDA accessors for RT shader push constants.
    // Valid only after init() — both buffers are created with SHADER_DEVICE_ADDRESS usage.
    VkDeviceAddress materialClassMappingBDA() const { return materialClassSSBO_.deviceAddress(); }
    VkDeviceAddress displacedFaceBDA(uint32_t frameIndex) const { return displacedFaceSSBOs_[frameIndex % framesInFlight_].deviceAddress(); }

    VkDeviceSize materialClassSSBOSize() const { return MATERIAL_CLASS_SSBO_SIZE; }
    VkDeviceSize tileLightSSBOSize(uint32_t frameIndex) const { return tileLightSSBOs_[frameIndex % framesInFlight_].size(); }

    // SHARC radiance cache buffers (BDA-accessible).
    // Three device-local buffers with capacity 2^21 entries:
    //   hashEntries: uint64 per entry  (16 MB)
    //   accumulation: SharcAccumulationData per entry (~32 MB, depends on struct size)
    //   resolved: SharcPackedData per entry (~16 MB)
    VkDeviceAddress sharcHashEntriesBDA() const { return sharcHashEntries_.deviceAddress(); }
    VkDeviceAddress sharcAccumulationBDA() const { return sharcAccumulation_.deviceAddress(); }
    VkDeviceAddress sharcResolvedBDA() const { return sharcResolved_.deviceAddress(); }
    bool sharcBuffersValid() const { return sharcHashEntries_.handle() != VK_NULL_HANDLE; }
    static constexpr uint32_t SHARC_CAPACITY = 1u << 21;  // 2M entries

    // Material class entry — uses the canonical shared.hpp definition (std430, 144 bytes each).
    // The GLSL shader reads materialClassMapping.entries[idx] via BDA using this exact layout.
    // Using the wrong (smaller) struct caused per-entry stride mismatch → GPU type=6 fault.
    using MaterialClassEntry = vk::Data::MaterialClassEntry;

    static constexpr uint32_t MATERIAL_CLASS_COUNT = static_cast<uint32_t>(vk::Data::MAX_MATERIAL_CLASSES); // 512, matches shader
    static constexpr VkDeviceSize MATERIAL_CLASS_SSBO_SIZE =
        static_cast<VkDeviceSize>(MATERIAL_CLASS_COUNT) * sizeof(MaterialClassEntry);

    // Tile light buffer constants (16x16 pixel tiles, sized for up to 3840x2160).
    static constexpr uint32_t TILE_SIZE = 16;
    static constexpr uint32_t MAX_TILES_X = 240;  // 3840 / 16
    static constexpr uint32_t MAX_TILES_Y = 135;  // 2160 / 16
    static constexpr uint32_t MAX_LIGHTS_PER_TILE = 32;
    // Per-tile: { uint lightCount; uint lightIndices[MAX_LIGHTS_PER_TILE]; } = 132 bytes
    static constexpr VkDeviceSize TILE_ENTRY_SIZE = sizeof(uint32_t) + MAX_LIGHTS_PER_TILE * sizeof(uint32_t);
    static constexpr VkDeviceSize TILE_LIGHT_SSBO_SIZE = MAX_TILES_X * MAX_TILES_Y * TILE_ENTRY_SIZE;

private:
    static constexpr VkDeviceSize BLUE_NOISE_SOBOL_SIZE = 256 * 256 * sizeof(uint32_t);
    static constexpr VkDeviceSize BLUE_NOISE_SCRAMBLING_SIZE = 128 * 128 * 8 * sizeof(uint32_t);
    static constexpr uint32_t ENERGY_LUT_DIM = 64;

    vk2::DeviceService* device_ = nullptr;
    ResourceGC* gc_ = nullptr;
    bool initialized_ = false;
    uint32_t framesInFlight_ = 1;

    // Per-frame UBOs (host-visible, persistently mapped for cheap updates)
    std::vector<vk2::Buffer> worldUBOs_;
    std::vector<vk2::Buffer> lastWorldUBOs_;
    std::vector<vk2::Buffer> skyUBOs_;

    // Static GPU resources
    vk2::Buffer blueNoiseSobol_;
    vk2::Buffer blueNoiseScrambling_;
    vk2::Image energyLUT_;

    // Single-instance SSBOs (not rewritten per frame — safe as shared buffers)
    vk2::Buffer blasOffsetsSSBO_;
    vk2::Buffer materialClassSSBO_;

    // Per-frame-slot double-buffered SSBOs. Each vector has framesInFlight_ entries so
    // that frame N reads slot N while frame N-1 may still be in flight on the GPU.
    // bridge().flush() writes to ALL slots (identical data); the GPU only reads its own slot.
    std::vector<vk2::Buffer> textureMappingSSBOs_;
    std::vector<vk2::Buffer> areaLightSSBOs_;
    std::vector<vk2::Buffer> tileLightSSBOs_;
    std::vector<vk2::Buffer> spriteRegistrySSBOs_;
    std::vector<vk2::Buffer> blenderPbrSSBOs_;
    std::vector<vk2::Buffer> displacedFaceSSBOs_;
    std::vector<vk2::Buffer> biomeColorSSBOs_;  // uvec4[] — grass/foliage/water tint per TLAS instance

    // Area light count (updated by uploadAreaLights, read by RayTracingAdapter)
    uint32_t areaLightCount_ = 0;

    // Per-emissive-block data (CPU cache, written into WorldUBO each frame)
    float emissionData_[50 * 4] = {};   // default: all zeros (no emission until Java uploads)
    float emissiveGamut_[13 * 4] = {};  // default: all zeros (overwritten to 1.0 if no upload)
    bool hasEmissionData_ = false;      // true once uploadEmissionData() has been called

    // SHARC radiance cache buffers (device-local, BDA-accessible)
    vk2::Buffer sharcHashEntries_;       // uint64_t per entry
    vk2::Buffer sharcAccumulation_;      // 16 bytes per entry (SharcAccumulationData)
    vk2::Buffer sharcResolved_;          // 8 bytes per entry (SharcPackedData)

    VkSampler linearSampler_ = VK_NULL_HANDLE;

    // Sub-pixel jitter from DLSS adapter (pixel-space, centered [-0.5, 0.5]).
    float jitterX_ = 0.0f;
    float jitterY_ = 0.0f;

    // Latest sky data (CPU-side, copied to per-frame UBOs on update)
    SkyUBOData latestSky_{};
    bool skyDirty_ = false;

    // Deferred first-frame init state.
    // energyLUT_ is device-local, so we can't memset it from the host. Instead we
    // allocate a host-visible staging buffer at init(), fill it with 1.0 on the CPU,
    // and have runDeferredInit() record a copy into the first processScene cmd buffer.
    bool firstFrameInitPending_ = true;
    vk2::Buffer energyLUTStaging_;

    // Helper: create a host-visible UBO buffer.
    vk2::Result<vk2::Buffer> createHostUBO(VkDeviceSize size, VkBufferUsageFlags usage);
    vk2::Result<vk2::Buffer> createDeviceSSBO(VkDeviceSize size);
    vk2::Result<vk2::Buffer> createHostSSBO(VkDeviceSize size);

    // Cached instance count last used to size biomeColorSSBOs_; avoids redundant reallocs.
    uint32_t biomeColorCapacity_ = 0;
};

} // namespace engine
