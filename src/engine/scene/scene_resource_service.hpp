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

#include <array>
#include <cstdint>
#include <vector>

namespace engine {

namespace vk2 { class DeviceService; }

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

    vk2::Result<void> init(vk2::DeviceService& device, uint32_t framesInFlight);
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

    // Update the WorldUBO for the current frame from camera + config.
    // Also snapshots current → last for temporal reprojection.
    void updateWorldUBO(uint32_t frameIndex, const CameraData& camera,
                        const ConfigSnapshot& config, uint64_t frameNumber);

    // Update SkyUBO from bridge data (sun direction, weather, etc.).
    void updateSkyUBO(const SkyUBOData& sky);

    // Replace texture mapping SSBO contents with raw bytes.
    // Reallocates the SSBO if the new size differs from the existing one.
    // Caller's data is copied — safe to free immediately after the call.
    void uploadTextureMapping(const void* data, size_t size);

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

    // Stub SSBOs for material/light data — populated via bridge later.
    VkBuffer textureMappingSSBO() const { return textureMappingSSBO_.handle(); }
    VkBuffer materialClassSSBO() const { return materialClassSSBO_.handle(); }
    VkBuffer areaLightSSBO() const { return areaLightSSBO_.handle(); }
    VkBuffer tileLightSSBO() const { return tileLightSSBO_.handle(); }
    VkBuffer spriteRegistrySSBO() const { return spriteRegistrySSBO_.handle(); }
    VkBuffer blenderPbrSSBO() const { return blenderPbrSSBO_.handle(); }
    VkBuffer displacedFaceSSBO() const { return displacedFaceSSBO_.handle(); }

private:
    static constexpr VkDeviceSize BLUE_NOISE_SOBOL_SIZE = 256 * 256 * sizeof(uint32_t);
    static constexpr VkDeviceSize BLUE_NOISE_SCRAMBLING_SIZE = 128 * 128 * 8 * sizeof(uint32_t);
    static constexpr uint32_t ENERGY_LUT_DIM = 64;

    vk2::DeviceService* device_ = nullptr;
    bool initialized_ = false;

    // Per-frame UBOs (host-visible, persistently mapped for cheap updates)
    std::vector<vk2::Buffer> worldUBOs_;
    std::vector<vk2::Buffer> lastWorldUBOs_;
    std::vector<vk2::Buffer> skyUBOs_;

    // Static GPU resources
    vk2::Buffer blueNoiseSobol_;
    vk2::Buffer blueNoiseScrambling_;
    vk2::Image energyLUT_;

    // Stub SSBOs (16 bytes each — populated by bridge later)
    vk2::Buffer textureMappingSSBO_;
    vk2::Buffer materialClassSSBO_;
    vk2::Buffer areaLightSSBO_;
    vk2::Buffer tileLightSSBO_;
    vk2::Buffer spriteRegistrySSBO_;
    vk2::Buffer blenderPbrSSBO_;
    vk2::Buffer displacedFaceSSBO_;

    VkSampler linearSampler_ = VK_NULL_HANDLE;

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
};

} // namespace engine
