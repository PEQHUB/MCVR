#pragma once

// V1-compatible 4-set ray tracing descriptor layout.
//
// Set 0 (textures, 6 bindings):    bindless texture array, atmosphere LUT/cube,
//                                   sprite albedo/specular/normal arrays
// Set 1 (geometry, 14 bindings):   TLAS, BLAS offsets, vtx/idx BDAs (cur+prev),
//                                   prev transforms, texture/material/light SSBOs,
//                                   blender PBR, displaced face data, sprite registry
// Set 2 (uniforms, 6 bindings):    WorldUBO (cur), LastWorldUBO, SkyUBO,
//                                   energy LUT (binding 4), blue noise sobol+scrambling
//                                   (bindings 9, 10) — note V1 has gaps in this set
// Set 3 (outputs, 34 bindings):    29 RT output storage images + 5 reservoir images
//
// All non-essential bindings are PARTIALLY_BOUND so test shaders can leave them
// unwritten until PR39 ports the real shaders.

#include "platform/vulkan/vk2_pipeline.hpp"
#include "platform/vulkan/vk2_result.hpp"

#include <array>
#include <cstdint>

#include <vulkan/vulkan.h>

namespace engine {

namespace rt_layout {

// --- Set 0 (textures) ---
constexpr uint32_t SET_TEXTURES = 0;
constexpr uint32_t TEX_BINDING_BINDLESS = 0;       // 4096 combined image samplers
constexpr uint32_t TEX_BINDING_ATMO_LUT = 1;
constexpr uint32_t TEX_BINDING_ATMO_CUBE = 2;
constexpr uint32_t TEX_BINDING_SPRITE_ALBEDO = 3;
constexpr uint32_t TEX_BINDING_SPRITE_SPECULAR = 4;
constexpr uint32_t TEX_BINDING_SPRITE_NORMAL = 5;
constexpr uint32_t TEX_BINDLESS_COUNT = 4096;

// --- Set 1 (geometry/buffers) ---
constexpr uint32_t SET_GEOMETRY = 1;
constexpr uint32_t GEO_BINDING_TLAS = 0;
constexpr uint32_t GEO_BINDING_BLAS_OFFSETS = 1;
constexpr uint32_t GEO_BINDING_VTX_ADDRS = 2;
constexpr uint32_t GEO_BINDING_IDX_ADDRS = 3;
constexpr uint32_t GEO_BINDING_LAST_VTX_ADDRS = 4;
constexpr uint32_t GEO_BINDING_LAST_IDX_ADDRS = 5;
constexpr uint32_t GEO_BINDING_LAST_TRANSFORMS = 6;
constexpr uint32_t GEO_BINDING_TEXTURE_MAPPING = 7;
constexpr uint32_t GEO_BINDING_AREA_LIGHT = 8;
constexpr uint32_t GEO_BINDING_TILE_LIGHT = 9;
constexpr uint32_t GEO_BINDING_BLENDER_PBR = 10;
constexpr uint32_t GEO_BINDING_MATERIAL_CLASS = 11;
constexpr uint32_t GEO_BINDING_DISPLACED_FACE = 12;
constexpr uint32_t GEO_BINDING_SPRITE_REGISTRY = 13;

// --- Set 2 (uniforms + blue noise + energy LUT) ---
constexpr uint32_t SET_UNIFORMS = 2;
constexpr uint32_t UBO_BINDING_WORLD = 0;
constexpr uint32_t UBO_BINDING_LAST_WORLD = 1;
constexpr uint32_t UBO_BINDING_SKY = 2;
// Note: V1 has a gap at binding 3 (no descriptor)
constexpr uint32_t UBO_BINDING_ENERGY_LUT = 4;     // combined image sampler
// Note: V1 has gaps at bindings 5-8 (these used to live in set 1)
constexpr uint32_t UBO_BINDING_BLUE_NOISE_SOBOL = 9;
constexpr uint32_t UBO_BINDING_BLUE_NOISE_SCRAMBLING = 10;

// --- Set 3 (output storage images) ---
constexpr uint32_t SET_OUTPUTS = 3;
constexpr uint32_t OUT_BINDING_HDR_NOISY = 0;
constexpr uint32_t OUT_BINDING_DIFFUSE_ALBEDO = 1;
constexpr uint32_t OUT_BINDING_SPECULAR_ALBEDO = 2;
constexpr uint32_t OUT_BINDING_NORMAL_ROUGHNESS = 3;
constexpr uint32_t OUT_BINDING_MOTION_VECTOR = 4;
constexpr uint32_t OUT_BINDING_LINEAR_DEPTH = 5;
constexpr uint32_t OUT_BINDING_SPECULAR_HIT_DEPTH = 6;
constexpr uint32_t OUT_BINDING_FIRST_HIT_DEPTH = 7;
constexpr uint32_t OUT_BINDING_FH_DIFFUSE_DIRECT = 8;
constexpr uint32_t OUT_BINDING_FH_DIFFUSE_INDIRECT = 9;
constexpr uint32_t OUT_BINDING_FH_SPECULAR = 10;
constexpr uint32_t OUT_BINDING_FH_CLEAR = 11;
constexpr uint32_t OUT_BINDING_FH_BASE_EMISSION = 12;
constexpr uint32_t OUT_BINDING_DIRECT_LIGHT_DEPTH = 13;
constexpr uint32_t OUT_BINDING_RESERVOIR_CURRENT = 14;
constexpr uint32_t OUT_BINDING_RESERVOIR_PREVIOUS = 15;
constexpr uint32_t OUT_BINDING_DIFFUSE_RAY_DIR = 16;
constexpr uint32_t OUT_BINDING_SPECULAR_RAY_DIR = 17;
constexpr uint32_t OUT_BINDING_BOUNCE_RESERVOIR_1 = 18;
constexpr uint32_t OUT_BINDING_BOUNCE_RESERVOIR_2 = 19;
constexpr uint32_t OUT_BINDING_BOUNCE_RESERVOIR_3 = 20;
constexpr uint32_t OUT_BINDING_REFLECTION_MV = 21;
constexpr uint32_t OUT_BINDING_ANIM_TEX_MASK = 22;
constexpr uint32_t OUT_BINDING_PARTICLE_MASK = 23;
constexpr uint32_t OUT_BINDING_BIAS_MASK = 24;
constexpr uint32_t OUT_BINDING_RT_HIT_DIST = 25;
constexpr uint32_t OUT_BINDING_MOTION_VEC_3D = 26;
constexpr uint32_t OUT_BINDING_GBUFFER_METALLIC = 27;
constexpr uint32_t OUT_BINDING_GBUFFER_SHADING_MODEL_ID = 28;
constexpr uint32_t OUT_BINDING_GBUFFER_MATERIAL_ID = 29;
constexpr uint32_t OUT_BINDING_POSITION_VIEW_SPACE = 30;
constexpr uint32_t OUT_BINDING_TRANSPARENCY_LAYER = 31;
constexpr uint32_t OUT_BINDING_TRANSPARENCY_OPACITY = 32;
constexpr uint32_t OUT_BINDING_TRANSPARENCY_MVECS = 33;
constexpr uint32_t OUT_BINDING_COUNT = 34;

} // namespace rt_layout

// V1-compatible push constant struct. Mirrors RayTracingPushConstant in
// core/render/modules/world/ray_tracing/ray_tracing_module.hpp.
// 160 bytes — must be byte-for-byte compatible with V1 once shaders are ported.
struct RtPushConstants {
    int32_t  numRayBounces;          // 0
    int32_t  flags;                  // 4   bits 0=simplifiedIndirect, 1=areaLights, 2=restir,
                                     //         3=simplifiedBRDF, 4=restirBounce, 5=sharc,
                                     //         6=noiseLOD, 7=multiScatterGGX, 8=eonDiffuse
    int32_t  areaLightCount;         // 8
    float    shadowSoftness;         // 12
    int32_t  risCandidates;          // 16
    int32_t  temporalMClamp;         // 20  (used as float in shader)
    int32_t  wClamp;                 // 24  (used as float in shader)
    float    preExposure;            // 28
    float    pomHeightScale;         // 32
    int32_t  pomSteps;               // 36
    int32_t  pomRefinement;          // 40
    float    pomFadeDistance;        // 44
    float    colorExpansion;         // 48
    uint32_t blueNoiseFrame;         // 52
    uint64_t sharcHashEntries;       // 56
    uint64_t sharcAccumulation;      // 64
    uint64_t sharcResolved;          // 72
    float    sharcCameraX;           // 80
    float    sharcCameraY;           // 84
    float    sharcCameraZ;           // 88
    float    sharcSceneScale;        // 92
    uint32_t sharcCapacity;          // 96
    float    sharcRadianceScale;     // 100
    uint32_t sharcFrameIndex;        // 104
    float    sharcRoughnessThreshold;// 108
    int32_t  sharcUpdateBlockSize;   // 112
    int32_t  sharcUpdateBounces;     // 116
    int32_t  offlineFlags;           // 120
    int32_t  accumFrameCount;        // 124
    float    aperture;               // 128
    float    focalDistance;          // 132
    uint64_t materialClassAddr;      // 136
    uint64_t displacedFaceDataAddr;  // 144
    int32_t  displacedFaceCount;     // 152
    int32_t  displacementQuality;    // 156
};                                   // 160 bytes total

static_assert(sizeof(RtPushConstants) == 160,
              "RtPushConstants must be 160 bytes — matches V1 RayTracingPushConstant");

// Owns the four V1-compatible descriptor set layouts and the combined pipeline layout.
// Adapter allocates per-frame descriptor sets from these layouts.
class RtDescriptorLayout {
public:
    RtDescriptorLayout() = default;
    ~RtDescriptorLayout() { shutdown(); }

    RtDescriptorLayout(const RtDescriptorLayout&) = delete;
    RtDescriptorLayout& operator=(const RtDescriptorLayout&) = delete;

    vk2::Result<void> init(VkDevice device);
    void shutdown();

    bool isInitialized() const { return device_ != VK_NULL_HANDLE; }

    VkDescriptorSetLayout setLayout(uint32_t set) const { return setLayouts_[set]; }
    VkPipelineLayout pipelineLayout() const { return pipelineLayout_.handle(); }

    // Pool sizes that match the maximum requirements of all 4 sets combined.
    // Multiply by maxSetsPerFrame in DescriptorAllocator::init.
    static std::vector<VkDescriptorPoolSize> poolSizes();

private:
    VkDevice device_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSetLayout, 4> setLayouts_{};
    vk2::PipelineLayout pipelineLayout_;
};

} // namespace engine
