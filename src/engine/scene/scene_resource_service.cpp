#include "scene_resource_service.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "frame/frame_context.hpp"
#include "frame/resource_gc.hpp"
#include "config/config_service.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

// Blue noise source data from FidelityFX SDK (SSSR component).
// Defines: static const int sobol_256spp_256d[256*256]
//          static const int scramblingTile[128*128*8]
#include "../../../extern/FidelityFX-SDK/sdk/src/components/sssr/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_1spp.cpp"

// ---------- CPU helpers for energy compensation LUT baking ----------

struct Vec3 { float x, y, z; };

Vec3 normalize3(Vec3 v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (len < 1e-8f) return {0, 0, 1};
    return {v.x/len, v.y/len, v.z/len};
}

Vec3 sampleGGXVNDF_CPU(Vec3 V, float ax, float ay, float r1, float r2) {
    Vec3 Vh = normalize3({ax * V.x, ay * V.y, V.z});
    float lensq = Vh.x*Vh.x + Vh.y*Vh.y;
    Vec3 T1 = lensq > 0 ? Vec3{-Vh.y / sqrtf(lensq), Vh.x / sqrtf(lensq), 0}
                         : Vec3{1, 0, 0};
    Vec3 T2 = {Vh.y*T1.z - Vh.z*T1.y, Vh.z*T1.x - Vh.x*T1.z, Vh.x*T1.y - Vh.y*T1.x};
    float r = sqrtf(r1);
    float phi = 6.28318530718f * r2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1*t1)) + s * t2;
    float nz = sqrtf(fmaxf(0.0f, 1.0f - t1*t1 - t2*t2));
    Vec3 Nh = {t1*T1.x + t2*T2.x + nz*Vh.x,
               t1*T1.y + t2*T2.y + nz*Vh.y,
               t1*T1.z + t2*T2.z + nz*Vh.z};
    return normalize3({ax * Nh.x, ay * Nh.y, fmaxf(0.0f, Nh.z)});
}

float smithG1_CPU(float NdotV, float alpha) {
    float a2 = alpha * alpha;
    float c = NdotV;
    return (2.0f * c) / (c + sqrtf(a2 + c*c - a2*c*c));
}

uint32_t xorshift32(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float randFloat(uint32_t &state) {
    return (float)(xorshift32(state) & 0xFFFFFF) / (float)0xFFFFFF;
}

float bakeGGXDirectionalAlbedo(float NdotV, float alpha, int numSamples) {
    NdotV = fmaxf(NdotV, 1e-4f);
    alpha = fmaxf(alpha, 1e-4f);
    Vec3 V = {sqrtf(1.0f - NdotV*NdotV), 0.0f, NdotV};
    float sum = 0.0f;
    uint32_t seed = (uint32_t)(NdotV * 12345.0f + alpha * 67890.0f + 1);
    for (int i = 0; i < numSamples; i++) {
        float r1 = randFloat(seed);
        float r2 = randFloat(seed);
        Vec3 H = sampleGGXVNDF_CPU(V, alpha, alpha, r1, r2);
        float VdotH = V.x*H.x + V.y*H.y + V.z*H.z;
        Vec3 L = {2.0f*VdotH*H.x - V.x, 2.0f*VdotH*H.y - V.y, 2.0f*VdotH*H.z - V.z};
        if (L.z > 0.0f) {
            sum += smithG1_CPU(L.z, alpha);
        }
    }
    return fminf(fmaxf(sum / (float)numSamples, 0.0f), 1.0f);
}

float bakeFONDirectionalAlbedo(float mu_o, float r, int numSamples) {
    mu_o = fmaxf(mu_o, 1e-4f);
    float sinTheta_o = sqrtf(1.0f - mu_o*mu_o);
    float A_F = 1.0f / (1.0f + (0.5f - 2.0f / (3.0f * 3.14159265f)) * r);
    float B_F = r * A_F;
    float sum = 0.0f;
    uint32_t seed = (uint32_t)(mu_o * 54321.0f + r * 98765.0f + 7);
    for (int i = 0; i < numSamples; i++) {
        float r1 = randFloat(seed);
        float r2 = randFloat(seed);
        float sqrtR1 = sqrtf(r1);
        float phi = 6.28318530718f * r2;
        float Lx = sqrtR1 * cosf(phi);
        float Ly = sqrtR1 * sinf(phi);
        float Lz = sqrtf(fmaxf(0.0f, 1.0f - r1));
        float mu_i = Lz;
        float VdotL = Lx * sinTheta_o + Lz * mu_o;
        float s = VdotL - mu_i * mu_o;
        float t = (s > 0.0f) ? fmaxf(mu_i, mu_o) : 1.0f;
        float f_FON = (1.0f / 3.14159265f) * (A_F + B_F * s / t);
        sum += f_FON * 3.14159265f;
    }
    return fminf(fmaxf(sum / (float)numSamples, 0.0f), 1.0f);
}

uint16_t floatToHalf(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFF;
    if (exponent <= 0) return (uint16_t)sign;
    if (exponent >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exponent << 10) | (mantissa >> 13));
}

// Column-major 4x4 inverse — used for cameraViewMatInv / cameraProjMatInv.
// Adapted from the same routine in raytracing_adapter.cpp.
void invertMatrix4(const float* m, float* out) {
    float s0 = m[0]*m[5] - m[4]*m[1], s1 = m[0]*m[6] - m[4]*m[2];
    float s2 = m[0]*m[7] - m[4]*m[3], s3 = m[1]*m[6] - m[5]*m[2];
    float s4 = m[1]*m[7] - m[5]*m[3], s5 = m[2]*m[7] - m[6]*m[3];
    float c5 = m[10]*m[15]-m[14]*m[11], c4 = m[9]*m[15]-m[13]*m[11];
    float c3 = m[9]*m[14]-m[13]*m[10], c2 = m[8]*m[15]-m[12]*m[11];
    float c1 = m[8]*m[14]-m[12]*m[10], c0 = m[8]*m[13]-m[12]*m[9];
    float det = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;
    if (std::abs(det) < 1e-12f) {
        std::memset(out, 0, 16 * sizeof(float));
        out[0] = out[5] = out[10] = out[15] = 1.0f;
        return;
    }
    float inv = 1.0f / det;
    out[0]  = ( m[5]*c5 - m[6]*c4 + m[7]*c3) * inv;
    out[1]  = (-m[1]*c5 + m[2]*c4 - m[3]*c3) * inv;
    out[2]  = ( m[13]*s5 - m[14]*s4 + m[15]*s3) * inv;
    out[3]  = (-m[9]*s5 + m[10]*s4 - m[11]*s3) * inv;
    out[4]  = (-m[4]*c5 + m[6]*c2 - m[7]*c1) * inv;
    out[5]  = ( m[0]*c5 - m[2]*c2 + m[3]*c1) * inv;
    out[6]  = (-m[12]*s5 + m[14]*s2 - m[15]*s1) * inv;
    out[7]  = ( m[8]*s5 - m[10]*s2 + m[11]*s1) * inv;
    out[8]  = ( m[4]*c4 - m[5]*c2 + m[7]*c0) * inv;
    out[9]  = (-m[0]*c4 + m[1]*c2 - m[3]*c0) * inv;
    out[10] = ( m[12]*s4 - m[13]*s2 + m[15]*s0) * inv;
    out[11] = (-m[8]*s4 + m[9]*s2 - m[11]*s0) * inv;
    out[12] = (-m[4]*c3 + m[5]*c1 - m[6]*c0) * inv;
    out[13] = ( m[0]*c3 - m[1]*c1 + m[2]*c0) * inv;
    out[14] = (-m[12]*s3 + m[13]*s1 - m[14]*s0) * inv;
    out[15] = ( m[8]*s3 - m[9]*s1 + m[10]*s0) * inv;
}

} // namespace

namespace engine {

vk2::Result<vk2::Buffer> SceneResourceService::createHostUBO(VkDeviceSize size, VkBufferUsageFlags usage) {
    vk2::Buffer::Desc bd{};
    bd.size = size;
    bd.usage = usage | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    return vk2::Buffer::create(device_->device(), device_->vma(), bd);
}

vk2::Result<vk2::Buffer> SceneResourceService::createDeviceSSBO(VkDeviceSize size) {
    vk2::Buffer::Desc bd{};
    bd.size = size;
    bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
             | VK_BUFFER_USAGE_TRANSFER_DST_BIT
             | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    return vk2::Buffer::create(device_->device(), device_->vma(), bd);
}

vk2::Result<vk2::Buffer> SceneResourceService::createHostSSBO(VkDeviceSize size) {
    vk2::Buffer::Desc bd{};
    bd.size = size;
    bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
             | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    return vk2::Buffer::create(device_->device(), device_->vma(), bd);
}

vk2::Result<void> SceneResourceService::init(vk2::DeviceService& device, uint32_t framesInFlight,
                                               ResourceGC* gc) {
    device_ = &device;
    gc_ = gc;
    framesInFlight_ = (framesInFlight > 0) ? framesInFlight : 1;

    // Per-frame UBOs
    worldUBOs_.reserve(framesInFlight);
    lastWorldUBOs_.reserve(framesInFlight);
    skyUBOs_.reserve(framesInFlight);
    for (uint32_t i = 0; i < framesInFlight; ++i) {
        auto w = createHostUBO(sizeof(WorldUBOData), 0);
        if (!w) return vk2::makeError(w.error().vkResult, "WorldUBO: " + w.error().message);
        worldUBOs_.push_back(std::move(w.value()));

        auto lw = createHostUBO(sizeof(WorldUBOData), 0);
        if (!lw) return vk2::makeError(lw.error().vkResult, "LastWorldUBO: " + lw.error().message);
        lastWorldUBOs_.push_back(std::move(lw.value()));

        auto s = createHostUBO(sizeof(SkyUBOData), 0);
        if (!s) return vk2::makeError(s.error().vkResult, "SkyUBO: " + s.error().message);
        skyUBOs_.push_back(std::move(s.value()));
    }

    // Blue noise buffers (host-visible, populated from FidelityFX blue noise data)
    {
        vk2::Buffer::Desc bd{};
        bd.size = BLUE_NOISE_SOBOL_SIZE;
        bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        auto r = vk2::Buffer::create(device_->device(), device_->vma(), bd);
        if (!r) return vk2::makeError(r.error().vkResult, "Blue noise Sobol: " + r.error().message);
        blueNoiseSobol_ = std::move(r.value());
        if (blueNoiseSobol_.mappedPtr()) {
            uint32_t* dst = static_cast<uint32_t*>(blueNoiseSobol_.mappedPtr());
            for (size_t i = 0; i < 256 * 256; ++i)
                dst[i] = static_cast<uint32_t>(sobol_256spp_256d[i]);
        }
    }
    {
        vk2::Buffer::Desc bd{};
        bd.size = BLUE_NOISE_SCRAMBLING_SIZE;
        bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        auto r = vk2::Buffer::create(device_->device(), device_->vma(), bd);
        if (!r) return vk2::makeError(r.error().vkResult, "Blue noise scrambling: " + r.error().message);
        blueNoiseScrambling_ = std::move(r.value());
        if (blueNoiseScrambling_.mappedPtr()) {
            uint32_t* dst = static_cast<uint32_t*>(blueNoiseScrambling_.mappedPtr());
            for (size_t i = 0; i < 128 * 128 * 8; ++i)
                dst[i] = static_cast<uint32_t>(scramblingTile[i]);
        }
    }

    // Energy LUT (64x64 RGBA16F, sampled). Initialized to all-1.0 (no compensation)
    // via deferred GPU copy from a host-visible staging buffer. The clear cannot happen
    // here because the engine has no transfer cmd buffer at init time — see runDeferredInit().
    {
        vk2::Image::Desc id{};
        id.width = ENERGY_LUT_DIM;
        id.height = ENERGY_LUT_DIM;
        id.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        id.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        auto r = vk2::Image::create(device_->device(), device_->vma(), id);
        if (!r) return vk2::makeError(r.error().vkResult, "Energy LUT: " + r.error().message);
        energyLUT_ = std::move(r.value());
    }

    // Energy LUT staging buffer: host-visible, baked with real energy compensation data.
    // Layout: RGBA16F, R=GGX E(NdotV,alpha), G=FON E(NdotV,r), B=GGX Eavg, A=FON Eavg
    // UV: u=NdotV [0,1], v=perceptual roughness r [0,1]; alpha = r*r
    {
        constexpr VkDeviceSize lutBytes = ENERGY_LUT_DIM * ENERGY_LUT_DIM * 4 * sizeof(uint16_t);
        constexpr int LUT_SIZE = static_cast<int>(ENERGY_LUT_DIM);
        constexpr int GGX_SAMPLES = 1024;
        constexpr int FON_SAMPLES = 512;

        vk2::Buffer::Desc bd{};
        bd.size = lutBytes;
        bd.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        auto r = vk2::Buffer::create(device_->device(), device_->vma(), bd);
        if (!r) return vk2::makeError(r.error().vkResult, "Energy LUT staging: " + r.error().message);
        energyLUTStaging_ = std::move(r.value());

        if (energyLUTStaging_.mappedPtr()) {
            // First pass: compute per-texel directional albedo
            std::vector<float> ggxE(LUT_SIZE * LUT_SIZE);
            std::vector<float> fonE(LUT_SIZE * LUT_SIZE);

            for (int y = 0; y < LUT_SIZE; y++) {
                float roughness = ((float)y + 0.5f) / (float)LUT_SIZE;
                float alpha = roughness * roughness;
                for (int x = 0; x < LUT_SIZE; x++) {
                    float NdotV = ((float)x + 0.5f) / (float)LUT_SIZE;
                    int idx = y * LUT_SIZE + x;
                    ggxE[idx] = bakeGGXDirectionalAlbedo(NdotV, alpha, GGX_SAMPLES);
                    fonE[idx] = bakeFONDirectionalAlbedo(NdotV, roughness, FON_SAMPLES);
                }
            }

            // Second pass: cosine-weighted average per roughness row
            std::vector<float> ggxEavg(LUT_SIZE);
            std::vector<float> fonEavg(LUT_SIZE);

            for (int y = 0; y < LUT_SIZE; y++) {
                float sumGGX = 0.0f, sumFON = 0.0f, sumWeight = 0.0f;
                for (int x = 0; x < LUT_SIZE; x++) {
                    float NdotV = ((float)x + 0.5f) / (float)LUT_SIZE;
                    float weight = NdotV;
                    int idx = y * LUT_SIZE + x;
                    sumGGX += ggxE[idx] * weight;
                    sumFON += fonE[idx] * weight;
                    sumWeight += weight;
                }
                ggxEavg[y] = sumWeight > 0 ? sumGGX / sumWeight : 0.5f;
                fonEavg[y] = sumWeight > 0 ? sumFON / sumWeight : 0.5f;
            }

            // Pack into RGBA16F staging buffer
            uint16_t* dst = static_cast<uint16_t*>(energyLUTStaging_.mappedPtr());
            for (int y = 0; y < LUT_SIZE; y++) {
                for (int x = 0; x < LUT_SIZE; x++) {
                    int idx = y * LUT_SIZE + x;
                    int pixelBase = idx * 4;
                    dst[pixelBase + 0] = floatToHalf(ggxE[idx]);     // R: GGX E(NdotV, alpha)
                    dst[pixelBase + 1] = floatToHalf(fonE[idx]);     // G: FON E(NdotV, r)
                    dst[pixelBase + 2] = floatToHalf(ggxEavg[y]);    // B: GGX E_avg(alpha)
                    dst[pixelBase + 3] = floatToHalf(fonEavg[y]);    // A: FON E_avg(r)
                }
            }

            log::info("scene-res", "Energy compensation LUT baked (" +
                      std::to_string(LUT_SIZE) + "x" + std::to_string(LUT_SIZE) +
                      ", GGX=" + std::to_string(GGX_SAMPLES) +
                      " FON=" + std::to_string(FON_SAMPLES) + " samples)");
        }
    }

    // Single-instance stub SSBO (blasOffsets — never rewritten per frame).
    {
        auto r = createHostSSBO(64);
        if (!r) return vk2::makeError(r.error().vkResult, "Stub SSBO (blasOffsets): " + r.error().message);
        blasOffsetsSSBO_ = std::move(r.value());
        if (blasOffsetsSSBO_.mappedPtr())
            std::memset(blasOffsetsSSBO_.mappedPtr(), 0, 64);
    }

    // Per-frame-slot double-buffered stub SSBOs. Each vector is sized to framesInFlight_
    // so that frame N writes to slot N without racing the GPU reading slot N-1.
    struct SlottedSpec {
        std::vector<vk2::Buffer>* vec;
        const char* name;
    };
    SlottedSpec slottedSpecs[] = {
        {&textureMappingSSBOs_,  "textureMappingSSBO"},
        {&areaLightSSBOs_,       "areaLightSSBO"},
        {&tileLightSSBOs_,       "tileLightSSBO"},
        {&spriteRegistrySSBOs_,  "spriteRegistrySSBO"},
        {&blenderPbrSSBOs_,      "blenderPbrSSBO"},
        {&displacedFaceSSBOs_,   "displacedFaceSSBO"},
        {&biomeColorSSBOs_,      "biomeColorSSBO"},
    };
    for (auto& spec : slottedSpecs) {
        spec.vec->clear();
        spec.vec->reserve(framesInFlight_);
        for (uint32_t slot = 0; slot < framesInFlight_; ++slot) {
            auto r = createHostSSBO(64);
            if (!r) return vk2::makeError(r.error().vkResult,
                         std::string("Stub SSBO (") + spec.name + "[" + std::to_string(slot) + "]): " + r.error().message);
            if (r.value().mappedPtr())
                std::memset(r.value().mappedPtr(), 0, 64);
            spec.vec->push_back(std::move(r.value()));
        }
    }

    // A3: Material class SSBO — 512 entries × 144 bytes = 73728 bytes.
    // Uses the canonical vk::Data::MaterialClassEntry (shared.hpp, std430, 144 bytes).
    // The GLSL shader reads materialClassMapping.entries[materialClassIdx] via BDA using
    // this exact layout. The prior 16-byte stub caused a stride mismatch: for idx >= 29
    // the shader read 29×144 = 4176 bytes in, past the end of the old 4096-byte buffer,
    // producing a GPU page fault (type=6, instruction-fetch) → DEVICE_LOST.
    {
        auto r = createHostSSBO(MATERIAL_CLASS_SSBO_SIZE);
        if (!r) return vk2::makeError(r.error().vkResult, "Material class SSBO: " + r.error().message);
        materialClassSSBO_ = std::move(r.value());

        void* rawPtr = materialClassSSBO_.mappedPtr();
        if (rawPtr) {
            // Zero the entire buffer first — all fields default to 0.
            std::memset(rawPtr, 0, static_cast<size_t>(MATERIAL_CLASS_SSBO_SIZE));

            auto* entries = static_cast<MaterialClassEntry*>(rawPtr);

            // Fill all entries with safe defaults (roughness=0.5, ior=1.5, noiseLacunarity=2.0).
            // The shader multiplies roughness by itself for GGX alpha, so 0.5 → alpha=0.25 (matte).
            for (uint32_t i = 0; i < MATERIAL_CLASS_COUNT; ++i) {
                entries[i].roughness       = 0.5f;
                entries[i].ior             = 1.5f;
                entries[i].noiseLacunarity = 2.0f;
            }

            // These legacy simple-material entries (0-11) are kept for backward compatibility
            // with materialType values that existed before the full MaterialClassEntry was used.

            // Entry 0: air / transparent (ior=1.0, not is_glass)
            entries[0].roughness = 0.0f;
            entries[0].ior       = 1.0f;

            // Entry 1: glass (ior=1.5, is_glass flag=bit1)
            entries[1].roughness    = 0.0f;
            entries[1].ior          = 1.5f;
            entries[1].transmission = 1.0f;
            entries[1].flags        = 0; // flags bit meaning differs from old stub; leave 0

            // Entry 2: water (ior=1.333)
            entries[2].roughness    = 0.0f;
            entries[2].ior          = 1.333f;
            entries[2].transmission = 1.0f;

            // Entry 3: diamond (ior=2.42)
            entries[3].roughness    = 0.0f;
            entries[3].ior          = 2.42f;
            entries[3].transmission = 1.0f;

            // Entries 4-11: metals (high reflectance, near-zero roughness)
            constexpr float metalIORs[] = {
                2.950f,  // 4: iron
                0.470f,  // 5: gold
                0.270f,  // 6: copper
                0.130f,  // 7: aluminum
                2.330f,  // 8: chromium
                1.910f,  // 9: nickel
                0.150f,  // 10: silver
                2.160f,  // 11: titanium
            };
            for (uint32_t i = 0; i < 8; ++i) {
                entries[4 + i].roughness = 0.05f;
                entries[4 + i].metallic  = 1.0f;
                entries[4 + i].ior       = metalIORs[i];
            }

            log::info("scene-res", "Material class SSBO initialized: " +
                      std::to_string(MATERIAL_CLASS_COUNT) + " entries × " +
                      std::to_string(sizeof(MaterialClassEntry)) + " bytes = " +
                      std::to_string(MATERIAL_CLASS_SSBO_SIZE) + " bytes");
        }
    }

    // D6: Tile light buffer — 240x135 tiles (16x16 px each, covers up to 3840x2160).
    // Per-tile: { uint lightCount; uint lightIndices[32]; } = 132 bytes.
    // Double-buffered: one slot per frame-in-flight. Zero-initialized — RT shaders fall
    // back to iterating all lights until a proper compute culling pass fills this buffer.
    {
        tileLightSSBOs_.clear();
        tileLightSSBOs_.reserve(framesInFlight_);
        for (uint32_t slot = 0; slot < framesInFlight_; ++slot) {
            auto r = createHostSSBO(TILE_LIGHT_SSBO_SIZE);
            if (!r) return vk2::makeError(r.error().vkResult,
                         "Tile light SSBO[" + std::to_string(slot) + "]: " + r.error().message);
            if (r.value().mappedPtr())
                std::memset(r.value().mappedPtr(), 0, static_cast<size_t>(TILE_LIGHT_SSBO_SIZE));
            tileLightSSBOs_.push_back(std::move(r.value()));
        }

        log::info("scene-res", "Tile light SSBO initialized: " +
                  std::to_string(MAX_TILES_X) + "x" + std::to_string(MAX_TILES_Y) +
                  " tiles (" + std::to_string(TILE_LIGHT_SSBO_SIZE) + " bytes, " +
                  std::to_string(framesInFlight_) + " slots)");
    }

    // SHARC radiance cache buffers — device-local with BDA support.
    // Sizes per NVIDIA SHARC SDK:
    //   hashEntries: uint64_t per entry (8 bytes)
    //   accumulation: SharcAccumulationData = 16 bytes per entry
    //   resolved: SharcPackedData = 8 bytes per entry
    {
        constexpr uint32_t cap = SHARC_CAPACITY;
        struct SharcBufSpec { vk2::Buffer* dst; VkDeviceSize entrySize; const char* name; };
        SharcBufSpec specs[] = {
            {&sharcHashEntries_,  8,  "SHARC hashEntries"},
            {&sharcAccumulation_, 16, "SHARC accumulation"},
            {&sharcResolved_,     8,  "SHARC resolved"},
        };
        for (auto& spec : specs) {
            vk2::Buffer::Desc bd{};
            bd.size = static_cast<VkDeviceSize>(cap) * spec.entrySize;
            bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                     | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                     | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            auto r = vk2::Buffer::create(device_->device(), device_->vma(), bd);
            if (!r) return vk2::makeError(r.error().vkResult, std::string(spec.name) + ": " + r.error().message);
            *spec.dst = std::move(r.value());
        }
        log::info("scene-res", "SHARC buffers allocated: capacity=" + std::to_string(cap)
                  + " hash=" + std::to_string(sharcHashEntries_.deviceAddress())
                  + " accum=" + std::to_string(sharcAccumulation_.deviceAddress())
                  + " resolved=" + std::to_string(sharcResolved_.deviceAddress()));
    }

    // Linear sampler for energy LUT and other sampled images
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device_->device(), &si, nullptr, &linearSampler_) != VK_SUCCESS) {
            return vk2::makeError(VK_ERROR_INITIALIZATION_FAILED, "linear sampler");
        }
    }

    initialized_ = true;
    log::info("scene-res", "SceneResourceService initialized: " +
              std::to_string(framesInFlight_) + " frames, " +
              std::to_string(sizeof(WorldUBOData)) + "B WorldUBO, " +
              "hot SSBOs double-buffered x" + std::to_string(framesInFlight_));
    return {};
}

void SceneResourceService::shutdown() {
    if (!initialized_) return;

    if (linearSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_->device(), linearSampler_, nullptr);
        linearSampler_ = VK_NULL_HANDLE;
    }

    worldUBOs_.clear();
    lastWorldUBOs_.clear();
    skyUBOs_.clear();
    blueNoiseSobol_ = {};
    blueNoiseScrambling_ = {};
    energyLUT_ = {};
    energyLUTStaging_ = {};
    blasOffsetsSSBO_ = {};
    textureMappingSSBOs_.clear();
    materialClassSSBO_ = {};
    areaLightSSBOs_.clear();
    tileLightSSBOs_.clear();
    spriteRegistrySSBOs_.clear();
    blenderPbrSSBOs_.clear();
    displacedFaceSSBOs_.clear();
    biomeColorSSBOs_.clear();
    sharcHashEntries_ = {};
    sharcAccumulation_ = {};
    sharcResolved_ = {};

    initialized_ = false;
    device_ = nullptr;
}

bool SceneResourceService::runDeferredInit(VkCommandBuffer cmd) {
    if (!initialized_ || !firstFrameInitPending_) return false;

    // 1. Transition energyLUT_ UNDEFINED -> TRANSFER_DST_OPTIMAL
    {
        VkImageMemoryBarrier2 bar{};
        bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        bar.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        bar.srcAccessMask = 0;
        bar.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        bar.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.image = energyLUT_.handle();
        bar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bar.subresourceRange.baseMipLevel = 0;
        bar.subresourceRange.levelCount = 1;
        bar.subresourceRange.baseArrayLayer = 0;
        bar.subresourceRange.layerCount = 1;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &bar;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // 2. Copy staging -> LUT
    {
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {ENERGY_LUT_DIM, ENERGY_LUT_DIM, 1};
        vkCmdCopyBufferToImage(cmd, energyLUTStaging_.handle(), energyLUT_.handle(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }

    // 3. Transition energyLUT_ TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    {
        VkImageMemoryBarrier2 bar{};
        bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        bar.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        bar.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        bar.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                         | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                         | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        bar.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        bar.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.image = energyLUT_.handle();
        bar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bar.subresourceRange.baseMipLevel = 0;
        bar.subresourceRange.levelCount = 1;
        bar.subresourceRange.baseArrayLayer = 0;
        bar.subresourceRange.layerCount = 1;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &bar;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    firstFrameInitPending_ = false;
    log::info("scene-res", "Energy LUT uploaded via deferred GPU copy");
    return true;
}

void SceneResourceService::updateWorldUBO(uint32_t frameIndex, const CameraData& camera,
                                          const ConfigSnapshot& config, uint64_t frameNumber) {
    if (!initialized_ || frameIndex >= worldUBOs_.size()) return;

    // Snapshot current → last (for temporal reprojection)
    void* lastDst = lastWorldUBOs_[frameIndex].mappedPtr();
    void* curSrc = worldUBOs_[frameIndex].mappedPtr();
    if (lastDst && curSrc) {
        std::memcpy(lastDst, curSrc, sizeof(WorldUBOData));
        lastWorldUBOs_[frameIndex].flush();
    }

    // Build new WorldUBO
    WorldUBOData ubo{};
    std::memcpy(ubo.cameraViewMat, camera.view, sizeof(ubo.cameraViewMat));
    std::memcpy(ubo.cameraEffectedViewMat, camera.view, sizeof(ubo.cameraEffectedViewMat));

    // Apply sub-pixel jitter to the projection matrix.
    // The jitter is in pixel-space [-0.5, 0.5]; convert to NDC by dividing
    // by the render resolution (extracted from the projection matrix diagonal).
    // Column-major: proj[0]=m00, proj[5]=m11.
    float jitteredProj[16];
    std::memcpy(jitteredProj, camera.projection, sizeof(jitteredProj));
    if (jitterX_ != 0.0f || jitterY_ != 0.0f) {
        // proj[8] = m20 (column 2, row 0), proj[9] = m21 (column 2, row 1)
        // Adding jitter * 2/res is equivalent to shifting the NDC center.
        // For a symmetric projection: m00 = 2n/w, so pixel-to-NDC = m00/W = 2/W_pixels.
        // But DLSS expects us to offset the projection center directly:
        //   proj[2][0] += 2*jitterX / renderWidth
        //   proj[2][1] += 2*jitterY / renderHeight
        // Since we don't have renderWidth/Height here, we use the standard identity:
        //   ndcJitterX = 2 * jitterX_pixels * proj[0] / renderWidth
        // which simplifies to: proj[8] += jitterX * 2.0 / renderWidth
        // However, for ray tracing the jitter is applied via the projection inverse,
        // so the simplest correct approach is to offset m[2][0] and m[2][1] directly.
        // These map to column-major indices [8] and [9].
        jitteredProj[8] += jitterX_ * 2.0f;   // will be divided by resolution in shader
        jitteredProj[9] += jitterY_ * 2.0f;
    }
    std::memcpy(ubo.cameraProjMat, jitteredProj, sizeof(ubo.cameraProjMat));

    // Compute inverses for ray generation (test_raygen and PR39 real shaders read these)
    invertMatrix4(camera.view, ubo.cameraViewMatInv);
    invertMatrix4(camera.view, ubo.cameraEffectedViewMatInv);
    invertMatrix4(jitteredProj, ubo.cameraProjMatInv);

    // Store jitter in WorldUBO for shaders that need the raw pixel-space offset.
    ubo.cameraJitter[0] = jitterX_;
    ubo.cameraJitter[1] = jitterY_;

    ubo.gameTime = static_cast<float>(frameNumber) / 60.0f;
    ubo.seed = static_cast<uint32_t>(frameNumber);
    ubo.rayBounces = static_cast<uint32_t>(config.data.rayBounces);
    ubo.cameraTmin = 0.001f;
    ubo.cameraPos[0] = camera.posX;
    ubo.cameraPos[1] = camera.posY;
    ubo.cameraPos[2] = camera.posZ;
    ubo.cameraPos[3] = 0.0;
    ubo.animTick = static_cast<uint32_t>(camera.gameTick);

    // Identity texture matrix
    ubo.textureMat[0] = ubo.textureMat[5] = ubo.textureMat[10] = ubo.textureMat[15] = 1.0f;

    // Emission data: use uploaded values if available, otherwise neutral defaults.
    // When hasEmissionData_ is false, emissionData_ is all-zeros which means
    // all emission multipliers are 0 — but that's correct because V2 hasn't received
    // the emission table from Java yet. Once uploadEmissionData() is called, the
    // real values flow through.
    if (hasEmissionData_) {
        std::memcpy(ubo.emissionData, emissionData_, sizeof(ubo.emissionData));
        std::memcpy(ubo.emissiveGamut, emissiveGamut_, sizeof(ubo.emissiveGamut));
    } else {
        // Neutral emissive gamut (1.0) — safe default before Java sends real data
        for (int i = 0; i < 13 * 4; ++i) ubo.emissiveGamut[i] = 1.0f;
    }

    // Write to mapped buffer
    if (curSrc) {
        std::memcpy(curSrc, &ubo, sizeof(ubo));
        worldUBOs_[frameIndex].flush(); // Ensure GPU-visible on non-coherent heaps
    }
}

void SceneResourceService::updateSkyUBO(const SkyUBOData& sky) {
    latestSky_ = sky;

    // Hardcoded atmosphere constants (matches V1 buffers.cpp:381-396).
    // These are physical constants that don't change at runtime — the bridge
    // command only sends dynamic sky state (colors, directions, weather).
    latestSky_.Rg = 6360000.0f;
    latestSky_.Rt = 6460000.0f;
    latestSky_.Hr = 8000.0f;
    latestSky_.Hm = 1200.0f;
    latestSky_.mieG = 0.80f;
    latestSky_.betaR[0] = 5.802e-6f;
    latestSky_.betaR[1] = 13.558e-6f;
    latestSky_.betaR[2] = 33.100e-6f;
    latestSky_.betaM[0] = 21.000e-6f;
    latestSky_.betaM[1] = 21.000e-6f;
    latestSky_.betaM[2] = 21.000e-6f;
    latestSky_.minViewCos = 0.02f;
    latestSky_.sunRadiance[0] = 100000.0f;
    latestSky_.sunRadiance[1] = 100000.0f;
    latestSky_.sunRadiance[2] = 100000.0f;
    latestSky_.moonRadiance[0] = 0.05f;
    latestSky_.moonRadiance[1] = 0.06f;
    latestSky_.moonRadiance[2] = 0.12f;
    latestSky_.hdrRadianceScale = 1.0f;

    skyDirty_ = true;
    // Write to all per-frame buffers (cheap — small data)
    for (auto& buf : skyUBOs_) {
        if (buf.mappedPtr()) {
            std::memcpy(buf.mappedPtr(), &latestSky_, sizeof(SkyUBOData));
            buf.flush();
        }
    }
}

void SceneResourceService::uploadTextureMapping(const void* data, size_t size) {
    if (!initialized_ || !data || size == 0) return;

    // Reallocate ALL per-frame-slot SSBOs if the size changed.
    // Since bridge().flush() is called right after the slot fence, at most one slot
    // is in-flight on the GPU while we write the others. By keeping all slots at the
    // same size we avoid the WAR hazard: each slot holds an independent copy.
    bool needRealloc = (textureMappingSSBOs_.size() != framesInFlight_) ||
                       (!textureMappingSSBOs_.empty() &&
                        textureMappingSSBOs_[0].size() != static_cast<VkDeviceSize>(size));
    if (needRealloc) {
        for (auto& buf : textureMappingSSBOs_) {
            if (gc_ && buf.handle()) {
                auto old = std::make_shared<vk2::Buffer>(std::move(buf));
                gc_->defer([old]() {});
            }
        }
        textureMappingSSBOs_.clear();
        textureMappingSSBOs_.reserve(framesInFlight_);
        for (uint32_t slot = 0; slot < framesInFlight_; ++slot) {
            auto r = createHostSSBO(static_cast<VkDeviceSize>(size));
            if (!r) {
                log::warn("scene-res", "Texture mapping SSBO[" + std::to_string(slot) +
                          "] realloc failed: " + r.error().message);
                return;
            }
            textureMappingSSBOs_.push_back(std::move(r.value()));
        }
        log::info("scene-res", "Texture mapping SSBO resized to " + std::to_string(size) +
                  " bytes (" + std::to_string(framesInFlight_) + " slots)");
    }

    // Write the same data to every slot so each frame index gets a consistent view.
    for (auto& buf : textureMappingSSBOs_) {
        void* dst = buf.mappedPtr();
        if (dst) std::memcpy(dst, data, size);
        buf.flush(); // Ensure GPU-visible on non-coherent heaps
    }
}

void SceneResourceService::uploadEmissionData(const float* emissionData, const float* gamutData) {
    if (!initialized_) return;
    if (emissionData) std::memcpy(emissionData_, emissionData, sizeof(emissionData_));
    if (gamutData) std::memcpy(emissiveGamut_, gamutData, sizeof(emissiveGamut_));
    hasEmissionData_ = true;
    log::debug("scene-res", "Emission data uploaded (50 blocks + 13 gamut vec4s)");
}

void SceneResourceService::uploadAreaLights(const void* data, size_t size, uint32_t lightCount) {
    if (!initialized_) return;

    areaLightCount_ = lightCount;

    // Empty upload — keep the stub SSBOs valid but set count to 0.
    if (!data || size == 0 || lightCount == 0) {
        areaLightCount_ = 0;
        return;
    }

    // Reallocate all per-frame-slot SSBOs if size changed (same WAR-avoidance as texture mapping).
    bool needRealloc = (areaLightSSBOs_.size() != framesInFlight_) ||
                       (!areaLightSSBOs_.empty() &&
                        areaLightSSBOs_[0].size() != static_cast<VkDeviceSize>(size));
    if (needRealloc) {
        for (auto& buf : areaLightSSBOs_) {
            if (gc_ && buf.handle()) {
                auto old = std::make_shared<vk2::Buffer>(std::move(buf));
                gc_->defer([old]() {});
            }
        }
        areaLightSSBOs_.clear();
        areaLightSSBOs_.reserve(framesInFlight_);
        for (uint32_t slot = 0; slot < framesInFlight_; ++slot) {
            auto r = createHostSSBO(static_cast<VkDeviceSize>(size));
            if (!r) {
                log::warn("scene-res", "Area light SSBO[" + std::to_string(slot) +
                          "] realloc failed: " + r.error().message);
                areaLightCount_ = 0;
                return;
            }
            areaLightSSBOs_.push_back(std::move(r.value()));
        }
        log::info("scene-res", "Area light SSBO resized to " + std::to_string(size)
                  + " bytes (" + std::to_string(lightCount) + " lights, "
                  + std::to_string(framesInFlight_) + " slots)");
    }

    // Write to every slot — identical data, each frame reads its own independent copy.
    for (auto& buf : areaLightSSBOs_) {
        void* dst = buf.mappedPtr();
        if (dst) std::memcpy(dst, data, size);
        buf.flush(); // Ensure GPU-visible on non-coherent heaps
    }
}
void SceneResourceService::resizeBiomeColors(uint32_t minInstances) {
    if (!initialized_ || minInstances == 0) return;

    // Already large enough — nothing to do.
    if (minInstances <= biomeColorCapacity_) return;

    // Round up to the next power of 2 (or at least 64) so we don't realloc every frame
    // during a world-load chunk storm.
    uint32_t newCap = 64;
    while (newCap < minInstances) newCap <<= 1;

    VkDeviceSize newSize = static_cast<VkDeviceSize>(newCap) * 16; // uvec4 = 16 bytes

    // Allocate all new per-frame-slot buffers into a temporary vector first.
    // Only replace the live vector on full success — this way a partial allocation
    // failure (OOM) leaves biomeColorSSBOs_ intact and fully sized, preventing the
    // caller from seeing an undersized vector and taking an OOB operator[] access.
    std::vector<vk2::Buffer> newSlots;
    newSlots.reserve(framesInFlight_);

    for (uint32_t slot = 0; slot < framesInFlight_; ++slot) {
        auto r = createHostSSBO(newSize);
        if (!r) {
            log::warn("scene-res", "biomeColorSSBO[" + std::to_string(slot) + "] grow failed (" +
                      std::to_string(newSize) + " bytes): " + r.error().message);
            // Discard the partial temp allocation; keep the existing live vector unchanged.
            return;
        }
        // Zero-fill: no biome tint applied (correct neutral default).
        if (r.value().mappedPtr())
            std::memset(r.value().mappedPtr(), 0, static_cast<size_t>(newSize));
        newSlots.push_back(std::move(r.value()));
    }

    // All slots allocated successfully — retire the old buffers via GC and swap in the new ones.
    for (auto& buf : biomeColorSSBOs_) {
        if (gc_ && buf.handle()) {
            auto old = std::make_shared<vk2::Buffer>(std::move(buf));
            gc_->defer([old]() {});
        }
    }
    biomeColorSSBOs_ = std::move(newSlots);
    biomeColorCapacity_ = newCap;
    log::info("scene-res", "biomeColorSSBOs grown to " + std::to_string(newCap) +
              " instances (" + std::to_string(newSize) + " bytes, " +
              std::to_string(framesInFlight_) + " slots)");
}

} // namespace engine
