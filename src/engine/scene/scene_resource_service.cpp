#include "scene_resource_service.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "frame/frame_context.hpp"
#include "config/config_service.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <cmath>
#include <cstring>

namespace {

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
    bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    return vk2::Buffer::create(device_->device(), device_->vma(), bd);
}

vk2::Result<vk2::Buffer> SceneResourceService::createHostSSBO(VkDeviceSize size) {
    vk2::Buffer::Desc bd{};
    bd.size = size;
    bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    return vk2::Buffer::create(device_->device(), device_->vma(), bd);
}

vk2::Result<void> SceneResourceService::init(vk2::DeviceService& device, uint32_t framesInFlight) {
    device_ = &device;

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

    // Blue noise buffers (host-visible, zeroed for now — real data comes via bridge later)
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
            std::memset(blueNoiseSobol_.mappedPtr(), 0, BLUE_NOISE_SOBOL_SIZE);
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
            std::memset(blueNoiseScrambling_.mappedPtr(), 0, BLUE_NOISE_SCRAMBLING_SIZE);
        }
    }

    // Energy LUT (64x64 RGBA16F, sampled). Initialized to all-1.0 (no compensation).
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

    // Stub SSBOs — 64 bytes each (small placeholder; real sizes come via bridge)
    for (auto* p : {&textureMappingSSBO_, &materialClassSSBO_, &areaLightSSBO_,
                    &tileLightSSBO_, &spriteRegistrySSBO_, &blenderPbrSSBO_,
                    &displacedFaceSSBO_}) {
        auto r = createDeviceSSBO(64);
        if (!r) return vk2::makeError(r.error().vkResult, "Stub SSBO: " + r.error().message);
        *p = std::move(r.value());
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
              std::to_string(framesInFlight) + " frames, " +
              std::to_string(sizeof(WorldUBOData)) + "B WorldUBO");
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
    textureMappingSSBO_ = {};
    materialClassSSBO_ = {};
    areaLightSSBO_ = {};
    tileLightSSBO_ = {};
    spriteRegistrySSBO_ = {};
    blenderPbrSSBO_ = {};
    displacedFaceSSBO_ = {};

    initialized_ = false;
    device_ = nullptr;
}

void SceneResourceService::updateWorldUBO(uint32_t frameIndex, const CameraData& camera,
                                          const ConfigSnapshot& config, uint64_t frameNumber) {
    if (!initialized_ || frameIndex >= worldUBOs_.size()) return;

    // Snapshot current → last (for temporal reprojection)
    void* lastDst = lastWorldUBOs_[frameIndex].mappedPtr();
    void* curSrc = worldUBOs_[frameIndex].mappedPtr();
    if (lastDst && curSrc) {
        std::memcpy(lastDst, curSrc, sizeof(WorldUBOData));
    }

    // Build new WorldUBO
    WorldUBOData ubo{};
    std::memcpy(ubo.cameraViewMat, camera.view, sizeof(ubo.cameraViewMat));
    std::memcpy(ubo.cameraEffectedViewMat, camera.view, sizeof(ubo.cameraEffectedViewMat));
    std::memcpy(ubo.cameraProjMat, camera.projection, sizeof(ubo.cameraProjMat));
    // Compute inverses for ray generation (test_raygen and PR39 real shaders read these)
    invertMatrix4(camera.view, ubo.cameraViewMatInv);
    invertMatrix4(camera.view, ubo.cameraEffectedViewMatInv);
    invertMatrix4(camera.projection, ubo.cameraProjMatInv);
    ubo.gameTime = static_cast<float>(frameNumber) / 60.0f;
    ubo.seed = static_cast<uint32_t>(frameNumber);
    ubo.rayBounces = static_cast<uint32_t>(config.data.rayBounces);
    ubo.cameraTmin = 0.001f;
    ubo.cameraPos[0] = camera.posX;
    ubo.cameraPos[1] = camera.posY;
    ubo.cameraPos[2] = camera.posZ;
    ubo.cameraPos[3] = 0.0;
    ubo.animTick = static_cast<uint32_t>(frameNumber);

    // Identity texture matrix
    ubo.textureMat[0] = ubo.textureMat[5] = ubo.textureMat[10] = ubo.textureMat[15] = 1.0f;

    // Neutral emissive gamut (1.0)
    for (int i = 0; i < 13 * 4; ++i) ubo.emissiveGamut[i] = 1.0f;

    // Write to mapped buffer
    if (curSrc) {
        std::memcpy(curSrc, &ubo, sizeof(ubo));
    }
}

void SceneResourceService::updateSkyUBO(const SkyUBOData& sky) {
    latestSky_ = sky;
    skyDirty_ = true;
    // Write to all per-frame buffers (cheap — small data)
    for (auto& buf : skyUBOs_) {
        if (buf.mappedPtr()) {
            std::memcpy(buf.mappedPtr(), &latestSky_, sizeof(SkyUBOData));
        }
    }
}

void SceneResourceService::uploadTextureMapping(const void* data, size_t size) {
    if (!initialized_ || !data || size == 0) return;

    // Reallocate if the existing SSBO is the wrong size (typical: stub 64B → 128KB on first call).
    // Host-visible so the CPU can write directly without staging — texture mapping is read-mostly.
    if (textureMappingSSBO_.size() != static_cast<VkDeviceSize>(size)) {
        auto r = createHostSSBO(static_cast<VkDeviceSize>(size));
        if (!r) {
            log::warn("scene-res", "Texture mapping SSBO realloc failed: " + r.error().message);
            return;
        }
        textureMappingSSBO_ = std::move(r.value());
        log::info("scene-res", "Texture mapping SSBO resized to " + std::to_string(size) + " bytes");
    }

    void* dst = textureMappingSSBO_.mappedPtr();
    if (dst) {
        std::memcpy(dst, data, size);
    }
}

} // namespace engine
