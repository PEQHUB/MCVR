#include "gpu_upload_service.hpp"
#include "chunk_registry.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <cstring>

namespace engine {

vk2::Result<void> GpuUploadService::init(vk2::DeviceService& device, uint32_t framesInFlight,
                                         VkDeviceSize stagingSize) {
    device_ = &device;
    stagingSize_ = stagingSize;
    framesInFlight_ = std::max(framesInFlight, 1u);
    slotSize_ = stagingSize_ / framesInFlight_;

    // Create staging buffer (host-visible, persistently mapped)
    vk2::Buffer::Desc bd{};
    bd.size = stagingSize;
    bd.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    auto r = vk2::Buffer::create(device.device(), device.vma(), bd);
    if (!r) return vk2::makeError(r.error().vkResult, "Staging buffer: " + r.error().message);
    stagingBuffer_ = std::move(r.value());

    initialized_ = true;
    log::info("gpu-upload", "Initialized with " + std::to_string(stagingSize / (1024 * 1024)) + " MB staging");
    return {};
}

void GpuUploadService::shutdown() {
    chunks_.clear();
    stagingBuffer_ = {};
    initialized_ = false;
}

uint32_t GpuUploadService::uploadDirtyChunks(VkCommandBuffer cmd,
                                              const std::vector<ChunkGeometry>& dirtyChunks,
                                              uint32_t frameIndex) {
    if (!initialized_ || dirtyChunks.empty()) return 0;

    resetStaging(frameIndex);
    uint32_t uploaded = 0;
    bool useBDA = device_->caps().bufferDeviceAddress;

    for (const auto& geo : dirtyChunks) {
        if (geo.empty()) {
            // Empty chunk — remove existing GPU data
            chunks_.erase(geo.id);
            continue;
        }

        VkDeviceSize vertSize = geo.vertexData.size();
        VkDeviceSize idxSize = geo.indexData.size() * sizeof(uint32_t);

        // Allocate staging space
        VkDeviceSize vertStaging = allocStaging(vertSize);
        VkDeviceSize idxStaging = allocStaging(idxSize);
        if (vertStaging == UINT64_MAX || idxStaging == UINT64_MAX) {
            log::warn("gpu-upload", "Staging buffer full, skipping remaining chunks");
            break;
        }

        // Copy vertex data to staging
        auto* staging = static_cast<uint8_t*>(stagingBuffer_.mappedPtr());
        std::memcpy(staging + vertStaging, geo.vertexData.data(), vertSize);
        std::memcpy(staging + idxStaging, geo.indexData.data(), idxSize);

        // Create device-local vertex buffer
        VkBufferUsageFlags vertUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        if (useBDA) vertUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        vk2::Buffer::Desc vbd{};
        vbd.size = vertSize;
        vbd.usage = vertUsage;
        auto vertR = vk2::Buffer::create(device_->device(), device_->vma(), vbd);
        if (!vertR) { log::warn("gpu-upload", "Vertex buffer failed"); continue; }

        // Create device-local index buffer
        VkBufferUsageFlags idxUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
            | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        if (useBDA) idxUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        vk2::Buffer::Desc ibd{};
        ibd.size = idxSize;
        ibd.usage = idxUsage;
        auto idxR = vk2::Buffer::create(device_->device(), device_->vma(), ibd);
        if (!idxR) { log::warn("gpu-upload", "Index buffer failed"); continue; }

        // Record copy commands
        VkBufferCopy vertCopy{vertStaging, 0, vertSize};
        vkCmdCopyBuffer(cmd, stagingBuffer_.handle(), vertR.value().handle(), 1, &vertCopy);

        VkBufferCopy idxCopy{idxStaging, 0, idxSize};
        vkCmdCopyBuffer(cmd, stagingBuffer_.handle(), idxR.value().handle(), 1, &idxCopy);

        // Store GPU data
        GpuChunkData data;
        data.triangleCount = geo.triangleCount;
        data.indexCount = static_cast<uint32_t>(geo.indexData.size());
        data.revision = geo.revision;
        data.originX = geo.originX;
        data.originY = geo.originY;
        data.originZ = geo.originZ;
        data.vertexAddress = vertR.value().deviceAddress();
        data.indexAddress = idxR.value().deviceAddress();
        data.vertexBuffer = std::move(vertR.value());
        data.indexBuffer = std::move(idxR.value());

        chunks_[geo.id] = std::move(data);
        ++uploaded;
    }

    return uploaded;
}

void GpuUploadService::removeChunk(ChunkId id) {
    chunks_.erase(id);
}

const GpuChunkData* GpuUploadService::getChunk(ChunkId id) const {
    auto it = chunks_.find(id);
    return it != chunks_.end() ? &it->second : nullptr;
}

VkDeviceSize GpuUploadService::allocStaging(VkDeviceSize size, VkDeviceSize alignment) {
    // Align up
    VkDeviceSize aligned = (stagingOffset_ + alignment - 1) & ~(alignment - 1);
    if (aligned + size > slotBase_ + slotSize_) return UINT64_MAX;
    stagingOffset_ = aligned + size;
    return aligned;
}

void GpuUploadService::resetStaging(uint32_t frameIndex) {
    slotBase_ = (frameIndex % framesInFlight_) * slotSize_;
    stagingOffset_ = slotBase_;
}

} // namespace engine
