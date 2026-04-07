#include "blas_service.hpp"
#include "gpu_upload_service.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>

namespace engine {

BlasService::~BlasService() { shutdown(); }

vk2::Result<void> BlasService::init(vk2::DeviceService& device, VkDeviceSize scratchSize) {
    device_ = &device;
    scratchSize_ = scratchSize;

    if (!device.caps().accelerationStructure) {
        log::warn("blas", "Acceleration structures not supported — BLAS service disabled");
        return {};
    }

    // Create scratch buffer
    vk2::Buffer::Desc bd{};
    bd.size = scratchSize;
    bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bd.minAlignment = 256; // AS scratch alignment requirement

    auto r = vk2::Buffer::create(device.device(), device.vma(), bd);
    if (!r) return vk2::makeError(r.error().vkResult, "BLAS scratch buffer: " + r.error().message);
    scratchBuffer_ = std::move(r.value());
    scratchAddress_ = scratchBuffer_.deviceAddress();

    initialized_ = true;
    log::info("blas", "Initialized with " + std::to_string(scratchSize / (1024 * 1024)) + " MB scratch");
    return {};
}

void BlasService::shutdown() {
    for (auto& [id, data] : blas_) destroyBlas(data);
    blas_.clear();
    scratchBuffer_ = {};
    initialized_ = false;
}

uint32_t BlasService::buildDirtyChunks(VkCommandBuffer cmd,
                                        const std::vector<ChunkId>& dirtyIds,
                                        const std::unordered_map<ChunkId, GpuChunkData>& gpuChunks) {
    if (!initialized_ || !device_->caps().accelerationStructure) return 0;

    uint32_t built = 0;
    VkDeviceSize scratchOffset = 0;

    for (const auto& chunkId : dirtyIds) {
        auto it = gpuChunks.find(chunkId);
        if (it == gpuChunks.end() || it->second.triangleCount == 0) continue;

        const auto& gpu = it->second;

        // Geometry description
        VkAccelerationStructureGeometryKHR geometry{};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geometry.geometry.triangles.vertexData.deviceAddress = gpu.vertexAddress;
        geometry.geometry.triangles.vertexStride = 96; // PBRTriangle stride
        geometry.geometry.triangles.maxVertex = gpu.triangleCount * 3 - 1;
        geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
        geometry.geometry.triangles.indexData.deviceAddress = gpu.indexAddress;

        // Build info
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geometry;

        // Query size requirements
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
        sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        uint32_t primCount = gpu.triangleCount;
        vkGetAccelerationStructureBuildSizesKHR(device_->device(),
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primCount, &sizeInfo);

        // Check scratch fits
        VkDeviceSize alignedScratch = (scratchOffset + 255) & ~255ULL;
        if (alignedScratch + sizeInfo.buildScratchSize > scratchSize_) {
            log::warn("blas", "Scratch buffer full, skipping remaining chunks");
            break;
        }

        // Create AS buffer
        vk2::Buffer::Desc asBd{};
        asBd.size = sizeInfo.accelerationStructureSize;
        asBd.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                   | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        auto asR = vk2::Buffer::create(device_->device(), device_->vma(), asBd);
        if (!asR) { log::warn("blas", "AS buffer alloc failed"); continue; }

        // Create AS
        VkAccelerationStructureCreateInfoKHR asCI{};
        asCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        asCI.buffer = asR.value().handle();
        asCI.size = sizeInfo.accelerationStructureSize;
        asCI.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

        VkAccelerationStructureKHR as = VK_NULL_HANDLE;
        if (vkCreateAccelerationStructureKHR(device_->device(), &asCI, nullptr, &as) != VK_SUCCESS) {
            log::warn("blas", "vkCreateAccelerationStructureKHR failed");
            continue;
        }

        // Get device address
        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addrInfo.accelerationStructure = as;
        VkDeviceAddress asAddr = vkGetAccelerationStructureDeviceAddressKHR(device_->device(), &addrInfo);

        // Build
        buildInfo.dstAccelerationStructure = as;
        buildInfo.scratchData.deviceAddress = scratchAddress_ + alignedScratch;

        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
        rangeInfo.primitiveCount = primCount;
        const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;

        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);

        scratchOffset = alignedScratch + sizeInfo.buildScratchSize;

        // Destroy old BLAS if exists
        auto existing = blas_.find(chunkId);
        if (existing != blas_.end()) destroyBlas(existing->second);

        // Store
        BlasData data;
        data.handle = as;
        data.buffer = std::move(asR.value());
        data.address = asAddr;
        data.vertexAddress = gpu.vertexAddress;
        data.indexAddress = gpu.indexAddress;
        data.revision = gpu.revision;
        blas_[chunkId] = std::move(data);
        ++built;
    }

    // Memory barrier after all builds
    if (built > 0) {
        VkMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
                             | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    return built;
}

void BlasService::removeChunk(ChunkId id) {
    auto it = blas_.find(id);
    if (it != blas_.end()) {
        destroyBlas(it->second);
        blas_.erase(it);
    }
}

const BlasData* BlasService::getBlas(ChunkId id) const {
    auto it = blas_.find(id);
    return it != blas_.end() ? &it->second : nullptr;
}

void BlasService::destroyBlas(BlasData& data) {
    if (data.handle != VK_NULL_HANDLE && device_) {
        vkDestroyAccelerationStructureKHR(device_->device(), data.handle, nullptr);
        data.handle = VK_NULL_HANDLE;
    }
}

} // namespace engine
