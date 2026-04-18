#include "entity_blas_service.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "frame/resource_gc.hpp"
#include "diagnostics/log.hpp"

#include <utility>
#include <cstring>
#include <volk.h>

namespace engine {

EntityBlasService::~EntityBlasService() { shutdown(); }

vk2::Result<void> EntityBlasService::init(vk2::DeviceService& device, ResourceGC* gc,
                                           VkDeviceSize scratchSize) {
    device_ = &device;
    gc_ = gc;
    scratchSize_ = scratchSize;

    if (!device.caps().accelerationStructure) {
        log::warn("entityBlas", "Acceleration structures not supported - EntityBlasService disabled");
        return {};
    }

    // Create scratch buffer
    vk2::Buffer::Desc bd{};
    bd.size = scratchSize;
    bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bd.minAlignment = 256;

    // Create double-buffered scratch buffers (one per frame-in-flight slot).
    // Frame N uses slot (N & 1); fence wait guarantees Frame N-2 is done.
    for (uint32_t s = 0; s < 2; ++s) {
        auto r = vk2::Buffer::create(device.device(), device.vma(), bd);
        if (!r) return vk2::makeError(r.error().vkResult,
            "Entity BLAS scratch buffer[" + std::to_string(s) + "]: " + r.error().message);
        scratchBuffers_[s] = std::move(r.value());
        scratchAddresses_[s] = scratchBuffers_[s].deviceAddress();
    }

    initialized_ = true;
    log::info("entityBlas", "Initialized with " + std::to_string(scratchSize / (1024 * 1024)) + " MB scratch");
    return {};
}

void EntityBlasService::shutdown() {
    destroyAllBlas();
    scratchBuffers_[0] = {};
    scratchBuffers_[1] = {};
    vertexBuffers_[0] = {}; vertexBuffers_[1] = {};
    indexBuffers_[0] = {};  indexBuffers_[1] = {};
    entityMotionBuffer_ = {};
    prevEntities_.clear();
    initialized_ = false;
    gc_ = nullptr;
}

VkBuffer EntityBlasService::entityMotionBuffer() const {
    return entityMotionBuffer_.handle();
}

VkDeviceSize EntityBlasService::entityMotionBufferSize() const {
    return entityMotionBuffer_.handle() ? entityMotionBuffer_.size() : 0;
}

void EntityBlasService::destroyAllBlas() {
    if (entities_.empty()) return;

    if (gc_) {
        // Deferred delete: keep handles/buffers alive until the GPU is done
        // reading them (ringSize frames later). Synchronous destroy here would
        // race with the previous frame slot still in-flight on the GPU.
        VkDevice dev = device_->device();
        for (auto& e : entities_) {
            if (e.handle == VK_NULL_HANDLE) continue;
            auto buf = std::make_shared<vk2::Buffer>(std::move(e.buffer));
            VkAccelerationStructureKHR handle = e.handle;
            e.handle = VK_NULL_HANDLE;
            gc_->defer([dev, handle, buf]() {
                vkDestroyAccelerationStructureKHR(dev, handle, nullptr);
                // buf shared_ptr releases here — vk2::Buffer dtor frees VMA allocation
            });
        }
    } else {
        // No GC wired — synchronous fallback (only safe if caller ensures GPU idle).
        for (auto& e : entities_) {
            if (e.handle != VK_NULL_HANDLE && device_) {
                vkDestroyAccelerationStructureKHR(device_->device(), e.handle, nullptr);
                e.handle = VK_NULL_HANDLE;
            }
        }
    }
    entities_.clear();
}

void EntityBlasService::clear() {
    destroyAllBlas();
}

uint32_t EntityBlasService::submitBatch(VkCommandBuffer cmd,
                                         std::vector<uint8_t> vertexData,
                                         std::vector<uint32_t> indexData,
                                         const std::vector<EntityBatchEntry>& entries,
                                         uint32_t frameIndex) {
    if (!initialized_ || !device_->caps().accelerationStructure) return 0;
    if (entries.empty() || vertexData.empty()) return 0;

    // Select the double-buffer slot for this frame.
    // The fence wait before this command buffer guarantees that this slot's
    // previous contents are no longer referenced by any in-flight GPU work.
    const uint32_t slot = frameIndex & 1;
    vk2::Buffer& vertexBuffer = vertexBuffers_[slot];
    vk2::Buffer& indexBuffer  = indexBuffers_[slot];

    // Upload vertex data into this frame's slot
    {
        VkDeviceSize vbSize = vertexData.size();
        if (!vertexBuffer.handle() || vertexBuffer.size() < vbSize) {
            vk2::Buffer::Desc bd{};
            bd.size = std::max(vbSize, VkDeviceSize(1024 * 1024)); // min 1MB
            bd.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                     | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                     | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                        | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            auto r = vk2::Buffer::create(device_->device(), device_->vma(), bd);
            if (!r) { log::error("entityBlas", "Vertex buffer alloc failed"); return 0; }
            vertexBuffer = std::move(r.value());
        }
        std::memcpy(vertexBuffer.mappedPtr(), vertexData.data(), vbSize);
        vertexBuffer.flush();
    }

    // Upload index data into this frame's slot
    {
        VkDeviceSize ibSize = indexData.size() * sizeof(uint32_t);
        if (!indexBuffer.handle() || indexBuffer.size() < ibSize) {
            vk2::Buffer::Desc bd{};
            bd.size = std::max(ibSize, VkDeviceSize(512 * 1024)); // min 512KB
            bd.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                     | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                     | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                        | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            auto r = vk2::Buffer::create(device_->device(), device_->vma(), bd);
            if (!r) { log::error("entityBlas", "Index buffer alloc failed"); return 0; }
            indexBuffer = std::move(r.value());
        }
        std::memcpy(indexBuffer.mappedPtr(), indexData.data(), ibSize);
        indexBuffer.flush();
    }

    VkDeviceAddress vbAddr = vertexBuffer.deviceAddress();
    VkDeviceAddress ibAddr = indexBuffer.deviceAddress();

    VkDeviceSize scratchOffset = 0;
    uint32_t built = 0;

    for (const auto& entry : entries) {
        if (entry.triangleCount == 0) continue;

        // Geometry description
        VkAccelerationStructureGeometryKHR geometry{};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geometry.geometry.triangles.vertexData.deviceAddress = vbAddr + entry.vertexOffset;
        geometry.geometry.triangles.vertexStride = 96; // PBRTriangle stride
        geometry.geometry.triangles.maxVertex = entry.triangleCount * 3 - 1;
        geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
        geometry.geometry.triangles.indexData.deviceAddress = ibAddr + entry.indexOffset * sizeof(uint32_t);

        // Build info - PREFER_FAST_BUILD for entities (rebuilt every frame)
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geometry;

        // Query size
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
        sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        uint32_t primCount = entry.triangleCount;
        vkGetAccelerationStructureBuildSizesKHR(device_->device(),
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primCount, &sizeInfo);

        // Check scratch fits
        VkDeviceSize alignedScratch = (scratchOffset + 255) & ~255ULL;
        if (alignedScratch + sizeInfo.buildScratchSize > scratchSize_) {
            log::warn("entityBlas", "Scratch exhausted at entity " + std::to_string(built)
                     + "/" + std::to_string(entries.size()));
            break;
        }

        // Create AS buffer
        vk2::Buffer::Desc asBd{};
        asBd.size = sizeInfo.accelerationStructureSize;
        asBd.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                   | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        auto asR = vk2::Buffer::create(device_->device(), device_->vma(), asBd);
        if (!asR) { log::warn("entityBlas", "AS buffer alloc failed"); continue; }

        // Create AS
        VkAccelerationStructureCreateInfoKHR asCI{};
        asCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        asCI.buffer = asR.value().handle();
        asCI.size = sizeInfo.accelerationStructureSize;
        asCI.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

        VkAccelerationStructureKHR as = VK_NULL_HANDLE;
        if (vkCreateAccelerationStructureKHR(device_->device(), &asCI, nullptr, &as) != VK_SUCCESS) {
            log::warn("entityBlas", "vkCreateAccelerationStructureKHR failed");
            continue;
        }

        // Get device address
        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addrInfo.accelerationStructure = as;
        VkDeviceAddress asAddr = vkGetAccelerationStructureDeviceAddressKHR(device_->device(), &addrInfo);

        // Build
        buildInfo.dstAccelerationStructure = as;
        buildInfo.scratchData.deviceAddress = scratchAddresses_[slot] + alignedScratch;

        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
        rangeInfo.primitiveCount = primCount;
        const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);

        scratchOffset = alignedScratch + sizeInfo.buildScratchSize;

        // Store
        EntityBlasData data;
        data.handle = as;
        data.buffer = std::move(asR.value());
        data.address = asAddr;
        data.vertexAddress = vbAddr + entry.vertexOffset;
        data.indexAddress = ibAddr + entry.indexOffset * sizeof(uint32_t);
        data.posX = entry.posX;
        data.posY = entry.posY;
        data.posZ = entry.posZ;
        data.rtFlag = entry.rtFlag;
        data.coordSystem = entry.coordSystem;
        data.hashCode = entry.hashCode;
        entities_.push_back(std::move(data));
        ++built;
    }

    // Build entity-motion SSBO and advance prev-frame tracking
    buildMotionBuffer();

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

void EntityBlasService::buildMotionBuffer() {
    const uint32_t count = static_cast<uint32_t>(entities_.size());
    if (count == 0) {
        // No entities this frame — just clear prev data
        prevEntities_.clear();
        return;
    }

    // 1. Allocate / grow the host-visible SSBO
    const VkDeviceSize needed = count * sizeof(EntityMotion);
    if (!entityMotionBuffer_.handle() || entityMotionBuffer_.size() < needed) {
        vk2::Buffer::Desc bd{};
        bd.size = std::max(needed, VkDeviceSize(4096)); // min 4 KB
        bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        auto r = vk2::Buffer::create(device_->device(), device_->vma(), bd);
        if (!r) {
            log::error("entityBlas", "Entity motion SSBO alloc failed");
            return;
        }
        entityMotionBuffer_ = std::move(r.value());
    }

    // 2. Fill motion deltas
    auto* dst = static_cast<EntityMotion*>(entityMotionBuffer_.mappedPtr());
    std::unordered_set<uint32_t> seenThisFrame;
    seenThisFrame.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        const auto& e = entities_[i];
        seenThisFrame.insert(e.hashCode);

        auto it = prevEntities_.find(e.hashCode);
        if (it != prevEntities_.end()) {
            dst[i].dx = e.posX - it->second.posX;
            dst[i].dy = e.posY - it->second.posY;
            dst[i].dz = e.posZ - it->second.posZ;
        } else {
            dst[i].dx = 0.0f;
            dst[i].dy = 0.0f;
            dst[i].dz = 0.0f;
        }
        dst[i].pad = 0.0f;
    }
    entityMotionBuffer_.flush(); // Ensure GPU-visible on non-coherent heaps

    // 3. Update prevEntities_ from current frame
    //    Erase stale entries first, then overwrite with current data.
    for (auto it = prevEntities_.begin(); it != prevEntities_.end(); ) {
        if (seenThisFrame.find(it->first) == seenThisFrame.end())
            it = prevEntities_.erase(it);
        else
            ++it;
    }

    for (const auto& e : entities_) {
        PrevEntityData& pd = prevEntities_[e.hashCode];
        pd.posX = e.posX;
        pd.posY = e.posY;
        pd.posZ = e.posZ;
        pd.vertexAddr = static_cast<uint64_t>(e.vertexAddress);
        pd.indexAddr  = static_cast<uint64_t>(e.indexAddress);
    }
}

} // namespace engine
