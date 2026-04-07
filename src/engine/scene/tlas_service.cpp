#include "tlas_service.hpp"
#include "blas_service.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <algorithm>
#include <cstring>
#include <vector>

namespace engine {

TlasService::~TlasService() { shutdown(); }

vk2::Result<void> TlasService::init(vk2::DeviceService& device) {
    device_ = &device;

    if (!device.caps().accelerationStructure) {
        log::warn("tlas", "Acceleration structures not supported — TLAS service disabled");
        return {};
    }

    initialized_ = true;
    log::info("tlas", "TLAS service initialized");
    return {};
}

void TlasService::shutdown() {
    destroyTlas();
    instanceBuffer_ = {};
    scratchBuffer_ = {};
    vertexBdaBuffer_ = {};
    indexBdaBuffer_ = {};
    chunkOriginBuffer_ = {};
    bdaCapacity_ = 0;
    originCapacity_ = 0;
    initialized_ = false;
}

bool TlasService::rebuild(VkCommandBuffer cmd, const BlasService& blasService) {
    if (!initialized_ || !device_->caps().accelerationStructure) return false;
    if (blasService.blasCount() == 0) return false;

    // Collect instances + parallel BDA arrays.
    // instanceCustomIndex = position in vertexBdas/indexBdas, so the closest-hit
    // shader can use gl_InstanceCustomIndexEXT to find this chunk's vertex/index buffers.
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    std::vector<uint64_t> vertexBdas;
    std::vector<uint64_t> indexBdas;
    std::vector<float>    chunkOrigins;  // 4 floats per instance (xyz + pad)
    instances.reserve(blasService.blasCount());
    vertexBdas.reserve(blasService.blasCount());
    indexBdas.reserve(blasService.blasCount());
    chunkOrigins.reserve(blasService.blasCount() * 4);

    uint32_t nextCustomIdx = 0;
    blasService.forEachBlas([&](const ChunkId& id, const BlasData& blas) {
        VkAccelerationStructureInstanceKHR inst{};
        // Identity transform (chunk origin baked into vertex data)
        inst.transform.matrix[0][0] = 1.0f;
        inst.transform.matrix[1][1] = 1.0f;
        inst.transform.matrix[2][2] = 1.0f;
        inst.instanceCustomIndex = nextCustomIdx;
        inst.mask = 0xFF;
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blas.address;
        instances.push_back(inst);

        vertexBdas.push_back(blas.vertexAddress);
        indexBdas.push_back(blas.indexAddress);
        // chunk origin is baked into vertex positions, so the "origin" is 0 for now.
        // TODO: when GpuChunkData.origin is bridged through, populate this from blas.
        chunkOrigins.push_back(0.0f);
        chunkOrigins.push_back(0.0f);
        chunkOrigins.push_back(0.0f);
        chunkOrigins.push_back(0.0f);

        ++nextCustomIdx;
    });

    if (instances.empty()) return false;

    instanceCount_ = static_cast<uint32_t>(instances.size());
    VkDeviceSize instanceSize = instanceCount_ * sizeof(VkAccelerationStructureInstanceKHR);

    // Create/resize instance buffer (host-visible for simplicity)
    if (!instanceBuffer_.handle() || instanceBuffer_.size() < instanceSize) {
        vk2::Buffer::Desc bd{};
        bd.size = instanceSize;
        bd.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                 | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        auto r = vk2::Buffer::create(device_->device(), device_->vma(), bd);
        if (!r) { log::error("tlas", "Instance buffer: " + r.error().message); return false; }
        instanceBuffer_ = std::move(r.value());
    }

    // Upload instance data
    std::memcpy(instanceBuffer_.mappedPtr(), instances.data(), instanceSize);

    // Allocate and upload BDA arrays (host-visible, used as storage buffers by RT shaders)
    VkDeviceSize bdaSize = instanceCount_ * sizeof(uint64_t);
    if (!vertexBdaBuffer_.handle() || bdaCapacity_ < instanceCount_) {
        vk2::Buffer::Desc bd{};
        // Allocate at least 64 bytes; round up to a comfy capacity to avoid frequent reallocs
        VkDeviceSize newCapacity = std::max<VkDeviceSize>(instanceCount_ * 2, 64);
        bd.size = newCapacity * sizeof(uint64_t);
        bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        auto rv = vk2::Buffer::create(device_->device(), device_->vma(), bd);
        if (!rv) { log::error("tlas", "Vertex BDA buffer: " + rv.error().message); return false; }
        vertexBdaBuffer_ = std::move(rv.value());

        auto ri = vk2::Buffer::create(device_->device(), device_->vma(), bd);
        if (!ri) { log::error("tlas", "Index BDA buffer: " + ri.error().message); return false; }
        indexBdaBuffer_ = std::move(ri.value());

        bdaCapacity_ = newCapacity;
    }
    std::memcpy(vertexBdaBuffer_.mappedPtr(), vertexBdas.data(), bdaSize);
    std::memcpy(indexBdaBuffer_.mappedPtr(), indexBdas.data(), bdaSize);

    // Chunk origin SSBO (vec4 per instance)
    VkDeviceSize originSize = instanceCount_ * sizeof(float) * 4;
    if (!chunkOriginBuffer_.handle() || originCapacity_ < instanceCount_) {
        vk2::Buffer::Desc bd{};
        VkDeviceSize newCapacity = std::max<VkDeviceSize>(instanceCount_ * 2, 64);
        bd.size = newCapacity * sizeof(float) * 4;
        bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        auto r = vk2::Buffer::create(device_->device(), device_->vma(), bd);
        if (!r) { log::error("tlas", "Chunk origin buffer: " + r.error().message); return false; }
        chunkOriginBuffer_ = std::move(r.value());
        originCapacity_ = newCapacity;
    }
    std::memcpy(chunkOriginBuffer_.mappedPtr(), chunkOrigins.data(), originSize);

    // Build geometry info
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = instanceBuffer_.deviceAddress();

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    // Query size
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(device_->device(),
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &instanceCount_, &sizeInfo);

    // Recreate TLAS buffer if needed
    if (!tlasBuffer_.handle() || tlasBuffer_.size() < sizeInfo.accelerationStructureSize) {
        destroyTlas();

        vk2::Buffer::Desc bd{};
        bd.size = sizeInfo.accelerationStructureSize;
        bd.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                 | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        auto r = vk2::Buffer::create(device_->device(), device_->vma(), bd);
        if (!r) { log::error("tlas", "TLAS buffer: " + r.error().message); return false; }
        tlasBuffer_ = std::move(r.value());

        VkAccelerationStructureCreateInfoKHR asCI{};
        asCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        asCI.buffer = tlasBuffer_.handle();
        asCI.size = sizeInfo.accelerationStructureSize;
        asCI.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

        if (vkCreateAccelerationStructureKHR(device_->device(), &asCI, nullptr, &tlas_) != VK_SUCCESS) {
            log::error("tlas", "vkCreateAccelerationStructureKHR failed");
            return false;
        }

        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addrInfo.accelerationStructure = tlas_;
        tlasAddress_ = vkGetAccelerationStructureDeviceAddressKHR(device_->device(), &addrInfo);
    }

    // Recreate scratch buffer if needed
    if (!scratchBuffer_.handle() || scratchBuffer_.size() < sizeInfo.buildScratchSize) {
        vk2::Buffer::Desc bd{};
        bd.size = sizeInfo.buildScratchSize;
        bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bd.minAlignment = 256;
        auto r = vk2::Buffer::create(device_->device(), device_->vma(), bd);
        if (!r) { log::error("tlas", "Scratch buffer: " + r.error().message); return false; }
        scratchBuffer_ = std::move(r.value());
    }

    // Build
    buildInfo.dstAccelerationStructure = tlas_;
    buildInfo.scratchData.deviceAddress = scratchBuffer_.deviceAddress();

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = instanceCount_;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;

    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);

    // Barrier: TLAS build → RT shader read
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);

    return true;
}

void TlasService::destroyTlas() {
    if (tlas_ != VK_NULL_HANDLE && device_) {
        vkDestroyAccelerationStructureKHR(device_->device(), tlas_, nullptr);
        tlas_ = VK_NULL_HANDLE;
        tlasAddress_ = 0;
    }
}

} // namespace engine
