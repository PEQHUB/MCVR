#include "core/vulkan/as.hpp"

#include "core/vulkan/command.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/vma.hpp"

#include <iostream>

vk::BLAS::BLAS(std::shared_ptr<Device> device,
               VkAccelerationStructureKHR blas,
               std::shared_ptr<DeviceLocalBuffer> blasBuffer)
    : device_(device), blas_(blas), blasBuffer_(blasBuffer) {
    VkAccelerationStructureDeviceAddressInfoKHR deviceAddressInfo{};
    deviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    deviceAddressInfo.accelerationStructure = blas_;
    blasDeviceAddress_ = vkGetAccelerationStructureDeviceAddressKHR(device->vkDevice(), &deviceAddressInfo);
}

vk::BLAS::~BLAS() {
    vkDestroyAccelerationStructureKHR(device_->vkDevice(), blas_, nullptr);
}

std::shared_ptr<vk::DeviceLocalBuffer> vk::BLAS::blasBuffer() {
    return blasBuffer_;
}

VkAccelerationStructureKHR &vk::BLAS::blas() {
    return blas_;
}

VkDeviceAddress &vk::BLAS::blasDeviceAddress() {
    return blasDeviceAddress_;
}

vk::TLAS::TLAS(std::shared_ptr<Device> device,
               VkAccelerationStructureKHR tlas,
               std::shared_ptr<DeviceLocalBuffer> tlasBuffer)
    : device_(device), tlas_(tlas), tlasBuffer_(tlasBuffer) {
    VkAccelerationStructureDeviceAddressInfoKHR deviceAddressInfo{};
    deviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    deviceAddressInfo.accelerationStructure = tlas_;
    tlasDeviceAddress_ = vkGetAccelerationStructureDeviceAddressKHR(device->vkDevice(), &deviceAddressInfo);
}

vk::TLAS::~TLAS() {
    vkDestroyAccelerationStructureKHR(device_->vkDevice(), tlas_, nullptr);
}

std::shared_ptr<vk::DeviceLocalBuffer> vk::TLAS::tlasBuffer() {
    return tlasBuffer_;
}

VkAccelerationStructureKHR &vk::TLAS::tlas() {
    return tlas_;
}

VkDeviceAddress &vk::TLAS::tlasDeviceAddress() {
    return tlasDeviceAddress_;
}

vk::BLASBuilder::BLASGeometryBuilder::BLASGeometryBuilder(vk::BLASBuilder &parent) : parent(parent) {}

vk::BLASBuilder::BLASGeometryBuilder &vk::BLASBuilder::BLASGeometryBuilder::definePlaceholderGeometry() {
    VkAccelerationStructureGeometryTrianglesDataKHR trianglesData{};
    trianglesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    trianglesData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT; // A format is needed, but it won't be used.
    trianglesData.vertexData.deviceAddress = 0;
    trianglesData.vertexStride = 0;
    trianglesData.maxVertex = 0;
    trianglesData.indexType = VK_INDEX_TYPE_NONE_KHR;
    trianglesData.indexData.deviceAddress = 0;
    trianglesData.transformData.deviceAddress = 0;

    VkAccelerationStructureGeometryKHR placeholderGeometry{};
    placeholderGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    placeholderGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    placeholderGeometry.geometry.triangles = trianglesData;

    placeholderGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    geometries.push_back(placeholderGeometry);
    primitiveCounts.push_back(0);

    return *this;
}

std::shared_ptr<vk::BLASBuilder> vk::BLASBuilder::BLASGeometryBuilder::endGeometries() {
    return parent.shared_from_this();
}

vk::BLASBuilder::BLASBuilder() : geometryBuilder_(*this) {}

std::shared_ptr<vk::BLASBuilder::BLASGeometryBuilder> vk::BLASBuilder::beginGeometries() {
    return std::shared_ptr<vk::BLASBuilder::BLASGeometryBuilder>(shared_from_this(), &geometryBuilder_);
}

std::shared_ptr<vk::BLASBuilder> vk::BLASBuilder::defineBuildProperty(VkBuildAccelerationStructureFlagsKHR flags) {
    mode_ = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    flags_ = flags;
    return shared_from_this();
}

std::shared_ptr<vk::BLASBuilder> vk::BLASBuilder::defineUpdateProperty(VkBuildAccelerationStructureFlagsKHR flags,
                                                                       VkAccelerationStructureKHR srcBLAS) {
    mode_ = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    flags_ = flags;
    srcBLAS_ = srcBLAS;
    return shared_from_this();
}

std::shared_ptr<vk::BLASBuilder> vk::BLASBuilder::querySizeInfo(std::shared_ptr<Device> device) {
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.mode = mode_;
    buildInfo.flags = flags_;
    buildInfo.geometryCount = static_cast<uint32_t>(geometryBuilder_.geometries.size());
    buildInfo.pGeometries = geometryBuilder_.geometries.data();

    sizeInfo_.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(device->vkDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &buildInfo, geometryBuilder_.primitiveCounts.data(), &sizeInfo_);

    return shared_from_this();
}

std::shared_ptr<vk::BLASBuilder> vk::BLASBuilder::allocateBuffers(std::shared_ptr<PhysicalDevice> physicalDevice,
                                                                  std::shared_ptr<Device> device,
                                                                  std::shared_ptr<VMA> vma) {
    blasBuffer_ = DeviceLocalBuffer::create(vma, device, false, sizeInfo_.accelerationStructureSize,
                                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                            0, VMA_MEMORY_USAGE_GPU_ONLY, 256);

    scratchBuffer_ = DeviceLocalBuffer::create(
        vma, device, false,
        mode_ == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR ? sizeInfo_.buildScratchSize :
                                                                  sizeInfo_.updateScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0, VMA_MEMORY_USAGE_GPU_ONLY,
        physicalDevice->accelerationStructProperties().minAccelerationStructureScratchOffsetAlignment);

    return shared_from_this();
}

std::shared_ptr<vk::BLAS> vk::BLASBuilder::buildAndSubmit(std::shared_ptr<Device> device,
                                                          std::shared_ptr<CommandBuffer> commandBuffer) {
    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = blasBuffer_->vkBuffer();
    createInfo.size = sizeInfo_.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    if (vkCreateAccelerationStructureKHR(device->vkDevice(), &createInfo, nullptr, &dstBLAS_) != VK_SUCCESS) {
        std::cout << "Cannot create BLAS" << std::endl;
        exit(EXIT_FAILURE);
    }

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.mode = mode_;
    buildInfo.flags = flags_;
    buildInfo.geometryCount = static_cast<uint32_t>(geometryBuilder_.geometries.size());
    buildInfo.pGeometries = geometryBuilder_.geometries.data();

    buildInfo.srcAccelerationStructure = srcBLAS_;
    buildInfo.dstAccelerationStructure = dstBLAS_;
    buildInfo.scratchData.deviceAddress = scratchBuffer_->bufferAddress();

    std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRanges;
    for (const auto &count : geometryBuilder_.primitiveCounts) {
        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = count;
        range.primitiveOffset = 0;
        buildRanges.push_back(range);
    }

    const VkAccelerationStructureBuildRangeInfoKHR *pBuildRanges = buildRanges.data();
    vkCmdBuildAccelerationStructuresKHR(commandBuffer->vkCommandBuffer(), 1, &buildInfo, &pBuildRanges);

    return BLAS::create(device, dstBLAS_, blasBuffer_);
}

std::shared_ptr<vk::BLAS> vk::BLASBuilder::build(std::shared_ptr<Device> device) {
    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = blasBuffer_->vkBuffer();
    createInfo.size = sizeInfo_.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    if (vkCreateAccelerationStructureKHR(device->vkDevice(), &createInfo, nullptr, &dstBLAS_) != VK_SUCCESS) {
        std::cout << "Cannot create BLAS" << std::endl;
        exit(EXIT_FAILURE);
    }

    return BLAS::create(device, dstBLAS_, blasBuffer_);
}

std::shared_ptr<vk::BLAS> vk::BLASBuilder::buildExternal(std::shared_ptr<Device> device,
                                                         std::shared_ptr<DeviceLocalBuffer> buffer,
                                                         VkDeviceSize offset) {
    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = buffer->vkBuffer();
    createInfo.size = sizeInfo_.accelerationStructureSize;
    createInfo.offset = offset;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    if (vkCreateAccelerationStructureKHR(device->vkDevice(), &createInfo, nullptr, &dstBLAS_) != VK_SUCCESS) {
        std::cout << "Cannot create BLAS" << std::endl;
        exit(EXIT_FAILURE);
    }

    return BLAS::create(device, dstBLAS_, buffer);
}

void vk::BLASBuilder::submit(std::shared_ptr<vk::CommandBuffer> commandBuffer) {
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.mode = mode_;
    buildInfo.flags = flags_;
    buildInfo.geometryCount = static_cast<uint32_t>(geometryBuilder_.geometries.size());
    buildInfo.pGeometries = geometryBuilder_.geometries.data();

    buildInfo.srcAccelerationStructure = srcBLAS_;
    buildInfo.dstAccelerationStructure = dstBLAS_;
    buildInfo.scratchData.deviceAddress = scratchBuffer_->bufferAddress();

    std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRanges;
    for (const auto &count : geometryBuilder_.primitiveCounts) {
        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = count;
        range.primitiveOffset = 0;
        buildRanges.push_back(range);
    }

    const VkAccelerationStructureBuildRangeInfoKHR *pBuildRanges = buildRanges.data();
    vkCmdBuildAccelerationStructuresKHR(commandBuffer->vkCommandBuffer(), 1, &buildInfo, &pBuildRanges);
}

void vk::BLASBuilder::submitExternal(std::shared_ptr<CommandBuffer> commandBuffer,
                                     VkDeviceAddress scratchBufferAddress) {
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.mode = mode_;
    buildInfo.flags = flags_;
    buildInfo.geometryCount = static_cast<uint32_t>(geometryBuilder_.geometries.size());
    buildInfo.pGeometries = geometryBuilder_.geometries.data();

    buildInfo.srcAccelerationStructure = srcBLAS_;
    buildInfo.dstAccelerationStructure = dstBLAS_;
    buildInfo.scratchData.deviceAddress = scratchBufferAddress;

    std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRanges;
    for (const auto &count : geometryBuilder_.primitiveCounts) {
        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = count;
        range.primitiveOffset = 0;
        buildRanges.push_back(range);
    }

    const VkAccelerationStructureBuildRangeInfoKHR *pBuildRanges = buildRanges.data();
    vkCmdBuildAccelerationStructuresKHR(commandBuffer->vkCommandBuffer(), 1, &buildInfo, &pBuildRanges);
}

void vk::BLASBuilder::batchSubmit(std::vector<std::shared_ptr<BLASBuilder>> &builders,
                                  std::shared_ptr<CommandBuffer> commandBuffer) {
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos;
    std::vector<std::vector<VkAccelerationStructureBuildRangeInfoKHR>> buildRanges;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR *> pbuildRanges;

    for (int i = 0; i < builders.size(); i++) {
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.mode = builders[i]->mode_;
        buildInfo.flags = builders[i]->flags_;
        buildInfo.geometryCount = static_cast<uint32_t>(builders[i]->geometryBuilder_.geometries.size());
        buildInfo.pGeometries = builders[i]->geometryBuilder_.geometries.data();

        buildInfo.srcAccelerationStructure = builders[i]->srcBLAS_;
        buildInfo.dstAccelerationStructure = builders[i]->dstBLAS_;
        buildInfo.scratchData.deviceAddress = builders[i]->scratchBuffer_->bufferAddress();
        buildInfos.push_back(buildInfo);

        std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRange;
        for (const auto &count : builders[i]->geometryBuilder_.primitiveCounts) {
            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = count;
            range.primitiveOffset = 0;
            buildRange.push_back(range);
        }
        buildRanges.push_back(buildRange);
        pbuildRanges.push_back(buildRanges[buildRanges.size() - 1].data());
    }

    vkCmdBuildAccelerationStructuresKHR(commandBuffer->vkCommandBuffer(), buildInfos.size(), buildInfos.data(),
                                        pbuildRanges.data());
}


void vk::BLASBuilder::batchSubmitExternal(std::vector<std::shared_ptr<BLASBuilder>> &builders,
                                          std::vector<VkDeviceAddress> scratchBufferAddress,
                                          std::shared_ptr<CommandBuffer> commandBuffer) {
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos;
    std::vector<std::vector<VkAccelerationStructureBuildRangeInfoKHR>> buildRanges;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR *> pbuildRanges;

    for (int i = 0; i < builders.size(); i++) {
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.mode = builders[i]->mode_;
        buildInfo.flags = builders[i]->flags_;
        buildInfo.geometryCount = static_cast<uint32_t>(builders[i]->geometryBuilder_.geometries.size());
        buildInfo.pGeometries = builders[i]->geometryBuilder_.geometries.data();

        buildInfo.srcAccelerationStructure = builders[i]->srcBLAS_;
        buildInfo.dstAccelerationStructure = builders[i]->dstBLAS_;
        buildInfo.scratchData.deviceAddress = scratchBufferAddress[i];
        buildInfos.push_back(buildInfo);

        std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRange;
        for (const auto &count : builders[i]->geometryBuilder_.primitiveCounts) {
            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = count;
            range.primitiveOffset = 0;
            buildRange.push_back(range);
        }
        buildRanges.push_back(buildRange);
        pbuildRanges.push_back(buildRanges[buildRanges.size() - 1].data());
    }

    vkCmdBuildAccelerationStructuresKHR(commandBuffer->vkCommandBuffer(), buildInfos.size(), buildInfos.data(),
                                        pbuildRanges.data());
}

std::shared_ptr<vk::BLASBuilder> vk::BLASBatchBuilder::defineBLASBuilder() {
    auto blasBuilder = BLASBuilder::create();
    builders_.push_back(blasBuilder);
    return blasBuilder;
}

std::shared_ptr<vk::BLASBatchBuilder> vk::BLASBatchBuilder::allocateBuffers(
    std::shared_ptr<PhysicalDevice> physicalDevice, std::shared_ptr<Device> device, std::shared_ptr<VMA> vma) {
    const VkDeviceSize scratchAlignment =
        physicalDevice->accelerationStructProperties().minAccelerationStructureScratchOffsetAlignment;
    const VkDeviceSize blasAlignment = 256; // VK_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT_KHR

    auto alignUp = [](VkDeviceSize value, VkDeviceSize alignment) -> VkDeviceSize {
        return (value + alignment - 1) & ~(alignment - 1);
    };

    VkDeviceSize totalBlasSize = 0;
    VkDeviceSize totalScratchSize = 0;

    for (auto builder : builders_) {
        blasOffsets_.push_back(totalBlasSize);
        totalBlasSize += builder->sizeInfo_.accelerationStructureSize;
        totalBlasSize = alignUp(totalBlasSize, blasAlignment);

        scratchOffsets_.push_back(totalScratchSize);
        totalScratchSize += builder->mode_ == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR ?
                                builder->sizeInfo_.buildScratchSize :
                                builder->sizeInfo_.updateScratchSize;
        totalScratchSize = alignUp(totalScratchSize, scratchAlignment);
    }

    blasBuffer_ = DeviceLocalBuffer::create(vma, device, false, totalBlasSize,
                                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                            0, VMA_MEMORY_USAGE_GPU_ONLY, 256);

    scratchBuffer_ = DeviceLocalBuffer::create(
        vma, device, false, totalScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0, VMA_MEMORY_USAGE_GPU_ONLY,
        physicalDevice->accelerationStructProperties().minAccelerationStructureScratchOffsetAlignment);

    for (int i = 0; i < scratchOffsets_.size(); i++) {
        scratchAddresses_.push_back(scratchBuffer_->bufferAddress() + scratchOffsets_[i]);
    }

    return shared_from_this();
}

std::vector<std::shared_ptr<vk::BLAS>> vk::BLASBatchBuilder::build(std::shared_ptr<Device> device) {
    std::vector<std::shared_ptr<vk::BLAS>> results;

    for (int i = 0; auto builder : builders_) {
        auto result = builder->buildExternal(device, blasBuffer_, blasOffsets_[i++]);
        results.push_back(result);
    }

    return results;
}

void vk::BLASBatchBuilder::submit(std::shared_ptr<vk::CommandBuffer> commandBuffer) {
    BLASBuilder::batchSubmitExternal(builders_, scratchAddresses_, commandBuffer);
}

vk::TLASBuilder::TLASInstanceBuilder::TLASInstanceBuilder(TLASBuilder &parent) : parent(parent) {}

vk::TLASBuilder::TLASInstanceBuilder &
vk::TLASBuilder::TLASInstanceBuilder::defineInstance(VkTransformMatrixKHR transform,
                                                     uint32_t customIndex,
                                                     uint32_t mask,
                                                     uint32_t offset,
                                                     VkGeometryInstanceFlagsKHR flag,
                                                     std::shared_ptr<BLAS> blas) {
    instances.emplace_back(transform, customIndex, mask, offset, flag, blas);
    return *this;
}

std::shared_ptr<vk::TLASBuilder>
vk::TLASBuilder::TLASInstanceBuilder::endInstanceBuilder(std::shared_ptr<Device> device, std::shared_ptr<VMA> vma) {
    std::vector<VkAccelerationStructureInstanceKHR> asInstances(instances.size());
    for (int i = 0; i < instances.size(); i++) {
        asInstances[i].transform = std::get<0>(instances[i]);
        asInstances[i].instanceCustomIndex = std::get<1>(instances[i]);
        asInstances[i].mask = std::get<2>(instances[i]);
        asInstances[i].instanceShaderBindingTableRecordOffset = std::get<3>(instances[i]);
        asInstances[i].flags = std::get<4>(instances[i]);
        asInstances[i].accelerationStructureReference = std::get<5>(instances[i])->blasDeviceAddress();
    }

    instanceBuffer =
        HostVisibleBuffer::create(vma, device, sizeof(VkAccelerationStructureInstanceKHR) * instances.size(),
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    instanceBuffer->uploadToBuffer(asInstances.data());

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = instanceBuffer->bufferAddress();
    geometries.push_back(geometry);

    return parent.shared_from_this();
}

vk::TLASBuilder::TLASBuilder() : tlasInstanceBuilder_(*this) {}

vk::TLASBuilder::TLASBuilder::TLASInstanceBuilder &vk::TLASBuilder::beginInstanceBuilder() {
    return tlasInstanceBuilder_;
}

std::shared_ptr<vk::TLASBuilder> vk::TLASBuilder::defineBuildProperty(VkBuildAccelerationStructureFlagsKHR flags) {
    mode_ = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    flags_ = flags;
    return shared_from_this();
}

std::shared_ptr<vk::TLASBuilder> vk::TLASBuilder::defineUpdateProperty(VkBuildAccelerationStructureFlagsKHR flags,
                                                                       VkAccelerationStructureKHR srcTLAS) {
    mode_ = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    flags_ = flags;
    srcTLAS_ = srcTLAS;
    return shared_from_this();
}

std::shared_ptr<vk::TLASBuilder> vk::TLASBuilder::querySizeInfo(std::shared_ptr<Device> device) {
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.mode = mode_;
    buildInfo.flags = flags_;
    buildInfo.geometryCount = static_cast<uint32_t>(tlasInstanceBuilder_.geometries.size()); // normally 1
    buildInfo.pGeometries = tlasInstanceBuilder_.geometries.data(); // TODO: actually not necessary

    uint32_t instanceCount = static_cast<uint32_t>(tlasInstanceBuilder_.instances.size());
    sizeInfo_.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(device->vkDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &buildInfo, &instanceCount, &sizeInfo_);
    return shared_from_this();
}

std::shared_ptr<vk::TLASBuilder> vk::TLASBuilder::allocateBuffers(std::shared_ptr<PhysicalDevice> physicalDevice,
                                                                  std::shared_ptr<Device> device,
                                                                  std::shared_ptr<VMA> vma) {
    tlasBuffer_ = DeviceLocalBuffer::create(vma, device, false, sizeInfo_.accelerationStructureSize,
                                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                            0, VMA_MEMORY_USAGE_GPU_ONLY, 256);

    scratchBuffer_ = DeviceLocalBuffer::create(
        vma, device, false,
        mode_ == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR ? sizeInfo_.buildScratchSize :
                                                                  sizeInfo_.updateScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0, VMA_MEMORY_USAGE_GPU_ONLY,
        physicalDevice->accelerationStructProperties().minAccelerationStructureScratchOffsetAlignment);

    return shared_from_this();
}

std::shared_ptr<vk::TLAS> vk::TLASBuilder::buildAndSubmit(std::shared_ptr<Device> device,
                                                          std::shared_ptr<CommandBuffer> commandBuffer) {
    VkAccelerationStructureCreateInfoKHR tlasCreateInfo{};
    tlasCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    tlasCreateInfo.buffer = tlasBuffer_->vkBuffer();
    tlasCreateInfo.size = sizeInfo_.accelerationStructureSize;
    tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    if (vkCreateAccelerationStructureKHR(device->vkDevice(), &tlasCreateInfo, nullptr, &dstTLAS_) != VK_SUCCESS) {
        std::cout << "Cannot create TLAS" << std::endl;
        exit(EXIT_FAILURE);
    }

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.mode = mode_;
    buildInfo.flags = flags_;
    buildInfo.geometryCount = static_cast<uint32_t>(tlasInstanceBuilder_.geometries.size());
    buildInfo.pGeometries = tlasInstanceBuilder_.geometries.data();

    buildInfo.srcAccelerationStructure = srcTLAS_;
    buildInfo.dstAccelerationStructure = dstTLAS_;
    buildInfo.scratchData.deviceAddress = scratchBuffer_->bufferAddress();

    VkAccelerationStructureBuildRangeInfoKHR buildRanges{};
    buildRanges.primitiveCount = tlasInstanceBuilder_.instances.size();
    buildRanges.primitiveOffset = 0;

    const VkAccelerationStructureBuildRangeInfoKHR *pBuildRanges = &buildRanges;
    vkCmdBuildAccelerationStructuresKHR(commandBuffer->vkCommandBuffer(), 1, &buildInfo, &pBuildRanges);

    return TLAS::create(device, dstTLAS_, tlasBuffer_);
}
