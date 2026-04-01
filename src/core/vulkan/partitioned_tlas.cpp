#include "core/vulkan/partitioned_tlas.hpp"

#include "core/vulkan/command.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/vma.hpp"

#include <cstring>
#include <iostream>

static std::ostream &ptlasCout() {
    return std::cout << "[PTLAS] ";
}

// Align a value up to the specified alignment
static VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

vk::PartitionedTLAS::PartitionedTLAS(std::shared_ptr<Device> device,
                                      std::shared_ptr<PhysicalDevice> physicalDevice,
                                      std::shared_ptr<VMA> vma,
                                      const Config &config)
    : device_(device), vma_(vma), config_(config) {

    // Fill input specification
    inputSpec_.sType = VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT_NV;
    inputSpec_.pNext = nullptr;
    inputSpec_.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                       VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    inputSpec_.instanceCount = config_.maxInstances;
    inputSpec_.maxInstancePerPartitionCount = config_.maxInstancePerPartition;
    inputSpec_.partitionCount = config_.maxPartitions;
    inputSpec_.maxInstanceInGlobalPartitionCount = config_.maxGlobalInstances;

    // Query sizes
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetPartitionedAccelerationStructuresBuildSizesNV(device_->vkDevice(), &inputSpec_, &sizeInfo);

    ptlasCout() << "PTLAS sizes: data=" << (sizeInfo.accelerationStructureSize / 1024 / 1024) << "MB"
                << " scratch=" << (sizeInfo.buildScratchSize / 1024 / 1024) << "MB"
                << " (instances=" << config_.maxInstances
                << " partitions=" << config_.maxPartitions << ")" << std::endl;

    // Allocate data buffer (the PTLAS itself — raw device memory, no VkAccelerationStructureKHR)
    VkDeviceSize scratchAlignment = physicalDevice->accelerationStructProperties().minAccelerationStructureScratchOffsetAlignment;

    dataBuffer_ = DeviceLocalBuffer::create(
        vma, device, false, sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        0, VMA_MEMORY_USAGE_GPU_ONLY, 256);
    dataBufferAddress_ = dataBuffer_->bufferAddress();

    // Allocate scratch buffer
    scratchBuffer_ = DeviceLocalBuffer::create(
        vma, device, false, sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        0, VMA_MEMORY_USAGE_GPU_ONLY, scratchAlignment);

    // Layout the indirect buffer:
    //   [0]: uint32_t srcInfosCount (= 1)
    //   [16-aligned]: VkBuildPartitionedAccelerationStructureIndirectCommandNV (command header)
    //   [16-aligned]: VkPartitionedAccelerationStructureWriteInstanceDataNV[] (instance data array)
    constexpr VkDeviceSize ALIGN = 16;
    countOffset_ = 0;
    commandOffset_ = alignUp(sizeof(uint32_t), ALIGN);
    instanceDataOffset_ = alignUp(
        commandOffset_ + sizeof(VkBuildPartitionedAccelerationStructureIndirectCommandNV), ALIGN);
    VkDeviceSize totalIndirectSize = instanceDataOffset_ +
        static_cast<VkDeviceSize>(config_.maxInstances) *
        sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV);

    indirectBuffer_ = HostVisibleBuffer::create(
        vma, device, totalIndirectSize,
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, ALIGN);

    ptlasCout() << "Indirect buffer: " << (totalIndirectSize / 1024) << "KB" << std::endl;
}

vk::PartitionedTLAS::~PartitionedTLAS() = default;

void vk::PartitionedTLAS::buildAllInstances(
    const std::vector<VkPartitionedAccelerationStructureWriteInstanceDataNV> &instances,
    std::shared_ptr<CommandBuffer> commandBuffer) {

    uint32_t numInstances = static_cast<uint32_t>(instances.size());
    if (numInstances == 0) return;

    auto *mapped = static_cast<uint8_t *>(indirectBuffer_->mappedPtr());
    VkDeviceAddress baseAddr = indirectBuffer_->bufferAddress();

    // Write srcInfosCount = 1 (one indirect command)
    uint32_t cmdCount = 1;
    std::memcpy(mapped + countOffset_, &cmdCount, sizeof(uint32_t));

    // Write instance data array
    VkDeviceSize instanceDataSize = numInstances *
        sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV);
    std::memcpy(mapped + instanceDataOffset_, instances.data(), instanceDataSize);

    // Write the indirect command header
    VkBuildPartitionedAccelerationStructureIndirectCommandNV cmd{};
    cmd.opType = VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_INSTANCE_NV;
    cmd.argCount = numInstances;
    cmd.argData.startAddress = baseAddr + instanceDataOffset_;
    cmd.argData.strideInBytes = sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV);
    std::memcpy(mapped + commandOffset_, &cmd, sizeof(cmd));

    // Flush to make writes visible to GPU
    indirectBuffer_->flush();

    // Fill the build info
    VkBuildPartitionedAccelerationStructureInfoNV buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_BUILD_PARTITIONED_ACCELERATION_STRUCTURE_INFO_NV;
    buildInfo.input = inputSpec_;
    // Override instanceCount with actual count for this build
    buildInfo.input.instanceCount = numInstances;
    buildInfo.srcAccelerationStructureData = initialBuildDone_ ? dataBufferAddress_ : 0;
    buildInfo.dstAccelerationStructureData = dataBufferAddress_;
    buildInfo.scratchData = scratchBuffer_->bufferAddress();
    buildInfo.srcInfos = baseAddr + commandOffset_;
    buildInfo.srcInfosCount = baseAddr + countOffset_;

    vkCmdBuildPartitionedAccelerationStructuresNV(
        commandBuffer->vkCommandBuffer(), &buildInfo);

    initialBuildDone_ = true;
}
