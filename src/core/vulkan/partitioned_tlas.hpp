#pragma once

#include "core/all_extern.hpp"
#include "core/vulkan/buffer.hpp"

namespace vk {
class Device;
class PhysicalDevice;
class VMA;
class CommandBuffer;

/// Partitioned TLAS (VK_NV_partitioned_acceleration_structure).
/// Replaces standard TLAS BUILD with incremental per-partition updates on Blackwell GPUs.
class PartitionedTLAS : public SharedObject<PartitionedTLAS> {
  public:
    struct Config {
        uint32_t maxInstances = 40000;
        uint32_t maxPartitions = 300;
        uint32_t maxInstancePerPartition = 512;
        uint32_t maxGlobalInstances = 2048;
    };

    PartitionedTLAS(std::shared_ptr<Device> device,
                    std::shared_ptr<PhysicalDevice> physicalDevice,
                    std::shared_ptr<VMA> vma,
                    const Config &config);
    ~PartitionedTLAS();

    /// Write all instances and record the PTLAS build command.
    /// Caller provides fully populated instance write data (with partition assignments).
    void buildAllInstances(
        const std::vector<VkPartitionedAccelerationStructureWriteInstanceDataNV> &instances,
        std::shared_ptr<CommandBuffer> commandBuffer);

    VkDeviceAddress dataBufferAddress() const { return dataBufferAddress_; }

    bool needsRecreate(uint32_t requiredInstances) const {
        return requiredInstances > config_.maxInstances;
    }

    const Config &config() const { return config_; }

  private:
    std::shared_ptr<Device> device_;
    std::shared_ptr<VMA> vma_;
    Config config_;

    // PTLAS data buffer (raw device memory — no VkAccelerationStructureKHR handle)
    std::shared_ptr<DeviceLocalBuffer> dataBuffer_;
    VkDeviceAddress dataBufferAddress_ = 0;

    // Scratch buffer (persisted across frames)
    std::shared_ptr<DeviceLocalBuffer> scratchBuffer_;

    // Indirect command buffer (host-visible, mapped):
    //   [0]: uint32_t srcInfosCount (= 1)
    //   [aligned]: VkBuildPartitionedAccelerationStructureIndirectCommandNV (command header)
    //   [aligned]: VkPartitionedAccelerationStructureWriteInstanceDataNV[] (instance data)
    std::shared_ptr<HostVisibleBuffer> indirectBuffer_;
    VkDeviceSize countOffset_ = 0;     // offset of srcInfosCount in indirectBuffer
    VkDeviceSize commandOffset_ = 0;   // offset of indirect command header
    VkDeviceSize instanceDataOffset_ = 0; // offset of instance data array

    VkPartitionedAccelerationStructureInstancesInputNV inputSpec_{};
    bool initialBuildDone_ = false;
    uint32_t prevInstanceCount_ = 0; // tracks max instances written for stale slot clearing
};

} // namespace vk
