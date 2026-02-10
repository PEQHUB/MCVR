#pragma once

#include "core/all_extern.hpp"

namespace vk {
class VMA;
class Device;
class CommandBuffer;

class Buffer {
  public:
    virtual size_t size() = 0;
    virtual VkBuffer &vkBuffer() = 0;
};

class HostVisibleBuffer : public Buffer, public SharedObject<HostVisibleBuffer> {
  public:
    HostVisibleBuffer(std::shared_ptr<VMA> vma, std::shared_ptr<Device> device, size_t size, VkBufferUsageFlags usage);
    HostVisibleBuffer(std::shared_ptr<VMA> vma, std::shared_ptr<Device> device, size_t size, VkBufferUsageFlags usage, VkDeviceSize minAlignment);
    ~HostVisibleBuffer();

    void downloadFromBuffer();
    void downloadFromBuffer(size_t size, size_t offset);

    void uploadToBuffer(void *src);
    void uploadToBuffer(void *src, size_t size, size_t offset);

    void flush();

    size_t size() override;
    VkBuffer &vkBuffer() override;
    void *mappedPtr();
    VkDeviceAddress &bufferAddress();

  private:
    std::shared_ptr<VMA> vma_;
    std::shared_ptr<Device> device_;

    size_t size_;
    VkBufferUsageFlags bufferUsage_;
    void *mappedPtr_ = nullptr;
    VkDeviceAddress bufferAddress_ = 0;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo_;
};

class DeviceLocalBuffer : public Buffer, public SharedObject<DeviceLocalBuffer> {
  public:
    DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                      std::shared_ptr<Device> device,
                      size_t size,
                      VkBufferUsageFlags usageExceptTransfer);
    DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                      std::shared_ptr<Device> device,
                      bool persistStaging,
                      size_t size,
                      VkBufferUsageFlags usageExceptTransfer);
    DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                      std::shared_ptr<Device> device,
                      bool persistStaging,
                      size_t size,
                      VkBufferUsageFlags usageExceptTransfer,
                      VmaAllocationCreateFlags allocationFlag,
                      VmaMemoryUsage vmaUsage);
    DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                      std::shared_ptr<Device> device,
                      bool persistStaging,
                      size_t size,
                      VkBufferUsageFlags usageExceptTransfer,
                      VmaAllocationCreateFlags allocationFlag,
                      VmaMemoryUsage vmaUsage,
                      VkDeviceSize minAlignment);
    ~DeviceLocalBuffer();

    void downloadFromStagingBuffer(void *dest);
    void downloadFromStagingBuffer(void *dest, size_t size, size_t offset);

    void uploadToStagingBuffer(void *src);
    void uploadToStagingBuffer(void *src, size_t size, size_t offset);
    void flushStagingBuffer();

    void downloadFromBuffer(VkCommandBuffer cmdBuffer);
    void downloadFromBuffer(VkCommandBuffer cmdBuffer, size_t size, size_t srcOffset, size_t dstOffset);

    void uploadToBuffer(VkCommandBuffer cmdBuffer);
    void uploadToBuffer(VkCommandBuffer cmdBuffer, size_t size, size_t srcOffset, size_t dstOffset);

    void uploadToBuffer(std::shared_ptr<CommandBuffer> cmdBuffer);
    void uploadToBuffer(std::shared_ptr<CommandBuffer> cmdBuffer, size_t size, size_t srcOffset, size_t dstOffset);

    size_t size() override;
    VkBuffer &vkStagingBuffer();
    VkBuffer &vkBuffer() override;
    void *mappedPtr();
    VkDeviceAddress &bufferAddress();

  private:
    std::shared_ptr<VMA> vma_;
    std::shared_ptr<Device> device_;

    bool persistStaging_;
    size_t size_;
    void *mappedPtr_ = nullptr;
    VkBufferUsageFlags bufferUsage_;
    VmaAllocationCreateFlags vmaAllocationFlags_;
    VmaMemoryUsage vmaUsage_;
    VkDeviceAddress bufferAddress_ = 0;
    VkBuffer stagingBuffer_ = VK_NULL_HANDLE;
    VmaAllocation stagingAllocation_ = VK_NULL_HANDLE;
    VmaAllocationInfo stagingAllocationInfo_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo_;
};
}; // namespace vk