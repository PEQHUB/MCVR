#pragma once

#include "core/all_extern.hpp"

#include <utility>
#include <vector>

namespace vk {
class PhysicalDevice;
class Device;
class DeviceLocalBuffer;
class DeviceLocalImage;
class DescriptorTable;
class GraphicsPipeline;
class ComputePipeline;
class RayTracingPipeline;
class RenderPass;
class Framebuffer;
class Buffer;
class Image;
class SBT;
class Fence;

class CommandPool : public SharedObject<CommandPool> {
  public:
    CommandPool(std::shared_ptr<PhysicalDevice> physicalDevice, std::shared_ptr<Device> device);
    CommandPool(std::shared_ptr<PhysicalDevice> physicalDevice, std::shared_ptr<Device> device, uint32_t queueIndex);
    ~CommandPool();

    VkCommandPool &vkCommandPool();

  private:
    std::shared_ptr<Device> device_;

    VkCommandPool commandPool_ = VK_NULL_HANDLE;
};

class CommandBuffer : public SharedObject<CommandBuffer> {
  public:
    struct SubmitInfo {
        std::vector<std::pair<VkSemaphore, VkPipelineStageFlags>> waitSemaphoresAndStageMasks;
        std::vector<VkSemaphore> signalSemaphores;
        VkFence signalFence;
    };

    struct BufferMemoryBarrier {
        VkPipelineStageFlags2 srcStageMask;
        VkAccessFlags2 srcAccessMask;
        VkPipelineStageFlags2 dstStageMask;
        VkAccessFlags2 dstAccessMask;
        uint32_t srcQueueFamilyIndex;
        uint32_t dstQueueFamilyIndex;
        std::shared_ptr<Buffer> buffer;
        VkDeviceSize offset = 0;
        VkDeviceSize size = VK_WHOLE_SIZE;
    };

    struct ImageMemoryBarrier {
        VkPipelineStageFlags2 srcStageMask;
        VkAccessFlags2 srcAccessMask;
        VkPipelineStageFlags2 dstStageMask;
        VkAccessFlags2 dstAccessMask;
        VkImageLayout oldLayout;
        VkImageLayout newLayout;
        uint32_t srcQueueFamilyIndex;
        uint32_t dstQueueFamilyIndex;
        std::shared_ptr<Image> image;
        VkImageSubresourceRange subresourceRange;
    };

    struct MemoryBarrier {
        VkPipelineStageFlags2 srcStageMask;
        VkAccessFlags2 srcAccessMask;
        VkPipelineStageFlags2 dstStageMask;
        VkAccessFlags2 dstAccessMask;
    };

    struct RenderPassBeginInfo {
        std::shared_ptr<RenderPass> renderPass;
        std::shared_ptr<Framebuffer> framebuffer;
        VkOffset2D renderAreaOffset = {0, 0};
        VkExtent2D renderAreaExtent;
        std::vector<VkClearValue> clearValues;
    };

  public:
    CommandBuffer(std::shared_ptr<Device> device, std::shared_ptr<CommandPool> commandPool_);

    VkCommandBuffer &vkCommandBuffer();

    std::shared_ptr<CommandBuffer> begin(VkCommandBufferUsageFlags flags = 0);
    std::shared_ptr<CommandBuffer> copyToDeviceLocalBuffer(std::shared_ptr<DeviceLocalBuffer> deviceLocalBuffer);
    std::shared_ptr<CommandBuffer> copyToDeviceLocalImage(std::shared_ptr<DeviceLocalImage> deviceLocalImage);
    std::shared_ptr<CommandBuffer> barriersBufferImage(std::vector<BufferMemoryBarrier> bufferBarriers,
                                                       std::vector<ImageMemoryBarrier> imageBarriers);
    std::shared_ptr<CommandBuffer> barriersMemory(std::vector<MemoryBarrier> memoryBarriers);
    std::shared_ptr<CommandBuffer> beginRenderPass(RenderPassBeginInfo renderPassBeginInfo);
    std::shared_ptr<CommandBuffer> endRenderPass();
    std::shared_ptr<CommandBuffer> bindDescriptorTable(std::shared_ptr<DescriptorTable> descriptorTable,
                                                       VkPipelineBindPoint bindPoint);
    std::shared_ptr<CommandBuffer> bindGraphicsPipeline(std::shared_ptr<GraphicsPipeline> pipeline);
    std::shared_ptr<CommandBuffer> bindRTPipeline(std::shared_ptr<RayTracingPipeline> pipeline);
    std::shared_ptr<CommandBuffer> bindComputePipeline(std::shared_ptr<ComputePipeline> pipeline);
    std::shared_ptr<CommandBuffer> bindVertexBuffers(std::shared_ptr<DeviceLocalBuffer> buffer);
    std::shared_ptr<CommandBuffer> bindIndexBuffer(std::shared_ptr<DeviceLocalBuffer> buffer);
    std::shared_ptr<CommandBuffer> bindIndexBuffer(std::shared_ptr<DeviceLocalBuffer> buffer, VkIndexType indexType);
    std::shared_ptr<CommandBuffer>
    draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstIndex = 0, uint32_t firstInstance = 0);
    std::shared_ptr<CommandBuffer> drawIndexed(uint32_t indexCount,
                                               uint32_t instanceCount,
                                               uint32_t firstIndex = 0,
                                               int32_t vertexOffset = 0,
                                               uint32_t firstInstance = 0);
    std::shared_ptr<CommandBuffer>
    raytracing(std::shared_ptr<SBT> sbt, uint32_t width, uint32_t height, uint32_t depth);
    std::shared_ptr<CommandBuffer> end();

    void submitMainQueueIndividual(std::shared_ptr<Device> device);
    void submitMainQueueIndividual(std::shared_ptr<Device> device, std::shared_ptr<Fence> fence);
    void submitMainQueue(std::shared_ptr<Device> device, SubmitInfo submitInfo);

  private:
    std::shared_ptr<Device> device_;
    std::shared_ptr<CommandPool> commandPool_;

    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
};
}; // namespace vk