#include "core/vulkan/command.hpp"

#include "core/vulkan/buffer.hpp"
#include "core/vulkan/descriptor.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/framebuffer.hpp"
#include "core/vulkan/image.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/pipeline.hpp"
#include "core/vulkan/render_pass.hpp"
#include "core/vulkan/sbt.hpp"
#include "core/vulkan/sync.hpp"

#include <iostream>

std::ostream &commandPoolCout() {
    return std::cout << "[CommandPool] ";
}

std::ostream &commandPoolCerr() {
    return std::cerr << "[CommandPool] ";
}

std::ostream &commandBufferCout() {
    return std::cout << "[CommandBuffer] ";
}

std::ostream &commandBufferCerr() {
    return std::cerr << "[CommandBuffer] ";
}

vk::CommandPool::CommandPool(std::shared_ptr<PhysicalDevice> physicalDevice, std::shared_ptr<Device> device)
    : CommandPool(physicalDevice, device, physicalDevice->mainQueueIndex()) {}

vk::CommandPool::CommandPool(std::shared_ptr<PhysicalDevice> physicalDevice,
                             std::shared_ptr<Device> device,
                             uint32_t queueIndex)
    : device_(device) {
    // Create graphics command pool
    VkCommandPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCreateInfo.queueFamilyIndex = queueIndex;

    if (vkCreateCommandPool(device_->vkDevice(), &poolCreateInfo, nullptr, &commandPool_) != VK_SUCCESS) {
        commandPoolCerr() << "failed to create command pool" << std::endl;
        exit(EXIT_FAILURE);
    } else {
#ifdef DEBUG
        commandPoolCout() << "created command pool" << std::endl;
#endif
    }
}

vk::CommandPool::~CommandPool() {
    vkDestroyCommandPool(device_->vkDevice(), commandPool_, nullptr);

#ifdef DEBUG
    commandPoolCout() << "command pool deconstructed" << std::endl;
#endif
}

VkCommandPool &vk::CommandPool::vkCommandPool() {
    return commandPool_;
}

vk::CommandBuffer::CommandBuffer(std::shared_ptr<Device> device, std::shared_ptr<CommandPool> commandPool_)
    : commandPool_(commandPool_), device_(device) {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_->vkCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device_->vkDevice(), &allocInfo, &commandBuffer_) != VK_SUCCESS) {
        commandBufferCerr() << "failed to allocate command buffer" << std::endl;
        exit(EXIT_FAILURE);
    } else {
#ifdef DEBUG
        commandBufferCout() << "allocated command buffer" << std::endl;
#endif
    }
}

VkCommandBuffer &vk::CommandBuffer::vkCommandBuffer() {
    return commandBuffer_;
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::begin(VkCommandBufferUsageFlags flags) {
    VkCommandBufferBeginInfo bufferBeginInfo = {};
    bufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bufferBeginInfo.flags = flags;

    vkBeginCommandBuffer(commandBuffer_, &bufferBeginInfo);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer>
vk::CommandBuffer::copyToDeviceLocalBuffer(std::shared_ptr<DeviceLocalBuffer> deviceLocalBuffer) {
    deviceLocalBuffer->uploadToBuffer(commandBuffer_);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer>
vk::CommandBuffer::copyToDeviceLocalImage(std::shared_ptr<DeviceLocalImage> deviceLocalImage) {
    deviceLocalImage->uploadToImage(commandBuffer_);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer>
vk::CommandBuffer::barriersBufferImage(std::vector<BufferMemoryBarrier> bufferBarriers,
                                       std::vector<ImageMemoryBarrier> imageBarriers) {
    std::vector<VkBufferMemoryBarrier2> vkBufferBarriers;
    for (int i = 0; i < bufferBarriers.size(); i++) {
        VkBufferMemoryBarrier2 vkBufferBarrier{};
        vkBufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        vkBufferBarrier.srcStageMask = bufferBarriers[i].srcStageMask;
        vkBufferBarrier.srcAccessMask = bufferBarriers[i].srcAccessMask;
        vkBufferBarrier.dstStageMask = bufferBarriers[i].dstStageMask;
        vkBufferBarrier.dstAccessMask = bufferBarriers[i].dstAccessMask;
        vkBufferBarrier.srcQueueFamilyIndex = bufferBarriers[i].srcQueueFamilyIndex;
        vkBufferBarrier.dstQueueFamilyIndex = bufferBarriers[i].dstQueueFamilyIndex;
        vkBufferBarrier.buffer = bufferBarriers[i].buffer->vkBuffer();
        vkBufferBarrier.offset = bufferBarriers[i].offset;
        vkBufferBarrier.size = bufferBarriers[i].size;

        vkBufferBarriers.push_back(vkBufferBarrier);
    }

    std::vector<VkImageMemoryBarrier2> vkImageBarriers;
    for (int i = 0; i < imageBarriers.size(); i++) {
        VkImageMemoryBarrier2 vkImageBarrier{};
        vkImageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        vkImageBarrier.srcStageMask = imageBarriers[i].srcStageMask;
        vkImageBarrier.srcAccessMask = imageBarriers[i].srcAccessMask;
        vkImageBarrier.dstStageMask = imageBarriers[i].dstStageMask;
        vkImageBarrier.dstAccessMask = imageBarriers[i].dstAccessMask;
        vkImageBarrier.oldLayout = imageBarriers[i].oldLayout;
        vkImageBarrier.newLayout = imageBarriers[i].newLayout;
        vkImageBarrier.srcQueueFamilyIndex = imageBarriers[i].srcQueueFamilyIndex;
        vkImageBarrier.dstQueueFamilyIndex = imageBarriers[i].dstQueueFamilyIndex;
        vkImageBarrier.image = imageBarriers[i].image->vkImage();
        vkImageBarrier.subresourceRange = imageBarriers[i].subresourceRange;

        vkImageBarriers.push_back(vkImageBarrier);
    }

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.bufferMemoryBarrierCount = vkBufferBarriers.size();
    dependencyInfo.pBufferMemoryBarriers = vkBufferBarriers.data();
    dependencyInfo.imageMemoryBarrierCount = vkImageBarriers.size();
    dependencyInfo.pImageMemoryBarriers = vkImageBarriers.data();
    dependencyInfo.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    vkCmdPipelineBarrier2(commandBuffer_, &dependencyInfo);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::barriersMemory(std::vector<MemoryBarrier> memoryBarriers) {
    std::vector<VkMemoryBarrier2> vkMemoryBarriers;
    for (int i = 0; i < memoryBarriers.size(); i++) {
        VkMemoryBarrier2 vkMemoryBarrier{};
        vkMemoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        vkMemoryBarrier.srcStageMask = memoryBarriers[i].srcStageMask;
        vkMemoryBarrier.srcAccessMask = memoryBarriers[i].srcAccessMask;
        vkMemoryBarrier.dstStageMask = memoryBarriers[i].dstStageMask;
        vkMemoryBarrier.dstAccessMask = memoryBarriers[i].dstAccessMask;

        vkMemoryBarriers.push_back(vkMemoryBarrier);
    }

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.memoryBarrierCount = vkMemoryBarriers.size();
    dependencyInfo.pMemoryBarriers = vkMemoryBarriers.data();

    vkCmdPipelineBarrier2(commandBuffer_, &dependencyInfo);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::beginRenderPass(RenderPassBeginInfo renderPassBeginInfo) {
    VkRenderPassBeginInfo vkRenderPassBeginInfo{};
    vkRenderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    vkRenderPassBeginInfo.renderPass = renderPassBeginInfo.renderPass->vkRenderPass();
    vkRenderPassBeginInfo.framebuffer = renderPassBeginInfo.framebuffer->vkFramebuffer();
    vkRenderPassBeginInfo.renderArea.offset = renderPassBeginInfo.renderAreaOffset;
    vkRenderPassBeginInfo.renderArea.extent = renderPassBeginInfo.renderAreaExtent;
    vkRenderPassBeginInfo.clearValueCount = renderPassBeginInfo.clearValues.size();
    vkRenderPassBeginInfo.pClearValues = renderPassBeginInfo.clearValues.data();
    vkCmdBeginRenderPass(commandBuffer_, &vkRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::endRenderPass() {
    vkCmdEndRenderPass(commandBuffer_);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer>
vk::CommandBuffer::bindDescriptorTable(std::shared_ptr<DescriptorTable> descriptorTable,
                                       VkPipelineBindPoint bindPoint) {
    vkCmdBindDescriptorSets(commandBuffer_, bindPoint, descriptorTable->vkPipelineLayout(), 0,
                            descriptorTable->descriptorSet().size(), descriptorTable->descriptorSet().data(), 0,
                            nullptr);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::bindGraphicsPipeline(std::shared_ptr<GraphicsPipeline> pipeline) {
    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->vkPipeline());
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::bindRTPipeline(std::shared_ptr<RayTracingPipeline> pipeline) {
    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline->vkPipeline());
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::bindComputePipeline(std::shared_ptr<ComputePipeline> pipeline) {
    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->vkPipeline());
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::bindVertexBuffers(std::shared_ptr<DeviceLocalBuffer> buffer) {
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(commandBuffer_, 0, 1, &buffer->vkBuffer(), &offset);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::bindIndexBuffer(std::shared_ptr<DeviceLocalBuffer> buffer) {
    vkCmdBindIndexBuffer(commandBuffer_, buffer->vkBuffer(), 0, VK_INDEX_TYPE_UINT32);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::bindIndexBuffer(std::shared_ptr<DeviceLocalBuffer> buffer,
                                                                      VkIndexType indexType) {
    vkCmdBindIndexBuffer(commandBuffer_, buffer->vkBuffer(), 0, indexType);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer>
vk::CommandBuffer::draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstIndex, uint32_t firstInstance) {
    vkCmdDraw(commandBuffer_, vertexCount, instanceCount, firstIndex, firstInstance);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::drawIndexed(
    uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
    vkCmdDrawIndexed(commandBuffer_, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer>
vk::CommandBuffer::raytracing(std::shared_ptr<SBT> sbt, uint32_t width, uint32_t height, uint32_t depth) {
    vkCmdTraceRaysKHR(commandBuffer_, &sbt->raygenRegion(), &sbt->missRegion(), &sbt->hitRegion(),
                      &sbt->callableRegion(), width, height, depth);
    return shared_from_this();
}

std::shared_ptr<vk::CommandBuffer> vk::CommandBuffer::end() {
    vkEndCommandBuffer(commandBuffer_);
    return shared_from_this();
}

void vk::CommandBuffer::submitMainQueueIndividual(std::shared_ptr<Device> device) {
    submitMainQueueIndividual(device, nullptr);
}

void vk::CommandBuffer::submitMainQueueIndividual(std::shared_ptr<vk::Device> device,
                                                  std::shared_ptr<vk::Fence> fence) {
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer_;

    vkQueueSubmit(device->mainVkQueue(), 1, &submitInfo, fence == nullptr ? VK_NULL_HANDLE : fence->vkFence());
}

void vk::CommandBuffer::submitMainQueue(std::shared_ptr<Device> device, SubmitInfo submitInfo) {
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkPipelineStageFlags> waitStageMasks;
    for (auto [semaphore, mask] : submitInfo.waitSemaphoresAndStageMasks) {
        waitSemaphores.push_back(semaphore);
        waitStageMasks.push_back(mask);
    }

    std::vector<VkSemaphore> signalSemaphores = submitInfo.signalSemaphores;

    VkSubmitInfo vkSubmitInfo = {};
    vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo.waitSemaphoreCount = waitSemaphores.size();
    vkSubmitInfo.pWaitSemaphores = waitSemaphores.data();
    vkSubmitInfo.pWaitDstStageMask = waitStageMasks.data();
    vkSubmitInfo.commandBufferCount = 1;
    vkSubmitInfo.pCommandBuffers = &commandBuffer_;
    vkSubmitInfo.signalSemaphoreCount = signalSemaphores.size();
    vkSubmitInfo.pSignalSemaphores = signalSemaphores.data();

    vkQueueSubmit(device->mainVkQueue(), 1, &vkSubmitInfo, submitInfo.signalFence);
}