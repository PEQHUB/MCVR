#include "frame_scheduler.hpp"
#include "app/engine_services.hpp"
#include "config/config_service.hpp"
#include "bridge/bridge_service.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "platform/vulkan/vk2_swapchain.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <cmath>

namespace engine {

FrameScheduler::FrameScheduler(EngineServices& services)
    : services_(services) {
    lastFrameTime_ = Clock::now();
}

FrameScheduler::~FrameScheduler() {
    shutdownSync();
    gc_.flush();
}

vk2::Result<void> FrameScheduler::initSync(vk2::DeviceService& device) {
    syncDevice_ = device.device();
    frameSync_.resize(imageCount_);

    // Command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = device.mainQueue().familyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkResult r = vkCreateCommandPool(syncDevice_, &poolInfo, nullptr, &cmdPool_);
    if (r != VK_SUCCESS) return vk2::makeError(r, "vkCreateCommandPool failed");

    // Allocate command buffers
    std::vector<VkCommandBuffer> cmds(imageCount_);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmdPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = imageCount_;

    r = vkAllocateCommandBuffers(syncDevice_, &allocInfo, cmds.data());
    if (r != VK_SUCCESS) return vk2::makeError(r, "vkAllocateCommandBuffers failed");

    // Per-frame sync
    for (uint32_t i = 0; i < imageCount_; ++i) {
        frameSync_[i].cmd = cmds[i];

        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        r = vkCreateSemaphore(syncDevice_, &semInfo, nullptr, &frameSync_[i].imageAcquired);
        if (r != VK_SUCCESS) return vk2::makeError(r, "vkCreateSemaphore (acquire) failed");

        r = vkCreateSemaphore(syncDevice_, &semInfo, nullptr, &frameSync_[i].renderComplete);
        if (r != VK_SUCCESS) return vk2::makeError(r, "vkCreateSemaphore (render) failed");

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // Signaled so first frame doesn't deadlock

        r = vkCreateFence(syncDevice_, &fenceInfo, nullptr, &frameSync_[i].inFlight);
        if (r != VK_SUCCESS) return vk2::makeError(r, "vkCreateFence failed");
    }

    syncInitialized_ = true;
    log::info("frame", "Sync objects created for " + std::to_string(imageCount_) + " frames");
    return {};
}

void FrameScheduler::shutdownSync() {
    if (!syncInitialized_ || syncDevice_ == VK_NULL_HANDLE) return;

    for (auto& fs : frameSync_) {
        if (fs.imageAcquired != VK_NULL_HANDLE)
            vkDestroySemaphore(syncDevice_, fs.imageAcquired, nullptr);
        if (fs.renderComplete != VK_NULL_HANDLE)
            vkDestroySemaphore(syncDevice_, fs.renderComplete, nullptr);
        if (fs.inFlight != VK_NULL_HANDLE)
            vkDestroyFence(syncDevice_, fs.inFlight, nullptr);
    }
    frameSync_.clear();

    if (cmdPool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(syncDevice_, cmdPool_, nullptr);
        cmdPool_ = VK_NULL_HANDLE;
    }

    syncInitialized_ = false;
}

FrameContext FrameScheduler::beginFrame() {
    auto now = Clock::now();
    if (!firstFrame_) {
        lastFrameTimeMs_ = std::chrono::duration<float, std::milli>(now - lastFrameTime_).count();
    }
    firstFrame_ = false;
    lastFrameTime_ = now;

    services_.bridge().flush();
    auto configSnap = services_.config().snapshot();
    gc_.tick();

    FrameContext ctx;
    ctx.frameIndex = frameIndex_;
    ctx.frameNumber = frameNumber_;
    ctx.deltaTime = lastFrameTimeMs_ / 1000.0f;
    ctx.config = std::move(configSnap);

    return ctx;
}

void FrameScheduler::executeClearFrame(
    vk2::DeviceService& device, vk2::SwapchainService& swapchain,
    const FrameContext& ctx) {

    if (!syncInitialized_) return;

    auto& sync = frameSync_[frameIndex_];

    // Wait for this frame's previous submission to complete
    vkWaitForFences(device.device(), 1, &sync.inFlight, VK_TRUE, UINT64_MAX);
    vkResetFences(device.device(), 1, &sync.inFlight);

    // Acquire swapchain image
    auto acquireResult = swapchain.acquireNextImage(sync.imageAcquired);
    if (!acquireResult) {
        log::warn("frame", "Acquire failed: " + acquireResult.error().message);
        return;
    }
    uint32_t imageIndex = acquireResult.value();

    // Record clear command
    VkCommandBuffer cmd = sync.cmd;
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Transition swapchain image to TRANSFER_DST
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = swapchain.image(imageIndex);
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Clear to a cycling color (proof of life — color changes each frame)
    float t = static_cast<float>(frameNumber_ % 360) / 360.0f;
    VkClearColorValue clearColor = {{
        0.5f * (1.0f + std::sin(t * 6.28318f)),
        0.5f * (1.0f + std::sin(t * 6.28318f + 2.094f)),
        0.5f * (1.0f + std::sin(t * 6.28318f + 4.189f)),
        1.0f
    }};
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, swapchain.image(imageIndex),
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &range);

    // Transition to PRESENT_SRC
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = 0;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(cmd);

    // Submit
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &sync.imageAcquired;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &sync.renderComplete;

    vkQueueSubmit(device.mainQueue().queue, 1, &submitInfo, sync.inFlight);

    // Present
    auto presentResult = swapchain.present(imageIndex, sync.renderComplete);
    if (!presentResult) {
        log::warn("frame", "Present failed: " + presentResult.error().message);
    }
}

void FrameScheduler::executeGraph(const FrameContext& ctx) {
    if (!graph_.valid()) return;
    for (const auto& pass : graph_.passes) {
        if (pass.execute) pass.execute(ctx, pass.resources);
    }
}

void FrameScheduler::setGraph(CompiledGraph graph) {
    graph_ = std::move(graph);
    if (graph_.valid()) {
        log::info("frame", "Render graph set: " + std::to_string(graph_.passCount()) +
                  " passes, " + std::to_string(graph_.resourceCount()) + " resources");
    }
}

void FrameScheduler::endFrame(const FrameContext& ctx) {
    ++frameNumber_;
    frameIndex_ = (frameIndex_ + 1) % imageCount_;
    services_.bridge().emit(EvtFrameComplete{ctx.frameNumber, lastFrameTimeMs_});
}

} // namespace engine
