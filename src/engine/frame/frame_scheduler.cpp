#include "frame_scheduler.hpp"
#include "offscreen_target.hpp"
#include "composite_pass.hpp"
#include "rendergraph/barrier_planner.hpp"
#include "app/engine_services.hpp"
#include "config/config_service.hpp"
#include "bridge/bridge_service.hpp"
#include "diagnostics/metrics_service.hpp"
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

vk2::Result<void> FrameScheduler::initSync(vk2::DeviceService& device, uint32_t framesInFlight) {
    syncDevice_ = device.device();
    framesInFlight_ = framesInFlight;
    frameSync_.resize(framesInFlight_);

    // Command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = device.mainQueue().familyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkResult r = vkCreateCommandPool(syncDevice_, &poolInfo, nullptr, &cmdPool_);
    if (r != VK_SUCCESS) return vk2::makeError(r, "vkCreateCommandPool failed");

    // Command buffers
    std::vector<VkCommandBuffer> cmds(framesInFlight_);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmdPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = framesInFlight_;

    r = vkAllocateCommandBuffers(syncDevice_, &allocInfo, cmds.data());
    if (r != VK_SUCCESS) return vk2::makeError(r, "vkAllocateCommandBuffers failed");

    // Per-frame sync
    for (uint32_t i = 0; i < framesInFlight_; ++i) {
        frameSync_[i].cmd = cmds[i];

        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        r = vkCreateSemaphore(syncDevice_, &semInfo, nullptr, &frameSync_[i].imageAcquired);
        if (r != VK_SUCCESS) return vk2::makeError(r, "vkCreateSemaphore (acquire) failed");

        r = vkCreateSemaphore(syncDevice_, &semInfo, nullptr, &frameSync_[i].renderComplete);
        if (r != VK_SUCCESS) return vk2::makeError(r, "vkCreateSemaphore (render) failed");

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        r = vkCreateFence(syncDevice_, &fenceInfo, nullptr, &frameSync_[i].inFlight);
        if (r != VK_SUCCESS) return vk2::makeError(r, "vkCreateFence failed");
    }

    syncInitialized_ = true;
    log::info("frame", "Sync objects created: " + std::to_string(framesInFlight_) +
              " frames in flight, " + std::to_string(imageCount_) + " swapchain images");
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

    // Record CPU time and read previous frame's GPU results
    if (lastFrameTimeMs_ > 0) {
        services_.metrics().recordCpuFrameTime(lastFrameTimeMs_);
    }
    services_.metrics().readGpuResults(frameIndex_);

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

bool FrameScheduler::executeClearFrame(
    vk2::DeviceService& device, vk2::SwapchainService& swapchain,
    FrameContext& ctx) {

    if (!syncInitialized_) return false;

    // Handle pending recreate from previous frame
    if (swapchain.isRecreateNeeded()) {
        device.waitIdle();
        auto r = swapchain.recreate(0, 0);
        if (!r) {
            log::warn("frame", "Swapchain recreate failed: " + r.error().message);
        } else {
            swapchain.clearRecreateNeeded();
            imageCount_ = swapchain.imageCount();
        }
        return true; // Frame skipped
    }

    auto& sync = frameSync_[frameIndex_];

    // Wait for this frame slot's previous submission
    vkWaitForFences(device.device(), 1, &sync.inFlight, VK_TRUE, UINT64_MAX);
    vkResetFences(device.device(), 1, &sync.inFlight);

    // Acquire swapchain image
    auto acquireResult = swapchain.acquireNextImage(sync.imageAcquired);
    if (!acquireResult) {
        if (acquireResult.error().vkResult == VK_ERROR_OUT_OF_DATE_KHR) {
            swapchain.markRecreateNeeded();
            return true; // Skip frame, recreate next time
        }
        log::warn("frame", "Acquire failed: " + acquireResult.error().message);
        return true;
    }
    ctx.swapchainImageIndex = acquireResult.value();

    // Record command buffer
    VkCommandBuffer cmd = sync.cmd;
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // GPU timing begin
    services_.metrics().beginGpuFrame(cmd, ctx.frameIndex);

    // Cycling clear color (proof of life)
    float t = static_cast<float>(frameNumber_ % 360) / 360.0f;
    VkClearColorValue clearColor = {{
        0.5f * (1.0f + std::sin(t * 6.28318f)),
        0.5f * (1.0f + std::sin(t * 6.28318f + 2.094f)),
        0.5f * (1.0f + std::sin(t * 6.28318f + 4.189f)),
        1.0f
    }};

    VkImage swapImage = swapchain.image(ctx.swapchainImageIndex);

    bool useDynamicRendering = device.caps().dynamicRendering;

    if (useDynamicRendering) {
        // --- Dynamic rendering path (preferred) ---

        // Transition UNDEFINED → COLOR_ATTACHMENT_OPTIMAL
        VkImageMemoryBarrier2 preBarrier{};
        preBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        preBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        preBarrier.srcAccessMask = VK_ACCESS_2_NONE;
        preBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        preBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        preBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        preBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        preBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBarrier.image = swapImage;
        preBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo preDep{};
        preDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        preDep.imageMemoryBarrierCount = 1;
        preDep.pImageMemoryBarriers = &preBarrier;
        vkCmdPipelineBarrier2(cmd, &preDep);

        // Begin dynamic rendering with loadOp=CLEAR
        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = swapchain.imageView(ctx.swapchainImageIndex);
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = clearColor;

        VkRenderingInfo renderInfo{};
        renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderInfo.renderArea = {{0, 0}, swapchain.extent()};
        renderInfo.layerCount = 1;
        renderInfo.colorAttachmentCount = 1;
        renderInfo.pColorAttachments = &colorAttachment;

        vkCmdBeginRendering(cmd, &renderInfo);
        vkCmdEndRendering(cmd);

        // Transition COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC
        VkImageMemoryBarrier2 postBarrier{};
        postBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        postBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        postBarrier.dstAccessMask = VK_ACCESS_2_NONE;
        postBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        postBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        postBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        postBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        postBarrier.image = swapImage;
        postBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo postDep{};
        postDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        postDep.imageMemoryBarrierCount = 1;
        postDep.pImageMemoryBarriers = &postBarrier;
        vkCmdPipelineBarrier2(cmd, &postDep);

    } else {
        // --- Transfer clear fallback ---

        VkImageMemoryBarrier2 preBarrier{};
        preBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        preBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        preBarrier.srcAccessMask = VK_ACCESS_2_NONE;
        preBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        preBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        preBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        preBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        preBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBarrier.image = swapImage;
        preBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo preDep{};
        preDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        preDep.imageMemoryBarrierCount = 1;
        preDep.pImageMemoryBarriers = &preBarrier;
        vkCmdPipelineBarrier2(cmd, &preDep);

        VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, swapImage,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &range);

        VkImageMemoryBarrier2 postBarrier{};
        postBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        postBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        postBarrier.dstAccessMask = VK_ACCESS_2_NONE;
        postBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        postBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        postBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        postBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        postBarrier.image = swapImage;
        postBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo postDep{};
        postDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        postDep.imageMemoryBarrierCount = 1;
        postDep.pImageMemoryBarriers = &postBarrier;
        vkCmdPipelineBarrier2(cmd, &postDep);
    }

    // GPU timing end
    services_.metrics().endGpuFrame(cmd, ctx.frameIndex);

    vkEndCommandBuffer(cmd);

    // Submit via vkQueueSubmit2
    VkSemaphoreSubmitInfo waitSemInfo{};
    waitSemInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    waitSemInfo.semaphore = sync.imageAcquired;
    waitSemInfo.stageMask = useDynamicRendering
        ? VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
        : VK_PIPELINE_STAGE_2_TRANSFER_BIT;

    VkSemaphoreSubmitInfo signalSemInfo{};
    signalSemInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalSemInfo.semaphore = sync.renderComplete;
    signalSemInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkCommandBufferSubmitInfo cmdInfo{};
    cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmdInfo.commandBuffer = cmd;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.waitSemaphoreInfoCount = 1;
    submitInfo.pWaitSemaphoreInfos = &waitSemInfo;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdInfo;
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos = &signalSemInfo;

    VkResult submitResult = vkQueueSubmit2(device.mainQueue().queue, 1, &submitInfo, sync.inFlight);
    if (submitResult == VK_ERROR_DEVICE_LOST) {
        log::error("frame", "DEVICE_LOST on vkQueueSubmit2 — initiating shutdown");
        services_.bridge().emit(EvtDeviceLost{"vkQueueSubmit2 returned DEVICE_LOST"});
        deviceLost_ = true;
        return false;
    }

    // Present
    auto presentResult = swapchain.present(ctx.swapchainImageIndex, sync.renderComplete);
    if (!presentResult) {
        // OUT_OF_DATE/SUBOPTIMAL already handled inside present() by marking recreateNeeded
    }

    return false; // Frame not skipped
}

bool FrameScheduler::executeOffscreenFrame(
    vk2::DeviceService& device, vk2::SwapchainService& swapchain,
    OffscreenTarget& offscreen, FrameContext& ctx) {

    if (!syncInitialized_) return false;

    // Handle pending recreate
    if (swapchain.isRecreateNeeded()) {
        device.waitIdle();
        auto r = swapchain.recreate(0, 0);
        if (!r) {
            log::warn("frame", "Swapchain recreate failed: " + r.error().message);
        } else {
            swapchain.clearRecreateNeeded();
            imageCount_ = swapchain.imageCount();
            // Recreate offscreen targets to match new extent
            auto ext = swapchain.extent();
            auto offR = offscreen.recreate(device, ext.width, ext.height);
            if (!offR) log::error("frame", "Offscreen recreate failed: " + offR.error().message);
        }
        return true; // Frame skipped
    }

    auto& sync = frameSync_[frameIndex_];

    // Wait for this frame slot's previous submission
    vkWaitForFences(device.device(), 1, &sync.inFlight, VK_TRUE, UINT64_MAX);
    vkResetFences(device.device(), 1, &sync.inFlight);

    // Acquire swapchain image
    auto acquireResult = swapchain.acquireNextImage(sync.imageAcquired);
    if (!acquireResult) {
        if (acquireResult.error().vkResult == VK_ERROR_OUT_OF_DATE_KHR) {
            swapchain.markRecreateNeeded();
            return true;
        }
        log::warn("frame", "Acquire failed: " + acquireResult.error().message);
        return true;
    }
    ctx.swapchainImageIndex = acquireResult.value();

    // Record command buffer
    VkCommandBuffer cmd = sync.cmd;
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    services_.metrics().beginGpuFrame(cmd, ctx.frameIndex);

    // Cycling clear color
    float t = static_cast<float>(frameNumber_ % 360) / 360.0f;
    VkClearColorValue clearColor = {{
        0.5f * (1.0f + std::sin(t * 6.28318f)),
        0.5f * (1.0f + std::sin(t * 6.28318f + 2.094f)),
        0.5f * (1.0f + std::sin(t * 6.28318f + 4.189f)),
        1.0f
    }};

    auto& offImg = offscreen.image(ctx.frameIndex);
    VkImage offVkImg = offImg.handle();
    VkImage swapImage = swapchain.image(ctx.swapchainImageIndex);
    VkExtent2D swapExtent = swapchain.extent();

    // 1. Offscreen: UNDEFINED → TRANSFER_DST_OPTIMAL
    {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        barrier.srcAccessMask = VK_ACCESS_2_NONE;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = offVkImg;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // 2. Clear offscreen image
    {
        VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, offVkImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &clearColor, 1, &range);
    }

    // 3. Offscreen: TRANSFER_DST → TRANSFER_SRC_OPTIMAL
    {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = offVkImg;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // 4-5. Composite: blit offscreen → swapchain, transition to PRESENT_SRC
    CompositePass::record(cmd, offVkImg, offscreen.width(), offscreen.height(),
                          swapImage, swapExtent.width, swapExtent.height);

    services_.metrics().endGpuFrame(cmd, ctx.frameIndex);
    vkEndCommandBuffer(cmd);

    // Submit via vkQueueSubmit2
    VkSemaphoreSubmitInfo waitSemInfo{};
    waitSemInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    waitSemInfo.semaphore = sync.imageAcquired;
    waitSemInfo.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

    VkSemaphoreSubmitInfo signalSemInfo{};
    signalSemInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalSemInfo.semaphore = sync.renderComplete;
    signalSemInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkCommandBufferSubmitInfo cmdInfo{};
    cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmdInfo.commandBuffer = cmd;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.waitSemaphoreInfoCount = 1;
    submitInfo.pWaitSemaphoreInfos = &waitSemInfo;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdInfo;
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos = &signalSemInfo;

    VkResult submitResult = vkQueueSubmit2(device.mainQueue().queue, 1, &submitInfo, sync.inFlight);
    if (submitResult == VK_ERROR_DEVICE_LOST) {
        log::error("frame", "DEVICE_LOST on vkQueueSubmit2 — initiating shutdown");
        services_.bridge().emit(EvtDeviceLost{"vkQueueSubmit2 returned DEVICE_LOST"});
        deviceLost_ = true;
        return false;
    }

    auto presentResult = swapchain.present(ctx.swapchainImageIndex, sync.renderComplete);
    if (!presentResult) {
        // OUT_OF_DATE/SUBOPTIMAL handled inside present()
    }

    return false;
}

bool FrameScheduler::executeGraphFrame(
    vk2::DeviceService& device, vk2::SwapchainService& swapchain,
    FrameContext& ctx) {

    if (!syncInitialized_ || !graph_.valid()) return false;

    // Handle pending recreate
    if (swapchain.isRecreateNeeded()) {
        device.waitIdle();
        auto r = swapchain.recreate(0, 0);
        if (!r) {
            log::warn("frame", "Swapchain recreate failed: " + r.error().message);
        } else {
            swapchain.clearRecreateNeeded();
            imageCount_ = swapchain.imageCount();
        }
        return true;
    }

    auto ext = swapchain.extent();

    // Allocate or recreate graph resources if needed
    if (!resourcePool_.isAllocated()) {
        auto allocR = resourcePool_.allocate(device, graph_, ext.width, ext.height);
        if (!allocR) {
            log::error("frame", "Graph resource allocation failed: " + allocR.error().message);
            return true;
        }
    } else if (resourcePool_.renderWidth() != ext.width || resourcePool_.renderHeight() != ext.height) {
        device.waitIdle();
        auto recreR = resourcePool_.recreate(device, graph_, ext.width, ext.height);
        if (!recreR) {
            log::error("frame", "Graph resource recreate failed: " + recreR.error().message);
            return true;
        }
    }

    // Plan barriers
    auto resolved = resourcePool_.resolved();
    BarrierPlanner::plan(graph_, resolved.images);

    // Wire resolved resources into each pass
    for (auto& pass : graph_.passes) {
        pass.resources.resolved = &resolved;
    }

    auto& sync = frameSync_[frameIndex_];

    // Wait for this frame slot
    vkWaitForFences(device.device(), 1, &sync.inFlight, VK_TRUE, UINT64_MAX);
    vkResetFences(device.device(), 1, &sync.inFlight);

    // Acquire swapchain image
    auto acquireResult = swapchain.acquireNextImage(sync.imageAcquired);
    if (!acquireResult) {
        if (acquireResult.error().vkResult == VK_ERROR_OUT_OF_DATE_KHR) {
            swapchain.markRecreateNeeded();
            return true;
        }
        log::warn("frame", "Acquire failed: " + acquireResult.error().message);
        return true;
    }
    ctx.swapchainImageIndex = acquireResult.value();

    // Record command buffer
    VkCommandBuffer cmd = sync.cmd;
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    services_.metrics().beginGpuFrame(cmd, ctx.frameIndex);

    // Pre-graph scene processing (upload, BLAS, TLAS)
    if (preGraphFn_) preGraphFn_(cmd);

    // Execute all passes with their pre-barriers
    for (auto& pass : graph_.passes) {
        pass.resources.cmd = cmd;
        // Record pre-barriers
        if (!pass.preBarriers.imageBarriers.empty()) {
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = static_cast<uint32_t>(pass.preBarriers.imageBarriers.size());
            dep.pImageMemoryBarriers = pass.preBarriers.imageBarriers.data();
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        // Execute pass
        if (pass.execute) {
            pass.execute(ctx, pass.resources);
        }
    }

    // Composite: transition final output to TRANSFER_SRC, then blit to swapchain
    if (graph_.finalOutput.valid() && graph_.finalOutput.index < resolved.count) {
        VkImage finalImg = resolved.images[graph_.finalOutput.index];

        // Final output → TRANSFER_SRC_OPTIMAL
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = finalImg;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);

        auto& finalRes = graph_.resources[graph_.finalOutput.index];
        uint32_t srcW = finalRes.fixedWidth  ? finalRes.fixedWidth  : ext.width;
        uint32_t srcH = finalRes.fixedHeight ? finalRes.fixedHeight : ext.height;

        VkImage swapImage = swapchain.image(ctx.swapchainImageIndex);
        CompositePass::record(cmd, finalImg, srcW, srcH,
                              swapImage, ext.width, ext.height);
    }

    services_.metrics().endGpuFrame(cmd, ctx.frameIndex);
    vkEndCommandBuffer(cmd);

    // Submit
    VkSemaphoreSubmitInfo waitSemInfo{};
    waitSemInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    waitSemInfo.semaphore = sync.imageAcquired;
    waitSemInfo.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

    VkSemaphoreSubmitInfo signalSemInfo{};
    signalSemInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalSemInfo.semaphore = sync.renderComplete;
    signalSemInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkCommandBufferSubmitInfo cmdInfo{};
    cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmdInfo.commandBuffer = cmd;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.waitSemaphoreInfoCount = 1;
    submitInfo.pWaitSemaphoreInfos = &waitSemInfo;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdInfo;
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos = &signalSemInfo;

    VkResult submitResult = vkQueueSubmit2(device.mainQueue().queue, 1, &submitInfo, sync.inFlight);
    if (submitResult == VK_ERROR_DEVICE_LOST) {
        log::error("frame", "DEVICE_LOST on vkQueueSubmit2 — initiating shutdown");
        services_.bridge().emit(EvtDeviceLost{"vkQueueSubmit2 returned DEVICE_LOST"});
        deviceLost_ = true;
        return false;
    }

    auto presentResult = swapchain.present(ctx.swapchainImageIndex, sync.renderComplete);
    if (!presentResult) {
        // OUT_OF_DATE/SUBOPTIMAL handled inside present()
    }

    return false;
}

void FrameScheduler::setGraph(CompiledGraph graph) {
    // Release old resources before replacing graph
    if (resourcePool_.isAllocated()) {
        resourcePool_.release(gc_);
    }
    graph_ = std::move(graph);
    if (graph_.valid()) {
        log::info("frame", "Render graph set: " + std::to_string(graph_.passCount()) +
                  " passes, " + std::to_string(graph_.resourceCount()) + " resources");
    }
}

void FrameScheduler::endFrame(const FrameContext& ctx) {
    ++frameNumber_;
    frameIndex_ = (frameIndex_ + 1) % framesInFlight_;
    services_.bridge().emit(EvtFrameComplete{ctx.frameNumber, lastFrameTimeMs_});
}

} // namespace engine
