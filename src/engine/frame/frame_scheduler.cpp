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
#include <cstdio>
#include <sstream>
#include <vector>

namespace engine {

namespace {

// Query and log VK_EXT_device_fault information after a DEVICE_LOST result.
// Safe to call even when the extension is disabled (acts as a no-op).
// This is our primary diagnostic tool for understanding GPU timeouts — without
// it, a DEVICE_LOST is completely opaque.
void logDeviceFaultInfo(vk2::DeviceService& device, const char* site) {
    if (!device.caps().deviceFault) {
        log::warn("frame", std::string("device-fault info unavailable (VK_EXT_device_fault not enabled) at ") + site);
        return;
    }
    if (!vkGetDeviceFaultInfoEXT) {
        log::warn("frame", "vkGetDeviceFaultInfoEXT function pointer is null");
        return;
    }

    // First call: query counts. The spec says call once with nullptrs to get counts,
    // then a second time with allocated arrays.
    VkDeviceFaultCountsEXT counts{};
    counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
    VkResult r = vkGetDeviceFaultInfoEXT(device.device(), &counts, nullptr);
    if (r != VK_SUCCESS) {
        log::error("frame", std::string("vkGetDeviceFaultInfoEXT (count query) returned ") + std::to_string(r));
        return;
    }

    std::vector<VkDeviceFaultAddressInfoEXT> addressInfos(counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT> vendorInfos(counts.vendorInfoCount);
    std::vector<uint8_t> vendorBinary(counts.vendorBinarySize);

    VkDeviceFaultInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
    info.pAddressInfos = addressInfos.empty() ? nullptr : addressInfos.data();
    info.pVendorInfos = vendorInfos.empty() ? nullptr : vendorInfos.data();
    info.pVendorBinaryData = vendorBinary.empty() ? nullptr : vendorBinary.data();

    r = vkGetDeviceFaultInfoEXT(device.device(), &counts, &info);
    if (r != VK_SUCCESS) {
        log::error("frame", std::string("vkGetDeviceFaultInfoEXT (data query) returned ") + std::to_string(r));
        return;
    }

    log::error("frame", std::string("=== DEVICE_FAULT at ") + site + " ===");
    log::error("frame", std::string("description: ") + info.description);
    log::error("frame", std::string("addressInfoCount=") + std::to_string(counts.addressInfoCount)
                      + " vendorInfoCount=" + std::to_string(counts.vendorInfoCount)
                      + " vendorBinarySize=" + std::to_string(counts.vendorBinarySize));

    for (uint32_t i = 0; i < counts.addressInfoCount; ++i) {
        const auto& a = addressInfos[i];
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "  address[%u] type=%u reportedAddress=0x%llx granularity=0x%llx",
            i, static_cast<uint32_t>(a.addressType),
            static_cast<unsigned long long>(a.reportedAddress),
            static_cast<unsigned long long>(a.addressPrecision));
        log::error("frame", buf);
    }
    for (uint32_t i = 0; i < counts.vendorInfoCount; ++i) {
        const auto& v = vendorInfos[i];
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "  vendor[%u] code=0x%llx data=0x%llx desc=%.200s",
            i, static_cast<unsigned long long>(v.vendorFaultCode),
            static_cast<unsigned long long>(v.vendorFaultData),
            v.description);
        log::error("frame", buf);
    }
    log::error("frame", "=== end DEVICE_FAULT ===");

    // NV GPU Checkpoints — last completed marker before DEVICE_LOST
    std::string lastCheckpoint;
    if (device.caps().nvCheckpoints && vkGetQueueCheckpointDataNV) {
        uint32_t cpCount = 0;
        vkGetQueueCheckpointDataNV(device.mainQueue().queue, &cpCount, nullptr);
        if (cpCount > 0) {
            std::vector<VkCheckpointDataNV> cpData(cpCount);
            for (auto& d : cpData) d.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV;
            vkGetQueueCheckpointDataNV(device.mainQueue().queue, &cpCount, cpData.data());
            if (!cpData.empty() && cpData.back().pCheckpointMarker) {
                lastCheckpoint = static_cast<const char*>(cpData.back().pCheckpointMarker);
                log::error("frame", "last NV checkpoint: " + lastCheckpoint);
            }
        }
    }

    log::event("frame", "device_lost", {
        {"site",             std::string(site)},
        {"addressInfoCount", std::to_string(counts.addressInfoCount)},
        {"vendorInfoCount",  std::to_string(counts.vendorInfoCount)},
        {"lastCheckpoint",   lastCheckpoint}
    });
}

} // namespace

// Wait on the in-flight fence and check for DEVICE_LOST. Returns false if the
// device was lost (caller should emit EvtDeviceLost and return), true on success.
bool FrameScheduler::waitForFrameFenceCheckDeviceLost(vk2::DeviceService& device, VkFence fence,
                                                      const char* site) {
    VkResult r = vkWaitForFences(device.device(), 1, &fence, VK_TRUE, UINT64_MAX);
    if (r == VK_ERROR_DEVICE_LOST) {
        log::error("frame", std::string("DEVICE_LOST on vkWaitForFences at ") + site);
        logDeviceFaultInfo(device, site);
        services_.bridge().emit(EvtDeviceLost{std::string("vkWaitForFences returned DEVICE_LOST at ") + site});
        deviceLost_ = true;
        return false;
    }
    if (r != VK_SUCCESS) {
        log::warn("frame", std::string("vkWaitForFences returned ") + std::to_string(r) + " at " + site);
    }
    vkResetFences(device.device(), 1, &fence);
    return true;
}

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

    // Per-frame sync (imageAcquired + inFlight fence). renderComplete is pooled
    // per swapchain image, not per frame-in-flight — see recreatePresentSemaphores.
    for (uint32_t i = 0; i < framesInFlight_; ++i) {
        frameSync_[i].cmd = cmds[i];

        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        r = vkCreateSemaphore(syncDevice_, &semInfo, nullptr, &frameSync_[i].imageAcquired);
        if (r != VK_SUCCESS) return vk2::makeError(r, "vkCreateSemaphore (acquire) failed");

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        r = vkCreateFence(syncDevice_, &fenceInfo, nullptr, &frameSync_[i].inFlight);
        if (r != VK_SUCCESS) return vk2::makeError(r, "vkCreateFence failed");
    }

    // Per-swapchain-image present semaphore pool (Bug D fix, VUID-03868).
    auto presSem = recreatePresentSemaphores();
    if (!presSem) return presSem;

    syncInitialized_ = true;
    log::info("frame", "Sync objects created: " + std::to_string(framesInFlight_) +
              " frames in flight, " + std::to_string(imageCount_) + " swapchain images (" +
              std::to_string(renderCompleteByImage_.size()) + " present semaphores)");
    return {};
}

vk2::Result<void> FrameScheduler::recreatePresentSemaphores() {
    if (syncDevice_ == VK_NULL_HANDLE) {
        return vk2::makeError(VK_ERROR_INITIALIZATION_FAILED,
                              "recreatePresentSemaphores called before syncDevice_ is valid");
    }
    // Destroy any existing semaphores first (safe no-op if pool is empty).
    destroyPresentSemaphores();

    const uint32_t count = imageCount_ > 0 ? imageCount_ : 3;
    renderCompleteByImage_.assign(count, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < count; ++i) {
        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkResult r = vkCreateSemaphore(syncDevice_, &semInfo, nullptr, &renderCompleteByImage_[i]);
        if (r != VK_SUCCESS) {
            return vk2::makeError(r, "vkCreateSemaphore (renderCompleteByImage[" + std::to_string(i) + "]) failed");
        }
    }
    return {};
}

void FrameScheduler::destroyPresentSemaphores() {
    if (syncDevice_ == VK_NULL_HANDLE) {
        renderCompleteByImage_.clear();
        return;
    }
    for (auto s : renderCompleteByImage_) {
        if (s != VK_NULL_HANDLE) vkDestroySemaphore(syncDevice_, s, nullptr);
    }
    renderCompleteByImage_.clear();
}

void FrameScheduler::shutdownSync() {
    if (!syncInitialized_ || syncDevice_ == VK_NULL_HANDLE) return;

    for (auto& fs : frameSync_) {
        if (fs.imageAcquired != VK_NULL_HANDLE)
            vkDestroySemaphore(syncDevice_, fs.imageAcquired, nullptr);
        if (fs.inFlight != VK_NULL_HANDLE)
            vkDestroyFence(syncDevice_, fs.inFlight, nullptr);
    }
    frameSync_.clear();

    destroyPresentSemaphores();

    if (cmdPool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(syncDevice_, cmdPool_, nullptr);
        cmdPool_ = VK_NULL_HANDLE;
    }

    syncInitialized_ = false;
}

VkCommandBuffer FrameScheduler::beginOneShotCommandBuffer() {
    if (!syncInitialized_ || cmdPool_ == VK_NULL_HANDLE || syncDevice_ == VK_NULL_HANDLE) {
        log::warn("frame", "beginOneShotCommandBuffer: scheduler sync not initialized");
        return VK_NULL_HANDLE;
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmdPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(syncDevice_, &allocInfo, &cmd) != VK_SUCCESS) {
        log::warn("frame", "beginOneShotCommandBuffer: vkAllocateCommandBuffers failed");
        return VK_NULL_HANDLE;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
        log::warn("frame", "beginOneShotCommandBuffer: vkBeginCommandBuffer failed");
        vkFreeCommandBuffers(syncDevice_, cmdPool_, 1, &cmd);
        return VK_NULL_HANDLE;
    }

    return cmd;
}

void FrameScheduler::endOneShotCommandBuffer(VkCommandBuffer cmd) {
    if (cmd == VK_NULL_HANDLE) {
        return;
    }
    if (!syncInitialized_ || cmdPool_ == VK_NULL_HANDLE || syncDevice_ == VK_NULL_HANDLE) {
        log::warn("frame", "endOneShotCommandBuffer: scheduler sync not initialized");
        return;
    }

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        log::warn("frame", "endOneShotCommandBuffer: vkEndCommandBuffer failed");
        vkFreeCommandBuffers(syncDevice_, cmdPool_, 1, &cmd);
        return;
    }

    // Submit synchronously — we wait on the main queue after submission so the
    // caller can free any resources the recorded commands depended on.
    VkCommandBufferSubmitInfo cmdSubmitInfo{};
    cmdSubmitInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmdSubmitInfo.commandBuffer = cmd;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdSubmitInfo;

    VkQueue queue = services_.device().mainQueue().queue;
    if (vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        log::warn("frame", "endOneShotCommandBuffer: vkQueueSubmit2 failed");
    } else {
        // Wait for the submission to complete. Acceptable because one-shot
        // buffers are only used for init-time operations on the critical path.
        vkQueueWaitIdle(queue);
    }

    vkFreeCommandBuffers(syncDevice_, cmdPool_, 1, &cmd);
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

    auto configSnap = services_.config().snapshot();
    // NOTE: bridge().flush() and gc_.tick() are intentionally NOT called here.
    // Both are called inside executeGraphFrame() / executeClearFrame(), immediately
    // after the fence wait, so that:
    //  - bridge().flush() writes to all per-frame-slot SSBOs (textureMappingSSBOs_,
    //    areaLightSSBOs_, etc.) only after the current slot fence is waited, so the
    //    GPU for the other slot is reading its own independent copy — no WAR hazard.
    //  - gc_.tick() only fires deferred destructors after the GPU has finished the
    //    corresponding frame slot — preventing use-after-free of deferred resources.

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
            recreatePresentSemaphores();
        }
        return true; // Frame skipped
    }

    auto& sync = frameSync_[frameIndex_];

    // Wait for this frame slot's previous submission
    if (!waitForFrameFenceCheckDeviceLost(device, sync.inFlight, "waitForFences")) {
        return false;
    }

    // Tick GC after fence wait (same reasoning as executeGraphFrame — must be post-fence).
    gc_.tick();

    // Flush bridge commands after the fence wait (same WAR-hazard reasoning as executeGraphFrame).
    services_.bridge().flush();

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

    // Solid black — menu/pre-world background.
    VkClearColorValue clearColor = {{ 0.0f, 0.0f, 0.0f, 1.0f }};

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
    signalSemInfo.semaphore = renderCompleteByImage_[ctx.swapchainImageIndex];
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
        logDeviceFaultInfo(device, "executeClearFrame:vkQueueSubmit2");
        services_.bridge().emit(EvtDeviceLost{"vkQueueSubmit2 returned DEVICE_LOST"});
        deviceLost_ = true;
        return false;
    }

    // Present
    auto presentResult = swapchain.present(ctx.swapchainImageIndex, renderCompleteByImage_[ctx.swapchainImageIndex]);
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
            recreatePresentSemaphores();
            // Recreate offscreen targets to match new extent
            auto ext = swapchain.extent();
            auto offR = offscreen.recreate(device, ext.width, ext.height);
            if (!offR) log::error("frame", "Offscreen recreate failed: " + offR.error().message);
        }
        return true; // Frame skipped
    }

    auto& sync = frameSync_[frameIndex_];

    // Wait for this frame slot's previous submission
    if (!waitForFrameFenceCheckDeviceLost(device, sync.inFlight, "waitForFences")) {
        return false;
    }

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
    signalSemInfo.semaphore = renderCompleteByImage_[ctx.swapchainImageIndex];
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
        logDeviceFaultInfo(device, "executeOffscreenFrame:vkQueueSubmit2");
        services_.bridge().emit(EvtDeviceLost{"vkQueueSubmit2 returned DEVICE_LOST"});
        deviceLost_ = true;
        return false;
    }

    auto presentResult = swapchain.present(ctx.swapchainImageIndex, renderCompleteByImage_[ctx.swapchainImageIndex]);
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
            recreatePresentSemaphores();
        }
        return true;
    }

    auto ext = swapchain.extent();

    // Compute the render resolution for graph resource allocation.
    // When DLSS upscaling is active, graphRenderWidth_/Height_ is set to the DLSS input
    // (render) resolution (e.g. 50% of display size for Performance mode). Resources that
    // use widthScale/heightScale=1.0 (e.g. all RT outputs) are allocated at render res.
    // Resources with fixedWidth/fixedHeight (e.g. the DLSS output) override this and
    // always get their explicit size. Pass 0,0 → use swapchain extent (no upscaling).
    uint32_t allocW = (graphRenderWidth_  > 0) ? graphRenderWidth_  : ext.width;
    uint32_t allocH = (graphRenderHeight_ > 0) ? graphRenderHeight_ : ext.height;

    // Allocate or recreate graph resources if needed
    if (!resourcePool_.isAllocated()) {
        auto allocR = resourcePool_.allocate(device, graph_, allocW, allocH);
        if (!allocR) {
            log::error("frame", "Graph resource allocation failed: " + allocR.error().message);
            return true;
        }
        // Fresh allocation → all layouts UNDEFINED
        graphLayout_.assign(graph_.resourceCount(), VK_IMAGE_LAYOUT_UNDEFINED);
    } else if (resourcePool_.renderWidth() != allocW || resourcePool_.renderHeight() != allocH) {
        device.waitIdle();
        auto recreR = resourcePool_.recreate(device, graph_, allocW, allocH, gc_);
        if (!recreR) {
            log::error("frame", "Graph resource recreate failed: " + recreR.error().message);
            return true;
        }
        // Resources reallocated → layouts reset
        graphLayout_.assign(graph_.resourceCount(), VK_IMAGE_LAYOUT_UNDEFINED);
    }

    // Plan barriers (persistent layout state across frames)
    auto resolved = resourcePool_.resolved();
    BarrierPlanner::plan(graph_, resolved.images, graphLayout_);

    // Wire resolved resources into each pass
    for (auto& pass : graph_.passes) {
        pass.resources.resolved = &resolved;
    }

    auto& sync = frameSync_[frameIndex_];

    // Wait for this frame slot
    if (!waitForFrameFenceCheckDeviceLost(device, sync.inFlight, "executeGraphFrame:waitForFences")) {
        return false;
    }

    // Flush bridge commands NOW — after the fence wait — so that CPU writes to
    // per-frame-slot SSBOs (textureMappingSSBOs_, areaLightSSBOs_, etc.) are safe:
    // each slot holds an independent copy so writing slot N while slot N-1's GPU is
    // still reading does not cause a write-after-read (WAR) hazard.
    services_.bridge().flush();

    // Tick the GC ring now that the fence for this slot has been waited.
    // Resources deferred in the previous use of this slot (Frame N-2) are guaranteed
    // done by the GPU; the ring will free items from ringSize frames ago.
    // This placement is critical: gc_.tick() MUST run after the fence wait, not before,
    // so that deferred destructors never fire while the matching GPU work is still in flight.
    gc_.tick();

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

    // Post-graph callback: invoked after all passes but before composite/present.
    // FrameGenAdapter uses this to tag depth, motion, and HUD-less resources for DLSS-G.
    if (postGraphFn_) postGraphFn_(cmd, ctx);

    // Composite: transition final output to TRANSFER_SRC, then blit to swapchain
    if (graph_.finalOutput.valid() && graph_.finalOutput.index < resolved.count) {
        VkImage finalImg = resolved.images[graph_.finalOutput.index];

        // Final output → TRANSFER_SRC_OPTIMAL
        // oldLayout is read from graphLayout_ (maintained per-frame by BarrierPlanner).
        // The final pass may leave the image in COLOR_ATTACHMENT_OPTIMAL (raster pass)
        // or GENERAL (compute/RT pass) — hardcoding GENERAL causes a validation error
        // and undefined blit results when a raster pass (PostRender, StarField) is last.
        VkImageLayout finalCurrentLayout = VK_IMAGE_LAYOUT_GENERAL;
        if (graph_.finalOutput.index < graphLayout_.size()) {
            finalCurrentLayout = graphLayout_[graph_.finalOutput.index];
        }

        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        barrier.oldLayout = finalCurrentLayout;
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

        // Tell next frame's BarrierPlanner about the new layout of the final
        // output so it can transition it back to GENERAL before reuse.
        if (graph_.finalOutput.index < graphLayout_.size()) {
            graphLayout_[graph_.finalOutput.index] = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        }

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
    signalSemInfo.semaphore = renderCompleteByImage_[ctx.swapchainImageIndex];
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
        logDeviceFaultInfo(device, "executeGraphFrame:vkQueueSubmit2");
        services_.bridge().emit(EvtDeviceLost{"vkQueueSubmit2 returned DEVICE_LOST"});
        deviceLost_ = true;
        return false;
    }

    auto presentResult = swapchain.present(ctx.swapchainImageIndex, renderCompleteByImage_[ctx.swapchainImageIndex]);
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
    // Reset persistent layout tracking — new graph, new layouts.
    graphLayout_.assign(graph_.resourceCount(), VK_IMAGE_LAYOUT_UNDEFINED);
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
