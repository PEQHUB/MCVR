#pragma once

// Ownership: EngineServices (1:1).
// Thread: Main thread only. No method is thread-safe.
// Dependencies: ConfigService (snapshot), BridgeService (flush),
//               ResourceGC, DeviceService + SwapchainService (V2 frames).
//
// Sync objects are sized to framesInFlight (CPU frame slots), NOT imageCount.
// frameIndex rotates through 0..framesInFlight-1 each frame.
// swapchainImageIndex comes from vkAcquireNextImageKHR and indexes swapchain images.

#include "frame_context.hpp"
#include "resource_gc.hpp"
#include "rendergraph/graph_types.hpp"
#include "rendergraph/graph_resource_pool.hpp"
#include "platform/vulkan/vk2_result.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine {

class EngineServices;
class OffscreenTarget;

namespace vk2 {
class DeviceService;
class SwapchainService;
}

class FrameScheduler {
public:
    explicit FrameScheduler(EngineServices& services);
    ~FrameScheduler();

    FrameScheduler(const FrameScheduler&) = delete;
    FrameScheduler& operator=(const FrameScheduler&) = delete;

    // --- Sync init/shutdown (V2 mode only) ---

    void setImageCount(uint32_t count) { imageCount_ = count; }

    // Create per-frame sync objects. framesInFlight <= imageCount.
    vk2::Result<void> initSync(vk2::DeviceService& device, uint32_t framesInFlight);
    void shutdownSync();

    // --- Frame lifecycle ---

    FrameContext beginFrame();

    // Acquire, render clear, present. Returns true if frame was skipped (recreate).
    bool executeClearFrame(vk2::DeviceService& device, vk2::SwapchainService& swapchain,
                           FrameContext& ctx);

    // Acquire, clear offscreen, blit to swapchain, present. Returns true if skipped.
    bool executeOffscreenFrame(vk2::DeviceService& device, vk2::SwapchainService& swapchain,
                               OffscreenTarget& offscreen, FrameContext& ctx);

    // Execute the render graph: allocate resources, plan barriers, run passes, composite.
    // Returns true if frame was skipped.
    bool executeGraphFrame(vk2::DeviceService& device, vk2::SwapchainService& swapchain,
                           FrameContext& ctx);

    void endFrame(const FrameContext& ctx);

    void setGraph(CompiledGraph graph);
    bool hasGraph() const { return graph_.valid(); }

    // Set a callback invoked with the active command buffer before graph passes.
    // Used for scene processing (upload, BLAS, TLAS builds).
    using PreGraphFn = std::function<void(VkCommandBuffer)>;
    void setPreGraphCallback(PreGraphFn fn) { preGraphFn_ = std::move(fn); }

    ResourceGC& gc() { return gc_; }
    GraphResourcePool& resourcePool() { return resourcePool_; }
    uint64_t frameNumber() const { return frameNumber_; }
    uint32_t currentFrameIndex() const { return frameIndex_; }
    uint32_t framesInFlight() const { return framesInFlight_; }
    float lastFrameTimeMs() const { return lastFrameTimeMs_; }
    bool isDeviceLost() const { return deviceLost_; }

private:
    EngineServices& services_;
    ResourceGC gc_{32};
    CompiledGraph graph_;
    GraphResourcePool resourcePool_;
    PreGraphFn preGraphFn_;

    uint64_t frameNumber_ = 0;
    uint32_t frameIndex_ = 0;
    uint32_t framesInFlight_ = 2;
    uint32_t imageCount_ = 3;

    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point lastFrameTime_;
    float lastFrameTimeMs_ = 0.0f;
    bool firstFrame_ = true;

    struct FrameSync {
        VkSemaphore imageAcquired = VK_NULL_HANDLE;
        VkSemaphore renderComplete = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
    };
    std::vector<FrameSync> frameSync_;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkDevice syncDevice_ = VK_NULL_HANDLE;
    bool syncInitialized_ = false;
    bool deviceLost_ = false;
};

} // namespace engine
