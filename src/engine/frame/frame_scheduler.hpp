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

    // Override the render resolution used when allocating graph resources.
    // Call this BEFORE setGraph() when DLSS is active so that RT resources
    // (widthScale=1.0) are sized to the DLSS input (render) resolution rather
    // than the full swapchain (display) resolution. Pass 0,0 to use swapchain
    // extent (default, no-upscaling path).
    void setGraphRenderResolution(uint32_t w, uint32_t h) {
        graphRenderWidth_ = w;
        graphRenderHeight_ = h;
    }

    // Set a callback invoked with the active command buffer before graph passes.
    // Used for scene processing (upload, BLAS, TLAS builds).
    using PreGraphFn = std::function<void(VkCommandBuffer)>;
    void setPreGraphCallback(PreGraphFn fn) { preGraphFn_ = std::move(fn); }

    // Set a callback invoked after all graph passes complete but before composite/present.
    // Used by FrameGenAdapter to tag depth, motion, and HUD-less resources for DLSS-G.
    using PostGraphFn = std::function<void(VkCommandBuffer, const FrameContext&)>;
    void setPostGraphCallback(PostGraphFn fn) { postGraphFn_ = std::move(fn); }

    ResourceGC& gc() { return gc_; }
    GraphResourcePool& resourcePool() { return resourcePool_; }
    uint64_t frameNumber() const { return frameNumber_; }
    uint32_t currentFrameIndex() const { return frameIndex_; }
    uint32_t framesInFlight() const { return framesInFlight_; }
    float lastFrameTimeMs() const { return lastFrameTimeMs_; }
    bool isDeviceLost() const { return deviceLost_; }

    // Allocate a one-shot primary command buffer from the scheduler's pool,
    // begin recording with ONE_TIME_SUBMIT_BIT, and return it. Caller records
    // commands then calls endOneShotCommandBuffer() to submit + wait + free.
    // Used for init-time operations that need a command buffer outside the
    // main frame loop (e.g. NGX feature creation, one-off image uploads).
    // Returns VK_NULL_HANDLE on failure.
    VkCommandBuffer beginOneShotCommandBuffer();

    // Ends recording on `cmd`, submits it to the main queue, waits for
    // completion, and frees the command buffer. Must be paired with a prior
    // beginOneShotCommandBuffer() call.
    void endOneShotCommandBuffer(VkCommandBuffer cmd);

private:
    EngineServices& services_;
    // Ring size = framesInFlight_ (default 2) + 1 safety margin. Resources deferred
    // on frame N are freed after ringSize ticks, which guarantees all GPU work for
    // that slot has completed (the fence for slot N is waited each time it reuses the slot).
    // A size-2 ring is the minimum correct value for 2 frames in flight; we add +1
    // so that resources deferred during the current tick are not freed until the tick
    // AFTER the next fence wait, giving one extra frame of headroom for async work.
    ResourceGC gc_{3};  // 2 frames in flight + 1 safety margin
    CompiledGraph graph_;
    GraphResourcePool resourcePool_;
    // Persistent per-resource layout state for BarrierPlanner::plan().
    // Initialized to UNDEFINED on graph (re)build; reset when resources are
    // reallocated at a new resolution. Updated by the planner AND manually
    // for the final-output composite transition each frame.
    std::vector<VkImageLayout> graphLayout_;
    PreGraphFn preGraphFn_;
    PostGraphFn postGraphFn_;

    // waitForFrameFenceCheckDeviceLost: waits on the frame-in-flight fence, and
    // queries VK_EXT_device_fault (if enabled), emits EvtDeviceLost, sets
    // deviceLost_ = true, and returns false. On success, resets the fence and
    // returns true.
    bool waitForFrameFenceCheckDeviceLost(vk2::DeviceService& device, VkFence fence,
                                          const char* site);

    // Bug D fix: the renderComplete binary semaphore must be keyed to swapchain
    // image index, not frame-in-flight index. framesInFlight_ (2) and imageCount_
    // (3 on triple-buffered swapchains) are different, so a per-frame semaphore
    // gets double-signaled before the presentation engine consumes the previous
    // signal. (VUID-vkQueueSubmit2-semaphore-03868.) Creates/destroys a pool
    // sized to imageCount_. Safe to call multiple times; previous pool is
    // destroyed first. Requires syncDevice_ to be valid.
    vk2::Result<void> recreatePresentSemaphores();
    void destroyPresentSemaphores();

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
        VkFence inFlight = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
    };
    std::vector<FrameSync> frameSync_;
    // Per-swapchain-image present semaphore (fixes Bug D, VUID-03868).
    std::vector<VkSemaphore> renderCompleteByImage_;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkDevice syncDevice_ = VK_NULL_HANDLE;
    bool syncInitialized_ = false;
    bool deviceLost_ = false;

    // Render resolution override for graph resource allocation (DLSS upscaling path).
    // 0 means "use swapchain extent" (default). Set via setGraphRenderResolution().
    uint32_t graphRenderWidth_  = 0;
    uint32_t graphRenderHeight_ = 0;
};
} // namespace engine
