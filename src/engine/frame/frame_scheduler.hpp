#pragma once

// Ownership: EngineServices (1:1).
// Thread: Main thread only. No method is thread-safe.
// Dependencies: ConfigService (for snapshot), BridgeService (for flush),
//               ResourceGC (for tick), DeviceService + SwapchainService (for V2 frames).

#include "frame_context.hpp"
#include "resource_gc.hpp"
#include "rendergraph/graph_types.hpp"
#include "platform/vulkan/vk2_result.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine {

class EngineServices;

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

    // Create per-frame sync objects (semaphores + fences + command pool).
    vk2::Result<void> initSync(vk2::DeviceService& device);
    void shutdownSync();

    // --- Frame lifecycle ---

    FrameContext beginFrame();

    // Record and submit a vkCmdClearColorImage + present. Proof of life.
    void executeClearFrame(vk2::DeviceService& device, vk2::SwapchainService& swapchain,
                           const FrameContext& ctx);

    void executeGraph(const FrameContext& ctx);

    void endFrame(const FrameContext& ctx);

    void setGraph(CompiledGraph graph);
    bool hasGraph() const { return graph_.valid(); }

    ResourceGC& gc() { return gc_; }
    uint64_t frameNumber() const { return frameNumber_; }
    float lastFrameTimeMs() const { return lastFrameTimeMs_; }

private:
    EngineServices& services_;
    ResourceGC gc_{32};
    CompiledGraph graph_;

    uint64_t frameNumber_ = 0;
    uint32_t frameIndex_ = 0;
    uint32_t imageCount_ = 3;

    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point lastFrameTime_;
    float lastFrameTimeMs_ = 0.0f;
    bool firstFrame_ = true;

    // Per-frame sync objects (sized to imageCount_)
    struct FrameSync {
        VkSemaphore imageAcquired = VK_NULL_HANDLE;
        VkSemaphore renderComplete = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
    };
    std::vector<FrameSync> frameSync_;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkDevice syncDevice_ = VK_NULL_HANDLE;  // Non-owning, for cleanup
    bool syncInitialized_ = false;
};

} // namespace engine
