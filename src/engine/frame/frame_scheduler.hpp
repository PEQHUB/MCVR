#pragma once

// Ownership: EngineServices (1:1).
// Thread: Main thread only. No method is thread-safe.
// Dependencies: ConfigService (for snapshot), BridgeService (for flush),
//               ResourceGC (for tick).

#include "frame_context.hpp"
#include "resource_gc.hpp"

#include <chrono>
#include <cstdint>
#include <memory>

namespace engine {

class EngineServices;

// Orchestrates the per-frame lifecycle:
//   beginFrame() → [render passes] → endFrame()
//
// Currently a no-op loop (no Vulkan submission). The frame shell proves
// the scheduling contract before real rendering is wired in.
class FrameScheduler {
public:
    explicit FrameScheduler(EngineServices& services);
    ~FrameScheduler();

    FrameScheduler(const FrameScheduler&) = delete;
    FrameScheduler& operator=(const FrameScheduler&) = delete;

    // Begin a new frame. Flushes bridge, snapshots config, ticks GC.
    // Returns the FrameContext for this frame.
    FrameContext beginFrame();

    // End the current frame. Updates timing, emits events.
    void endFrame(const FrameContext& ctx);

    // Access the GC (for deferring resource destruction).
    ResourceGC& gc() { return gc_; }

    uint64_t frameNumber() const { return frameNumber_; }
    float lastFrameTimeMs() const { return lastFrameTimeMs_; }

private:
    EngineServices& services_;
    ResourceGC gc_{32};

    uint64_t frameNumber_ = 0;
    uint32_t frameIndex_ = 0;       // Round-robin swapchain index (placeholder)
    uint32_t imageCount_ = 3;       // Default triple-buffer assumption

    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point lastFrameTime_;
    float lastFrameTimeMs_ = 0.0f;
    bool firstFrame_ = true;
};

} // namespace engine
