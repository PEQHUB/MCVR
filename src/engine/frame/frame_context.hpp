#pragma once

// Ownership: Created by FrameScheduler at beginFrame(), consumed during the frame,
//            dropped at endFrame(). One instance per in-flight frame.
// Thread: Created on main thread. Passed to render passes (may cross threads
//         for async compute, but each pass gets its own command buffer).

#include "config/config_service.hpp"

#include <cstdint>
#include <memory>

namespace engine {

struct FrameContext {
    uint32_t frameIndex = 0;          // Swapchain image index (0..imageCount-1)
    uint64_t frameNumber = 0;         // Monotonic frame counter
    float deltaTime = 0.0f;           // Seconds since last frame
    std::shared_ptr<const ConfigSnapshot> config;  // Frozen config for this frame
};

} // namespace engine
