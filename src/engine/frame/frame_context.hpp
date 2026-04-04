#pragma once

// Ownership: Created by FrameScheduler at beginFrame(), consumed during the frame.
// Thread: Main thread. Passed to render passes by const ref.
//
// frameIndex: CPU frame slot (0..framesInFlight-1). Indexes sync objects.
// swapchainImageIndex: From vkAcquireNextImageKHR. Indexes swapchain images/views.
// These are NOT the same value in general.

#include "config/config_service.hpp"

#include <cstdint>
#include <memory>

namespace engine {

struct FrameContext {
    uint32_t frameIndex = 0;              // CPU frame slot (0..framesInFlight-1)
    uint32_t swapchainImageIndex = 0;     // From vkAcquireNextImageKHR
    uint64_t frameNumber = 0;             // Monotonic frame counter
    float deltaTime = 0.0f;              // Seconds since last frame
    std::shared_ptr<const ConfigSnapshot> config;
};

} // namespace engine
