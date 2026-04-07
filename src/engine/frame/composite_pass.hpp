#pragma once

// Ownership: FrameScheduler (or graph executor).
// Thread: Main thread only.
//
// Terminal node of every render graph — blits a source image to the swapchain
// and transitions the swapchain to PRESENT_SRC_KHR.
//
// Caller must ensure srcImage is in TRANSFER_SRC_OPTIMAL before calling record().

#include <vulkan/vulkan.h>

namespace engine {

class CompositePass {
public:
    // Record blit from src → dst (swapchain) plus layout transitions.
    // srcImage must already be in TRANSFER_SRC_OPTIMAL.
    // dstImage is transitioned: UNDEFINED → TRANSFER_DST → PRESENT_SRC.
    static void record(VkCommandBuffer cmd,
                       VkImage srcImage, uint32_t srcWidth, uint32_t srcHeight,
                       VkImage dstImage, uint32_t dstWidth, uint32_t dstHeight);
};

} // namespace engine
