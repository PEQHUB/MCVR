#include "composite_pass.hpp"

#include <volk.h>

namespace engine {

void CompositePass::record(VkCommandBuffer cmd,
                           VkImage srcImage, uint32_t srcWidth, uint32_t srcHeight,
                           VkImage dstImage, uint32_t dstWidth, uint32_t dstHeight) {
    // Transition swapchain: UNDEFINED → TRANSFER_DST_OPTIMAL
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
        barrier.image = dstImage;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // Blit src → dst
    {
        VkImageBlit2 region{};
        region.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.srcOffsets[0] = {0, 0, 0};
        region.srcOffsets[1] = {static_cast<int32_t>(srcWidth),
                                static_cast<int32_t>(srcHeight), 1};
        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstOffsets[0] = {0, 0, 0};
        region.dstOffsets[1] = {static_cast<int32_t>(dstWidth),
                                static_cast<int32_t>(dstHeight), 1};

        VkBlitImageInfo2 blitInfo{};
        blitInfo.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
        blitInfo.srcImage = srcImage;
        blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        blitInfo.dstImage = dstImage;
        blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        blitInfo.regionCount = 1;
        blitInfo.pRegions = &region;
        blitInfo.filter = VK_FILTER_LINEAR;

        vkCmdBlitImage2(cmd, &blitInfo);
    }

    // Transition swapchain: TRANSFER_DST → PRESENT_SRC_KHR
    {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        barrier.dstAccessMask = VK_ACCESS_2_NONE;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = dstImage;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
}

} // namespace engine
