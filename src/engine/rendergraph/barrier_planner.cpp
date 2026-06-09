#include "barrier_planner.hpp"

namespace engine {

void BarrierPlanner::plan(CompiledGraph& graph, const VkImage* resolvedImages) {
    uint32_t resourceCount = graph.resourceCount();

    // Track current layout per resource — starts UNDEFINED
    std::vector<VkImageLayout> currentLayout(resourceCount, VK_IMAGE_LAYOUT_UNDEFINED);

    for (auto& pass : graph.passes) {
        pass.preBarriers.imageBarriers.clear();

        // Process inputs — need read-appropriate layout
        for (const auto& input : pass.resources.inputs) {
            if (!input.valid() || input.index >= resourceCount) continue;

            VkImageLayout required = layoutForInput(pass.queue);
            if (currentLayout[input.index] != required) {
                VkImageMemoryBarrier2 barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                barrier.srcStageMask = stageForLayout(currentLayout[input.index]);
                barrier.srcAccessMask = accessForLayout(currentLayout[input.index], true);
                barrier.dstStageMask = stageForLayout(required);
                barrier.dstAccessMask = accessForLayout(required, false);
                barrier.oldLayout = currentLayout[input.index];
                barrier.newLayout = required;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = resolvedImages[input.index];
                barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

                pass.preBarriers.imageBarriers.push_back(barrier);
                currentLayout[input.index] = required;
            }
        }

        // Process outputs — need write-appropriate layout
        for (const auto& output : pass.resources.outputs) {
            if (!output.valid() || output.index >= resourceCount) continue;

            VkImageLayout required = layoutForOutput(pass.queue);
            if (currentLayout[output.index] != required) {
                VkImageMemoryBarrier2 barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                barrier.srcStageMask = stageForLayout(currentLayout[output.index]);
                barrier.srcAccessMask = accessForLayout(currentLayout[output.index], true);
                barrier.dstStageMask = stageForLayout(required);
                barrier.dstAccessMask = accessForLayout(required, true);
                barrier.oldLayout = currentLayout[output.index];
                barrier.newLayout = required;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = resolvedImages[output.index];
                barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

                pass.preBarriers.imageBarriers.push_back(barrier);
                currentLayout[output.index] = required;
            }
        }
    }
}

VkImageLayout BarrierPlanner::layoutForOutput(QueueAffinity queue) {
    switch (queue) {
        case QueueAffinity::Compute:  return VK_IMAGE_LAYOUT_GENERAL;
        case QueueAffinity::Transfer: return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        default:                      return VK_IMAGE_LAYOUT_GENERAL; // Storage write from compute or graphics
    }
}

VkImageLayout BarrierPlanner::layoutForInput(QueueAffinity queue) {
    switch (queue) {
        case QueueAffinity::Transfer: return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        default:                      return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
}

VkPipelineStageFlags2 BarrierPlanner::stageForLayout(VkImageLayout layout) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            return VK_PIPELINE_STAGE_2_NONE;
        case VK_IMAGE_LAYOUT_GENERAL:
            // GENERAL is used by compute, ray tracing, graphics, and transfer (vkCmdClearColorImage)
            return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                 | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR
                 | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
                 | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            return VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                 | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                 | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        default:
            return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
}

VkAccessFlags2 BarrierPlanner::accessForLayout(VkImageLayout layout, bool write) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            return VK_ACCESS_2_NONE;
        case VK_IMAGE_LAYOUT_GENERAL:
            return write ? (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT
                          | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)
                         : VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            return write ? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
                         : VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            return VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return VK_ACCESS_2_TRANSFER_READ_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return VK_ACCESS_2_TRANSFER_WRITE_BIT;
        default:
            return write ? VK_ACCESS_2_MEMORY_WRITE_BIT : VK_ACCESS_2_MEMORY_READ_BIT;
    }
}

} // namespace engine
