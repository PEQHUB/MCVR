#pragma once

// Ownership: Transient — called once per frame.
// Thread: Main thread only.
//
// Walks a CompiledGraph with resolved physical images and inserts
// VkImageMemoryBarrier2 batches into each pass's preBarriers.
// Tracks per-resource layout state across frames so passes can read history
// buffers that were written in the previous frame.
// Caller owns the currentLayout vector and must initialize it to UNDEFINED
// on first frame / graph recreate, and update it for any layout transitions
// that happen OUTSIDE the planned passes (e.g. composite blit).

#include "graph_types.hpp"

#include <cstdint>
#include <vector>

namespace engine {

class BarrierPlanner {
public:
    // Plan barriers for all passes in the graph.
    // resolvedImages: VkImage handles indexed by ResourceHandle.
    // currentLayout: in/out per-resource layout tracking. Size must match
    //                graph.resourceCount(). Initialize to UNDEFINED on graph
    //                (re)build. Updated in-place as passes are planned.
    // Populates each CompiledPass::preBarriers.
    static void plan(CompiledGraph& graph,
                     const VkImage* resolvedImages,
                     std::vector<VkImageLayout>& currentLayout);

private:
    // What layout a resource needs when used as an input or output
    static VkImageLayout layoutForOutput(QueueAffinity queue);
    static VkImageLayout layoutForInput(QueueAffinity queue);

    static VkPipelineStageFlags2 stageForLayout(VkImageLayout layout);
    static VkAccessFlags2 accessForLayout(VkImageLayout layout, bool write);
};

} // namespace engine
