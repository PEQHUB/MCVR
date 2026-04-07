#pragma once

// Ownership: Transient — called once per graph compilation or per-frame.
// Thread: Main thread only.
//
// Walks a CompiledGraph with resolved physical images and inserts
// VkImageMemoryBarrier2 batches into each pass's preBarriers.
// Tracks per-resource layout state: UNDEFINED at start, transitions based on
// whether a pass reads or writes each resource.

#include "graph_types.hpp"

#include <cstdint>
#include <vector>

namespace engine {

class BarrierPlanner {
public:
    // Plan barriers for all passes in the graph.
    // resolvedImages: VkImage handles indexed by ResourceHandle.
    // Populates each CompiledPass::preBarriers.
    static void plan(CompiledGraph& graph, const VkImage* resolvedImages);

private:
    // What layout a resource needs when used as an input or output
    static VkImageLayout layoutForOutput(QueueAffinity queue);
    static VkImageLayout layoutForInput(QueueAffinity queue);

    static VkPipelineStageFlags2 stageForLayout(VkImageLayout layout);
    static VkAccessFlags2 accessForLayout(VkImageLayout layout, bool write);
};

} // namespace engine
