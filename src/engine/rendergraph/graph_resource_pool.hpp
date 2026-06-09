#pragma once

// Ownership: FrameScheduler (1:1).
// Thread: Main thread only.
// Dependencies: DeviceService (VkDevice, VMA), CompiledGraph.
//
// Allocates physical vk2::Image objects for each resource declared in a CompiledGraph.
// Usage flags are derived from how each resource is used across all passes.
// Recreated when render resolution changes.

#include "graph_types.hpp"
#include "platform/vulkan/vk2_image.hpp"
#include "platform/vulkan/vk2_result.hpp"

#include <cstdint>
#include <vector>

namespace engine {

namespace vk2 { class DeviceService; }
class ResourceGC;

class GraphResourcePool {
public:
    GraphResourcePool() = default;
    ~GraphResourcePool() = default;

    GraphResourcePool(const GraphResourcePool&) = delete;
    GraphResourcePool& operator=(const GraphResourcePool&) = delete;

    // Allocate images for all resources in the graph at the given render resolution.
    vk2::Result<void> allocate(vk2::DeviceService& device, const CompiledGraph& graph,
                               uint32_t renderWidth, uint32_t renderHeight);

    // Recreate all images at a new resolution. No-op if resolution unchanged.
    vk2::Result<void> recreate(vk2::DeviceService& device, const CompiledGraph& graph,
                               uint32_t renderWidth, uint32_t renderHeight);

    // Release all images via deferred GC (safe for in-flight frames).
    void release(ResourceGC& gc);

    // Release immediately (call only after device idle).
    void releaseImmediate();

    bool isAllocated() const { return !images_.empty(); }
    uint32_t renderWidth() const { return renderWidth_; }
    uint32_t renderHeight() const { return renderHeight_; }

    // Fill a ResolvedResources struct pointing into our backing arrays.
    ResolvedResources resolved() const;

    // Access individual images (by ResourceHandle index).
    vk2::Image& image(uint32_t index) { return images_[index]; }
    const vk2::Image& image(uint32_t index) const { return images_[index]; }

private:
    std::vector<vk2::Image> images_;
    std::vector<VkImage> imageHandles_;       // parallel array for ResolvedResources
    std::vector<VkImageView> imageViews_;     // parallel array for ResolvedResources
    uint32_t renderWidth_ = 0;
    uint32_t renderHeight_ = 0;

    static VkImageUsageFlags deriveUsage(const CompiledGraph& graph, uint32_t resourceIndex);
};

} // namespace engine
