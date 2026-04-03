#pragma once

// Render graph type definitions shared across builder, compiler, and executor.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace engine {

struct FrameContext;

// --- Handles (indices into graph arrays) ---

struct PassHandle { uint32_t index = UINT32_MAX; bool valid() const { return index != UINT32_MAX; } };
struct ResourceHandle { uint32_t index = UINT32_MAX; bool valid() const { return index != UINT32_MAX; } };

// --- Queue affinity ---

enum class QueueAffinity {
    Graphics,  // Main queue (graphics + compute + transfer)
    Compute,   // Async compute (if available, else falls back to Graphics)
    Transfer   // DMA transfer (if available, else falls back to Graphics)
};

// --- Resource descriptors ---

struct ImageResourceDesc {
    std::string name;
    VkFormat format = VK_FORMAT_UNDEFINED;
    // Size is derived from render resolution unless overridden:
    float widthScale = 1.0f;     // Relative to render width
    float heightScale = 1.0f;    // Relative to render height
    uint32_t fixedWidth = 0;     // If non-zero, overrides widthScale
    uint32_t fixedHeight = 0;    // If non-zero, overrides heightScale
    VkImageUsageFlags extraUsage = 0;
    bool isTransient = false;    // Can alias memory with non-overlapping passes
};

// --- Pass descriptors ---

// PassResources is provided to the execute callback with resolved images.
struct PassResources {
    // Inputs and outputs are indices into the graph's resource array.
    // The executor resolves them to actual VkImage/VkImageView handles.
    std::vector<ResourceHandle> inputs;
    std::vector<ResourceHandle> outputs;
};

using PassExecuteFn = std::function<void(const FrameContext&, const PassResources&)>;

struct PassDescriptor {
    std::string name;
    std::vector<ResourceHandle> inputs;
    std::vector<ResourceHandle> outputs;
    QueueAffinity queue = QueueAffinity::Graphics;
    PassExecuteFn execute;  // Called during graph execution
};

// --- Compiled graph (output of GraphBuilder::compile) ---

struct CompiledPass {
    uint32_t passIndex;
    std::string name;
    QueueAffinity queue;
    PassResources resources;
    PassExecuteFn execute;
    // Barriers to insert before this pass (computed by compiler)
    // Will be populated when barrier planning is implemented.
};

struct CompiledGraph {
    std::vector<CompiledPass> passes;          // Topologically sorted
    std::vector<ImageResourceDesc> resources;  // All declared resources

    bool valid() const { return !passes.empty(); }
    uint32_t passCount() const { return static_cast<uint32_t>(passes.size()); }
    uint32_t resourceCount() const { return static_cast<uint32_t>(resources.size()); }
};

} // namespace engine
