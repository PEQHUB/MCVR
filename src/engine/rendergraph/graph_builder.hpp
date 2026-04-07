#pragma once

// Ownership: Created per-pipeline-build (not per-frame).
// Thread: Main thread only during pipeline setup.
// Dependencies: None (pure data builder).

#include "graph_types.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace engine {

// Builds a render graph declaratively, then compiles it into an execution order.
//
// Usage:
//   GraphBuilder builder;
//   auto radiance = builder.addResource({.name="radiance", .format=...});
//   auto denoised = builder.addResource({.name="denoised", .format=...});
//   auto rtPass = builder.addPass({.name="ray_tracing", .outputs={radiance}});
//   auto dlssPass = builder.addPass({.name="dlss", .inputs={radiance}, .outputs={denoised}});
//   auto graph = builder.compile();
//
class GraphBuilder {
public:
    GraphBuilder() = default;

    // Declare a resource (image). Returns a handle for use in passes.
    ResourceHandle addResource(ImageResourceDesc desc);

    // Declare a pass. Returns a handle. Inputs/outputs reference resource handles.
    PassHandle addPass(PassDescriptor desc);

    // Connect an output resource of one pass to an input resource of another.
    // This is the YAML "connect(outputConfig, inputConfig)" equivalent.
    void connect(ResourceHandle output, ResourceHandle input);

    // Set which resource is the final output (composited to swapchain).
    void setFinalOutput(ResourceHandle handle) { finalOutput_ = handle; }

    // Look up a resource by name. Returns invalid handle if not found.
    ResourceHandle findResource(const std::string& name) const;

    // Compile the graph into topologically sorted execution order.
    // Returns empty graph on cycle or validation error.
    CompiledGraph compile() const;

    // Import a module from YAML-style config (matching existing pipeline YAML format).
    // Creates resources for all declared inputs/outputs and a pass for the module.
    // Returns the pass handle. Resources are accessible via findResource(name).
    PassHandle importModule(const std::string& moduleName,
                            const std::vector<ImageResourceDesc>& inputs,
                            const std::vector<ImageResourceDesc>& outputs,
                            PassExecuteFn execute = nullptr);

    uint32_t passCount() const { return static_cast<uint32_t>(passes_.size()); }
    uint32_t resourceCount() const { return static_cast<uint32_t>(resources_.size()); }

private:
    std::vector<ImageResourceDesc> resources_;
    std::vector<PassDescriptor> passes_;
    std::unordered_map<std::string, uint32_t> resourceNameMap_;
    ResourceHandle finalOutput_;

    // Adjacency for topological sort: pass → set of passes it depends on
    // (derived from shared resources: if pass A outputs R and pass B inputs R, B depends on A)
    struct Edge { uint32_t from; uint32_t to; };
    std::vector<Edge> edges_;

    void buildEdges(std::vector<Edge>& edges) const;
};

} // namespace engine
