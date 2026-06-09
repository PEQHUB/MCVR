#include "graph_builder.hpp"
#include "diagnostics/log.hpp"

#include <algorithm>
#include <queue>
#include <unordered_set>

namespace engine {

ResourceHandle GraphBuilder::addResource(ImageResourceDesc desc) {
    uint32_t idx = static_cast<uint32_t>(resources_.size());
    resourceNameMap_[desc.name] = idx;
    resources_.push_back(std::move(desc));
    return {idx};
}

PassHandle GraphBuilder::addPass(PassDescriptor desc) {
    uint32_t idx = static_cast<uint32_t>(passes_.size());
    passes_.push_back(std::move(desc));
    return {idx};
}

void GraphBuilder::connect(ResourceHandle output, ResourceHandle input) {
    // "connect" means: pass A's output resource IS the same physical resource
    // as pass B's input. We redirect input references to point at the output
    // resource. Output arrays are never modified — each pass owns its outputs.
    if (output.index == input.index) return;  // Already the same resource

    // Redirect: all INPUT references to input.index → output.index
    for (auto& pass : passes_) {
        for (auto& r : pass.inputs) {
            if (r.index == input.index) r.index = output.index;
        }
        // Intentionally NOT touching pass.outputs — a pass's output identity
        // must never change, or two passes could silently share an output.
    }

    // Update name map so findResource() resolves to the canonical resource
    if (input.index < resources_.size()) {
        resourceNameMap_[resources_[input.index].name] = output.index;
    }
}

ResourceHandle GraphBuilder::findResource(const std::string& name) const {
    auto it = resourceNameMap_.find(name);
    if (it != resourceNameMap_.end()) return {it->second};
    return {};  // Invalid handle
}

void GraphBuilder::buildEdges(std::vector<Edge>& edges) const {
    // For each resource, find which pass(es) write it and which read it.
    // Writers must execute before readers.
    std::unordered_map<uint32_t, std::vector<uint32_t>> writers;  // resource → pass indices
    std::unordered_map<uint32_t, std::vector<uint32_t>> readers;

    for (uint32_t pi = 0; pi < passes_.size(); ++pi) {
        for (const auto& r : passes_[pi].outputs) writers[r.index].push_back(pi);
        for (const auto& r : passes_[pi].inputs) readers[r.index].push_back(pi);
    }

    for (const auto& [resIdx, writerPasses] : writers) {
        auto it = readers.find(resIdx);
        if (it == readers.end()) continue;
        for (uint32_t w : writerPasses) {
            for (uint32_t r : it->second) {
                if (w != r) {
                    edges.push_back({w, r});
                }
            }
        }
    }
}

CompiledGraph GraphBuilder::compile() const {
    CompiledGraph graph;
    graph.resources = resources_;
    graph.finalOutput = finalOutput_;

    if (passes_.empty()) return graph;

    // Build dependency edges
    std::vector<Edge> edges;
    buildEdges(edges);

    // Kahn's algorithm for topological sort
    uint32_t n = static_cast<uint32_t>(passes_.size());
    std::vector<uint32_t> inDegree(n, 0);
    std::vector<std::vector<uint32_t>> adj(n);

    for (const auto& e : edges) {
        adj[e.from].push_back(e.to);
        ++inDegree[e.to];
    }

    std::queue<uint32_t> q;
    for (uint32_t i = 0; i < n; ++i) {
        if (inDegree[i] == 0) q.push(i);
    }

    std::vector<uint32_t> order;
    order.reserve(n);
    while (!q.empty()) {
        uint32_t u = q.front(); q.pop();
        order.push_back(u);
        for (uint32_t v : adj[u]) {
            if (--inDegree[v] == 0) q.push(v);
        }
    }

    if (order.size() != n) {
        if (log::isInitialized()) {
            log::error("rendergraph", "Cycle detected in render graph — " +
                       std::to_string(order.size()) + "/" + std::to_string(n) + " passes sorted");
        }
        return {};  // Empty graph signals error
    }

    // Build compiled passes in sorted order
    for (uint32_t pi : order) {
        const auto& pass = passes_[pi];
        CompiledPass cp;
        cp.passIndex = pi;
        cp.name = pass.name;
        cp.queue = pass.queue;
        cp.resources.inputs = pass.inputs;
        cp.resources.outputs = pass.outputs;
        cp.execute = pass.execute;
        graph.passes.push_back(std::move(cp));
    }

    return graph;
}

PassHandle GraphBuilder::importModule(const std::string& moduleName,
                                       const std::vector<ImageResourceDesc>& inputs,
                                       const std::vector<ImageResourceDesc>& outputs,
                                       PassExecuteFn execute) {
    PassDescriptor pass;
    pass.name = moduleName;
    pass.execute = std::move(execute);

    // Create or find resources for inputs
    for (const auto& desc : inputs) {
        auto existing = findResource(desc.name);
        if (existing.valid()) {
            pass.inputs.push_back(existing);
        } else {
            pass.inputs.push_back(addResource(desc));
        }
    }

    // Create resources for outputs (always new — a module owns its outputs)
    for (const auto& desc : outputs) {
        auto existing = findResource(desc.name);
        if (existing.valid()) {
            pass.outputs.push_back(existing);
        } else {
            pass.outputs.push_back(addResource(desc));
        }
    }

    return addPass(std::move(pass));
}

} // namespace engine
