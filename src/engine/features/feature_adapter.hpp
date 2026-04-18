#pragma once

// Base interface for vendor feature adapters (DLSS, FSR, NRD, tone mapping, etc.).
// Each adapter wraps a legacy WorldModule as a render graph pass.
//
// Ownership: Render graph owns via PassDescriptor. EngineServices provides DI.
// Thread: execute() runs on the queue declared by the pass.
// Invariant: Adapters never read global state. All inputs come from PassResources.

#include "frame/frame_context.hpp"
#include "rendergraph/graph_types.hpp"

#include <cstdint>
#include <limits>

namespace engine {

class EngineServices;

class FeatureAdapter {
public:
    virtual ~FeatureAdapter() = default;

    // Initialize the adapter with engine services.
    virtual void init(EngineServices& services) = 0;

    // Update configuration from a frozen config snapshot.
    virtual void configure(const ConfigSnapshot& config) = 0;

    // Execute the pass for one frame. Called by the render graph executor.
    virtual void execute(const FrameContext& ctx, const PassResources& resources) = 0;

    // Release all resources.
    virtual void shutdown() = 0;

    // Human-readable name for logging.
    virtual const char* name() const = 0;

protected:
    // GPU profiler slot registered via MetricsService::registerTimer().
    // UINT32_MAX means "not registered" (default). Adapters that want
    // per-pass GPU timing set this during init() and wrap execute() with
    // services.metrics().beginNamedTimer / endNamedTimer.
    uint32_t profileSlot_ = UINT32_MAX;
};

} // namespace engine
