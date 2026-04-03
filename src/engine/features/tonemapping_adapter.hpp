#pragma once

// Ownership: Render graph (via PassDescriptor).
// Thread: execute() on Graphics queue.
// Dependencies: ConfigService (for tone mapping settings).
//
// This is the first FeatureAdapter — proves the adapter pattern works.
// It wraps the tone mapping pass configuration without touching the
// legacy ToneMappingModule code (which still runs in the legacy path).

#include "feature_adapter.hpp"

#include <cstdint>

namespace engine {

class GraphBuilder;

class ToneMappingAdapter : public FeatureAdapter {
public:
    void init(EngineServices& services) override;
    void configure(const ConfigSnapshot& config) override;
    void execute(const FrameContext& ctx, const PassResources& resources) override;
    void shutdown() override;
    const char* name() const override { return "ToneMapping"; }

    // Register this pass with a GraphBuilder.
    // Creates input (denoised_radiance) and output (mapped_output) resources,
    // and a pass that calls execute().
    static PassHandle registerPass(GraphBuilder& builder, ToneMappingAdapter& adapter);

private:
    // Cached config values (updated in configure())
    uint32_t tonemapMode_ = 0;
    bool psychoEnabled_ = true;
    float saturation_ = 1.3f;
    float Lwhite_ = 4.0f;
    float casSharpness_ = 0.5f;
    uint32_t sharpenerMode_ = 0;
    float psychoPeakSDR_ = 1.0f;
    float colorExpansion_ = 1.0f;

    bool initialized_ = false;
};

} // namespace engine
