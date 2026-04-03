#include "tonemapping_adapter.hpp"
#include "app/engine_services.hpp"
#include "config/config_service.hpp"
#include "rendergraph/graph_builder.hpp"
#include "diagnostics/log.hpp"

namespace engine {

void ToneMappingAdapter::init(EngineServices& /*services*/) {
    log::info("features", "ToneMappingAdapter initialized");
    initialized_ = true;
}

void ToneMappingAdapter::configure(const ConfigSnapshot& config) {
    tonemapMode_ = config.data.tonemappingMode;
    psychoEnabled_ = config.data.psychoEnabled;
    saturation_ = config.data.saturation;
    Lwhite_ = config.data.Lwhite;
    casSharpness_ = config.data.casSharpness;
    sharpenerMode_ = config.data.sharpenerMode;
    psychoPeakSDR_ = config.data.psychoPeakSDR;
    colorExpansion_ = config.data.colorExpansion;
}

void ToneMappingAdapter::execute(const FrameContext& ctx, const PassResources& /*resources*/) {
    if (!initialized_) return;

    // Configure from this frame's frozen config
    configure(*ctx.config);

    // In the real implementation, this would:
    // 1. Bind the tone mapping compute pipeline
    // 2. Set push constants (tonemapMode, saturation, Lwhite, etc.)
    // 3. Dispatch compute shader over the input image
    // 4. Apply sharpening if sharpenerMode > 0
    //
    // For now, this is a no-op that proves the adapter → render graph → frame
    // scheduler pipeline works end to end. The actual Vulkan dispatch will be
    // added when the legacy ToneMappingModule is ported.

    log::trace("features", "ToneMapping execute frame=" + std::to_string(ctx.frameNumber) +
               " mode=" + std::to_string(tonemapMode_));
}

void ToneMappingAdapter::shutdown() {
    log::info("features", "ToneMappingAdapter shutdown");
    initialized_ = false;
}

PassHandle ToneMappingAdapter::registerPass(GraphBuilder& builder, ToneMappingAdapter& adapter) {
    // Declare resources matching tone_mapping.yaml
    ImageResourceDesc inputDesc;
    inputDesc.name = "denoised_radiance";
    inputDesc.format = VK_FORMAT_R32G32B32A32_SFLOAT;

    ImageResourceDesc outputDesc;
    outputDesc.name = "mapped_output";
    outputDesc.format = VK_FORMAT_R8G8B8A8_UNORM;

    // Import as a module (creates/finds resources, creates pass)
    return builder.importModule(
        "tone_mapping",
        {inputDesc},
        {outputDesc},
        [&adapter](const FrameContext& ctx, const PassResources& res) {
            adapter.execute(ctx, res);
        });
}

} // namespace engine
