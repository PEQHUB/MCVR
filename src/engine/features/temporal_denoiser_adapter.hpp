#pragma once

// V2 M8 Temporal Denoiser Adapter (stepping stone toward full SVGF).
//
// Sits between the RT pass and ToneMapping. Consumes the four RT outputs:
//   - rt_radiance     (sampled, current frame radiance)
//   - rt_normal       (sampled, world-space normal)
//   - rt_linear_depth (sampled, ray distance)
//   - rt_motion_vector (sampled, pixel-space motion)
//
// Plus a persistent history image declared as an OUTPUT (carries state across
// frames via the GraphResourcePool, which keeps the same VkImage alive across
// frames once the BarrierPlanner tracks layouts persistently).
//
// Writes:
//   - denoise_output  (rgba16f, fed to ToneMapping)
//   - denoise_history (rgba16f, persistent, read+written by next frame)
//
// This is intentionally minimal: a TAA-style temporal accumulation with a
// neighborhood color clamp for ghosting suppression. The full SVGF port lands
// in M8b once the RT adapter exposes specular/diffuse separation.

#include "feature_adapter.hpp"
#include "platform/vulkan/vk2_descriptor.hpp"
#include "platform/vulkan/vk2_pipeline.hpp"
#include "platform/vulkan/vk2_shader.hpp"
#include "rendergraph/graph_types.hpp"

#include <cstdint>

namespace engine {

class GraphBuilder;

struct TemporalDenoisePush {
    float alpha;
    float normalThreshold;
    float depthThreshold;
    uint32_t frameNumber;
    int32_t imageWidth;
    int32_t imageHeight;
    int32_t padA;
    int32_t padB;
};

class TemporalDenoiserAdapter : public FeatureAdapter {
public:
    void init(EngineServices& services) override;
    void configure(const ConfigSnapshot& config) override;
    void execute(const FrameContext& ctx, const PassResources& resources) override;
    void shutdown() override;
    const char* name() const override { return "TemporalDenoiser"; }

    // Register denoiser pass after RT. Inputs come from the RT adapter; output
    // is consumed by the ToneMapping adapter.
    void registerPass(GraphBuilder& builder,
                      ResourceHandle radianceIn,
                      ResourceHandle normalIn,
                      ResourceHandle depthIn,
                      ResourceHandle motionIn);

    ResourceHandle outputHandle() const { return outputHandle_; }

private:
    EngineServices* services_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;

    vk2::ShaderModule shader_;
    vk2::PipelineLayout pipelineLayout_;
    vk2::ComputePipeline pipeline_;

    VkDescriptorSetLayout descSetLayout_ = VK_NULL_HANDLE;
    vk2::DescriptorAllocator descAllocator_;
    VkSampler linearSampler_ = VK_NULL_HANDLE;

    float alpha_ = 0.1f;             // 10% current, 90% history once accepted
    float normalThreshold_ = 0.9f;
    float depthThreshold_ = 0.1f;
    uint32_t frameNumber_ = 0;

    // Inputs (declared by RT adapter)
    ResourceHandle radianceInput_;
    ResourceHandle normalInput_;
    ResourceHandle depthInput_;
    ResourceHandle motionInput_;

    // Outputs (declared here)
    ResourceHandle historyHandle_;   // persistent rgba16f
    ResourceHandle outputHandle_;    // denoised rgba16f → ToneMap

    bool initialized_ = false;
};

} // namespace engine
