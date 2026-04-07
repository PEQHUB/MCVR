#pragma once

// PR41: CAS (Contrast-Adaptive Sharpening) post-process adapter.
//
// Wraps FidelityFX CAS as a single compute dispatch. Lives between ToneMapping
// and Composite. Reads tone-mapped LDR input, writes sharpened LDR output.
//
// Push constants are 32 bytes (2 uvec4) computed each frame from input/output
// resolution + sharpness via the FFX CAS setup formula.

#include "feature_adapter.hpp"
#include "platform/vulkan/vk2_shader.hpp"
#include "platform/vulkan/vk2_pipeline.hpp"
#include "platform/vulkan/vk2_descriptor.hpp"

#include <array>
#include <cstdint>

namespace engine {

class GraphBuilder;

struct CasPushConstants {
    std::array<uint32_t, 4> const0{};
    std::array<uint32_t, 4> const1{};
};

class PostProcessAdapter : public FeatureAdapter {
public:
    void init(EngineServices& services) override;
    void configure(const ConfigSnapshot& config) override;
    void execute(const FrameContext& ctx, const PassResources& resources) override;
    void shutdown() override;
    const char* name() const override { return "PostProcess"; }

    // Register CAS pass: reads inputHandle (tone-mapped LDR), writes a new
    // sharpened LDR resource. Marks itself as the final output of the graph.
    void registerPass(GraphBuilder& builder, ResourceHandle inputHandle);

private:
    EngineServices* services_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;

    vk2::ShaderModule casShader_;
    vk2::PipelineLayout pipelineLayout_;
    vk2::ComputePipeline casPipeline_;

    VkDescriptorSetLayout descSetLayout_ = VK_NULL_HANDLE;
    vk2::DescriptorAllocator descAllocator_;

    VkSampler linearSampler_ = VK_NULL_HANDLE;

    float sharpness_ = 0.5f;

    ResourceHandle inputHandle_;
    ResourceHandle outputHandle_;
    bool initialized_ = false;
};

} // namespace engine
