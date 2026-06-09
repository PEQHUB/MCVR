#pragma once

// RT adapter: traces rays against the world TLAS when geometry is available.
// Falls back to sky gradient (miss shader only) when TLAS is empty.
//
// PR38: uses V1-compatible 4-set descriptor layout via RtDescriptorLayout.
// Test shaders consume only TLAS (set 1 binding 0), WorldUBO (set 2 binding 0),
// and the output image (set 3 binding 0). All other bindings remain unbound
// (PARTIALLY_BOUND) until PR39 ports the real shaders.

#include "feature_adapter.hpp"
#include "rt_descriptor_layout.hpp"
#include "platform/vulkan/vk2_shader.hpp"
#include "platform/vulkan/vk2_pipeline.hpp"
#include "platform/vulkan/vk2_rt_pipeline.hpp"
#include "platform/vulkan/vk2_descriptor.hpp"

namespace engine {

class GraphBuilder;

class RayTracingAdapter : public FeatureAdapter {
public:
    void init(EngineServices& services) override;
    void configure(const ConfigSnapshot& config) override;
    void execute(const FrameContext& ctx, const PassResources& resources) override;
    void shutdown() override;
    const char* name() const override { return "RayTracing"; }

    // Register RT pass in graph. Returns output handle (HDR radiance).
    ResourceHandle registerPass(GraphBuilder& builder);

private:
    EngineServices* services_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;

    vk2::ShaderModule rgenShader_;
    vk2::ShaderModule rchitShader_;
    vk2::ShaderModule rmissShader_;
    vk2::RtPipeline rtPipeline_;
    vk2::ShaderBindingTable sbt_;

    RtDescriptorLayout layout_;          // PR38: V1-compatible 4-set layout
    vk2::DescriptorAllocator descAllocator_;

    ResourceHandle outputHandle_;
    bool initialized_ = false;
};

} // namespace engine
