#pragma once

// ColorToDepthAdapter: Converts the RT first-hit-depth storage image (R32F)
// into a proper D32_SFLOAT depth attachment via fullscreen triangle rendering.
//
// The produced depth buffer is NOT a graph resource — it's adapter-owned
// because the graph resource pool doesn't support DEPTH_STENCIL_ATTACHMENT usage.
// Downstream adapters (PostRenderAdapter, StarFieldAdapter) receive the depth
// image handle directly via setDepthImage().
//
// Pipeline: fullscreen triangle (3 vertices, no vertex buffer)
//   - Vertex: v2_color_to_depth.vert (hardcoded NDC triangle)
//   - Fragment: v2_color_to_depth.frag (reads firstHitDepth, writes gl_FragDepth)

#include "feature_adapter.hpp"
#include "platform/vulkan/vk2_shader.hpp"
#include "platform/vulkan/vk2_pipeline.hpp"
#include "platform/vulkan/vk2_descriptor.hpp"
#include "platform/vulkan/vk2_image.hpp"

namespace engine {

class GraphBuilder;

class ColorToDepthAdapter : public FeatureAdapter {
public:
    void init(EngineServices& services) override;
    void configure(const ConfigSnapshot& config) override;
    void execute(const FrameContext& ctx, const PassResources& resources) override;
    void shutdown() override;
    const char* name() const override { return "ColorToDepth"; }

    void registerPass(GraphBuilder& builder, ResourceHandle firstHitDepthHandle);

    // Access the adapter-owned D32_SFLOAT depth image (for downstream passes).
    vk2::Image& depthImage() { return depthImage_; }
    const vk2::Image& depthImage() const { return depthImage_; }

    // Graph ordering token: downstream passes (PostRender, StarField) can declare
    // this as an input to guarantee C2D executes first.
    ResourceHandle orderingHandle() const { return orderingHandle_; }

private:
    EngineServices* services_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;

    vk2::ShaderModule vertShader_;
    vk2::ShaderModule fragShader_;

    VkDescriptorSetLayout descSetLayout_ = VK_NULL_HANDLE;
    vk2::PipelineLayout pipelineLayout_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;   // manually managed (depth-only, no color)

    vk2::DescriptorAllocator descAllocator_;

    // Adapter-owned depth image (D32_SFLOAT)
    vk2::Image depthImage_;
    uint32_t lastWidth_ = 0;
    uint32_t lastHeight_ = 0;

    ResourceHandle inputHandle_;
    ResourceHandle orderingHandle_;  // Dummy resource for graph ordering
    bool initialized_ = false;

    void ensureDepthImage(uint32_t width, uint32_t height);
    void destroyPipeline();
};

} // namespace engine
