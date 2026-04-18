#pragma once

// PostRenderAdapter: Entity rasterization pass (particles, hand items, weather).
//
// Rasterizes post-RT entities on top of the tone-mapped + CAS output using
// the D32_SFLOAT depth buffer from ColorToDepthAdapter for correct depth testing.
// Alpha blending (srcAlpha, oneMinusSrcAlpha) composites translucent entities.
//
// Pipeline: PBRTriangle (96-byte) vertex input, dynamic rendering, push constants
// for view/projection matrices, one descriptor set for texture atlas sampling.
//
// Data flow: Java submits CmdEntityPostDraw via bridge → EngineApp stores pending
// → processScene() calls setPendingDrawData() → execute() uploads to GPU and draws.

#include "feature_adapter.hpp"
#include "platform/vulkan/vk2_shader.hpp"
#include "platform/vulkan/vk2_pipeline.hpp"
#include "platform/vulkan/vk2_descriptor.hpp"
#include "platform/vulkan/vk2_buffer.hpp"
#include "platform/vulkan/vk2_image.hpp"
#include "bridge/bridge_service.hpp"

#include <vector>

namespace engine {

class GraphBuilder;
class ColorToDepthAdapter;

class PostRenderAdapter : public FeatureAdapter {
public:
    void init(EngineServices& services) override;
    void configure(const ConfigSnapshot& config) override;
    void execute(const FrameContext& ctx, const PassResources& resources) override;
    void shutdown() override;
    const char* name() const override { return "PostRender"; }

    void registerPass(GraphBuilder& builder, ResourceHandle colorInputHandle,
                      ResourceHandle c2dOrderingHandle);

    ResourceHandle outputHandle() const { return outputHandle_; }

    // Set the depth image from ColorToDepthAdapter (not a graph resource).
    void setDepthSource(ColorToDepthAdapter* c2d) { c2dAdapter_ = c2d; }

    // Receive pending post-entity draw data from EngineApp (main thread).
    // Called from processScene() after bridge flush.
    void setPendingDrawData(CmdEntityPostDraw data);

private:
    EngineServices* services_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;

    ColorToDepthAdapter* c2dAdapter_ = nullptr;

    // Graph handles
    ResourceHandle inputHandle_;
    ResourceHandle outputHandle_;

    // Pipeline objects
    vk2::ShaderModule vertShader_;
    vk2::ShaderModule fragShader_;
    VkDescriptorSetLayout texDescSetLayout_ = VK_NULL_HANDLE;
    vk2::PipelineLayout pipelineLayout_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    // Descriptor allocator for per-frame texture atlas binding
    vk2::DescriptorAllocator descAllocator_;

    // Per-frame GPU buffers (host-visible, recreated when size changes)
    vk2::Buffer vertexBuffer_;
    vk2::Buffer indexBuffer_;
    VkDeviceSize vertexBufferCapacity_ = 0;
    VkDeviceSize indexBufferCapacity_ = 0;

    // Pending draw data (set by setPendingDrawData, consumed by execute)
    CmdEntityPostDraw pendingDraw_;
    bool hasPendingDraw_ = false;

    // Push constant layout matching the vertex shader
    struct PushConstants {
        float viewProj[16];
        float view[16];
        float cameraPos[4];
    };

    bool initialized_ = false;

    void destroyPipeline();
    void ensureBuffers(VkDeviceSize vertexBytes, VkDeviceSize indexBytes);
};

} // namespace engine
