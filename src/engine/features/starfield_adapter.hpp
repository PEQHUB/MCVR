#pragma once

// StarFieldAdapter: Procedural star field rendering for V2 engine.
//
// Renders 3000 procedural star billboard quads with additive blending at night.
// Uses the D32_SFLOAT depth buffer from ColorToDepthAdapter for depth testing.
// Stars are placed at a fixed depth of 0.95 (behind world geometry, in front of sky).
//
// Star sphere is rotated each frame so that the exclusion zone (no stars near
// the sun/moon axis) tracks the sun direction from the SkyUBO.
//
// Pipeline:
//   - Vertex: v2_starfield.vert (push constants for viewProj, sunDirection, visibility)
//   - Fragment: v2_starfield.frag (additive blend, fixed gl_FragDepth = 0.95)
//   - Blending: additive (src=ONE, dst=ONE for color; alpha: src=ZERO, dst=ONE)
//   - Depth: test LESS, write TRUE

#include "feature_adapter.hpp"
#include "platform/vulkan/vk2_shader.hpp"
#include "platform/vulkan/vk2_pipeline.hpp"
#include "platform/vulkan/vk2_buffer.hpp"
#include "platform/vulkan/vk2_image.hpp"

namespace engine {

class GraphBuilder;
class ColorToDepthAdapter;

class StarFieldAdapter : public FeatureAdapter {
public:
    void init(EngineServices& services) override;
    void configure(const ConfigSnapshot& config) override;
    void execute(const FrameContext& ctx, const PassResources& resources) override;
    void shutdown() override;
    const char* name() const override { return "StarField"; }

    void registerPass(GraphBuilder& builder, ResourceHandle colorInputHandle);

    ResourceHandle outputHandle() const { return outputHandle_; }

    // Set the depth image from ColorToDepthAdapter (not a graph resource).
    void setDepthSource(ColorToDepthAdapter* c2d) { c2dAdapter_ = c2d; }

    // Star vertex: position + color. 28 bytes per vertex.
    struct StarVertex {
        float pos[3];    // World-space position on star sphere
        float color[4];  // RGBA (alpha encodes per-star brightness)
    };
    static_assert(sizeof(StarVertex) == 28, "StarVertex must be 28 bytes");

    // Push constants matching the vertex shader layout.
    struct PushConstants {
        float viewProj[16];    // mat4
        float sunDirection[3]; // vec3
        float starVisibility;  // 0 = daytime, 1 = nighttime
        float rainGradient;    // Rain attenuation factor
        uint32_t skyType;      // 1 = overworld
        float pad[2];          // Align to 16 bytes
    };
    static_assert(sizeof(PushConstants) == 96, "PushConstants must be 96 bytes");

    static constexpr uint32_t STAR_COUNT = 3000;
    static constexpr uint32_t VERTICES_PER_STAR = 6;
    static constexpr uint32_t TOTAL_VERTICES = STAR_COUNT * VERTICES_PER_STAR;

private:
    EngineServices* services_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;

    ColorToDepthAdapter* c2dAdapter_ = nullptr;

    // Shaders
    vk2::ShaderModule vertShader_;
    vk2::ShaderModule fragShader_;

    // Pipeline
    VkDescriptorSetLayout descSetLayout_ = VK_NULL_HANDLE;  // empty (no descriptors)
    vk2::PipelineLayout pipelineLayout_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    // Star vertex buffer (device-local, static)
    vk2::Buffer vertexBuffer_;

    ResourceHandle inputHandle_;
    ResourceHandle outputHandle_;
    bool initialized_ = false;

    void generateStars();
    void destroyPipeline();
};

} // namespace engine
