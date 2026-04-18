#pragma once

// V2 Atmosphere Adapter
//
// Generates two atmosphere textures consumed by the RT pipeline:
//   1. Transmittance LUT  (512x128 RGBA16F) — computed once on first frame
//   2. Sky cubemap         (512x512x6 RGBA16F) — re-computed every frame
//
// Owns its own internal compute pipelines and descriptor layouts (set 0).
// Exposes VkImageView + VkSampler accessors so the RT adapter can bind them
// at set 0 bindings 1 (LUT) and 2 (cubemap).
//
// Not a FeatureAdapter — it doesn't participate in the render graph.
// Instead, EngineApp calls recordCommands() from processScene() each frame.

#include "platform/vulkan/vk2_shader.hpp"
#include "platform/vulkan/vk2_pipeline.hpp"
#include "platform/vulkan/vk2_descriptor.hpp"
#include "platform/vulkan/vk2_image.hpp"

#include <cstdint>

namespace engine {

class EngineServices;

class AtmosphereAdapter {
public:
    AtmosphereAdapter() = default;
    ~AtmosphereAdapter() = default;

    AtmosphereAdapter(const AtmosphereAdapter&) = delete;
    AtmosphereAdapter& operator=(const AtmosphereAdapter&) = delete;

    // Initialize pipelines, images, samplers.
    void init(EngineServices& services);

    // Record atmosphere compute dispatches into the command buffer.
    // On first call: dispatches LUT compute + cubemap compute.
    // On subsequent calls: dispatches cubemap compute only.
    void recordCommands(VkCommandBuffer cmd, uint32_t frameIndex);

    // Shutdown and release all resources.
    void shutdown();

    bool isInitialized() const { return initialized_; }

    // Accessors for RT descriptor binding.
    VkImageView lutView() const { return lutImage_.defaultView(); }
    // cubeView() returns a VK_IMAGE_VIEW_TYPE_CUBE view (index 1 in views_).
    // The default view at index 0 is VK_IMAGE_VIEW_TYPE_2D_ARRAY and must NOT
    // be used for imageCube storage or samplerCube bindings — it causes a
    // GPU instruction fault (type=6 DEVICE_LOST) on the first dispatch.
    VkImageView cubeView() const { return cubeImage_.view(cubeViewIdx_); }
    VkSampler   sampler() const { return linearSampler_; }

private:
    static constexpr uint32_t LUT_WIDTH  = 512;
    static constexpr uint32_t LUT_HEIGHT = 128;
    static constexpr uint32_t CUBE_DIM   = 512;

    EngineServices* services_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;

    // Images
    vk2::Image lutImage_;   // 512x128 RGBA16F 2D
    vk2::Image cubeImage_;  // 512x512x6 RGBA16F cubemap

    // Sampler (shared by LUT and cubemap)
    VkSampler linearSampler_ = VK_NULL_HANDLE;

    // LUT compute pipeline (set 0: binding 0 = output image, binding 1 = sky UBO)
    vk2::ShaderModule lutShader_;
    VkDescriptorSetLayout lutDescLayout_ = VK_NULL_HANDLE;
    vk2::PipelineLayout lutPipelineLayout_;
    vk2::ComputePipeline lutPipeline_;
    vk2::DescriptorAllocator lutDescAllocator_;

    // Cubemap compute pipeline (set 0: binding 0 = output cube, binding 1 = LUT sampler, binding 2 = sky UBO)
    vk2::ShaderModule cubeShader_;
    VkDescriptorSetLayout cubeDescLayout_ = VK_NULL_HANDLE;
    vk2::PipelineLayout cubePipelineLayout_;
    vk2::ComputePipeline cubePipeline_;
    vk2::DescriptorAllocator cubeDescAllocator_;

    struct CubePushConstant {
        int32_t face;
    };

    bool initialized_ = false;
    bool lutGenerated_ = false;
    uint32_t profileSlot_ = UINT32_MAX;  // MetricsService named timer slot
    uint32_t cubeViewIdx_ = 0;  // index of VK_IMAGE_VIEW_TYPE_CUBE view in cubeImage_.views_
};

} // namespace engine
