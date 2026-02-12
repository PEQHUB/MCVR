#pragma once

#include "common/shared.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <filesystem>
#include <vector>

class Framework;

/// Composites the PQ-encoded HDR world and sRGB UI overlay onto the swapchain.
///
/// Only used when HDR10 output is active. The shader decodes PQ, converts UI
/// from BT.709 to BT.2020 at the user-chosen nit level, alpha-blends in linear
/// light, re-encodes PQ, and applies anti-banding dither.
///
/// SDR path never touches this class.
class HdrCompositePass : public SharedObject<HdrCompositePass> {
  public:
    HdrCompositePass() = default;
    ~HdrCompositePass();

    /// Create and initialize all Vulkan resources.
    /// @param framework   The active Framework (device, swapchain, etc.)
    void init(std::shared_ptr<Framework> framework);

    /// Tear down and recreate for a new swapchain (e.g. resize, HDR toggle).
    void recreate(std::shared_ptr<Framework> framework);

    /// Destroy all Vulkan resources.
    void destroy();

    /// Record composite commands into the given command buffer.
    ///
    /// Handles all image layout transitions internally:
    ///   worldImage  → SHADER_READ_ONLY_OPTIMAL
    ///   overlayImage → SHADER_READ_ONLY_OPTIMAL
    ///   swapchainImage → COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR
    ///
    /// @param cmd              The fuse command buffer (already begun)
    /// @param frameIndex       Swapchain image index
    /// @param uiBrightnessNits UI brightness push constant (50–300)
    /// @param worldImage       Tone-mapped HDR world (PQ-encoded, A2B10G10R10)
    /// @param overlayImage     Minecraft UI overlay (R8G8B8A8_SRGB, transparent background)
    /// @param swapchainImage   Target swapchain image (A2B10G10R10_UNORM)
    /// @param mainQueueIndex   Queue family index for barriers
    void record(std::shared_ptr<vk::CommandBuffer> cmd,
                uint32_t frameIndex,
                float uiBrightnessNits,
                std::shared_ptr<vk::DeviceLocalImage> worldImage,
                std::shared_ptr<vk::DeviceLocalImage> overlayImage,
                std::shared_ptr<vk::SwapchainImage> swapchainImage,
                uint32_t mainQueueIndex);

  private:
    void initShaders();
    void initSampler();
    void initDescriptorSets();
    void initRenderPass();
    void initFramebuffers();
    void initPipeline();

    std::weak_ptr<Framework> framework_;

    // Shaders
    std::shared_ptr<vk::Shader> vertShader_;
    std::shared_ptr<vk::Shader> fragShader_;

    // Sampler (nearest — dimensions match, no filtering needed)
    std::shared_ptr<vk::Sampler> sampler_;

    // Descriptor sets — one per swapchain image
    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables_;

    // Render pass + framebuffers (one per swapchain image, targeting swapchain)
    std::shared_ptr<vk::RenderPass> renderPass_;
    std::vector<std::shared_ptr<vk::Framebuffer>> framebuffers_;

    // Graphics pipeline
    std::shared_ptr<vk::GraphicsPipeline> pipeline_;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
};
