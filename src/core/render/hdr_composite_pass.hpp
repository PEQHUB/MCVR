#pragma once

#include "common/shared.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <filesystem>
#include <vector>

class Framework;

/// Composites the rendered world and UI overlay onto the swapchain.
///
/// HDR10 mode: expects PQ-encoded world + sRGB UI overlay, composites in linear
/// light, outputs PQ-encoded HDR10 to the HDR swapchain.
///
/// SDR mode: expects display-referred SDR world (gamma-encoded in UNORM) + sRGB
/// UI overlay (with alpha), composites in linear light, outputs sRGB-encoded
/// values to the SDR swapchain (which is typically UNORM + SRGB_NONLINEAR).
class HdrCompositePass : public SharedObject<HdrCompositePass> {
  public:
    HdrCompositePass() = default;
    ~HdrCompositePass();

    enum class OutputMode : uint32_t {
        Sdr = 0,
        Hdr10 = 1,
    };

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
    /// @param mode             Composite mode (SDR or HDR10)
    /// @param uiBrightnessNits UI brightness push constant (HDR only; ignored in SDR)
    /// @param worldImage       Tone-mapped HDR world (PQ-encoded, A2B10G10R10)
    /// @param overlayImage     Minecraft UI overlay (R8G8B8A8_SRGB, transparent background)
    /// @param swapchainImage   Target swapchain image (A2B10G10R10_UNORM)
    /// @param mainQueueIndex   Queue family index for barriers
    void record(std::shared_ptr<vk::CommandBuffer> cmd,
                uint32_t frameIndex,
                OutputMode mode,
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
    std::shared_ptr<vk::Shader> fragShaderHdr_;
    std::shared_ptr<vk::Shader> fragShaderSdr_;

    // Sampler (nearest — dimensions match, no filtering needed)
    std::shared_ptr<vk::Sampler> sampler_;

    // Descriptor sets — one per swapchain image
    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables_;

    // Render pass + framebuffers (one per swapchain image, targeting swapchain)
    std::shared_ptr<vk::RenderPass> renderPass_;
    std::vector<std::shared_ptr<vk::Framebuffer>> framebuffers_;

    // Graphics pipeline
    std::shared_ptr<vk::GraphicsPipeline> pipelineHdr_;
    std::shared_ptr<vk::GraphicsPipeline> pipelineSdr_;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
};
