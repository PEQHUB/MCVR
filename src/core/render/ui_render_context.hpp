#pragma once

#include "core/vulkan/all_core_vulkan.hpp"
#include "core/render/modules/ui_module.hpp"

#include <atomic>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <mutex>

class Framework;
class OverlayCompositor;

/// Standalone overlay rendering context for the UI thread.
/// Owns its own command pool (secondary queue), double-buffered images,
/// framebuffers, and fences — completely independent of the per-frame
/// FrameworkContext used by the render thread.
///
/// Shares UIModule's read-only pipeline objects (render pass, shaders,
/// descriptor layouts) to avoid duplicating GPU resources.
class UIRenderContext {
  public:
    UIRenderContext() = default;
    ~UIRenderContext();

    /// Initialize all Vulkan resources. Returns false on failure.
    bool init(std::shared_ptr<Framework> framework, UIModule *uiModule);

    /// Destroy all resources (safe to call multiple times).
    void destroy();

    // ── Frame lifecycle (called from UI thread) ──

    /// Begin a new UI frame: wait for previous fence, reset command buffer,
    /// begin render pass with clear.
    void beginFrame();

    /// End the current render pass.
    void endFrame();

    /// Submit command buffer to secondary queue, wait fence, present via
    /// DComp overlay, then waitForDisplayReady() for pacing.
    void submitAndPresent();

    // ── Draw commands (called between beginFrame/endFrame) ──

    /// Bind pipeline + vertex/index buffers, issue indexed draw.
    void drawIndexed(std::shared_ptr<vk::DeviceLocalBuffer> vertexBuffer,
                     std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer,
                     OverlayDrawPipelineType pipelineType,
                     uint32_t indexCount,
                     VkIndexType indexType);

    /// Set dynamic viewport.
    void setViewport(int x, int y, int width, int height);

    /// Set dynamic scissor.
    void setScissor(int x, int y, int width, int height);
    void setScissorEnabled(bool enabled);

    /// Clear color attachment.
    void clearColor(float r, float g, float b, float a);

    /// Sync all dynamic state to the active command buffer.
    void syncDynamicState();

    // ── Overlay state setters (mirror UIModuleContext) ──
    void setBlendEnable(bool enable);
    void setBlendFuncSeparate(int srcColor, int srcAlpha, int dstColor, int dstAlpha);
    void setBlendOpSeparate(int colorOp, int alphaOp);
    void setColorWriteMask(int mask);
    void setColorLogicOpEnable(bool enable);
    void setColorLogicOp(int op);
    void setBlendConstants(float c0, float c1, float c2, float c3);
    void setDepthTestEnable(bool enable);
    void setDepthWriteEnable(bool enable);
    void setDepthCompareOp(int op);
    void setStencilTestEnable(bool enable);
    void setStencilFrontFunc(int compareOp, int reference, int compareMask);
    void setStencilBackFunc(int compareOp, int reference, int compareMask);
    void setStencilFrontOp(int failOp, int depthFailOp, int passOp);
    void setStencilBackOp(int failOp, int depthFailOp, int passOp);
    void setStencilFrontWriteMask(int writeMask);
    void setStencilBackWriteMask(int writeMask);
    void setCullMode(int cullMode);
    void setFrontFace(int frontFace);
    void setPolygonMode(int polygonMode);
    void setDepthBiasEnable(int polygonMode, bool enable);
    void setDepthBias(float slopeFactor, float constantFactor);
    void setLineWidth(float lineWidth);

    // ── Texture binding ──
    void bindTexture(std::shared_ptr<vk::Sampler> sampler,
                     std::shared_ptr<vk::DeviceLocalImage> image,
                     int index);

    // ── Pause/resume (called from render thread during recreate) ──

    /// Synchronous pause: blocks caller until UI thread acknowledges.
    void pause();
    void resume();
    bool isPaused() const { return paused_.load(std::memory_order_relaxed); }

    /// Rebuild images/framebuffers after swapchain recreation. Must be called while paused.
    void onSwapchainRecreate();

    /// Set the overlay compositor for DComp presentation.
    void setOverlayCompositor(OverlayCompositor *compositor);

    /// Whether a frame is currently in-flight (between beginFrame and submitAndPresent).
    bool isFrameActive() const { return frameActive_; }

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

  private:
    void createResources();
    void destroyResources();
    void diagLog(const char *fmt, ...);

    std::weak_ptr<Framework> framework_;
    std::shared_ptr<vk::Device> device_;
    UIModule *uiModule_ = nullptr;  // non-owning — UIModule outlives us

    uint32_t width_ = 0;
    uint32_t height_ = 0;

    // Double-buffered resources (owned by this context)
    static constexpr int kBufferCount = 2;
    std::shared_ptr<vk::CommandPool> commandPool_;
    std::shared_ptr<vk::CommandBuffer> commandBuffers_[kBufferCount];
    std::shared_ptr<vk::DeviceLocalImage> colorImages_[kBufferCount];
    std::shared_ptr<vk::DeviceLocalImage> depthImages_[kBufferCount];
    std::shared_ptr<vk::Framebuffer> framebuffers_[kBufferCount];
    std::shared_ptr<vk::DescriptorTable> descriptorTables_[kBufferCount];
    std::shared_ptr<vk::Sampler> colorImageSamplers_[kBufferCount];
    VkFence fences_[kBufferCount] = {};
    int currentIndex_ = 0;

    // Frame state
    bool frameActive_ = false;
    bool renderPassActive_ = false;

    // Dynamic state (mirrors UIModuleContext fields)
    VkViewport viewport_{};
    VkRect2D scissor_{};
    bool scissorEnabled_ = false;

    VkBool32 blendEnabled_ = VK_TRUE;
    VkColorBlendEquationEXT colorBlendEquation_{};
    VkColorComponentFlags colorWriteMask_ = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    bool colorLogicOpEnable_ = false;
    VkLogicOp colorLogicOp_ = VK_LOGIC_OP_COPY;
    std::array<float, 4> blendConstants_ = {0, 0, 0, 0};

    bool depthTestEnable_ = false;
    bool depthWriteEnable_ = false;
    VkCompareOp depthCompareOp_ = VK_COMPARE_OP_LESS_OR_EQUAL;
    bool stencilTestEnable_ = false;
    std::array<VkStencilOp, 2> failOp_ = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP};
    std::array<VkStencilOp, 2> passOp_ = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP};
    std::array<VkStencilOp, 2> depthFailOp_ = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP};
    std::array<VkCompareOp, 2> stencilCompareOp_ = {VK_COMPARE_OP_ALWAYS, VK_COMPARE_OP_ALWAYS};
    std::array<uint32_t, 2> stencilReference_ = {0, 0};
    std::array<uint32_t, 2> stencilCompareMask_ = {0xFF, 0xFF};
    std::array<uint32_t, 2> stencilWriteMask_ = {0xFF, 0xFF};

    VkCullModeFlags cullMode_ = VK_CULL_MODE_NONE;
    VkFrontFace frontFace_ = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkPolygonMode polygonMode_ = VK_POLYGON_MODE_FILL;
    bool depthBiasEnable_ = false;
    float depthBiasSlopeFactor_ = 0.0f;
    float depthBiasConstantFactor_ = 0.0f;
    float lineWidth_ = 1.0f;

    std::array<float, 4> clearColor_ = {0, 0, 0, 0};

    // DComp overlay
    std::atomic<OverlayCompositor *> overlayCompositor_{nullptr};

    // Tracks whether any thread is driving the beginFrame/endFrame/submit loop
    std::atomic<bool> loopActive_{false};

    // Pause protocol
    std::atomic<bool> paused_{false};
    std::atomic<bool> pauseRequested_{false};
    std::atomic<bool> pauseAcknowledged_{false};
    std::mutex pauseMtx_;
    std::condition_variable pauseCv_;

    // Diagnostics
    std::ofstream diagFile_;
};
