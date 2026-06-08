#pragma once
#ifdef _WIN32

#include "core/all_extern.hpp"
#include <memory>

namespace vk {
class Device;
class DeviceLocalImage;
class CommandBuffer;
class Shader;
class GraphicsPipeline;
class RenderPass;
class DescriptorTable;
class Sampler;
class Framebuffer;
}

struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11Texture2D;
struct IDXGISwapChain1;
struct IDXGISwapChain2;
struct IDCompositionDevice;
struct IDCompositionTarget;
struct IDCompositionVisual;
struct IDXGIKeyedMutex;

class Framework;

class OverlayCompositor {
  public:
    OverlayCompositor() = default;
    ~OverlayCompositor();

    /// Initialize all resources. Returns false if any precondition fails.
    bool init(std::shared_ptr<Framework> framework);

    /// Destroy all resources (safe to call multiple times).
    void destroy();

    /// Record premultiply pass into command buffer (called by fuseFinal on render thread).
    /// Reads overlayImage, writes to shared image.
    void recordPremultiply(std::shared_ptr<vk::CommandBuffer> cmd,
                           std::shared_ptr<vk::DeviceLocalImage> overlayImage,
                           uint32_t queueFamilyIndex);

    /// Present the shared image content via D3D11 -> DXGI -> DComp.
    /// Called by PresentThread (may re-present the same frame at display rate).
    void present();

    /// Wait for the DXGI swapchain to be ready for the next frame.
    /// Returns true if ready, false on timeout. Paces to display refresh rate.
    bool waitForDisplayReady(uint32_t timeoutMs = 8);

    /// Handle game window resize. Recreates DXGI swapchain + shared image.
    void resize(uint32_t width, uint32_t height);

    bool isActive() const { return active_; }

    /// Return CSV diagnostic string for debug bridge queries.
    std::string getDiagnostics() const;

  private:
    void diagLog(const char *fmt, ...);
    std::string initFailReason_;
    bool checkPreconditions();
    bool createOverlayWindow();
    bool createD3D11Device();
    bool createDxgiSwapChain();
    bool createDirectComposition();
    bool createSharedVulkanImage(int index);
    bool createPremultiplyPipeline();
    void destroyOverlayWindow();
    void destroySharedImages();
    void syncPosition();

    std::weak_ptr<Framework> framework_;

    // Win32
    void *gameHwnd_ = nullptr;    // HWND
    void *overlayHwnd_ = nullptr; // HWND
    uint32_t width_ = 0;
    uint32_t height_ = 0;

    // D3D11 / DXGI / DirectComposition
    ID3D11Device *d3dDevice_ = nullptr;
    ID3D11DeviceContext *d3dContext_ = nullptr;
    IDXGISwapChain1 *dxgiSwapChain_ = nullptr;
    void *frameLatencyWaitable_ = nullptr;  // HANDLE from GetFrameLatencyWaitableObject
    IDCompositionDevice *dcompDevice_ = nullptr;
    IDCompositionTarget *dcompTarget_ = nullptr;
    IDCompositionVisual *dcompVisual_ = nullptr;

    // Double-buffered Vulkan shared images (avoids race between render and present)
    static constexpr int kSharedImageCount = 2;
    VkImage sharedImages_[kSharedImageCount] = {};
    VkDeviceMemory sharedMemories_[kSharedImageCount] = {};
    VkImageView sharedViews_[kSharedImageCount] = {};
    void *sharedHandles_[kSharedImageCount] = {};         // HANDLE (NT)
    ID3D11Texture2D *d3dSharedTextures_[kSharedImageCount] = {};
    IDXGIKeyedMutex *sharedMutexes_[kSharedImageCount] = {};
    VkFramebuffer premultiplyFramebuffers_[kSharedImageCount] = {};
    uint32_t currentSharedIndex_ = 0;

    // Vulkan premultiply pipeline
    std::shared_ptr<vk::Shader> premultiplyVert_;
    std::shared_ptr<vk::Shader> premultiplyFrag_;
    std::shared_ptr<vk::RenderPass> premultiplyRenderPass_;
    std::shared_ptr<vk::GraphicsPipeline> premultiplyPipeline_;
    std::shared_ptr<vk::DescriptorTable> premultiplyDescriptor_;
    std::shared_ptr<vk::Sampler> premultiplySampler_;

    bool active_ = false;
};

#else
// Linux stub
class OverlayCompositor {
  public:
    bool init(std::shared_ptr<class Framework>) { return false; }
    void destroy() {}
    void recordPremultiply(std::shared_ptr<class vk::CommandBuffer>,
                           std::shared_ptr<class vk::DeviceLocalImage>, uint32_t) {}
    void present() {}
    void resize(uint32_t, uint32_t) {}
    bool isActive() const { return false; }
    std::string getDiagnostics() const { return "platform=linux,supported=0"; }
};
#endif
