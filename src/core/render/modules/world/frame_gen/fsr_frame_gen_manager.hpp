#pragma once

#include "core/vulkan/all_core_vulkan.hpp"
#include "core/render/render_framework.hpp"
#include <memory>

/// Manages FSR 3.1 Frame Generation: proxy swapchain + per-frame prepare/configure.
/// Static class mirroring FrameGenManager's architecture. Not a WorldModule.
/// Uses FidelityFX SDK FFX API v4 (same as FSR3 upscaler).
class FsrFrameGenManager {
public:
    /// Initialize (query FFX availability). Called once at framework init.
    static bool init(std::shared_ptr<vk::Device> device,
                     std::shared_ptr<vk::PhysicalDevice> physDevice);

    /// Query if FSR FG is currently active (proxy swapchain installed).
    static bool isActive();

    /// Called each frame BEFORE upscaling: dispatches FG Prepare with depth+MVs.
    static void prepare(VkCommandBuffer cmdBuffer,
                        VkImage depthImage, VkFormat depthFormat,
                        uint32_t depthWidth, uint32_t depthHeight,
                        VkImage mvImage, VkFormat mvFormat,
                        uint32_t mvWidth, uint32_t mvHeight,
                        float jitterX, float jitterY,
                        float mvScaleX, float mvScaleY,
                        float frameTimeDelta,
                        float cameraNear, float cameraFar,
                        float cameraFovVertical,
                        const float cameraPos[3],
                        const float cameraUp[3],
                        const float cameraRight[3],
                        const float cameraForward[3]);

    /// Called each frame AFTER compositing: configure FG for this frame.
    static void configureFrame(std::shared_ptr<FrameworkContext> context,
                               VkImage hudlessImage, VkFormat hudlessFormat,
                               uint32_t hudlessWidth, uint32_t hudlessHeight,
                               VkImage uiImage, VkFormat uiFormat,
                               uint32_t uiWidth, uint32_t uiHeight);

    /// Present via proxy swapchain's replacement vkQueuePresentKHR.
    static VkResult present(VkQueue queue, const VkPresentInfoKHR* presentInfo);

    /// Before swapchain recreation: destroy proxy context.
    static void beforeSwapchainRecreate();

    /// After swapchain recreation: create proxy if FSR FG is wanted.
    static void afterSwapchainRecreate(std::shared_ptr<vk::Swapchain> swapchain,
                                       std::shared_ptr<vk::Device> device,
                                       std::shared_ptr<vk::PhysicalDevice> physDevice);

    static void shutdown();

    /// Proxy swapchain replacement functions (valid only when active)
    static PFN_vkAcquireNextImageKHR   getAcquireNextImage();
    static PFN_vkGetSwapchainImagesKHR getGetSwapchainImages();

private:
    static bool initialized_;
    static bool active_;
    static bool pendingEnable_;
    static uint64_t frameId_;

    static std::shared_ptr<vk::Device> device_;
    static std::shared_ptr<vk::PhysicalDevice> physDevice_;

    // Proxy swapchain replacement function pointers
    static PFN_vkQueuePresentKHR       proxyPresent_;
    static PFN_vkAcquireNextImageKHR   proxyAcquire_;
    static PFN_vkGetSwapchainImagesKHR proxyGetImages_;

    // FFX contexts are opaque — stored as void* to avoid header pollution
    static void* fgContext_;           // Frame generation context
    static void* fgSwapChainContext_;  // Proxy swapchain context
    static bool fgContextCreated_;
    static bool fgSwapChainCreated_;
};
