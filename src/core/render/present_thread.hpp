#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "core/render/frame_slot_ring.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

class Framework;
class OverlayCompositor;

/// Dedicated thread for presenting frames at display rate.
/// Active when frame generation is OFF. Paused when FG is ON
/// (Streamline interposer owns present in FG mode).
class PresentThread {
  public:
    PresentThread() = default;
    ~PresentThread();

    void init(std::shared_ptr<Framework> framework, FrameSlotRing *ring);
    void start();
    void pause();
    void resume();
    void stop();
    bool isRunning() const { return running_.load(std::memory_order_relaxed); }
    bool isPaused() const { return paused_.load(std::memory_order_relaxed); }

    /// Recreate Vulkan resources after swapchain recreation. Must be called while paused.
    void onSwapchainRecreate();

    /// Enable overlay-only mode: PresentThread does D3D11 copy + DXGI present (no Vulkan swapchain).
    void setOverlayMode(bool enabled, OverlayCompositor *compositor);

  private:
    void threadFunc();
    void createVulkanResources();
    void destroyVulkanResources();

    std::weak_ptr<Framework> framework_;
    std::shared_ptr<vk::Device> device_;
    FrameSlotRing *ring_ = nullptr;

    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> pauseAcknowledged_{false};
    std::mutex pauseMtx_;
    std::condition_variable pauseCv_;

    // Overlay compositor mode (atomic: written by render thread, read by present thread)
    std::atomic<OverlayCompositor *> overlayCompositor_{nullptr};
    std::atomic<bool> overlayMode_{false};

    // Vulkan resources (used exclusively by present thread)
    std::shared_ptr<vk::CommandPool> commandPool_;
    std::vector<std::shared_ptr<vk::CommandBuffer>> commandBuffers_;
    VkFence compositeFence_ = VK_NULL_HANDLE;
    VkSemaphore acquireSemaphore_ = VK_NULL_HANDLE;
    VkSemaphore compositeSemaphore_ = VK_NULL_HANDLE;
};
