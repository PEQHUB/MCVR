#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>

#include "core/vulkan/all_core_vulkan.hpp"

/// Per-slot data for render->present handoff.
struct FrameSlot {
    std::shared_ptr<vk::DeviceLocalImage> worldImage;
    std::shared_ptr<vk::DeviceLocalImage> overlayImage;
    VkFence renderDoneFence = VK_NULL_HANDLE;
    bool hdrOutput = false;
    float hdrUiBrightnessNits = 0.0f;
    std::atomic<bool> ready{false};
};

/// Lock-free triple buffer for render->present handoff.
/// Three slots avoid producer-consumer contention: render writes to one
/// while present reads another, third ages.
class FrameSlotRing {
  public:
    /// Publish a completed frame. Called by render thread after GPU submit.
    void publish(std::shared_ptr<vk::DeviceLocalImage> world,
                 std::shared_ptr<vk::DeviceLocalImage> overlay,
                 VkFence renderDoneFence,
                 bool hdr, float uiBrightnessNits) {
        int idx = writeIndex_.load(std::memory_order_relaxed);
        auto &slot = slots_[idx];
        slot.worldImage = world;
        slot.overlayImage = overlay;
        slot.renderDoneFence = renderDoneFence;
        slot.hdrOutput = hdr;
        slot.hdrUiBrightnessNits = uiBrightnessNits;
        slot.ready.store(true, std::memory_order_release);
        writeIndex_.store((idx + 1) % 3, std::memory_order_relaxed);
        cv_.notify_one();
    }

    /// Consume the latest ready slot. Returns nullptr if none ready.
    /// Marks all slots as not-ready.
    FrameSlot *consume() {
        int w = writeIndex_.load(std::memory_order_relaxed);
        for (int i = 0; i < 3; i++) {
            int idx = (w - 1 - i + 6) % 3;
            if (slots_[idx].ready.load(std::memory_order_acquire)) {
                for (int j = 0; j < 3; j++)
                    slots_[j].ready.store(false, std::memory_order_relaxed);
                return &slots_[idx];
            }
        }
        return nullptr;
    }

    /// Block until a slot is ready, then consume it.
    FrameSlot *waitAndConsume(std::chrono::milliseconds timeout = std::chrono::milliseconds(16)) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait_for(lk, timeout, [this] {
            for (int i = 0; i < 3; i++)
                if (slots_[i].ready.load(std::memory_order_acquire)) return true;
            return false;
        });
        lk.unlock();
        return consume();
    }

    void reset() {
        for (auto &slot : slots_) {
            slot.worldImage = nullptr;
            slot.overlayImage = nullptr;
            slot.renderDoneFence = VK_NULL_HANDLE;
            slot.ready.store(false, std::memory_order_relaxed);
        }
        writeIndex_.store(0, std::memory_order_relaxed);
    }

  private:
    FrameSlot slots_[3];
    std::atomic<int> writeIndex_{0};
    std::mutex mtx_;
    std::condition_variable cv_;
};
