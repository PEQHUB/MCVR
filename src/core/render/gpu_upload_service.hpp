#pragma once

#include "core/vulkan/all_core_vulkan.hpp"
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

/// Persistent GPU upload service for texture_loader_v4.
///
/// Owns a mapped staging ring buffer and submits texture/buffer uploads
/// asynchronously via a dedicated transfer queue (when available) or
/// the main graphics queue with timeline semaphore completion tracking.
///
/// No vkDeviceWaitIdle during normal load/reload.
/// No submit-and-wait blocking.
class GpuUploadService {
public:
    enum class Priority : uint8_t {
        FirstFrameAlbedo = 0,
        FirstFrameAux    = 1,
        FirstFrameMips   = 2,
        NearVisibleCtm   = 3,
        BackgroundCtm    = 4,
        Animation        = 5,
        Count
    };

    enum class Kind : uint8_t {
        TextureSubresource,
        BufferRange,
        MipGeneration,
        DescriptorUpdate
    };

    struct TextureUpload {
        uint64_t generation   = 0;
        uint32_t namespaceId  = 0;
        uint32_t tier         = 0;
        uint32_t page         = 0;
        uint32_t layer        = 0;
        uint32_t layerCount   = 0;
        uint32_t width        = 0;
        uint32_t height       = 0;
        uint32_t bytesPerLayer = 0;
        VkFormat format       = VK_FORMAT_R8G8B8A8_UNORM;
        VkImage dstImage      = VK_NULL_HANDLE;
        uint32_t mipLevels    = 1;
        const uint8_t* data   = nullptr;
        uint64_t bytes        = 0;
        bool visible          = false;
        Priority priority     = Priority::BackgroundCtm;
        std::function<void(uint64_t)> onComplete;
    };

    struct BufferUpload {
        uint64_t generation   = 0;
        VkBuffer dst          = VK_NULL_HANDLE;
        VkDeviceSize dstOffset = 0;
        const uint8_t* data   = nullptr;
        uint64_t bytes        = 0;
        const VkBufferCopy* regions = nullptr;
        uint32_t regionCount = 0;
        bool visible          = false;
        Priority priority     = Priority::BackgroundCtm;
        std::shared_ptr<vk::DeviceLocalBuffer> dstOwner;
    };

    struct Status {
        bool initialized              = false;
        bool hasDedicatedTransferQueue = false;
        bool asyncTransferQueueUpload = false;
        bool asyncMainQueueUpload     = false;
        bool blockingFenceUploads     = false;
        uint64_t pendingUploadBytes   = 0;
        uint64_t pendingVisibleUploadBytes = 0;
        uint64_t queuedBytes          = 0;
        uint64_t queuedVisibleBytes   = 0;
        uint64_t inFlightBytes        = 0;
        uint64_t inFlightVisibleBytes = 0;
        uint64_t submittedBytes       = 0;
        uint64_t completedBytes       = 0;
        uint64_t completedVisibleBytes = 0;
        uint64_t failedBytes          = 0;
        uint64_t timelineSubmissions  = 0;
        uint64_t actualVkCopyCommands = 0;
        uint64_t actualVkCopyBufferToImageCommands = 0;
        uint64_t actualVkBufferCopyCommands = 0;
        uint64_t vkDeviceWaitIdleDuringLoad = 0;
    };

    bool initialize(std::shared_ptr<vk::Device> device, std::shared_ptr<vk::VMA> vma, uint64_t stagingBytes);
    void shutdown();
    void cancelGeneration(uint64_t generation);

    bool enqueueTextureUpload(const TextureUpload& upload);
    bool enqueueBufferUpload(const BufferUpload& upload);

    /// Process pending uploads within a frame byte budget.
    void pump(uint64_t frameBudgetBytes);

    /// Poll timeline semaphore for completed submissions.
    void pollCompletions();

    /// True if a generation has no pending uploads (optionally visible-only).
    bool generationIdle(uint64_t generation, bool visibleOnly) const;

    Status status() const;
    std::string statusJson() const;

private:
    struct Pending {
        Kind kind;
        Priority priority;
        uint64_t generation;
        bool visible;
        // Texture fields
        uint32_t namespaceId, tier, page, layer, layerCount;
        uint32_t width, height, bytesPerLayer;
        VkFormat format;
        VkImage dstImage;
        uint32_t mipLevels;
        // Buffer fields
        VkBuffer dstBuffer;
        VkDeviceSize dstOffset;
        std::shared_ptr<vk::DeviceLocalBuffer> dstOwner;
        std::vector<VkBufferCopy> copyRegions;
        // Data
        std::vector<uint8_t> payload;
        std::function<void(uint64_t)> onComplete;
    };

    struct InFlight {
        uint64_t generation;
        VkCommandBuffer cmd;
        VkCommandPool cmdPool;
        std::shared_ptr<vk::HostVisibleBuffer> stagingBuffer;
        std::shared_ptr<vk::DeviceLocalBuffer> dstOwner;
        std::shared_ptr<vk::TimelineSemaphore> timeline;
        uint64_t timelineValue;
        uint64_t bytes;
        bool visible;
        std::function<void(uint64_t)> onComplete;
    };

    bool submitTextureUploadLocked(const Pending& pending);
    bool submitBufferUploadLocked(const Pending& pending);
    bool ensureStagingCapacityLocked(uint64_t bytes);

    std::shared_ptr<vk::Device> device_;
    std::shared_ptr<vk::VMA> vma_;

    // Transfer infrastructure
    std::shared_ptr<vk::CommandPool> transferCmdPool_;
    std::shared_ptr<vk::TimelineSemaphore> uploadTimeline_;
    uint64_t nextTimelineValue_ = 1;

    // Staging ring buffer
    std::shared_ptr<vk::HostVisibleBuffer> stagingRing_;
    uint64_t stagingRingSize_ = 0;
    uint64_t stagingRingOffset_ = 0;

    std::deque<Pending> queues_[static_cast<int>(Priority::Count)];
    std::deque<InFlight> inFlight_;

    mutable std::mutex mutex_;

    std::atomic<uint64_t> pendingUploadBytes_{0};
    std::atomic<uint64_t> pendingVisibleUploadBytes_{0};
    std::atomic<uint64_t> queuedBytes_{0};
    std::atomic<uint64_t> queuedVisibleBytes_{0};
    std::atomic<uint64_t> inFlightBytes_{0};
    std::atomic<uint64_t> inFlightVisibleBytes_{0};
    std::atomic<uint64_t> submittedBytes_{0};
    std::atomic<uint64_t> completedBytes_{0};
    std::atomic<uint64_t> completedVisibleBytes_{0};
    std::atomic<uint64_t> failedBytes_{0};
    std::atomic<uint64_t> timelineSubmissions_{0};
    std::atomic<uint64_t> actualVkCopyCommands_{0};
    std::atomic<uint64_t> actualVkCopyBufferToImageCommands_{0};
    std::atomic<uint64_t> actualVkBufferCopyCommands_{0};
    std::atomic<uint64_t> vkDeviceWaitIdleDuringLoad_{0};

    bool initialized_ = false;
    bool hasDedicatedTransferQueue_ = false;
    bool asyncTransferQueueUpload_ = false;
    bool asyncMainQueueUpload_ = false;
};
