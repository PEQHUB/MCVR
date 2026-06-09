#pragma once

#include "core/render/gpu_upload_service.hpp"
#include "core/render/texture_page_pool.hpp"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

/// Native side of texture_loader_v4.
///
/// Owns the GpuUploadService and TexturePagePool.
/// Receives generation-scoped upload requests from Java via JNI.
/// Publishes readiness state for first-frame gate.
class TextureLoaderV4 {
public:
    struct UploadRequest {
        uint64_t generation = 0;
        uint32_t namespaceId = 0;
        uint32_t tier = 0;
        uint32_t page = 0;
        uint32_t startLayer = 0;
        uint32_t layerCount = 0;
        uint32_t layerCapacity = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        const uint8_t* albedoData = nullptr;
        const uint8_t* specularData = nullptr;
        const uint8_t* normalData = nullptr;
        const uint8_t* flagData = nullptr;
        uint64_t bytesPerLayer = 0;
        uint32_t channelMask = 0;
        bool visible = false;
    };

    bool initialize(std::shared_ptr<vk::Device> device, std::shared_ptr<vk::VMA> vma);
    void shutdown();

    /// Begin a new generation. Allocates page pools.
    bool beginGeneration(uint64_t generation);

    /// Enqueue a tiered page upload.
    bool enqueueUpload(const UploadRequest& request);

    /// Commit a generation. Finalizes page pools and publishes.
    bool commitGeneration(uint64_t generation);

    /// Cancel a generation. Discards all pending uploads.
    bool cancelGeneration(uint64_t generation, int reasonCode);

    /// Per-frame pump: process pending uploads within budget.
    void pump(uint64_t frameBudgetBytes);

    /// Poll for completed uploads.
    void pollCompletions();

    /// True if a generation has no pending work.
    bool generationIdle(uint64_t generation, bool visibleOnly) const;

    /// Status JSON for DebugBridge.
    std::string statusJson() const;
    std::string tierStatusJson() const;
    std::string uploadQueueStatusJson() const;
    std::string pagePoolStatusJson() const;
    std::string firstFrameReadinessJson(uint64_t generation) const;

    GpuUploadService& uploadService() { return uploadService_; }
    TexturePagePool& pagePool() { return pagePool_; }

    bool isInitialized() const { return initialized_.load(std::memory_order_acquire); }

private:
    GpuUploadService uploadService_;
    TexturePagePool pagePool_;

    std::atomic<uint64_t> activeGeneration_{0};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> generationCommitted_{false};
    mutable std::mutex mutex_;
};
