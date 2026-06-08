#pragma once

#include "core/vulkan/all_core_vulkan.hpp"
#include <string>
#include <vector>
#include <mutex>
#include <atomic>

/**
 * GPU timestamp profiler. Inserts vkCmdWriteTimestamp calls around each pipeline
 * module, reads back results, and exposes per-module GPU timings in milliseconds.
 *
 * Usage:
 *   profiler.beginFrame(cmdBuffer)   — reset queries, write frame-start timestamp
 *   profiler.beginModule(cmdBuffer, "RayTracing")
 *   profiler.beginModule(cmdBuffer, "RT:MainTrace") // nested scopes are supported
 *   ... module render ...
 *   profiler.endModule(cmdBuffer)
 *   profiler.endModule(cmdBuffer)
 *   profiler.endFrame(cmdBuffer)     — write frame-end timestamp
 *   // After fence wait (next frame):
 *   profiler.readResults(device)
 *   profiler.getModuleTimings()      — returns vector of {name, ms}
 */
class GpuProfiler {
public:
    struct ModuleTiming {
        std::string name;
        float ms;  // GPU milliseconds
    };

    GpuProfiler() = default;
    ~GpuProfiler();

    void init(std::shared_ptr<vk::Device> device, std::shared_ptr<vk::PhysicalDevice> physicalDevice,
              uint32_t maxModules, uint32_t frameCount);
    void destroy();

    // Per-frame recording (called on the render thread)
    void beginFrame(VkCommandBuffer cmd, uint32_t frameIndex);
    void beginModule(VkCommandBuffer cmd, const std::string& moduleName);
    void endModule(VkCommandBuffer cmd);
    void endFrame(VkCommandBuffer cmd);

    // Read results from previous frame (called after fence wait)
    void readResults(uint32_t frameIndex);

    // Get latest results (thread-safe, can be called from JNI)
    std::vector<ModuleTiming> getModuleTimings() const;
    float getTotalGpuMs() const;

    bool isEnabled() const { return enabled_.load(); }
    void setEnabled(bool enabled) { enabled_.store(enabled); }

private:
    static constexpr uint32_t MAX_TIMESTAMPS = 128; // frame bounds + nested module pairs

    struct PerFrame {
        struct Event {
            std::string name;
            uint32_t beginQuery = UINT32_MAX;
            uint32_t endQuery = UINT32_MAX;
        };

        VkQueryPool queryPool = VK_NULL_HANDLE;
        uint32_t queryCount = 0;
        std::vector<Event> events;
        std::vector<uint32_t> moduleStack;
    };

    VkDevice device_ = VK_NULL_HANDLE;
    float timestampPeriodNs_ = 1.0f; // nanoseconds per tick
    std::vector<PerFrame> frames_;
    std::atomic<bool> enabled_{false};  // OFF by default — zero cost until explicitly enabled
    std::atomic<uint32_t> activeFrame_{UINT32_MAX}; // which frame is currently recording

    // Latest results (protected by mutex for cross-thread reads)
    mutable std::mutex resultsMutex_;
    std::vector<ModuleTiming> latestTimings_;
    float latestTotalMs_ = 0.0f;
};
