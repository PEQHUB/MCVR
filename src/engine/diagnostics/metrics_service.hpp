#pragma once

// Ownership: EngineServices (1:1).
// Thread: init/shutdown on main thread. recordCpuFrameTime/readGpuResults on main thread.
//         beginGpuFrame/endGpuFrame called during command recording (main thread).
//         statusLine() and queryVram() are thread-safe (read-only or mutex-guarded).

#include "platform/vulkan/vk2_result.hpp"

#include <cstdint>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>

namespace engine {

namespace vk2 { class DeviceService; }

class MetricsService {
public:
    MetricsService() = default;
    ~MetricsService();

    MetricsService(const MetricsService&) = delete;
    MetricsService& operator=(const MetricsService&) = delete;

    vk2::Result<void> init(vk2::DeviceService& device, uint32_t framesInFlight);
    void shutdown();

    // --- CPU timing ---
    void recordCpuFrameTime(float ms);
    float cpuFrameTimeAvg() const;
    float cpuFrameTimeStdDev() const;
    uint32_t hitchCount() const;  // frames > 2x avg in last N frames

    // --- GPU timing ---
    void beginGpuFrame(VkCommandBuffer cmd, uint32_t frameIndex);
    void endGpuFrame(VkCommandBuffer cmd, uint32_t frameIndex);
    void readGpuResults(uint32_t frameIndex);
    float gpuFrameTimeMs() const;

    // --- VRAM ---
    struct VramSnapshot {
        uint64_t budgetBytes = 0;
        uint64_t usageBytes = 0;
    };
    VramSnapshot queryVram() const;

    // --- Status line ---
    std::string statusLine() const;

    // --- Counters ---
    uint32_t swapchainRecreateCount() const { return swapchainRecreates_; }
    void recordSwapchainRecreate() { ++swapchainRecreates_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator vma_ = VK_NULL_HANDLE;
    float timestampPeriodNs_ = 0.0f;
    bool gpuTimingAvailable_ = false;

    struct PerFrame {
        VkQueryPool queryPool = VK_NULL_HANDLE;
    };
    std::vector<PerFrame> frames_;

    // CPU timing ring
    static constexpr uint32_t CPU_RING_SIZE = 300;
    float cpuTimes_[CPU_RING_SIZE]{};
    uint32_t cpuRingPos_ = 0;
    uint32_t cpuRingCount_ = 0;

    mutable std::mutex resultsMutex_;
    float latestGpuMs_ = 0.0f;

    // Cached device info
    std::string deviceName_;
    std::string profileName_;
    bool memoryBudgetAvailable_ = false;
    uint32_t heapCount_ = 0;
    bool deviceLocalHeaps_[VK_MAX_MEMORY_HEAPS]{};
    uint32_t swapchainRecreates_ = 0;
};

} // namespace engine
