#pragma once

// Ownership: EngineServices (1:1).
// Thread: init/shutdown on main thread. recordCpuFrameTime/readGpuResults on main thread.
//         beginGpuFrame/endGpuFrame called during command recording (main thread).
//         beginNamedTimer/endNamedTimer called during command recording (main thread).
//         statusLine(), queryVram(), getProfileString() are thread-safe (mutex-guarded).

#include "platform/vulkan/vk2_result.hpp"

#include <cassert>
#include <cstdint>
#include <cmath>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>
#include <vk_mem_alloc.h>

namespace engine {

namespace vk2 { class DeviceService; }

class MetricsService {
public:
    // Maximum number of named per-adapter GPU timers.
    // Query pool slots: 0=frame-begin, 1=frame-end, then 2*kMaxNamedTimers pairs.
    static constexpr uint32_t kMaxNamedTimers = 16;

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

    // --- GPU timing (whole frame) ---
    void beginGpuFrame(VkCommandBuffer cmd, uint32_t frameIndex);
    void endGpuFrame(VkCommandBuffer cmd, uint32_t frameIndex);
    void readGpuResults(uint32_t frameIndex);
    float gpuFrameTimeMs() const;

    // --- GPU timing (named per-adapter timers) ---
    // Register a named timer. Returns the slot index (< kMaxNamedTimers).
    // Must be called before the first frame. Asserts if kMaxNamedTimers is exceeded.
    uint32_t registerTimer(std::string_view name);

    // Write begin/end timestamp for the given slot into the command buffer.
    // slot must be < timerNames_.size(). No-op if GPU timing unavailable.
    void beginNamedTimer(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t slot);
    void endNamedTimer(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t slot);

    // Returns the last-frame GPU time in milliseconds for the given slot.
    // Returns 0.0 if slot is out of range or result not yet available.
    float namedTimerMs(uint32_t slot) const;

    // Builds "RayTracing:5.234,DLSS:1.023,...,TOTAL:8.357" matching V1 GpuProfiler format.
    // Thread-safe.
    std::string getProfileString() const;

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

    // Total queries per frame: 2 (frame begin/end) + 2*kMaxNamedTimers (named pairs).
    static constexpr uint32_t kTotalQueryCount = 2 + 2 * kMaxNamedTimers;

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

    // Named timer registration (indexed by slot).
    std::vector<std::string> timerNames_;                   // slot → name
    std::vector<float> namedTimerResults_;                  // slot → last-frame ms

    // Cached device info
    std::string deviceName_;
    std::string profileName_;
    bool memoryBudgetAvailable_ = false;
    uint32_t heapCount_ = 0;
    bool deviceLocalHeaps_[VK_MAX_MEMORY_HEAPS]{};
    uint32_t swapchainRecreates_ = 0;
};

} // namespace engine
