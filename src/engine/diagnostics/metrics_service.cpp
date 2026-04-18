#include "metrics_service.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cassert>

namespace engine {

MetricsService::~MetricsService() {
    shutdown();
}

vk2::Result<void> MetricsService::init(vk2::DeviceService& device, uint32_t framesInFlight) {
    device_ = device.device();
    vma_ = device.vma();
    deviceName_ = device.caps().deviceName;
    memoryBudgetAvailable_ = device.caps().memoryBudget;

    auto profile = device.caps().profile;
    profileName_ = (profile == vk2::DeviceProfile::AdvancedRT ? "AdvancedRT" :
                    profile == vk2::DeviceProfile::BaselineRenderer ? "BaselineRenderer" :
                    "BootstrapPresent");

    // Check if GPU timestamps are available
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device.physicalDevice(), &props);
    timestampPeriodNs_ = props.limits.timestampPeriod;

    // Cache device-local heap flags for VRAM queries
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(device.physicalDevice(), &memProps);
    heapCount_ = memProps.memoryHeapCount;
    for (uint32_t i = 0; i < heapCount_; ++i) {
        deviceLocalHeaps_[i] = (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
    }

    if (timestampPeriodNs_ <= 0.0f) {
        log::warn("metrics", "GPU timestamps not supported (period=0)");
        gpuTimingAvailable_ = false;
    } else {
        gpuTimingAvailable_ = true;
        frames_.resize(framesInFlight);

        for (uint32_t i = 0; i < framesInFlight; ++i) {
            VkQueryPoolCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            ci.queryCount = kTotalQueryCount; // 2 frame + 2*kMaxNamedTimers named

            VkResult r = vkCreateQueryPool(device_, &ci, nullptr, &frames_[i].queryPool);
            if (r != VK_SUCCESS) {
                return vk2::makeError(r, "vkCreateQueryPool for metrics failed");
            }
        }

        log::info("metrics", "GPU timing initialized (period=" +
                  std::to_string(timestampPeriodNs_) + "ns)");
    }

    return {};
}

void MetricsService::shutdown() {
    for (auto& f : frames_) {
        if (f.queryPool != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device_, f.queryPool, nullptr);
            f.queryPool = VK_NULL_HANDLE;
        }
    }
    frames_.clear();
}

// --- CPU timing ---

void MetricsService::recordCpuFrameTime(float ms) {
    cpuTimes_[cpuRingPos_] = ms;
    cpuRingPos_ = (cpuRingPos_ + 1) % CPU_RING_SIZE;
    if (cpuRingCount_ < CPU_RING_SIZE) cpuRingCount_++;
}

float MetricsService::cpuFrameTimeAvg() const {
    if (cpuRingCount_ == 0) return 0.0f;
    float sum = 0.0f;
    for (uint32_t i = 0; i < cpuRingCount_; ++i) sum += cpuTimes_[i];
    return sum / static_cast<float>(cpuRingCount_);
}

float MetricsService::cpuFrameTimeStdDev() const {
    if (cpuRingCount_ < 2) return 0.0f;
    float avg = cpuFrameTimeAvg();
    float sumSq = 0.0f;
    for (uint32_t i = 0; i < cpuRingCount_; ++i) {
        float d = cpuTimes_[i] - avg;
        sumSq += d * d;
    }
    return std::sqrt(sumSq / static_cast<float>(cpuRingCount_));
}

uint32_t MetricsService::hitchCount() const {
    if (cpuRingCount_ < 10) return 0;
    float avg = cpuFrameTimeAvg();
    float threshold = avg * 2.0f;
    uint32_t count = 0;
    for (uint32_t i = 0; i < cpuRingCount_; ++i) {
        if (cpuTimes_[i] > threshold) count++;
    }
    return count;
}

// --- GPU timing ---

void MetricsService::beginGpuFrame(VkCommandBuffer cmd, uint32_t frameIndex) {
    if (!gpuTimingAvailable_ || frameIndex >= frames_.size()) return;
    // Reset the entire pool (frame timestamps + all named timer slots)
    vkCmdResetQueryPool(cmd, frames_[frameIndex].queryPool, 0, kTotalQueryCount);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                          frames_[frameIndex].queryPool, 0);
}

void MetricsService::endGpuFrame(VkCommandBuffer cmd, uint32_t frameIndex) {
    if (!gpuTimingAvailable_ || frameIndex >= frames_.size()) return;
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                          frames_[frameIndex].queryPool, 1);
}

void MetricsService::readGpuResults(uint32_t frameIndex) {
    if (!gpuTimingAvailable_ || frameIndex >= frames_.size()) return;

    struct TimestampWithAvail {
        uint64_t timestamp;
        uint64_t available;
    };

    // Read all queries in one call: 2 frame + 2*kMaxNamedTimers named pairs.
    TimestampWithAvail results[kTotalQueryCount]{};

    VkResult r = vkGetQueryPoolResults(
        device_, frames_[frameIndex].queryPool, 0, kTotalQueryCount,
        sizeof(results), results, sizeof(TimestampWithAvail),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

    if (r != VK_SUCCESS) return;

    std::lock_guard lock(resultsMutex_);

    // Whole-frame time (slots 0 and 1)
    if (results[0].available && results[1].available) {
        uint64_t delta = results[1].timestamp - results[0].timestamp;
        latestGpuMs_ = static_cast<float>(delta) * timestampPeriodNs_ / 1e6f;
    }

    // Named timer pairs (slot i → query indices 2+i*2 and 2+i*2+1)
    uint32_t registeredCount = static_cast<uint32_t>(timerNames_.size());
    for (uint32_t i = 0; i < registeredCount; ++i) {
        uint32_t beginIdx = 2 + i * 2;
        uint32_t endIdx   = 2 + i * 2 + 1;
        if (results[beginIdx].available && results[endIdx].available) {
            uint64_t delta = results[endIdx].timestamp - results[beginIdx].timestamp;
            namedTimerResults_[i] = static_cast<float>(delta) * timestampPeriodNs_ / 1e6f;
        }
    }
}

float MetricsService::gpuFrameTimeMs() const {
    std::lock_guard lock(resultsMutex_);
    return latestGpuMs_;
}

// --- Named per-adapter GPU timers ---

uint32_t MetricsService::registerTimer(std::string_view name) {
    uint32_t slot = static_cast<uint32_t>(timerNames_.size());
    assert(slot < kMaxNamedTimers && "MetricsService: too many named timers (> kMaxNamedTimers)");
    timerNames_.emplace_back(name);
    namedTimerResults_.push_back(0.0f);
    return slot;
}

void MetricsService::beginNamedTimer(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t slot) {
    if (!gpuTimingAvailable_ || frameIndex >= frames_.size()) return;
    if (slot >= timerNames_.size()) return;
    uint32_t queryIdx = 2 + slot * 2;
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                          frames_[frameIndex].queryPool, queryIdx);
}

void MetricsService::endNamedTimer(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t slot) {
    if (!gpuTimingAvailable_ || frameIndex >= frames_.size()) return;
    if (slot >= timerNames_.size()) return;
    uint32_t queryIdx = 2 + slot * 2 + 1;
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                          frames_[frameIndex].queryPool, queryIdx);
}

float MetricsService::namedTimerMs(uint32_t slot) const {
    std::lock_guard lock(resultsMutex_);
    if (slot >= namedTimerResults_.size()) return 0.0f;
    return namedTimerResults_[slot];
}

std::string MetricsService::getProfileString() const {
    std::lock_guard lock(resultsMutex_);
    if (timerNames_.empty()) return "";

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    bool first = true;
    for (uint32_t i = 0; i < static_cast<uint32_t>(timerNames_.size()); ++i) {
        if (!first) ss << ",";
        ss << timerNames_[i] << ":" << namedTimerResults_[i];
        first = false;
    }
    ss << ",TOTAL:" << latestGpuMs_;
    return ss.str();
}

// --- VRAM ---

MetricsService::VramSnapshot MetricsService::queryVram() const {
    VramSnapshot snap;
    if (!memoryBudgetAvailable_ || vma_ == VK_NULL_HANDLE) return snap;

    VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
    vmaGetHeapBudgets(const_cast<VmaAllocator>(vma_), budgets);

    // Only sum device-local heaps (avoid over-reporting on UMA/shared-memory)
    for (uint32_t i = 0; i < heapCount_; ++i) {
        if (deviceLocalHeaps_[i]) {
            snap.budgetBytes += budgets[i].budget;
            snap.usageBytes += budgets[i].usage;
        }
    }
    return snap;
}

// --- Status line ---

std::string MetricsService::statusLine() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "V2 " << profileName_ << " | " << deviceName_;
    ss << " | CPU " << cpuFrameTimeAvg() << "ms";
    ss << " | GPU " << gpuFrameTimeMs() << "ms";

    auto vram = queryVram();
    if (vram.budgetBytes > 0) {
        double usageGB = static_cast<double>(vram.usageBytes) / (1024.0 * 1024.0 * 1024.0);
        double budgetGB = static_cast<double>(vram.budgetBytes) / (1024.0 * 1024.0 * 1024.0);
        ss << " | VRAM " << std::setprecision(1) << usageGB << "/" << budgetGB << " GB";
    }

    return ss.str();
}

} // namespace engine
