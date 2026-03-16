#include "core/render/gpu_profiler.hpp"
#include <iostream>

GpuProfiler::~GpuProfiler() {
    destroy();
}

void GpuProfiler::init(std::shared_ptr<vk::Device> device, std::shared_ptr<vk::PhysicalDevice> physicalDevice,
                        uint32_t maxModules, uint32_t frameCount) {
    device_ = device->vkDevice();
    timestampPeriodNs_ = physicalDevice->properties().limits.timestampPeriod;

    if (timestampPeriodNs_ <= 0.0f) {
        std::cerr << "[GpuProfiler] timestampPeriod is 0 — GPU timestamps not supported" << std::endl;
        enabled_.store(false);
        return;
    }

    frames_.resize(frameCount);
    for (auto& frame : frames_) {
        VkQueryPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        ci.queryCount = MAX_TIMESTAMPS;

        VkResult result = vkCreateQueryPool(device_, &ci, nullptr, &frame.queryPool);
        if (result != VK_SUCCESS) {
            std::cerr << "[GpuProfiler] vkCreateQueryPool failed: " << result << std::endl;
            enabled_.store(false);
            return;
        }
        frame.moduleNames.reserve(maxModules);
    }

    std::cout << "[GpuProfiler] initialized: " << frameCount << " frames, "
              << "timestampPeriod=" << timestampPeriodNs_ << "ns" << std::endl;
}

void GpuProfiler::destroy() {
    for (auto& frame : frames_) {
        if (frame.queryPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device_, frame.queryPool, nullptr);
            frame.queryPool = VK_NULL_HANDLE;
        }
    }
    frames_.clear();
}

void GpuProfiler::beginFrame(VkCommandBuffer cmd, uint32_t frameIndex) {
    if (!enabled_.load() || frameIndex >= frames_.size()) return;
    auto& frame = frames_[frameIndex];

    vkCmdResetQueryPool(cmd, frame.queryPool, 0, MAX_TIMESTAMPS);
    frame.queryCount = 0;
    frame.moduleNames.clear();
    activeFrame_.store(frameIndex);

    // Query 0: frame start
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame.queryPool, 0);
    frame.queryCount = 1;
}

void GpuProfiler::beginModule(VkCommandBuffer cmd, const std::string& moduleName) {
    if (!enabled_.load()) return;
    uint32_t fi = activeFrame_.load();
    if (fi >= frames_.size()) return;
    auto& frame = frames_[fi];
    if (frame.queryCount >= MAX_TIMESTAMPS - 1) return;

    frame.moduleNames.push_back(moduleName);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame.queryPool, frame.queryCount);
    frame.queryCount++;
}

void GpuProfiler::endModule(VkCommandBuffer cmd) {
    if (!enabled_.load()) return;
    uint32_t fi = activeFrame_.load();
    if (fi >= frames_.size()) return;
    auto& frame = frames_[fi];
    if (frame.queryCount >= MAX_TIMESTAMPS) return;

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame.queryPool, frame.queryCount);
    frame.queryCount++;
}

void GpuProfiler::endFrame(VkCommandBuffer cmd) {
    if (!enabled_.load()) return;
    uint32_t fi = activeFrame_.load();
    if (fi >= frames_.size()) return;
    auto& frame = frames_[fi];
    if (frame.queryCount >= MAX_TIMESTAMPS) return;

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame.queryPool, frame.queryCount);
    frame.queryCount++;
    activeFrame_.store(UINT32_MAX); // No longer recording
}

void GpuProfiler::readResults(uint32_t frameIndex) {
    if (!enabled_.load() || frameIndex >= frames_.size()) return;
    auto& frame = frames_[frameIndex];
    if (frame.queryCount < 3) return;

    // Non-blocking read with availability check
    struct TimestampWithAvail { uint64_t timestamp; uint64_t available; };
    std::vector<TimestampWithAvail> data(frame.queryCount);
    VkResult result = vkGetQueryPoolResults(
        device_, frame.queryPool, 0, frame.queryCount,
        frame.queryCount * sizeof(TimestampWithAvail), data.data(),
        sizeof(TimestampWithAvail),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

    if (result != VK_SUCCESS && result != VK_NOT_READY) return;

    // Verify all timestamps are available
    std::vector<uint64_t> timestamps(frame.queryCount);
    for (uint32_t i = 0; i < frame.queryCount; i++) {
        if (!data[i].available) return; // Not all ready yet
        timestamps[i] = data[i].timestamp;
    }

    // Layout: [frame_start, mod0_begin, mod0_end, mod1_begin, mod1_end, ..., frame_end]
    float nsToMs = timestampPeriodNs_ / 1e6f;
    std::vector<ModuleTiming> timings;

    uint32_t qi = 1; // skip frame_start
    for (size_t m = 0; m < frame.moduleNames.size() && qi + 1 < frame.queryCount; m++) {
        uint64_t begin = timestamps[qi];
        uint64_t end = timestamps[qi + 1];
        float ms = (float)(end - begin) * nsToMs;
        timings.push_back({frame.moduleNames[m], ms});
        qi += 2;
    }

    float totalMs = (float)(timestamps[frame.queryCount - 1] - timestamps[0]) * nsToMs;

    std::lock_guard<std::mutex> lock(resultsMutex_);
    latestTimings_ = std::move(timings);
    latestTotalMs_ = totalMs;
}

std::vector<GpuProfiler::ModuleTiming> GpuProfiler::getModuleTimings() const {
    std::lock_guard<std::mutex> lock(resultsMutex_);
    return latestTimings_;
}

float GpuProfiler::getTotalGpuMs() const {
    std::lock_guard<std::mutex> lock(resultsMutex_);
    return latestTotalMs_;
}
