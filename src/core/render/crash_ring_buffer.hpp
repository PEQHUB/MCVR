#pragma once

#include <cstdint>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <vulkan/vulkan.h>

// Lockless crash-safe ring buffer. Always active, near-zero overhead.
// Records the last 64 frame events so VK_ERROR_DEVICE_LOST crashes
// leave a readable trail in radiance/logs/crash_ring.txt.

struct CrashRingEntry {
    uint64_t frameNumber;
    uint64_t timestampNs;
    int32_t  vkResult;
    char     tag[52]; // null-terminated event description
};
static_assert(sizeof(CrashRingEntry) == 72, "CrashRingEntry must be 72 bytes");

class CrashRingBuffer {
public:
    static constexpr int CAPACITY = 64;

    inline void advanceFrame() { frameCounter_++; }

    inline void record(const char* tag, int vkResult = 0) {
        auto& e = entries_[writeIndex_ % CAPACITY];
        e.frameNumber = frameCounter_;
        e.timestampNs = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        e.vkResult = vkResult;
        std::strncpy(e.tag, tag, sizeof(e.tag) - 1);
        e.tag[sizeof(e.tag) - 1] = '\0';
        writeIndex_++;
    }

    void dumpToFile(const std::filesystem::path& dir) const;

    uint64_t frameCount() const { return frameCounter_; }

private:
    CrashRingEntry entries_[CAPACITY] = {};
    uint32_t writeIndex_ = 0;
    uint64_t frameCounter_ = 0;
};

// Global instance
inline CrashRingBuffer g_crashRing;

// Helper: dump ring buffer + context, then exit.
void crashExit(int vkResult, const char* context);
// Same as crashExit but also queries GPU diagnostics from the given queue.
void crashExitWithQueue(int vkResult, const char* context, VkQueue queue);
