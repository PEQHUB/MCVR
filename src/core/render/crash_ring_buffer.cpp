#include "core/render/crash_ring_buffer.hpp"
#include "core/render/renderer.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>

void CrashRingBuffer::dumpToFile(const std::filesystem::path& dir) const {
    try {
        std::filesystem::create_directories(dir);
        std::filesystem::path filePath = dir / "crash_ring.txt";

        std::ofstream f(filePath, std::ios::trunc);
        if (!f.is_open()) {
            std::cerr << "[CrashRing] Failed to open " << filePath << std::endl;
            return;
        }

        // Header
        auto now = std::chrono::system_clock::now();
        auto nowTime = std::chrono::system_clock::to_time_t(now);
        f << "=== Radiance Crash Ring Buffer ===" << std::endl;
        f << "Dump time: " << std::put_time(std::localtime(&nowTime), "%Y-%m-%d %H:%M:%S") << std::endl;
        f << "Total frames: " << frameCounter_ << std::endl;
        f << "Entries (oldest first):" << std::endl;
        f << std::endl;

        // Determine range
        uint32_t count = (writeIndex_ < CAPACITY) ? writeIndex_ : CAPACITY;
        uint32_t start = (writeIndex_ < CAPACITY) ? 0 : (writeIndex_ % CAPACITY);

        for (uint32_t i = 0; i < count; i++) {
            const auto& e = entries_[(start + i) % CAPACITY];
            f << "  frame=" << std::setw(8) << e.frameNumber
              << "  vk=" << std::setw(4) << e.vkResult
              << "  " << e.tag
              << std::endl;
        }

        f << std::endl << "=== End ===" << std::endl;
        f.close();

        std::cerr << "[CrashRing] Dumped " << count << " entries to " << filePath << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "[CrashRing] Exception during dump: " << ex.what() << std::endl;
    }
}

void crashExit(int vkResult, const char* context) {
    std::cerr << "[CRASH] " << context << " (VkResult=" << vkResult << ")" << std::endl;
    g_crashRing.record(context, vkResult);

    auto logsDir = Renderer::folderPath / "logs";
    g_crashRing.dumpToFile(logsDir);

    exit(EXIT_FAILURE);
}
