#include "core/render/crash_ring_buffer.hpp"
#include "core/render/gpu_diagnostics.hpp"
#include "core/render/aftermath_integration.hpp"
#include "core/render/render_framework.hpp"
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
        f << "Total frames: " << frameCounter_.load(std::memory_order_relaxed) << std::endl;
        f << "Entries (oldest first):" << std::endl;
        f << std::endl;

        // Determine range
        uint32_t writeIndex = writeIndex_.load(std::memory_order_acquire);
        uint32_t count = (writeIndex < CAPACITY) ? writeIndex : CAPACITY;
        uint32_t start = (writeIndex < CAPACITY) ? 0 : (writeIndex % CAPACITY);

        for (uint32_t i = 0; i < count; i++) {
            const auto& e = entries_[(start + i) % CAPACITY];
            f << "  frame=" << std::setw(8) << e.frameNumber
              << "  t=" << std::setw(16) << e.timestampNs
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
    crashExitWithQueue(vkResult, context, VK_NULL_HANDLE);
}

void crashExitWithQueue(int vkResult, const char* context, VkQueue queue) {
    std::cerr << "[CRASH] " << context << " (VkResult=" << vkResult << ")" << std::endl;
    g_crashRing.record(context, vkResult);

    auto logsDir = Renderer::folderPath / "logs";

    // GPU diagnostics: query checkpoints + device fault before dump
    if (vkResult == -4 /* VK_ERROR_DEVICE_LOST */) {
        // Also query secondary queue checkpoints if available
        auto framework = Renderer::instance().framework();
        if (framework && framework->device()) {
            VkQueue secQueue = framework->device()->secondaryQueue();
            if (secQueue != VK_NULL_HANDLE && secQueue != queue) {
                auto writeToStderr = [](const std::string& line) { std::cerr << line << std::endl; };
                std::cerr << "\n--- Secondary Queue ---" << std::endl;
                GpuDiag::queryAndPrintCheckpoints(secQueue, "Secondary Queue", writeToStderr);
                std::cerr << "--- End Secondary Queue ---\n" << std::endl;
            }
        }
        GpuDiag::onDeviceLost(queue, logsDir, g_crashRing.frameCount());

        // Wait for Aftermath to collect GPU crash dump (up to 5 seconds)
        std::string dumpPath = AftermathIntegration::waitForCrashDump(5000);
        if (!dumpPath.empty()) {
            std::cerr << "[CRASH] Aftermath crash dump: " << dumpPath << std::endl;
        }
    }

    g_crashRing.dumpToFile(logsDir);
    AftermathIntegration::shutdown();

    exit(EXIT_FAILURE);
}
