#include "boot_trace.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

namespace engine::boot_trace {

namespace {

std::mutex g_mutex;
std::string g_logPath;
std::string g_logDir;
std::string g_lastBreadcrumb;
std::string g_lastBridgeCommand;
std::atomic<uint64_t> g_lastFrame{0};
std::atomic<bool> g_initialized{false};
std::atomic<bool> g_frozen{false};

std::string formatTimestamp() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms  = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    auto t   = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    char out[40];
    std::snprintf(out, sizeof(out), "%s.%03d", buf, static_cast<int>(ms.count()));
    return out;
}

std::string compactTimestamp() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t   = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return buf;
}

void writeLineToLog(const std::string& line) {
    if (g_logPath.empty() || g_frozen.load(std::memory_order_acquire)) return;
    std::ofstream out(g_logPath, std::ios::app | std::ios::binary);
    if (!out.is_open()) {
        // Surface the failure — stderr is the last resort when the primary log
        // is unwritable (permissions, locked file, disk full).
        std::fprintf(stderr,
            "[V2-BOOT] WARN cannot open %s for append; line lost: %s\n",
            g_logPath.c_str(), line.c_str());
        return;
    }
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
    out.put('\n');
    out.flush();
    // ofstream destructor closes the file — line is committed before any
    // subsequent instruction (including a crash) runs.
}

// Resolve %TEMP% (Windows) or /tmp (POSIX) for a fallback log location when
// the requested logDir is unwritable. Returns empty string if no fallback
// candidate is available.
std::string tempFallbackPath() {
#if defined(_WIN32)
    if (const char* t = std::getenv("TEMP")) {
        return (std::filesystem::path(t) / "radiance_v2_boot.log").string();
    }
    if (const char* t = std::getenv("TMP")) {
        return (std::filesystem::path(t) / "radiance_v2_boot.log").string();
    }
    return {};
#else
    return "/tmp/radiance_v2_boot.log";
#endif
}

} // anonymous namespace

void init(const std::string& logDir) {
    std::lock_guard lock(g_mutex);
    g_logDir = logDir;

    std::error_code ec;
    std::filesystem::create_directories(logDir, ec);
    if (ec) {
        std::fprintf(stderr,
            "[V2-BOOT] WARN create_directories(%s) failed: %s\n",
            logDir.c_str(), ec.message().c_str());
    }

    g_logPath = (std::filesystem::path(logDir) / "v2_boot.log").string();

    auto tryTruncate = [&](const std::string& path) -> bool {
        std::ofstream truncate(path, std::ios::trunc | std::ios::binary);
        if (!truncate.is_open()) return false;
        std::string header = formatTimestamp() + " [V2-BOOT] === boot trace start ===\n";
        truncate.write(header.data(), static_cast<std::streamsize>(header.size()));
        truncate.flush();
        return truncate.good();
    };

    if (!tryTruncate(g_logPath)) {
        std::fprintf(stderr,
            "[V2-BOOT] WARN primary log path %s unwritable; trying fallback\n",
            g_logPath.c_str());
        std::string fallback = tempFallbackPath();
        if (!fallback.empty() && tryTruncate(fallback)) {
            g_logPath = fallback;
            std::fprintf(stderr,
                "[V2-BOOT] INFO boot log redirected to %s\n", g_logPath.c_str());
        } else {
            std::fprintf(stderr,
                "[V2-BOOT] ERROR no writable location for v2_boot.log; "
                "breadcrumbs will stream to stderr only\n");
            g_logPath.clear();
        }
    }

    g_lastBreadcrumb.clear();
    g_lastBridgeCommand.clear();
    g_lastFrame.store(0, std::memory_order_release);
    g_initialized.store(true, std::memory_order_release);
    g_frozen.store(false, std::memory_order_release);
}

void breadcrumb(std::string_view stage,
                std::string_view status,
                std::string_view detail) {
    std::ostringstream oss;
    oss << formatTimestamp()
        << " [V2-BOOT] " << stage << ' ' << status;
    if (!detail.empty()) {
        oss << " - " << detail;
    }
    std::string line = oss.str();

    {
        std::lock_guard lock(g_mutex);
        g_lastBreadcrumb = line;
        writeLineToLog(line);
    }

    // Also echo to stderr so JVM-captured output shows boot progress without
    // the user having to dig through v2_boot.log when debugging interactively.
    std::fprintf(stderr, "%s\n", line.c_str());
}

void recordBridgeCommand(std::string_view cmdType, uint64_t seq) {
    std::ostringstream oss;
    oss << cmdType << " #" << seq;
    std::lock_guard lock(g_mutex);
    g_lastBridgeCommand = oss.str();
}

void recordFrame(uint64_t frameIndex) {
    g_lastFrame.store(frameIndex, std::memory_order_release);
}

std::string lastBreadcrumb() {
    std::lock_guard lock(g_mutex);
    return g_lastBreadcrumb;
}

std::string lastBridgeCommand() {
    std::lock_guard lock(g_mutex);
    return g_lastBridgeCommand;
}

uint64_t lastFrameIndex() {
    return g_lastFrame.load(std::memory_order_acquire);
}

std::string writeCrashDump(std::string_view reason,
                           int vkResult,
                           std::string_view extraDetail) {
    std::lock_guard lock(g_mutex);

    std::string crashPath;
    if (!g_logDir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(g_logDir, ec);
        crashPath = (std::filesystem::path(g_logDir) / ("v2_crash_" + compactTimestamp() + ".log")).string();
    }
    if (crashPath.empty()) return {};

    std::ofstream out(crashPath, std::ios::trunc | std::ios::binary);
    if (!out.is_open()) return {};

    auto writeKv = [&](std::string_view k, const std::string& v) {
        out << k << ": " << v << '\n';
    };
    out << "=== V2 crash dump ===\n";
    writeKv("timestamp",      formatTimestamp());
    writeKv("reason",         std::string(reason));
    writeKv("last_breadcrumb", g_lastBreadcrumb.empty() ? std::string("<none>") : g_lastBreadcrumb);
    writeKv("last_bridge_cmd", g_lastBridgeCommand.empty() ? std::string("<none>") : g_lastBridgeCommand);
    writeKv("last_frame_index", std::to_string(g_lastFrame.load(std::memory_order_acquire)));
    if (vkResult != 0) {
        writeKv("vk_result", std::to_string(vkResult));
    }
    if (!extraDetail.empty()) {
        out << "extra_detail:\n" << extraDetail << '\n';
    }
    out.flush();

    // Also drop a one-line marker into v2_boot.log so the chronology is clear.
    std::string marker = formatTimestamp() + " [V2-BOOT] crash classifier wrote " + crashPath;
    if (!g_logPath.empty()) {
        std::ofstream blog(g_logPath, std::ios::app | std::ios::binary);
        if (blog.is_open()) {
            blog.write(marker.data(), static_cast<std::streamsize>(marker.size()));
            blog.put('\n');
            blog.flush();
        }
    }
    std::fprintf(stderr, "%s\n", marker.c_str());

    return crashPath;
}

void writeTickHeartbeat(uint64_t frameIndex, std::string_view extra) {
    std::string path;
    {
        std::lock_guard lock(g_mutex);
        if (g_logDir.empty()) return;
        path = (std::filesystem::path(g_logDir) / "v2_tick.log").string();
    }
    std::ofstream out(path, std::ios::app | std::ios::binary);
    if (!out.is_open()) return;
    std::ostringstream oss;
    oss << formatTimestamp() << " [V2-TICK] frame=" << frameIndex;
    if (!extra.empty()) {
        oss << ' ' << extra;
    }
    oss << '\n';
    std::string line = oss.str();
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
    out.flush();
}

void freezeBootLog() { g_frozen.store(true,  std::memory_order_release); }
void unfreezeBootLog() { g_frozen.store(false, std::memory_order_release); }

} // namespace engine::boot_trace
