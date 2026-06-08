#include "core/render/radiance_logger.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

static bool s_enabled = false;
static std::ofstream s_file;
static std::mutex s_mutex;
static std::filesystem::path s_filePath;
static constexpr size_t MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

static std::string currentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

static std::string makeLogFileName() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::ostringstream oss;
    oss << "radiance_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".log";
    return oss.str();
}

void RadianceLogger::setEnabled(bool enabled, const std::filesystem::path& folderPath) {
    std::lock_guard<std::mutex> lock(s_mutex);

    if (enabled && !s_enabled) {
        // Opening
        try {
            auto logsDir = folderPath / "logs";
            std::filesystem::create_directories(logsDir);
            s_filePath = logsDir / makeLogFileName();
            s_file.open(s_filePath, std::ios::app);
            if (s_file.is_open()) {
                s_enabled = true;
                s_file << "[" << currentTimestamp() << "] [Logger] [INFO] Logging started: "
                       << s_filePath.string() << std::endl;
                std::cout << "[RadianceLogger] Opened " << s_filePath << std::endl;
            } else {
                std::cerr << "[RadianceLogger] Failed to open " << s_filePath << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[RadianceLogger] Init error: " << e.what() << std::endl;
        }
    } else if (!enabled && s_enabled) {
        // Closing
        s_file << "[" << currentTimestamp() << "] [Logger] [INFO] Logging stopped" << std::endl;
        s_file.close();
        s_enabled = false;
        std::cout << "[RadianceLogger] Closed" << std::endl;
    }
}

bool RadianceLogger::isEnabled() {
    return s_enabled;
}

void RadianceLogger::log(const char* component, const char* level, const char* fmt, ...) {
    if (!s_enabled) return;

    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_file.is_open()) return;

    s_file << "[" << currentTimestamp() << "] [" << component << "] [" << level << "] "
           << buf << std::endl;

    // Rolling: check size
    auto pos = s_file.tellp();
    if (pos > 0 && static_cast<size_t>(pos) > MAX_FILE_SIZE) {
        s_file.close();
        try {
            auto bakPath = s_filePath;
            bakPath.replace_extension(".bak");
            std::filesystem::rename(s_filePath, bakPath);
        } catch (...) {}
        s_file.open(s_filePath, std::ios::trunc);
        if (s_file.is_open()) {
            s_file << "[" << currentTimestamp() << "] [Logger] [INFO] Log rotated (10MB limit)" << std::endl;
        }
    }
}

void RadianceLogger::logVkResult(const char* context, int result) {
    if (!s_enabled || result == 0) return; // VK_SUCCESS = 0
    log("Vulkan", "WARN", "%s returned VkResult=%d", context, result);
}

void RadianceLogger::shutdown() {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (s_file.is_open()) {
        s_file << "[" << currentTimestamp() << "] [Logger] [INFO] Shutdown" << std::endl;
        s_file.close();
    }
    s_enabled = false;
}
