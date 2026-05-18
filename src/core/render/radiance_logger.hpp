#pragma once

#include <cstdarg>
#include <filesystem>

// Opt-in structured logger. When disabled, isEnabled() returns false and
// all log() calls are no-ops (single branch, no allocation, no I/O).
// When enabled, writes timestamped entries to radiance/logs/radiance_<timestamp>.log.
// Thread-safe via mutex. Rolling file at 10MB.

class RadianceLogger {
public:
    static void setEnabled(bool enabled, const std::filesystem::path& folderPath);
    static bool isEnabled();

    // Printf-style logging. Component = "RayTracing", "Chunks", etc.
    // Level = "INFO", "WARN", "ERROR".
    static void log(const char* component, const char* level, const char* fmt, ...);

    // Convenience: log a Vulkan result if non-success
    static void logVkResult(const char* context, int result);

    // Flush and close the log file
    static void shutdown();
};
