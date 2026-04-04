#pragma once

// Ownership: LogService owned by EngineServices (1:1 lifetime).
// Thread: All methods are thread-safe (spdlog handles internal synchronization).
// Dependencies: None (standalone).

#include <string_view>
#include <string>
#include <unordered_map>
#include <cstdint>

namespace engine::log {

struct LogConfig {
    std::string logDir = "logs";        // Directory for log files
    std::string consolePattern;         // Empty = default colored pattern
    uint32_t maxFileSizeMB = 5;         // Per-file size before rotation
    uint32_t maxFiles = 3;              // Number of rotated files to keep
    bool consoleEnabled = true;
    bool fileEnabled = true;
    bool jsonlEnabled = true;           // Structured JSONL sink
};

// Initialize logging. Call once at startup.
void init(const LogConfig& config);

// Flush all sinks and shut down. Call once at shutdown.
void shutdown();

// Standard severity levels
void trace(std::string_view category, std::string_view msg);
void debug(std::string_view category, std::string_view msg);
void info(std::string_view category, std::string_view msg);
void warn(std::string_view category, std::string_view msg);
void error(std::string_view category, std::string_view msg);

// Fatal: flushes all sinks, writes crash context, then calls std::abort().
[[noreturn]] void fatal(std::string_view category, std::string_view msg);

// Structured event (written to JSONL sink).
void event(std::string_view category, std::string_view eventId,
           const std::unordered_map<std::string, std::string>& fields = {});

// Runtime category filter (all enabled by default).
void setCategoryEnabled(std::string_view category, bool enabled);

// Check if initialized (safe to call before init).
bool isInitialized();

} // namespace engine::log

// Debug-only assertion — fires log::fatal (aborts) on invariant violation.
#ifndef NDEBUG
#define ENGINE_ASSERT(cond, msg) \
    do { if (!(cond)) { engine::log::fatal("assert", msg); } } while(0)
#else
#define ENGINE_ASSERT(cond, msg) ((void)0)
#endif
