#include "log.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/pattern_formatter.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_set>

namespace engine::log {

namespace {

std::shared_ptr<spdlog::logger> g_console;
std::shared_ptr<spdlog::logger> g_file;
std::shared_ptr<spdlog::logger> g_jsonl;
std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> g_crashRing;
std::atomic<bool> g_initialized{false};
std::string g_logDir = "logs";

std::mutex g_filterMutex;
std::unordered_set<std::string> g_disabledCategories;

bool isCategoryEnabled(std::string_view category) {
    std::lock_guard lock(g_filterMutex);
    return g_disabledCategories.find(std::string(category)) == g_disabledCategories.end();
}

std::string formatMessage(std::string_view category, std::string_view msg) {
    std::string result;
    result.reserve(category.size() + msg.size() + 4);
    result.append("[");
    result.append(category);
    result.append("] ");
    result.append(msg);
    return result;
}

void jsonEscape(std::string& out, std::string_view sv) {
    for (char c : sv) {
        switch (c) {
            case '"':  out.append("\\\""); break;
            case '\\': out.append("\\\\"); break;
            case '\b': out.append("\\b");  break;
            case '\f': out.append("\\f");  break;
            case '\n': out.append("\\n");  break;
            case '\r': out.append("\\r");  break;
            case '\t': out.append("\\t");  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out.append(buf);
                } else {
                    out.push_back(c);
                }
        }
    }
}

std::string formatJsonl(std::string_view category, std::string_view eventId,
                        const std::unordered_map<std::string, std::string>& fields) {
    std::string json = "{\"cat\":\"";
    jsonEscape(json, category);
    json.append("\",\"event\":\"");
    jsonEscape(json, eventId);
    json.append("\"");
    for (const auto& [k, v] : fields) {
        json.append(",\"");
        jsonEscape(json, k);
        json.append("\":\"");
        jsonEscape(json, v);
        json.append("\"");
    }
    json.append("}");
    return json;
}

} // anonymous namespace

void init(const LogConfig& config) {
    if (g_initialized.exchange(true)) return;

    g_logDir = config.logDir;

    std::vector<spdlog::sink_ptr> sinks;

    // Console sink (stderr, colored)
    if (config.consoleEnabled) {
        auto consoleSink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        if (!config.consolePattern.empty()) {
            consoleSink->set_pattern(config.consolePattern);
        } else {
            consoleSink->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
        }
        sinks.push_back(consoleSink);
    }

    // Crash ring buffer (last 256 entries, always active)
    g_crashRing = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(256);
    sinks.push_back(g_crashRing);

    // Create log directory
    std::filesystem::path logDir(config.logDir);
    if (config.fileEnabled || config.jsonlEnabled) {
        std::error_code ec;
        std::filesystem::create_directories(logDir, ec);
    }

    // Rotating text file sink
    if (config.fileEnabled) {
        auto fileSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            (logDir / "engine.log").string(),
            config.maxFileSizeMB * 1024 * 1024,
            config.maxFiles);
        fileSink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
        sinks.push_back(fileSink);
    }

    // JSONL structured file sink (separate logger)
    if (config.jsonlEnabled) {
        auto jsonlSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            (logDir / "engine.jsonl").string(),
            config.maxFileSizeMB * 1024 * 1024,
            config.maxFiles);
        jsonlSink->set_pattern("%v");  // Raw message only (we format JSON ourselves)
        g_jsonl = std::make_shared<spdlog::logger>("jsonl", jsonlSink);
        g_jsonl->set_level(spdlog::level::info);
        // Flush on info so every structured event hits disk immediately.
        // JSONL volume is low (one line per event) so the cost is negligible
        // and guarantees records survive a DEVICE_LOST / JVM abort.
        g_jsonl->flush_on(spdlog::level::info);
    }

    // Main logger uses all non-JSONL sinks
    g_console = std::make_shared<spdlog::logger>("engine", sinks.begin(), sinks.end());
    g_console->set_level(spdlog::level::trace);
    g_console->flush_on(spdlog::level::info);

    spdlog::set_default_logger(g_console);
}

void shutdown() {
    // Mark uninitialized FIRST so any concurrent or subsequent log calls (including
    // those fired from ~EngineApp during dllmain_crt_process_detach) bail out at
    // the g_initialized guard before touching g_filterMutex / g_disabledCategories,
    // which may already be in a destructed state during Windows DLL unload.
    if (!g_initialized.exchange(false)) return;

    if (g_console) g_console->flush();
    if (g_file) g_file->flush();
    if (g_jsonl) g_jsonl->flush();

    spdlog::shutdown();
    g_console.reset();
    g_file.reset();
    g_jsonl.reset();
    g_crashRing.reset();
}

void trace(std::string_view category, std::string_view msg) {
    if (!g_initialized.load() || !isCategoryEnabled(category)) return;
    g_console->trace(formatMessage(category, msg));
}

void debug(std::string_view category, std::string_view msg) {
    if (!g_initialized.load() || !isCategoryEnabled(category)) return;
    g_console->debug(formatMessage(category, msg));
}

void info(std::string_view category, std::string_view msg) {
    if (!g_initialized.load() || !isCategoryEnabled(category)) return;
    g_console->info(formatMessage(category, msg));
}

void warn(std::string_view category, std::string_view msg) {
    if (!g_initialized.load() || !isCategoryEnabled(category)) return;
    g_console->warn(formatMessage(category, msg));
}

void error(std::string_view category, std::string_view msg) {
    if (!g_initialized.load() || !isCategoryEnabled(category)) return;
    g_console->error(formatMessage(category, msg));
    // Force JSONL flush on every error so structured events (device_lost, etc.)
    // are guaranteed on disk before any abort or JVM shutdown path runs.
    if (g_jsonl) g_jsonl->flush();
}

[[noreturn]] void fatal(std::string_view category, std::string_view msg) {
    auto formatted = formatMessage(category, msg);

    if (g_initialized.load()) {
        g_console->critical(formatted);
        g_console->flush();

        // Dump crash ring to file
        if (g_crashRing) {
            auto entries = g_crashRing->last_formatted();
            std::ofstream crash(g_logDir + "/engine_crash_ring.txt");
            if (crash.is_open()) {
                for (const auto& entry : entries) {
                    crash << entry;
                }
            }
        }

        if (g_jsonl) {
            g_jsonl->info(formatJsonl(category, "fatal", {{"msg", std::string(msg)}}));
            g_jsonl->flush();
        }
    }

    std::abort();
}

void event(std::string_view category, std::string_view eventId,
           const std::unordered_map<std::string, std::string>& fields) {
    if (!g_initialized.load()) return;
    if (g_jsonl) {
        g_jsonl->info(formatJsonl(category, eventId, fields));
    }
    // Also log to console at info level for visibility
    g_console->info("[{}] event:{}", category, eventId);
}

void setCategoryEnabled(std::string_view category, bool enabled) {
    std::lock_guard lock(g_filterMutex);
    if (enabled) {
        g_disabledCategories.erase(std::string(category));
    } else {
        g_disabledCategories.insert(std::string(category));
    }
}

void setMinLevel(spdlog::level::level_enum level) {
    if (g_console) g_console->set_level(level);
    // Clamp the JSONL sink to info: structured events are always at info level
    // and must reach disk regardless of the console verbosity setting.
    // diagLevel=0 (warn) would otherwise block every log::event() call.
    if (g_jsonl) g_jsonl->set_level(std::min(level, spdlog::level::info));
}

void setMinLevel(int level) {
    spdlog::level::level_enum spLevel;
    switch (level) {
        case 0:  spLevel = spdlog::level::warn;  break;
        case 1:  spLevel = spdlog::level::info;  break;
        case 2:  spLevel = spdlog::level::debug; break;
        case 3:  spLevel = spdlog::level::trace; break;
        default: spLevel = spdlog::level::warn;  break;
    }
    setMinLevel(spLevel);
}

bool isInitialized() {
    return g_initialized.load();
}

} // namespace engine::log
