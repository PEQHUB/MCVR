#pragma once

// Boot trace — flush-per-line breadcrumbs for V2 init/shutdown/tick failures.
//
// Independent of the spdlog logger (which is itself initialized as one of the
// breadcrumb stages, so it cannot trace its own birth). Writes one line,
// flushes, closes the stream every call so a post-crash inspector can read
// the file even after std::abort, EXCEPTION_ACCESS_VIOLATION, or DLL detach.
//
// The "last" state (breadcrumb / bridge command / frame index) is kept in
// thread-safe storage so the crash classifier can dump it from any thread.

#include <cstdint>
#include <string>
#include <string_view>

namespace engine::boot_trace {

// Initialize the boot trace. Truncates {logDir}/v2_boot.log on entry so the
// file always reflects the current process. Safe to call before log::init().
// If logDir cannot be created, falls back to writing alongside the executable.
void init(const std::string& logDir);

// Emit one breadcrumb. Line format:
//   <YYYY-MM-DD HH:MM:SS.mmm> [V2-BOOT] <stage> <status> [- <detail>]
// Always opens-writes-flushes-closes the file, so the line is on disk before
// the next instruction runs. Updates the "last breadcrumb" cache.
void breadcrumb(std::string_view stage,
                std::string_view status,
                std::string_view detail = "");

// Record the most recent bridge command (type name + monotonic sequence).
// Cheap: updates in-memory cache only — does not write to the boot log.
void recordBridgeCommand(std::string_view cmdType, uint64_t seq);

// Record the most recent frame index (bumped at the top of EngineApp::tick).
// Cheap: in-memory only.
void recordFrame(uint64_t frameIndex);

// Read the last breadcrumb / command / frame for the crash classifier.
std::string lastBreadcrumb();
std::string lastBridgeCommand();
uint64_t    lastFrameIndex();

// Write a v2_crash_<timestamp>.log with the full classifier dump:
//   reason, last breadcrumb, last bridge command, last frame, vkResult,
//   extra detail. Caller passes whatever device-fault / checkpoint data
//   it managed to gather from VK_EXT_device_fault / VK_NV_device_diagnostic_checkpoints.
// Always flushes. Returns the absolute path written, or empty on failure.
std::string writeCrashDump(std::string_view reason,
                           int vkResult = 0,
                           std::string_view extraDetail = "");

// Per-tick heartbeat: appends a line to {logDir}/v2_tick.log.
//
// Written to a SEPARATE file from v2_boot.log so freezeBootLog() doesn't
// silence it and post-boot ticks are always captured. Caller should rate-limit
// (e.g. every N frames) — every call opens-writes-flushes-closes so it is not
// cheap enough to run every frame at 240 Hz, but fine at 2 Hz.
//
// Format: <timestamp> [V2-TICK] frame=<frameIndex> <extra...>
void writeTickHeartbeat(uint64_t frameIndex, std::string_view extra = "");

// After the engine has fully booted, the boot log can be frozen so steady-state
// breadcrumbs (e.g. shutdown stages) are still cached but no longer hit disk.
// The crash classifier still reads lastBreadcrumb()/lastFrameIndex().
void freezeBootLog();

// Re-arm the boot log (writes resume). Call from shutdown so the teardown
// stages are persisted.
void unfreezeBootLog();

} // namespace engine::boot_trace
