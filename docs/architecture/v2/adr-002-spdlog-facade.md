# ADR-002: spdlog Behind engine::log Facade

**Status**: Accepted
**Date**: 2026-04-03

## Context

Current logging is ad-hoc: raw `std::cout`, `std::cerr`, `fprintf`, and occasional
`OutputDebugStringA` scattered across 91 .cpp files. No structured output, no categories,
no severity levels, no rotation. Crash investigation relies on crash_ring.txt (custom)
and Aftermath dumps — there is no general-purpose log file.

Buffer creation failures in buffer.cpp log and continue, leaving partially valid objects
that crash later in unrelated code paths.

## Decision

Add spdlog as extern dependency (`extern/spdlog/`, header-only mode).

Wrap behind `engine::log` facade in `src/engine/diagnostics/log.hpp`:

```cpp
namespace engine::log {
    void init(const LogConfig& config);
    void shutdown();
    
    void trace(std::string_view category, std::string_view msg);
    void debug(std::string_view category, std::string_view msg);
    void info(std::string_view category, std::string_view msg);
    void warn(std::string_view category, std::string_view msg);
    void error(std::string_view category, std::string_view msg);
    void fatal(std::string_view category, std::string_view msg);
    
    // Structured event (JSONL sink)
    void event(std::string_view category, std::string_view eventId, 
               const std::unordered_map<std::string, std::string>& fields);
}
```

Three sinks:
1. **Console** — human-readable, color-coded, stderr
2. **JSONL file** — `Radiance/logs/engine.jsonl`, rotating (5 MB x 3 files)
3. **Crash session** — last 256 entries in memory ring, flushed on fatal/crash

Categories: `vulkan`, `blas`, `tlas`, `pipeline`, `frame`, `bridge`, `config`, `shader`.

Fatal error model: `engine::log::fatal()` captures crash context, flushes all sinks,
writes crash_ring, then calls `std::abort()`. No partial-object survival.

## Alternatives Considered

**Custom logger**: Full control but reinventing rotation, thread safety, formatting.
Not worth the maintenance cost for a logging backend.

**No facade (spdlog directly)**: Locks the entire codebase to spdlog's API. Facade
costs ~50 lines and allows swap later.

## Consequences

### Positive
- Structured, searchable logs for crash investigation
- Category filtering reduces noise
- Fatal errors can't leave zombie objects
- JSONL enables automated log analysis

### Negative
- New extern dependency (~400KB headers)
- Facade overhead is negligible but non-zero

### Migration Impact
- New code uses `engine::log::*` exclusively
- Legacy `std::cout`/`std::cerr` stays until each file is touched for other reasons
- Ban enforced: no new raw console logging in engine/ code
