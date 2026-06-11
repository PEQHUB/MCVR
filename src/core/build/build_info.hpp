#pragma once
// Build identity for MCVR native core.
//
// Generated at build time by CMake configure_file from build_info.hpp.in.
// Every runtime log and DebugBridge buildInfo command must print these fields.

#include <cstdint>
#include <string>

namespace build_info {

inline constexpr int kTextureLoaderAbiVersion = 5;
inline constexpr int kCacheSchemaVersion = 5;

// Filled by CMake configure step
inline constexpr const char* kRepoCommit = "86b482c03320fb2a93b0637f00ff2bed4817cb33";
inline constexpr const char* kBranch = "fix/texture-loader-v4-critical-correctness";
inline constexpr bool kDirty = 1;
inline constexpr const char* kBuildTimestamp = "2026-06-11T03:38:19Z";
inline constexpr const char* kDllSha256 = "unknown";

std::string summary();

} // namespace build_info
