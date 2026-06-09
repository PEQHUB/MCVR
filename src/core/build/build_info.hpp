#pragma once
// Build identity for MCVR native core.
//
// Generated at build time by CMake configure_file from build_info.hpp.in.
// Every runtime log and DebugBridge buildInfo command must print these fields.

#include <cstdint>
#include <string>

namespace build_info {

inline constexpr int kTextureLoaderAbiVersion = 4;
inline constexpr int kCacheSchemaVersion = 4;

// Filled by CMake configure step
inline constexpr const char* kRepoCommit = "c4a98a3ee80ca7c90ffd47448a0fd46622618880";
inline constexpr const char* kBranch = "main";
inline constexpr bool kDirty = 1;
inline constexpr const char* kBuildTimestamp = "2026-06-09T18:04:57Z";
inline constexpr const char* kDllSha256 = "unknown";

std::string summary();

} // namespace build_info
