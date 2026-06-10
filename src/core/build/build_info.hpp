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
inline constexpr const char* kRepoCommit = "8d8ee191edb0755b1d842e382f9e51625a72f084";
inline constexpr const char* kBranch = "fix/texture-loader-v4-critical-correctness";
inline constexpr bool kDirty = 1;
inline constexpr const char* kBuildTimestamp = "2026-06-10T08:16:47Z";
inline constexpr const char* kDllSha256 = "unknown";

std::string summary();

} // namespace build_info
