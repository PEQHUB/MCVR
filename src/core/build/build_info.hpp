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
inline constexpr const char* kRepoCommit = "f572477f0cab0d74c149b961a6dc6b93fb5bc76f";
inline constexpr const char* kBranch = "fix/texture-loader-v4-critical-correctness";
inline constexpr bool kDirty = 1;
inline constexpr const char* kBuildTimestamp = "2026-06-10T12:00:00Z";
inline constexpr const char* kDllSha256 = "unknown";

std::string summary();

} // namespace build_info
