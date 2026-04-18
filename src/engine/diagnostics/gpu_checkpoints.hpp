#pragma once
// NV_device_diagnostic_checkpoints helper.
// insertCheckpoint() inserts a named breadcrumb into the command stream.
// On DEVICE_LOST, query with vkGetQueueCheckpointDataNV to find the last completed marker.
// marker MUST be a string literal (static storage) — the driver may read it asynchronously.
#include <vulkan/vulkan.h>
#include "diag_flags.hpp"
#include "config/engine_config.hpp"

namespace engine {

inline void insertCheckpoint(VkCommandBuffer cmd, bool enabled, bool supported, const char* marker) {
    if (!enabled || !supported) return;
#ifdef VK_NV_device_diagnostic_checkpoints
    vkCmdSetCheckpointNV(cmd, static_cast<const void*>(marker));
#endif
}

} // namespace engine
