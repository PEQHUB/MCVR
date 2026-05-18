#pragma once

#include <filesystem>
#include <string>

// Lightweight Aftermath GPU crash dump integration.
// Call initAftermath() before Vulkan device creation.
// On DEVICE_LOST, call waitForCrashDump() to let Aftermath collect data.
// The crash dump is saved as .nv-gpudmp in the logs directory.

namespace AftermathIntegration {

// Initialize Aftermath crash dump collection for Vulkan.
// Must be called before vkCreateDevice.
// Returns true if Aftermath SDK loaded successfully.
bool init(const std::filesystem::path& logsDir);

// Called on DEVICE_LOST. Polls Aftermath until crash dump is collected or timeout.
// Returns path to .nv-gpudmp file, or empty if none generated.
std::string waitForCrashDump(int timeoutMs = 5000);

// Shutdown Aftermath crash dump collection.
void shutdown();

} // namespace AftermathIntegration
