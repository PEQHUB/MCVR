#pragma once

// Dry-run Vulkan capability probe.
//
// Phase 1 of resilient-booting-lighthouse.md: never mutate GLFW/window state
// until Vulkan is proven healthy. This probe creates a temporary VkInstance,
// enumerates physical devices, checks queue families, and verifies that the
// extensions V2 strictly requires (swapchain, RT pipeline, accel-struct,
// buffer-device-address) are present on the chosen physical device. Then it
// tears everything back down. No window, no GLFW, no long-lived handles.
//
// Caller (MinecraftClientMixins.initRenderer redirect) must run this BEFORE
// the GLFW_NO_API + UnsafeManager.allocateInstance(WindowFramebuffer.class)
// sequence. If the probe fails, the renderer falls back to legacy without
// having corrupted the window or framebuffer state.

#include <cstdint>
#include <string>
#include <vector>

namespace engine::vk2 {

struct ProbeResult {
    bool ok = false;                          // overall: all hard requirements met
    std::string reason;                       // single-line failure summary

    // Identity (filled when an instance + at least one device exist)
    std::string gpuName;
    uint32_t vendorId = 0;
    uint32_t driverVersion = 0;
    uint32_t deviceApiVersion = 0;            // physical-device's apiVersion
    uint32_t loaderApiVersion = 0;            // vkEnumerateInstanceVersion result

    // Capability checks on the chosen physical device
    bool hasSwapchainExt = false;             // VK_KHR_swapchain
    bool hasRayTracingPipeline = false;       // VK_KHR_ray_tracing_pipeline
    bool hasAccelerationStructure = false;    // VK_KHR_acceleration_structure
    bool hasBufferDeviceAddress = false;      // VK_KHR_buffer_device_address (or core in 1.2+)
    bool hasGraphicsQueue = false;            // any queue family with VK_QUEUE_GRAPHICS_BIT

    // Core 1.2/1.3 features the baseline renderer requires (see vk2_device.cpp
    // `canBaseline` gate). Probed with VkPhysicalDeviceFeatures2 chain so the
    // probe catches drivers that advertise the extensions but don't actually
    // support the feature bits — failure mode we've seen before on older
    // Mesa/AMD drivers.
    bool hasSynchronization2 = false;
    bool hasDynamicRendering = false;
    bool hasDescriptorIndexing = false;
    bool hasTimelineSemaphore = false;

    // What's missing — populated only if !ok. Joined into reason but kept
    // separate so the diagnostic file can list them line-by-line.
    std::vector<std::string> missingExtensions;
    std::vector<std::string> missingFeatures;

    int physicalDeviceCount = 0;
};

// Run the probe. Always destroys all created Vulkan handles before returning.
//
// `enableValidation` controls whether the validation layer is requested for
// the temporary instance — enabling it ensures the user sees validation
// errors during the probe even if their normal config has it off.
ProbeResult probeVulkan(bool enableValidation);

// Format the result as a human-readable diagnostic written to
// {logDir}/v2_cannot_start.txt. Returns the absolute path written, or empty
// on I/O failure. Always called on probe failure; safe to call on success
// (writes a "probe ok" file useful for CI).
std::string writeProbeReport(const std::string& logDir, const ProbeResult& result);

} // namespace engine::vk2
