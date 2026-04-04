#pragma once

// Ownership: EngineServices (1:1 lifetime, outlives all GPU resources).
// Thread: Created on main thread. Device handle is immutable after init.
// Dependencies: GLFW (for surface/window), volk (for function loading).

#include "vk2_result.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>

struct GLFWwindow;

namespace engine::vk2 {

struct DeviceCaps {
    bool rayTracingPipeline = false;
    bool accelerationStructure = false;
    bool timelineSemaphore = false;
    bool synchronization2 = false;
    bool bufferDeviceAddress = false;
    bool descriptorIndexing = false;
    bool opacityMicromap = false;
    bool shaderExecutionReorder = false;
    bool shaderClock = false;
    bool maintenance5 = false;
    bool deviceFault = false;
    uint32_t maxRayRecursionDepth = 0;
    std::string deviceName;
    uint32_t vendorId = 0;
};

struct QueueInfo {
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t familyIndex = UINT32_MAX;
};

// Wraps VkInstance + VkPhysicalDevice + VkDevice + VMA.
// Uses Result<T> for all fallible operations.
class DeviceService {
public:
    DeviceService() = default;
    ~DeviceService();

    DeviceService(const DeviceService&) = delete;
    DeviceService& operator=(const DeviceService&) = delete;

    struct InitConfig {
        GLFWwindow* window = nullptr;          // Required: existing GLFW window for surface
        bool enableValidation = false;         // Vulkan validation layers
        bool enableDiagnostics = false;        // NV device diagnostics
        std::vector<const char*> extraInstanceExtensions;
        std::vector<const char*> extraDeviceExtensions;
    };

    // Initialize the full Vulkan stack. Returns error on any failure.
    Result<void> init(const InitConfig& config);

    // Clean shutdown (waits for device idle first).
    void shutdown();

    bool isInitialized() const { return device_ != VK_NULL_HANDLE; }

    // --- Accessors (valid after init) ---

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    VkDevice device() const { return device_; }
    VmaAllocator vma() const { return vma_; }
    VkSurfaceKHR surface() const { return surface_; }

    const QueueInfo& mainQueue() const { return mainQueue_; }
    const QueueInfo& secondaryQueue() const { return secondaryQueue_; }
    const DeviceCaps& caps() const { return caps_; }

    void waitIdle();

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator vma_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;

    QueueInfo mainQueue_;
    QueueInfo secondaryQueue_;
    DeviceCaps caps_;

    Result<void> createInstance(const InitConfig& config);
    Result<void> pickPhysicalDevice();
    Result<void> createDevice(const InitConfig& config);
    Result<void> createVma();
    void findQueueFamilies();

    uint32_t mainQueueFamilyIndex_ = UINT32_MAX;
    uint32_t secondaryQueueFamilyIndex_ = UINT32_MAX;
    bool sameQueueFamily_ = false;
};

} // namespace engine::vk2
