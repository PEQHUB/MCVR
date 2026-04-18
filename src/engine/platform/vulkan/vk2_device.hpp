#pragma once

// Ownership: EngineServices (1:1 lifetime, outlives all GPU resources).
// Thread: Created on main thread. Device handle is immutable after init.
// Dependencies: volk (for function loading). GLFW optional (standalone mode only).

#include "vk2_result.hpp"

#include <volk.h>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>
#include <vk_mem_alloc.h>

struct GLFWwindow;

namespace engine::vk2 {

// Device capability tier. Determined at init based on what the GPU actually supports.
enum class DeviceProfile {
    BootstrapPresent,   // Swapchain + present only (any GPU that can show a frame)
    BaselineRenderer,   // + sync2, timeline semaphores, descriptor indexing, BDA
    AdvancedRT          // + RT pipeline, acceleration structure
};

struct DeviceCaps {
    DeviceProfile profile = DeviceProfile::BootstrapPresent;

    // Feature flags — true only if actually enabled on the device
    bool rayTracingPipeline = false;
    bool accelerationStructure = false;
    bool timelineSemaphore = false;
    bool synchronization2 = false;
    bool bufferDeviceAddress = false;
    bool descriptorIndexing = false;
    bool dynamicRendering = false;
    bool opacityMicromap = false;
    bool shaderExecutionReorder = false;
    bool shaderClock = false;
    bool maintenance5 = false;
    bool deviceFault = false;
    bool debugUtils = false;
    bool memoryBudget = false;
    bool nvCheckpoints = false;   // VK_NV_device_diagnostic_checkpoints

    uint32_t maxRayRecursionDepth = 0;

    // Device identity
    std::string deviceName;
    uint32_t vendorId = 0;
    uint32_t driverVersion = 0;
    uint32_t vulkanVersion = 0;

    // Memory heaps
    struct HeapInfo {
        uint64_t size = 0;
        uint64_t budget = 0;     // 0 if VK_EXT_memory_budget not available
        uint64_t usage = 0;
        bool deviceLocal = false;
    };
    std::vector<HeapInfo> memoryHeaps;

    // Present modes supported by current surface
    std::vector<VkPresentModeKHR> presentModes;
};

struct QueueInfo {
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t familyIndex = UINT32_MAX;
};

// Wraps VkInstance + VkPhysicalDevice + VkDevice + VMA.
// Uses Result<T> for all fallible operations.
// GPU selection is scoring-based (not filtering). Feature enabling is query-first.
class DeviceService {
public:
    DeviceService() = default;
    ~DeviceService();

    DeviceService(const DeviceService&) = delete;
    DeviceService& operator=(const DeviceService&) = delete;

    // Callback for creating a VkSurface without depending on GLFW.
    using SurfaceFactory = std::function<VkResult(VkInstance, VkSurfaceKHR*)>;

    struct InitConfig {
        GLFWwindow* window = nullptr;          // GLFW window for surface (standalone mode)
        SurfaceFactory surfaceFactory;         // Custom surface creation (JNI mode)
        std::vector<const char*> instanceExtensions;  // Override GLFW instance extensions
        bool enableValidation = false;         // Vulkan validation layers
        bool enableDiagnostics = false;        // NV device diagnostics (Aftermath)
        std::string logsDir;                   // Directory for Aftermath crash dumps (requires enableDiagnostics)
        std::vector<const char*> extraInstanceExtensions;
        std::vector<const char*> extraDeviceExtensions;
        // Callback fired AFTER physical device is picked but BEFORE logical device is
        // created. Returns additional device extension names to enable. Used to query
        // NGX/DLSS for required extensions that depend on the chosen physical device.
        // The returned strings must remain valid for the duration of init().
        std::function<std::vector<const char*>(VkInstance, VkPhysicalDevice)>
            extraDeviceExtensionsCallback;

        // Streamline / DLSS-G integration.
        // When true, StreamlineContext::init() is called before vkCreateInstance
        // and Volk's proc-addr function is overridden with the SL interposer.
        bool enableFrameGen = false;
        // Absolute path to the directory containing sl.*.dll files.
        std::wstring streamlinePluginDir;
    };

    Result<void> init(const InitConfig& config);
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

    // Supported extensions on the chosen physical device (populated during pickPhysicalDevice)
    std::unordered_set<std::string> supportedExtensions_;

    Result<void> createInstance(const InitConfig& config);
    Result<void> pickPhysicalDevice();
    Result<void> createDevice(const InitConfig& config);
    Result<void> createVma();
    void findQueueFamilies();
    void queryDeviceCaps();

    int scoreDevice(VkPhysicalDevice pd) const;
    bool hasExtension(const char* name) const { return supportedExtensions_.count(name) > 0; }

    uint32_t mainQueueFamilyIndex_ = UINT32_MAX;
    uint32_t secondaryQueueFamilyIndex_ = UINT32_MAX;
    bool sameQueueFamily_ = false;
    bool slInterposerActive_ = false;  // true when SL interposer overrides Volk
};

} // namespace engine::vk2

// NV GPU checkpoint helper.
// Inserts a string marker into the command buffer so vkGetQueueCheckpointDataNV
// can report the last completed marker before a DEVICE_LOST event.
// No-op when the extension is unsupported or the crash-diag flag is off.
inline void nvInsertCheckpoint(VkCommandBuffer cmd, bool enabled, bool supported, const char* marker) {
    if (!enabled || !supported) return;
#if defined(VK_NV_device_diagnostic_checkpoints)
    vkCmdSetCheckpointNV(cmd, static_cast<const void*>(marker));
#endif
}
