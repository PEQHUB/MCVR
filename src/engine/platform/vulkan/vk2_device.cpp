// VMA_IMPLEMENTATION is provided by core/vulkan/vma.cpp at link time.
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#include "vk2_device.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstring>
#include <set>

namespace engine::vk2 {

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /*userData*/) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        log::error("vulkan", data->pMessage);
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        log::warn("vulkan", data->pMessage);
    }
    return VK_FALSE;
}

DeviceService::~DeviceService() {
    shutdown();
}

Result<void> DeviceService::init(const InitConfig& config) {
    if (volkInitialize() != VK_SUCCESS) {
        return makeError(VK_ERROR_INITIALIZATION_FAILED, "volkInitialize failed");
    }

    if (!config.window && !config.surfaceFactory) {
        return makeError(VK_ERROR_INITIALIZATION_FAILED, "No window or surface factory provided");
    }

    auto r = createInstance(config);
    if (!r) return r;

    volkLoadInstance(instance_);

    // Create surface
    VkResult sr;
    if (config.surfaceFactory) {
        sr = config.surfaceFactory(instance_, &surface_);
    } else {
        sr = glfwCreateWindowSurface(instance_, config.window, nullptr, &surface_);
    }
    if (sr != VK_SUCCESS) {
        return makeError(sr, "Surface creation failed");
    }

    r = pickPhysicalDevice();
    if (!r) return r;

    findQueueFamilies();

    r = createDevice(config);
    if (!r) return r;

    log::info("vulkan", "volkLoadDevice...");
    volkLoadDevice(device_);

    log::info("vulkan", "Getting queues (main=" + std::to_string(mainQueueFamilyIndex_) +
              ", sec=" + std::to_string(secondaryQueueFamilyIndex_) +
              ", same=" + std::to_string(sameQueueFamily_) + ")");

    // Get queues
    vkGetDeviceQueue(device_, mainQueueFamilyIndex_, 0, &mainQueue_.queue);
    mainQueue_.familyIndex = mainQueueFamilyIndex_;
    log::info("vulkan", "Main queue acquired");

    uint32_t secIdx = sameQueueFamily_ ? 1 : 0;
    vkGetDeviceQueue(device_, secondaryQueueFamilyIndex_, secIdx, &secondaryQueue_.queue);
    secondaryQueue_.familyIndex = secondaryQueueFamilyIndex_;
    log::info("vulkan", "Secondary queue acquired");

    r = createVma();
    if (!r) return r;
    log::info("vulkan", "VMA created");

    queryDeviceCaps();

    log::info("vulkan", "DeviceService initialized: " + caps_.deviceName +
              " [" + std::string(
                  caps_.profile == DeviceProfile::AdvancedRT ? "AdvancedRT" :
                  caps_.profile == DeviceProfile::BaselineRenderer ? "BaselineRenderer" :
                  "BootstrapPresent") + "]");
    return {};
}

void DeviceService::shutdown() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }
    if (vma_ != VK_NULL_HANDLE) {
        vmaDestroyAllocator(vma_);
        vma_ = VK_NULL_HANDLE;
    }
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (surface_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }
    if (debugMessenger_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
        vkDestroyDebugUtilsMessengerEXT(instance_, debugMessenger_, nullptr);
        debugMessenger_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

void DeviceService::waitIdle() {
    if (device_ != VK_NULL_HANDLE) vkDeviceWaitIdle(device_);
}

// --- Instance ---

Result<void> DeviceService::createInstance(const InitConfig& config) {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "RadSER";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 2, 0);
    appInfo.pEngineName = "RadSER Engine V2";
    appInfo.engineVersion = VK_MAKE_VERSION(2, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    // Gather extensions
    std::vector<const char*> extensions;
    if (!config.instanceExtensions.empty()) {
        extensions = config.instanceExtensions;
    } else {
        uint32_t glfwExtCount = 0;
        const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
        extensions.assign(glfwExts, glfwExts + glfwExtCount);
    }
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    if (config.enableValidation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    for (auto ext : config.extraInstanceExtensions) {
        extensions.push_back(ext);
    }

    // Deduplicate
    std::set<std::string> unique(extensions.begin(), extensions.end());
    std::vector<std::string> storage(unique.begin(), unique.end());
    std::vector<const char*> uniqueExts;
    for (auto& s : storage) uniqueExts.push_back(s.c_str());

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(uniqueExts.size());
    createInfo.ppEnabledExtensionNames = uniqueExts.data();

    const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    if (config.enableValidation) {
        createInfo.enabledLayerCount = 1;
        createInfo.ppEnabledLayerNames = &validationLayer;
    }

    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        return makeError(result, "vkCreateInstance failed");
    }

    if (config.enableValidation) {
        VkDebugUtilsMessengerCreateInfoEXT dbgInfo{};
        dbgInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dbgInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
        dbgInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dbgInfo.pfnUserCallback = debugCallback;
        vkCreateDebugUtilsMessengerEXT(instance_, &dbgInfo, nullptr, &debugMessenger_);
        caps_.debugUtils = true;
    }

    return {};
}

// --- Physical device selection (scoring, not filtering) ---

int DeviceService::scoreDevice(VkPhysicalDevice pd) const {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);

    // Must support at least one queue family with graphics + present
    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qFamilies(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, qFamilies.data());

    bool canPresent = false;
    for (uint32_t i = 0; i < qCount; ++i) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface_, &present);
        if (present && (qFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
            canPresent = true;
    }
    if (!canPresent) return -1;

    int score = 0;

    // Prefer discrete GPU
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 10000;
    else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 1000;

    // Prefer NVIDIA
    if (props.vendorID == 0x10DE) score += 500;

    // Bonus for RT pipeline support
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, exts.data());
    for (const auto& ext : exts) {
        if (strcmp(ext.extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) == 0)
            score += 5000;
    }

    // Bonus for VRAM (per 256MB device-local)
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(pd, &memProps);
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            score += static_cast<int>(memProps.memoryHeaps[i].size / (256 * 1024 * 1024));
    }

    return score;
}

Result<void> DeviceService::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) {
        return makeError(VK_ERROR_INITIALIZATION_FAILED, "No Vulkan physical devices found");
    }

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    int bestScore = -1;
    VkPhysicalDevice bestDevice = VK_NULL_HANDLE;

    for (auto pd : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);

        int score = scoreDevice(pd);
        log::debug("vulkan", std::string("GPU candidate: ") + props.deviceName +
                   " score=" + std::to_string(score));

        if (score > bestScore) {
            bestScore = score;
            bestDevice = pd;
        }
    }

    if (bestScore < 0) {
        return makeError(VK_ERROR_INITIALIZATION_FAILED,
                         "No GPU can present to the target surface");
    }

    physicalDevice_ = bestDevice;

    // Cache supported extensions for this device
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, exts.data());
    supportedExtensions_.clear();
    for (const auto& ext : exts) {
        supportedExtensions_.insert(ext.extensionName);
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice_, &props);
    caps_.deviceName = props.deviceName;
    caps_.vendorId = props.vendorID;
    caps_.driverVersion = props.driverVersion;
    caps_.vulkanVersion = props.apiVersion;

    log::info("vulkan", "Selected GPU: " + caps_.deviceName +
              " (Vulkan " + std::to_string(VK_VERSION_MAJOR(caps_.vulkanVersion)) + "." +
              std::to_string(VK_VERSION_MINOR(caps_.vulkanVersion)) + ")");

    return {};
}

// --- Queue families ---

void DeviceService::findQueueFamilies() {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &count, families.data());

    for (uint32_t i = 0; i < count; ++i) {
        auto flags = families[i].queueFlags;
        bool hasGraphics = (flags & VK_QUEUE_GRAPHICS_BIT) != 0;
        bool hasCompute = (flags & VK_QUEUE_COMPUTE_BIT) != 0;
        bool hasTransfer = (flags & VK_QUEUE_TRANSFER_BIT) != 0;

        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice_, i, surface_, &present);

        if (hasGraphics && hasCompute && hasTransfer && present) {
            mainQueueFamilyIndex_ = i;
            if (families[i].queueCount >= 2) {
                secondaryQueueFamilyIndex_ = i;
                sameQueueFamily_ = true;
            }
            break;
        }
    }

    if (secondaryQueueFamilyIndex_ == UINT32_MAX) {
        for (uint32_t i = 0; i < count; ++i) {
            if (i == mainQueueFamilyIndex_) continue;
            auto flags = families[i].queueFlags;
            if ((flags & VK_QUEUE_COMPUTE_BIT) && (flags & VK_QUEUE_TRANSFER_BIT)) {
                secondaryQueueFamilyIndex_ = i;
                sameQueueFamily_ = false;
                break;
            }
        }
    }

    if (secondaryQueueFamilyIndex_ == UINT32_MAX) {
        secondaryQueueFamilyIndex_ = mainQueueFamilyIndex_;
        sameQueueFamily_ = true;
    }
}

// --- Device creation (capability-driven) ---

Result<void> DeviceService::createDevice(const InitConfig& config) {
    // --- 1. Query supported features ---
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR supportedRT{};
    supportedRT.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR supportedAS{};
    supportedAS.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    supportedAS.pNext = &supportedRT;

    VkPhysicalDeviceVulkan13Features supported13{};
    supported13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    supported13.pNext = &supportedAS;

    VkPhysicalDeviceVulkan12Features supported12{};
    supported12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    supported12.pNext = &supported13;

    VkPhysicalDeviceFeatures2 supportedFeatures{};
    supportedFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supportedFeatures.pNext = &supported12;

    vkGetPhysicalDeviceFeatures2(physicalDevice_, &supportedFeatures);

    // --- 2. Determine achievable profile ---
    bool canRT = hasExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) &&
                 hasExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
                 hasExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) &&
                 hasExtension(VK_KHR_SPIRV_1_4_EXTENSION_NAME) &&
                 supportedRT.rayTracingPipeline == VK_TRUE &&
                 supportedAS.accelerationStructure == VK_TRUE;

    bool canBaseline = supported13.synchronization2 == VK_TRUE &&
                       supported12.timelineSemaphore == VK_TRUE &&
                       supported12.bufferDeviceAddress == VK_TRUE &&
                       supported12.descriptorIndexing == VK_TRUE &&
                       supported13.dynamicRendering == VK_TRUE;

    DeviceProfile profile;
    if (canRT && canBaseline)
        profile = DeviceProfile::AdvancedRT;
    else if (canBaseline)
        profile = DeviceProfile::BaselineRenderer;
    else
        profile = DeviceProfile::BootstrapPresent;

    caps_.profile = profile;

    log::info("vulkan", "Device profile: " + std::string(
        profile == DeviceProfile::AdvancedRT ? "AdvancedRT" :
        profile == DeviceProfile::BaselineRenderer ? "BaselineRenderer" :
        "BootstrapPresent"));

    // --- 3. Build extension list based on profile ---
    std::vector<const char*> extensions;
    extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    if (profile >= DeviceProfile::BaselineRenderer) {
        if (hasExtension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME))
            extensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
        if (hasExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME))
            extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        if (hasExtension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME))
            extensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    }

    if (profile >= DeviceProfile::AdvancedRT) {
        extensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        extensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
    }

    // Optional extensions (any profile)
    if (hasExtension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
        extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        caps_.memoryBudget = true;
    }
    if (hasExtension(VK_EXT_DEVICE_FAULT_EXTENSION_NAME)) {
        extensions.push_back(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
        caps_.deviceFault = true;
    }

    for (auto ext : config.extraDeviceExtensions) {
        if (hasExtension(ext)) extensions.push_back(ext);
    }

    // Deduplicate
    std::set<std::string> uniqueExts(extensions.begin(), extensions.end());
    std::vector<std::string> extStorage(uniqueExts.begin(), uniqueExts.end());
    std::vector<const char*> finalExts;
    for (auto& s : extStorage) finalExts.push_back(s.c_str());

    // Log enabled extensions
    for (const auto& ext : extStorage) {
        log::debug("vulkan", "  ext: " + ext);
    }

    // --- 4. Build feature enable chain (only enable what was queried as supported) ---
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR enableRT{};
    enableRT.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR enableAS{};
    enableAS.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;

    VkPhysicalDeviceVulkan13Features enable13{};
    enable13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

    VkPhysicalDeviceVulkan12Features enable12{};
    enable12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceFeatures2 enableFeatures{};
    enableFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    // Wire pNext chain based on profile
    void** pNextTail = &enableFeatures.pNext;

    // Vulkan 1.2 features (always in chain for timeline semaphore query)
    *pNextTail = &enable12;
    pNextTail = &enable12.pNext;

    // Vulkan 1.3 features
    *pNextTail = &enable13;
    pNextTail = &enable13.pNext;

    if (profile >= DeviceProfile::AdvancedRT) {
        *pNextTail = &enableAS;
        pNextTail = &enableAS.pNext;
        *pNextTail = &enableRT;
        pNextTail = &enableRT.pNext;
    }

    // Enable features based on profile + what's actually supported
    if (profile >= DeviceProfile::BaselineRenderer) {
        enable12.timelineSemaphore = supported12.timelineSemaphore;
        enable12.bufferDeviceAddress = supported12.bufferDeviceAddress;
        enable12.descriptorIndexing = supported12.descriptorIndexing;
        enable12.runtimeDescriptorArray = supported12.runtimeDescriptorArray;
        enable12.shaderSampledImageArrayNonUniformIndexing = supported12.shaderSampledImageArrayNonUniformIndexing;
        enable12.descriptorBindingPartiallyBound = supported12.descriptorBindingPartiallyBound;
        enable12.descriptorBindingUpdateUnusedWhilePending = supported12.descriptorBindingUpdateUnusedWhilePending;
        enable12.scalarBlockLayout = supported12.scalarBlockLayout;
        enable12.shaderBufferInt64Atomics = supported12.shaderBufferInt64Atomics;
        enable13.synchronization2 = supported13.synchronization2;
        enable13.dynamicRendering = supported13.dynamicRendering;

        caps_.timelineSemaphore = supported12.timelineSemaphore;
        caps_.bufferDeviceAddress = supported12.bufferDeviceAddress;
        caps_.descriptorIndexing = supported12.descriptorIndexing;
        caps_.synchronization2 = supported13.synchronization2;
        caps_.dynamicRendering = supported13.dynamicRendering;
    } else {
        // BootstrapPresent: enable minimal 1.3 features for dynamic rendering if available
        if (supported13.dynamicRendering) {
            enable13.dynamicRendering = VK_TRUE;
            caps_.dynamicRendering = true;
        }
        if (supported13.synchronization2) {
            enable13.synchronization2 = VK_TRUE;
            caps_.synchronization2 = true;
        }
        if (supported12.timelineSemaphore) {
            enable12.timelineSemaphore = VK_TRUE;
            caps_.timelineSemaphore = true;
        }
    }

    if (profile >= DeviceProfile::AdvancedRT) {
        enableRT.rayTracingPipeline = VK_TRUE;
        enableAS.accelerationStructure = VK_TRUE;
        caps_.rayTracingPipeline = true;
        caps_.accelerationStructure = true;
    }

    // Optional core features (any profile)
    enableFeatures.features.shaderInt64 = supportedFeatures.features.shaderInt64;
    enableFeatures.features.shaderFloat64 = supportedFeatures.features.shaderFloat64;
    enableFeatures.features.shaderInt16 = supportedFeatures.features.shaderInt16;
    enableFeatures.features.fillModeNonSolid = supportedFeatures.features.fillModeNonSolid;

    // --- 5. Queue creation ---
    // Priorities must outlive queueInfos (pQueuePriorities is a pointer)
    float queuePriorities[] = {1.0f, 0.0f};

    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    if (sameQueueFamily_) {
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = mainQueueFamilyIndex_;
        qi.queueCount = 2;
        qi.pQueuePriorities = queuePriorities;
        queueInfos.push_back(qi);
    } else {
        VkDeviceQueueCreateInfo qi0{};
        qi0.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi0.queueFamilyIndex = mainQueueFamilyIndex_;
        qi0.queueCount = 1;
        qi0.pQueuePriorities = &queuePriorities[0];
        queueInfos.push_back(qi0);

        VkDeviceQueueCreateInfo qi1{};
        qi1.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi1.queueFamilyIndex = secondaryQueueFamilyIndex_;
        qi1.queueCount = 1;
        qi1.pQueuePriorities = &queuePriorities[1];
        queueInfos.push_back(qi1);
    }

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.pNext = &enableFeatures;
    deviceInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    deviceInfo.pQueueCreateInfos = queueInfos.data();
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(finalExts.size());
    deviceInfo.ppEnabledExtensionNames = finalExts.data();

    log::info("vulkan", "Calling vkCreateDevice (queues=" + std::to_string(queueInfos.size()) +
              ", exts=" + std::to_string(finalExts.size()) +
              ", vkCreateDevice ptr=" + std::to_string(reinterpret_cast<uintptr_t>(vkCreateDevice)) + ")");

    if (!vkCreateDevice) {
        return makeError(VK_ERROR_INITIALIZATION_FAILED, "vkCreateDevice function pointer is null");
    }

    VkResult result = vkCreateDevice(physicalDevice_, &deviceInfo, nullptr, &device_);
    log::info("vulkan", "vkCreateDevice returned: " + std::to_string(result));
    if (result != VK_SUCCESS) {
        return makeError(result, "vkCreateDevice failed (profile=" + std::string(
            profile == DeviceProfile::AdvancedRT ? "AdvancedRT" :
            profile == DeviceProfile::BaselineRenderer ? "BaselineRenderer" :
            "BootstrapPresent") + ")");
    }

    // Query RT properties if RT is enabled
    if (caps_.rayTracingPipeline) {
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
        rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &rtProps;
        vkGetPhysicalDeviceProperties2(physicalDevice_, &props2);
        caps_.maxRayRecursionDepth = rtProps.maxRayRecursionDepth;
    }

    return {};
}

// --- VMA ---

Result<void> DeviceService::createVma() {
    // VMA was compiled with STATIC=0 DYNAMIC=0 (see all_extern.hpp).
    // Must provide ALL function pointers explicitly from volk globals.
    VmaVulkanFunctions vulkanFunctions{};
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    vulkanFunctions.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
    vulkanFunctions.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
    vulkanFunctions.vkAllocateMemory = vkAllocateMemory;
    vulkanFunctions.vkFreeMemory = vkFreeMemory;
    vulkanFunctions.vkMapMemory = vkMapMemory;
    vulkanFunctions.vkUnmapMemory = vkUnmapMemory;
    vulkanFunctions.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
    vulkanFunctions.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges;
    vulkanFunctions.vkBindBufferMemory = vkBindBufferMemory;
    vulkanFunctions.vkBindImageMemory = vkBindImageMemory;
    vulkanFunctions.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
    vulkanFunctions.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
    vulkanFunctions.vkCreateBuffer = vkCreateBuffer;
    vulkanFunctions.vkDestroyBuffer = vkDestroyBuffer;
    vulkanFunctions.vkCreateImage = vkCreateImage;
    vulkanFunctions.vkDestroyImage = vkDestroyImage;
    vulkanFunctions.vkCmdCopyBuffer = vkCmdCopyBuffer;
    vulkanFunctions.vkGetBufferMemoryRequirements2KHR = vkGetBufferMemoryRequirements2;
    vulkanFunctions.vkGetImageMemoryRequirements2KHR = vkGetImageMemoryRequirements2;
    vulkanFunctions.vkBindBufferMemory2KHR = vkBindBufferMemory2;
    vulkanFunctions.vkBindImageMemory2KHR = vkBindImageMemory2;
    vulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2;
    vulkanFunctions.vkGetDeviceBufferMemoryRequirements = vkGetDeviceBufferMemoryRequirements;
    vulkanFunctions.vkGetDeviceImageMemoryRequirements = vkGetDeviceImageMemoryRequirements;

    VmaAllocatorCreateInfo createInfo{};
    createInfo.physicalDevice = physicalDevice_;
    createInfo.device = device_;
    createInfo.instance = instance_;
    createInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    createInfo.pVulkanFunctions = &vulkanFunctions;

    if (caps_.bufferDeviceAddress)
        createInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    if (caps_.memoryBudget)
        createInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;

    VkResult result = vmaCreateAllocator(&createInfo, &vma_);
    if (result != VK_SUCCESS) {
        return makeError(result, "vmaCreateAllocator failed");
    }
    return {};
}

// --- Post-init capability query ---

void DeviceService::queryDeviceCaps() {
    // Memory heaps
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);
    caps_.memoryHeaps.resize(memProps.memoryHeapCount);
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        caps_.memoryHeaps[i].size = memProps.memoryHeaps[i].size;
        caps_.memoryHeaps[i].deviceLocal =
            (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
    }

    // Memory budget (if extension available and VMA initialized)
    if (caps_.memoryBudget && vma_ != VK_NULL_HANDLE) {
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
        vmaGetHeapBudgets(vma_, budgets);
        for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
            caps_.memoryHeaps[i].budget = budgets[i].budget;
            caps_.memoryHeaps[i].usage = budgets[i].usage;
        }
    }

    // Present modes
    uint32_t pmCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_, &pmCount, nullptr);
    caps_.presentModes.resize(pmCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_, &pmCount,
                                               caps_.presentModes.data());

    // Log summary
    uint64_t totalVram = 0;
    for (const auto& heap : caps_.memoryHeaps) {
        if (heap.deviceLocal) totalVram += heap.size;
    }
    log::info("vulkan", "VRAM: " + std::to_string(totalVram / (1024 * 1024)) + " MB, " +
              std::to_string(caps_.presentModes.size()) + " present modes");
}

} // namespace engine::vk2
