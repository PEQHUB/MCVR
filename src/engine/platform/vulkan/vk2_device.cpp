#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#include "vk2_device.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <set>
#include <cstring>

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

    // Create surface from GLFW window
    if (!config.window) {
        return makeError(VK_ERROR_INITIALIZATION_FAILED, "No GLFW window provided");
    }

    auto r = createInstance(config);
    if (!r) return r;

    volkLoadInstance(instance_);

    // Create surface
    VkResult sr = glfwCreateWindowSurface(instance_, config.window, nullptr, &surface_);
    if (sr != VK_SUCCESS) {
        return makeError(sr, "glfwCreateWindowSurface failed");
    }

    r = pickPhysicalDevice();
    if (!r) return r;

    findQueueFamilies();

    r = createDevice(config);
    if (!r) return r;

    volkLoadDevice(device_);

    // Get queues
    vkGetDeviceQueue(device_, mainQueueFamilyIndex_, 0, &mainQueue_.queue);
    mainQueue_.familyIndex = mainQueueFamilyIndex_;

    uint32_t secIdx = sameQueueFamily_ ? 1 : 0;
    vkGetDeviceQueue(device_, secondaryQueueFamilyIndex_, secIdx, &secondaryQueue_.queue);
    secondaryQueue_.familyIndex = secondaryQueueFamilyIndex_;

    r = createVma();
    if (!r) return r;

    log::info("vulkan", "DeviceService initialized: " + caps_.deviceName);
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

Result<void> DeviceService::createInstance(const InitConfig& config) {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "RadSER";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 2, 0);
    appInfo.pEngineName = "RadSER Engine V2";
    appInfo.engineVersion = VK_MAKE_VERSION(2, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    // Gather extensions
    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    if (config.enableValidation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    for (auto ext : config.extraInstanceExtensions) {
        extensions.push_back(ext);
    }

    // Deduplicate
    std::set<std::string> unique(extensions.begin(), extensions.end());
    std::vector<const char*> uniqueExts;
    std::vector<std::string> storage;
    storage.assign(unique.begin(), unique.end());
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

    // Set up debug messenger
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
    }

    return {};
}

Result<void> DeviceService::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) {
        return makeError(VK_ERROR_INITIALIZATION_FAILED, "No Vulkan physical devices found");
    }

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    // Pick first discrete GPU with RT support
    for (auto pd : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);

        if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;

        // Check for RT pipeline extension
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, exts.data());

        bool hasRT = false;
        for (const auto& ext : exts) {
            if (strcmp(ext.extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) == 0) {
                hasRT = true;
                break;
            }
        }

        if (hasRT) {
            physicalDevice_ = pd;
            caps_.deviceName = props.deviceName;
            caps_.vendorId = props.vendorID;
            log::info("vulkan", "Selected GPU: " + caps_.deviceName);
            return {};
        }
    }

    return makeError(VK_ERROR_INITIALIZATION_FAILED, "No discrete GPU with RT support found");
}

void DeviceService::findQueueFamilies() {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &count, families.data());

    // Find a family with graphics + compute + transfer + present
    for (uint32_t i = 0; i < count; ++i) {
        auto flags = families[i].queueFlags;
        bool hasGraphics = (flags & VK_QUEUE_GRAPHICS_BIT) != 0;
        bool hasCompute = (flags & VK_QUEUE_COMPUTE_BIT) != 0;
        bool hasTransfer = (flags & VK_QUEUE_TRANSFER_BIT) != 0;

        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice_, i, surface_, &present);

        if (hasGraphics && hasCompute && hasTransfer && present) {
            mainQueueFamilyIndex_ = i;
            // If this family has 2+ queues, use the same family for secondary
            if (families[i].queueCount >= 2) {
                secondaryQueueFamilyIndex_ = i;
                sameQueueFamily_ = true;
            }
            break;
        }
    }

    // If no secondary yet, find a compute+transfer family
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

    // Fallback: use main for both
    if (secondaryQueueFamilyIndex_ == UINT32_MAX) {
        secondaryQueueFamilyIndex_ = mainQueueFamilyIndex_;
        sameQueueFamily_ = true;
    }
}

Result<void> DeviceService::createDevice(const InitConfig& config) {
    // Core extensions
    std::vector<const char*> extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    };

    for (auto ext : config.extraDeviceExtensions) {
        extensions.push_back(ext);
    }

    // Filter to supported extensions
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> supported(extCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, supported.data());

    std::vector<const char*> filtered;
    for (auto ext : extensions) {
        bool found = false;
        for (const auto& s : supported) {
            if (strcmp(ext, s.extensionName) == 0) { found = true; break; }
        }
        if (found) filtered.push_back(ext);
        else log::warn("vulkan", std::string("Extension not supported, skipping: ") + ext);
    }

    // Feature chain (minimal for Phase 3)
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{};
    rtFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtFeatures.rayTracingPipeline = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeatures.pNext = &rtFeatures;
    asFeatures.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceVulkan13Features vk13{};
    vk13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vk13.pNext = &asFeatures;
    vk13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features vk12{};
    vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk12.pNext = &vk13;
    vk12.bufferDeviceAddress = VK_TRUE;
    vk12.descriptorIndexing = VK_TRUE;
    vk12.runtimeDescriptorArray = VK_TRUE;
    vk12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    vk12.descriptorBindingPartiallyBound = VK_TRUE;
    vk12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    vk12.scalarBlockLayout = VK_TRUE;
    vk12.timelineSemaphore = VK_TRUE;
    vk12.shaderBufferInt64Atomics = VK_TRUE;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &vk12;
    features2.features.shaderInt64 = VK_TRUE;
    features2.features.shaderFloat64 = VK_TRUE;
    features2.features.shaderInt16 = VK_TRUE;
    features2.features.fillModeNonSolid = VK_TRUE;

    // Queue creation
    float mainPriority = 1.0f;
    float secPriority = 0.0f;

    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    if (sameQueueFamily_) {
        float priorities[] = {mainPriority, secPriority};
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = mainQueueFamilyIndex_;
        qi.queueCount = 2;
        qi.pQueuePriorities = priorities;
        queueInfos.push_back(qi);
    } else {
        VkDeviceQueueCreateInfo qi0{};
        qi0.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi0.queueFamilyIndex = mainQueueFamilyIndex_;
        qi0.queueCount = 1;
        qi0.pQueuePriorities = &mainPriority;
        queueInfos.push_back(qi0);

        VkDeviceQueueCreateInfo qi1{};
        qi1.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi1.queueFamilyIndex = secondaryQueueFamilyIndex_;
        qi1.queueCount = 1;
        qi1.pQueuePriorities = &secPriority;
        queueInfos.push_back(qi1);
    }

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.pNext = &features2;
    deviceInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    deviceInfo.pQueueCreateInfos = queueInfos.data();
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(filtered.size());
    deviceInfo.ppEnabledExtensionNames = filtered.data();

    VkResult result = vkCreateDevice(physicalDevice_, &deviceInfo, nullptr, &device_);
    if (result != VK_SUCCESS) {
        return makeError(result, "vkCreateDevice failed");
    }

    // Query RT properties for caps
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
    rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rtProps;
    vkGetPhysicalDeviceProperties2(physicalDevice_, &props2);

    caps_.rayTracingPipeline = true;
    caps_.accelerationStructure = true;
    caps_.timelineSemaphore = true;
    caps_.synchronization2 = true;
    caps_.bufferDeviceAddress = true;
    caps_.descriptorIndexing = true;
    caps_.maxRayRecursionDepth = rtProps.maxRayRecursionDepth;

    return {};
}

Result<void> DeviceService::createVma() {
    VmaVulkanFunctions vulkanFunctions{};
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo createInfo{};
    createInfo.physicalDevice = physicalDevice_;
    createInfo.device = device_;
    createInfo.instance = instance_;
    createInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    createInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    createInfo.pVulkanFunctions = &vulkanFunctions;

    VkResult result = vmaCreateAllocator(&createInfo, &vma_);
    if (result != VK_SUCCESS) {
        return makeError(result, "vmaCreateAllocator failed");
    }
    return {};
}

} // namespace engine::vk2
