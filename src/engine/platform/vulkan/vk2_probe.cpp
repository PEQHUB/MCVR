#include "vk2_probe.hpp"

#include "diagnostics/log.hpp"
#include "diagnostics/boot_trace.hpp"

#include <volk.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace engine::vk2 {

namespace {

constexpr const char* kRequiredDeviceExtensions[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    // Acceleration-structure dependencies — listed for completeness even though
    // a driver supporting accel-struct will always also expose these.
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
};

int scoreDeviceForProbe(VkPhysicalDevice pd) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);

    int score = 0;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)        score += 10000;
    else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 1000;
    if (props.vendorID == 0x10DE) score += 500; // NVIDIA bias

    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, exts.data());
    for (const auto& ext : exts) {
        if (std::strcmp(ext.extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) == 0) {
            score += 5000;
            break;
        }
    }
    return score;
}

bool hasGraphicsQueueFamily(VkPhysicalDevice pd) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
    if (count == 0) return false;
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, families.data());
    for (const auto& f : families) {
        if (f.queueFlags & VK_QUEUE_GRAPHICS_BIT) return true;
    }
    return false;
}

std::string joinComma(const std::vector<std::string>& items) {
    std::string out;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i) out += ", ";
        out += items[i];
    }
    return out;
}

} // anonymous namespace

ProbeResult probeVulkan(bool enableValidation) {
    ProbeResult r;

    // --- Loader ---
    if (volkInitialize() != VK_SUCCESS) {
        r.reason = "volkInitialize failed (Vulkan loader missing or unable to find drivers)";
        return r;
    }

    uint32_t loaderApi = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion) {
        if (vkEnumerateInstanceVersion(&loaderApi) != VK_SUCCESS) {
            loaderApi = VK_API_VERSION_1_0;
        }
    }
    r.loaderApiVersion = loaderApi;
    if (loaderApi < VK_API_VERSION_1_2) {
        r.reason = "Vulkan loader is too old (need 1.2+, found "
                 + std::to_string(VK_VERSION_MAJOR(loaderApi)) + "."
                 + std::to_string(VK_VERSION_MINOR(loaderApi)) + ")";
        return r;
    }

    // --- Instance ---
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "RadSER Probe";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 2, 0);
    appInfo.pEngineName = "RadSER Engine V2 Probe";
    appInfo.engineVersion = VK_MAKE_VERSION(2, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> instanceExts = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    };

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &appInfo;
    ci.enabledExtensionCount = static_cast<uint32_t>(instanceExts.size());
    ci.ppEnabledExtensionNames = instanceExts.data();

    const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    if (enableValidation) {
        ci.enabledLayerCount = 1;
        ci.ppEnabledLayerNames = &validationLayer;
    }

    VkInstance instance = VK_NULL_HANDLE;
    VkResult ir = vkCreateInstance(&ci, nullptr, &instance);
    if (ir != VK_SUCCESS) {
        // Validation layer might be missing — retry without it before declaring failure.
        if (enableValidation && ir == VK_ERROR_LAYER_NOT_PRESENT) {
            ci.enabledLayerCount = 0;
            ir = vkCreateInstance(&ci, nullptr, &instance);
        }
        if (ir != VK_SUCCESS) {
            r.reason = "vkCreateInstance failed (VkResult=" + std::to_string(ir) + ")";
            return r;
        }
    }

    volkLoadInstance(instance);

    // --- Physical devices ---
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    r.physicalDeviceCount = static_cast<int>(deviceCount);
    if (deviceCount == 0) {
        r.reason = "No Vulkan-capable physical devices found";
        vkDestroyInstance(instance, nullptr);
        return r;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    int bestScore = -1;
    VkPhysicalDevice best = VK_NULL_HANDLE;
    for (auto pd : devices) {
        int s = scoreDeviceForProbe(pd);
        if (s > bestScore) {
            bestScore = s;
            best = pd;
        }
    }
    if (best == VK_NULL_HANDLE) {
        r.reason = "Could not select a physical device";
        vkDestroyInstance(instance, nullptr);
        return r;
    }

    // Identity
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(best, &props);
    r.gpuName          = props.deviceName;
    r.vendorId         = props.vendorID;
    r.driverVersion    = props.driverVersion;
    r.deviceApiVersion = props.apiVersion;

    // Queue families
    r.hasGraphicsQueue = hasGraphicsQueueFamily(best);
    if (!r.hasGraphicsQueue) {
        r.missingFeatures.push_back("graphics queue family");
    }

    // Extensions
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(best, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(best, nullptr, &extCount, exts.data());
    std::set<std::string> have;
    for (const auto& e : exts) have.insert(e.extensionName);

    auto check = [&](const char* name, bool& flag) {
        bool present = have.count(name) > 0;
        flag = present;
        if (!present) r.missingExtensions.push_back(name);
        return present;
    };

    check(VK_KHR_SWAPCHAIN_EXTENSION_NAME,             r.hasSwapchainExt);
    check(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,  r.hasRayTracingPipeline);
    check(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,r.hasAccelerationStructure);
    // Buffer-device-address: prefer the KHR ext, but Vulkan 1.2 promotes it
    // to core. Treat either as "present".
    {
        bool extPresent = have.count(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) > 0;
        bool corePromotion = props.apiVersion >= VK_API_VERSION_1_2;
        r.hasBufferDeviceAddress = extPresent || corePromotion;
        if (!r.hasBufferDeviceAddress) {
            r.missingExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        }
    }
    {
        bool dho = have.count(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) > 0;
        if (!dho) r.missingExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    }
    (void)kRequiredDeviceExtensions; // referenced for documentation in the header

    // --- Core 1.2/1.3 feature bits (must match vk2_device.cpp `canBaseline`) ---
    // An extension being present does not guarantee the feature bit is on —
    // chain the features2 query to verify the capabilities this renderer uses.
    if (props.apiVersion >= VK_API_VERSION_1_2 && vkGetPhysicalDeviceFeatures2) {
        VkPhysicalDeviceVulkan13Features f13{};
        f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

        VkPhysicalDeviceVulkan12Features f12{};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        f12.pNext = &f13;

        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &f12;

        vkGetPhysicalDeviceFeatures2(best, &f2);

        r.hasSynchronization2   = f13.synchronization2   == VK_TRUE;
        r.hasDynamicRendering   = f13.dynamicRendering   == VK_TRUE;
        r.hasDescriptorIndexing = f12.descriptorIndexing == VK_TRUE;
        r.hasTimelineSemaphore  = f12.timelineSemaphore  == VK_TRUE;

        if (!r.hasSynchronization2)   r.missingFeatures.push_back("synchronization2 (1.3)");
        if (!r.hasDynamicRendering)   r.missingFeatures.push_back("dynamicRendering (1.3)");
        if (!r.hasDescriptorIndexing) r.missingFeatures.push_back("descriptorIndexing (1.2)");
        if (!r.hasTimelineSemaphore)  r.missingFeatures.push_back("timelineSemaphore (1.2)");
    } else {
        r.missingFeatures.push_back(
            "Vulkan 1.2+ required on physical device for features2 chain");
    }

    // --- Verdict ---
    r.ok = r.hasGraphicsQueue
        && r.hasSwapchainExt
        && r.hasRayTracingPipeline
        && r.hasAccelerationStructure
        && r.hasBufferDeviceAddress
        && r.hasSynchronization2
        && r.hasDynamicRendering
        && r.hasDescriptorIndexing
        && r.hasTimelineSemaphore;

    if (!r.ok) {
        std::ostringstream oss;
        oss << "Vulkan probe rejected " << r.gpuName << ": ";
        if (!r.hasGraphicsQueue)             oss << "no graphics queue; ";
        if (!r.hasSwapchainExt)              oss << "no swapchain; ";
        if (!r.hasRayTracingPipeline)        oss << "no RT pipeline; ";
        if (!r.hasAccelerationStructure)     oss << "no acceleration structure; ";
        if (!r.hasBufferDeviceAddress)       oss << "no buffer-device-address; ";
        if (!r.hasSynchronization2)          oss << "no synchronization2; ";
        if (!r.hasDynamicRendering)          oss << "no dynamicRendering; ";
        if (!r.hasDescriptorIndexing)        oss << "no descriptorIndexing; ";
        if (!r.hasTimelineSemaphore)         oss << "no timelineSemaphore; ";
        r.reason = oss.str();
    }

    // --- Cleanup ---
    vkDestroyInstance(instance, nullptr);

    boot_trace::breadcrumb("probe", r.ok ? "ok" : "fail",
        r.ok ? r.gpuName
             : (r.gpuName.empty() ? r.reason : (r.gpuName + ": " + r.reason)));
    return r;
}

std::string writeProbeReport(const std::string& logDir, const ProbeResult& r) {
    std::error_code ec;
    std::filesystem::create_directories(logDir, ec);
    std::filesystem::path filename = r.ok ? "v2_probe_ok.txt" : "v2_cannot_start.txt";
    std::filesystem::path full = std::filesystem::path(logDir) / filename;

    std::ofstream out(full, std::ios::trunc);
    if (!out.is_open()) return {};

    out << (r.ok ? "=== V2 Vulkan probe: OK ===\n"
                 : "=== V2 Vulkan probe: FAILED ===\n\n"
                   "The Radiance V2 engine cannot start on this system.\n"
                   "The check below ran BEFORE any window state was modified, so\n"
                   "Minecraft will continue to run on the legacy renderer (or fail\n"
                   "to start cleanly if useV2Engine=true). To recover:\n"
                   "  - Update your GPU driver to a recent version that exposes\n"
                   "    Vulkan 1.3 + ray tracing.\n"
                   "  - On NVIDIA: install driver 535+ (RTX) for full RT support.\n"
                   "  - Set useV2Engine=false in radiance/options.properties to\n"
                   "    silence this check until your driver is updated.\n\n");

    out << "GPU:                " << (r.gpuName.empty() ? "<none>" : r.gpuName) << "\n";
    out << "Vendor ID:          0x" << std::hex << r.vendorId << std::dec << "\n";
    out << "Driver version:     0x" << std::hex << r.driverVersion << std::dec << "\n";
    out << "Loader Vulkan ver:  "
        << VK_VERSION_MAJOR(r.loaderApiVersion) << "."
        << VK_VERSION_MINOR(r.loaderApiVersion) << "."
        << VK_VERSION_PATCH(r.loaderApiVersion) << "\n";
    out << "Device Vulkan ver:  "
        << VK_VERSION_MAJOR(r.deviceApiVersion) << "."
        << VK_VERSION_MINOR(r.deviceApiVersion) << "."
        << VK_VERSION_PATCH(r.deviceApiVersion) << "\n";
    out << "Physical devices:   " << r.physicalDeviceCount << "\n\n";

    out << "Capability checks:\n";
    auto bool3 = [](bool b) { return b ? "yes" : "NO "; };
    out << "  graphics queue:           " << bool3(r.hasGraphicsQueue) << "\n";
    out << "  VK_KHR_swapchain:         " << bool3(r.hasSwapchainExt) << "\n";
    out << "  RT pipeline:              " << bool3(r.hasRayTracingPipeline) << "\n";
    out << "  acceleration structure:   " << bool3(r.hasAccelerationStructure) << "\n";
    out << "  buffer device address:    " << bool3(r.hasBufferDeviceAddress) << "\n";
    out << "  synchronization2 (1.3):   " << bool3(r.hasSynchronization2) << "\n";
    out << "  dynamicRendering (1.3):   " << bool3(r.hasDynamicRendering) << "\n";
    out << "  descriptorIndexing (1.2): " << bool3(r.hasDescriptorIndexing) << "\n";
    out << "  timelineSemaphore (1.2):  " << bool3(r.hasTimelineSemaphore) << "\n\n";

    if (!r.missingExtensions.empty()) {
        out << "Missing device extensions:\n";
        for (const auto& e : r.missingExtensions) out << "  - " << e << "\n";
        out << "\n";
    }
    if (!r.missingFeatures.empty()) {
        out << "Missing features:\n";
        for (const auto& f : r.missingFeatures) out << "  - " << f << "\n";
        out << "\n";
    }
    if (!r.reason.empty()) {
        out << "Summary: " << r.reason << "\n\n";
    }

    out << "See radiance/logs/v2_boot.log for the boot breadcrumb trace.\n";
    out.flush();

    return full.string();
}

} // namespace engine::vk2
