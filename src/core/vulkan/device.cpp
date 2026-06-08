#include "core/vulkan/device.hpp"

#include "core/render/gpu_diagnostics.hpp"
#include "core/render/aftermath_integration.hpp"
#include "core/render/modules/world/dlss/dlss_wrapper.hpp"
#include "core/render/streamline_context.hpp"
#include "core/vulkan/debug_utils.hpp"
#include "core/vulkan/instance.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/sync.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

std::ostream &deviceCout() {
    return std::cout << "[Device] ";
}

std::ostream &deviceCerr() {
    return std::cerr << "[Device] ";
}

vk::Device::Device(std::shared_ptr<Instance> instance,
                   std::shared_ptr<Window> window,
                   std::shared_ptr<PhysicalDevice> physicalDevice)
    : instance_(instance), window_(window), physicalDevice_(physicalDevice) {
    // enabled device extensions
    std::vector<const char *> enabledExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                                   VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                                                   VK_KHR_RAY_QUERY_EXTENSION_NAME,
                                                   VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                                   VK_KHR_SPIRV_1_4_EXTENSION_NAME,
                                                   VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                                                   VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
                                                   VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
                                                   VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
                                                   VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME,
                                                   VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
                                                   VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME,
                                                   VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
                                                   // HDR10: enables vkSetHdrMetadataEXT for SMPTE ST.2086 mastering display metadata
                                                   VK_EXT_HDR_METADATA_EXTENSION_NAME,
                                                   // SER: Shader Execution Reordering for material coherence
                                                   // Try EXT first (promoted), fall back to NV (original)
                                                   VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME,
                                                   VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME,
                                                   // Shader clock: per-pixel profiling instrumentation
                                                   VK_KHR_SHADER_CLOCK_EXTENSION_NAME,
                                                   // GPU diagnostics: breadcrumbs for DEVICE_LOST debugging
                                                   VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME,
                                                   VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME,
                                                   VK_EXT_DEVICE_FAULT_EXTENSION_NAME,
    // Nsight Graphics: NonSemantic Vulkan debug info for source-level shader profiling
    VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
#ifdef _WIN32
                                                   // External memory: Vulkan-D3D11 texture sharing for DComp overlay
                                                   VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
#endif
    };

    std::vector<VkExtensionProperties> dlssExtensions;
    NVSDK_NGX_Result dlssResult =
        NgxContext::getDlssRRRequiredDeviceExtensions(instance_, physicalDevice_, dlssExtensions);
    if (NVSDK_NGX_FAILED(dlssResult)) {
        deviceCerr() << "dlss device extensions unavailable; skipping." << std::endl;
    } else {
#ifdef DEBUG
        deviceCout() << "dlss instance extensions:" << std::endl;
#endif
        for (int i = 0; i < dlssExtensions.size(); i++) {
            if (std::strcmp(dlssExtensions[i].extensionName, "VK_EXT_buffer_device_address") == 0)
                continue; // already enabled using PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
#ifdef DEBUG
            deviceCout() << "\t" << dlssExtensions[i].extensionName << std::endl;
#endif
            enabledExtensions.push_back(dlssExtensions[i].extensionName);
        }
    }

    // Streamline SDK required device extensions (Reflex, DLSS-G)
    // Storage must outlive enabledExtensions pointers
    const auto &slDevExts = StreamlineContext::getRequiredDeviceExtensions();
    for (const auto &ext : slDevExts) {
        enabledExtensions.push_back(ext.c_str());
    }

    uint32_t deviceExtensionCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice_->vkPhysicalDevice(), nullptr, &deviceExtensionCount, nullptr);
    std::vector<VkExtensionProperties> deviceExtensions(deviceExtensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice_->vkPhysicalDevice(), nullptr, &deviceExtensionCount,
                                         deviceExtensions.data());

    std::unordered_set<std::string> supportedExtensions;
    supportedExtensions.reserve(deviceExtensions.size());
    for (const auto &ext : deviceExtensions) { supportedExtensions.insert(ext.extensionName); }

    std::vector<const char *> filteredExtensions;
    filteredExtensions.reserve(enabledExtensions.size());
    std::unordered_set<std::string> seenExtensions;
    for (const auto *ext : enabledExtensions) {
        if (supportedExtensions.find(ext) == supportedExtensions.end()) {
            deviceCerr() << "extension not supported, skipping: " << ext << std::endl;
            continue;
        }
        if (!seenExtensions.insert(ext).second) { continue; }
        filteredExtensions.push_back(ext);
    }

#ifdef DEBUG
    deviceCout() << "selected instance extensions:" << std::endl;
    for (int i = 0; i < filteredExtensions.size(); i++) { deviceCout() << "\t" << filteredExtensions[i] << std::endl; }
#endif

    // query supported features
    VkPhysicalDeviceShaderClockFeaturesKHR supportedShaderClockFeatures{};
    supportedShaderClockFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR;

    VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV supportedSERFeatures{};
    supportedSERFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV;
    supportedSERFeatures.pNext = &supportedShaderClockFeatures;
    VkPhysicalDeviceMaintenance5Features supportedMaintenance5{};
    supportedMaintenance5.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES;
    supportedMaintenance5.pNext = &supportedSERFeatures;

    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT supportedVertexInputDynamicState{};
    supportedVertexInputDynamicState.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT;
    supportedVertexInputDynamicState.pNext = &supportedMaintenance5;

    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT supportedExtendedDynamicState3{};
    supportedExtendedDynamicState3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
    supportedExtendedDynamicState3.pNext = &supportedVertexInputDynamicState;

    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT supportedExtendedDynamicState2{};
    supportedExtendedDynamicState2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT;
    supportedExtendedDynamicState2.pNext = &supportedExtendedDynamicState3;

    VkPhysicalDeviceVulkan13Features supportedVulkan13{};
    supportedVulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    supportedVulkan13.pNext = &supportedExtendedDynamicState2;

    VkPhysicalDeviceVulkan12Features supportedVulkan12{};
    supportedVulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    supportedVulkan12.pNext = &supportedVulkan13;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR supportedAccelerationStructureFeatures{};
    supportedAccelerationStructureFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    supportedAccelerationStructureFeatures.pNext = &supportedVulkan12;

    VkPhysicalDeviceRayQueryFeaturesKHR supportedRayQueryFeatures{};
    supportedRayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    supportedRayQueryFeatures.pNext = &supportedAccelerationStructureFeatures;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR supportedRayTracingFeatures{};
    supportedRayTracingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    supportedRayTracingFeatures.pNext = &supportedRayQueryFeatures;

    VkPhysicalDeviceFeatures2 supportedFeatures2{};
    supportedFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supportedFeatures2.pNext = &supportedRayTracingFeatures;

    vkGetPhysicalDeviceFeatures2(physicalDevice_->vkPhysicalDevice(), &supportedFeatures2);

    std::unordered_set<std::string> selectedExtensions;
    selectedExtensions.reserve(filteredExtensions.size());
    for (const auto *ext : filteredExtensions) { selectedExtensions.insert(ext); }
    auto hasExtension = [&](const char *name) { return selectedExtensions.find(name) != selectedExtensions.end(); };
    // enabling features

    // SER: Shader Execution Reordering for material coherence in RT
    // Accept either EXT (promoted) or NV (original) extension
    serSupported_ = (hasExtension(VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME) ||
                     hasExtension(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME)) &&
                    supportedSERFeatures.rayTracingInvocationReorder == VK_TRUE;

    VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV serFeatures{};
    serFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV;
    serFeatures.pNext = nullptr;
    serFeatures.rayTracingInvocationReorder = serSupported_ ? VK_TRUE : VK_FALSE;

    // Shader clock: per-pixel profiling instrumentation
    shaderClockSupported_ = hasExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME) &&
                            supportedShaderClockFeatures.shaderDeviceClock == VK_TRUE;

    VkPhysicalDeviceShaderClockFeaturesKHR shaderClockFeatures{};
    shaderClockFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR;
    shaderClockFeatures.pNext = &serFeatures;
    shaderClockFeatures.shaderDeviceClock = shaderClockSupported_ ? VK_TRUE : VK_FALSE;
    shaderClockFeatures.shaderSubgroupClock = shaderClockSupported_ ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceMaintenance5Features maintenance5Features{};
    maintenance5Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES;
    maintenance5Features.pNext = &shaderClockFeatures;
    maintenance5Features.maintenance5 =
        hasExtension(VK_KHR_MAINTENANCE_5_EXTENSION_NAME) ? supportedMaintenance5.maintenance5 : VK_FALSE;

    // GPU diagnostics: NV checkpoints + EXT device fault (zero cost when not queried)
    checkpointsSupported_ = hasExtension(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
    deviceFaultSupported_ = hasExtension(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);

    VkPhysicalDeviceDiagnosticsConfigFeaturesNV diagConfigFeatures{};
    diagConfigFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV;
    diagConfigFeatures.pNext = &maintenance5Features;
    diagConfigFeatures.diagnosticsConfig = hasExtension(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME) ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceFaultFeaturesEXT faultFeatures{};
    faultFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT;
    faultFeatures.pNext = &diagConfigFeatures;
    faultFeatures.deviceFault = deviceFaultSupported_ ? VK_TRUE : VK_FALSE;
    faultFeatures.deviceFaultVendorBinary = VK_FALSE;

    deviceCout() << "Shader Execution Reordering (SER): " << (serSupported_ ? "YES" : "NO")
                 << " (EXT=" << hasExtension(VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME)
                 << " NV=" << hasExtension(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME)
                 << " feature=" << supportedSERFeatures.rayTracingInvocationReorder << ")" << std::endl;
    deviceCout() << "Shader Clock: " << (shaderClockSupported_ ? "YES" : "NO") << std::endl;
    deviceCout() << "GPU Diagnostics Checkpoints: " << (checkpointsSupported_ ? "YES" : "NO") << std::endl;
    deviceCout() << "GPU Device Fault: " << (deviceFaultSupported_ ? "YES" : "NO") << std::endl;

    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT vertexInputDynamicState{};
    vertexInputDynamicState.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT;
    vertexInputDynamicState.pNext = &faultFeatures;
    vertexInputDynamicState.vertexInputDynamicState = hasExtension(VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME) ?
                                                          supportedVertexInputDynamicState.vertexInputDynamicState :
                                                          VK_FALSE;

    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT extendedDynamicState3{};
    extendedDynamicState3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
    extendedDynamicState3.pNext = &vertexInputDynamicState;
    if (hasExtension(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME)) {
        extendedDynamicState3.extendedDynamicState3PolygonMode =
            supportedExtendedDynamicState3.extendedDynamicState3PolygonMode;
        extendedDynamicState3.extendedDynamicState3ColorBlendEnable =
            supportedExtendedDynamicState3.extendedDynamicState3ColorBlendEnable;
        extendedDynamicState3.extendedDynamicState3ColorBlendEquation =
            supportedExtendedDynamicState3.extendedDynamicState3ColorBlendEquation;
        extendedDynamicState3.extendedDynamicState3ColorWriteMask =
            supportedExtendedDynamicState3.extendedDynamicState3ColorWriteMask;
        extendedDynamicState3.extendedDynamicState3LogicOpEnable =
            supportedExtendedDynamicState3.extendedDynamicState3LogicOpEnable;
    }

    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT extendedDynamicState2{};
    extendedDynamicState2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT;
    extendedDynamicState2.pNext = &extendedDynamicState3;
    if (hasExtension(VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME)) {
        extendedDynamicState2.extendedDynamicState2 = supportedExtendedDynamicState2.extendedDynamicState2;
#if defined(USE_AMD)
        // AMD drivers have issues with extendedDynamicState2LogicOp on both Linux and Windows
        VkBool32 wantLogicOp = VK_FALSE;
#else
        VkBool32 wantLogicOp = VK_TRUE;
#endif
        extendedDynamicState2.extendedDynamicState2LogicOp =
            (supportedExtendedDynamicState2.extendedDynamicState2LogicOp && wantLogicOp) ? VK_TRUE : VK_FALSE;

        // Store the flag for runtime checks
        extendedDynamicState2LogicOp_ = (extendedDynamicState2.extendedDynamicState2LogicOp == VK_TRUE);

#ifdef DEBUG
        std::cout << "[Device-Debug] extendedDynamicState2: "
                  << (extendedDynamicState2.extendedDynamicState2 ? "YES" : "NO") << std::endl;
        std::cout << "[Device-Debug] extendedDynamicState2LogicOp: "
                  << (extendedDynamicState2.extendedDynamicState2LogicOp ? "YES" : "NO") << std::endl;
#endif

        extendedDynamicState2.extendedDynamicState2PatchControlPoints = VK_FALSE;
    } else {
        std::cerr << "[Device-Debug] VK_EXT_extended_dynamic_state2 NOT FOUND!" << std::endl;
    }

    VkPhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.pNext = &extendedDynamicState2;
    vulkan13Features.shaderDemoteToHelperInvocation = supportedVulkan13.shaderDemoteToHelperInvocation;
    vulkan13Features.synchronization2 = supportedVulkan13.synchronization2;

    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.pNext = &vulkan13Features;
    vulkan12Features.bufferDeviceAddress = supportedVulkan12.bufferDeviceAddress;
    vulkan12Features.descriptorBindingUpdateUnusedWhilePending =
        supportedVulkan12.descriptorBindingUpdateUnusedWhilePending;
    vulkan12Features.descriptorBindingPartiallyBound = supportedVulkan12.descriptorBindingPartiallyBound;
    vulkan12Features.descriptorIndexing = supportedVulkan12.descriptorIndexing;
    vulkan12Features.runtimeDescriptorArray = supportedVulkan12.runtimeDescriptorArray;
    vulkan12Features.shaderSampledImageArrayNonUniformIndexing =
        supportedVulkan12.shaderSampledImageArrayNonUniformIndexing;
    vulkan12Features.descriptorBindingUniformBufferUpdateAfterBind =
        supportedVulkan12.descriptorBindingUniformBufferUpdateAfterBind;
    vulkan12Features.descriptorBindingSampledImageUpdateAfterBind =
        supportedVulkan12.descriptorBindingSampledImageUpdateAfterBind;
    vulkan12Features.descriptorBindingStorageImageUpdateAfterBind =
        supportedVulkan12.descriptorBindingStorageImageUpdateAfterBind;
    vulkan12Features.descriptorBindingStorageBufferUpdateAfterBind =
        supportedVulkan12.descriptorBindingStorageBufferUpdateAfterBind;
    vulkan12Features.shaderFloat16 = supportedVulkan12.shaderFloat16;
    vulkan12Features.shaderBufferInt64Atomics = supportedVulkan12.shaderBufferInt64Atomics;
    vulkan12Features.scalarBlockLayout = supportedVulkan12.scalarBlockLayout;
    vulkan12Features.timelineSemaphore = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures = {};
    accelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accelerationStructureFeatures.pNext = &vulkan12Features;
    if (hasExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) {
        accelerationStructureFeatures.accelerationStructure =
            supportedAccelerationStructureFeatures.accelerationStructure;
        accelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind =
            supportedAccelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind;
    }

    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};
    rayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rayQueryFeatures.pNext = &accelerationStructureFeatures;
    if (hasExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME)) {
        rayQueryFeatures.rayQuery = supportedRayQueryFeatures.rayQuery;
    }

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingFeatures = {};
    rayTracingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rayTracingFeatures.pNext = &rayQueryFeatures;
    if (hasExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) {
        rayTracingFeatures.rayTracingPipeline = supportedRayTracingFeatures.rayTracingPipeline;
    }

    VkPhysicalDeviceFeatures features = {};
    features.shaderClipDistance = supportedFeatures2.features.shaderClipDistance;
    features.shaderCullDistance = supportedFeatures2.features.shaderCullDistance;
    features.logicOp = supportedFeatures2.features.logicOp;
    features.fillModeNonSolid = supportedFeatures2.features.fillModeNonSolid;
    features.depthBiasClamp = supportedFeatures2.features.depthBiasClamp;
    features.shaderInt64 = supportedFeatures2.features.shaderInt64;
    features.shaderFloat64 = supportedFeatures2.features.shaderFloat64;
    features.shaderInt16 = supportedFeatures2.features.shaderInt16;

    // Aftermath-level diagnostics: shader debug info + resource tracking + auto checkpoints.
    // Enables the NVIDIA driver to report exact shader source location and resource names
    // on DEVICE_LOST via VK_EXT_device_fault. Zero overhead when no crash occurs.
    VkDeviceDiagnosticsConfigCreateInfoNV diagCreateInfo{};
    diagCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV;
    diagCreateInfo.pNext = &rayTracingFeatures;
    if (hasExtension(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME)) {
        diagCreateInfo.flags =
            VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV |
            VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV |
            VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV |
            VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_ERROR_REPORTING_BIT_NV;
        deviceCout() << "GPU Diagnostics Config: shader debug + resource tracking + auto checkpoints ENABLED" << std::endl;
    }

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &diagCreateInfo;
    features2.features = features;

    // Initialize Aftermath GPU crash dump collection before device creation
    {
        auto logsDir = std::filesystem::current_path() / "radiance" / "logs";
        AftermathIntegration::init(logsDir);
    }

    // Streamline interposer: use interposer's vkCreateDevice so it can track the VkDevice.
    // Without this, slSetVulkanInfo fails with eErrorInvalidIntegration because the
    // interposer has no record of the device handle.
    auto slCreateDevice = reinterpret_cast<PFN_vkCreateDevice>(StreamlineContext::getVkCreateDevice());
    PFN_vkCreateDevice createDeviceFn = slCreateDevice ? slCreateDevice : vkCreateDevice;

    // create logical device
    VkDeviceCreateInfo deviceCreateInfo = {};
    if (physicalDevice_->mainQueueIndex() == physicalDevice_->secondaryQueueIndex()) {
        std::vector<float> queuePriorities{{1.0, 0.0}};
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = physicalDevice_->mainQueueIndex();
        queueCreateInfo.queueCount = physicalDevice_->mainQueueCount() >= 2 ? 2 : 1;
        queueCreateInfo.pQueuePriorities = queuePriorities.data();

        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(filteredExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = filteredExtensions.data();
        deviceCreateInfo.pNext = &features2;
        deviceCreateInfo.pEnabledFeatures = nullptr;

        if (createDeviceFn(physicalDevice_->vkPhysicalDevice(), &deviceCreateInfo, nullptr, &device_) != VK_SUCCESS) {
            deviceCerr() << "Failed to create logical device!" << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        std::vector<float> queuePriorities{{1.0, 0.0}};
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos(2);
        queueCreateInfos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfos[0].queueFamilyIndex = physicalDevice_->mainQueueIndex();
        queueCreateInfos[0].queueCount = 1;
        queueCreateInfos[0].pQueuePriorities = &queuePriorities[0];

        queueCreateInfos[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfos[1].queueFamilyIndex = physicalDevice_->secondaryQueueIndex();
        queueCreateInfos[1].queueCount = 1;
        queueCreateInfos[1].pQueuePriorities = &queuePriorities[1];

        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = queueCreateInfos.size();
        deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(filteredExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = filteredExtensions.data();
        deviceCreateInfo.pNext = &features2;
        deviceCreateInfo.pEnabledFeatures = nullptr;

        if (createDeviceFn(physicalDevice_->vkPhysicalDevice(), &deviceCreateInfo, nullptr, &device_) != VK_SUCCESS) {
            deviceCerr() << "Failed to create logical device!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Streamline: the interposer's vkCreateDevice hook already registered the device
    // and initialized plugins. Just load feature function pointers (Reflex, etc.).
    StreamlineContext::onDeviceCreated();

    volkLoadDevice(device_);

    // Streamline interposer: volkLoadDevice replaces all device-level function pointers with
    // raw driver dispatch (via vkGetDeviceProcAddr), bypassing the interposer's hooks.
    // Re-override the mandatory hooks that SL needs for Reflex/DLSS-G (present, acquire, swapchain).
    auto slGDPA = reinterpret_cast<PFN_vkGetDeviceProcAddr>(StreamlineContext::getVkGetDeviceProcAddr());
    if (slGDPA) {
        vkQueuePresentKHR = reinterpret_cast<PFN_vkQueuePresentKHR>(slGDPA(device_, "vkQueuePresentKHR"));
        vkCreateSwapchainKHR = reinterpret_cast<PFN_vkCreateSwapchainKHR>(slGDPA(device_, "vkCreateSwapchainKHR"));
        vkDestroySwapchainKHR = reinterpret_cast<PFN_vkDestroySwapchainKHR>(slGDPA(device_, "vkDestroySwapchainKHR"));
        vkGetSwapchainImagesKHR = reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(slGDPA(device_, "vkGetSwapchainImagesKHR"));
        vkAcquireNextImageKHR = reinterpret_cast<PFN_vkAcquireNextImageKHR>(slGDPA(device_, "vkAcquireNextImageKHR"));
        vkDeviceWaitIdle = reinterpret_cast<PFN_vkDeviceWaitIdle>(slGDPA(device_, "vkDeviceWaitIdle"));
    }

#ifdef DEBUG
    deviceCout() << "Logical device created successfully!" << std::endl;
#endif

    vkGetDeviceQueue(device_, physicalDevice_->mainQueueIndex(), 0, &mainQueue_);
    vkGetDeviceQueue(device_, physicalDevice_->secondaryQueueIndex(),
                     physicalDevice_->mainQueueIndex() == physicalDevice_->secondaryQueueIndex() &&
                             physicalDevice_->mainQueueCount() >= 2
                         ? 1
                         : 0,
                     &secondaryQueue_);

    vk::DebugUtils::setObjectName(device_, VK_OBJECT_TYPE_DEVICE, device_, "Radiance VkDevice");
    vk::DebugUtils::setObjectName(device_, VK_OBJECT_TYPE_QUEUE, mainQueue_, "Radiance Main Queue");
    vk::DebugUtils::setObjectName(device_, VK_OBJECT_TYPE_QUEUE, secondaryQueue_, "Radiance Secondary BLAS Queue");

    loadPipelineCache();

    // Initialize GPU diagnostics (checkpoints + device fault)
    GpuDiag::init(device_, physicalDevice_->vkPhysicalDevice(), checkpointsSupported_, deviceFaultSupported_);
}

void vk::Device::createTimelineSemaphores() {
    blasSemaphore_ = TimelineSemaphore::create(shared_from_this(), 0);
    materialUploadSemaphore_ = TimelineSemaphore::create(shared_from_this(), 0);
}

vk::Device::~Device() {
    savePipelineCache();
    if (pipelineCache_ != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(device_, pipelineCache_, nullptr);
    }
    blasSemaphore_.reset();
    materialUploadSemaphore_.reset();
    vkDestroyDevice(device_, nullptr);

#ifdef DEBUG
    deviceCout() << "device deconstructed" << std::endl;
#endif
}

std::string vk::Device::pipelineCachePath() {
    // Store alongside the game's config directory
    const char *appdata = std::getenv("APPDATA");
    if (appdata) {
        return std::string(appdata) + "/Radiance/pipeline_cache.bin";
    }
    return "pipeline_cache.bin";
}

void vk::Device::loadPipelineCache() {
    auto path = pipelineCachePath();
    std::vector<uint8_t> cacheData;

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        auto size = file.tellg();
        if (size > 0) {
            cacheData.resize(static_cast<size_t>(size));
            file.seekg(0);
            file.read(reinterpret_cast<char *>(cacheData.data()), size);
        }
        file.close();
        deviceCout() << "Loaded pipeline cache (" << cacheData.size() << " bytes) from " << path << std::endl;
    }

    VkPipelineCacheCreateInfo cacheCreateInfo{};
    cacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheCreateInfo.initialDataSize = cacheData.size();
    cacheCreateInfo.pInitialData = cacheData.empty() ? nullptr : cacheData.data();

    VkResult result = vkCreatePipelineCache(device_, &cacheCreateInfo, nullptr, &pipelineCache_);
    if (result != VK_SUCCESS) {
        deviceCerr() << "Failed to create pipeline cache (VkResult=" << result << "), retrying empty" << std::endl;
        // Retry with empty cache (stale data)
        cacheCreateInfo.initialDataSize = 0;
        cacheCreateInfo.pInitialData = nullptr;
        vkCreatePipelineCache(device_, &cacheCreateInfo, nullptr, &pipelineCache_);
    }
}

void vk::Device::savePipelineCache() {
    if (pipelineCache_ == VK_NULL_HANDLE) return;

    size_t dataSize = 0;
    if (vkGetPipelineCacheData(device_, pipelineCache_, &dataSize, nullptr) != VK_SUCCESS || dataSize == 0) return;

    std::vector<uint8_t> data(dataSize);
    if (vkGetPipelineCacheData(device_, pipelineCache_, &dataSize, data.data()) != VK_SUCCESS) return;

    auto path = pipelineCachePath();
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char *>(data.data()), dataSize);
        file.close();
        deviceCout() << "Saved pipeline cache (" << dataSize << " bytes) to " << path << std::endl;
    }
}

VkDevice &vk::Device::vkDevice() {
    return device_;
}

VkQueue &vk::Device::mainVkQueue() {
    return mainQueue_;
}

VkQueue &vk::Device::secondaryQueue() {
    return secondaryQueue_;
}
