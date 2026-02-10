#include "fsr3_upscaler.hpp"
#include "core/render/modules/world/fsr_upscaler/fsr_setup.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

#ifdef MCVR_ENABLE_FFX_UPSCALER
// Prevent FFX API from trying to export symbols from our own DLL
#    ifdef _WIN32
#        include <windows.h>
#        define FFX_API_ENTRY
#        pragma warning(push)
#        pragma warning(disable : 4005)
#    endif

#    include <ffx_api/ffx_api.hpp>
#    include <ffx_api/ffx_upscale.hpp>
#    include <ffx_api/vk/ffx_api_vk.hpp>

#    ifdef _WIN32
#        pragma warning(pop)
#    endif

#    include <volk.h>

// Helper to convert quality mode to FFX quality mode
static uint32_t toFFXQualityMode(mcvr::UpscalerQualityMode mode) {
    switch (mode) {
        case mcvr::UpscalerQualityMode::NativeAA: return FFX_UPSCALE_QUALITY_MODE_NATIVEAA;
        case mcvr::UpscalerQualityMode::Quality: return FFX_UPSCALE_QUALITY_MODE_QUALITY;
        case mcvr::UpscalerQualityMode::Balanced: return FFX_UPSCALE_QUALITY_MODE_BALANCED;
        case mcvr::UpscalerQualityMode::Performance: return FFX_UPSCALE_QUALITY_MODE_PERFORMANCE;
        case mcvr::UpscalerQualityMode::UltraPerformance: return FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE;
        default: return FFX_UPSCALE_QUALITY_MODE_QUALITY;
    }
}
#endif // MCVR_ENABLE_FFX_UPSCALER

namespace mcvr {

FSR3Upscaler::FSR3Upscaler() {
#ifdef MCVR_ENABLE_FFX_UPSCALER
    m_fsrContext = nullptr;
#endif
}

FSR3Upscaler::~FSR3Upscaler() {
    destroy();
}

bool FSR3Upscaler::isAvailable() const {
#ifdef MCVR_ENABLE_FFX_UPSCALER
    return true;
#else
    return false;
#endif
}

bool FSR3Upscaler::initialize(const UpscalerConfig &config) {
#ifdef DEBUG
    std::cout << "FSR3Upscaler::initialize called" << std::endl;
#endif

#ifndef MCVR_ENABLE_FFX_UPSCALER
    (void)config;
    std::cerr << "FSR3Upscaler::initialize: FSR3 support not compiled in" << std::endl;
    return false;
#else
    if (m_initialized) {
#    ifdef DEBUG
        std::cerr << "[DEBUG] FSR3: Destroying old context" << std::endl;
        std::flush(std::cerr);
#    endif
        destroy();
    }

#    ifdef DEBUG
    std::cerr << "[DEBUG] FSR3: Assigning handles" << std::endl;
#    endif
    m_device = config.device;
    m_physicalDevice = config.physicalDevice;
#    ifdef DEBUG
    std::cerr << "[DEBUG] FSR3: Device = " << (void *)m_device << std::endl;
    std::flush(std::cerr);
#    endif

    m_commandPool = config.commandPool;
    m_graphicsQueue = config.graphicsQueue;
    m_graphicsQueueFamily = config.graphicsQueueFamily;
    m_config = config;

    m_renderWidth = config.maxRenderWidth;
    m_renderHeight = config.maxRenderHeight;
    m_displayWidth = config.maxDisplayWidth;
    m_displayHeight = config.maxDisplayHeight;

#    ifdef DEBUG
    std::cout << "[FSR3] init config: render=" << m_renderWidth << "x" << m_renderHeight
              << " display=" << m_displayWidth << "x" << m_displayHeight
              << " quality=" << static_cast<int>(config.qualityMode) << " hdr=" << config.hdr
              << " depthInverted=" << config.depthInverted << " depthInfinite=" << config.depthInfinite
              << " autoExposure=" << config.autoExposure << " sharpening=" << config.enableSharpening
              << " sharpness=" << config.sharpness << std::endl;
#    endif

#    ifdef DEBUG
    std::cerr << "[DEBUG] FSR3: Calling createContext()" << std::endl;
    std::flush(std::cerr);
#    endif
    if (!createContext()) {
        std::cerr << "FSR3Upscaler::initialize: Failed to create FFX context" << std::endl;
        return false;
    }

    ffx::QueryDescUpscaleGetJitterPhaseCount jitterQuery{};
    jitterQuery.renderWidth = m_renderWidth;
    jitterQuery.displayWidth = m_displayWidth;
    jitterQuery.pOutPhaseCount = &m_jitterPhaseCount;
    ffx::Query(reinterpret_cast<ffx::Context &>(m_fsrContext), jitterQuery);

    m_jitterIndex = 0;
    m_initialized = true;

#    ifdef DEBUG
    std::cerr << "=== FSR3 INITIALIZED ===" << std::endl;
    std::cerr << "Render: " << m_renderWidth << "x" << m_renderHeight << " -> Display: " << m_displayWidth << "x"
              << m_displayHeight << std::endl;
    std::cerr << "Context: " << (m_fsrContext != nullptr ? "OK" : "FAILED") << std::endl;
#    endif

    m_debugLogged = false;

    return true;
#endif
}

bool FSR3Upscaler::createContext() {
#ifndef MCVR_ENABLE_FFX_UPSCALER
    return false;
#else
#    ifdef DEBUG
    std::cerr << "[DEBUG] FSR3: Entering createContext" << std::endl;
    std::flush(std::cerr);
#    endif

    // Query for available FSR3 versions
    std::vector<uint64_t> fsrVersionIds;
    {
        ffx::QueryDescGetVersions versionQuery{};
        versionQuery.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
#    ifdef _WIN32
        // On Windows, FFX SDK assumes 'device' is ID3D12Device* and crashes when passed a VkDevice.
        // Passing nullptr avoids this check and safely uses the internal FSR3 provider.
        versionQuery.device = nullptr;
#    else
        versionQuery.device = m_device;
#    endif
        uint64_t versionCount = 0;
        versionQuery.outputCount = &versionCount;

        ffxQuery(nullptr, &versionQuery.header);

        if (versionCount > 0) {
            fsrVersionIds.resize(versionCount);
            versionQuery.versionIds = fsrVersionIds.data();
            versionQuery.versionNames = nullptr;
            versionQuery.outputCount = &versionCount;
            ffxQuery(nullptr, &versionQuery.header);

#    ifdef DEBUG
            std::cout << "FSR3: Found " << versionCount << " available FSR version(s):" << std::endl;
            for (uint64_t i = 0; i < versionCount; ++i) {
                std::cout << "  Version[" << i << "] = 0x" << std::hex << fsrVersionIds[i] << std::dec << std::endl;
            }
#    endif

        } else {
            std::cerr << "FSR3: No alternative versions available" << std::endl;
        }
    }

    // Check for required Vulkan extensions
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extensionCount, extensions.data());

    auto hasExtension = [&](const char *name) {
        return std::any_of(extensions.begin(), extensions.end(),
                           [&](const VkExtensionProperties &ext) { return strcmp(ext.extensionName, name) == 0; });
    };

#    ifdef DEBUG
    std::cout << "FSR3: Checking required extensions..." << std::endl;
#    endif

    const char *requiredExtensions[] = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
                                        VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME,
                                        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
                                        VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME};

    for (const char *ext : requiredExtensions) {
        bool available = hasExtension(ext);
#    ifdef DEBUG
        std::cerr << "  " << ext << ": " << (available ? "OK" : "MISSING") << std::endl;
#    endif
        if (!available && strcmp(ext, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) == 0) {
            std::cerr << "FSR3: Missing required extension: " << ext << std::endl;
        }
    }

    // Create Vulkan backend use volk's loaded function pointers via our wrapper
    ffx::CreateBackendVKDesc backendDesc{};
    backendDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backendDesc.vkDevice = m_device;
    backendDesc.vkPhysicalDevice = m_physicalDevice;

    PFN_vkGetDeviceProcAddr deviceProcAddr =
        reinterpret_cast<PFN_vkGetDeviceProcAddr>(mcvr::fsr::customVkGetDeviceProcAddr);
    if (deviceProcAddr == nullptr) {
        std::cerr << "FSR3 ERROR: customVkGetDeviceProcAddr is NULL!" << std::endl;
        return false;
    }

    PFN_vkCreateComputePipelines testCreateComputePipelines =
        (PFN_vkCreateComputePipelines)deviceProcAddr(m_device, "vkCreateComputePipelines");
    PFN_vkCmdDispatch testCmdDispatch = (PFN_vkCmdDispatch)deviceProcAddr(m_device, "vkCmdDispatch");
    PFN_vkCreateSampler testCreateSampler = (PFN_vkCreateSampler)deviceProcAddr(m_device, "vkCreateSampler");

#    ifdef DEBUG
    std::cerr << "FSR3: Function pointer check:" << std::endl;
    std::cerr << "  vkCreateComputePipelines: " << (void *)testCreateComputePipelines << std::endl;
    std::cerr << "  vkCmdDispatch: " << (void *)testCmdDispatch << std::endl;
    std::cerr << "  vkCreateSampler: " << (void *)testCreateSampler << std::endl;
#    endif

    if (testCreateComputePipelines == nullptr || testCmdDispatch == nullptr) {
        std::cerr << "FSR3 ERROR: Failed to get required Vulkan function pointers" << std::endl;
        return false;
    }

    backendDesc.vkDeviceProcAddr = deviceProcAddr;
#    ifdef DEBUG
    std::cerr << "FSR3: vkDeviceProcAddr = " << reinterpret_cast<void *>(deviceProcAddr) << std::endl;
#    endif

    ffx::CreateContextDescUpscale createFsr{};
    createFsr.maxUpscaleSize.width = m_displayWidth;
    createFsr.maxUpscaleSize.height = m_displayHeight;
    createFsr.maxRenderSize.width = m_renderWidth;
    createFsr.maxRenderSize.height = m_renderHeight;

    createFsr.flags = 0;
    if (m_config.hdr) { createFsr.flags |= FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE; }
    if (m_config.depthInfinite) { createFsr.flags |= FFX_UPSCALE_ENABLE_DEPTH_INFINITE; }
    if (m_config.depthInverted) { createFsr.flags |= FFX_UPSCALE_ENABLE_DEPTH_INVERTED; }
    // Motion vectors are de-jittered in our prepare pass; do not enable FSR jitter cancellation
    if (m_config.autoExposure) { createFsr.flags |= FFX_UPSCALE_ENABLE_AUTO_EXPOSURE; }

    createFsr.fpMessage = mcvr::fsr::messageCallback;

    ffx::ReturnCode retCode =
        ffx::CreateContext(reinterpret_cast<ffx::Context &>(m_fsrContext), nullptr, createFsr, backendDesc);

    if (retCode != ffx::ReturnCode::Ok && !fsrVersionIds.empty()) {
        std::cerr << "FSR3: Initial CreateContext failed with error: " << static_cast<uint32_t>(retCode) << std::endl;
#    ifdef DEBUG
        std::cerr << "FSR3: Trying alternative shader versions..." << std::endl;
#    endif

        uint64_t bestVersionId = fsrVersionIds[0];
        uint32_t bestMajor = 0xFFFFFFFF;
        uint32_t bestIndex = 0;

        for (uint32_t i = 0; i < fsrVersionIds.size(); ++i) {
            uint32_t version = static_cast<uint32_t>(fsrVersionIds[i] & 0xFFFFFFFFu);
            uint32_t major = version >> 22;
            if (major < bestMajor) {
                bestMajor = major;
                bestVersionId = fsrVersionIds[i];
                bestIndex = i;
            }
        }

#    ifdef DEBUG
        std::cerr << "FSR3: Trying version[" << bestIndex << "] = 0x" << std::hex << bestVersionId << std::dec
                  << std::endl;
#    endif

        ffx::CreateContextDescOverrideVersion versionOverride{};
        versionOverride.versionId = bestVersionId;
        retCode = ffx::CreateContext(reinterpret_cast<ffx::Context &>(m_fsrContext), nullptr, createFsr, backendDesc,
                                     versionOverride);

        if (retCode == ffx::ReturnCode::Ok) {
#    ifdef DEBUG
            std::cout << "FSR3: CreateContext succeeded with version override!" << std::endl;
#    endif
        }
    }

    if (retCode != ffx::ReturnCode::Ok) {
        std::cerr << "FSR3: CreateContext failed with error: " << static_cast<uint32_t>(retCode) << std::endl;

        std::cerr << "FSR3: Device=" << m_device << std::endl;
        std::cerr << "FSR3: PhysicalDevice=" << m_physicalDevice << std::endl;

        if (retCode == ffx::ReturnCode::ErrorRuntimeError) {
            std::cerr << "FSR3: Runtime error - check Vulkan validation layers for details" << std::endl;
        }

        m_fsrContext = nullptr;
        return false;
    }

#    ifdef DEBUG
    std::cout << "FSR3: CreateContext succeeded!" << std::endl;
#    endif

    m_contextCreated = true;
    return true;
#endif
}

void FSR3Upscaler::destroyContext() {
#ifdef MCVR_ENABLE_FFX_UPSCALER
    if (m_contextCreated && m_fsrContext) {
        ffx::DestroyContext(reinterpret_cast<ffx::Context &>(m_fsrContext));
        m_fsrContext = nullptr;
        m_contextCreated = false;
    }
#endif
}

void FSR3Upscaler::dispatch(const UpscalerInput &input) {
#ifndef MCVR_ENABLE_FFX_UPSCALER
    (void)input;
    return;
#else
    if (!m_initialized || !m_contextCreated) { return; }

    if (!m_debugLogged || input.renderWidth != m_lastRenderWidth || input.renderHeight != m_lastRenderHeight ||
        input.displayWidth != m_lastDisplayWidth || input.displayHeight != m_lastDisplayHeight) {
        m_debugLogged = true;
        m_lastRenderWidth = input.renderWidth;
        m_lastRenderHeight = input.renderHeight;
        m_lastDisplayWidth = input.displayWidth;
        m_lastDisplayHeight = input.displayHeight;

#    ifdef DEBUG
        std::cout << "[FSR3] dispatch: render=" << input.renderWidth << "x" << input.renderHeight
                  << " display=" << input.displayWidth << "x" << input.displayHeight
                  << " depthFmt=" << static_cast<int>(input.depthFormat)
                  << " depthFfx=" << static_cast<int>(mcvr::fsr::vkToFfxFormat(input.depthFormat)) << " jitter=("
                  << input.jitterOffsetX << "," << input.jitterOffsetY << ")" << " mvScale=("
                  << input.motionVectorScaleX << "," << input.motionVectorScaleY << ")" << " reset=" << input.reset
                  << " preExposure=" << input.preExposure << " exposure=" << (input.exposureImage != VK_NULL_HANDLE)
                  << " reactive=" << (input.reactiveImage != VK_NULL_HANDLE) << std::endl;
        std::cout << "[FSR3] formats: color=R16G16B16A16_SFLOAT motion=R16G16_SFLOAT output=R16G16B16A16_SFLOAT"
                  << std::endl;
#    endif
    }

    auto createResource = [](VkImage image, uint32_t width, uint32_t height, VkFormat format, FfxApiResourceState state,
                             FfxApiResourceUsage usage = FFX_API_RESOURCE_USAGE_READ_ONLY) -> FfxApiResource {
        FfxApiResource resource = {};
        resource.resource = reinterpret_cast<void *>(image);
        resource.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
        resource.description.format = mcvr::fsr::vkToFfxFormat(format);
        resource.description.width = width;
        resource.description.height = height;
        resource.description.depth = 1;
        resource.description.mipCount = 1;
        resource.description.usage = usage;
        resource.state = state;
        return resource;
    };

    ffx::DispatchDescUpscale dispatchUpscale{};
    dispatchUpscale.commandList = input.commandBuffer;

    dispatchUpscale.color = createResource(input.colorImage, input.renderWidth, input.renderHeight,
                                           VK_FORMAT_R16G16B16A16_SFLOAT, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);

    VkFormat depthFormat = input.depthFormat != VK_FORMAT_UNDEFINED ? input.depthFormat : VK_FORMAT_D32_SFLOAT;
    dispatchUpscale.depth = createResource(input.depthImage, input.renderWidth, input.renderHeight, depthFormat,
                                           FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);

    dispatchUpscale.motionVectors = createResource(input.motionVectorImage, input.renderWidth, input.renderHeight,
                                                   VK_FORMAT_R16G16_SFLOAT, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);

    if (input.exposureImage != VK_NULL_HANDLE) {
        dispatchUpscale.exposure =
            createResource(input.exposureImage, 1, 1, VK_FORMAT_R32_SFLOAT, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    } else {
        dispatchUpscale.exposure = {};
    }

    if (input.reactiveImage != VK_NULL_HANDLE) {
        dispatchUpscale.reactive = createResource(input.reactiveImage, input.renderWidth, input.renderHeight,
                                                  VK_FORMAT_R8_UNORM, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    } else {
        dispatchUpscale.reactive = {};
    }

    dispatchUpscale.transparencyAndComposition = {};

    // Output resource
    dispatchUpscale.output =
        createResource(input.outputImage, input.displayWidth, input.displayHeight, VK_FORMAT_R16G16B16A16_SFLOAT,
                       FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, FFX_API_RESOURCE_USAGE_UAV);

    // Jitter offset
    dispatchUpscale.jitterOffset.x = input.jitterOffsetX;
    dispatchUpscale.jitterOffset.y = input.jitterOffsetY;

    // Motion vector scale
    dispatchUpscale.motionVectorScale.x = input.motionVectorScaleX;
    dispatchUpscale.motionVectorScale.y = input.motionVectorScaleY;

    // Resolution
    dispatchUpscale.renderSize.width = input.renderWidth;
    dispatchUpscale.renderSize.height = input.renderHeight;
    dispatchUpscale.upscaleSize.width = input.displayWidth;
    dispatchUpscale.upscaleSize.height = input.displayHeight;

    // Sharpening
    dispatchUpscale.enableSharpening = input.enableSharpening;
    dispatchUpscale.sharpness = input.sharpness;

    // Frame parameters
    dispatchUpscale.frameTimeDelta = input.frameTimeDelta;
    dispatchUpscale.preExposure = input.preExposure;
    dispatchUpscale.reset = input.reset;

    // Camera parameters
    dispatchUpscale.cameraNear = input.cameraNear;
    dispatchUpscale.cameraFar = input.cameraFar;
    dispatchUpscale.cameraFovAngleVertical = input.cameraFovVertical;
    dispatchUpscale.viewSpaceToMetersFactor = 1.0f;

    // Flags
    dispatchUpscale.flags = 0;

    // Dispatch FSR3
    static int dispatchCount = 0;
    ffx::ReturnCode retCode = ffx::Dispatch(reinterpret_cast<ffx::Context &>(m_fsrContext), dispatchUpscale);
    if (retCode != ffx::ReturnCode::Ok) {
        std::cerr << "FSR3 DISPATCH FAILED: " << static_cast<uint32_t>(retCode) << std::endl;
    }
    // else {
    //     dispatchCount++;
    //     if (dispatchCount == 1 || dispatchCount % 300 == 0) {
    //         std::cerr << "FSR3 dispatch OK (count: " << dispatchCount << ")" << std::endl;
    //     }
    // }

    // Increment jitter index
    m_jitterIndex++;
#endif
}

void FSR3Upscaler::resize(uint32_t renderWidth, uint32_t renderHeight, uint32_t displayWidth, uint32_t displayHeight) {
#ifndef MCVR_ENABLE_FFX_UPSCALER
    (void)renderWidth;
    (void)renderHeight;
    (void)displayWidth;
    (void)displayHeight;
#else
    if (!m_initialized) { return; }

    // Recreate context with new resolution
    m_renderWidth = renderWidth;
    m_renderHeight = renderHeight;
    m_displayWidth = displayWidth;
    m_displayHeight = displayHeight;

    destroyContext();
    createContext();

    // Re-query jitter phase count
    ffx::QueryDescUpscaleGetJitterPhaseCount jitterQuery{};
    jitterQuery.renderWidth = m_renderWidth;
    jitterQuery.displayWidth = m_displayWidth;
    jitterQuery.pOutPhaseCount = &m_jitterPhaseCount;
    ffx::Query(reinterpret_cast<ffx::Context &>(m_fsrContext), jitterQuery);

    m_jitterIndex = 0;
#endif
}

void FSR3Upscaler::destroy() {
#ifdef MCVR_ENABLE_FFX_UPSCALER
    if (!m_initialized) { return; }

    destroyContext();

    m_device = VK_NULL_HANDLE;
    m_physicalDevice = VK_NULL_HANDLE;
    m_commandPool = VK_NULL_HANDLE;
    m_graphicsQueue = VK_NULL_HANDLE;
    m_initialized = false;
#endif
}

} // namespace mcvr
