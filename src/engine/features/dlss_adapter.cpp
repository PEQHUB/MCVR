// Copyright (c) 2026 Radiance -- V2 Engine DLSS Ray Reconstruction (DLSS-RR) adapter.
//
// Implementation notes:
//  - This adapter replaces the temporal denoiser entirely. DLSS-RR consumes the
//    1spp noisy HDR radiance plus 24 guide buffers (normals, albedo, roughness,
//    motion vectors, hit distances, etc.) and produces denoised+upscaled output.
//  - Uses NVSDK_NGX_Feature_RayReconstruction (not SuperSampling).
//  - Halton(2,3) jitter is generated internally and exposed via currentJitter()
//    so the RT adapter can inject it into the WorldUBO before tracing.
//  - Pre-exposure is driven by auto-exposure feedback from ToneMappingAdapter.
//  - NGX requires strict shutdown ordering: release feature handle BEFORE
//    destroying parameters BEFORE calling NVSDK_NGX_VULKAN_Shutdown1, all while
//    the VkDevice is still valid.

#include "dlss_adapter.hpp"

#include "app/engine_services.hpp"
#include "config/config_service.hpp"
#include "diagnostics/log.hpp"
#include "diagnostics/metrics_service.hpp"
#include "config/engine_config.hpp"
#include "frame/frame_context.hpp"
#include "frame/frame_scheduler.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "rendergraph/graph_builder.hpp"

// NGX SDK headers -- included only in the .cpp to keep the .hpp NGX-free.
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_defs_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd_vk.h>

#include <volk.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string>

namespace engine {

namespace {

constexpr unsigned long long kNgxApplicationId = 0xbaadf00dbaadcafe;

const char* ngxResultToString(NVSDK_NGX_Result result) {
    switch (result) {
        case NVSDK_NGX_Result_Success: return "Success";
        case NVSDK_NGX_Result_Fail: return "Fail";
        case NVSDK_NGX_Result_FAIL_FeatureNotSupported: return "FeatureNotSupported";
        case NVSDK_NGX_Result_FAIL_PlatformError: return "PlatformError";
        case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists: return "FeatureAlreadyExists";
        case NVSDK_NGX_Result_FAIL_FeatureNotFound: return "FeatureNotFound";
        case NVSDK_NGX_Result_FAIL_InvalidParameter: return "InvalidParameter";
        case NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall: return "ScratchBufferTooSmall";
        case NVSDK_NGX_Result_FAIL_NotInitialized: return "NotInitialized";
        case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat: return "UnsupportedInputFormat";
        case NVSDK_NGX_Result_FAIL_RWFlagMissing: return "RWFlagMissing";
        case NVSDK_NGX_Result_FAIL_MissingInput: return "MissingInput";
        case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature: return "UnableToInitializeFeature";
        case NVSDK_NGX_Result_FAIL_OutOfDate: return "OutOfDate";
        case NVSDK_NGX_Result_FAIL_OutOfGPUMemory: return "OutOfGPUMemory";
        case NVSDK_NGX_Result_FAIL_UnsupportedFormat: return "UnsupportedFormat";
        case NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath: return "UnableToWriteToAppDataPath";
        case NVSDK_NGX_Result_FAIL_UnsupportedParameter: return "UnsupportedParameter";
        case NVSDK_NGX_Result_FAIL_Denied: return "Denied";
        case NVSDK_NGX_Result_FAIL_NotImplemented: return "NotImplemented";
        default: return "Unknown";
    }
}

// Build a FeatureDiscoveryInfo for DLSS-RR queries.
void fillDiscoveryInfo(NVSDK_NGX_FeatureDiscoveryInfo& info,
                       NVSDK_NGX_FeatureCommonInfo& commonInfo) {
    commonInfo = {};
    info = {};
    info.SDKVersion = NVSDK_NGX_Version_API;
    info.FeatureID = NVSDK_NGX_Feature_RayReconstruction;
    info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
    info.Identifier.v.ApplicationId = kNgxApplicationId;
    info.ApplicationDataPath = L"";
    info.FeatureInfo = &commonInfo;
}

std::wstring narrowToWide(const std::string& narrow) {
    return std::wstring(narrow.begin(), narrow.end());
}

// Halton sequence for base b, zero-indexed. Returns value in [0, 1).
static float haltonBase(int index, int base) {
    float result = 0.0f;
    float f = 1.0f;
    int i = index;
    while (i > 0) {
        f /= static_cast<float>(base);
        result += f * static_cast<float>(i % base);
        i /= base;
    }
    return result;
}

// Halton(2,3) jitter centered in [-0.5, 0.5].
static std::pair<float, float> halton23(int index) {
    return {haltonBase(index, 2) - 0.5f, haltonBase(index, 3) - 0.5f};
}

} // namespace

// Static storage for extension name strings (defined outside the class).
std::vector<std::string> DlssAdapter::instanceExtStorage_;
std::vector<std::string> DlssAdapter::deviceExtStorage_;

std::vector<const char*> DlssAdapter::stringStorageToCStrings(const std::vector<std::string>& storage) {
    std::vector<const char*> result;
    result.reserve(storage.size());
    for (const auto& s : storage) {
        result.push_back(s.c_str());
    }
    return result;
}

DlssAdapter::~DlssAdapter() {
    shutdown();
}

std::vector<const char*> DlssAdapter::getRequiredInstanceExtensions() {
    if (!instanceExtStorage_.empty()) {
        return stringStorageToCStrings(instanceExtStorage_);
    }

    NVSDK_NGX_FeatureDiscoveryInfo info;
    NVSDK_NGX_FeatureCommonInfo commonInfo;
    fillDiscoveryInfo(info, commonInfo);

    uint32_t extCount = 0;
    VkExtensionProperties* extProps = nullptr;
    NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
        &info, &extCount, &extProps);

    if (result != NVSDK_NGX_Result_Success) {
        log::warn("dlss", std::string("instance extension query failed: ") + ngxResultToString(result));
        return {};
    }

    instanceExtStorage_.clear();
    instanceExtStorage_.reserve(extCount);
    for (uint32_t i = 0; i < extCount; ++i) {
        instanceExtStorage_.emplace_back(extProps[i].extensionName);
    }

    log::info("dlss", std::string("instance extensions required: ") + std::to_string(extCount));
    for (const auto& n : instanceExtStorage_) {
        log::info("dlss", std::string("  ") + n);
    }

    return stringStorageToCStrings(instanceExtStorage_);
}

std::vector<const char*> DlssAdapter::getRequiredDeviceExtensions(VkInstance instance,
                                                                  VkPhysicalDevice physicalDevice) {
    if (instance == VK_NULL_HANDLE || physicalDevice == VK_NULL_HANDLE) {
        log::warn("dlss", "getRequiredDeviceExtensions: null instance or physical device");
        return {};
    }

    if (!deviceExtStorage_.empty()) {
        return stringStorageToCStrings(deviceExtStorage_);
    }

    NVSDK_NGX_FeatureDiscoveryInfo info;
    NVSDK_NGX_FeatureCommonInfo commonInfo;
    fillDiscoveryInfo(info, commonInfo);

    uint32_t extCount = 0;
    VkExtensionProperties* extProps = nullptr;
    NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
        instance, physicalDevice, &info, &extCount, &extProps);

    if (result != NVSDK_NGX_Result_Success) {
        log::warn("dlss", std::string("device extension query failed: ") + ngxResultToString(result));
        return {};
    }

    deviceExtStorage_.clear();
    deviceExtStorage_.reserve(extCount);
    for (uint32_t i = 0; i < extCount; ++i) {
        deviceExtStorage_.emplace_back(extProps[i].extensionName);
    }

    log::info("dlss", std::string("device extensions required: ") + std::to_string(extCount));
    for (const auto& n : deviceExtStorage_) {
        log::info("dlss", std::string("  ") + n);
    }

    return stringStorageToCStrings(deviceExtStorage_);
}

void DlssAdapter::init(EngineServices& services) {
    if (initialized_) {
        log::warn("dlss", "init called twice");
        return;
    }

    services_ = &services;
    vkDevice_ = services.device().device();
    vkPhysicalDevice_ = services.device().physicalDevice();
    vkInstance_ = services.device().instance();

    // Register GPU profiler slot regardless of DLSS availability
    profileSlot_ = services.metrics().registerTimer("DLSS");

    if (vkDevice_ == VK_NULL_HANDLE || vkPhysicalDevice_ == VK_NULL_HANDLE || vkInstance_ == VK_NULL_HANDLE) {
        log::warn("dlss", "init: device not fully initialized");
        return;
    }

    std::wstring appDataPath = narrowToWide(services.resourceDir());

    NVSDK_NGX_FeatureCommonInfo featureInfo = {};
    featureInfo.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;

    NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_Init(
        kNgxApplicationId,
        appDataPath.c_str(),
        vkInstance_,
        vkPhysicalDevice_,
        vkDevice_,
        vkGetInstanceProcAddr,
        vkGetDeviceProcAddr,
        &featureInfo);

    if (result != NVSDK_NGX_Result_Success) {
        log::warn("dlss", std::string("NVSDK_NGX_VULKAN_Init failed: ") + ngxResultToString(result));
        initialized_ = true;
        return;
    }
    ngxInitialized_ = true;

    result = NVSDK_NGX_VULKAN_GetCapabilityParameters(&ngxParams_);
    if (result != NVSDK_NGX_Result_Success) {
        log::warn("dlss", std::string("GetCapabilityParameters failed: ") + ngxResultToString(result));
        initialized_ = true;
        return;
    }

    // Probe DLSS-RR (Ray Reconstruction) availability on this hardware.
    int rrAvailable = 0;
    result = ngxParams_->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &rrAvailable);
    if (result != NVSDK_NGX_Result_Success || rrAvailable == 0) {
        int needsUpdatedDriver = 0;
        ngxParams_->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver, &needsUpdatedDriver);
        if (needsUpdatedDriver) {
            log::warn("dlss", "DLSS-RR not available: driver needs update");
        } else {
            log::warn("dlss", std::string("DLSS-RR not available: ") + ngxResultToString(result));
        }
        dlssAvailable_ = false;
        initialized_ = true;
        return;
    }

    dlssAvailable_ = true;
    log::info("dlss", "DLSS-RR (Ray Reconstruction) available on this GPU");

    initialized_ = true;
    log::info("dlss", "DlssAdapter initialized (DLSS-RR mode)");
}

void DlssAdapter::configure(const ConfigSnapshot& /*snapshot*/) {
    // Resolution is driven by configureDlss() which is called explicitly from
    // EngineApp after swapchain creation. This method is a hook for future
    // per-frame config updates and is currently a no-op.
}

bool DlssAdapter::configureDlss(uint32_t inputWidth, uint32_t inputHeight,
                                uint32_t outputWidth, uint32_t outputHeight,
                                uint32_t qualityMode) {
    if (!initialized_ || !ngxInitialized_ || !dlssAvailable_) {
        log::warn("dlss", "configureDlss called but DLSS-RR not available");
        return false;
    }

    if (dlssHandle_ != nullptr &&
        inputWidth == inputWidth_ && inputHeight == inputHeight_ &&
        outputWidth == outputWidth_ && outputHeight == outputHeight_) {
        return true;
    }

    releaseFeature();

    inputWidth_ = inputWidth;
    inputHeight_ = inputHeight;
    outputWidth_ = outputWidth;
    outputHeight_ = outputHeight;
    frameCounter_ = 0;
    haltonIndex_ = 0;

    // Create DLSS-RR feature using NVSDK_NGX_DLSSD_Create_Params.
    NVSDK_NGX_DLSSD_Create_Params dlssdParams{};
    dlssdParams.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
    dlssdParams.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;
    dlssdParams.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;
    dlssdParams.InWidth = inputWidth;
    dlssdParams.InHeight = inputHeight;
    dlssdParams.InTargetWidth = outputWidth;
    dlssdParams.InTargetHeight = outputHeight;
    dlssdParams.InFeatureCreateFlags =
        NVSDK_NGX_DLSS_Feature_Flags_IsHDR |
        NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    // Map qualityMode (0=DLAA, 1=Quality, 2=Balanced, 3=Performance, 4=UltraPerformance)
    // to NVSDK_NGX_PerfQuality_Value.
    NVSDK_NGX_PerfQuality_Value perfQuality;
    const char* qualityName;
    switch (qualityMode) {
        case 1:  perfQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality;         qualityName = "Quality";          break;
        case 2:  perfQuality = NVSDK_NGX_PerfQuality_Value_Balanced;           qualityName = "Balanced";         break;
        case 3:  perfQuality = NVSDK_NGX_PerfQuality_Value_MaxPerf;            qualityName = "Performance";      break;
        case 4:  perfQuality = NVSDK_NGX_PerfQuality_Value_UltraPerformance;   qualityName = "UltraPerformance"; break;
        default: perfQuality = NVSDK_NGX_PerfQuality_Value_DLAA;               qualityName = "DLAA";             break;
    }
    dlssdParams.InPerfQualityValue = perfQuality;
    dlssdParams.InEnableOutputSubrects = false;

    // Set render preset on all quality levels to Preset E (latest transformer model).
    ngxParams_->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA,
                    static_cast<int>(NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E));
    ngxParams_->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality,
                    static_cast<int>(NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E));
    ngxParams_->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced,
                    static_cast<int>(NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E));
    ngxParams_->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance,
                    static_cast<int>(NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E));
    ngxParams_->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance,
                    static_cast<int>(NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E));
    ngxParams_->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality,
                    static_cast<int>(NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E));

    VkCommandBuffer oneShotCmd = services_->frame().beginOneShotCommandBuffer();
    if (oneShotCmd == VK_NULL_HANDLE) {
        log::warn("dlss", "configureDlss: failed to allocate one-shot command buffer");
        return false;
    }

    NVSDK_NGX_Result result = NGX_VULKAN_CREATE_DLSSD_EXT1(
        vkDevice_,
        oneShotCmd,
        1 /* creationNodeMask */,
        1 /* visibilityNodeMask */,
        &dlssHandle_,
        ngxParams_,
        &dlssdParams);

    if (result != NVSDK_NGX_Result_Success) {
        log::warn("dlss", std::string("NGX_VULKAN_CREATE_DLSSD_EXT1 failed: ") + ngxResultToString(result));
        services_->frame().endOneShotCommandBuffer(oneShotCmd);
        dlssHandle_ = nullptr;
        return false;
    }

    services_->frame().endOneShotCommandBuffer(oneShotCmd);

    // Initialize jitter for the first frame.
    auto [jx, jy] = halton23(1);
    jitterX_ = jx;
    jitterY_ = jy;
    haltonIndex_ = 1;

    log::info("dlss", std::string("DLSS-RR feature created: ") +
              std::to_string(inputWidth) + "x" + std::to_string(inputHeight) + " -> " +
              std::to_string(outputWidth) + "x" + std::to_string(outputHeight) +
              " (" + qualityName + ", Preset E)");
    return true;
}

void DlssAdapter::releaseFeature() {
    if (dlssHandle_) {
        NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_ReleaseFeature(dlssHandle_);
        if (result != NVSDK_NGX_Result_Success) {
            log::warn("dlss", std::string("ReleaseFeature failed: ") + ngxResultToString(result));
        }
        dlssHandle_ = nullptr;
    }
    // Reset per-feature evaluate failure trackers so reconfigure gets clean logging.
    lastLoggedEvalFailure_  = 0;
    lastLoggedEvalFailure2_ = 0;
}

void DlssAdapter::shutdown() {
    if (!initialized_) {
        return;
    }

    releaseFeature();

    if (ngxParams_) {
        NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_DestroyParameters(ngxParams_);
        if (result != NVSDK_NGX_Result_Success) {
            log::warn("dlss", std::string("DestroyParameters failed: ") + ngxResultToString(result));
        }
        ngxParams_ = nullptr;
    }

    if (ngxInitialized_ && vkDevice_ != VK_NULL_HANDLE) {
        NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_Shutdown1(vkDevice_);
        if (result != NVSDK_NGX_Result_Success) {
            log::warn("dlss", std::string("Shutdown1 failed: ") + ngxResultToString(result));
        }
        ngxInitialized_ = false;
    }

    services_ = nullptr;
    vkDevice_ = VK_NULL_HANDLE;
    vkPhysicalDevice_ = VK_NULL_HANDLE;
    vkInstance_ = VK_NULL_HANDLE;
    inputWidth_ = inputHeight_ = outputWidth_ = outputHeight_ = 0;
    dlssAvailable_ = false;
    initialized_ = false;
    rrPath_ = false;
    log::info("dlss", "DlssAdapter shut down");
}

ResourceHandle DlssAdapter::registerPass(GraphBuilder& builder,
                                         ResourceHandle inputColor,
                                         ResourceHandle motionVector,
                                         ResourceHandle linearDepth) {
    rrPath_ = false;
    inputColorHandle_ = inputColor;
    motionVectorHandle_ = motionVector;
    linearDepthHandle_ = linearDepth;

    ImageResourceDesc outDesc;
    outDesc.name = "dlss_output";
    outDesc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    outDesc.extraUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    // DLSS output is always at display (output) resolution, even when the pool
    // is allocated at render (input) resolution for DLSS upscaling modes.
    outDesc.fixedWidth  = outputWidth_;
    outDesc.fixedHeight = outputHeight_;
    outputColorHandle_ = builder.addResource(outDesc);

    PassDescriptor pass;
    pass.name = "dlss";
    pass.queue = QueueAffinity::Graphics;
    pass.inputs.push_back(inputColor);
    pass.inputs.push_back(motionVector);
    pass.inputs.push_back(linearDepth);
    pass.outputs.push_back(outputColorHandle_);
    pass.execute = [this](const FrameContext& ctx, const PassResources& res) {
        execute(ctx, res);
    };

    builder.addPass(pass);
    return outputColorHandle_;
}

ResourceHandle DlssAdapter::registerPassRR(GraphBuilder& builder,
    ResourceHandle hdrNoisy,
    ResourceHandle diffuseAlbedo,
    ResourceHandle specularAlbedo,
    ResourceHandle normalRoughness,
    ResourceHandle motionVector,
    ResourceHandle linearDepth,
    ResourceHandle specularHitDepth,
    ResourceHandle firstHitDepth,
    ResourceHandle fhDiffuseRayDir,
    ResourceHandle fhSpecularRayDir,
    ResourceHandle reflectionMv,
    ResourceHandle animTexMask,
    ResourceHandle particleMask,
    ResourceHandle fhBaseEmission,
    ResourceHandle biasMask,
    ResourceHandle rtHitDist,
    ResourceHandle motionVec3d,
    ResourceHandle gbufferMetallic,
    ResourceHandle gbufferShadingModelId,
    ResourceHandle gbufferMaterialId,
    ResourceHandle positionViewSpace,
    ResourceHandle transparencyLayer,
    ResourceHandle transparencyOpacity,
    ResourceHandle transparencyMvecs) {

    rrPath_ = true;

    // Store all handles.
    rrHdrNoisy_ = hdrNoisy;
    rrDiffuseAlbedo_ = diffuseAlbedo;
    rrSpecularAlbedo_ = specularAlbedo;
    rrNormalRoughness_ = normalRoughness;
    rrMotionVector_ = motionVector;
    rrLinearDepth_ = linearDepth;
    rrSpecularHitDepth_ = specularHitDepth;
    rrFirstHitDepth_ = firstHitDepth;
    rrDiffuseRayDir_ = fhDiffuseRayDir;
    rrSpecularRayDir_ = fhSpecularRayDir;
    rrReflectionMv_ = reflectionMv;
    rrAnimTexMask_ = animTexMask;
    rrParticleMask_ = particleMask;
    rrFhBaseEmission_ = fhBaseEmission;
    rrBiasMask_ = biasMask;
    rrRtHitDist_ = rtHitDist;
    rrMotionVec3d_ = motionVec3d;
    rrGbufferMetallic_ = gbufferMetallic;
    rrGbufferShadingModelId_ = gbufferShadingModelId;
    rrGbufferMaterialId_ = gbufferMaterialId;
    rrPositionViewSpace_ = positionViewSpace;
    rrTransparencyLayer_ = transparencyLayer;
    rrTransparencyOpacity_ = transparencyOpacity;
    rrTransparencyMvecs_ = transparencyMvecs;

    // Also set legacy handles for the common ones (used in fallback code paths).
    inputColorHandle_ = hdrNoisy;
    motionVectorHandle_ = motionVector;
    linearDepthHandle_ = linearDepth;

    // DLSS-RR output.
    ImageResourceDesc outDesc;
    outDesc.name = "dlss_rr_output";
    outDesc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    outDesc.extraUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    // DLSS output is always at display (output) resolution, even when the pool
    // is allocated at render (input) resolution for DLSS upscaling modes.
    // fixedWidth/fixedHeight override the pool's base render resolution.
    outDesc.fixedWidth  = outputWidth_;
    outDesc.fixedHeight = outputHeight_;
    outputColorHandle_ = builder.addResource(outDesc);

    PassDescriptor pass;
    pass.name = "dlss_rr";
    pass.queue = QueueAffinity::Graphics;

    // Declare all 24 guide inputs.
    pass.inputs.push_back(hdrNoisy);
    pass.inputs.push_back(diffuseAlbedo);
    pass.inputs.push_back(specularAlbedo);
    pass.inputs.push_back(normalRoughness);
    pass.inputs.push_back(motionVector);
    pass.inputs.push_back(linearDepth);
    pass.inputs.push_back(specularHitDepth);
    pass.inputs.push_back(firstHitDepth);
    pass.inputs.push_back(fhDiffuseRayDir);
    pass.inputs.push_back(fhSpecularRayDir);
    pass.inputs.push_back(reflectionMv);
    pass.inputs.push_back(animTexMask);
    pass.inputs.push_back(particleMask);
    pass.inputs.push_back(fhBaseEmission);
    pass.inputs.push_back(biasMask);
    pass.inputs.push_back(rtHitDist);
    pass.inputs.push_back(motionVec3d);
    pass.inputs.push_back(gbufferMetallic);
    pass.inputs.push_back(gbufferShadingModelId);
    pass.inputs.push_back(gbufferMaterialId);
    pass.inputs.push_back(positionViewSpace);
    pass.inputs.push_back(transparencyLayer);
    pass.inputs.push_back(transparencyOpacity);
    pass.inputs.push_back(transparencyMvecs);

    pass.outputs.push_back(outputColorHandle_);
    pass.execute = [this](const FrameContext& ctx, const PassResources& res) {
        execute(ctx, res);
    };

    builder.addPass(pass);
    return outputColorHandle_;
}

void DlssAdapter::execute(const FrameContext& ctx, const PassResources& resources) {
    if (!initialized_ || !dlssAvailable_ || dlssHandle_ == nullptr) {
        return;
    }

    if (resources.resolved == nullptr || resources.cmd == VK_NULL_HANDLE) {
        return;
    }

    // Begin per-pass GPU timer
    if (profileSlot_ != UINT32_MAX)
        services_->metrics().beginNamedTimer(resources.cmd, ctx.frameIndex, profileSlot_);

    // Helper to wrap a VkImage+VkImageView as an NVSDK_NGX_Resource_VK.
    auto makeResource = [](VkImage img, VkImageView view, VkFormat format,
                           uint32_t w, uint32_t h, bool readWrite) {
        NVSDK_NGX_Resource_VK r = {};
        r.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW;
        r.Resource.ImageViewInfo.ImageView = view;
        r.Resource.ImageViewInfo.Image = img;
        r.Resource.ImageViewInfo.SubresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        r.Resource.ImageViewInfo.SubresourceRange.baseMipLevel = 0;
        r.Resource.ImageViewInfo.SubresourceRange.levelCount = 1;
        r.Resource.ImageViewInfo.SubresourceRange.baseArrayLayer = 0;
        r.Resource.ImageViewInfo.SubresourceRange.layerCount = 1;
        r.Resource.ImageViewInfo.Format = format;
        r.Resource.ImageViewInfo.Width = w;
        r.Resource.ImageViewInfo.Height = h;
        r.ReadWrite = readWrite;
        return r;
    };

    // Helper to resolve a ResourceHandle into an NGX resource (returns nullptr VK resource if handle is invalid).
    auto resolveHandle = [&](ResourceHandle h, VkFormat fmt, bool rw = false) -> NVSDK_NGX_Resource_VK {
        if (!h.valid() || h.index >= resources.resolved->count) {
            NVSDK_NGX_Resource_VK null = {};
            return null;
        }
        return makeResource(
            resources.resolved->images[h.index],
            resources.resolved->views[h.index],
            fmt, inputWidth_, inputHeight_, rw);
    };

    // Snapshot camera matrices from the FrameContext for DLSS-RR.
    std::memcpy(worldToView_, ctx.camera.view, sizeof(worldToView_));
    std::memcpy(viewToClip_, ctx.camera.projection, sizeof(viewToClip_));

    if (rrPath_) {
        // ---- DLSS-RR (Ray Reconstruction) evaluate path ----

        NVSDK_NGX_Resource_VK inColorRes = resolveHandle(rrHdrNoisy_, VK_FORMAT_R16G16B16A16_SFLOAT);
        NVSDK_NGX_Resource_VK outColorRes = resolveHandle(outputColorHandle_, VK_FORMAT_R16G16B16A16_SFLOAT, true);
        NVSDK_NGX_Resource_VK inDepthRes = resolveHandle(rrLinearDepth_, VK_FORMAT_R32_SFLOAT);
        NVSDK_NGX_Resource_VK inMotionRes = resolveHandle(rrMotionVector_, VK_FORMAT_R16G16_SFLOAT);
        NVSDK_NGX_Resource_VK inDiffuseAlbedoRes = resolveHandle(rrDiffuseAlbedo_, VK_FORMAT_R16G16B16A16_SFLOAT);
        NVSDK_NGX_Resource_VK inSpecularAlbedoRes = resolveHandle(rrSpecularAlbedo_, VK_FORMAT_R16G16B16A16_SFLOAT);
        NVSDK_NGX_Resource_VK inNormalRoughnessRes = resolveHandle(rrNormalRoughness_, VK_FORMAT_R16G16B16A16_SFLOAT);
        NVSDK_NGX_Resource_VK inSpecHitDepthRes = resolveHandle(rrSpecularHitDepth_, VK_FORMAT_R32_SFLOAT);
        NVSDK_NGX_Resource_VK inFirstHitDepthRes = resolveHandle(rrFirstHitDepth_, VK_FORMAT_R32_SFLOAT);
        NVSDK_NGX_Resource_VK inDiffRayDirRes = resolveHandle(rrDiffuseRayDir_, VK_FORMAT_R16G16B16A16_SFLOAT);
        NVSDK_NGX_Resource_VK inSpecRayDirRes = resolveHandle(rrSpecularRayDir_, VK_FORMAT_R16G16B16A16_SFLOAT);
        NVSDK_NGX_Resource_VK inReflMvRes = resolveHandle(rrReflectionMv_, VK_FORMAT_R16G16_SFLOAT);
        NVSDK_NGX_Resource_VK inAnimTexMaskRes = resolveHandle(rrAnimTexMask_, VK_FORMAT_R8_UNORM);
        NVSDK_NGX_Resource_VK inParticleMaskRes = resolveHandle(rrParticleMask_, VK_FORMAT_R8_UNORM);
        NVSDK_NGX_Resource_VK inEmissionRes = resolveHandle(rrFhBaseEmission_, VK_FORMAT_R16G16B16A16_SFLOAT);
        NVSDK_NGX_Resource_VK inBiasMaskRes = resolveHandle(rrBiasMask_, VK_FORMAT_R8_UNORM);
        NVSDK_NGX_Resource_VK inRtHitDistRes = resolveHandle(rrRtHitDist_, VK_FORMAT_R16_SFLOAT);
        NVSDK_NGX_Resource_VK inMotion3dRes = resolveHandle(rrMotionVec3d_, VK_FORMAT_R16G16B16A16_SFLOAT);
        NVSDK_NGX_Resource_VK inMetallicRes = resolveHandle(rrGbufferMetallic_, VK_FORMAT_R8_UNORM);
        NVSDK_NGX_Resource_VK inShadingModelRes = resolveHandle(rrGbufferShadingModelId_, VK_FORMAT_R8_UINT);
        NVSDK_NGX_Resource_VK inMaterialIdRes = resolveHandle(rrGbufferMaterialId_, VK_FORMAT_R8_UINT);
        NVSDK_NGX_Resource_VK inPosViewRes = resolveHandle(rrPositionViewSpace_, VK_FORMAT_R16G16B16A16_SFLOAT);
        NVSDK_NGX_Resource_VK inTransLayerRes = resolveHandle(rrTransparencyLayer_, VK_FORMAT_R16G16B16A16_SFLOAT);
        NVSDK_NGX_Resource_VK inTransOpacityRes = resolveHandle(rrTransparencyOpacity_, VK_FORMAT_R8_UNORM);
        NVSDK_NGX_Resource_VK inTransMvecsRes = resolveHandle(rrTransparencyMvecs_, VK_FORMAT_R16G16_SFLOAT);

        NVSDK_NGX_VK_DLSSD_Eval_Params evalParams = {};

        // Core inputs (required).
        evalParams.pInColor = &inColorRes;
        evalParams.pInOutput = &outColorRes;
        evalParams.pInDepth = &inDepthRes;
        evalParams.pInMotionVectors = &inMotionRes;

        // RR-specific demodulated inputs.
        evalParams.pInDiffuseAlbedo = &inDiffuseAlbedoRes;
        evalParams.pInSpecularAlbedo = &inSpecularAlbedoRes;
        evalParams.pInNormals = &inNormalRoughnessRes;
        // Roughness packed in normals.w (NVSDK_NGX_DLSS_Roughness_Mode_Packed), so no separate roughness.
        evalParams.pInRoughness = nullptr;

        // Jitter (DLSS expects negated jitter offsets in pixel space).
        evalParams.InJitterOffsetX = -jitterX_;
        evalParams.InJitterOffsetY = -jitterY_;

        evalParams.InRenderSubrectDimensions.Width = inputWidth_;
        evalParams.InRenderSubrectDimensions.Height = inputHeight_;

        evalParams.InReset = (frameCounter_ == 0) ? 1 : 0;

        evalParams.InMVScaleX = 1.0f;
        evalParams.InMVScaleY = 1.0f;

        // Pre-exposure: auto-exposure feedback from ToneMappingAdapter (1-frame delayed).
        // RT output is multiplied by preExposure; DLSS-RR undoes it via InExposureScale = 1/E.
        evalParams.InPreExposure = preExposure_;
        evalParams.InExposureScale = 1.0f / preExposure_;

        // Frame delta for temporal pacing.
        evalParams.InFrameTimeDeltaInMsec = ctx.deltaTime * 1000.0f;

        // World-to-view and view-to-clip matrices.
        evalParams.pInWorldToViewMatrix = worldToView_;
        evalParams.pInViewToClipMatrix = viewToClip_;

        // G-buffer surface attributes via the GBufferSurface array.
        evalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_ALBEDO] = &inDiffuseAlbedoRes;
        evalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_ROUGHNESS] = &inNormalRoughnessRes;
        evalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_METALLIC] = &inMetallicRes;
        evalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_SPECULAR] = &inSpecularAlbedoRes;
        evalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_NORMALS] = &inNormalRoughnessRes;
        evalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_SHADINGMODELID] = &inShadingModelRes;
        evalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_MATERIALID] = &inMaterialIdRes;
        evalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_EMISSIVE] = &inEmissionRes;

        // Extended guide buffers.
        evalParams.pInMotionVectors3D = &inMotion3dRes;
        evalParams.pInIsParticleMask = &inParticleMaskRes;
        evalParams.pInAnimatedTextureMask = &inAnimTexMaskRes;
        evalParams.pInDepthHighRes = &inFirstHitDepthRes;
        evalParams.pInPositionViewSpace = &inPosViewRes;
        evalParams.pInRayTracingHitDistance = &inRtHitDistRes;
        evalParams.pInMotionVectorsReflections = &inReflMvRes;
        evalParams.pInBiasCurrentColorMask = &inBiasMaskRes;

        // Specular hit distance via diffuse/specular hit distance slots.
        evalParams.pInSpecularHitDistance = &inSpecHitDepthRes;

        // Ray direction guides.
        evalParams.pInDiffuseRayDirection = &inDiffRayDirRes;
        evalParams.pInSpecularRayDirection = &inSpecRayDirRes;

        // Transparency layer inputs.
        evalParams.pInTransparencyLayer = &inTransLayerRes;
        evalParams.pInTransparencyLayerOpacity = &inTransOpacityRes;
        evalParams.pInTransparencyLayerMvecs = &inTransMvecsRes;

        // Tone mapper type: none (our input is linear HDR, not tone-mapped).
        evalParams.InToneMapperType = NVSDK_NGX_TONEMAPPER_STRING;

        nvInsertCheckpoint(resources.cmd,
                           ctx.config && (ctx.config->data.diagFlags & DiagFlags::CRASH) != 0,
                           services_->device().caps().nvCheckpoints,
                           "DLSS_START");
        NVSDK_NGX_Result result = NGX_VULKAN_EVALUATE_DLSSD_EXT(
            resources.cmd,
            dlssHandle_,
            ngxParams_,
            &evalParams);

        if (result != NVSDK_NGX_Result_Success) {
            // Use instance member (not static) so the suppression resets on reconfigure.
            if (static_cast<int>(result) != lastLoggedEvalFailure_) {
                log::warn("dlss", std::string("NGX_VULKAN_EVALUATE_DLSSD_EXT failed: ") + ngxResultToString(result));
                lastLoggedEvalFailure_ = static_cast<int>(result);
            }
        }
    } else {
        // ---- Legacy DLSS-SR evaluate path (backward compatibility) ----

        VkImage inColorImg = resources.resolved->images[inputColorHandle_.index];
        VkImage inMotionImg = resources.resolved->images[motionVectorHandle_.index];
        VkImage inDepthImg = resources.resolved->images[linearDepthHandle_.index];
        VkImage outColorImg = resources.resolved->images[outputColorHandle_.index];
        VkImageView inColorView = resources.resolved->views[inputColorHandle_.index];
        VkImageView inMotionView = resources.resolved->views[motionVectorHandle_.index];
        VkImageView inDepthView = resources.resolved->views[linearDepthHandle_.index];
        VkImageView outColorView = resources.resolved->views[outputColorHandle_.index];

        if (inColorImg == VK_NULL_HANDLE || outColorImg == VK_NULL_HANDLE) {
            return;
        }

        NVSDK_NGX_Resource_VK inColorRes = makeResource(
            inColorImg, inColorView, VK_FORMAT_R16G16B16A16_SFLOAT, inputWidth_, inputHeight_, false);
        NVSDK_NGX_Resource_VK inMotionRes = makeResource(
            inMotionImg, inMotionView, VK_FORMAT_R16G16_SFLOAT, inputWidth_, inputHeight_, false);
        NVSDK_NGX_Resource_VK inDepthRes = makeResource(
            inDepthImg, inDepthView, VK_FORMAT_R32_SFLOAT, inputWidth_, inputHeight_, false);
        NVSDK_NGX_Resource_VK outColorRes = makeResource(
            outColorImg, outColorView, VK_FORMAT_R16G16B16A16_SFLOAT, outputWidth_, outputHeight_, true);

        NVSDK_NGX_VK_DLSS_Eval_Params evalParams = {};
        evalParams.Feature.pInColor = &inColorRes;
        evalParams.Feature.pInOutput = &outColorRes;
        evalParams.pInDepth = &inDepthRes;
        evalParams.pInMotionVectors = &inMotionRes;
        evalParams.InJitterOffsetX = 0.0f;
        evalParams.InJitterOffsetY = 0.0f;
        evalParams.InRenderSubrectDimensions.Width = inputWidth_;
        evalParams.InRenderSubrectDimensions.Height = inputHeight_;
        evalParams.InReset = (frameCounter_ == 0) ? 1 : 0;
        evalParams.InMVScaleX = 1.0f;
        evalParams.InMVScaleY = 1.0f;
        evalParams.InFrameTimeDeltaInMsec = ctx.deltaTime * 1000.0f;

        nvInsertCheckpoint(resources.cmd,
                           ctx.config && (ctx.config->data.diagFlags & DiagFlags::CRASH) != 0,
                           services_->device().caps().nvCheckpoints,
                           "DLSS_START");
        NVSDK_NGX_Result result = NGX_VULKAN_EVALUATE_DLSS_EXT(
            resources.cmd,
            dlssHandle_,
            ngxParams_,
            &evalParams);

        if (result != NVSDK_NGX_Result_Success) {
            // Use instance member (not static) so the suppression resets on reconfigure.
            if (static_cast<int>(result) != lastLoggedEvalFailure2_) {
                log::warn("dlss", std::string("NGX_VULKAN_EVALUATE_DLSS_EXT failed: ") + ngxResultToString(result));
                lastLoggedEvalFailure2_ = static_cast<int>(result);
            }
        }
    }

    // Advance Halton jitter for the next frame.
    ++frameCounter_;
    ++haltonIndex_;
    auto [jx, jy] = halton23(haltonIndex_);
    jitterX_ = jx;
    jitterY_ = jy;

    // Cache the output image for FrameGenAdapter HUD-less tagging.
    // The output handle is valid in both SR and RR paths.
    if (outputColorHandle_.valid() && resources.resolved &&
        outputColorHandle_.index < resources.resolved->count) {
        lastHudlessVkImage_ = resources.resolved->images[outputColorHandle_.index];
    }

    // End per-pass GPU timer
    if (profileSlot_ != UINT32_MAX)
        services_->metrics().endNamedTimer(resources.cmd, ctx.frameIndex, profileSlot_);
}

} // namespace engine
