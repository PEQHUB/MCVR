// V2 DLSS-G Frame Generation + NVIDIA Reflex adapter.
//
// Key constraints:
//  - ALL StreamlineContext calls guarded by #ifdef _WIN32.
//  - tagFrame() skips when world is not renderable (menu screens, loading).
//  - Two-recreate lifecycle: beforeSwapchainRecreate → afterSwapchainRecreate.
//    First recreate: unload feature (SL hooks detach).
//    Second recreate: load feature (SL hooks attach to new swapchain), then activate.

#include "frame_gen_adapter.hpp"

#include "app/engine_services.hpp"
#include "config/config_service.hpp"
#include "diagnostics/log.hpp"
#include "frame/frame_scheduler.hpp"
#include "platform/vulkan/vk2_device.hpp"

// StreamlineContext is Windows-only; compile the full body only on Win32.
#include "core/render/streamline_context.hpp"

#ifdef _WIN32
#include <sl.h>
#include <sl_dlss_g.h>
#include <sl_reflex.h>
#include <sl_pcl.h>
#endif

#include <volk.h>

#include <cmath>
#include <cstring>

namespace engine {

namespace {

#ifdef _WIN32

/// Map frameGenMultiplier (2-4) to SL numFramesToGenerate (1-3, i.e. multiplier-1).
static uint32_t multiplierToSL(uint32_t m, uint32_t maxFrames) {
    // SL: numFramesToGenerate=1 → 2x, =2 → 3x, =3 → 4x
    uint32_t slVal = (m >= 2) ? (m - 1) : 1;
    if (slVal > maxFrames) slVal = maxFrames;
    if (slVal < 1) slVal = 1;
    return slVal;
}

/// Build an sl::float4x4 from a column-major 4x4 (GLM/engine layout).
/// SL expects row-major, so we transpose by reading [col][row].
static sl::float4x4 toSlMat(const float m[16]) {
    sl::float4x4 out;
    for (int r = 0; r < 4; r++) {
        out[r] = sl::float4(m[r + 0], m[r + 4], m[r + 8], m[r + 12]);
    }
    return out;
}

/// Compute the inverse of a 4x4 column-major matrix into dst.
/// Uses Cramer's rule. Returns false if matrix is singular.
static bool mat4InverseColMajor(const float m[16], float dst[16]) {
    // Cofactor expansion (standard 4x4 inversion via cofactors)
    float a00 = m[0],  a10 = m[1],  a20 = m[2],  a30 = m[3];
    float a01 = m[4],  a11 = m[5],  a21 = m[6],  a31 = m[7];
    float a02 = m[8],  a12 = m[9],  a22 = m[10], a32 = m[11];
    float a03 = m[12], a13 = m[13], a23 = m[14], a33 = m[15];

    float b00 = a00*a11 - a10*a01;
    float b01 = a00*a21 - a20*a01;
    float b02 = a00*a31 - a30*a01;
    float b03 = a10*a21 - a20*a11;
    float b04 = a10*a31 - a30*a11;
    float b05 = a20*a31 - a30*a21;
    float b06 = a02*a13 - a12*a03;
    float b07 = a02*a23 - a22*a03;
    float b08 = a02*a33 - a32*a03;
    float b09 = a12*a23 - a22*a13;
    float b10 = a12*a33 - a32*a13;
    float b11 = a22*a33 - a32*a23;

    float det = b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06;
    if (std::abs(det) < 1e-12f) return false;
    float invDet = 1.0f / det;

    dst[0]  = ( a11*b11 - a21*b10 + a31*b09) * invDet;
    dst[1]  = (-a10*b11 + a20*b10 - a30*b09) * invDet;
    dst[2]  = ( a13*b05 - a23*b04 + a33*b03) * invDet;
    dst[3]  = (-a12*b05 + a22*b04 - a32*b03) * invDet;
    dst[4]  = (-a01*b11 + a21*b08 - a31*b07) * invDet;
    dst[5]  = ( a00*b11 - a20*b08 + a30*b07) * invDet;
    dst[6]  = (-a03*b05 + a23*b02 - a33*b01) * invDet;
    dst[7]  = ( a02*b05 - a22*b02 + a32*b01) * invDet;
    dst[8]  = ( a01*b10 - a11*b08 + a31*b06) * invDet;
    dst[9]  = (-a00*b10 + a10*b08 - a30*b06) * invDet;
    dst[10] = ( a03*b04 - a13*b02 + a33*b00) * invDet;
    dst[11] = (-a02*b04 + a12*b02 - a32*b00) * invDet;
    dst[12] = (-a01*b09 + a11*b07 - a21*b06) * invDet;
    dst[13] = ( a00*b09 - a10*b07 + a20*b06) * invDet;
    dst[14] = (-a03*b03 + a13*b01 - a23*b00) * invDet;
    dst[15] = ( a02*b03 - a12*b01 + a22*b00) * invDet;
    return true;
}

/// Multiply two 4x4 column-major matrices: out = a * b
static void mat4Mul(const float a[16], const float b[16], float out[16]) {
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += a[row + k*4] * b[k + col*4];
            }
            out[row + col*4] = sum;
        }
    }
}

#endif // _WIN32

} // namespace

FrameGenAdapter::~FrameGenAdapter() {
    if (initialized_) shutdown();
}

void FrameGenAdapter::init(EngineServices& services) {
    if (initialized_) return;
    services_ = &services;
    initialized_ = true;

#ifdef _WIN32
    if (!StreamlineContext::isDlssGSupported()) {
        log::info("framegen", "DLSS-G not supported (hardware/driver/plugin)");
        return;
    }

    sl::DLSSGState state{};
    if (StreamlineContext::getDlssGState(state)) {
        maxFrames_ = state.numFramesToGenerateMax;
        log::info("framegen", "DLSS-G supported: maxFramesToGenerate=" +
                  std::to_string(maxFrames_) +
                  " status=" + std::to_string(static_cast<uint32_t>(state.status)));
    } else {
        log::warn("framegen", "DLSS-G state query failed");
    }

    supported_ = (maxFrames_ > 0);
    log::info("framegen", std::string("FrameGenAdapter init: supported=") + (supported_ ? "yes" : "no"));
#endif
}

void FrameGenAdapter::configure(const ConfigSnapshot& config) {
    if (!initialized_) return;

    const auto& cfg = config.data;

#ifdef _WIN32
    if (!supported_) return;

    bool wantActive = cfg.frameGenEnabled;

    // Update DLSS-G mode: changes take effect via swapchain recreate cycle.
    // Mark needRecreate when the desired state differs from current state.
    if (wantActive != (active_ || deferredActivation_)) {
        services_->config().live().needRecreate = true;
        log::info("framegen", std::string("Frame gen desired state changed → needRecreate (want=") +
                  (wantActive ? "on" : "off") + ")");
    }
#endif

    // Apply Reflex settings whenever they change.
    bool reflexChanged = (cfg.reflexEnabled != reflexEnabled_) ||
                         (cfg.reflexBoost   != reflexBoost_)   ||
                         (cfg.maxFpsLimit   != maxFpsLimit_);
    if (reflexChanged) {
        reflexEnabled_ = cfg.reflexEnabled;
        reflexBoost_   = cfg.reflexBoost;
        maxFpsLimit_   = cfg.maxFpsLimit;
        applyReflexSettings(cfg);
    }
}

void FrameGenAdapter::applyReflexSettings(const EngineConfig& cfg) {
#ifdef _WIN32
    if (!StreamlineContext::isAvailable()) return;

    sl::ReflexMode mode = sl::ReflexMode::eOff;
    if (cfg.reflexEnabled) {
        mode = cfg.reflexBoost ? sl::ReflexMode::eLowLatencyWithBoost
                               : sl::ReflexMode::eLowLatency;
    }
    uint32_t frameLimitUs = (cfg.maxFpsLimit > 0) ? (1000000u / cfg.maxFpsLimit) : 0u;
    bool ok = StreamlineContext::setReflexOptions(mode, frameLimitUs);
    log::info("framegen", std::string("Reflex: ") +
              (cfg.reflexEnabled ? (cfg.reflexBoost ? "LowLatency+Boost" : "LowLatency") : "Off") +
              " fpsLimit=" + std::to_string(cfg.maxFpsLimit) +
              (ok ? "" : " [FAILED]"));
#else
    (void)cfg;
#endif
}

void FrameGenAdapter::pclSimulationStart() {
#ifdef _WIN32
    if (StreamlineContext::isAvailable()) {
        StreamlineContext::pclSetMarker(sl::PCLMarker::eSimulationStart);
    }
#endif
}

void FrameGenAdapter::pclRenderStart() {
#ifdef _WIN32
    if (StreamlineContext::isAvailable()) {
        StreamlineContext::pclSetMarker(sl::PCLMarker::eRenderSubmitStart);
    }
#endif
}

void FrameGenAdapter::tagFrame(VkCommandBuffer cmd,
                               const FrameContext& ctx,
                               VkImage depthImage,  uint32_t depthW,  uint32_t depthH,
                               VkImage mvImage,     uint32_t mvW,     uint32_t mvH,
                               VkImage hudlessImage, uint32_t hudlessW, uint32_t hudlessH) {
#ifdef _WIN32
    if (!supported_) return;

    // Always advance the SL frame token once per rendered frame.
    StreamlineContext::advanceFrame();

    // --- Deferred activation: activate when world becomes renderable ---
    bool shouldRender = latestCamera_.valid && services_ != nullptr;

    if (deferredActivation_ && shouldRender) {
        auto cfgSnap = services_->config().snapshot();
        if (cfgSnap) {
            uint32_t multiplier = cfgSnap->data.frameGenMultiplier;
            uint32_t slVal = multiplierToSL(multiplier, maxFrames_);
            StreamlineContext::setDlssGOptions(sl::DLSSGMode::eOn, slVal);
        }
        active_ = true;
        deferredActivation_ = false;
        currentMode_ = 1;
        log::info("framegen", "activated (deferred, world became renderable)");
    }

    if (!active_) return;

    // Auto-pause when world is not renderable (menus, loading screens).
    if (!shouldRender) {
        StreamlineContext::setDlssGOptions(sl::DLSSGMode::eOff, 1);
        active_ = false;
        deferredActivation_ = true;
        log::info("framegen", "auto-paused (world not renderable)");
        return;
    }

    // --- Set SL Common Constants ---
    sl::Constants consts{};

    const float* view = ctx.camera.view;       // column-major 4x4
    const float* proj = ctx.camera.projection; // column-major 4x4

    float viewInv[16];
    float projInv[16];
    if (!mat4InverseColMajor(view, viewInv)) std::memcpy(viewInv, view, 64);
    if (!mat4InverseColMajor(proj, projInv)) std::memcpy(projInv, proj, 64);

    consts.cameraViewToClip = toSlMat(proj);
    consts.clipToCameraView = toSlMat(projInv);

    // clipToPrevClip = prevProj * prevView * viewInv * projInv
    // prevClipToClip = proj * view * prevViewInv * prevProjInv
    if (firstFrame_) {
        std::memcpy(prevView_,    view,    64);
        std::memcpy(prevProj_,    proj,    64);
        std::memcpy(prevViewInv_, viewInv, 64);
        std::memcpy(prevProjInv_, projInv, 64);
        firstFrame_ = false;
    }

    float tmp1[16], tmp2[16], clipToPrevClip[16], prevClipToClip[16];
    // clipToPrevClip = prevProj * (prevView * (viewInv * projInv))
    mat4Mul(viewInv, projInv, tmp1);
    mat4Mul(prevView_, tmp1, tmp2);
    mat4Mul(prevProj_, tmp2, clipToPrevClip);
    // prevClipToClip = proj * (view * (prevViewInv * prevProjInv))
    mat4Mul(prevViewInv_, prevProjInv_, tmp1);
    mat4Mul(view, tmp1, tmp2);
    mat4Mul(proj, tmp2, prevClipToClip);

    consts.clipToPrevClip = toSlMat(clipToPrevClip);
    consts.prevClipToClip = toSlMat(prevClipToClip);

    std::memcpy(prevView_,    view,    64);
    std::memcpy(prevProj_,    proj,    64);
    std::memcpy(prevViewInv_, viewInv, 64);
    std::memcpy(prevProjInv_, projInv, 64);

    // Jitter: not tracked here (DLSS-G doesn't need per-FG-frame jitter;
    // it reads jitter from the tagged motion vectors).
    consts.jitterOffset = sl::float2(0.0f, 0.0f);
    consts.cameraPinholeOffset = sl::float2(0.0f, 0.0f);

    // MV scale: normalize pixel-space MVs to [-1, 1] NDC
    if (mvW > 0 && mvH > 0) {
        consts.mvecScale = sl::float2(1.0f / static_cast<float>(mvW),
                                       1.0f / static_cast<float>(mvH));
    } else {
        consts.mvecScale = sl::float2(1.0f, 1.0f);
    }

    // Camera world-space vectors (from view matrix inverse — column 3 = position,
    // columns 0,1,2 = right, up, -forward in camera space)
    consts.cameraPos   = sl::float3(viewInv[12], viewInv[13], viewInv[14]);
    consts.cameraRight = sl::float3(viewInv[0],  viewInv[1],  viewInv[2]);
    consts.cameraUp    = sl::float3(viewInv[4],  viewInv[5],  viewInv[6]);
    consts.cameraFwd   = sl::float3(-viewInv[8], -viewInv[9], -viewInv[10]);

    // Near/far from projection matrix (column-major: proj[2][2]=m22, proj[3][2]=m32)
    // Using column-major indexing: element [col*4+row]
    float p22 = proj[2*4+2]; // column 2, row 2
    float p32 = proj[3*4+2]; // column 3, row 2
    float cameraNear, cameraFar;
    if (std::abs(p22) < 1e-6f) {
        cameraNear = std::abs(p32);
        cameraFar  = 1e6f;
    } else {
        cameraNear = std::abs(p32 / p22);
        cameraFar  = std::abs(p32 / (p22 + 1.0f));
    }
    consts.cameraNear = cameraNear;
    consts.cameraFar  = cameraFar;

    // FOV and aspect from projection matrix
    float p11 = proj[1*4+1]; // column 1, row 1
    float p00 = proj[0*4+0]; // column 0, row 0
    consts.cameraFOV         = (std::abs(p11) > 1e-6f) ? 2.0f * std::atan(1.0f / std::abs(p11)) : 1.0f;
    consts.cameraAspectRatio = (std::abs(p00) > 1e-6f) ? std::abs(p11 / p00) : 1.0f;

    // Flags
    consts.depthInverted          = sl::Boolean::eFalse;
    consts.cameraMotionIncluded   = sl::Boolean::eTrue;
    consts.motionVectors3D        = sl::Boolean::eFalse;
    consts.reset                  = firstFrame_ ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    consts.motionVectorsDilated   = sl::Boolean::eFalse;
    consts.motionVectorsJittered  = sl::Boolean::eFalse;

    StreamlineContext::setConstants(consts);

    // --- Tag Resources ---
    uint32_t tagCount = 0;
    sl::Resource resources[3] = {};
    sl::ResourceTag tags[3] = {};
    sl::Extent extents[3] = {};

    auto fillResource = [](sl::Resource& res, VkImage img, uint32_t w, uint32_t h) {
        res.type   = sl::ResourceType::eTex2d;
        res.native = reinterpret_cast<void*>(img);
        res.memory = nullptr;       // Not required for tagged-image path
        res.view   = nullptr;
        res.state  = static_cast<uint32_t>(VK_IMAGE_LAYOUT_GENERAL);
        res.width  = w;
        res.height = h;
        // Format is inferred by SL from the swapchain/pipeline state
        res.nativeFormat = 0;
        res.mipLevels  = 1;
        res.arrayLayers = 1;
    };

    if (depthImage != VK_NULL_HANDLE && depthW > 0 && depthH > 0) {
        fillResource(resources[tagCount], depthImage, depthW, depthH);
        extents[tagCount] = {0, 0, depthW, depthH};
        tags[tagCount] = sl::ResourceTag(&resources[tagCount], sl::kBufferTypeDepth,
                                          sl::ResourceLifecycle::eValidUntilPresent,
                                          &extents[tagCount]);
        tagCount++;
    }

    if (mvImage != VK_NULL_HANDLE && mvW > 0 && mvH > 0) {
        fillResource(resources[tagCount], mvImage, mvW, mvH);
        extents[tagCount] = {0, 0, mvW, mvH};
        tags[tagCount] = sl::ResourceTag(&resources[tagCount], sl::kBufferTypeMotionVectors,
                                          sl::ResourceLifecycle::eValidUntilPresent,
                                          &extents[tagCount]);
        tagCount++;
    }

    if (hudlessImage != VK_NULL_HANDLE && hudlessW > 0 && hudlessH > 0) {
        fillResource(resources[tagCount], hudlessImage, hudlessW, hudlessH);
        extents[tagCount] = {0, 0, hudlessW, hudlessH};
        tags[tagCount] = sl::ResourceTag(&resources[tagCount], sl::kBufferTypeHUDLessColor,
                                          sl::ResourceLifecycle::eValidUntilPresent,
                                          &extents[tagCount]);
        tagCount++;
    }

    if (tagCount > 0) {
        StreamlineContext::tagResources(tags, tagCount,
                                        cmd != VK_NULL_HANDLE ? reinterpret_cast<void*>(cmd) : nullptr);
    }

    // PCL: render submit marker (required every frame for Reflex + PCL)
    StreamlineContext::pclSetMarker(sl::PCLMarker::eRenderSubmitEnd);

#else
    (void)cmd; (void)ctx;
    (void)depthImage;  (void)depthW;  (void)depthH;
    (void)mvImage;     (void)mvW;     (void)mvH;
    (void)hudlessImage; (void)hudlessW; (void)hudlessH;
#endif
}

void FrameGenAdapter::beforeSwapchainRecreate() {
#ifdef _WIN32
    if (!initialized_ || !supported_) return;

    // §19.0 from V1: DLSS-G must be eOff before swapchain teardown.
    if (active_) {
        StreamlineContext::setDlssGOptions(sl::DLSSGMode::eOff, 1);
        active_ = false;
        deferredActivation_ = false;
        log::info("framegen", "eOff before swapchain recreate");
    }

    // Unload feature when user disabled FG (avoids "only one swapchain" rejection).
    auto cfgSnap = services_ ? services_->config().snapshot() : nullptr;
    bool wantActive = cfgSnap && cfgSnap->data.frameGenEnabled;
    if (!wantActive && featureLoaded_) {
        StreamlineContext::setFeatureLoaded(sl::kFeatureDLSS_G, false);
        featureLoaded_ = false;
        deferredActivation_ = false;
        log::info("framegen", "feature unloaded");
    }
#endif
}

void FrameGenAdapter::afterSwapchainRecreate() {
#ifdef _WIN32
    if (!initialized_ || !supported_) return;

    auto cfgSnap = services_ ? services_->config().snapshot() : nullptr;
    bool wantActive = cfgSnap && cfgSnap->data.frameGenEnabled;

    // First recreate after enabling FG: load the feature (SL hooks attach).
    // Trigger a SECOND recreate so the hooks are active on the new swapchain.
    if (wantActive && !featureLoaded_) {
        StreamlineContext::setFeatureLoaded(sl::kFeatureDLSS_G, true);
        featureLoaded_ = true;
        StreamlineContext::setDlssGOptions(sl::DLSSGMode::eOff, 1);
        if (services_) services_->config().live().needRecreate = true;
        deferredActivation_ = true;
        log::info("framegen", "feature loaded + initialized, triggering second recreate");
        return;
    }

    // Second recreate (or normal recreate with feature loaded): activate.
    if (wantActive && featureLoaded_) {
        bool worldReady = latestCamera_.valid;
        if (worldReady && cfgSnap) {
            uint32_t multiplier = cfgSnap->data.frameGenMultiplier;
            uint32_t slVal = multiplierToSL(multiplier, maxFrames_);
            StreamlineContext::setDlssGOptions(sl::DLSSGMode::eOn, slVal);
            active_ = true;
            currentMode_ = 1;
            log::info("framegen", "activated after swapchain recreate (multiplier=" +
                      std::to_string(multiplier) + ", slVal=" + std::to_string(slVal) + ")");
        } else {
            deferredActivation_ = true;
            log::info("framegen", "deferred activation (waiting for world)");
        }
    }
#endif
}

void FrameGenAdapter::shutdown() {
    if (!initialized_) return;

#ifdef _WIN32
    if (active_) {
        StreamlineContext::setDlssGOptions(sl::DLSSGMode::eOff, 1);
        active_ = false;
    }
    if (featureLoaded_) {
        StreamlineContext::setFeatureLoaded(sl::kFeatureDLSS_G, false);
        featureLoaded_ = false;
    }
    log::info("framegen", "FrameGenAdapter shut down");
#endif

    services_    = nullptr;
    initialized_ = false;
    supported_   = false;
    firstFrame_  = true;
}

} // namespace engine
