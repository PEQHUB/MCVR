#include "core/render/modules/world/frame_gen/frame_gen_manager.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/world.hpp"
#include "core/render/streamline_context.hpp"
#include "core/render/buffers.hpp"
#include "common/shared.hpp"

#ifdef _WIN32
#include <sl.h>
#include <sl_dlss_g.h>
#endif

#include <cmath>
#include <iostream>

static std::ostream &fgCout() { return std::cout << "[FrameGen] "; }
static std::ostream &fgCerr() { return std::cerr << "[FrameGen ERROR] "; }

#ifdef _WIN32
/// Map Options::frameGenMode (0=Off, 1=On, 2=Auto) to Streamline enum.
static sl::DLSSGMode toSlMode(uint32_t mode) {
    switch (mode) {
    case 1:  return sl::DLSSGMode::eOn;
    case 2:  return sl::DLSSGMode::eAuto;
    default: return sl::DLSSGMode::eOff;
    }
}
#endif

bool FrameGenManager::initialized_ = false;
bool FrameGenManager::active_ = false;
uint32_t FrameGenManager::maxFrames_ = 0;
uint32_t FrameGenManager::currentMode_ = 0;
bool FrameGenManager::deferredActivation_ = false;
bool FrameGenManager::featureLoaded_ = false;
bool FrameGenManager::recreateInProgress_ = false;
uint32_t FrameGenManager::recreateGeneration_ = 0;

bool FrameGenManager::init() {
#ifdef _WIN32
    if (initialized_) return active_;

    if (!StreamlineContext::isDlssGSupported()) {
        fgCout() << "DLSS-G not supported (hardware/driver/plugin)" << std::endl;
        initialized_ = true;
        return false;
    }

    // Query hardware capability
    sl::DLSSGState state{};
    if (StreamlineContext::getDlssGState(state)) {
        maxFrames_ = state.numFramesToGenerateMax;
        fgCout() << "DLSS-G supported: maxFramesToGenerate=" << maxFrames_
                 << " minWidthOrHeight=" << state.minWidthOrHeight
                 << " status=" << static_cast<uint32_t>(state.status) << std::endl;
    } else {
        fgCout() << "DLSS-G state query failed" << std::endl;
    }

    initialized_ = true;
    return maxFrames_ > 0;
#else
    return false;
#endif
}

void FrameGenManager::setMode(uint32_t mode, uint32_t numFramesToGenerate) {
    // Mode changes are applied via beforeSwapchainRecreate/afterSwapchainRecreate
    // on the render thread. This method only updates the intent — the actual
    // slDLSSGSetOptions call happens during swapchain recreation.
    // Safe to call from any thread (only writes to Options, sets needRecreate).
    Renderer::options.frameGenMode = mode;
    Renderer::options.frameGenMultiplier = numFramesToGenerate;
    Renderer::options.frameGenEnabled = (mode != 0);
    Renderer::options.needRecreate = true;
}

void FrameGenManager::tagFrame(std::shared_ptr<FrameworkContext> context,
                               std::shared_ptr<vk::DeviceLocalImage> worldOutput) {
#ifdef _WIN32
    if (!StreamlineContext::isDlssGSupported()) return;
    if (!context) return;

    if (recreateInProgress_) return;

    bool shouldRender = Renderer::instance().world()->shouldRender();

    // Deferred activation: feature is loaded, waiting for shouldRender to go true
    if (deferredActivation_ && shouldRender && !Renderer::options.needRecreate) {
        sl::DLSSGMode slMode = toSlMode(Renderer::options.frameGenMode);
        uint32_t multiplier = Renderer::options.frameGenMultiplier;
        // In Auto mode, give the plugin the full hardware range to dynamically vary
        if (slMode == sl::DLSSGMode::eAuto) multiplier = maxFrames_;
        if (multiplier > maxFrames_) multiplier = maxFrames_;
        if (multiplier < 1) multiplier = 1;
        if (StreamlineContext::setDlssGOptions(slMode, multiplier)) {
            active_ = true;
            deferredActivation_ = false;
            currentMode_ = Renderer::options.frameGenMode;
            fgCout() << "activated (deferred, shouldRender became true)" << std::endl;
        } else {
            Renderer::options.frameGenEnabled = false;
            Renderer::options.frameGenMode = 0;
            deferredActivation_ = false;
            fgCerr() << "deferred activation failed; disabling Frame Generation" << std::endl;
        }
    }

    if (!active_) return;

    // Auto-pause FG when world stops rendering (menu/loading transition)
    // so the SL interposer doesn't process presents with zero tagged resources.
    if (!shouldRender) {
        StreamlineContext::setDlssGOptions(sl::DLSSGMode::eOff, 1);
        active_ = false;
        deferredActivation_ = true;  // will re-activate when shouldRender returns
        fgCout() << "auto-paused (shouldRender=false)" << std::endl;
        return;
    }

    // Don't tag during offline accumulation — generated frames would corrupt the Welford accumulator
    if (Renderer::options.offlineState == 2) return;

    uint32_t frameIndex = context->frameIndex;

    // --- Set SL Common Constants (camera matrices, MV scale, jitter) ---
    auto buffersPtr = Renderer::instance().buffers();
    if (!buffersPtr || !buffersPtr->worldUniformBuffer()) return;
    auto worldUBO = static_cast<vk::Data::WorldUBO *>(buffersPtr->worldUniformBuffer()->mappedPtr());
    if (!worldUBO) return;

    sl::Constants consts{};

    // SL sl::Constants requires row-major matrices. GLM stores column-major.
    // Transpose by reading GLM m[col][row] into SL out[row] = {col0, col1, col2, col3}.
    // This matches NRD's matrix copy pattern (nrd_module.cpp:247-253).
    auto toSlMat = [](const glm::mat4 &m) -> sl::float4x4 {
        sl::float4x4 out;
        for (int r = 0; r < 4; r++)
            out[r] = sl::float4(m[0][r], m[1][r], m[2][r], m[3][r]);
        return out;
    };

    glm::mat4 viewMat = worldUBO->cameraEffectedViewMat;
    glm::mat4 projMat = worldUBO->cameraProjMat;
    glm::mat4 viewMatInv = worldUBO->cameraEffectedViewMatInv;
    glm::mat4 projMatInv = worldUBO->cameraProjMatInv;

    consts.cameraViewToClip = toSlMat(projMat);
    consts.clipToCameraView = toSlMat(projMatInv);

    // clipToPrevClip = prevProj * prevView * currentViewInv * currentProjInv
    // prevClipToClip = currentProj * currentView * prevViewInv * prevProjInv
    // Cache previous frame matrices + inverses to avoid per-frame glm::inverse().
    // Reset to current frame on first call or after recreation (frameCount tracks).
    static glm::mat4 prevViewMat{};
    static glm::mat4 prevProjMat{};
    static glm::mat4 prevViewMatInv{};
    static glm::mat4 prevProjMatInv{};
    static uint32_t lastTaggedFrame = UINT32_MAX;

    bool firstFrame = (lastTaggedFrame == UINT32_MAX) ||
                      (StreamlineContext::getFrameIndex() != lastTaggedFrame + 1);
    if (firstFrame) {
        prevViewMat = viewMat;
        prevProjMat = projMat;
        prevViewMatInv = viewMatInv;
        prevProjMatInv = projMatInv;
    }
    lastTaggedFrame = StreamlineContext::getFrameIndex();

    glm::mat4 clipToPrevClip = prevProjMat * prevViewMat * viewMatInv * projMatInv;
    glm::mat4 prevClipToClip = projMat * viewMat * prevViewMatInv * prevProjMatInv;

    consts.clipToPrevClip = toSlMat(clipToPrevClip);
    consts.prevClipToClip = toSlMat(prevClipToClip);

    prevViewMat = viewMat;
    prevProjMat = projMat;
    prevViewMatInv = viewMatInv;
    prevProjMatInv = projMatInv;

    // Jitter offset (pixel space)
    consts.jitterOffset = sl::float2(worldUBO->cameraJitter.x, worldUBO->cameraJitter.y);

    // Camera pinhole offset — (0,0) for standard centered pinhole projection
    consts.cameraPinholeOffset = sl::float2(0.0f, 0.0f);

    // Motion vector scale: our MVs are in pixel space already, normalize to [-1,1]
    // MV format: screen-space pixel offsets at render resolution
    uint32_t renderWidth = 0, renderHeight = 0;
    if (frameIndex < Renderer::frameGenMotionVectorImages.size() && Renderer::frameGenMotionVectorImages[frameIndex]) {
        renderWidth = Renderer::frameGenMotionVectorImages[frameIndex]->width();
        renderHeight = Renderer::frameGenMotionVectorImages[frameIndex]->height();
    }
    if (renderWidth > 0 && renderHeight > 0) {
        consts.mvecScale = sl::float2(1.0f / static_cast<float>(renderWidth),
                                       1.0f / static_cast<float>(renderHeight));
    } else {
        consts.mvecScale = sl::float2(1.0f, 1.0f);
    }

    // Camera vectors (world space)
    glm::vec3 camPos = glm::vec3(viewMatInv[3]);
    glm::vec3 camRight = glm::vec3(viewMatInv[0]);
    glm::vec3 camUp = glm::vec3(viewMatInv[1]);
    glm::vec3 camFwd = -glm::vec3(viewMatInv[2]); // negative Z is forward in view space

    consts.cameraPos = sl::float3(camPos.x, camPos.y, camPos.z);
    consts.cameraUp = sl::float3(camUp.x, camUp.y, camUp.z);
    consts.cameraRight = sl::float3(camRight.x, camRight.y, camRight.z);
    consts.cameraFwd = sl::float3(camFwd.x, camFwd.y, camFwd.z);

    // Near/far from projection matrix.
    // GLM column-major: projMat[2][2] and projMat[3][2] encode near/far.
    // Reversed-Z infinite far: projMat[2][2] = 0, projMat[3][2] = near.
    // Reversed-Z finite far:   projMat[2][2] = near/(far-near), projMat[3][2] = far*near/(far-near).
    float p22 = projMat[2][2];
    float p32 = projMat[3][2];
    float cameraNear, cameraFar;
    if (std::abs(p22) < 1e-6f) {
        // Infinite far plane (reversed-Z): p32 = near
        cameraNear = p32;
        cameraFar = 1e6f; // SL needs a finite value; use a large stand-in
    } else {
        cameraNear = p32 / p22;
        cameraFar = p32 / (p22 + 1.0f);
    }
    // Ensure positive values (reversed-Z may negate them)
    consts.cameraNear = std::abs(cameraNear);
    consts.cameraFar = std::abs(cameraFar);

    // FOV and aspect ratio from projection matrix (guard against zero/NaN)
    float p11 = projMat[1][1];
    float p00 = projMat[0][0];
    consts.cameraFOV = (std::abs(p11) > 1e-6f) ? 2.0f * std::atan(1.0f / std::abs(p11)) : 1.0f;
    consts.cameraAspectRatio = (std::abs(p00) > 1e-6f) ? std::abs(p11 / p00) : 1.0f;

    // Flags
    consts.depthInverted = sl::Boolean::eFalse;        // Linear depth buffer (0=near, large=far)
    consts.cameraMotionIncluded = sl::Boolean::eTrue;  // MVs include camera motion
    consts.motionVectors3D = sl::Boolean::eFalse;      // 2D screen-space
    consts.reset = (firstFrame || Renderer::resetExposureAdaptation)
                       ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    consts.motionVectorsDilated = sl::Boolean::eFalse;
    consts.motionVectorsJittered = sl::Boolean::eFalse;

    StreamlineContext::setConstants(consts);

    // --- Tag Resources ---
    // NVIDIA vk_streamline sample: tag depth, MV, and HUDless only.
    // Do NOT tag kBufferTypeUIColorAndAlpha — DLSS-G diffs HUDless vs the
    // final backbuffer (after UI composite) to identify UI pixels automatically.

    auto fillResource = [](sl::Resource &res, std::shared_ptr<vk::DeviceLocalImage> img, uint32_t layoutOverride = UINT_MAX) {
        res.type = sl::ResourceType::eTex2d;
        res.native = reinterpret_cast<void *>(img->vkImage());
        res.memory = reinterpret_cast<void *>(img->vkDeviceMemory());
        res.view = reinterpret_cast<void *>(img->vkImageView());
        res.state = (layoutOverride != UINT_MAX) ? layoutOverride : static_cast<uint32_t>(img->imageLayout());
        res.width = img->width();
        res.height = img->height();
        res.nativeFormat = static_cast<uint32_t>(img->vkFormat());
        res.mipLevels = 1;
        res.arrayLayers = 1;
    };

    sl::Resource resources[3] = {};
    sl::ResourceTag tags[3] = {};
    sl::Extent extents[3] = {};
    uint32_t tagCount = 0;

    // Depth (render resolution)
    if (frameIndex < Renderer::frameGenDepthImages.size() && Renderer::frameGenDepthImages[frameIndex]) {
        auto depthImg = Renderer::frameGenDepthImages[frameIndex];
        fillResource(resources[tagCount], depthImg, static_cast<uint32_t>(VK_IMAGE_LAYOUT_GENERAL));
        extents[tagCount] = {0, 0, depthImg->width(), depthImg->height()};
        tags[tagCount] = sl::ResourceTag(&resources[tagCount], sl::kBufferTypeDepth,
                                          sl::ResourceLifecycle::eValidUntilPresent, &extents[tagCount]);
        tagCount++;
    }

    // Motion Vectors (render resolution)
    if (frameIndex < Renderer::frameGenMotionVectorImages.size() && Renderer::frameGenMotionVectorImages[frameIndex]) {
        auto mvImg = Renderer::frameGenMotionVectorImages[frameIndex];
        fillResource(resources[tagCount], mvImg, static_cast<uint32_t>(VK_IMAGE_LAYOUT_GENERAL));
        extents[tagCount] = {0, 0, mvImg->width(), mvImg->height()};
        tags[tagCount] = sl::ResourceTag(&resources[tagCount], sl::kBufferTypeMotionVectors,
                                          sl::ResourceLifecycle::eValidUntilPresent, &extents[tagCount]);
        tagCount++;
    }

    // HUD-less Color (display resolution — world output after tone mapping, before UI).
    // Same format as swapchain (R8G8B8A8_UNORM). Where overlay alpha=0, the SDR
    // composite fast-path writes worldOutput pixels verbatim to the swapchain, so
    // HUDless == backbuffer for all non-UI pixels. DLSS-G diffs to find UI.
    if (worldOutput) {
        fillResource(resources[tagCount], worldOutput);
        extents[tagCount] = {0, 0, worldOutput->width(), worldOutput->height()};
        tags[tagCount] = sl::ResourceTag(&resources[tagCount], sl::kBufferTypeHUDLessColor,
                                          sl::ResourceLifecycle::eValidUntilPresent, &extents[tagCount]);
        tagCount++;
    }

    if (tagCount > 0) {
        void *cmdBuf = context->fuseCommandBuffer
                         ? reinterpret_cast<void *>(context->fuseCommandBuffer->vkCommandBuffer())
                         : nullptr;
        StreamlineContext::tagResources(tags, tagCount, cmdBuf);
    }
#endif
}

bool FrameGenManager::isActive() {
    return active_;
}

uint32_t FrameGenManager::maxFramesToGenerate() {
    return maxFrames_;
}

void FrameGenManager::beforeSwapchainRecreate() {
#ifdef _WIN32
    if (!initialized_ || !StreamlineContext::isDlssGSupported()) return;

    recreateInProgress_ = true;
    recreateGeneration_++;

    bool wantActive = Renderer::options.frameGenEnabled;

    // §19.0: DLSS-G must be eOff before swapchain teardown to prevent deadlocks.
    if (active_) {
        StreamlineContext::setDlssGOptions(sl::DLSSGMode::eOff, 1);
        active_ = false;
        deferredActivation_ = false;
        fgCout() << "eOff before swapchain recreate (gen=" << recreateGeneration_ << ")" << std::endl;
    }

    // Only unload when user explicitly disables FG. Keep feature loaded across
    // normal recreates (resize, alt-tab) so hooks stay active on the swapchain
    // and FG can reactivate immediately without a reload cycle.
    // The "only one swap-chain" Streamline rejection was caused by the DComp overlay
    // compositor (now disabled with wantOverlay=false), not by the feature lifecycle.
    if (!wantActive && featureLoaded_) {
        StreamlineContext::setFeatureLoaded(sl::kFeatureDLSS_G, false);
        featureLoaded_ = false;
        deferredActivation_ = false;
        fgCout() << "feature unloaded" << std::endl;
    }
#endif
}

void FrameGenManager::afterSwapchainRecreate() {
#ifdef _WIN32
    if (!initialized_ || !StreamlineContext::isDlssGSupported()) return;

    bool wantActive = Renderer::options.frameGenEnabled;

    // Feature load happens here — AFTER swapchain reconstruct (new swapchain exists).
    // The old swapchain was destroyed without the feature loaded, avoiding the
    // sl.dlss_g.dll crash. SL hooks now attach to future swapchain operations,
    // so we need one more recreate for hooks to be active on a fresh swapchain.
    if (wantActive && !featureLoaded_) {
        if (!StreamlineContext::setFeatureLoaded(sl::kFeatureDLSS_G, true)) {
            Renderer::options.frameGenEnabled = false;
            Renderer::options.frameGenMode = 0;
            deferredActivation_ = false;
            recreateInProgress_ = false;
            fgCerr() << "feature load failed; disabling Frame Generation" << std::endl;
            return;
        }
        featureLoaded_ = true;
        // Initialize DLSS-G state immediately (eOff) so that the NEXT recreate's
        // vkDestroySwapchainKHR hook won't crash on uninitialized state.
        if (!StreamlineContext::setDlssGOptions(sl::DLSSGMode::eOff, 1)) {
            StreamlineContext::setFeatureLoaded(sl::kFeatureDLSS_G, false);
            featureLoaded_ = false;
            Renderer::options.frameGenEnabled = false;
            Renderer::options.frameGenMode = 0;
            deferredActivation_ = false;
            recreateInProgress_ = false;
            fgCerr() << "initial eOff failed; disabling Frame Generation" << std::endl;
            return;
        }
        Renderer::options.needRecreate = true;
        deferredActivation_ = true;
        fgCout() << "feature loaded + initialized, triggering second recreate (gen="
                 << recreateGeneration_ << ")" << std::endl;
        return;
    }

    if (wantActive && featureLoaded_) {
        // Second recreate (or normal recreate with feature loaded): hooks active.
        if (Renderer::instance().world()->shouldRender()) {
            sl::DLSSGMode slMode = toSlMode(Renderer::options.frameGenMode);
            uint32_t multiplier = Renderer::options.frameGenMultiplier;
            if (slMode == sl::DLSSGMode::eAuto) multiplier = maxFrames_;
            if (multiplier > maxFrames_) multiplier = maxFrames_;
            if (multiplier < 1) multiplier = 1;
            if (StreamlineContext::setDlssGOptions(slMode, multiplier)) {
                active_ = true;
                currentMode_ = Renderer::options.frameGenMode;
                fgCout() << "activated after swapchain recreate (gen="
                         << recreateGeneration_ << ")" << std::endl;
            } else {
                Renderer::options.frameGenEnabled = false;
                Renderer::options.frameGenMode = 0;
                active_ = false;
                deferredActivation_ = false;
                fgCerr() << "activation failed after swapchain recreate; disabling Frame Generation" << std::endl;
            }
        } else {
            deferredActivation_ = true;
            fgCout() << "deferred activation (waiting for shouldRender, gen="
                     << recreateGeneration_ << ")" << std::endl;
        }
    }

    recreateInProgress_ = false;
#endif
}
