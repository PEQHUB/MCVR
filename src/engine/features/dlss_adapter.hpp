#pragma once

// V2 DLSS Ray Reconstruction (DLSS-RR) adapter.
//
// Replaces the temporal denoiser entirely. Consumes the 1spp noisy HDR
// radiance plus all RT G-buffer outputs (25 guide images) and produces a
// denoised+upscaled output via NVSDK_NGX_Feature_RayReconstruction.
//
// Static usage (called before EngineApp::init creates the DeviceService):
//   - getRequiredInstanceExtensions()
//   - getRequiredDeviceExtensions(instance, physicalDevice)
//
// Runtime usage:
//   - init(services)                      — NGX init + RR availability probe.
//   - configureDlss(w, h, outW, outH)     — (re)creates the DLSS-RR feature handle.
//   - registerPassRR(builder, ...25 handles...) — wire into graph (RR path).
//   - registerPass(builder, color, motion, depth) — legacy SR fallback.
//   - execute(ctx, res)                   — called automatically by the graph.
//   - shutdown()                          — must run BEFORE DeviceService::shutdown().
//
// Halton(2,3) jitter is managed internally. After each execute(), currentJitter()
// returns the pixel-space offset for the NEXT frame so the RT adapter can
// inject it into the WorldUBO / projection matrix before tracing.

#include "feature_adapter.hpp"
#include "rendergraph/graph_types.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Forward-declare NGX types to keep NGX headers out of this public header.
struct NVSDK_NGX_Parameter;
struct NVSDK_NGX_Handle;

namespace engine {

class GraphBuilder;

class DlssAdapter : public FeatureAdapter {
public:
    ~DlssAdapter() override;

    // ---- Static NGX queries (no NGX init required) ----

    static std::vector<const char*> getRequiredInstanceExtensions();
    static std::vector<const char*> getRequiredDeviceExtensions(VkInstance instance,
                                                                VkPhysicalDevice physicalDevice);

    // ---- FeatureAdapter overrides ----

    void init(EngineServices& services) override;
    void configure(const ConfigSnapshot& config) override;
    void execute(const FrameContext& ctx, const PassResources& resources) override;
    void shutdown() override;
    const char* name() const override { return "DlssAdapter"; }

    // ---- DLSS-RR-specific API ----

    bool configureDlss(uint32_t inputWidth, uint32_t inputHeight,
                       uint32_t outputWidth, uint32_t outputHeight,
                       uint32_t qualityMode = 0);

    bool isInitialized() const { return initialized_; }
    bool isAvailable() const { return dlssAvailable_; }
    bool isFeatureCreated() const { return dlssHandle_ != nullptr; }

    // DLSS input (render) resolution — the resolution at which RT shaders trace.
    // Valid after a successful configureDlss() call. Used by EngineApp to set
    // the graph resource pool allocation size.
    uint32_t renderInputWidth()  const { return inputWidth_; }
    uint32_t renderInputHeight() const { return inputHeight_; }

    // Legacy SR register path (backward-compatible, kept for fallback).
    ResourceHandle registerPass(GraphBuilder& builder,
                                ResourceHandle inputColor,
                                ResourceHandle motionVector,
                                ResourceHandle linearDepth);

    // Full DLSS-RR register path — all 25 RT guide inputs.
    ResourceHandle registerPassRR(GraphBuilder& builder,
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
        ResourceHandle transparencyMvecs);

    ResourceHandle outputHandle() const { return outputColorHandle_; }

    // Returns the VkImage for the DLSS output (HUD-less color) from the last
    // executed frame. Valid after the first execute() call. Used by FrameGenAdapter
    // to tag the HUD-less resource for DLSS-G without needing graph handle resolution.
    VkImage hudlessVkImage() const { return lastHudlessVkImage_; }
    uint32_t hudlessWidth()  const { return outputWidth_; }
    uint32_t hudlessHeight() const { return outputHeight_; }

    // Jitter for the NEXT frame (pixel-space, centered [-0.5, 0.5]).
    // Call after execute() has advanced the Halton sequence.
    std::pair<float, float> currentJitter() const { return {jitterX_, jitterY_}; }

    // Pre-exposure value from tonemapping adapter (auto-exposure feedback loop).
    // Called each frame before execute(). Default 1.0.
    void setPreExposure(float e) { preExposure_ = e; }

    // Reset DLSS temporal history (forces InReset=1 on the next evaluate call).
    // Must be called on world load to prevent stale menu-screen history (accumulated
    // with zero G-buffers while inWorld_ was false) from bleeding into the first
    // in-world frames and causing white-screen saturation.
    void resetHistory() { frameCounter_ = 0; }

private:
    static std::vector<const char*> stringStorageToCStrings(const std::vector<std::string>& storage);
    void releaseFeature();

    // NGX state
    NVSDK_NGX_Parameter* ngxParams_ = nullptr;
    NVSDK_NGX_Handle* dlssHandle_ = nullptr;

    // Cached services + device handles
    EngineServices* services_ = nullptr;
    VkDevice vkDevice_ = VK_NULL_HANDLE;
    VkPhysicalDevice vkPhysicalDevice_ = VK_NULL_HANDLE;
    VkInstance vkInstance_ = VK_NULL_HANDLE;

    // Configured resolution
    uint32_t inputWidth_ = 0;
    uint32_t inputHeight_ = 0;
    uint32_t outputWidth_ = 0;
    uint32_t outputHeight_ = 0;

    // --- Graph handles (SR legacy path) ---
    ResourceHandle inputColorHandle_;
    ResourceHandle motionVectorHandle_;
    ResourceHandle linearDepthHandle_;
    ResourceHandle outputColorHandle_;

    // --- Graph handles (RR path — all 25 guide inputs) ---
    bool rrPath_ = false;  // true when registerPassRR() was used
    ResourceHandle rrHdrNoisy_;
    ResourceHandle rrDiffuseAlbedo_;
    ResourceHandle rrSpecularAlbedo_;
    ResourceHandle rrNormalRoughness_;
    ResourceHandle rrMotionVector_;
    ResourceHandle rrLinearDepth_;
    ResourceHandle rrSpecularHitDepth_;
    ResourceHandle rrFirstHitDepth_;
    ResourceHandle rrDiffuseRayDir_;
    ResourceHandle rrSpecularRayDir_;
    ResourceHandle rrReflectionMv_;
    ResourceHandle rrAnimTexMask_;
    ResourceHandle rrParticleMask_;
    ResourceHandle rrFhBaseEmission_;
    ResourceHandle rrBiasMask_;
    ResourceHandle rrRtHitDist_;
    ResourceHandle rrMotionVec3d_;
    ResourceHandle rrGbufferMetallic_;
    ResourceHandle rrGbufferShadingModelId_;
    ResourceHandle rrGbufferMaterialId_;
    ResourceHandle rrPositionViewSpace_;
    ResourceHandle rrTransparencyLayer_;
    ResourceHandle rrTransparencyOpacity_;
    ResourceHandle rrTransparencyMvecs_;

    // State flags
    bool initialized_ = false;
    bool ngxInitialized_ = false;
    bool dlssAvailable_ = false;

    // Per-instance NGX evaluate error tracking — reset on releaseFeature() so that a
    // reconfigure or device-reset doesn't silently swallow repeated evaluate failures.
    // Stored as int to avoid pulling NGX headers into this public header; compared
    // against the raw NVSDK_NGX_Result value cast to int.
    int lastLoggedEvalFailure_  = 0;  // 0 == NVSDK_NGX_Result_Success
    int lastLoggedEvalFailure2_ = 0;  // legacy SR path

    // Frame counter + Halton jitter
    uint64_t frameCounter_ = 0;
    int haltonIndex_ = 0;
    float jitterX_ = 0.0f;
    float jitterY_ = 0.0f;
    float preExposure_ = 1.0f;  // auto-exposure feedback, updated per frame

    // Cached HUD-less VkImage (output image) from the most recent execute().
    // Set inside execute() from the resolved resources; safe to read after execute().
    VkImage lastHudlessVkImage_ = VK_NULL_HANDLE;

    // World-to-view and view-to-clip matrices for DLSS-RR (updated from FrameContext)
    float worldToView_[16] = {};
    float viewToClip_[16] = {};

    // Static storage for extension name strings.
    static std::vector<std::string> instanceExtStorage_;
    static std::vector<std::string> deviceExtStorage_;
};

} // namespace engine
