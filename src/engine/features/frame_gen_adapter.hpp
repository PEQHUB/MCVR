#pragma once

// V2 DLSS Frame Generation (DLSS-G) + NVIDIA Reflex adapter.
//
// Wraps StreamlineContext to provide DLSS-G frame generation via the SL interposer.
// Streamline intercepts Vulkan at the swapchain level so generated frames are
// presented automatically without any changes to the render graph.
//
// Lifecycle:
//   1. init(services) — probe DLSS-G support, query hardware maxFrames.
//   2. configure(config) — enable/disable FG and Reflex from live config.
//   3. setCameraData(cam) — call each frame from processScene() before tagFrame.
//   4. tagFrame(cmd, ctx, depthImg, mvImg, hudlessImg) — set SL constants + tag resources.
//   5. beforeSwapchainRecreate() / afterSwapchainRecreate() — two-recreate lifecycle.
//   6. shutdown() — unload SL feature, safe to call even if not initialized.
//
// All Streamline calls are guarded by #ifdef _WIN32. On Linux, everything is a no-op.

#include "feature_adapter.hpp"
#include "frame/frame_context.hpp"

#include <cstdint>
#include <vulkan/vulkan.h>

namespace engine {

class EngineServices;

class FrameGenAdapter : public FeatureAdapter {
public:
    FrameGenAdapter() = default;
    ~FrameGenAdapter() override;

    // ---- FeatureAdapter overrides ----
    void init(EngineServices& services) override;
    void configure(const ConfigSnapshot& config) override;
    void execute(const FrameContext& ctx, const PassResources& resources) override {}
    void shutdown() override;
    const char* name() const override { return "FrameGenAdapter"; }

    // ---- Frame Gen API ----

    /// Supply current frame camera data before tagFrame().
    /// Called each frame from EngineApp::processScene() (or its caller).
    void setCameraData(const CameraData& cam) { latestCamera_ = cam; }

    /// Tag depth, motion vector, and HUD-less color resources for DLSS-G.
    /// Must be called after all graph passes complete but before present.
    /// @param cmd   Active command buffer (may be VK_NULL_HANDLE).
    /// @param ctx   Current frame context.
    /// @param depthImage   VkImage of the linear depth buffer (render res).
    /// @param mvImage      VkImage of the motion vector buffer (render res).
    /// @param hudlessImage VkImage of the HUD-less color output (display res).
    void tagFrame(VkCommandBuffer cmd,
                  const FrameContext& ctx,
                  VkImage depthImage,  uint32_t depthW,  uint32_t depthH,
                  VkImage mvImage,     uint32_t mvW,     uint32_t mvH,
                  VkImage hudlessImage, uint32_t hudlessW, uint32_t hudlessH);

    /// Must be called before swapchain recreation when toggling DLSS-G.
    void beforeSwapchainRecreate();

    /// Must be called after swapchain recreation when toggling DLSS-G.
    void afterSwapchainRecreate();

    bool isSupported() const { return supported_; }
    bool isActive()    const { return active_; }
    uint32_t maxFramesToGenerate() const { return maxFrames_; }

    // Reflex: call once per frame regardless of Reflex on/off state.
    void pclSimulationStart();
    void pclRenderStart();

private:
    void applyReflexSettings(const EngineConfig& cfg);

    EngineServices* services_ = nullptr;
    CameraData latestCamera_;

    bool initialized_        = false;
    bool supported_          = false;  // hardware supports DLSS-G
    bool active_             = false;  // DLSS-G currently generating frames
    bool featureLoaded_      = false;  // sl::kFeatureDLSS_G loaded
    bool deferredActivation_ = false;  // waiting for world to be renderable
    uint32_t maxFrames_      = 0;
    uint32_t currentMode_    = 0;

    // Track previous camera matrices for SL clipToPrevClip computation
    float prevView_[16] = {};
    float prevProj_[16] = {};
    float prevViewInv_[16] = {};
    float prevProjInv_[16] = {};
    bool firstFrame_         = true;

    // Reflex
    bool reflexEnabled_ = false;
    bool reflexBoost_   = false;
    uint32_t maxFpsLimit_ = 0;
};

} // namespace engine
