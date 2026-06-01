#pragma once

#include "core/vulkan/all_core_vulkan.hpp"
#include "core/render/render_framework.hpp"
#include <memory>
#include <string>

/// Manages DLSS-G frame generation: sets SL constants and tags resources each frame.
/// Not a WorldModule — operates at the framework/present level, not the pipeline level.
class FrameGenManager {
  public:
    /// Initialize DLSS-G if hardware supports it.
    /// Returns true if DLSS-G is available and ready to be enabled.
    static bool init();

    /// Enable or disable frame generation. Triggers swapchain recreation if needed.
    /// @param mode 0=Off, 1=On, 2=Auto
    /// @param numFramesToGenerate 1=2x, 2=3x, 3=4x (clamped to hardware max)
    static void setMode(uint32_t mode, uint32_t numFramesToGenerate);

    /// Called every frame after world render completes and before present.
    /// Sets SL common constants (camera matrices) and tags depth/MV/HUD-less resources.
    /// DLSS-G diffs HUDless vs backbuffer to identify UI pixels automatically.
    static void tagFrame(std::shared_ptr<FrameworkContext> context,
                         std::shared_ptr<vk::DeviceLocalImage> worldOutput);

    /// Capture the DLSS-G input-processing completion fence for a presented frame slot.
    static void captureInputCompletion(uint32_t frameSlot, const char *source);

    /// Wait for Streamline to finish processing tagged resources before reusing the slot.
    static bool waitForInputCompletion(uint32_t frameSlot, VkDevice device, uint64_t timeoutNs,
                                       uint64_t *elapsedUs = nullptr);

    /// Wait for all captured DLSS-G input-processing fences before swapchain/resource teardown.
    static bool waitForAllInputCompletions(VkDevice device, uint64_t timeoutNs);

    /// Flat diagnostics string for debug snapshots and the in-game debug menu.
    static std::string latencyDiagnostics();

    /// Query whether DLSS-G is currently active (enabled + hardware supports it).
    static bool isActive();

    /// Query hardware max frames capability (0 if unsupported).
    static uint32_t maxFramesToGenerate();

    /// Must be called before swapchain recreation when toggling DLSS-G.
    static void beforeSwapchainRecreate();

    /// Must be called after swapchain recreation when toggling DLSS-G.
    static void afterSwapchainRecreate();

    /// Whether the DLSS-G Streamline feature is currently loaded (hooks active).
    static bool isFeatureLoaded() { return featureLoaded_; }

    /// Current recreate generation counter, incremented before each swapchain recreate.
    static uint32_t recreateGeneration() { return recreateGeneration_; }

    /// Whether a swapchain recreation is currently in progress.
    static bool isRecreateInProgress() { return recreateInProgress_; }

  private:
    static bool initialized_;
    static bool active_;
    static uint32_t maxFrames_;
    static uint32_t currentMode_;
    static bool deferredActivation_;  // eOn deferred until shouldRender() becomes true
    static bool featureLoaded_;       // Whether sl::kFeatureDLSS_G has been loaded
    static bool recreateInProgress_;  // Blocks tagFrame activation during Framework::recreate()
    static uint32_t recreateGeneration_;
};
