#pragma once

#include "core/vulkan/all_core_vulkan.hpp"
#include "core/render/render_framework.hpp"
#include <memory>

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
    /// Sets SL common constants (camera matrices) and tags depth/MV/HUD-less/UI resources.
    static void tagFrame(std::shared_ptr<FrameworkContext> context,
                         std::shared_ptr<vk::DeviceLocalImage> worldOutput,
                         std::shared_ptr<vk::DeviceLocalImage> overlayOutput);

    /// Query whether DLSS-G is currently active (enabled + hardware supports it).
    static bool isActive();

    /// Query hardware max frames capability (0 if unsupported).
    static uint32_t maxFramesToGenerate();

    /// Must be called before swapchain recreation when toggling DLSS-G.
    static void beforeSwapchainRecreate();

    /// Must be called after swapchain recreation when toggling DLSS-G.
    static void afterSwapchainRecreate();

  private:
    static bool initialized_;
    static bool active_;
    static uint32_t maxFrames_;
    static uint32_t currentMode_;
    static bool needsSwapchainRecreate_;
    static bool pendingEnable_;  // Deferred slDLSSGSetOptions after swapchain recreate
};
