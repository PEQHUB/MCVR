#pragma once

#include <cstdint>
#include <string>
#include <vector>

#ifdef _WIN32

// Include SL headers for type/struct definitions only.
// We never call the SL_API-declared functions directly — all loaded via GetProcAddress.
// The inline helpers (slReflexSetOptions etc.) in sl_reflex.h reference slGetFeatureFunction,
// but since we never call those inlines, the linker won't try to resolve the symbol.
#include <sl.h>
#include <sl_pcl.h>
#include <sl_reflex.h>

class StreamlineContext {
  public:
    /// Initialize Streamline — load sl.interposer.dll, call slInit().
    /// Must be called BEFORE any Vulkan initialization (before volkInitialize).
    /// @param pluginPath directory containing sl.*.dll files
    static bool init(const wchar_t *pluginPath);

    /// After VkDevice creation — provide Vulkan info to Streamline.
    /// Only needed in non-interposer mode. With interposer, use onDeviceCreated().
    static bool setVulkanInfo(void *instance, void *physicalDevice, void *device,
                              uint32_t graphicsQueueFamily, uint32_t graphicsQueueIndex,
                              uint32_t computeQueueFamily, uint32_t computeQueueIndex);

    /// Call after vkCreateDevice when using the interposer model.
    /// The interposer already registered the device; this just loads feature functions.
    static bool onDeviceCreated();

    /// Shutdown Streamline — call before destroying Vulkan device.
    static void shutdown();

    // --- Query ---
    static bool isAvailable();       ///< Was SL successfully initialized?
    static bool isReflexAvailable(); ///< Is sl.reflex plugin loaded and hardware supports it?

    /// Get vkGetInstanceProcAddr from sl.interposer.dll (for volkInitializeCustom).
    /// Returns nullptr if interposer not loaded.
    static void *getVkGetInstanceProcAddr();

    /// Get vkCreateDevice from sl.interposer.dll.
    /// Must be used instead of volk's vkCreateDevice so the interposer tracks the device.
    static void *getVkCreateDevice();

    /// Get vkGetDeviceProcAddr from sl.interposer.dll.
    /// After volkLoadDevice, use this to re-override mandatory SL hooks.
    static void *getVkGetDeviceProcAddr();

    // --- Feature requirements (call after init, before instance/device creation) ---
    static const std::vector<std::string> &getRequiredInstanceExtensions();
    static const std::vector<std::string> &getRequiredDeviceExtensions();

    // --- Reflex API ---
    static bool setReflexOptions(sl::ReflexMode mode, uint32_t frameLimitUs = 0);
    static bool reflexSleep();
    static bool getReflexState(sl::ReflexState &state);

    // --- PCL (PC Latency) API ---
    /// Set a PCL marker for the current frame token.
    /// Must be called every frame regardless of Reflex mode (on/off).
    static bool pclSetMarker(sl::PCLMarker marker);

    // --- Frame management ---
    static void advanceFrame(); ///< Increments frame counter + gets new SL frame token
    static sl::FrameToken *getCurrentFrameToken();

  private:
    static void *interposerModule_; // HMODULE
    static bool initialized_;
    static bool vulkanInfoSet_;
    static bool reflexSupported_;
    static uint32_t frameIndex_;
    static sl::FrameToken *currentFrameToken_;

    // Required Vulkan extensions (populated by slGetFeatureRequirements after slInit)
    static std::vector<std::string> requiredInstanceExtensions_;
    static std::vector<std::string> requiredDeviceExtensions_;

    // Core SL function pointers (loaded via GetProcAddress from sl.interposer.dll)
    static PFun_slInit *pfnSlInit;
    static PFun_slShutdown *pfnSlShutdown;
    static PFun_slSetVulkanInfo *pfnSlSetVulkanInfo;
    static PFun_slGetFeatureFunction *pfnSlGetFeatureFunction;
    static PFun_slGetNewFrameToken *pfnSlGetNewFrameToken;
    static PFun_slIsFeatureSupported *pfnSlIsFeatureSupported;
    static PFun_slGetFeatureRequirements *pfnSlGetFeatureRequirements;

    // Reflex function pointers (loaded via slGetFeatureFunction after device set)
    static PFun_slReflexSetOptions *pfnReflexSetOptions;
    static PFun_slReflexSleep *pfnReflexSleep;
    static PFun_slReflexGetState *pfnReflexGetState;

    // PCL function pointers (loaded via slGetFeatureFunction after device set)
    static PFun_slPCLSetMarker *pfnPCLSetMarker;
    static PFun_slPCLGetState *pfnPCLGetState;

    static bool loadCoreFunctions();
    static bool loadReflexFunctions();
    static bool loadPCLFunctions();
    static void queryFeatureRequirements();
};

#else // Linux — Streamline is Windows-only; provide no-op stubs

class StreamlineContext {
  public:
    static bool init(const wchar_t *) { return false; }
    static bool setVulkanInfo(void *, void *, void *, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
    static bool onDeviceCreated() { return false; }
    static void shutdown() {}
    static bool isAvailable() { return false; }
    static bool isReflexAvailable() { return false; }
    static void *getVkGetInstanceProcAddr() { return nullptr; }
    static void *getVkCreateDevice() { return nullptr; }
    static void *getVkGetDeviceProcAddr() { return nullptr; }
    static const std::vector<std::string> &getRequiredInstanceExtensions() {
        static const std::vector<std::string> empty;
        return empty;
    }
    static const std::vector<std::string> &getRequiredDeviceExtensions() {
        static const std::vector<std::string> empty;
        return empty;
    }
    static void advanceFrame() {}
    static bool reflexSleep() { return false; }
};

#endif // _WIN32
