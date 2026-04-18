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
#include <sl_dlss_g.h>

class StreamlineContext {
  public:
    /// Initialize Streamline — load sl.interposer.dll, call slInit().
    /// Must be called BEFORE any Vulkan initialization (before volkInitialize).
    /// @param pluginPath directory containing sl.*.dll files
    static bool init(const wchar_t *pluginPath);

    /// Call after vkCreateDevice when using the interposer model.
    /// The interposer already registered the device; this just loads feature functions.
    static bool onDeviceCreated();

    /// Shutdown Streamline — call before destroying Vulkan device.
    static void shutdown();

    // --- Query ---
    static bool isAvailable();       ///< Was SL successfully initialized?
    static bool isReflexAvailable(); ///< Is sl.reflex plugin loaded and hardware supports it?
    static bool isDlssGSupported();  ///< Is DLSS-G plugin loaded and hardware supports it?

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

    // --- DLSS-G (Frame Generation) API ---
    /// Set DLSS-G options (mode, numFramesToGenerate, etc.)
    static bool setDlssGOptions(sl::DLSSGMode mode, uint32_t numFramesToGenerate = 1);
    /// Query current DLSS-G state (status, VRAM usage, max frames, etc.)
    static bool getDlssGState(sl::DLSSGState &state);
    /// Set common constants (camera matrices, MV scale, jitter) for current frame.
    static bool setConstants(const sl::Constants &consts);
    /// Tag a resource for the current frame.
    /// @param tags Array of ResourceTag structs describing tagged resources.
    /// @param numTags Number of tags in the array.
    /// @param cmdBuffer Optional command buffer (can be nullptr for eValidUntilPresent).
    static bool tagResources(const sl::ResourceTag *tags, uint32_t numTags, void *cmdBuffer = nullptr);
    /// Load or unload a Streamline feature at runtime (for swapchain recreation on DLSS-G toggle).
    static bool setFeatureLoaded(sl::Feature feature, bool loaded);

    // --- Frame management ---
    static void advanceFrame(); ///< Increments frame counter + gets new SL frame token
    static sl::FrameToken *getCurrentFrameToken();
    static uint32_t getFrameIndex(); ///< Current frame index

  private:
    static void *interposerModule_; // HMODULE
    static bool initialized_;
    static bool vulkanInfoSet_;
    static bool reflexSupported_;
    static bool dlssGSupported_;
    static uint32_t frameIndex_;
    static sl::FrameToken *currentFrameToken_;

    // Required Vulkan extensions (populated by slGetFeatureRequirements after slInit)
    static std::vector<std::string> requiredInstanceExtensions_;
    static std::vector<std::string> requiredDeviceExtensions_;

    // Core SL function pointers (loaded via GetProcAddress from sl.interposer.dll)
    static PFun_slInit *pfnSlInit;
    static PFun_slShutdown *pfnSlShutdown;
    static PFun_slGetFeatureFunction *pfnSlGetFeatureFunction;
    static PFun_slGetNewFrameToken *pfnSlGetNewFrameToken;
    static PFun_slGetFeatureRequirements *pfnSlGetFeatureRequirements;

    // Reflex function pointers (loaded via slGetFeatureFunction after device set)
    static PFun_slReflexSetOptions *pfnReflexSetOptions;
    static PFun_slReflexSleep *pfnReflexSleep;
    static PFun_slReflexGetState *pfnReflexGetState;

    // PCL function pointers (loaded via slGetFeatureFunction after device set)
    static PFun_slPCLSetMarker *pfnPCLSetMarker;
    static PFun_slPCLGetState *pfnPCLGetState;

    // DLSS-G function pointers (loaded via slGetFeatureFunction after device set)
    static PFun_slDLSSGSetOptions *pfnDLSSGSetOptions;
    static PFun_slDLSSGGetState *pfnDLSSGGetState;

    // Core SL functions needed for DLSS-G resource tagging and constants
    static PFun_slSetConstants *pfnSlSetConstants;
    static PFun_slSetTagForFrame *pfnSlSetTagForFrame;
    static PFun_slSetFeatureLoaded *pfnSlSetFeatureLoaded;

    static bool loadCoreFunctions();
    static bool loadReflexFunctions();
    static bool loadPCLFunctions();
    static bool loadDlssGFunctions();
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
    static bool isDlssGSupported() { return false; }
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
    static uint32_t getFrameIndex() { return 0; }
    // Reflex/PCL stubs
    static bool setReflexOptions(int /*mode*/, uint32_t /*frameLimitUs*/ = 0) { return false; }
    static bool pclSetMarker(int /*marker*/) { return false; }
    // DLSS-G stubs
    static bool setDlssGOptions(int /*mode*/, uint32_t /*numFramesToGenerate*/ = 1) { return false; }
    static bool getDlssGState(void * /*state*/) { return false; }
    static bool setConstants(const void * /*consts*/) { return false; }
    static bool tagResources(const void * /*tags*/, uint32_t /*numTags*/, void * /*cmdBuffer*/ = nullptr) { return false; }
    static bool setFeatureLoaded(int /*feature*/, bool /*loaded*/) { return false; }
    static void *getCurrentFrameToken() { return nullptr; }
};

#endif // _WIN32
