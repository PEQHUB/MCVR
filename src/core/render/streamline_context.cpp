#include "core/render/streamline_context.hpp"

#ifdef _WIN32

#include <Windows.h>
#include <filesystem>
#include <fstream>
#include <iostream>

// ---- static member definitions ----
void *StreamlineContext::interposerModule_ = nullptr;
bool StreamlineContext::initialized_ = false;
bool StreamlineContext::vulkanInfoSet_ = false;
bool StreamlineContext::reflexSupported_ = false;
uint32_t StreamlineContext::frameIndex_ = 0;
sl::FrameToken *StreamlineContext::currentFrameToken_ = nullptr;

std::vector<std::string> StreamlineContext::requiredInstanceExtensions_;
std::vector<std::string> StreamlineContext::requiredDeviceExtensions_;

PFun_slInit *StreamlineContext::pfnSlInit = nullptr;
PFun_slShutdown *StreamlineContext::pfnSlShutdown = nullptr;
PFun_slSetVulkanInfo *StreamlineContext::pfnSlSetVulkanInfo = nullptr;
PFun_slGetFeatureFunction *StreamlineContext::pfnSlGetFeatureFunction = nullptr;
PFun_slGetNewFrameToken *StreamlineContext::pfnSlGetNewFrameToken = nullptr;
PFun_slIsFeatureSupported *StreamlineContext::pfnSlIsFeatureSupported = nullptr;
PFun_slGetFeatureRequirements *StreamlineContext::pfnSlGetFeatureRequirements = nullptr;

PFun_slReflexSetOptions *StreamlineContext::pfnReflexSetOptions = nullptr;
PFun_slReflexSleep *StreamlineContext::pfnReflexSleep = nullptr;
PFun_slReflexGetState *StreamlineContext::pfnReflexGetState = nullptr;

PFun_slPCLSetMarker *StreamlineContext::pfnPCLSetMarker = nullptr;
PFun_slPCLGetState *StreamlineContext::pfnPCLGetState = nullptr;

// ---- file-based logger (stdout doesn't reach Minecraft logs) ----

static std::ofstream &slLogFile() {
    static std::ofstream file;
    if (!file.is_open()) {
        // Write log next to core.dll
        wchar_t modulePath[MAX_PATH];
        HMODULE coreModule = nullptr;
        static const int anchor = 0;
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&anchor), &coreModule);
        GetModuleFileNameW(coreModule, modulePath, MAX_PATH);
        auto logPath = std::filesystem::path(modulePath).parent_path() / "streamline.log";
        file.open(logPath, std::ios::trunc);
    }
    return file;
}

static std::ofstream &slCout() { auto &f = slLogFile(); f << "[Streamline] "; return f; }
static std::ofstream &slCerr() { auto &f = slLogFile(); f << "[Streamline ERROR] "; return f; }

// Callback for Streamline SDK internal log messages (captures the "why" behind error codes)
static void slLogMessageCallback(sl::LogType type, const char *msg) {
    if (!msg) return;
    auto &f = slLogFile();
    switch (type) {
    case sl::LogType::eInfo:  f << "[SL-SDK INFO] " << msg << std::endl; break;
    case sl::LogType::eWarn:  f << "[SL-SDK WARN] " << msg << std::endl; break;
    case sl::LogType::eError: f << "[SL-SDK ERROR] " << msg << std::endl; break;
    default:                  f << "[SL-SDK] " << msg << std::endl; break;
    }
}

// Wide string storage for pathToLogsAndData (must outlive slInit call)
static std::wstring slLogDirPath_;

template <typename T>
static bool loadProc(HMODULE mod, const char *name, T *&out) {
    out = reinterpret_cast<T *>(GetProcAddress(mod, name));
    if (!out) {
        slCerr() << "failed to resolve " << name << std::endl;
        return false;
    }
    return true;
}

// ---- core function loading ----

bool StreamlineContext::loadCoreFunctions() {
    auto mod = static_cast<HMODULE>(interposerModule_);
    bool ok = true;
    ok &= loadProc(mod, "slInit", pfnSlInit);
    ok &= loadProc(mod, "slShutdown", pfnSlShutdown);
    ok &= loadProc(mod, "slSetVulkanInfo", pfnSlSetVulkanInfo);
    ok &= loadProc(mod, "slGetFeatureFunction", pfnSlGetFeatureFunction);
    ok &= loadProc(mod, "slGetNewFrameToken", pfnSlGetNewFrameToken);
    ok &= loadProc(mod, "slGetFeatureRequirements", pfnSlGetFeatureRequirements);
    // slIsFeatureSupported is optional — might not exist in all builds
    loadProc(mod, "slIsFeatureSupported", pfnSlIsFeatureSupported);
    return ok;
}

bool StreamlineContext::loadReflexFunctions() {
    if (!pfnSlGetFeatureFunction) return false;

    auto getFunc = [](sl::Feature feature, const char *name, void *&out) -> bool {
        sl::Result res = pfnSlGetFeatureFunction(feature, name, out);
        if (res != sl::Result::eOk) {
            slCerr() << "failed to load " << name << " (result=" << static_cast<int>(res) << ")" << std::endl;
            return false;
        }
        return true;
    };

    bool ok = true;
    ok &= getFunc(sl::kFeatureReflex, "slReflexSetOptions", reinterpret_cast<void *&>(pfnReflexSetOptions));
    ok &= getFunc(sl::kFeatureReflex, "slReflexSleep", reinterpret_cast<void *&>(pfnReflexSleep));
    ok &= getFunc(sl::kFeatureReflex, "slReflexGetState", reinterpret_cast<void *&>(pfnReflexGetState));

    if (ok) {
        // If all three Reflex function pointers resolved, the plugin is loaded and
        // the hardware/driver is capable.  Don't gate on lowLatencyAvailable here —
        // some drivers only report it after the first slReflexSetOptions call.
        reflexSupported_ = true;
        slCout() << "Reflex available (all function pointers resolved)" << std::endl;
    } else {
        slCout() << "Reflex NOT available (failed to resolve function pointers)" << std::endl;
    }

    return ok;
}

bool StreamlineContext::loadPCLFunctions() {
    if (!pfnSlGetFeatureFunction) return false;

    auto getFunc = [](sl::Feature feature, const char *name, void *&out) -> bool {
        sl::Result res = pfnSlGetFeatureFunction(feature, name, out);
        if (res != sl::Result::eOk) {
            slCerr() << "failed to load " << name << " (result=" << static_cast<int>(res) << ")" << std::endl;
            return false;
        }
        return true;
    };

    bool ok = true;
    ok &= getFunc(sl::kFeaturePCL, "slPCLSetMarker", reinterpret_cast<void *&>(pfnPCLSetMarker));
    ok &= getFunc(sl::kFeaturePCL, "slPCLGetState", reinterpret_cast<void *&>(pfnPCLGetState));

    if (ok) {
        slCout() << "PCL available (all function pointers resolved)" << std::endl;
    } else {
        slCout() << "PCL NOT available (failed to resolve function pointers)" << std::endl;
    }

    return ok;
}

void StreamlineContext::queryFeatureRequirements() {
    if (!pfnSlGetFeatureRequirements) {
        slCout() << "slGetFeatureRequirements not available, skipping" << std::endl;
        return;
    }

    // Query requirements for each feature we loaded
    sl::Feature features[] = {sl::kFeatureReflex, sl::kFeaturePCL};
    for (auto feature : features) {
        sl::FeatureRequirements reqs{};
        sl::Result res = pfnSlGetFeatureRequirements(feature, reqs);
        if (res != sl::Result::eOk) {
            slCout() << "slGetFeatureRequirements for feature " << static_cast<int>(feature)
                     << " failed (result=" << static_cast<int>(res) << ")" << std::endl;
            continue;
        }

        slCout() << "Feature " << static_cast<int>(feature) << " requirements:" << std::endl;

        // Instance extensions
        for (uint32_t i = 0; i < reqs.vkNumInstanceExtensions; i++) {
            std::string ext = reqs.vkInstanceExtensions[i];
            slCout() << "  instance ext: " << ext << std::endl;
            // Avoid duplicates
            bool found = false;
            for (const auto &e : requiredInstanceExtensions_) {
                if (e == ext) { found = true; break; }
            }
            if (!found) requiredInstanceExtensions_.push_back(ext);
        }

        // Device extensions
        for (uint32_t i = 0; i < reqs.vkNumDeviceExtensions; i++) {
            std::string ext = reqs.vkDeviceExtensions[i];
            slCout() << "  device ext: " << ext << std::endl;
            bool found = false;
            for (const auto &e : requiredDeviceExtensions_) {
                if (e == ext) { found = true; break; }
            }
            if (!found) requiredDeviceExtensions_.push_back(ext);
        }

        // Vulkan 1.2/1.3 features
        for (uint32_t i = 0; i < reqs.vkNumFeatures12; i++) {
            slCout() << "  vk1.2 feature: " << reqs.vkFeatures12[i] << std::endl;
        }
        for (uint32_t i = 0; i < reqs.vkNumFeatures13; i++) {
            slCout() << "  vk1.3 feature: " << reqs.vkFeatures13[i] << std::endl;
        }

        slCout() << "  queues: " << reqs.vkNumGraphicsQueuesRequired << " graphics, "
                 << reqs.vkNumComputeQueuesRequired << " compute" << std::endl;

        if (reqs.vkNumInstanceExtensions == 0 && reqs.vkNumDeviceExtensions == 0) {
            slCout() << "  (no additional Vulkan extensions required)" << std::endl;
        }
    }
}

// ---- public API ----

bool StreamlineContext::init(const wchar_t *pluginPath) {
    if (initialized_) return true;

    // 1. Load sl.interposer.dll
    std::wstring interposerPath = std::wstring(pluginPath) + L"\\sl.interposer.dll";
    // Convert wide path to narrow for the log file
    {
        int len = WideCharToMultiByte(CP_UTF8, 0, pluginPath, -1, nullptr, 0, nullptr, nullptr);
        std::string narrowPath(len - 1, '\0');
        WideCharToMultiByte(CP_UTF8, 0, pluginPath, -1, narrowPath.data(), len, nullptr, nullptr);
        slCout() << "plugin path: " << narrowPath << std::endl;
    }
    interposerModule_ = static_cast<void *>(LoadLibraryW(interposerPath.c_str()));
    if (!interposerModule_) {
        DWORD err = GetLastError();
        slCerr() << "LoadLibraryW failed for sl.interposer.dll (error=" << err << ")" << std::endl;
        return false;
    }
    slCout() << "sl.interposer.dll loaded" << std::endl;

    // 2. Load core function pointers
    if (!loadCoreFunctions()) {
        slCerr() << "failed to load core functions from sl.interposer.dll" << std::endl;
        FreeLibrary(static_cast<HMODULE>(interposerModule_));
        interposerModule_ = nullptr;
        return false;
    }

    // 3. Configure preferences
    sl::Preferences pref{};
    pref.showConsole = false;
    pref.logLevel = sl::LogLevel::eVerbose; // TODO: set to eOff after debugging
    pref.logMessageCallback = slLogMessageCallback;
    // Write SL SDK's own log files next to core.dll for additional diagnostics
    {
        wchar_t modPath[MAX_PATH];
        HMODULE coreMod = nullptr;
        static const int anchor2 = 0;
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&anchor2), &coreMod);
        GetModuleFileNameW(coreMod, modPath, MAX_PATH);
        slLogDirPath_ = std::filesystem::path(modPath).parent_path().wstring();
    }
    pref.pathToLogsAndData = slLogDirPath_.c_str();
    pref.renderAPI = sl::RenderAPI::eVulkan;
    // NOTE: Do NOT set eUseManualHooking. We use the interposer model: sl.interposer.dll
    // intercepts vkGetInstanceProcAddr and vkCreateDevice, automatically registering the
    // device and initializing plugins during vkCreateDevice. Manual hooking mode prevents
    // some command-level hooks from working properly.

    // Features to load
    sl::Feature features[] = {sl::kFeatureReflex, sl::kFeaturePCL};
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = 2;

    // Plugin search paths — where to find sl.reflex.dll, sl.pcl.dll, etc.
    pref.pathsToPlugins = &pluginPath;
    pref.numPathsToPlugins = 1;

    // Engine identity
    pref.engine = sl::EngineType::eCustom;
    pref.engineVersion = "1.0.0";
    pref.projectId = "radiance-mcvr";

    // 4. Initialize Streamline
    sl::Result result = pfnSlInit(pref, sl::kSDKVersion);
    if (result != sl::Result::eOk) {
        slCerr() << "slInit failed (result=" << static_cast<int>(result) << ")" << std::endl;
        FreeLibrary(static_cast<HMODULE>(interposerModule_));
        interposerModule_ = nullptr;
        return false;
    }

    initialized_ = true;
    slCout() << "initialized successfully" << std::endl;

    // 5. Query feature requirements for instance/device extension lists
    //    Must be done AFTER slInit but BEFORE vkCreateInstance/vkCreateDevice
    queryFeatureRequirements();

    return true;
}

bool StreamlineContext::setVulkanInfo(void *instance, void *physicalDevice, void *device,
                                     uint32_t graphicsQueueFamily, uint32_t graphicsQueueIndex,
                                     uint32_t computeQueueFamily, uint32_t computeQueueIndex) {
    // Kept for potential future use (e.g., non-interposer mode).
    // With the interposer model, the interposer's vkCreateDevice hook already
    // registers the device and initializes plugins. Use onDeviceCreated() instead.
    if (!initialized_ || !pfnSlSetVulkanInfo) return false;
    if (vulkanInfoSet_) return true;

    sl::VulkanInfo info{};
    info.device = device;
    info.instance = instance;
    info.physicalDevice = physicalDevice;
    info.graphicsQueueIndex = graphicsQueueIndex;
    info.graphicsQueueFamily = graphicsQueueFamily;
    info.computeQueueIndex = computeQueueIndex;
    info.computeQueueFamily = computeQueueFamily;

    sl::Result result = pfnSlSetVulkanInfo(info);
    if (result != sl::Result::eOk) {
        slCerr() << "slSetVulkanInfo failed (result=" << static_cast<int>(result) << ")" << std::endl;
        return false;
    }

    vulkanInfoSet_ = true;
    loadReflexFunctions();
    return true;
}

bool StreamlineContext::onDeviceCreated() {
    if (!initialized_) return false;
    if (vulkanInfoSet_) return true; // already done

    // The interposer's vkCreateDevice hook automatically registers the device
    // and initializes plugins. We don't need to call slSetVulkanInfo.
    // Just mark the device as ready and load feature-specific function pointers.
    vulkanInfoSet_ = true;
    slCout() << "device created — loading feature functions" << std::endl;

    loadReflexFunctions();
    loadPCLFunctions();

    // Per NVIDIA docs: "slReflexSetOptions needs to be called at least once,
    // even when Reflex Low Latency is Off and there is no Reflex UI."
    // Without this initial call, the Reflex pipeline never properly initializes.
    if (reflexSupported_ && pfnReflexSetOptions) {
        sl::ReflexOptions initOpts{};
        initOpts.mode = sl::ReflexMode::eOff;
        initOpts.frameLimitUs = 0;
        sl::Result res = pfnReflexSetOptions(initOpts);
        slCout() << "initial slReflexSetOptions(eOff) result=" << static_cast<int>(res) << std::endl;
    }

    return true;
}

void StreamlineContext::shutdown() {
    if (!initialized_) return;

    // Disable Reflex before shutdown
    if (reflexSupported_ && pfnReflexSetOptions) {
        sl::ReflexOptions opts{};
        opts.mode = sl::ReflexMode::eOff;
        pfnReflexSetOptions(opts);
    }

    if (pfnSlShutdown) {
        pfnSlShutdown();
    }

    if (interposerModule_) {
        // Note: Don't FreeLibrary the interposer — it may have hooked Vulkan calls
        // that are still in flight. Let the process exit handle cleanup.
        interposerModule_ = nullptr;
    }

    initialized_ = false;
    vulkanInfoSet_ = false;
    reflexSupported_ = false;
    currentFrameToken_ = nullptr;
    frameIndex_ = 0;

    slCout() << "shut down" << std::endl;
}

bool StreamlineContext::isAvailable() { return initialized_ && vulkanInfoSet_; }

bool StreamlineContext::isReflexAvailable() { return isAvailable() && reflexSupported_; }

void *StreamlineContext::getVkGetInstanceProcAddr() {
    if (!interposerModule_) return nullptr;
    return reinterpret_cast<void *>(
        GetProcAddress(static_cast<HMODULE>(interposerModule_), "vkGetInstanceProcAddr"));
}

void *StreamlineContext::getVkCreateDevice() {
    if (!interposerModule_) return nullptr;
    return reinterpret_cast<void *>(
        GetProcAddress(static_cast<HMODULE>(interposerModule_), "vkCreateDevice"));
}

void *StreamlineContext::getVkGetDeviceProcAddr() {
    if (!interposerModule_) return nullptr;
    return reinterpret_cast<void *>(
        GetProcAddress(static_cast<HMODULE>(interposerModule_), "vkGetDeviceProcAddr"));
}

const std::vector<std::string> &StreamlineContext::getRequiredInstanceExtensions() {
    return requiredInstanceExtensions_;
}

const std::vector<std::string> &StreamlineContext::getRequiredDeviceExtensions() {
    return requiredDeviceExtensions_;
}

bool StreamlineContext::setReflexOptions(sl::ReflexMode mode, uint32_t frameLimitUs) {
    if (!reflexSupported_ || !pfnReflexSetOptions) return false;

    sl::ReflexOptions options{};
    options.mode = mode;
    options.frameLimitUs = frameLimitUs;

    sl::Result result = pfnReflexSetOptions(options);
    if (result != sl::Result::eOk) {
        slCerr() << "slReflexSetOptions failed (result=" << static_cast<int>(result) << ")" << std::endl;
        return false;
    }

    slCout() << "Reflex mode set to " << static_cast<int>(mode) << std::endl;
    return true;
}

bool StreamlineContext::reflexSleep() {
    if (!reflexSupported_ || !pfnReflexSleep || !currentFrameToken_) return false;
    return pfnReflexSleep(*currentFrameToken_) == sl::Result::eOk;
}

bool StreamlineContext::getReflexState(sl::ReflexState &state) {
    if (!reflexSupported_ || !pfnReflexGetState) return false;
    return pfnReflexGetState(state) == sl::Result::eOk;
}

bool StreamlineContext::pclSetMarker(sl::PCLMarker marker) {
    if (!pfnPCLSetMarker || !currentFrameToken_) return false;
    return pfnPCLSetMarker(marker, *currentFrameToken_) == sl::Result::eOk;
}

void StreamlineContext::advanceFrame() {
    if (!initialized_ || !pfnSlGetNewFrameToken) return;
    frameIndex_++;
    sl::Result res = pfnSlGetNewFrameToken(currentFrameToken_, &frameIndex_);

    // Log diagnostics every 300 frames
    if (frameIndex_ % 300 == 1) {
        slCout() << "frame " << frameIndex_
                 << " | token=" << (currentFrameToken_ ? "valid" : "NULL")
                 << " | getToken result=" << static_cast<int>(res) << std::endl;

        // Periodic Reflex state check
        if (pfnReflexGetState) {
            sl::ReflexState state{};
            sl::Result rres = pfnReflexGetState(state);
            slCout() << "  ReflexState: result=" << static_cast<int>(rres)
                     << " lowLatencyAvailable=" << state.lowLatencyAvailable
                     << " latencyReportAvailable=" << state.latencyReportAvailable
                     << " flashIndicatorDriverControlled=" << state.flashIndicatorDriverControlled
                     << std::endl;

            if (state.latencyReportAvailable) {
                auto &r = state.frameReport[0];
                slCout() << "  frameReport[0]: simStart=" << r.simStartTime
                         << " simEnd=" << r.simEndTime
                         << " renderSubmitStart=" << r.renderSubmitStartTime
                         << " renderSubmitEnd=" << r.renderSubmitEndTime
                         << " presentStart=" << r.presentStartTime
                         << " presentEnd=" << r.presentEndTime
                         << " gpuRenderStart=" << r.gpuRenderStartTime
                         << " gpuRenderEnd=" << r.gpuRenderEndTime
                         << std::endl;
            }
        }
    }
}

sl::FrameToken *StreamlineContext::getCurrentFrameToken() { return currentFrameToken_; }

#endif // _WIN32
