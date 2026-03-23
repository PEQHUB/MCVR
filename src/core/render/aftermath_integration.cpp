#include "aftermath_integration.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// Aftermath SDK types — loaded dynamically to avoid hard DLL dependency
#include <GFSDK_Aftermath_Defines.h>
#include <GFSDK_Aftermath_GpuCrashDump.h>

static std::filesystem::path s_logsDir;
static std::mutex s_dumpMutex;
static std::string s_lastDumpPath;
static bool s_initialized = false;

// Dynamically loaded function pointers
static HMODULE s_aftermathModule = nullptr;
static PFN_GFSDK_Aftermath_EnableGpuCrashDumps s_pfnEnable = nullptr;
static PFN_GFSDK_Aftermath_DisableGpuCrashDumps s_pfnDisable = nullptr;
static PFN_GFSDK_Aftermath_GetCrashDumpStatus s_pfnGetStatus = nullptr;

static void GFSDK_AFTERMATH_CALL gpuCrashDumpCb(
    const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize, void*) {
    std::lock_guard<std::mutex> lock(s_dumpMutex);

    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char timeBuf[64];
    std::strftime(timeBuf, sizeof(timeBuf), "%Y%m%d_%H%M%S", std::localtime(&time));

    std::filesystem::create_directories(s_logsDir);
    auto path = s_logsDir / (std::string("crash_") + timeBuf + ".nv-gpudmp");

    std::ofstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.write(static_cast<const char*>(pGpuCrashDump), gpuCrashDumpSize);
        file.close();
        std::cerr << "[Aftermath] Crash dump saved: " << path.string()
                  << " (" << gpuCrashDumpSize << " bytes)" << std::endl;
        s_lastDumpPath = path.string();
    } else {
        std::cerr << "[Aftermath] Failed to write: " << path.string() << std::endl;
    }
}

static void GFSDK_AFTERMATH_CALL shaderDebugInfoCb(
    const void*, const uint32_t, void*) {
    // Shader debug info collected by driver — included in crash dump automatically
}

static void GFSDK_AFTERMATH_CALL descriptionCb(
    PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addValue, void*) {
    addValue(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, "RadSER/MCVR");
    addValue(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationVersion, "displacement");
}

static void GFSDK_AFTERMATH_CALL resolveMarkerCb(
    const void*, const uint32_t, void*, void** out, uint32_t* outSize) {
    *out = nullptr;
    *outSize = 0;
}

bool AftermathIntegration::init(const std::filesystem::path& logsDir) {
    s_logsDir = logsDir;
    s_lastDumpPath.clear();

    // Try to load Aftermath DLL dynamically
    // Search in: radiance dir (next to core.dll), then system PATH
    auto dllPath = logsDir.parent_path() / "GFSDK_Aftermath_Lib.x64.dll";
    s_aftermathModule = LoadLibraryA(dllPath.string().c_str());
    if (!s_aftermathModule) {
        s_aftermathModule = LoadLibraryA("GFSDK_Aftermath_Lib.x64.dll");
    }
    if (!s_aftermathModule) {
        std::cerr << "[Aftermath] DLL not found — crash dumps unavailable" << std::endl;
        return false;
    }

    s_pfnEnable = (PFN_GFSDK_Aftermath_EnableGpuCrashDumps)
        GetProcAddress(s_aftermathModule, "GFSDK_Aftermath_EnableGpuCrashDumps");
    s_pfnDisable = (PFN_GFSDK_Aftermath_DisableGpuCrashDumps)
        GetProcAddress(s_aftermathModule, "GFSDK_Aftermath_DisableGpuCrashDumps");
    s_pfnGetStatus = (PFN_GFSDK_Aftermath_GetCrashDumpStatus)
        GetProcAddress(s_aftermathModule, "GFSDK_Aftermath_GetCrashDumpStatus");

    if (!s_pfnEnable || !s_pfnDisable || !s_pfnGetStatus) {
        std::cerr << "[Aftermath] Failed to resolve SDK functions" << std::endl;
        FreeLibrary(s_aftermathModule);
        s_aftermathModule = nullptr;
        return false;
    }

    GFSDK_Aftermath_Result result = s_pfnEnable(
        GFSDK_Aftermath_Version_API,
        GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan,
        GFSDK_Aftermath_GpuCrashDumpFeatureFlags_DeferDebugInfoCallbacks,
        gpuCrashDumpCb, shaderDebugInfoCb, descriptionCb, resolveMarkerCb, nullptr);

    if (result == GFSDK_Aftermath_Result_Success) {
        std::cout << "[Aftermath] GPU crash dump collection ENABLED" << std::endl;
        s_initialized = true;
        return true;
    } else {
        std::cerr << "[Aftermath] Enable failed: result=" << result << std::endl;
        return false;
    }
}

std::string AftermathIntegration::waitForCrashDump(int timeoutMs) {
    if (!s_initialized || !s_pfnGetStatus) return "";

    auto start = std::chrono::steady_clock::now();
    GFSDK_Aftermath_CrashDump_Status status = GFSDK_Aftermath_CrashDump_Status_Unknown;

    while (true) {
        s_pfnGetStatus(&status);
        if (status == GFSDK_Aftermath_CrashDump_Status_Finished ||
            status == GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed)
            break;
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >= timeoutMs) {
            std::cerr << "[Aftermath] Timeout (status=" << status << ")" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::lock_guard<std::mutex> lock(s_dumpMutex);
    return s_lastDumpPath;
}

void AftermathIntegration::shutdown() {
    if (s_initialized && s_pfnDisable) {
        s_pfnDisable();
        s_initialized = false;
    }
    if (s_aftermathModule) {
        FreeLibrary(s_aftermathModule);
        s_aftermathModule = nullptr;
    }
}
