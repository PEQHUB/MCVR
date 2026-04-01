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
#include <GFSDK_Aftermath_GpuCrashDumpDecoding.h>

static std::filesystem::path s_logsDir;
static std::mutex s_dumpMutex;
static std::string s_lastDumpPath;
static bool s_initialized = false;

// Dynamically loaded function pointers
static HMODULE s_aftermathModule = nullptr;
static PFN_GFSDK_Aftermath_EnableGpuCrashDumps s_pfnEnable = nullptr;
static PFN_GFSDK_Aftermath_DisableGpuCrashDumps s_pfnDisable = nullptr;
static PFN_GFSDK_Aftermath_GetCrashDumpStatus s_pfnGetStatus = nullptr;

// Decoder function pointers
static PFN_GFSDK_Aftermath_GpuCrashDump_CreateDecoder s_pfnCreateDecoder = nullptr;
static PFN_GFSDK_Aftermath_GpuCrashDump_DestroyDecoder s_pfnDestroyDecoder = nullptr;
static PFN_GFSDK_Aftermath_GpuCrashDump_GenerateJSON s_pfnGenerateJSON = nullptr;
static PFN_GFSDK_Aftermath_GpuCrashDump_GetJSON s_pfnGetJSON = nullptr;
static PFN_GFSDK_Aftermath_GpuCrashDump_GetPageFaultInfo s_pfnGetPageFault = nullptr;
static PFN_GFSDK_Aftermath_GpuCrashDump_GetPageFaultResourceInfo s_pfnGetPageFaultResources = nullptr;
static PFN_GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfoCount s_pfnGetShaderCount = nullptr;
static PFN_GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfo s_pfnGetShaderInfo = nullptr;
static PFN_GFSDK_Aftermath_GetShaderHashForShaderInfo s_pfnGetShaderHash = nullptr;

// Decode a crash dump in-place and write a JSON report alongside the binary dump.
static void decodeCrashDump(const void* pGpuCrashDump, uint32_t gpuCrashDumpSize,
                             const std::filesystem::path& jsonPath) {
    if (!s_pfnCreateDecoder || !s_pfnGenerateJSON || !s_pfnGetJSON || !s_pfnDestroyDecoder) {
        std::cerr << "[Aftermath] Decoder functions not available — skipping decode" << std::endl;
        return;
    }

    GFSDK_Aftermath_GpuCrashDump_Decoder decoder = nullptr;
    auto result = s_pfnCreateDecoder(GFSDK_Aftermath_Version_API,
                                      pGpuCrashDump, gpuCrashDumpSize, &decoder);
    if (result != GFSDK_Aftermath_Result_Success || !decoder) {
        std::cerr << "[Aftermath] CreateDecoder failed: " << result << std::endl;
        return;
    }

    // Query page fault info directly and log to stderr for immediate visibility
    if (s_pfnGetPageFault) {
        GFSDK_Aftermath_GpuCrashDump_PageFaultInfo pf{};
        if (s_pfnGetPageFault(decoder, &pf) == GFSDK_Aftermath_Result_Success) {
            std::cerr << "[Aftermath] PAGE FAULT: VA=0x" << std::hex << pf.faultingGpuVA
                      << " faultType=" << std::dec << pf.faultType
                      << " accessType=" << pf.accessType
                      << " engine=" << pf.engine
                      << " client=" << pf.client
                      << " resourceCount=" << pf.resourceInfoCount << std::endl;

            // Query resource info for the faulting address
            if (pf.resourceInfoCount > 0 && s_pfnGetPageFaultResources) {
                std::vector<GFSDK_Aftermath_GpuCrashDump_ResourceInfo> resources(pf.resourceInfoCount);
                if (s_pfnGetPageFaultResources(decoder, pf.resourceInfoCount, resources.data())
                    == GFSDK_Aftermath_Result_Success) {
                    for (uint32_t i = 0; i < pf.resourceInfoCount; i++) {
                        auto &r = resources[i];
                        std::cerr << "[Aftermath]   Resource[" << i << "]: VA=0x" << std::hex << r.gpuVa
                                  << " size=" << std::dec << r.size
                                  << " " << r.width << "x" << r.height << "x" << r.depth
                                  << " format=" << r.format
                                  << " handle=0x" << std::hex << r.apiResource << std::dec << std::endl;
                    }
                }
            }
        }
    }

    // Query active shaders and log
    if (s_pfnGetShaderCount && s_pfnGetShaderInfo) {
        uint32_t shaderCount = 0;
        if (s_pfnGetShaderCount(decoder, &shaderCount) == GFSDK_Aftermath_Result_Success && shaderCount > 0) {
            std::vector<GFSDK_Aftermath_GpuCrashDump_ShaderInfo> shaders(shaderCount);
            if (s_pfnGetShaderInfo(decoder, shaderCount, shaders.data()) == GFSDK_Aftermath_Result_Success) {
                std::cerr << "[Aftermath] ACTIVE SHADERS (" << shaderCount << "):" << std::endl;
                for (uint32_t i = 0; i < shaderCount; i++) {
                    auto &s = shaders[i];
                    std::cerr << "[Aftermath]   [" << i << "] hash=0x" << std::hex << s.shaderHash
                              << " instance=0x" << s.shaderInstance
                              << std::dec << " type=" << s.shaderType
                              << " internal=" << s.isInternal << std::endl;
                }
            }
        }
    }

    // Generate full JSON report (no shader source callbacks — we don't have SPIR-V at crash time)
    uint32_t jsonSize = 0;
    const uint32_t flags = 0x3FFF; // GFSDK_Aftermath_GpuCrashDumpDecoderFlags_ALL_INFO
    result = s_pfnGenerateJSON(decoder, flags, 0, nullptr, nullptr, nullptr, nullptr, &jsonSize);
    if (result == GFSDK_Aftermath_Result_Success && jsonSize > 0) {
        std::vector<char> json(jsonSize);
        result = s_pfnGetJSON(decoder, jsonSize, json.data());
        if (result == GFSDK_Aftermath_Result_Success) {
            std::ofstream jsonFile(jsonPath);
            if (jsonFile.is_open()) {
                jsonFile.write(json.data(), jsonSize - 1); // exclude null terminator
                jsonFile.close();
                std::cerr << "[Aftermath] Decoded report: " << jsonPath.string()
                          << " (" << jsonSize << " bytes)" << std::endl;
            }
        }
    } else {
        std::cerr << "[Aftermath] GenerateJSON failed: " << result << std::endl;
    }

    s_pfnDestroyDecoder(decoder);
}

static void GFSDK_AFTERMATH_CALL gpuCrashDumpCb(
    const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize, void*) {
    std::lock_guard<std::mutex> lock(s_dumpMutex);

    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char timeBuf[64];
    std::strftime(timeBuf, sizeof(timeBuf), "%Y%m%d_%H%M%S", std::localtime(&time));

    std::filesystem::create_directories(s_logsDir);
    auto baseName = std::string("crash_") + timeBuf;
    auto dumpPath = s_logsDir / (baseName + ".nv-gpudmp");
    auto jsonPath = s_logsDir / (baseName + ".json");

    std::ofstream file(dumpPath, std::ios::binary);
    if (file.is_open()) {
        file.write(static_cast<const char*>(pGpuCrashDump), gpuCrashDumpSize);
        file.close();
        std::cerr << "[Aftermath] Crash dump saved: " << dumpPath.string()
                  << " (" << gpuCrashDumpSize << " bytes)" << std::endl;
        s_lastDumpPath = dumpPath.string();

        // Decode the dump inline — extract page fault, active shaders, full JSON report
        decodeCrashDump(pGpuCrashDump, gpuCrashDumpSize, jsonPath);
    } else {
        std::cerr << "[Aftermath] Failed to write: " << dumpPath.string() << std::endl;
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

    // Load decoder functions (optional — decode fails gracefully if unavailable)
    s_pfnCreateDecoder = (PFN_GFSDK_Aftermath_GpuCrashDump_CreateDecoder)
        GetProcAddress(s_aftermathModule, "GFSDK_Aftermath_GpuCrashDump_CreateDecoder");
    s_pfnDestroyDecoder = (PFN_GFSDK_Aftermath_GpuCrashDump_DestroyDecoder)
        GetProcAddress(s_aftermathModule, "GFSDK_Aftermath_GpuCrashDump_DestroyDecoder");
    s_pfnGenerateJSON = (PFN_GFSDK_Aftermath_GpuCrashDump_GenerateJSON)
        GetProcAddress(s_aftermathModule, "GFSDK_Aftermath_GpuCrashDump_GenerateJSON");
    s_pfnGetJSON = (PFN_GFSDK_Aftermath_GpuCrashDump_GetJSON)
        GetProcAddress(s_aftermathModule, "GFSDK_Aftermath_GpuCrashDump_GetJSON");
    s_pfnGetPageFault = (PFN_GFSDK_Aftermath_GpuCrashDump_GetPageFaultInfo)
        GetProcAddress(s_aftermathModule, "GFSDK_Aftermath_GpuCrashDump_GetPageFaultInfo");
    s_pfnGetPageFaultResources = (PFN_GFSDK_Aftermath_GpuCrashDump_GetPageFaultResourceInfo)
        GetProcAddress(s_aftermathModule, "GFSDK_Aftermath_GpuCrashDump_GetPageFaultResourceInfo");
    s_pfnGetShaderCount = (PFN_GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfoCount)
        GetProcAddress(s_aftermathModule, "GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfoCount");
    s_pfnGetShaderInfo = (PFN_GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfo)
        GetProcAddress(s_aftermathModule, "GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfo");
    s_pfnGetShaderHash = (PFN_GFSDK_Aftermath_GetShaderHashForShaderInfo)
        GetProcAddress(s_aftermathModule, "GFSDK_Aftermath_GetShaderHashForShaderInfo");

    if (s_pfnCreateDecoder) {
        std::cout << "[Aftermath] Crash dump decoder available" << std::endl;
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
