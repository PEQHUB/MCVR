#include "com_radiance_client_proxy_vulkan_RendererProxy.h"

#include "core/all_extern.hpp"
#include "core/build/build_info.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/entities.hpp"
#include "core/render/modules/ui_module.hpp"
#include "core/render/modules/world/frame_gen/frame_gen_manager.hpp"
#include "core/render/modules/world/ray_tracing/ray_tracing_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/overlay_compositor.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/textures.hpp"
#include "core/render/world.hpp"
#include "core/vulkan/vma.hpp"

#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#    include <windows.h>
#    include <bcrypt.h>
using DYNLIB_HANDLE = HMODULE;

static DYNLIB_HANDLE try_get_loaded_handle(const wchar_t *wname) {
    return GetModuleHandleW(wname);
}

static FARPROC getproc(DYNLIB_HANDLE h, const char *sym) {
    FARPROC p = GetProcAddress(h, sym);
    if (!p) {
        std::cerr << "GetProcAddress failed: " << sym << std::endl;
        std::abort();
    }
    return p;
}

#elif defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#    include <dlfcn.h>
using DYNLIB_HANDLE = void *;

static DYNLIB_HANDLE try_get_loaded_handle(const char *name) {
    return dlopen(name, RTLD_NOW | RTLD_NOLOAD);
}

static void *getproc(DYNLIB_HANDLE h, const char *sym) {
    void *p = dlsym(h, sym);
    if (!p) {
        std::cerr << "dlsym failed: " << sym << " — " << dlerror() << std::endl;
        std::abort();
    }
    return p;
}

#else
#    error "Unsupported platform"
#endif

namespace {
std::recursive_mutex g_rendererJniMtx;
std::atomic<bool> g_rendererShuttingDown{false};
std::atomic<bool> g_rendererClosed{false};

std::string json_escape(const char *text) {
    std::string out;
    if (text == nullptr) return out;
    for (const char c : std::string(text)) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += c; break;
        }
    }
    return out;
}

std::string json_escape(const std::string& text) {
    return json_escape(text.c_str());
}

struct FileIdentity {
    std::string path;
    std::string sha256 = "unavailable";
    uint64_t sizeBytes = 0;
    std::string error;
};

#if defined(_WIN32)
std::string utf8FromWide(const std::wstring& text) {
    if (text.empty()) return {};
    int size = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()),
                                   nullptr, 0, nullptr, nullptr);
    if (size <= 0) return {};
    std::string out(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()),
                        out.data(), size, nullptr, nullptr);
    return out;
}

std::string sha256File(const std::filesystem::path& path, std::string& error) {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD objectLength = 0;
    DWORD bytesWritten = 0;
    std::vector<UCHAR> hashObject;
    std::array<UCHAR, 32> digest{};

    auto cleanup = [&]() {
        if (hash) {
            BCryptDestroyHash(hash);
        }
        if (algorithm) {
            BCryptCloseAlgorithmProvider(algorithm, 0);
        }
    };

    NTSTATUS status = BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0);
    if (!BCRYPT_SUCCESS(status)) {
        error = "BCryptOpenAlgorithmProvider failed";
        cleanup();
        return "unavailable";
    }
    status = BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                               reinterpret_cast<PUCHAR>(&objectLength),
                               sizeof(objectLength), &bytesWritten, 0);
    if (!BCRYPT_SUCCESS(status) || objectLength == 0) {
        error = "BCryptGetProperty(BCRYPT_OBJECT_LENGTH) failed";
        cleanup();
        return "unavailable";
    }
    hashObject.resize(objectLength);
    status = BCryptCreateHash(algorithm, &hash, hashObject.data(), objectLength, nullptr, 0, 0);
    if (!BCRYPT_SUCCESS(status)) {
        error = "BCryptCreateHash failed";
        cleanup();
        return "unavailable";
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "failed to open DLL for hashing";
        cleanup();
        return "unavailable";
    }

    std::array<char, 1024 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize read = input.gcount();
        if (read > 0) {
            status = BCryptHashData(hash, reinterpret_cast<PUCHAR>(buffer.data()),
                                    static_cast<ULONG>(read), 0);
            if (!BCRYPT_SUCCESS(status)) {
                error = "BCryptHashData failed";
                cleanup();
                return "unavailable";
            }
        }
    }
    if (input.bad()) {
        error = "DLL read failed";
        cleanup();
        return "unavailable";
    }
    status = BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0);
    if (!BCRYPT_SUCCESS(status)) {
        error = "BCryptFinishHash failed";
        cleanup();
        return "unavailable";
    }

    std::ostringstream out;
    out << std::uppercase << std::hex << std::setfill('0');
    for (const auto byte : digest) {
        out << std::setw(2) << static_cast<int>(byte);
    }
    cleanup();
    return out.str();
}

FileIdentity currentDllIdentity() {
    FileIdentity identity;
    HMODULE module = nullptr;
    const DWORD flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (!GetModuleHandleExW(flags, reinterpret_cast<LPCWSTR>(&g_rendererClosed), &module) || module == nullptr) {
        identity.error = "GetModuleHandleExW failed";
        return identity;
    }

    std::wstring modulePath(MAX_PATH, L'\0');
    while (true) {
        DWORD length = GetModuleFileNameW(module, modulePath.data(), static_cast<DWORD>(modulePath.size()));
        if (length == 0) {
            identity.error = "GetModuleFileNameW failed";
            return identity;
        }
        if (length < modulePath.size() - 1) {
            modulePath.resize(length);
            break;
        }
        modulePath.resize(modulePath.size() * 2);
    }

    std::filesystem::path path(modulePath);
    identity.path = utf8FromWide(modulePath);
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        identity.error = "file_size failed: " + ec.message();
    } else {
        identity.sizeBytes = static_cast<uint64_t>(size);
    }
    std::string hashError;
    identity.sha256 = sha256File(path, hashError);
    if (!hashError.empty()) {
        identity.error = identity.error.empty() ? hashError : identity.error + "; " + hashError;
    }
    return identity;
}
#else
FileIdentity currentDllIdentity() {
    FileIdentity identity;
    identity.error = "runtime DLL hashing is only implemented on Windows";
    return identity;
}
#endif

inline bool rendererUsable() {
    return Renderer::is_initialized() &&
           !g_rendererShuttingDown.load(std::memory_order_acquire) &&
           !g_rendererClosed.load(std::memory_order_acquire);
}
} // namespace

static DYNLIB_HANDLE bind_handle_from_candidates(JNIEnv *env, jobjectArray jnames) {
    jsize n = env->GetArrayLength(jnames);
    if (n == 0) return nullptr;
#if defined(_WIN32)
    for (jsize i = 0; i < n; ++i) {
        jstring s = (jstring)env->GetObjectArrayElement(jnames, i);
        const jchar *w = env->GetStringChars(s, nullptr);
        DYNLIB_HANDLE h = try_get_loaded_handle(reinterpret_cast<const wchar_t *>(w));
        env->ReleaseStringChars(s, w);
        env->DeleteLocalRef(s);
        if (h) return h;
    }
#else
    for (jsize i = 0; i < n; ++i) {
        jstring s = (jstring)env->GetObjectArrayElement(jnames, i);
        const char *c = env->GetStringUTFChars(s, nullptr);
        DYNLIB_HANDLE h = try_get_loaded_handle(c);
        env->ReleaseStringUTFChars(s, c);
        env->DeleteLocalRef(s);
        if (h) return h;
    }
#endif
    return nullptr;
}

static void bind_symbols(DYNLIB_HANDLE h) {
#if defined(_WIN32)
    auto gp = [&](const char *sym) { return getproc(h, sym); };
#else
    auto gp = [&](const char *sym) { return getproc(h, sym); };
#endif
    p_glfwInit = reinterpret_cast<PFN_glfwInit>(gp("glfwInit"));
    p_glfwTerminate = reinterpret_cast<PFN_glfwTerminate>(gp("glfwTerminate"));
    p_glfwGetWindowSize = reinterpret_cast<PFN_glfwGetWindowSize>(gp("glfwGetWindowSize"));
    p_glfwCreateWindowSurface = reinterpret_cast<PFN_glfwCreateWindowSurface>(gp("glfwCreateWindowSurface"));
    p_glfwGetRequiredInstanceExtensions =
        reinterpret_cast<PFN_glfwGetRequiredInstanceExtensions>(gp("glfwGetRequiredInstanceExtensions"));
    p_glfwSetWindowTitle = reinterpret_cast<PFN_glfwSetWindowTitle>(gp("glfwSetWindowTitle"));
    p_glfwSetFramebufferSizeCallback =
        reinterpret_cast<PFN_glfwSetFramebufferSizeCallback>(gp("glfwSetFramebufferSizeCallback"));
    p_glfwGetFramebufferSize = reinterpret_cast<PFN_glfwGetFramebufferSize>(gp("glfwGetFramebufferSize"));
    p_glfwWaitEvents = reinterpret_cast<PFN_glfwWaitEvents>(gp("glfwWaitEvents"));
    p_glfwPollEvents = reinterpret_cast<PFN_glfwPollEvents>(gp("glfwPollEvents"));
    p_glfwGetWindowMonitor = reinterpret_cast<PFN_glfwGetWindowMonitor>(gp("glfwGetWindowMonitor"));
    p_glfwGetPrimaryMonitor = reinterpret_cast<PFN_glfwGetPrimaryMonitor>(gp("glfwGetPrimaryMonitor"));
    p_glfwGetVideoMode = reinterpret_cast<PFN_glfwGetVideoMode>(gp("glfwGetVideoMode"));
    p_glfwGetWindowPos = reinterpret_cast<PFN_glfwGetWindowPos>(gp("glfwGetWindowPos"));
    p_glfwSetWindowPos = reinterpret_cast<PFN_glfwSetWindowPos>(gp("glfwSetWindowPos"));
    p_glfwSetWindowSize = reinterpret_cast<PFN_glfwSetWindowSize>(gp("glfwSetWindowSize"));
#ifdef _WIN32
    p_glfwGetWin32Window = reinterpret_cast<PFN_glfwGetWin32Window>(gp("glfwGetWin32Window"));
#endif
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_initFolderPath(JNIEnv *env,
                                                                                          jclass,
                                                                                          jstring folderPath) {
    if (folderPath == NULL) { return; }

    const char *nativeString = env->GetStringUTFChars(folderPath, nullptr);

    if (nativeString == nullptr) { return; }

    std::string pathStr(nativeString);

    env->ReleaseStringUTFChars(folderPath, nativeString);

    Renderer::folderPath = std::filesystem::path(pathStr);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_initRenderer(JNIEnv *env,
                                                                                        jclass,
                                                                                        jobjectArray candidates,
                                                                                        jlong windowHandle) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    g_rendererShuttingDown.store(false, std::memory_order_release);
    g_rendererClosed.store(false, std::memory_order_release);

    DYNLIB_HANDLE h = bind_handle_from_candidates(env, candidates);
    if (!h) {
        std::cerr << "[GLFW-Bind] Could not find already-loaded GLFW via NOLOAD/GetModuleHandle."
                     " Ensure Java(LWJGL) loads GLFW before JNI and pass correct names/paths."
                  << std::endl;
        std::abort();
    }
    bind_symbols(h);

    GLFWwindow *window = (GLFWwindow *)(intptr_t)windowHandle;
    Renderer::init(window);
    Renderer::instance().framework()->acquireContext();
}

extern "C" JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_maxSupportedTextureSize(JNIEnv *, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return 16384; // V2 mode: return safe default for texture atlas sizing
    auto maxImageSize = Renderer::instance().framework()->physicalDevice()->properties().limits.maxImageDimension2D;
    return maxImageSize;
}

extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_nativeBuildInfoJson(JNIEnv *env,
                                                                                                             jclass) {
    const FileIdentity dllIdentity = currentDllIdentity();
    std::ostringstream json;
    json << "{";
    json << "\"repository\":\"radser-mcvr\",";
    json << "\"commit\":\"" << json_escape(build_info::kRepoCommit) << "\",";
    json << "\"branch\":\"" << json_escape(build_info::kBranch) << "\",";
    json << "\"dirty\":" << (build_info::kDirty ? "true" : "false") << ",";
    json << "\"buildTimestamp\":\"" << json_escape(build_info::kBuildTimestamp) << "\",";
    json << "\"dllSha256\":\"" << json_escape(dllIdentity.sha256) << "\",";
    json << "\"dllPath\":\"" << json_escape(dllIdentity.path) << "\",";
    json << "\"dllSizeBytes\":" << dllIdentity.sizeBytes << ",";
    json << "\"dllHashError\":\"" << json_escape(dllIdentity.error) << "\",";
    json << "\"compileTimeDllSha256\":\"" << json_escape(build_info::kDllSha256) << "\",";
    json << "\"textureLoaderAbiVersion\":" << build_info::kTextureLoaderAbiVersion << ",";
    json << "\"cacheSchemaVersion\":" << build_info::kCacheSchemaVersion << ",";
    json << "\"features\":{";
#ifdef MCVR_ENABLE_NRD
    json << "\"nrd\":true,";
#else
    json << "\"nrd\":false,";
#endif
#ifdef MCVR_ENABLE_FFX_UPSCALER
    json << "\"ffxUpscaler\":true,";
#else
    json << "\"ffxUpscaler\":false,";
#endif
#ifdef MCVR_ENABLE_SHARC
    json << "\"sharc\":true,";
#else
    json << "\"sharc\":false,";
#endif
#ifdef MCVR_ENABLE_SHARC_MAIN_TRACE_QUERY
    json << "\"sharcMainTraceQuery\":true";
#else
    json << "\"sharcMainTraceQuery\":false";
#endif
    json << "}}";
    return env->NewStringUTF(json.str().c_str());
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_acquireContext(JNIEnv *, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return;
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->acquireContext();
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_submitCommand(JNIEnv *, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return;
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->submitCommand();
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_present(JNIEnv *, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return;
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->present();
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_drawOverlay(
    JNIEnv *, jclass, jint vertexId, jint indexId, jint pipelineType, jint indexCount, jint indexType) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return;
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto vertexBuffer = Renderer::instance().buffers()->getBuffer(vertexId);
    auto indexBuffer = Renderer::instance().buffers()->getBuffer(indexId);
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->drawIndexed(vertexBuffer, indexBuffer,
                                                  static_cast<OverlayDrawPipelineType>(pipelineType), indexCount,
                                                  static_cast<VkIndexType>(indexType));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_fuseWorld(JNIEnv *, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return;
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->fuseWorld();
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_postBlur(JNIEnv *, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return;
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto world = Renderer::instance().world();
    if (world != nullptr && world->shouldRender()) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->postBlur(6);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_close(JNIEnv *, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (g_rendererClosed.load(std::memory_order_acquire)) return;

    g_rendererShuttingDown.store(true, std::memory_order_release);
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) {
        g_rendererClosed.store(true, std::memory_order_release);
        return;
    }
    Renderer::instance().close();
    g_rendererClosed.store(true, std::memory_order_release);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_RendererProxy_shouldRenderWorld(JNIEnv *, jclass, jboolean shouldRenderWorld) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return;
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    world->shouldRender() = shouldRenderWorld;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_takeScreenshot(
    JNIEnv *, jclass, jboolean withUI, jint width, jint height, jint channel, jlong pointer) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return;
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->takeScreenshot(withUI, width, height, channel, reinterpret_cast<void *>(pointer));
}

extern "C" JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_takeScreenshotRawHdrPacked(
    JNIEnv *, jclass, jboolean withUI, jint width, jint height, jlong pointer, jint byteSize) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return static_cast<jint>(VK_FORMAT_UNDEFINED);
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return static_cast<jint>(VK_FORMAT_UNDEFINED);

    VkFormat format = framework->takeScreenshotRawHdrPacked(
        withUI, width, height, reinterpret_cast<void *>(pointer), byteSize);
    return static_cast<jint>(format);
}

// --- Window position/size persistence ---

extern "C" JNIEXPORT jintArray JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_nativeGetWindowPosSize(
    JNIEnv *env, jclass) {
    jintArray result = env->NewIntArray(4);
    if (!rendererUsable() || !Renderer::is_initialized()) return result;
    auto *r = Renderer::try_instance();
    if (!r || !r->framework() || !r->framework()->window()) return result;
    auto fw = r->framework();
    GLFWwindow *w = fw->window()->window();
    int x = 0, y = 0, width = 0, height = 0;
    if (GLFW_GetWindowPos) GLFW_GetWindowPos(w, &x, &y);
    GLFW_GetWindowSize(w, &width, &height);
    jint buf[4] = { x, y, width, height };
    env->SetIntArrayRegion(result, 0, 4, buf);
    return result;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_nativeSetWindowPos(
    JNIEnv *, jclass, jint x, jint y) {
    if (!rendererUsable() || !Renderer::is_initialized()) return;
    auto *r = Renderer::try_instance();
    if (!r || !r->framework() || !r->framework()->window()) return;
    auto fw = r->framework();
    if (GLFW_SetWindowPos) GLFW_SetWindowPos(fw->window()->window(), x, y);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_nativeSetWindowSize(
    JNIEnv *, jclass, jint width, jint height) {
    if (!rendererUsable() || !Renderer::is_initialized()) return;
    auto *r = Renderer::try_instance();
    if (!r || !r->framework() || !r->framework()->window()) return;
    auto fw = r->framework();
    if (GLFW_SetWindowSize) GLFW_SetWindowSize(fw->window()->window(), width, height);
}

/**
 * Returns GPU profiler timings as a flat string: "ModuleName:ms,ModuleName:ms,...,TOTAL:ms"
 * Empty string if profiler disabled or no data available.
 */
extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_nativeGetGpuProfile(
    JNIEnv *env, jclass) {
    auto& profiler = Renderer::gpuProfiler;
    if (!profiler.isEnabled()) return env->NewStringUTF("");

    auto timings = profiler.getModuleTimings();
    if (timings.empty()) return env->NewStringUTF("");

    std::string result;
    for (auto& t : timings) {
        if (!result.empty()) result += ",";
        // Format: name:milliseconds (3 decimal places)
        char buf[128];
        snprintf(buf, sizeof(buf), "%s:%.3f", t.name.c_str(), t.ms);
        result += buf;
    }
    // Append total
    char buf[128];
    snprintf(buf, sizeof(buf), ",TOTAL:%.3f", profiler.getTotalGpuMs());
    result += buf;

    return env->NewStringUTF(result.c_str());
}

/**
 * Enable/disable GPU profiler at runtime.
 */
extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_nativeSetGpuProfileEnabled(
    JNIEnv *, jclass, jboolean enabled) {
    Renderer::gpuProfiler.setEnabled(enabled);
}

/**
 * Transient RT.MainTrace diagnostic flags. DebugBridge sweeps these and restores them;
 * they are intentionally not persisted in Java options.
 */
extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_nativeSetRtDebugFlags(
    JNIEnv *, jclass, jint flags) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    Renderer::options.rtDebugFlags = flags < 0 ? 0u : static_cast<uint32_t>(flags);
}

extern "C" JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_nativeGetRtDebugFlags(
    JNIEnv *, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    return static_cast<jint>(Renderer::options.rtDebugFlags);
}

/**
 * Returns renderer feature truth as a flat string. This reports compiled/native reality,
 * not just Java option intent.
 */
extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_nativeGetFeatureTruth(
    JNIEnv *env, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable() || !Renderer::is_initialized()) {
        return env->NewStringUTF("rendererUsable:0");
    }

    auto* renderer = Renderer::try_instance();
    if (!renderer) return env->NewStringUTF("rendererUsable:0");

    auto framework = renderer->framework();
    auto device = framework ? framework->device() : nullptr;
    const bool serDevice = device && device->hasSER();
    const bool serActive = serDevice && Renderer::options.serEnabled;

    std::ostringstream out;
    out << "rendererUsable:1"
        << ",gpuProfilerEnabled:" << (Renderer::gpuProfiler.isEnabled() ? 1 : 0)
        << ",rayBounces:" << Renderer::options.rayBounces
        << ",simplifiedIndirectOption:" << (Renderer::options.simplifiedIndirect ? 1 : 0)
        << ",serDevice:" << (serDevice ? 1 : 0)
        << ",serOption:" << (Renderer::options.serEnabled ? 1 : 0)
        << ",serHintsOption:" << (Renderer::options.serHintsEnabled ? 1 : 0)
        << ",serActive:" << (serActive ? 1 : 0)
        << ",rtDebugFlags:" << Renderer::options.rtDebugFlags;

    bool rayTracingModuleFound = false;
    auto pipeline = framework ? framework->pipeline() : nullptr;
    auto worldPipeline = pipeline ? pipeline->worldPipeline() : nullptr;
    if (worldPipeline) {
        for (const auto& module : worldPipeline->worldModules()) {
            auto rayTracingModule = std::dynamic_pointer_cast<RayTracingModule>(module);
            if (!rayTracingModule) continue;
            rayTracingModuleFound = true;
            out << ",rayTracingModule:1," << rayTracingModule->diagnosticFeatureTruth();
            break;
        }
    }
    if (!rayTracingModuleFound) {
        out << ",rayTracingModule:0"
#ifdef MCVR_ENABLE_SHARC
            << ",sharcCompiled:1";
#else
            << ",sharcCompiled:0";
#endif
    }

    auto world = renderer->world();
    auto entities = world ? world->entities() : nullptr;
    if (entities) {
        std::string entityDiag = entities->diagnosticsString();
        if (!entityDiag.empty()) out << "," << entityDiag;
    }

    return env->NewStringUTF(out.str().c_str());
}

/**
 * Returns color-pipeline state as a flat CSV string.
 */
extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_nativeGetColorPipelineDiagnostics(
    JNIEnv *env, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable() || !Renderer::is_initialized()) {
        return env->NewStringUTF("rendererUsable:0");
    }

    auto* renderer = Renderer::try_instance();
    if (!renderer) return env->NewStringUTF("rendererUsable:0");

    auto framework = renderer->framework();
    auto swapchain = framework ? framework->swapchain() : nullptr;
    VkSurfaceFormatKHR surfaceFormat{};
    bool hdrSwapchain = false;
    bool hdr10Swapchain = false;
    bool scRgbSwapchain = false;
    bool transferSrc = false;
    if (swapchain) {
        surfaceFormat = swapchain->vkSurfaceFormat();
        hdrSwapchain = swapchain->isHDR();
        hdr10Swapchain = swapchain->isHDR10();
        scRgbSwapchain = swapchain->isScRGB();
        transferSrc = swapchain->supportsTransferSrc();
    }

    std::ostringstream out;
    out << "rendererUsable:1"
        << ",hdrOption:" << (Renderer::options.hdrEnabled ? 1 : 0)
        << ",hdrScrgbRequested:" << (Renderer::options.hdrScrgbMode ? 1 : 0)
        << ",swapchainHdr:" << (hdrSwapchain ? 1 : 0)
        << ",swapchainHdr10:" << (hdr10Swapchain ? 1 : 0)
        << ",swapchainScRgb:" << (scRgbSwapchain ? 1 : 0)
        << ",swapchainFormat:" << surfaceFormat.format
        << ",swapchainColorSpace:" << surfaceFormat.colorSpace
        << ",swapchainTransferSrc:" << (transferSrc ? 1 : 0)
        << ",sdrTonemapMode:" << Renderer::options.tonemappingMode
        << ",sdrWorkingSpace:BT709"
        << ",sdrFinalGamutMap:none"
        << ",sdrHardClamp:1"
        << ",sdrPsychoWorkingSpace:BT709"
        << ",hdrTonemapMode:" << Renderer::options.hdrTonemapMode
        << ",saturation:" << Renderer::options.saturation
        << ",saturationAdaptive:" << (Renderer::options.saturationAdaptive ? 1 : 0)
        << ",sdrTransferFunction:" << Renderer::options.sdrTransferFunction
        << ",paperWhiteNits:" << Renderer::options.hdrPaperWhiteNits
        << ",peakNits:" << Renderer::options.hdrPeakNits
        << ",sharpenerMode:" << Renderer::options.sharpenerMode;

    return env->NewStringUTF(out.str().c_str());
}

/**
 * Returns DLSS-G latency diagnostics as a flat CSV string.
 */
extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_nativeGetDlssgLatencyDiag(
    JNIEnv *env, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable() || !Renderer::is_initialized()) {
        return env->NewStringUTF("rendererUsable:0");
    }

    return env->NewStringUTF(FrameGenManager::latencyDiagnostics().c_str());
}

/**
 * Returns VMA memory statistics as a CSV string.
 */
extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_nativeGetVmaStats(
    JNIEnv *env, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return env->NewStringUTF("");
    auto framework = Renderer::instance().framework();
    if (!framework) return env->NewStringUTF("");
    auto vma = framework->vma();
    if (!vma) return env->NewStringUTF("");

    auto stats = vma->getStats();
    char buf[256];
    snprintf(buf, sizeof(buf),
             "totalAllocMB:%.1f,usedMB:%.1f,budgetMB:%.1f,budgetUsageMB:%.1f,allocations:%u,blocks:%u",
             stats.totalAllocBytes / (1024.0 * 1024.0),
             stats.totalUsedBytes / (1024.0 * 1024.0),
             stats.budgetBytes / (1024.0 * 1024.0),
             stats.budgetUsageBytes / (1024.0 * 1024.0),
             stats.allocationCount,
             stats.blockCount);
    return env->NewStringUTF(buf);
}

/**
 * Returns texture reload diagnostics as a flat CSV string.
 */
extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_nativeGetTextureReloadDiagnostics(
    JNIEnv *env, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable() || !Renderer::is_initialized()) {
        return env->NewStringUTF("rendererUsable:0");
    }

    auto* renderer = Renderer::try_instance();
    if (!renderer) return env->NewStringUTF("rendererUsable:0");

    const uint64_t textureGeneration = Renderer::textureSystem.generation();
    const bool textureDebugDumped =
        Renderer::textureSystem.dumpDebug("C:/RadSER/texture_system_full.csv", 0);
    std::ostringstream out;
    out << Renderer::textureSystem.statusString()
        << ",textureDebugDumped:" << (textureDebugDumped ? 1 : 0);

    uint32_t chunkTotal = 0;
    uint32_t chunksWithBlas = 0;
    uint32_t chunksMatchingGeneration = 0;
    uint32_t chunksStaleGeneration = 0;
    uint32_t chunksWithoutBlas = 0;
    uint32_t chunkInputQueue = 0;
    std::map<uint64_t, uint32_t> generationHistogram;

    auto world = renderer->world();
    auto chunks = world ? world->chunks() : nullptr;
    if (chunks) {
        std::unique_lock<std::recursive_mutex> chunkLock(chunks->mutex());
        auto& chunk1s = chunks->chunks();
        chunkTotal = static_cast<uint32_t>(chunk1s.size());
        for (const auto& chunk : chunk1s) {
            if (!chunk) continue;
            generationHistogram[chunk->textureGeneration]++;
            if (!chunk->blas) {
                chunksWithoutBlas++;
                continue;
            }
            chunksWithBlas++;
            if (textureGeneration == 0 || chunk->textureGeneration == textureGeneration) {
                chunksMatchingGeneration++;
            } else {
                chunksStaleGeneration++;
            }
        }
        chunkInputQueue = chunks->getInputQueueSize();
    }

    out << ",chunkTotal:" << chunkTotal
        << ",chunksWithBlas:" << chunksWithBlas
        << ",chunksMatchingGeneration:" << chunksMatchingGeneration
        << ",chunksStaleGeneration:" << chunksStaleGeneration
        << ",chunksWithoutBlas:" << chunksWithoutBlas
        << ",chunkInputQueue:" << chunkInputQueue;

    for (const auto& [generation, count] : generationHistogram) {
        out << ",chunkGen_" << generation << ':' << count;
    }

    return env->NewStringUTF(out.str().c_str());
}

/**
 * Get overlay compositor diagnostic string (CSV key=value pairs).
 */
extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_nativeGetOverlayDiag(
    JNIEnv *env, jclass) {
    auto fw = Renderer::instance().framework();
    if (!fw) return env->NewStringUTF("framework=null");

    auto *compositor = fw->overlayCompositor();
    if (!compositor) {
        // No compositor object — report why
        std::string s = "compositor=null,frameGenEnabled=";
        s += std::to_string(Renderer::options.frameGenEnabled ? 1 : 0);
        s += ",decoupledPresent=";
        s += std::to_string(fw->isDecoupledPresent() ? 1 : 0);
        return env->NewStringUTF(s.c_str());
    }

    return env->NewStringUTF(compositor->getDiagnostics().c_str());
}
