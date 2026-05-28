#include "com_radiance_client_proxy_vulkan_RendererProxy.h"

#include "core/all_extern.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/modules/ui_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/overlay_compositor.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/textures.hpp"
#include "core/render/world.hpp"
#include "core/vulkan/vma.hpp"

#include <atomic>
#include <map>
#include <mutex>
#include <sstream>

#if defined(_WIN32)
#    include <windows.h>
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

inline bool rendererUsable() {
    return !g_rendererShuttingDown.load(std::memory_order_acquire) &&
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
    if (!rendererUsable()) return 0;
    auto maxImageSize = Renderer::instance().framework()->physicalDevice()->properties().limits.maxImageDimension2D;
    return maxImageSize;
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
        << ",textureDebugDumped:" << (textureDebugDumped ? 1 : 0)
        << ",blockModelTableLoaded:" << (Renderer::blockModelTable.isLoaded() ? 1 : 0)
        << ",blockModelTableGeneration:" << Renderer::blockModelTable.generation()
        << ",blockModelStateCount:" << Renderer::blockModelTable.stateCount()
        << ",blockModelMaxStateId:" << Renderer::blockModelTable.maxStateId();

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
