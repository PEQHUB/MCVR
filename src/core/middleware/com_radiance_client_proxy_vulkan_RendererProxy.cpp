#include "com_radiance_client_proxy_vulkan_RendererProxy.h"

#include "core/all_extern.hpp"
#include "core/render/buffers.hpp"
#include "core/render/modules/ui_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/textures.hpp"
#include "core/render/world.hpp"

#include <atomic>
#include <mutex>

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
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_initFolderPath(JNIEnv *env,
                                                                                          jclass,
                                                                                          jstring folderPath) {
    if (folderPath == NULL) { return; }

    const char *nativeString = env->GetStringUTFChars(folderPath, nullptr);

    if (nativeString == nullptr) { return; }

    std::string pathStr(nativeString);

    env->ReleaseStringUTFChars(folderPath, nativeString);

    Renderer::folderPath = std::filesystem::path(pathStr);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_initRenderer(JNIEnv *env,
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

JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_maxSupportedTextureSize(JNIEnv *, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return 0;
    auto maxImageSize = Renderer::instance().framework()->physicalDevice()->properties().limits.maxImageDimension2D;
    return maxImageSize;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_acquireContext(JNIEnv *, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return;
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->acquireContext();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_submitCommand(JNIEnv *, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return;
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->submitCommand();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_present(JNIEnv *, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return;
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->present();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_drawOverlay(
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

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_fuseWorld(JNIEnv *, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return;
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->fuseWorld();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_postBlur(JNIEnv *, jclass) {
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

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_close(JNIEnv *, jclass) {
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

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_RendererProxy_shouldRenderWorld(JNIEnv *, jclass, jboolean shouldRenderWorld) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return;
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    world->shouldRender() = shouldRenderWorld;
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_takeScreenshot(
    JNIEnv *, jclass, jboolean withUI, jint width, jint height, jint channel, jlong pointer) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return;
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    framework->takeScreenshot(withUI, width, height, channel, reinterpret_cast<void *>(pointer));
}

JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_RendererProxy_takeScreenshotRawHdrPacked(
    JNIEnv *, jclass, jboolean withUI, jint width, jint height, jlong pointer, jint byteSize) {
    std::lock_guard<std::recursive_mutex> guard(g_rendererJniMtx);
    if (!rendererUsable()) return static_cast<jint>(VK_FORMAT_UNDEFINED);
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return static_cast<jint>(VK_FORMAT_UNDEFINED);

    VkFormat format = framework->takeScreenshotRawHdrPacked(
        withUI, width, height, reinterpret_cast<void *>(pointer), byteSize);
    return static_cast<jint>(format);
}
