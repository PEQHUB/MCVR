#if defined(_WIN32)
#    define VK_USE_PLATFORM_WIN32_KHR
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include "core/render/streamline_context.hpp"
#endif

#include "engine_app.hpp"
#include "engine_session.hpp"
#include "engine_services.hpp"
#include "config/config_service.hpp"
#include "bridge/bridge_service.hpp"
#include "frame/frame_scheduler.hpp"
#include "frame/offscreen_target.hpp"
#include "diagnostics/metrics_service.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "platform/vulkan/vk2_swapchain.hpp"
#include "rendergraph/graph_builder.hpp"
#include "features/feature_adapter.hpp"
#include "features/test_compute_adapter.hpp"
#include "features/tonemapping_adapter.hpp"
#include "features/raytracing_adapter.hpp"
#include "features/dlss_adapter.hpp"
#include "features/frame_gen_adapter.hpp"
#include "features/rt_descriptor_layout.hpp"
#include "features/postprocess_adapter.hpp"
#include "features/temporal_denoiser_adapter.hpp"
#include "features/atmosphere_adapter.hpp"
#include "features/color_to_depth_adapter.hpp"
#include "features/post_render_adapter.hpp"
#include "features/starfield_adapter.hpp"
#include "scene/scene_service.hpp"
#include "scene/chunk_registry.hpp"
#include "scene/gpu_upload_service.hpp"
#include "scene/blas_service.hpp"
#include "scene/entity_blas_service.hpp"
#include "scene/tlas_service.hpp"
#include "scene/scene_resource_service.hpp"
#include "scene/texture_service.hpp"
#include "diagnostics/log.hpp"
#include "diagnostics/diag_flags.hpp"
#include "diagnostics/boot_trace.hpp"
#include "diagnostics/replay/replay_recorder.hpp"

// V1 light type definitions — used by area light gathering in processScene()
#include "core/render/lights.hpp"
#include "core/render/renderer.hpp"

// AreaLight GPU struct (48 bytes, std430)
#include "common/shared.hpp"

#include <volk.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace engine {

EngineApp* EngineApp::s_instance = nullptr;

EngineApp::~EngineApp() {
    if (initialized_) {
        // Defensively shut down the logger FIRST before any other teardown
        // logging. If the JVM exits without calling nativeShutdown() (crash,
        // System.exit(-1)), this destructor fires during dllmain_crt_process_detach
        // after the CRT has already run static destructors — including spdlog's
        // own sink objects. Logging after that point causes EXCEPTION_ACCESS_VIOLATION
        // inside isCategoryEnabled (g_filterMutex destroyed). Silencing the logger
        // here makes the teardown silent but crash-free.
        log::shutdown();
        shutdown();
    }
}

bool EngineApp::init(const EngineInitConfig& config) {
    if (initialized_) return true;

    mode_ = config.mode;
    s_instance = this;

    // Boot trace must come up FIRST, before the spdlog logger. spdlog::init()
    // is itself one of the breadcrumb stages and may fail; the boot trace is
    // independent so we can still see "logging fail" in v2_boot.log.
    boot_trace::init(config.configDir + "/logs");
    boot_trace::breadcrumb("init", "begin",
        std::string("mode=") + (mode_ == EngineMode::V2 ? "v2" : "legacy"));

    // Cleanup on failure — releases partial state so legacy can start clean
    auto rollback = [this]() {
        if (services_) {
            if (mode_ == EngineMode::V2 && services_->device().isInitialized()) {
                services_->device().waitIdle();
                for (auto& a : adapters_) a->shutdown();
                adapters_.clear();
                services_->offscreen().shutdown();
                services_->metrics().shutdown();
                services_->frame().shutdownSync();
                services_->swapchain().shutdown();
                services_->device().shutdown();
            }
            services_.reset();
        }
        session_.reset();
        log::shutdown();
        s_instance = nullptr;
    };

    // Initialize logging first
    boot_trace::breadcrumb("logging", "begin");
    log::LogConfig logConfig;
    logConfig.logDir = config.configDir + "/logs";
    log::init(logConfig);
    boot_trace::breadcrumb("logging", "ok");

    log::info("app", "EngineApp initializing");
    log::info("app", std::string("Mode: ") + (mode_ == EngineMode::V2 ? "v2" : "legacy"));

    // Create services
    boot_trace::breadcrumb("services", "begin");
    services_ = std::make_unique<EngineServices>();
    services_->setResourceDir(config.configDir);
    boot_trace::breadcrumb("services", "ok");

    // Load config
    boot_trace::breadcrumb("config", "begin");
    std::string configPath = config.configDir + "/options.properties";
    services_->config().load(configPath);
    boot_trace::breadcrumb("config", "ok", configPath);

    // Apply initial diagnostic level from loaded config
    applyDiagLevel(services_->config().live().diagLevel,
                   services_->config().live().diagFlags);

    // Wire bridge handler
    boot_trace::breadcrumb("bridge", "begin");
    services_->bridge().setHandler([this](const BridgeCommand& cmd) {
        std::visit([this](const auto& c) { handleCommand(c); }, cmd);
    });
    boot_trace::breadcrumb("bridge", "ok");

    // Initialize Vulkan in V2 mode
    if (mode_ == EngineMode::V2) {
        if (!config.window && !config.nativeWindowHandle) {
            log::error("app", "V2 mode requires a window (GLFW or native handle)");
            boot_trace::breadcrumb("device", "fail", "no window/native handle provided");
            boot_trace::writeCrashDump("init: no window provided");
            rollback();
            return false;
        }

        vk2::DeviceService::InitConfig deviceConfig;
        deviceConfig.enableValidation = config.enableValidation
            || (services_->config().snapshot() && services_->config().snapshot()->data.validationLayers);
        deviceConfig.enableDiagnostics = services_->config().snapshot()->data.gpuDiagnostics;
        deviceConfig.logsDir = config.configDir + "/logs";

        if (config.nativeWindowHandle) {
            // JNI mode: create Win32 surface directly, bypassing GLFW
#if defined(_WIN32)
            HWND hwnd = static_cast<HWND>(config.nativeWindowHandle);
            deviceConfig.surfaceFactory = [hwnd](VkInstance instance, VkSurfaceKHR* surface) {
                // Load vkCreateWin32SurfaceKHR manually — volk may not have it
                // if VOLK_STATIC_DEFINES didn't include VK_USE_PLATFORM_WIN32_KHR
                auto pfn = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
                    vkGetInstanceProcAddr(instance, "vkCreateWin32SurfaceKHR"));
                if (!pfn) return VK_ERROR_EXTENSION_NOT_PRESENT;
                VkWin32SurfaceCreateInfoKHR ci{};
                ci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
                ci.hinstance = GetModuleHandle(nullptr);
                ci.hwnd = hwnd;
                return pfn(instance, &ci, nullptr, surface);
            };
            deviceConfig.instanceExtensions = {
                VK_KHR_SURFACE_EXTENSION_NAME,
                VK_KHR_WIN32_SURFACE_EXTENSION_NAME
            };
            log::info("app", "Using Win32 native surface (JNI mode)");
#else
            log::error("app", "Native window handle not supported on this platform");
            rollback();
            return false;
#endif
        } else {
            deviceConfig.window = config.window;
        }

        // Query DLSS/NGX required Vulkan extensions BEFORE device creation.
        // These queries are static SDK calls that don't require a live instance.
        // The device extension list requires a physical device, so we plumb it
        // through a callback that the DeviceService invokes between
        // pickPhysicalDevice and createDevice.
        {
            auto ngxInstExts = DlssAdapter::getRequiredInstanceExtensions();
            for (const char* ext : ngxInstExts) {
                // Append to extraInstanceExtensions so we don't clobber the JNI/GLFW
                // provided base set. DeviceService::init deduplicates internally.
                deviceConfig.extraInstanceExtensions.push_back(ext);
            }
            if (!ngxInstExts.empty()) {
                log::info("app", std::string("DLSS instance extensions added: ") +
                          std::to_string(ngxInstExts.size()));
            }

            deviceConfig.extraDeviceExtensionsCallback = [](VkInstance inst, VkPhysicalDevice pd) {
                return DlssAdapter::getRequiredDeviceExtensions(inst, pd);
            };
        }

        // Streamline / DLSS-G: must be initialized BEFORE vkCreateInstance.
        // The plugin dir is one level above configDir (e.g. .minecraft/radiance/sl/).
        // enableFrameGen is loaded from options.properties; streamlinePluginDir
        // is the absolute path to the sl.*.dll directory.
        {
            const auto& cfg = services_->config().live();
            deviceConfig.enableFrameGen = cfg.frameGenEnabled;
#ifdef _WIN32
            if (cfg.frameGenEnabled) {
                // Convert configDir (UTF-8) to wide string for StreamlineContext::init().
                // configDir is typically ".../radiance/config", plugin dir is sibling "sl/".
                std::string pluginDirUtf8 = config.configDir + "/../sl";
                int wlen = MultiByteToWideChar(CP_UTF8, 0,
                    pluginDirUtf8.c_str(), -1, nullptr, 0);
                if (wlen > 0) {
                    std::wstring pluginDirW(wlen, L'\0');
                    MultiByteToWideChar(CP_UTF8, 0,
                        pluginDirUtf8.c_str(), -1, &pluginDirW[0], wlen);
                    pluginDirW.resize(wlen - 1); // strip null terminator
                    deviceConfig.streamlinePluginDir = std::move(pluginDirW);
                }
                log::info("app", "Frame gen enabled: SL plugin dir = " + pluginDirUtf8);
            }
#endif
        }

        boot_trace::breadcrumb("device", "begin");
        auto deviceResult = services_->device().init(deviceConfig);
        if (!deviceResult) {
            log::error("app", "DeviceService init failed: " + deviceResult.error().message);
            boot_trace::breadcrumb("device", "fail", deviceResult.error().message);
            boot_trace::writeCrashDump("device init failed", 0, deviceResult.error().message);
            rollback();
            return false;
        }
        boot_trace::breadcrumb("device", "ok", services_->device().caps().deviceName);

        boot_trace::breadcrumb("swapchain", "begin");
        auto swapResult = services_->swapchain().init(services_->device());
        if (!swapResult) {
            log::error("app", "SwapchainService init failed: " + swapResult.error().message);
            boot_trace::breadcrumb("swapchain", "fail", swapResult.error().message);
            boot_trace::writeCrashDump("swapchain init failed", 0, swapResult.error().message);
            rollback();
            return false;
        }
        {
            auto ext = services_->swapchain().extent();
            boot_trace::breadcrumb("swapchain", "ok",
                std::to_string(ext.width) + "x" + std::to_string(ext.height) +
                " (" + std::to_string(services_->swapchain().imageCount()) + " images)");
        }

        // Initialize frame scheduler
        boot_trace::breadcrumb("scheduler", "begin");
        uint32_t imgCount = services_->swapchain().imageCount();
        services_->frame().setImageCount(imgCount);
        uint32_t framesInFlight = std::min(imgCount, 2u);

        auto initSyncResult = services_->frame().initSync(services_->device(), framesInFlight);
        if (!initSyncResult) {
            log::error("app", "Frame sync init failed: " + initSyncResult.error().message);
            boot_trace::breadcrumb("scheduler", "fail", initSyncResult.error().message);
            boot_trace::writeCrashDump("frame sync init failed", 0, initSyncResult.error().message);
            rollback();
            return false;
        }
        boot_trace::breadcrumb("scheduler", "ok",
            std::to_string(framesInFlight) + " frames-in-flight");

        // Initialize metrics
        boot_trace::breadcrumb("metrics", "begin");
        auto metricsResult = services_->metrics().init(services_->device(), framesInFlight);
        if (!metricsResult) {
            log::warn("app", "Metrics init failed (non-fatal): " + metricsResult.error().message);
            boot_trace::breadcrumb("metrics", "warn", metricsResult.error().message);
        } else {
            boot_trace::breadcrumb("metrics", "ok");
        }

        // Initialize scene services (with deferred-delete via FrameScheduler::gc())
        boot_trace::breadcrumb("scene", "begin");
        {
            auto& gc = services_->frame().gc();
            auto gpuR = services_->gpuUpload().init(services_->device(), framesInFlight, &gc);
            if (!gpuR) {
                log::warn("app", "GpuUpload init failed (non-fatal): " + gpuR.error().message);
                boot_trace::breadcrumb("scene.gpuUpload", "warn", gpuR.error().message);
            }
            auto blasR = services_->blas().init(services_->device(), &gc);
            if (!blasR) {
                log::warn("app", "BLAS init failed (non-fatal): " + blasR.error().message);
                boot_trace::breadcrumb("scene.blas", "warn", blasR.error().message);
            }
            // Propagate OMM config: when ommEnabled, drop the OPAQUE geometry flag so
            // any-hit shaders fire for alpha testing. CPU-side OMM baking is not possible
            // in V2 (alpha data is GPU-side post-finalize), so this is the correct fallback.
            services_->blas().setOmmEnabled(
                services_->config().snapshot()->data.ommEnabled);
            auto entityBlasR = services_->entityBlas().init(services_->device(), &gc);
            if (!entityBlasR) {
                log::warn("app", "EntityBlas init failed (non-fatal): " + entityBlasR.error().message);
                boot_trace::breadcrumb("scene.entityBlas", "warn", entityBlasR.error().message);
            }
            auto tlasR = services_->tlas().init(services_->device(), &gc);
            if (!tlasR) {
                log::warn("app", "TLAS init failed (non-fatal): " + tlasR.error().message);
                boot_trace::breadcrumb("scene.tlas", "warn", tlasR.error().message);
            }
            auto sceneResR = services_->sceneRes().init(services_->device(), framesInFlight, &gc);
            if (!sceneResR) {
                log::warn("app", "SceneRes init failed (non-fatal): " + sceneResR.error().message);
                boot_trace::breadcrumb("scene.sceneRes", "warn", sceneResR.error().message);
            }
            auto texR = services_->texture().init(services_->device(), &gc);
            if (!texR) {
                log::warn("app", "TextureService init failed (non-fatal): " + texR.error().message);
                boot_trace::breadcrumb("scene.texture", "warn", texR.error().message);
            }
        }
        boot_trace::breadcrumb("scene", "ok");

        // Initialize offscreen render targets
        boot_trace::breadcrumb("offscreen", "begin");
        auto offscreenResult = services_->offscreen().init(
            services_->device(), framesInFlight,
            services_->swapchain().extent().width,
            services_->swapchain().extent().height);
        if (!offscreenResult) {
            log::error("app", "Offscreen init failed: " + offscreenResult.error().message);
            boot_trace::breadcrumb("offscreen", "fail", offscreenResult.error().message);
            boot_trace::writeCrashDump("offscreen init failed", 0, offscreenResult.error().message);
            rollback();
            return false;
        }
        boot_trace::breadcrumb("offscreen", "ok");

        log::info("app", "Vulkan initialized: " + services_->device().caps().deviceName +
                  ", swapchain " + std::to_string(services_->swapchain().extent().width) +
                  "x" + std::to_string(services_->swapchain().extent().height));
    }

    // Create session
    boot_trace::breadcrumb("session", "begin");
    session_ = std::make_unique<EngineSession>(*services_);
    session_->setState(SessionState::Running);
    boot_trace::breadcrumb("session", "ok");

    initialized_ = true;

    // Boot summary
    if (mode_ == EngineMode::V2) {
        const auto& caps = services_->device().caps();
        auto ext = services_->swapchain().extent();
        log::info("app", "=== V2 Bootstrap Gold ===");
        log::info("app", "Profile: " + std::string(
            caps.profile == vk2::DeviceProfile::AdvancedRT ? "AdvancedRT" :
            caps.profile == vk2::DeviceProfile::BaselineRenderer ? "BaselineRenderer" :
            "BootstrapPresent"));
        log::info("app", "GPU: " + caps.deviceName);
        log::info("app", "Vulkan: " + std::to_string(VK_VERSION_MAJOR(caps.vulkanVersion)) + "." +
                  std::to_string(VK_VERSION_MINOR(caps.vulkanVersion)));
        log::info("app", "Swapchain: " + std::to_string(ext.width) + "x" +
                  std::to_string(ext.height) + " (" +
                  std::to_string(services_->swapchain().imageCount()) + " images)");
        log::info("app", "Sync2: " + std::string(caps.synchronization2 ? "yes" : "no") +
                  "  DynRender: " + std::string(caps.dynamicRendering ? "yes" : "no") +
                  "  RT: " + std::string(caps.rayTracingPipeline ? "yes" : "no"));
        log::info("app", "========================");

        // Wire scene processing callback
        services_->frame().setPreGraphCallback([this](VkCommandBuffer cmd) {
            processScene(cmd);
        });

        // Wire post-graph callback for DLSS-G resource tagging.
        // After all passes execute, tag depth/MV/hudless for Streamline frame generation.
        // fgAdapter_ is set below; this lambda captures it by pointer safely because
        // adapters_ lifetime equals EngineApp lifetime and fgAdapter_ is reset in shutdown.
        services_->frame().setPostGraphCallback([this](VkCommandBuffer cmd, const FrameContext& ctx) {
            if (!fgAdapter_ || !fgAdapter_->isActive()) return;
            if (!dlssAdapter_ || !rtAdapter_) return;

            // Resolve VkImages for depth and motion from the graph resource pool.
            // The pool is fully allocated at this point (passes have already executed).
            auto& pool = services_->frame().resourcePool();
            ResourceHandle depthHandle = rtAdapter_->linearDepthHandle();
            ResourceHandle mvHandle    = rtAdapter_->motionHandle();

            VkImage depthImg = VK_NULL_HANDLE;
            uint32_t depthW = 0, depthH = 0;
            VkImage mvImg = VK_NULL_HANDLE;
            uint32_t mvW = 0, mvH = 0;

            if (depthHandle.valid() && pool.isAllocated() &&
                    depthHandle.index < pool.imageCount()) {
                const auto& depthRes = pool.image(depthHandle.index);
                depthImg = depthRes.handle();
                depthW   = depthRes.width();
                depthH   = depthRes.height();
            }
            if (mvHandle.valid() && pool.isAllocated() &&
                    mvHandle.index < pool.imageCount()) {
                const auto& mvRes = pool.image(mvHandle.index);
                mvImg = mvRes.handle();
                mvW   = mvRes.width();
                mvH   = mvRes.height();
            }

            // HUD-less = DLSS output (display resolution) — cached by DlssAdapter.execute()
            VkImage hudlessImg = dlssAdapter_->hudlessVkImage();
            uint32_t hudlessW  = dlssAdapter_->hudlessWidth();
            uint32_t hudlessH  = dlssAdapter_->hudlessHeight();

            fgAdapter_->tagFrame(cmd, ctx,
                depthImg,   depthW,   depthH,
                mvImg,      mvW,      mvH,
                hudlessImg, hudlessW, hudlessH);
        });

        // Build render graph:
        //   DLSS-RR active: RT → DLSS-RR → ToneMapping → PostProcess (CAS) → Composite
        //   Fallback:        RT → TemporalDenoiser → ToneMapping → PostProcess (CAS) → Composite
        // Initialize atmosphere adapter (generates LUT + cubemap, not a graph pass)
        boot_trace::breadcrumb("adapter.atmosphere", "begin");
        atmosphere_.init(*services_);
        boot_trace::breadcrumb("adapter.atmosphere", "ok");

        boot_trace::breadcrumb("adapter.raytracing", "begin");
        auto rtAdapter = std::make_unique<RayTracingAdapter>();
        rtAdapter->setAtmosphere(&atmosphere_);
        rtAdapter->init(*services_);
        rtAdapter_ = rtAdapter.get();  // keep raw pointer for pre-exposure wiring
        boot_trace::breadcrumb("adapter.raytracing", "ok",
            rtAdapter->isFullShaderSuiteLoaded() ? "full shader suite" : "v2_min stability path");

        boot_trace::breadcrumb("adapter.temporal", "begin");
        auto denoiseAdapter = std::make_unique<TemporalDenoiserAdapter>();
        denoiseAdapter->init(*services_);
        boot_trace::breadcrumb("adapter.temporal", "ok");

        // DLSS-RR (Ray Reconstruction). Created regardless of availability —
        // shutdown is safe either way. If DLSS-RR is available, it replaces
        // the temporal denoiser entirely and consumes all RT guide buffers.
        boot_trace::breadcrumb("adapter.dlss", "begin");
        auto dlssAdapter = std::make_unique<DlssAdapter>();
        dlssAdapter->init(*services_);
        dlssAdapter_ = dlssAdapter.get();  // keep raw pointer for jitter access
        boot_trace::breadcrumb("adapter.dlss", "ok",
            dlssAdapter->isAvailable() ? "available" : "unavailable");
        if (dlssAdapter->isAvailable()) {
            uint32_t dlssQuality = services_->config().live().dlssQuality;

            // Compute render resolution based on DLSS quality mode.
            // DLAA: input == output (no upscaling, denoising only).
            // Other modes scale down the input (render) resolution.
            uint32_t inputW = ext.width;
            uint32_t inputH = ext.height;
            switch (dlssQuality) {
                case 1:  // Quality: 2/3 resolution
                    inputW = (ext.width * 2) / 3;
                    inputH = (ext.height * 2) / 3;
                    break;
                case 2:  // Balanced: 58% resolution
                    inputW = static_cast<uint32_t>(ext.width * 0.58f);
                    inputH = static_cast<uint32_t>(ext.height * 0.58f);
                    break;
                case 3:  // Performance: 1/2 resolution
                    inputW = ext.width / 2;
                    inputH = ext.height / 2;
                    break;
                case 4:  // UltraPerformance: 1/3 resolution
                    inputW = ext.width / 3;
                    inputH = ext.height / 3;
                    break;
                default: // 0 = DLAA: input == output
                    break;
            }

            if (!dlssAdapter->configureDlss(inputW, inputH, ext.width, ext.height, dlssQuality)) {
                log::warn("app", "DLSS-RR configureDlss failed — falling back to TemporalDenoiser");
            }
        }

        boot_trace::breadcrumb("adapter.tonemapping", "begin");
        auto tmAdapter = std::make_unique<ToneMappingAdapter>();
        tmAdapter->init(*services_);
        tmAdapter_ = tmAdapter.get();  // keep raw pointer for exposure readback
        boot_trace::breadcrumb("adapter.tonemapping", "ok");

        boot_trace::breadcrumb("adapter.postprocess", "begin");
        auto ppAdapter = std::make_unique<PostProcessAdapter>();
        ppAdapter->init(*services_);
        boot_trace::breadcrumb("adapter.postprocess", "ok");

        boot_trace::breadcrumb("adapter.color2depth", "begin");
        auto c2dAdapter = std::make_unique<ColorToDepthAdapter>();
        c2dAdapter->init(*services_);
        boot_trace::breadcrumb("adapter.color2depth", "ok");

        boot_trace::breadcrumb("adapter.postrender", "begin");
        auto prAdapter = std::make_unique<PostRenderAdapter>();
        prAdapter->setDepthSource(c2dAdapter.get());
        prAdapter->init(*services_);
        prAdapter_ = prAdapter.get();  // keep raw pointer for post-entity data
        boot_trace::breadcrumb("adapter.postrender", "ok");

        boot_trace::breadcrumb("adapter.starfield", "begin");
        auto sfAdapter = std::make_unique<StarFieldAdapter>();
        sfAdapter->setDepthSource(c2dAdapter.get());
        sfAdapter->init(*services_);
        boot_trace::breadcrumb("adapter.starfield", "ok");

        // FrameGenAdapter: probe DLSS-G support and configure Reflex.
        // Must be created AFTER the device is up (Streamline hooks device during init).
        // Non-owning raw pointer kept for per-frame wiring.
        boot_trace::breadcrumb("adapter.framegen", "begin");
        auto fgAdapter = std::make_unique<FrameGenAdapter>();
        fgAdapter->init(*services_);
        if (services_->config().live().frameGenEnabled) {
            fgAdapter->configure(*services_->config().snapshot());
        }
        fgAdapter_ = fgAdapter.get();
        boot_trace::breadcrumb("adapter.framegen", "ok");

        // CloudAdapter: volumetric cloud compositing (after DLSS/denoiser, before tone mapping).
        // Disabled at runtime when cloudQuality == 0 (pass-through).
        boot_trace::breadcrumb("adapter.cloud", "begin");
        auto cloudAdapter = std::make_unique<CloudAdapter>();
        cloudAdapter->init(*services_);
        cloudAdapter_ = cloudAdapter.get();  // keep raw pointer for graph wiring
        boot_trace::breadcrumb("adapter.cloud", "ok");

        boot_trace::breadcrumb("graph", "begin");
        GraphBuilder builder;
        auto rtRadiance = rtAdapter->registerPass(builder);
        const bool fullRtActive = rtAdapter->isFullShaderSuiteLoaded();

        ResourceHandle toneMapInput;
        const bool dlssActive = fullRtActive &&
                                dlssAdapter->isAvailable() &&
                                dlssAdapter->isFeatureCreated();
        if (dlssActive) {
            // DLSS-RR replaces the temporal denoiser entirely.
            toneMapInput = dlssAdapter->registerPassRR(builder,
                rtAdapter->radianceHandle(),
                rtAdapter->diffuseAlbedoHandle(),
                rtAdapter->specularAlbedoHandle(),
                rtAdapter->normalHandle(),
                rtAdapter->motionHandle(),
                rtAdapter->linearDepthHandle(),
                rtAdapter->specularHitDepthHandle(),
                rtAdapter->firstHitDepthHandle(),
                rtAdapter->outputHandle(rt_layout::OUT_BINDING_DIFFUSE_RAY_DIR),
                rtAdapter->outputHandle(rt_layout::OUT_BINDING_SPECULAR_RAY_DIR),
                rtAdapter->reflectionMvHandle(),
                rtAdapter->outputHandle(rt_layout::OUT_BINDING_ANIM_TEX_MASK),
                rtAdapter->outputHandle(rt_layout::OUT_BINDING_PARTICLE_MASK),
                rtAdapter->fhBaseEmissionHandle(),
                rtAdapter->outputHandle(rt_layout::OUT_BINDING_BIAS_MASK),
                rtAdapter->outputHandle(rt_layout::OUT_BINDING_RT_HIT_DIST),
                rtAdapter->motionVec3dHandle(),
                rtAdapter->gbufferMetallicHandle(),
                rtAdapter->outputHandle(rt_layout::OUT_BINDING_GBUFFER_SHADING_MODEL_ID),
                rtAdapter->outputHandle(rt_layout::OUT_BINDING_GBUFFER_MATERIAL_ID),
                rtAdapter->positionViewSpaceHandle(),
                rtAdapter->outputHandle(rt_layout::OUT_BINDING_TRANSPARENCY_LAYER),
                rtAdapter->outputHandle(rt_layout::OUT_BINDING_TRANSPARENCY_OPACITY),
                rtAdapter->outputHandle(rt_layout::OUT_BINDING_TRANSPARENCY_MVECS));
        } else {
            // Fallback: temporal denoiser (no DLSS-RR).
            denoiseAdapter->registerPass(builder,
                                         rtRadiance,
                                         rtAdapter->normalHandle(),
                                         rtAdapter->linearDepthHandle(),
                                         rtAdapter->motionHandle());
            toneMapInput = denoiseAdapter->outputHandle();
        }

        // Cloud pass: insert AFTER DLSS/denoiser output, BEFORE tone mapping.
        // If the cloud adapter failed to initialize (shaders missing, etc.) we skip it
        // and tone mapping receives the DLSS/denoiser output directly.
        if (fullRtActive && cloudAdapter_ && cloudAdapter_->isInitialized()) {
            toneMapInput = cloudAdapter_->registerPass(builder,
                                                       toneMapInput,
                                                       rtAdapter->linearDepthHandle());
        }

        tmAdapter->registerPass(builder, toneMapInput);
        // ToneMapping declares its output via outputHandle_; PostProcess takes over as final.
        ppAdapter->registerPass(builder, tmAdapter->outputHandle());

        // Post-RT passes: color-to-depth conversion, entity rasterization, star field.
        // These run after CAS/bloom and composite over the final color output.
        // IMPORTANT: PostRender and StarField must be SEQUENTIAL (not both in-place
        // on the same resource) to avoid a graph cycle. PostRender writes in-place
        // on ppOutput. StarField then reads ppOutput as input-only (no output decl)
        // to avoid the bidirectional edge that would create a cycle.
        if (fullRtActive) {
            c2dAdapter->registerPass(builder, rtAdapter->firstHitDepthHandle());
            prAdapter->registerPass(builder, ppAdapter->outputHandle(), c2dAdapter->orderingHandle());
            sfAdapter->registerPass(builder, prAdapter->outputHandle());
        }

        auto graph = builder.compile();

        if (graph.valid()) {
            // Set render resolution BEFORE setGraph() so the first resource allocation
            // in executeGraphFrame uses the correct dimensions. When DLSS is active,
            // RT resources (widthScale=1.0) are sized to the DLSS render (input) resolution;
            // the DLSS output resource has fixedWidth/Height set to display resolution.
            // When DLSS is inactive (or DLAA), render res == display res, so 0,0 → swapchain extent.
            if (dlssActive) {
                uint32_t renderW = dlssAdapter_->renderInputWidth()  > 0 ? dlssAdapter_->renderInputWidth()  : ext.width;
                uint32_t renderH = dlssAdapter_->renderInputHeight() > 0 ? dlssAdapter_->renderInputHeight() : ext.height;
                services_->frame().setGraphRenderResolution(renderW, renderH);
                log::info("app", "Graph render resolution: " + std::to_string(renderW) + "x" + std::to_string(renderH)
                          + " (display: " + std::to_string(ext.width) + "x" + std::to_string(ext.height) + ")");
            } else {
                services_->frame().setGraphRenderResolution(0, 0);  // use swapchain extent
            }
            services_->frame().setGraph(std::move(graph));
            const bool cloudActive = fullRtActive && cloudAdapter_ && cloudAdapter_->isInitialized();
            if (dlssActive) {
                log::info("app", std::string("Render graph active: RT -> DLSS-RR ->") +
                          (cloudActive ? " Cloud ->" : "") +
                          " ToneMapping -> PostProcess(CAS) -> C2D -> PostRender -> StarField -> Composite");
            } else if (fullRtActive) {
                log::info("app", std::string("Render graph active: RT -> TemporalDenoiser ->") +
                          (cloudActive ? " Cloud ->" : "") +
                          " ToneMapping -> PostProcess(CAS) -> C2D -> PostRender -> StarField -> Composite");
            } else {
                log::warn("app",
                          "Render graph active: RT(v2_min stability path) -> TemporalDenoiser -> "
                          "ToneMapping -> PostProcess -> Composite");
            }
        }

        adapters_.push_back(std::move(rtAdapter));
        adapters_.push_back(std::move(denoiseAdapter));
        adapters_.push_back(std::move(dlssAdapter));
        adapters_.push_back(std::move(cloudAdapter));
        adapters_.push_back(std::move(tmAdapter));
        adapters_.push_back(std::move(ppAdapter));
        adapters_.push_back(std::move(c2dAdapter));
        adapters_.push_back(std::move(prAdapter));
        adapters_.push_back(std::move(sfAdapter));
        adapters_.push_back(std::move(fgAdapter));
        boot_trace::breadcrumb("graph", "ok",
            std::to_string(adapters_.size()) + " adapters registered");
    }

    boot_trace::breadcrumb("init", "ok");
    // Freeze the on-disk boot log: the "last breadcrumb" cache keeps updating
    // from this point (so the crash classifier still gets the most recent
    // stage), but steady-state tick breadcrumbs no longer spam v2_boot.log.
    // Unfreeze happens in shutdown() so teardown stages are persisted.
    boot_trace::freezeBootLog();
    return true;
}

bool EngineApp::tick() {
    if (!initialized_ || !session_ || session_->state() == SessionState::ShuttingDown) {
        return false;
    }

    // Check for device-lost from previous frame
    if (mode_ == EngineMode::V2 && services_->frame().isDeviceLost()) {
        log::error("app", "Device lost detected — shutting down V2 engine");
        boot_trace::writeCrashDump("device lost (mid-tick)",
                                   static_cast<int>(VK_ERROR_DEVICE_LOST));
        session_->setState(SessionState::ShuttingDown);
        return false;
    }

    // Bump the frame counter so the crash classifier knows the most recent
    // tick index. Cheap atomic store, in-memory only.
    boot_trace::recordFrame(tickCount_++);

    // Periodic heartbeat to v2_tick.log so a hang (tick thread parked on a
    // fence, an infinite wait, etc.) is obvious by the log's mtime. 120 frames
    // ≈ 2s at 60 fps / 0.5s at 240 fps — low enough overhead to stay on in
    // release builds.
    if ((tickCount_ & 127u) == 0u) {
        std::string extra = "mode=" + std::string(mode_ == EngineMode::V2 ? "V2" : "V1")
                          + " inWorld=" + (inWorld_ ? "1" : "0");
        boot_trace::writeTickHeartbeat(tickCount_, extra);
    }

    if (mode_ == EngineMode::V2) {
        // Flush bridge and tick GC every frame regardless
        auto ctx = services_->frame().beginFrame();
        ctx.camera = latestCamera_;

        // Reflex/PCL: mark simulation start for the current frame.
        // Required every frame regardless of Reflex on/off state.
        if (fgAdapter_) fgAdapter_->pclSimulationStart();

        if (!inWorld_ && !services_->swapchain().isZeroExtent()) {
            // Not in world — run executeClearFrame so the swapchain gets a present
            // (cycling black clear). Without this the Win32 HWND surface is never
            // painted and shows solid white (the system COLOR_WINDOW brush).
            // executeClearFrame handles its own fence wait, gc_.tick(), bridge().flush(),
            // acquire, record, submit and present — exactly what the menu needs.
            bool skipped = services_->frame().executeClearFrame(
                services_->device(), services_->swapchain(), ctx);
            if (!skipped) {
                services_->frame().endFrame(ctx);
            }
        } else if (services_->swapchain().isZeroExtent()) {
            // Minimized — skip present entirely, just drain pending recreates.
            if (services_->swapchain().isRecreateNeeded()) {
                services_->device().waitIdle();
                services_->swapchain().recreate(0, 0);
            }
            services_->frame().endFrame(ctx);
        } else {
            bool skipped;
            if (services_->frame().hasGraph()) {
                skipped = services_->frame().executeGraphFrame(
                    services_->device(), services_->swapchain(), ctx);
            } else {
                skipped = services_->frame().executeOffscreenFrame(
                    services_->device(), services_->swapchain(),
                    services_->offscreen(), ctx);
            }
            // Reflex sleep: after GPU submit/present, before next frame processing.
            // Reduces CPU-GPU latency when Reflex is enabled. Safe no-op when disabled.
            if (fgAdapter_) {
#ifdef _WIN32
                if (StreamlineContext::isAvailable() &&
                        services_->config().live().reflexEnabled) {
                    StreamlineContext::reflexSleep();
                }
#endif
            }
            if (!skipped) {
                services_->frame().endFrame(ctx);
            }
        }
    } else {
        // Legacy mode: just flush bridge and tick GC
        auto ctx = services_->frame().beginFrame();
        services_->frame().endFrame(ctx);
    }

    return true;
}

void EngineApp::shutdown() {
    if (!initialized_) return;

    // Re-arm the boot log so every shutdown stage is persisted to v2_boot.log.
    // If the teardown crashes, the last line in the log names the stage.
    boot_trace::unfreezeBootLog();
    boot_trace::breadcrumb("shutdown", "begin");

    log::info("app", "EngineApp shutting down");

    if (session_) {
        session_->setState(SessionState::ShuttingDown);
        session_.reset();
    }

    if (mode_ == EngineMode::V2) {
        // Breadcrumb logs — if shutdown crashes mid-teardown, the last log
        // line pinpoints exactly which stage died. Needed after the
        // 2026-04-08 shutdown crash (hs_err_pid23164) where the log just
        // stopped after the 3 adapter shutdowns with no indication why.
        log::info("app", "shutdown: waitIdle");
        services_->device().waitIdle();

        log::info("app", "shutdown: atmosphere.shutdown");
        atmosphere_.shutdown();

        log::info("app", "shutdown: adapters.shutdown");
        for (auto& adapter : adapters_) adapter->shutdown();

        log::info("app", "shutdown: adapters.clear (destructors)");
        adapters_.clear();
        // Null out the raw non-owning pointers that referred into adapters_.
        dlssAdapter_  = nullptr;
        cloudAdapter_ = nullptr;
        rtAdapter_    = nullptr;
        tmAdapter_    = nullptr;
        prAdapter_    = nullptr;
        fgAdapter_    = nullptr;

        log::info("app", "shutdown: frame.resourcePool.releaseImmediate");
        services_->frame().resourcePool().releaseImmediate();

        log::info("app", "shutdown: tlas.shutdown");
        services_->tlas().shutdown();

        log::info("app", "shutdown: blas.shutdown");
        services_->blas().shutdown();

        log::info("app", "shutdown: entityBlas.shutdown");
        services_->entityBlas().shutdown();

        log::info("app", "shutdown: gpuUpload.shutdown");
        services_->gpuUpload().shutdown();

        log::info("app", "shutdown: sceneRes.shutdown");
        services_->sceneRes().shutdown();

        log::info("app", "shutdown: texture.shutdown");
        services_->texture().shutdown();

        log::info("app", "shutdown: offscreen.shutdown");
        services_->offscreen().shutdown();

        log::info("app", "shutdown: metrics.shutdown");
        services_->metrics().shutdown();

        log::info("app", "shutdown: frame.shutdownSync");
        services_->frame().shutdownSync();

        log::info("app", "shutdown: swapchain.shutdown");
        services_->swapchain().shutdown();

        // CRITICAL: drain the ResourceGC ring BEFORE destroying the device.
        //
        // Deferred destructors captured VkDevice by value at the time of deferral.
        // Over the run, chunk BLAS retirements (retireBlas), TLAS retirements
        // (retireTlas), and chunk buffer retirements (retireChunk) all pile up in
        // the GC ring with lambdas that will call vkDestroy*() on those captured
        // handles. If we let ~FrameScheduler()'s gc_.flush() run from services_.reset()
        // BELOW device.shutdown(), those destructors fire against an already-destroyed
        // VkDevice — the classic use-after-free that lands deep inside nvoglv64.dll
        // (NVIDIA's Vulkan ICD). At rd=8 the ring usually drained during normal ticks
        // so shutdown was silently correct; at rd=32 there are ~50-80 accumulated
        // deferrals from ambient world rebuilds and we always crash.
        //
        // Flush here (device still valid) so the ring is empty when ~FrameScheduler
        // runs. Second flush in the destructor is a safe no-op.
        log::info("app", "shutdown: frame.gc.flush");
        services_->frame().gc().flush();

        log::info("app", "shutdown: device.shutdown");
        services_->device().shutdown();
    }

    log::info("app", "shutdown: services.reset");
    services_.reset();

    log::info("app", "EngineApp shutdown complete");
    boot_trace::breadcrumb("shutdown", "ok");
    log::shutdown();

    initialized_ = false;
    s_instance = nullptr;
}

EngineApp* EngineApp::get() {
    return s_instance;
}

// --- Command handlers ---

void EngineApp::handleCommand(const CmdPing& cmd) {
    log::debug("bridge", "Ping received, ts=" + std::to_string(cmd.timestamp));
}

void EngineApp::handleCommand(const CmdWindowResize& cmd) {
    log::info("bridge", "Window resize: " + std::to_string(cmd.width) + "x" + std::to_string(cmd.height));
    if (mode_ == EngineMode::V2) {
        // Always mark recreate needed — even if swapchain is not currently initialized
        // (e.g., after zero-extent destroyed it). The tick loop will handle recovery.
        services_->swapchain().markRecreateNeeded();
        if (services_->swapchain().isInitialized()) {
            services_->device().waitIdle();
            // Notify DLSS-G before teardown; after recreate let it re-attach.
            if (fgAdapter_) fgAdapter_->beforeSwapchainRecreate();
            auto r = services_->swapchain().recreate(cmd.width, cmd.height);
            if (!r) log::error("bridge", "Swapchain recreate failed: " + r.error().message);
            if (fgAdapter_) fgAdapter_->afterSwapchainRecreate();
        }
    }
}

void EngineApp::handleCommand(const CmdWorldLoad& cmd) {
    log::info("bridge", "World load: " + cmd.regionPath);
    inWorld_ = true;
    // Reset DLSS temporal history so the first in-world frame sends InReset=1.
    // Without this, DLSS-RR carries over stale history accumulated during menu
    // rendering (sky-blue radiance + zero G-buffers), causing white-screen
    // divergence on the first few frames after entering the world.
    if (dlssAdapter_) dlssAdapter_->resetHistory();
}

void EngineApp::handleCommand(const CmdWorldUnload&) {
    log::info("bridge", "World unload");
    inWorld_ = false;
    latestCamera_.valid = false;

    // Reset accumulation counter — world geometry is gone, any accumulated
    // frames would composite over stale data in the next world.
    resetAccum();

    // Discard any pending entity data from the old world
    pendingEntityBatch_.reset();
    pendingPostDraw_.reset();

    // Clear scene state so stale geometry doesn't bleed into the next world.
    // Order matters: BLAS/GPU data must be retired before the registry is cleared,
    // because the GC ring captures VkDevice handles that remain valid until flush.
    if (services_) {
        // Wait for all in-flight GPU work to complete before clearing
        if (mode_ == EngineMode::V2 && services_->device().isInitialized()) {
            services_->device().waitIdle();
        }

        if (services_->entityBlas().isInitialized()) {
            services_->entityBlas().clear();
        }
        if (services_->blas().isInitialized()) {
            services_->blas().clearAll();
        }
        if (services_->gpuUpload().isInitialized()) {
            services_->gpuUpload().clearAll();
        }
        services_->scene().clear();

        log::info("bridge", "World unload: scene cleared (chunks, BLAS, GPU uploads, entities)");
    }
}



void EngineApp::handleCommand(const CmdShutdown&) {
    log::info("bridge", "Shutdown requested via bridge");
    if (session_) session_->setState(SessionState::ShuttingDown);
}

void EngineApp::handleCommand(const CmdConfigPatch& cmd) {
    cmd.apply();
}

void EngineApp::handleCommand(const CmdResetAccumulation&) {
    resetAccum();
    log::debug("app", "Offline accumulation reset via bridge command");
}

void EngineApp::handleCommand(const CmdChunkSubmit& cmd) {
    ChunkGeometry geo;
    geo.id = {cmd.chunkX, cmd.sectionY, cmd.chunkZ};
    geo.vertexData = cmd.vertexData;
    geo.indexData = cmd.indexData;
    geo.triangleCount = cmd.triangleCount;
    geo.originX = static_cast<float>(cmd.originX);
    geo.originY = static_cast<float>(cmd.originY);
    geo.originZ = static_cast<float>(cmd.originZ);

    // Unpack per-chunk light entries (16 bytes each: float x, float y, float z, int typeId)
    if (cmd.lightCount > 0 && cmd.lightData.size() >= cmd.lightCount * 16) {
        geo.lightEntries.resize(cmd.lightCount);
        const uint8_t* src = cmd.lightData.data();
        for (uint32_t i = 0; i < cmd.lightCount; ++i) {
            std::memcpy(&geo.lightEntries[i].worldX, src + 0, sizeof(float));
            std::memcpy(&geo.lightEntries[i].worldY, src + 4, sizeof(float));
            std::memcpy(&geo.lightEntries[i].worldZ, src + 8, sizeof(float));
            std::memcpy(&geo.lightEntries[i].lightTypeId, src + 12, sizeof(int32_t));
            src += 16;
        }
    }

    // Diagnostic: log first 5 chunks then every 200th
    static std::atomic<uint32_t> chunkSubmitCount{0};
    uint32_t n = chunkSubmitCount.fetch_add(1) + 1;
    if (n <= 5 || n % 200 == 0) {
        log::info("app", "CmdChunkSubmit #" + std::to_string(n) + ": id=("
            + std::to_string(geo.id.x) + "," + std::to_string(geo.id.y) + ","
            + std::to_string(geo.id.z) + ") tris=" + std::to_string(geo.triangleCount)
            + " vbytes=" + std::to_string(geo.vertexData.size())
            + " ibytes=" + std::to_string(geo.indexData.size() * sizeof(uint32_t))
            + " origin=(" + std::to_string(cmd.originX) + ","
            + std::to_string(cmd.originY) + "," + std::to_string(cmd.originZ) + ")");
    }

    ChunkId insertedId = geo.id;  // capture before move
    services_->scene().chunks().insert(insertedId, std::move(geo));

    if (replayRecorder_) {
        replayRecorder_->recordChunkInsert(
            insertedId,
            cmd.vertexData.data(), static_cast<uint32_t>(cmd.vertexData.size()),
            cmd.indexData.data(), static_cast<uint32_t>(cmd.indexData.size() * sizeof(uint32_t)));
    }
}

void EngineApp::handleCommand(const CmdChunkRemove& cmd) {
    ChunkId id{cmd.chunkX, cmd.sectionY, cmd.chunkZ};
    services_->scene().chunks().remove(id);
    services_->gpuUpload().removeChunk(id);
    services_->blas().removeChunk(id);

    if (replayRecorder_) {
        replayRecorder_->recordChunkRemove(id);
    }
}

void EngineApp::processScene(VkCommandBuffer cmd) {
    // Reflex/PCL: mark render submit start.
    // Must be called each frame before GPU work begins, regardless of Reflex state.
    if (fgAdapter_) fgAdapter_->pclRenderStart();

    // Finalize textures if pending (records transfer commands into this cmd buffer)
    if (services_->texture().isInitialized() && !services_->texture().isFinalized()) {
        services_->texture().finalize(cmd);

        // V2 mode normalizes BlockModelTable quad UVs from atlas space into
        // sprite-local [0,1] using V2's TextureService bounds. The V1 path in
        // BlockModelBridge skips this in V2 because Renderer::is_initialized()
        // is false. Without this, the v2_world.rchit shader samples near (0,0)
        // of every sprite layer and all blocks render flat-colored.
        if (Renderer::blockModelTable.isLoaded()) {
            auto& tex = services_->texture();
            // DIAG: spriteId histogram (first 8 buckets) over a sample of blocks
            // before normalization happens. Helps confirm whether quads carry
            // varied spriteIds (the all-same-texture symptom could mean every
            // quad has the same spriteId, not just same baseLayer mapping).
            // Sample faces 0..5 (real cube faces) of the first ~50 model-bearing block states.
            // Print actual spriteId values so we can see if they're genuinely varied.
            uint32_t blockSampled = 0;
            std::string sample;
            sample.reserve(2048);
            uint32_t uniqueIds[64];
            uint32_t uniqueCount = 0;
            uint32_t totalQuads = 0;
            for (uint32_t s = 1; s < Renderer::blockModelTable.maxStateId() && blockSampled < 50; ++s) {
                const auto* e = Renderer::blockModelTable.getEntry(s);
                if (!e || e->renderType != 1 || e->totalQuadCount == 0) continue;
                sample.append(" S").append(std::to_string(s)).append(":");
                for (uint8_t dir = 0; dir < 6; ++dir) {
                    uint8_t cnt = 0;
                    const auto* q = Renderer::blockModelTable.getFaceQuads(*e, dir, cnt);
                    for (uint8_t qi = 0; qi < cnt; ++qi) {
                        uint16_t sid = q[qi].spriteId;
                        sample.append(std::to_string(sid)).append(",");
                        ++totalQuads;
                        bool found = false;
                        for (uint32_t u = 0; u < uniqueCount; ++u)
                            if (uniqueIds[u] == sid) { found = true; break; }
                        if (!found && uniqueCount < 64) uniqueIds[uniqueCount++] = sid;
                    }
                }
                ++blockSampled;
                if (sample.size() > 1500) break;
            }
            log::warn("app",
                "DIAG facequad spriteIds (" + std::to_string(blockSampled) + " blocks, "
                + std::to_string(totalQuads) + " quads, " + std::to_string(uniqueCount)
                + " unique):" + sample);

            Renderer::blockModelTable.normalizeQuadUVsWithBounds(
                [&tex](uint16_t spriteId) -> glm::vec4 {
                    return tex.spriteBounds(spriteId);
                });
        }
    }

    // First-frame deferred init: clear energy LUT to 1.0.
    // Idempotent — returns false on subsequent calls.
    if (services_->sceneRes().isInitialized()) {
        services_->sceneRes().runDeferredInit(cmd);
    }

    // Inject DLSS-RR jitter into scene resources before WorldUBO update.
    // The DlssAdapter advances the Halton(2,3) sequence each frame in execute();
    // currentJitter() returns the jitter for THIS frame (set at end of previous execute).
    if (dlssAdapter_ && dlssAdapter_->isFeatureCreated()) {
        auto [jx, jy] = dlssAdapter_->currentJitter();
        services_->sceneRes().setJitter(jx, jy);
    }

    // Pre-exposure: must be CONSTANT across frames — varying pre-exposure contaminates
    // DLSS-RR's temporal history (accumulated at different scales) and causes the NGX
    // network's history reprojection to dereference a stale/invalid address, producing
    // a READ_INVALID GPU fault and TDR. 0.1f matches V1's hard-coded value
    // (ray_tracing_module.cpp:1754) and maps Minecraft's typical luminance range into
    // fp16's sweet spot. DLSS-RR undoes it via InExposureScale = 1/0.1 = 10.
    constexpr float kDlssPreExposure = 0.1f;
    if (rtAdapter_)   rtAdapter_->setPreExposure(kDlssPreExposure);
    if (dlssAdapter_) dlssAdapter_->setPreExposure(kDlssPreExposure);

    // Offline accumulation state update.
    // When enabled: detect camera movement → reset counter; otherwise increment.
    // The counter is forwarded to the RT push constants via setAccumState() so
    // the RT shader can perform Welford running-average accumulation each frame.
    if (accumState_.enabled && latestCamera_.valid) {
        constexpr float kMoveTolerance = 0.001f;
        bool moved =
            std::abs(latestCamera_.posX - accumState_.prevCamPosX) > kMoveTolerance ||
            std::abs(latestCamera_.posY - accumState_.prevCamPosY) > kMoveTolerance ||
            std::abs(latestCamera_.posZ - accumState_.prevCamPosZ) > kMoveTolerance ||
            std::abs(latestCamera_.dirX - accumState_.prevCamDirX) > kMoveTolerance ||
            std::abs(latestCamera_.dirY - accumState_.prevCamDirY) > kMoveTolerance ||
            std::abs(latestCamera_.dirZ - accumState_.prevCamDirZ) > kMoveTolerance;
        if (moved) {
            accumState_.accumFrameCount = 0;
        } else {
            ++accumState_.accumFrameCount;
        }
        accumState_.prevCamPosX = latestCamera_.posX;
        accumState_.prevCamPosY = latestCamera_.posY;
        accumState_.prevCamPosZ = latestCamera_.posZ;
        accumState_.prevCamDirX = latestCamera_.dirX;
        accumState_.prevCamDirY = latestCamera_.dirY;
        accumState_.prevCamDirZ = latestCamera_.dirZ;
    }
    // Forward accum state to the RT adapter every frame (setAccumState is always safe).
    if (rtAdapter_) {
        auto cfgSnap = services_->config().snapshot();
        float offAperture   = cfgSnap ? cfgSnap->data.offlineAperture      : 0.0f;
        float offFocalDist  = cfgSnap ? cfgSnap->data.offlineFocalDistance  : 10.0f;
        uint32_t offBounces = cfgSnap ? cfgSnap->data.offlineBounces        : 0u;
        rtAdapter_->setAccumState(
            accumState_.enabled ? accumState_.accumFrameCount : 0u,
            accumState_.enabled,
            offAperture,
            offFocalDist,
            accumState_.enabled ? offBounces : 0u);
    }

    // Update scene resources (WorldUBO with current camera/config)
    if (services_->sceneRes().isInitialized() && latestCamera_.valid) {
        auto cfg = services_->config().snapshot();
        if (cfg) {
            services_->sceneRes().updateWorldUBO(
                services_->frame().currentFrameIndex(),
                latestCamera_, *cfg, services_->frame().frameNumber());
        }
    }

    // Atmosphere LUT (once) + sky cubemap (every frame)
    if (atmosphere_.isInitialized()) {
        atmosphere_.recordCommands(cmd, services_->frame().currentFrameIndex());
    }

    // Extract scene snapshot
    auto scene = services_->scene().extractFrame(services_->frame().frameNumber());
    if (!scene) return;

    // Propagate diagnostic flags to services that gate structured log::event calls.
    // Use g_effectiveDiagFlags (diagFlags | level-implied flags) so that diagLevel=2
    // automatically enables TLAS/BLAS/UPLOAD/FRAME events without requiring the user
    // to set diagFlags explicitly.
    {
        int effectiveFlags = engine::g_effectiveDiagFlags.load(std::memory_order_relaxed);
        services_->gpuUpload().setDiagFlags(effectiveFlags);
        services_->blas().setDiagFlags(effectiveFlags);
        services_->tlas().setDiagFlags(effectiveFlags);
    }

    // Upload dirty chunks
    uint32_t uploaded = 0;
    uint32_t reQueued = 0;
    if (!scene->dirtyChunkGeometries.empty() && services_->gpuUpload().isInitialized()) {
        auto uploadResult = services_->gpuUpload().uploadDirtyChunks(
            cmd, scene->dirtyChunkGeometries, services_->frame().currentFrameIndex());
        uploaded = uploadResult.uploaded;

        // Re-mark skipped chunks as dirty so they're retried next frame.
        // Without this, chunks that were dropped due to staging ring exhaustion
        // would be silently lost (the SceneService already called clearDirty()
        // before handing us the snapshot).
        if (!uploadResult.skipped.empty()) {
            auto& chunks = services_->scene().chunks();
            for (const auto& id : uploadResult.skipped) {
                chunks.markDirty(id);
            }
            reQueued = static_cast<uint32_t>(uploadResult.skipped.size());
        }
    }

    // Diagnostic: per-frame stats every 60 frames + first frame
    static uint64_t lastDiagFrame = 0;
    static uint32_t lastUploaded = 0, lastBuilt = 0, lastTlasInst = 0;
    uint64_t fnum = services_->frame().frameNumber();
    if (services_->config().live().diagLevel >= 1 &&
            (uploaded > 0 || reQueued > 0 || (fnum - lastDiagFrame) >= 60)) {
        log::info("scene", "frame=" + std::to_string(fnum)
            + " dirty=" + std::to_string(scene->dirtyChunkGeometries.size())
            + " uploaded=" + std::to_string(uploaded)
            + " reQueued=" + std::to_string(reQueued)
            + " gpuUpload.init=" + std::to_string(services_->gpuUpload().isInitialized() ? 1 : 0)
            + " allChunks=" + std::to_string(services_->gpuUpload().allChunks().size()));
        lastDiagFrame = fnum;
        lastUploaded = uploaded;
    }

    // Barrier: upload copies must complete before BLAS builds read vertex/index data,
    // AND before the RT shader reads vertex/index data directly via stored BDAs.
    // Two destination stages are needed:
    //   1. ACCELERATION_STRUCTURE_BUILD — so BLAS build sees the vertex/index writes
    //   2. RAY_TRACING_SHADER — so the RT closest-hit shader sees vertex/index data
    //      when it dereferences the BDA (vertexBufferAddrs.addrs[i]) stored in the SSBO.
    //      The TLAS barrier covers AS_READ and SHADER_STORAGE_READ for the BDA-array
    //      SSBO itself, but NOT the BDA-pointed vertex/index buffer contents, which
    //      were written by TRANSFER, not by the AS build. Without this dstStage coverage
    //      the RT shader can read stale vertex data and compute garbage world positions,
    //      or dereference a partially-written BDA — producing the recurring instruction-
    //      fetch DEVICE_FAULT at 0x14012add00 / 0x14012add60.
    if (uploaded > 0) {
        VkMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
                              | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
                              | VK_ACCESS_2_SHADER_READ_BIT;

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // Build BLAS for dirty chunks
    uint32_t built = 0;
    if (services_->blas().isInitialized()) {
        std::vector<ChunkId> dirtyIds;
        for (const auto& geo : scene->dirtyChunkGeometries) {
            if (!geo.empty()) dirtyIds.push_back(geo.id);
        }
        if (!dirtyIds.empty()) {
            auto blasResult = services_->blas().buildDirtyChunks(
                cmd, dirtyIds, services_->gpuUpload().allChunks(),
                services_->frame().currentFrameIndex());
            built = blasResult.built;

            // Re-mark BLAS-skipped chunks as dirty for next-frame retry.
            // These chunks made it through upload but couldn't fit in the BLAS
            // scratch buffer this frame.
            if (!blasResult.skipped.empty()) {
                auto& chunks = services_->scene().chunks();
                for (const auto& id : blasResult.skipped) {
                    chunks.markDirty(id);
                }
                reQueued += static_cast<uint32_t>(blasResult.skipped.size());
            }
        }
    }

    // --- Entity BLAS building ---
    if (services_->entityBlas().isInitialized()) {
        services_->entityBlas().clear();
        if (pendingEntityBatch_ && !pendingEntityBatch_->entities.empty()) {
            std::vector<EntityBatchEntry> entries;
            entries.reserve(pendingEntityBatch_->entities.size());
            for (const auto& e : pendingEntityBatch_->entities) {
                EntityBatchEntry be;
                be.hashCode = e.hashCode;
                be.posX = e.posX;
                be.posY = e.posY;
                be.posZ = e.posZ;
                be.rtFlag = e.rtFlag;
                be.coordSystem = e.coordSystem;
                be.vertexOffset = e.vertexOffset;
                be.indexOffset = e.indexOffset;
                be.triangleCount = e.triangleCount;
                entries.push_back(be);
            }
            uint32_t entityBuilt = services_->entityBlas().submitBatch(
                cmd,
                std::move(pendingEntityBatch_->vertexData),
                std::move(pendingEntityBatch_->indexData),
                entries,
                services_->frame().currentFrameIndex());
            if (entityBuilt > 0) {
                static uint64_t lastEntityLog = 0;
                if (fnum - lastEntityLog >= 60) {
                    log::info("scene", "Entity BLASes built: " + std::to_string(entityBuilt));
                    lastEntityLog = fnum;
                }
            }
            pendingEntityBatch_.reset();
        }
    }

    // --- Post-entity draw data (particles, hand, weather) ---
    if (prAdapter_ && pendingPostDraw_ && !pendingPostDraw_->drawCalls.empty()) {
        prAdapter_->setPendingDrawData(std::move(*pendingPostDraw_));
        pendingPostDraw_.reset();
    } else if (pendingPostDraw_) {
        pendingPostDraw_.reset();  // empty batch, discard
    }

    // Rebuild TLAS
    if (services_->tlas().isInitialized() &&
        (services_->blas().blasCount() > 0 || services_->entityBlas().entityCount() > 0)) {
        const EntityBlasService* entSvc = services_->entityBlas().isInitialized()
            ? &services_->entityBlas() : nullptr;
        const CameraData* cam = latestCamera_.valid ? &latestCamera_ : nullptr;
        services_->tlas().rebuild(cmd, services_->blas(),
                                   services_->frame().currentFrameIndex(),
                                   entSvc, cam);
    }

    // FrameGenAdapter: supply camera data for SL constants (used in tagFrame).
    if (fgAdapter_ && latestCamera_.valid) {
        fgAdapter_->setCameraData(latestCamera_);
    }

    // --- Area light gathering ---
    // Iterate all chunk geometries, collect per-chunk light entries near the camera,
    // convert to GPU AreaLight structs (48 bytes each), sort by contribution, cap at 512,
    // and upload to the area light SSBO.
    //
    // GUARD: Only gather from chunk registry if chunks have been submitted to V2.
    // If the registry is empty, skip — Java's AreaLightPacker may have already
    // uploaded lights via CmdAreaLightUpload, and we must not overwrite them with zeros.
    if (latestCamera_.valid && services_->sceneRes().isInitialized()
        && services_->scene().chunks().size() > 0) {
        auto cfgSnap = services_->config().snapshot();
        bool areaLightsEnabled = cfgSnap && cfgSnap->data.areaLightsEnabled;

        if (areaLightsEnabled) {
            constexpr float MAX_RANGE = 48.0f;
            constexpr int MAX_AREA_LIGHTS = 512;
            constexpr float VERTICAL_CULL_BELOW = 48.0f;
            float camX = static_cast<float>(latestCamera_.posX);
            float camY = static_cast<float>(latestCamera_.posY);
            float camZ = static_cast<float>(latestCamera_.posZ);

            struct GatheredLight {
                vk::Data::AreaLight al;
                float contribution;
            };
            std::vector<GatheredLight> gathered;
            gathered.reserve(256);

            services_->scene().chunks().forEachGeometry([&](const ChunkGeometry& geo) {
                for (const auto& entry : geo.lightEntries) {
                    if (entry.lightTypeId < 0 || entry.lightTypeId >= LIGHT_TYPE_COUNT) continue;

                    const auto& def = LIGHT_DEFS[entry.lightTypeId];

                    // Camera-relative position
                    float rx = entry.worldX - camX;
                    float ry = (entry.worldY + def.yOffset) - camY;
                    float rz = entry.worldZ - camZ;

                    // Vertical cull
                    if (ry < -VERTICAL_CULL_BELOW) continue;

                    // Distance cull
                    float d2 = rx * rx + ry * ry + rz * rz;
                    float maxRange = MAX_RANGE;
                    if (d2 > maxRange * maxRange) continue;

                    // Compute intensity
                    float intensity = def.lumens * LUMENS_TO_INTENSITY;

                    // Contribution score for sorting (brightest/nearest first)
                    float contrib = intensity / std::max(d2, 1.0f);

                    vk::Data::AreaLight al{};
                    al.position = glm::vec3(rx, ry, rz);
                    al.halfExtent = def.halfExtent;
                    al.color = glm::vec3(def.color);
                    al.intensity = intensity;
                    al.radius = maxRange;

                    gathered.push_back({al, contrib});
                }
            });

            // Sort by contribution (brightest/nearest first), cap at 512
            if (gathered.size() > MAX_AREA_LIGHTS) {
                std::partial_sort(gathered.begin(), gathered.begin() + MAX_AREA_LIGHTS,
                                  gathered.end(),
                                  [](const GatheredLight& a, const GatheredLight& b) {
                                      return a.contribution > b.contribution;
                                  });
                gathered.resize(MAX_AREA_LIGHTS);
            } else if (!gathered.empty()) {
                std::sort(gathered.begin(), gathered.end(),
                          [](const GatheredLight& a, const GatheredLight& b) {
                              return a.contribution > b.contribution;
                          });
            }

            // Upload to SSBO
            if (!gathered.empty()) {
                std::vector<vk::Data::AreaLight> gpuLights(gathered.size());
                for (size_t i = 0; i < gathered.size(); ++i) {
                    gpuLights[i] = gathered[i].al;
                }
                services_->sceneRes().uploadAreaLights(
                    gpuLights.data(),
                    gpuLights.size() * sizeof(vk::Data::AreaLight),
                    static_cast<uint32_t>(gpuLights.size()));
            } else {
                services_->sceneRes().uploadAreaLights(nullptr, 0, 0);
            }
        } else {
            // Area lights disabled — ensure count is zero
            services_->sceneRes().uploadAreaLights(nullptr, 0, 0);
        }
    }

    // Diagnostic: log BLAS/TLAS state when something changed or every 60 frames
    if (services_->config().live().diagLevel >= 1 &&
            (built > 0 || (fnum - lastDiagFrame == 0 && uploaded > 0) || services_->blas().blasCount() != lastTlasInst)) {
        log::info("scene", "frame=" + std::to_string(fnum)
            + " built=" + std::to_string(built)
            + " blasCount=" + std::to_string(services_->blas().blasCount())
            + " tlasInst=" + std::to_string(services_->tlas().instanceCount())
            + " tlasValid=" + std::to_string(services_->tlas().isValid() ? 1 : 0));
        lastBuilt = built;
        lastTlasInst = services_->blas().blasCount();
    }

    // Frame complete structured event (DiagFlags::FRAME + diagLevel >= 2)
    {
        auto& liveCfg = services_->config().live();
        if (liveCfg.diagLevel >= 2 && (liveCfg.diagFlags & DiagFlags::FRAME) != 0) {
            auto vram = services_->metrics().queryVram();
            log::event("frame", "frame_complete", {
                {"frame",        std::to_string(fnum)},
                {"cpuMs",        std::to_string(services_->metrics().cpuFrameTimeAvg())},
                {"gpuMs",        std::to_string(services_->metrics().gpuFrameTimeMs())},
                {"hitchCount",   std::to_string(services_->metrics().hitchCount())},
                {"vramUsageMB",  std::to_string(vram.usageBytes / 1048576)},
                {"vramBudgetMB", std::to_string(vram.budgetBytes / 1048576)},
                {"tlasInst",     std::to_string(services_->tlas().instanceCount())},
                {"blasCount",    std::to_string(services_->blas().blasCount())}
            });
        }
    }

    // Per-frame GPU profile log (DiagFlags::FRAME gate)
    {
        auto cfgSnap = services_->config().snapshot();
        if (cfgSnap && (cfgSnap->data.diagFlags & DiagFlags::FRAME) != 0 && (fnum % 60 == 0)) {
            log::debug("metrics", "gpu=" + services_->metrics().getProfileString());
        }
    }

    // Replay recorder: close frame record
    if (replayRecorder_) {
        replayRecorder_->endFrame();
    }
}

void EngineApp::handleCommand(const CmdCameraUpdate& cmd) {
    // Do NOT set inWorld_ here. inWorld_ is only set by CmdWorldLoad so that
    // the menu panoramic camera (which also sends CmdCameraUpdate) does not
    // prematurely switch the engine into in-world rendering mode and feed
    // zeroed G-buffers to DLSS-RR, which causes white-screen divergence.
    std::memcpy(latestCamera_.view, cmd.view, sizeof(cmd.view));
    std::memcpy(latestCamera_.projection, cmd.projection, sizeof(cmd.projection));
    latestCamera_.posX = cmd.posX;
    latestCamera_.posY = cmd.posY;
    latestCamera_.posZ = cmd.posZ;
    latestCamera_.dirX = cmd.dirX;
    latestCamera_.dirY = cmd.dirY;
    latestCamera_.dirZ = cmd.dirZ;
    latestCamera_.nearPlane = cmd.nearPlane;
    latestCamera_.farPlane = cmd.farPlane;
    latestCamera_.gameTick = cmd.tick;
    latestCamera_.valid = true;
}

void EngineApp::handleCommand(const CmdSkyUpdate& cmd) {
    if (!services_->sceneRes().isInitialized()) return;

    SkyUBOData sky{};
    std::memcpy(sky.baseColor, cmd.baseColor, sizeof(sky.baseColor));
    sky.skyType = cmd.skyType;
    std::memcpy(sky.horizonColor, cmd.horizonColor, sizeof(sky.horizonColor));
    std::memcpy(sky.sunDirection, cmd.sunDirection, sizeof(sky.sunDirection));
    sky.isSunRisingOrSetting = cmd.sunRisingOrSetting;
    std::memcpy(sky.moonDirection, cmd.moonDirection, sizeof(sky.moonDirection));
    sky.isSkyDark = cmd.skyDark;
    sky.hasBlindnessOrDarkness = cmd.hasBlindnessOrDarkness;
    sky.cameraSubmersionType = cmd.cameraSubmersionType;
    sky.moonPhase = cmd.moonPhase;
    sky.rainGradient = cmd.rainGradient;
    sky.thunderGradient = cmd.thunderGradient;
    sky.sunTextureID = cmd.sunTextureID;
    sky.moonTextureID = cmd.moonTextureID;

    services_->sceneRes().updateSkyUBO(sky);
}

void EngineApp::handleCommand(const CmdTextureMappingUpdate& cmd) {
    if (!services_->sceneRes().isInitialized()) return;
    if (cmd.data.empty()) return;
    services_->sceneRes().uploadTextureMapping(cmd.data.data(), cmd.data.size());
}

void EngineApp::handleCommand(const CmdSpriteTableUpload& cmd) {
    if (!services_->texture().isInitialized()) return;
    services_->texture().receiveSpriteTable(
        cmd.metadata.data(), cmd.count,
        cmd.atlasWidth, cmd.atlasHeight, cmd.spriteSize);
}

void EngineApp::handleCommand(const CmdSpritePixelsUpload& cmd) {
    if (!services_->texture().isInitialized()) return;
    services_->texture().receiveSpritePixels(cmd.pixels.data(), cmd.pixels.size());
}

void EngineApp::handleCommand(const CmdSpriteAuxPixelsUpload& cmd) {
    if (!services_->texture().isInitialized()) return;
    services_->texture().receiveAuxPixels(
        cmd.specularPixels.data(), cmd.specularPixels.size(),
        cmd.normalPixels.data(), cmd.normalPixels.size());
}

void EngineApp::handleCommand(const CmdAnimationFramesUpload& cmd) {
    if (!services_->texture().isInitialized()) return;
    services_->texture().receiveAnimationFrames(cmd.data.data(), cmd.data.size());
}

void EngineApp::handleCommand(const CmdTextureFinalize&) {
    // Finalization is deferred to processScene() where we have a command buffer.
    // The TextureService's pendingFinalize_ flag is already set by receiveSpriteTable().
    log::info("bridge", "TextureFinalize command received — will finalize in next processScene");
}

void EngineApp::handleCommand(const CmdAreaLightUpload& cmd) {
    if (!services_->sceneRes().isInitialized()) return;
    services_->sceneRes().uploadAreaLights(
        cmd.data.empty() ? nullptr : cmd.data.data(),
        cmd.data.size(), cmd.lightCount);
}

void EngineApp::handleCommand(const CmdEmissionDataUpload& cmd) {
    if (!services_->sceneRes().isInitialized()) return;
    services_->sceneRes().uploadEmissionData(cmd.emissionData, cmd.emissiveGamut);
}

void EngineApp::handleCommand(const CmdEntityBatchSubmit& cmd) {
    // Store the entity batch for processing in processScene().
    // We need a mutable copy since we'll move the data out.
    pendingEntityBatch_ = std::make_unique<CmdEntityBatchSubmit>(cmd);
}

void EngineApp::handleCommand(const CmdEntityPostDraw& cmd) {
    // Store post-entity draw data for the PostRenderAdapter.
    pendingPostDraw_ = std::make_unique<CmdEntityPostDraw>(cmd);
}

// --- Bridge symbols (consumed by generated config_bridge.cpp) ---

EngineConfig& activeConfig() {
    return EngineApp::get()->services().config().live();
}

BridgeService& activeBridge() {
    return EngineApp::get()->services().bridge();
}

void notifyConfigChange(ConfigKey key) {
    auto* app = EngineApp::get();
    if (app) app->services().config().notifyChange(key);
}

// Apply diagLevel + diagFlags → spdlog minimum level, g_effectiveDiagFlags, per-category enables,
// and replay recorder activation/deactivation.
// 0 = silent (warn+), 1 = operational (info+), 2 = diagnostic (debug+), 3 = forensic (trace).
void EngineApp::applyDiagLevel(int level, int diagFlags) {
    // Set spdlog minimum severity level via integer overload
    log::setMinLevel(level);  // 0=warn, 1=info, 2=debug, 3=trace

    // Compute effective flags: level implies minimum flags
    int impliedFlags = 0;
    if (level >= 2) impliedFlags = DiagFlags::TLAS | DiagFlags::BLAS | DiagFlags::UPLOAD | DiagFlags::FRAME;
    if (level >= 3) impliedFlags = 0xFF;  // all flags
    engine::g_effectiveDiagFlags.store(diagFlags | impliedFlags, std::memory_order_relaxed);

    // Per-category enable/disable
    bool verbose     = (level >= 2);
    bool operational = (level >= 1);
    log::setCategoryEnabled("scene",        operational);
    log::setCategoryEnabled("blas",         verbose);
    log::setCategoryEnabled("entityBlas",   verbose);
    log::setCategoryEnabled("gpu-upload",   verbose);
    log::setCategoryEnabled("tlas",         verbose);
    log::setCategoryEnabled("texture",      operational);
    log::setCategoryEnabled("rendergraph",  operational);
    log::setCategoryEnabled("graph-pool",   operational);
    log::setCategoryEnabled("replay",       verbose);
    log::setCategoryEnabled("metrics",      operational);
    // Always-on categories (lifecycle events):
    log::setCategoryEnabled("app",          true);
    log::setCategoryEnabled("bridge",       operational);
    log::setCategoryEnabled("config",       operational);
    log::setCategoryEnabled("frame",        true);  // always — DEVICE_LOST must log
    log::setCategoryEnabled("vulkan",       operational);
    log::setCategoryEnabled("atmosphere",   operational);
    log::setCategoryEnabled("cloud",        operational);
    log::setCategoryEnabled("c2d",          operational);
    log::setCategoryEnabled("dlss",         operational);
    log::setCategoryEnabled("framegen",     operational);
    log::setCategoryEnabled("postprocess",  operational);
    log::setCategoryEnabled("rt-adapter",   operational);
    log::setCategoryEnabled("tonemapping",  operational);
    log::setCategoryEnabled("post-render",  operational);
    log::setCategoryEnabled("scene-res",    operational);
    log::setCategoryEnabled("starfield",    false);  // never needed

    log::info("config", "diagLevel → " + std::to_string(level)
              + " diagFlags=0x" + [](int v) {
                  char buf[12]; std::snprintf(buf, sizeof(buf), "%x", v); return std::string(buf);
              }(diagFlags | impliedFlags));

    // Activate or deactivate replay recorder based on level/flags
    if (services_) {
        bool wantReplay = (level >= 3) || ((diagFlags & DiagFlags::REPLAY) != 0);
        if (wantReplay && !replayRecorder_) {
            auto now   = std::chrono::system_clock::now();
            auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
                             now.time_since_epoch()).count();
            std::string logDir = services_->resourceDir() + "/logs";
            std::string replayPath = logDir + "/replay_" + std::to_string(epoch) + ".bin";
            replayRecorder_ = std::make_unique<ReplayRecorder>(replayPath);
            log::info("replay", "Replay recorder started: " + replayPath);
        } else if (!wantReplay && replayRecorder_) {
            replayRecorder_.reset();  // ~ReplayRecorder patches frame count in header
            log::info("replay", "Replay recorder stopped");
        }
    }
}

void onConfigSideEffect(ConfigKey key) {
    auto* app = EngineApp::get();
    if (!app) return;

    switch (key) {
        case ConfigKey::OMM_ENABLED:
            log::debug("config", "OMM enabled changed");
            break;
        case ConfigKey::OFFLINE_ACCUM_ENABLED: {
            // Sync the AccumState::enabled flag and reset the counter so the
            // accumulation starts fresh whenever the setting is toggled.
            auto snap = app->services().config().snapshot();
            if (snap) {
                app->syncAccumEnabled(snap->data.offlineAccumEnabled);
                log::debug("config", std::string("offlineAccumEnabled → ") +
                           (snap->data.offlineAccumEnabled ? "true (reset counter)" : "false"));
            }
            break;
        }
        case ConfigKey::DIAG_LEVEL:
        case ConfigKey::DIAG_FLAGS: {
            auto snap = app->services().config().snapshot();
            if (snap) app->applyDiagLevel(snap->data.diagLevel, snap->data.diagFlags);
            break;
        }
        default:
            break;
    }
}

} // namespace engine
