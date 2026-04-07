#if defined(_WIN32)
#    define VK_USE_PLATFORM_WIN32_KHR
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
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
#include "features/postprocess_adapter.hpp"
#include "scene/scene_service.hpp"
#include "scene/chunk_registry.hpp"
#include "scene/gpu_upload_service.hpp"
#include "scene/blas_service.hpp"
#include "scene/tlas_service.hpp"
#include "scene/scene_resource_service.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <cstring>

namespace engine {

EngineApp* EngineApp::s_instance = nullptr;

EngineApp::~EngineApp() {
    if (initialized_) shutdown();
}

bool EngineApp::init(const EngineInitConfig& config) {
    if (initialized_) return true;

    mode_ = config.mode;
    s_instance = this;

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
    log::LogConfig logConfig;
    logConfig.logDir = config.configDir + "/logs";
    log::init(logConfig);

    log::info("app", "EngineApp initializing");
    log::info("app", std::string("Mode: ") + (mode_ == EngineMode::V2 ? "v2" : "legacy"));

    // Create services
    services_ = std::make_unique<EngineServices>();
    services_->setResourceDir(config.configDir);

    // Load config
    std::string configPath = config.configDir + "/options.properties";
    services_->config().load(configPath);

    // Wire bridge handler
    services_->bridge().setHandler([this](const BridgeCommand& cmd) {
        std::visit([this](const auto& c) { handleCommand(c); }, cmd);
    });

    // Initialize Vulkan in V2 mode
    if (mode_ == EngineMode::V2) {
        if (!config.window && !config.nativeWindowHandle) {
            log::error("app", "V2 mode requires a window (GLFW or native handle)");
            rollback();
            return false;
        }

        vk2::DeviceService::InitConfig deviceConfig;
        deviceConfig.enableValidation = config.enableValidation;

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

        auto deviceResult = services_->device().init(deviceConfig);
        if (!deviceResult) {
            log::error("app", "DeviceService init failed: " + deviceResult.error().message);
            rollback();
            return false;
        }

        auto swapResult = services_->swapchain().init(services_->device());
        if (!swapResult) {
            log::error("app", "SwapchainService init failed: " + swapResult.error().message);
            rollback();
            return false;
        }

        // Initialize frame scheduler
        uint32_t imgCount = services_->swapchain().imageCount();
        services_->frame().setImageCount(imgCount);
        uint32_t framesInFlight = std::min(imgCount, 2u);

        auto initSyncResult = services_->frame().initSync(services_->device(), framesInFlight);
        if (!initSyncResult) {
            log::error("app", "Frame sync init failed: " + initSyncResult.error().message);
            rollback();
            return false;
        }

        // Initialize metrics
        auto metricsResult = services_->metrics().init(services_->device(), framesInFlight);
        if (!metricsResult) {
            log::warn("app", "Metrics init failed (non-fatal): " + metricsResult.error().message);
        }

        // Initialize scene services
        {
            auto gpuR = services_->gpuUpload().init(services_->device(), framesInFlight);
            if (!gpuR) log::warn("app", "GpuUpload init failed (non-fatal): " + gpuR.error().message);
            auto blasR = services_->blas().init(services_->device());
            if (!blasR) log::warn("app", "BLAS init failed (non-fatal): " + blasR.error().message);
            auto tlasR = services_->tlas().init(services_->device());
            if (!tlasR) log::warn("app", "TLAS init failed (non-fatal): " + tlasR.error().message);
            auto sceneResR = services_->sceneRes().init(services_->device(), framesInFlight);
            if (!sceneResR) log::warn("app", "SceneRes init failed (non-fatal): " + sceneResR.error().message);
        }

        // Initialize offscreen render targets
        auto offscreenResult = services_->offscreen().init(
            services_->device(), framesInFlight,
            services_->swapchain().extent().width,
            services_->swapchain().extent().height);
        if (!offscreenResult) {
            log::error("app", "Offscreen init failed: " + offscreenResult.error().message);
            rollback();
            return false;
        }

        log::info("app", "Vulkan initialized: " + services_->device().caps().deviceName +
                  ", swapchain " + std::to_string(services_->swapchain().extent().width) +
                  "x" + std::to_string(services_->swapchain().extent().height));
    }

    // Create session
    session_ = std::make_unique<EngineSession>(*services_);
    session_->setState(SessionState::Running);

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

        // Build render graph: RT → ToneMapping → PostProcess (CAS) → Composite
        auto rtAdapter = std::make_unique<RayTracingAdapter>();
        rtAdapter->init(*services_);

        auto tmAdapter = std::make_unique<ToneMappingAdapter>();
        tmAdapter->init(*services_);

        auto ppAdapter = std::make_unique<PostProcessAdapter>();
        ppAdapter->init(*services_);

        GraphBuilder builder;
        auto rtOutput = rtAdapter->registerPass(builder);
        tmAdapter->registerPass(builder, rtOutput);
        // ToneMapping declares its output via outputHandle_; PostProcess takes over as final.
        // We need to fish that handle out — easiest via a small accessor on the adapter.
        ppAdapter->registerPass(builder, tmAdapter->outputHandle());
        auto graph = builder.compile();

        if (graph.valid()) {
            services_->frame().setGraph(std::move(graph));
            log::info("app", "Render graph active: RT → ToneMapping → PostProcess(CAS) → Composite");
        }

        adapters_.push_back(std::move(rtAdapter));
        adapters_.push_back(std::move(tmAdapter));
        adapters_.push_back(std::move(ppAdapter));
    }

    return true;
}

bool EngineApp::tick() {
    if (!initialized_ || !session_ || session_->state() == SessionState::ShuttingDown) {
        return false;
    }

    // Check for device-lost from previous frame
    if (mode_ == EngineMode::V2 && services_->frame().isDeviceLost()) {
        log::error("app", "Device lost detected — shutting down V2 engine");
        session_->setState(SessionState::ShuttingDown);
        return false;
    }

    if (mode_ == EngineMode::V2) {
        // Flush bridge and tick GC every frame regardless
        auto ctx = services_->frame().beginFrame();
        ctx.camera = latestCamera_;

        if (!inWorld_ || services_->swapchain().isZeroExtent()) {
            // Not in world or minimized — don't present, let Minecraft show its UI
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

    log::info("app", "EngineApp shutting down");

    if (session_) {
        session_->setState(SessionState::ShuttingDown);
        session_.reset();
    }

    if (mode_ == EngineMode::V2) {
        services_->device().waitIdle();
        for (auto& adapter : adapters_) adapter->shutdown();
        adapters_.clear();
        services_->frame().resourcePool().releaseImmediate();
        services_->tlas().shutdown();
        services_->blas().shutdown();
        services_->gpuUpload().shutdown();
        services_->offscreen().shutdown();
        services_->metrics().shutdown();
        services_->frame().shutdownSync();
        services_->swapchain().shutdown();
        services_->device().shutdown();
    }

    services_.reset();

    log::info("app", "EngineApp shutdown complete");
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
            auto r = services_->swapchain().recreate(cmd.width, cmd.height);
            if (!r) log::error("bridge", "Swapchain recreate failed: " + r.error().message);
        }
    }
}

void EngineApp::handleCommand(const CmdWorldLoad& cmd) {
    log::info("bridge", "World load: " + cmd.regionPath);
    inWorld_ = true;
}

void EngineApp::handleCommand(const CmdWorldUnload&) {
    log::info("bridge", "World unload");
    inWorld_ = false;
    latestCamera_.valid = false;
}



void EngineApp::handleCommand(const CmdShutdown&) {
    log::info("bridge", "Shutdown requested via bridge");
    if (session_) session_->setState(SessionState::ShuttingDown);
}

void EngineApp::handleCommand(const CmdConfigPatch& cmd) {
    cmd.apply();
}

void EngineApp::handleCommand(const CmdChunkSubmit& cmd) {
    ChunkGeometry geo;
    geo.id = {cmd.chunkX, cmd.chunkZ};
    geo.vertexData = cmd.vertexData;
    geo.indexData = cmd.indexData;
    geo.triangleCount = cmd.triangleCount;
    geo.originX = static_cast<float>(cmd.originX);
    geo.originY = static_cast<float>(cmd.originY);
    geo.originZ = static_cast<float>(cmd.originZ);
    services_->scene().chunks().insert(geo.id, std::move(geo));
}

void EngineApp::handleCommand(const CmdChunkRemove& cmd) {
    ChunkId id{cmd.chunkX, cmd.chunkZ};
    services_->scene().chunks().remove(id);
    services_->gpuUpload().removeChunk(id);
    services_->blas().removeChunk(id);
}

void EngineApp::processScene(VkCommandBuffer cmd) {
    // Update scene resources (WorldUBO with current camera/config)
    if (services_->sceneRes().isInitialized() && latestCamera_.valid) {
        auto cfg = services_->config().snapshot();
        if (cfg) {
            services_->sceneRes().updateWorldUBO(
                services_->frame().currentFrameIndex(),
                latestCamera_, *cfg, services_->frame().frameNumber());
        }
    }

    // Extract scene snapshot
    auto scene = services_->scene().extractFrame(services_->frame().frameNumber());
    if (!scene) return;

    // Upload dirty chunks
    uint32_t uploaded = 0;
    if (!scene->dirtyChunkGeometries.empty() && services_->gpuUpload().isInitialized()) {
        uploaded = services_->gpuUpload().uploadDirtyChunks(
            cmd, scene->dirtyChunkGeometries, services_->frame().currentFrameIndex());
        if (uploaded > 0) {
            log::debug("scene", "Uploaded " + std::to_string(uploaded) + " chunks");
        }
    }

    // Barrier: upload copies must complete before BLAS builds read vertex/index data
    if (uploaded > 0) {
        VkMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // Build BLAS for dirty chunks
    if (services_->blas().isInitialized()) {
        std::vector<ChunkId> dirtyIds;
        for (const auto& geo : scene->dirtyChunkGeometries) {
            if (!geo.empty()) dirtyIds.push_back(geo.id);
        }
        if (!dirtyIds.empty()) {
            uint32_t built = services_->blas().buildDirtyChunks(
                cmd, dirtyIds, services_->gpuUpload().allChunks());
            if (built > 0) {
                log::debug("scene", "Built " + std::to_string(built) + " BLAS");
            }
        }
    }

    // Rebuild TLAS
    if (services_->tlas().isInitialized() && services_->blas().blasCount() > 0) {
        services_->tlas().rebuild(cmd, services_->blas());
    }
}

void EngineApp::handleCommand(const CmdCameraUpdate& cmd) {
    inWorld_ = true;
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

void onConfigSideEffect(ConfigKey key) {
    auto* app = EngineApp::get();
    if (!app) return;

    switch (key) {
        case ConfigKey::OMM_ENABLED:
            log::debug("config", "OMM enabled changed");
            break;
        default:
            break;
    }
}

} // namespace engine
