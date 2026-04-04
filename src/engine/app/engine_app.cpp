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
#include "diagnostics/metrics_service.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "platform/vulkan/vk2_swapchain.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>

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
        if (services_->swapchain().isZeroExtent()) {
            // Window minimized — idle, but try to recover on each tick
            auto ctx = services_->frame().beginFrame();
            // Attempt recreate to detect when window is restored
            if (services_->swapchain().isRecreateNeeded()) {
                services_->device().waitIdle();
                services_->swapchain().recreate(0, 0);
            }
            services_->frame().endFrame(ctx);
        } else {
            auto ctx = services_->frame().beginFrame();
            bool skipped = services_->frame().executeClearFrame(
                services_->device(), services_->swapchain(), ctx);
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
}

void EngineApp::handleCommand(const CmdWorldUnload&) {
    log::info("bridge", "World unload");
}

void EngineApp::handleCommand(const CmdShutdown&) {
    log::info("bridge", "Shutdown requested via bridge");
    if (session_) session_->setState(SessionState::ShuttingDown);
}

void EngineApp::handleCommand(const CmdConfigPatch& cmd) {
    cmd.apply();
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
