#include "engine_app.hpp"
#include "engine_session.hpp"
#include "engine_services.hpp"
#include "config/config_service.hpp"
#include "bridge/bridge_service.hpp"
#include "frame/frame_scheduler.hpp"
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
        if (!config.window) {
            log::error("app", "V2 mode requires a GLFW window");
            return false;
        }

        vk2::DeviceService::InitConfig deviceConfig;
        deviceConfig.window = config.window;
        deviceConfig.enableValidation = config.enableValidation;

        auto deviceResult = services_->device().init(deviceConfig);
        if (!deviceResult) {
            log::error("app", "DeviceService init failed: " + deviceResult.error().message);
            return false;
        }

        auto swapResult = services_->swapchain().init(services_->device());
        if (!swapResult) {
            log::error("app", "SwapchainService init failed: " + swapResult.error().message);
            return false;
        }

        // Initialize frame scheduler with real swapchain info
        services_->frame().setImageCount(services_->swapchain().imageCount());

        auto initSyncResult = services_->frame().initSync(services_->device());
        if (!initSyncResult) {
            log::error("app", "Frame sync init failed: " + initSyncResult.error().message);
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
    log::info("app", "EngineApp initialized");
    return true;
}

bool EngineApp::tick() {
    if (!initialized_ || !session_ || session_->state() == SessionState::ShuttingDown) {
        return false;
    }

    if (mode_ == EngineMode::V2) {
        // Real Vulkan frame: acquire → clear → present
        auto ctx = services_->frame().beginFrame();

        services_->frame().executeClearFrame(
            services_->device(), services_->swapchain(), ctx);

        services_->frame().endFrame(ctx);
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
    if (mode_ == EngineMode::V2 && services_->swapchain().isInitialized()) {
        services_->device().waitIdle();
        auto r = services_->swapchain().recreate(cmd.width, cmd.height);
        if (!r) log::error("bridge", "Swapchain recreate failed: " + r.error().message);
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
