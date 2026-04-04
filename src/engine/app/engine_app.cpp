#include "engine_app.hpp"
#include "engine_session.hpp"
#include "engine_services.hpp"
#include "config/config_service.hpp"
#include "bridge/bridge_service.hpp"
#include "diagnostics/log.hpp"

namespace engine {

EngineApp* EngineApp::s_instance = nullptr;

EngineApp::~EngineApp() {
    if (initialized_) shutdown();
}

void EngineApp::init(const std::string& configDir, EngineMode mode) {
    if (initialized_) return;

    mode_ = mode;
    s_instance = this;

    // Initialize logging first so all subsequent code can log.
    log::LogConfig logConfig;
    logConfig.logDir = configDir + "/logs";
    log::init(logConfig);

    log::info("app", "EngineApp initializing");
    log::info("app", std::string("Mode: ") + (mode == EngineMode::V2 ? "v2" : "legacy"));

    // Create services
    services_ = std::make_unique<EngineServices>();

    // Load config
    std::string configPath = configDir + "/options.properties";
    services_->config().load(configPath);

    // Wire default command handler
    services_->bridge().setHandler([this](const BridgeCommand& cmd) {
        std::visit([this](const auto& c) { handleCommand(c); }, cmd);
    });

    // Create session
    session_ = std::make_unique<EngineSession>(*services_);
    session_->setState(SessionState::Running);

    initialized_ = true;
    log::info("app", "EngineApp initialized");
}

void EngineApp::shutdown() {
    if (!initialized_) return;

    log::info("app", "EngineApp shutting down");

    if (session_) {
        session_->setState(SessionState::ShuttingDown);
        session_.reset();
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
    // Will trigger SwapchainService::recreate() once that service exists.
}

void EngineApp::handleCommand(const CmdWorldLoad& cmd) {
    log::info("bridge", "World load: " + cmd.regionPath);
    // Will trigger SceneService reset + streaming start once those services exist.
}

void EngineApp::handleCommand(const CmdWorldUnload&) {
    log::info("bridge", "World unload");
    // Will trigger SceneService cleanup once that service exists.
}

void EngineApp::handleCommand(const CmdShutdown&) {
    log::info("bridge", "Shutdown requested via bridge");
    if (session_) session_->setState(SessionState::ShuttingDown);
}

void EngineApp::handleCommand(const CmdConfigPatch& cmd) {
    // Applied on main thread during bridge.flush() — safe to mutate live config.
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

    app->services().config().notifyChange(key);

    // Key-specific dispatch (expanded as services are added)
    switch (key) {
        case ConfigKey::OMM_ENABLED:
            // Will dispatch to BlasService::resetScheduler() once that service exists.
            log::debug("config", "OMM enabled changed — scheduler reset deferred until BlasService exists");
            break;
        default:
            break;
    }
}

} // namespace engine
