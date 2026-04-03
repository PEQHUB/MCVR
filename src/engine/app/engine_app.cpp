#include "engine_app.hpp"
#include "engine_session.hpp"
#include "engine_services.hpp"
#include "config/config_service.hpp"
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

// --- Bridge symbols (consumed by generated config_bridge.cpp) ---

EngineConfig& activeConfig() {
    return EngineApp::get()->services().config().live();
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
