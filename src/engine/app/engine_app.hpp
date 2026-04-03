#pragma once

// Ownership: Static lifetime (one instance per process).
// Thread: Created and destroyed on the main thread.
// Dependencies: None at construction. Services initialized via init().

#include <memory>
#include <string>

namespace engine {

class EngineSession;
class EngineServices;

enum class EngineMode {
    Legacy,  // Use existing Renderer path
    V2       // Use v2 engine path
};

class EngineApp {
public:
    EngineApp() = default;
    ~EngineApp();

    EngineApp(const EngineApp&) = delete;
    EngineApp& operator=(const EngineApp&) = delete;

    // Initialize engine services. Does NOT start rendering.
    // configDir: directory containing options.properties
    // mode: determined from options or command line
    void init(const std::string& configDir, EngineMode mode = EngineMode::Legacy);

    // Shut down all services and release resources.
    void shutdown();

    bool isInitialized() const { return initialized_; }
    EngineMode mode() const { return mode_; }
    EngineSession& session() { return *session_; }
    EngineServices& services() { return *services_; }

    // Global access (replaces Renderer::instance() for v2 code).
    // Returns nullptr if not initialized.
    static EngineApp* get();

private:
    bool initialized_ = false;
    EngineMode mode_ = EngineMode::Legacy;
    std::unique_ptr<EngineServices> services_;
    std::unique_ptr<EngineSession> session_;

    static EngineApp* s_instance;
};

} // namespace engine
