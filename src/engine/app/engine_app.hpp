#pragma once

// Ownership: Static lifetime (one instance per process).
// Thread: Created and destroyed on the main thread.
// Dependencies: None at construction. Services initialized via init().

#include "features/feature_adapter.hpp"
#include "frame/frame_context.hpp"

#include <memory>
#include <string>
#include <vector>

struct GLFWwindow;

namespace engine {

class EngineSession;
class EngineServices;
struct CmdPing;
struct CmdWindowResize;
struct CmdWorldLoad;
struct CmdWorldUnload;
struct CmdShutdown;
struct CmdConfigPatch;

enum class EngineMode {
    Legacy,  // Use existing Renderer path
    V2       // Use v2 engine path
};

struct EngineInitConfig {
    std::string configDir;             // Directory containing options.properties
    GLFWwindow* window = nullptr;      // Existing GLFW window (standalone V2 mode)
    void* nativeWindowHandle = nullptr; // Platform window handle (HWND on Win32, for JNI mode)
    EngineMode mode = EngineMode::Legacy;
    bool enableValidation = false;
};

class EngineApp {
public:
    EngineApp() = default;
    ~EngineApp();

    EngineApp(const EngineApp&) = delete;
    EngineApp& operator=(const EngineApp&) = delete;

    // Initialize engine services. In V2 mode, creates Vulkan device + swapchain.
    // Returns false on fatal init failure.
    bool init(const EngineInitConfig& config);

    // Run one frame (acquire, clear, present). Returns false if shutdown requested.
    bool tick();

    // Shut down all services and release resources.
    void shutdown();

    bool isInitialized() const { return initialized_; }
    bool isInWorld() const { return inWorld_; }
    EngineMode mode() const { return mode_; }
    EngineSession& session() { return *session_; }
    EngineServices& services() { return *services_; }

    static EngineApp* get();

private:
    bool initialized_ = false;
    EngineMode mode_ = EngineMode::Legacy;
    std::unique_ptr<EngineServices> services_;
    std::unique_ptr<EngineSession> session_;
    std::vector<std::unique_ptr<FeatureAdapter>> adapters_;
    CameraData latestCamera_;
    bool inWorld_ = false;

    static EngineApp* s_instance;

    void handleCommand(const struct CmdPing& cmd);
    void handleCommand(const struct CmdWindowResize& cmd);
    void handleCommand(const struct CmdWorldLoad& cmd);
    void handleCommand(const struct CmdWorldUnload& cmd);
    void handleCommand(const struct CmdShutdown& cmd);
    void handleCommand(const struct CmdConfigPatch& cmd);
    void handleCommand(const struct CmdChunkSubmit& cmd);
    void handleCommand(const struct CmdChunkRemove& cmd);
    void handleCommand(const struct CmdCameraUpdate& cmd);
    void handleCommand(const struct CmdSkyUpdate& cmd);
    void handleCommand(const struct CmdTextureMappingUpdate& cmd);

    void processScene(VkCommandBuffer cmd);
};

} // namespace engine
