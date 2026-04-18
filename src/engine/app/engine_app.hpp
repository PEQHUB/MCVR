#pragma once

// Ownership: Static lifetime (one instance per process).
// Thread: Created and destroyed on the main thread.
// Dependencies: None at construction. Services initialized via init().

#include "features/feature_adapter.hpp"
#include "diagnostics/replay/replay_recorder.hpp"
#include "features/atmosphere_adapter.hpp"
#include "features/cloud_adapter.hpp"
#include "features/dlss_adapter.hpp"
#include "features/frame_gen_adapter.hpp"
#include "features/raytracing_adapter.hpp"
#include "features/tonemapping_adapter.hpp"
#include "frame/frame_context.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct GLFWwindow;

namespace engine {

class EngineSession;
class EngineServices;
class PostRenderAdapter;
struct CmdPing;
struct CmdWindowResize;
struct CmdWorldLoad;
struct CmdWorldUnload;
struct CmdShutdown;
struct CmdConfigPatch;
struct CmdResetAccumulation;
struct CmdSpriteTableUpload;
struct CmdSpritePixelsUpload;
struct CmdSpriteAuxPixelsUpload;
struct CmdAnimationFramesUpload;
struct CmdTextureFinalize;
struct CmdAreaLightUpload;
struct CmdEmissionDataUpload;
struct CmdEntityBatchSubmit;
struct CmdEntityPostDraw;

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

// Offline accumulation state — tracks per-frame Welford counter and previous camera
// position so processScene() can detect movement and reset the accumulator.
// All fields are main-thread-only (processScene runs on the engine/main thread).
struct AccumState {
    uint32_t accumFrameCount = 0;       // Welford N (0 = not accumulating yet)
    float prevCamPosX = 0.0f;
    float prevCamPosY = 0.0f;
    float prevCamPosZ = 0.0f;
    float prevCamDirX = 0.0f;
    float prevCamDirY = 0.0f;
    float prevCamDirZ = 0.0f;
    bool enabled = false;               // mirrors config.offlineAccumEnabled
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

    // Reset the offline accumulation counter to 0.
    // Thread: main thread only (called from onConfigSideEffect and bridge commands).
    void resetAccum() {
        accumState_.accumFrameCount = 0;
    }

    // Sync the AccumState::enabled flag from the current config.
    // Resets the counter as a side-effect.
    void syncAccumEnabled(bool enabled) {
        accumState_.enabled = enabled;
        accumState_.accumFrameCount = 0;
    }

    static EngineApp* get();

    // Apply diagnostic verbosity tier and per-subsystem flags to all engine services.
    // Called at init and whenever DIAG_LEVEL / DIAG_FLAGS config keys change.
    void applyDiagLevel(int level, int diagFlags);

private:
    bool initialized_ = false;
    EngineMode mode_ = EngineMode::Legacy;
    std::unique_ptr<EngineServices> services_;
    std::unique_ptr<EngineSession> session_;
    std::vector<std::unique_ptr<FeatureAdapter>> adapters_;
    AtmosphereAdapter atmosphere_;
    DlssAdapter* dlssAdapter_ = nullptr;     // non-owning; lifetime managed by adapters_
    CloudAdapter* cloudAdapter_ = nullptr;   // non-owning; lifetime managed by adapters_
    RayTracingAdapter* rtAdapter_ = nullptr;  // non-owning; lifetime managed by adapters_
    ToneMappingAdapter* tmAdapter_ = nullptr; // non-owning; lifetime managed by adapters_
    PostRenderAdapter* prAdapter_ = nullptr;   // non-owning; lifetime managed by adapters_
    FrameGenAdapter* fgAdapter_ = nullptr;    // non-owning; lifetime managed by adapters_
    CameraData latestCamera_;
    bool inWorld_ = false;
    uint64_t tickCount_ = 0;  // monotonic per-frame counter for boot_trace classifier
    AccumState accumState_;
    std::unique_ptr<struct CmdEntityBatchSubmit> pendingEntityBatch_;
    std::unique_ptr<struct CmdEntityPostDraw> pendingPostDraw_;
    std::unique_ptr<ReplayRecorder> replayRecorder_;

    static EngineApp* s_instance;

    void handleCommand(const struct CmdPing& cmd);
    void handleCommand(const struct CmdWindowResize& cmd);
    void handleCommand(const struct CmdWorldLoad& cmd);
    void handleCommand(const struct CmdWorldUnload& cmd);
    void handleCommand(const struct CmdShutdown& cmd);
    void handleCommand(const struct CmdConfigPatch& cmd);
    void handleCommand(const struct CmdResetAccumulation& cmd);
    void handleCommand(const struct CmdChunkSubmit& cmd);
    void handleCommand(const struct CmdChunkRemove& cmd);
    void handleCommand(const struct CmdCameraUpdate& cmd);
    void handleCommand(const struct CmdSkyUpdate& cmd);
    void handleCommand(const struct CmdTextureMappingUpdate& cmd);
    void handleCommand(const struct CmdSpriteTableUpload& cmd);
    void handleCommand(const struct CmdSpritePixelsUpload& cmd);
    void handleCommand(const struct CmdSpriteAuxPixelsUpload& cmd);
    void handleCommand(const struct CmdAnimationFramesUpload& cmd);
    void handleCommand(const struct CmdTextureFinalize& cmd);
    void handleCommand(const struct CmdAreaLightUpload& cmd);
    void handleCommand(const struct CmdEmissionDataUpload& cmd);
    void handleCommand(const struct CmdEntityBatchSubmit& cmd);
    void handleCommand(const struct CmdEntityPostDraw& cmd);

    void processScene(VkCommandBuffer cmd);
};

} // namespace engine
