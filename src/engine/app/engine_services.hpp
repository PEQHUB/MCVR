#pragma once

// Ownership: EngineApp (1:1 lifetime).
// Thread: Main thread creates. Individual services declare their own affinity.
// Dependencies: None — this is the service locator root.

#include <memory>

namespace engine {

class ConfigService;
class BridgeService;
class FrameScheduler;
class SceneService;

// Service locator — no globals. All subsystem access goes through here.
// New services are added as the rewrite progresses (device, frame, scene, rt, etc.).
class EngineServices {
public:
    EngineServices();
    ~EngineServices();

    EngineServices(const EngineServices&) = delete;
    EngineServices& operator=(const EngineServices&) = delete;

    ConfigService& config() { return *config_; }
    const ConfigService& config() const { return *config_; }

    BridgeService& bridge() { return *bridge_; }
    const BridgeService& bridge() const { return *bridge_; }

    FrameScheduler& frame() { return *frame_; }
    const FrameScheduler& frame() const { return *frame_; }

    SceneService& scene() { return *scene_; }
    const SceneService& scene() const { return *scene_; }

    // Future services (uncommented as implemented):
    // DeviceService& device();
    // SwapchainService& swapchain();
    // BlasService& blas();
    // TlasService& tlas();

private:
    std::unique_ptr<ConfigService> config_;
    std::unique_ptr<BridgeService> bridge_;
    std::unique_ptr<FrameScheduler> frame_;
    std::unique_ptr<SceneService> scene_;
};

} // namespace engine
