#include "engine_services.hpp"
#include "config/config_service.hpp"
#include "bridge/bridge_service.hpp"
#include "frame/frame_scheduler.hpp"
#include "scene/scene_service.hpp"

namespace engine {

EngineServices::EngineServices()
    : config_(std::make_unique<ConfigService>()),
      bridge_(std::make_unique<BridgeService>()),
      frame_(std::make_unique<FrameScheduler>(*this)),
      scene_(std::make_unique<SceneService>()) {}

EngineServices::~EngineServices() = default;

} // namespace engine
