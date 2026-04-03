#include "engine_services.hpp"
#include "config/config_service.hpp"
#include "bridge/bridge_service.hpp"
#include "frame/frame_scheduler.hpp"

namespace engine {

EngineServices::EngineServices()
    : config_(std::make_unique<ConfigService>()),
      bridge_(std::make_unique<BridgeService>()),
      frame_(std::make_unique<FrameScheduler>(*this)) {}

EngineServices::~EngineServices() = default;

} // namespace engine
