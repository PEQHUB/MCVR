#include "engine_services.hpp"
#include "config/config_service.hpp"

namespace engine {

EngineServices::EngineServices()
    : config_(std::make_unique<ConfigService>()) {}

EngineServices::~EngineServices() = default;

} // namespace engine
