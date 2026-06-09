#include "engine_session.hpp"
#include "engine_services.hpp"

namespace engine {

EngineSession::EngineSession(EngineServices& services)
    : services_(services) {}

EngineSession::~EngineSession() = default;

} // namespace engine
