// Ownership: Temporary stubs — replaced by EngineApp in PR4.
// Provides the symbols that generated config_bridge.cpp depends on.

#include "engine_config.hpp"

namespace engine {

// Global config instance — will move into EngineSession in PR4.
static EngineConfig g_config{};

EngineConfig& activeConfig() {
    return g_config;
}

// Side-effect callback — no-op until EngineApp wires service dispatch.
// OMM resetScheduler, etc. are handled here once services exist.
void onConfigSideEffect(ConfigKey /*key*/) {
    // Intentionally empty. PR4+ will dispatch to BlasService, etc.
}

} // namespace engine
