#pragma once
#include <atomic>

namespace engine {

// Effective diagnostic flags — OR of diagFlags and level-implied flags.
// Written by engine_app::applyDiagLevel(), read on hot paths.
extern std::atomic<int> g_effectiveDiagFlags;

} // namespace engine
