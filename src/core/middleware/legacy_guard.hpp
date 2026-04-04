#pragma once

// Guard macro for legacy JNI middleware functions.
// Returns early (void or default) when the legacy Renderer singleton is not initialized
// (e.g., when V2 engine is active). Prevents C++ exceptions from reaching the JVM.

#include "core/render/renderer.hpp"

#define LEGACY_GUARD() \
    do { if (!Renderer::is_initialized()) return; } while(0)

#define LEGACY_GUARD_RETURN(val) \
    do { if (!Renderer::is_initialized()) return (val); } while(0)
