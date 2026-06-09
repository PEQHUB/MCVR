# ADR-003: Build Flag + Runtime Flag for V2 Activation

**Status**: Accepted
**Date**: 2026-04-03

## Context

The rewrite must coexist with legacy code for months. We need a mechanism to:
1. Compile new code without affecting the existing build
2. Activate v2 subsystems at runtime for testing without rebuilding
3. Fall back to legacy if v2 regresses

## Decision

**Build flag**: `MCVR_ENABLE_ENGINE_V2` (CMake option, default OFF)
- Gates compilation of `src/engine/` entirely
- When OFF, zero impact on existing build — no new symbols, no new includes
- CI turns it ON by default once the v2 shell compiles cleanly

**Runtime flag**: `engine.mode` in options.properties
- Values: `legacy` (default) | `v2`
- Read at startup before any subsystem init
- Controls which init path runs (legacy Renderer vs EngineApp)
- Can be set via DebugBridge: `{"cmd":"set","key":"engineMode","value":"v2"}`

**Rollout**:
- PR4-PR6: build flag gates compilation while shell is immature
- After v2 shell compiles cleanly: build flag ON in default CMake preset
- Runtime flag stays permanently for activation and fallback
- No long-term dual mode — once parity is proven per-subsystem, legacy path is deleted

## Alternatives Considered

**Runtime-only flag**: Would compile both paths always, slowing builds and
risking accidental symbol collisions during early development.

**Build-only flag**: No way to A/B test in the field without rebuilding.
Prevents quick fallback during user testing.

## Consequences

### Positive
- Zero risk to existing users during development
- Clean compilation boundary
- Runtime testability without rebuild

### Negative
- Two code paths exist temporarily (complexity budget)
- Must be disciplined about not extending dual mode beyond necessity

### Migration Impact
- CMakeLists.txt gets new option and conditional add_subdirectory
- options.properties gets new `engineMode=legacy` default
- Each subsystem cutover removes one piece of legacy, eventually eliminating the flag
