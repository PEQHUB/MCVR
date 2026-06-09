# ADR-001: JSON Schema + Python Codegen for Config

**Status**: Accepted
**Date**: 2026-04-03

## Context

Options.java (5,247 lines, 157 native methods) and com_radiance_client_option_Options.cpp
manually mirror every setting between Java and C++. Adding a new option requires editing
both files, writing a JNI setter, and updating UI metadata — 4 touch points minimum.
Drift between the two sides causes UnsatisfiedLinkError at runtime.

## Decision

Single source of truth: `C:\RadSER\schema\engine_config.schema.json`

Generate:
- `MCVR/src/generated/config/*` — C++ structs, enums, defaults, validation, patch/apply
- `Radiance/src/generated/java/com/radiance/v2/config/*` — Java classes, UI metadata
- `MCVR/src/generated/bridge/*` — typed JNI option patch messages
- `Radiance/src/generated/java/com/radiance/v2/bridge/*` — Java-side bridge types

Schema extensions (x-* metadata):
- `x-category` — UI grouping
- `x-ui-control` — slider, toggle, dropdown, color picker
- `x-range` — [min, max] for numeric types
- `x-step` — slider increment
- `x-restart-scope` — none | pipeline | renderer | game
- `x-runtime-scope` — frame | session | startup
- `x-experimental` — hidden unless experimental mode enabled
- `x-doc` — tooltip / description text

Codegen tool: Python script (`tools/codegen/gen_config.py`), invoked by:
- CMake custom command (before core target)
- Gradle task (before compileJava)

## Alternatives Considered

**Protobuf**: Strong serialization, poor fit for human-maintained settings schema.
No native x-metadata support. Would need wrapper layer for UI metadata anyway.

**FlatBuffers**: Zero-copy transport is overkill for config. Better candidate for
replay packets or high-volume bridge events later.

**Custom TOML/YAML DSL**: Maximum control but requires building parser + validator
from scratch. JSON Schema has existing validation tooling.

## Consequences

### Positive
- One file to edit when adding a new option
- Type-safe JNI bridge with no manual mirroring
- UI metadata lives next to the option definition
- Schema validation catches type mismatches at build time

### Negative
- Python dependency in build toolchain
- Generated code is harder to debug (mitigated by keeping generator simple)
- Migration of existing 157 options is tedious (one-time cost)

### Migration Impact
- Options.java shrinks from 5,247 lines to ~500 (bootstrap + custom logic)
- com_radiance_client_option_Options.cpp replaced entirely by generated bridge
- Existing options.properties format preserved (lossless migration)
- Rollback: delete generated/, revert to manual Options.java
