# V2 Architecture

This directory contains the design documents for the RadSER engine rewrite.

## Documents

| File | Purpose |
|------|---------|
| [baseline.md](baseline.md) | Codebase metrics snapshot, parity checklist |
| [migration-rules.md](migration-rules.md) | Hard/soft rules for all new code |
| [legacy-to-v2-mapping.md](legacy-to-v2-mapping.md) | Every legacy file → v2 replacement, by phase |

## ADRs (Architecture Decision Records)

| ADR | Title | Status |
|-----|-------|--------|
| [001](adr-001-json-schema-codegen.md) | JSON Schema + Python Codegen for Config | Accepted |
| [002](adr-002-spdlog-facade.md) | spdlog Behind engine::log Facade | Accepted |
| [003](adr-003-dual-mode-activation.md) | Build Flag + Runtime Flag for V2 Activation | Accepted |
