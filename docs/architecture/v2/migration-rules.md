# V2 Migration Rules

These rules apply to ALL code changes from 2026-04-03 onward.

## Hard Rules (enforced, no exceptions)

1. **No new `Renderer::options` reads in engine/ code.**
   Config comes from EngineServices → ConfigService. Legacy code keeps its pattern.

2. **No new JNI one-off setters/getters** unless they go through the new bridge.
   All new options use the generated typed bridge.

3. **No new mixins that cancel large vanilla flows** without an ADR.
   Existing cancellation mixins are legacy debt. New extraction hooks only.

4. **No Vulkan creation function may leave a partially valid object alive.**
   Buffer/image/pipeline creation either succeeds fully or returns an error.
   No log-and-continue with null handles.

5. **Every new subsystem header documents ownership and thread-affinity.**
   Format:
   ```cpp
   // Ownership: EngineSession (1:1 lifetime)
   // Thread: main thread only (frame scheduling)
   // Dependencies: SwapchainService, DeviceService
   ```

6. **96-byte PBRTriangle stride is sacred.** Never changes in v2 either.

7. **No new raw `std::cout`, `std::cerr`, `exit(...)` in engine/ code.**
   Use `engine::log::*` and `engine::fatal()`.

## Soft Rules (follow unless ADR justifies exception)

1. **Prefer moving a file to engine/ over modifying it in core/.**
   If you're doing a major rewrite of a core/ file, move it to engine/ instead.

2. **Don't refactor legacy code "while you're in there."**
   Touch legacy only for bugfixes. The rewrite is the cleanup.

3. **One subsystem cutover per PR.**
   Don't bundle frame services + scene extraction in one PR.

4. **Test against parity checklist before declaring cutover complete.**
   See baseline.md for the full checklist.

5. **Pipeline YAML configs stay as the user-facing format.**
   The render graph compiler reads them. Don't invent a new format.

## File Placement

| New code goes in | NOT in |
|------------------|--------|
| `src/engine/` | `src/core/` |
| `src/generated/` | `src/core/middleware/` |
| Radiance `v2/` packages | Radiance `client/` packages |

## Naming Conventions

- Services: `FooService` (e.g., `SwapchainService`, `BlasService`)
- Contexts: `FooContext` (per-frame state, e.g., `FrameContext`)
- Builders: `FooBuilder` (construction helpers)
- Namespace: `engine::` for all v2 C++ code
- Java package: `com.radiance.v2.*`

## Deletion Protocol

Before deleting legacy code:
1. All parity tests pass with `engine.mode=v2`
2. One full benchmark run shows no regression
3. PR explicitly lists deleted files and the v2 replacement
4. 48-hour soak test (play sessions) with no crashes
