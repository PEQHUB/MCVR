# Phase 5: Alpha, OMM, SER, Geometry, And Divergence

Goal: attack shader divergence, any-hit overhead, and geometry bandwidth once main trace and AS scopes are split.

## Alpha And Any-Hit

Current code uses any-hit alpha tests for transparent/cutout geometry. Do not remove that without replacement.

Leads:

- Add counters for any-hit invocations and ignored intersections.
- Test OMM on alpha-heavy geometry when available.
- Measure OMM bake time, memory, trace time, and visual correctness.
- Use approximate/two-state OMM only where acceptable for indirect rays.

## SER

SER is currently off. The shader has SER paths, and the GPU supports modern SER capabilities, but existing comments say prior profiling found SER harmful.

Plan:

- Re-test with current RTX 5090, current driver, and current shader.
- Measure active threads per warp and live-state/spill metrics in Nsight GPU Trace.
- Compare SER off, SER on, SER hints on.
- Do not default it on unless the current workload benefits.

## Vertex Format And Bandwidth

Full 96-byte `PBRTriangle` is currently forced because compact/lossless layout had issues.

Plan:

- Fix layout correctness in an isolated branch.
- Validate full, compact, and lossless paths against visual and crash tests.
- Measure trace time, memory bandwidth, BLAS build cost, and material correctness.
- Keep the 96-byte stride invariant until replacement is proven.

## Shader Live State

Large raygen and closest-hit shaders can accumulate expensive live state across trace calls.

Plan:

- Use Nsight GPU Trace shader profiler metrics when export/source view is available.
- Reduce payload size where possible.
- Move variables out of live regions around `traceRayEXT`.
- Consider smaller specialized paths only after measured hotspots prove they matter.

## Exit Criteria

- Any-hit, SER, and vertex-format changes have isolated A/B data.
- No alpha, glass, water, foliage, or material correctness regressions are accepted for a small perf win.

