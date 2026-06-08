# Phase 3: Acceleration Structures

Goal: reduce the about 2.2 ms `RT.BLAS_TLAS` block.

## Evidence

- `RT.BLAS_TLAS` wraps `worldPrepareContext->render()`.
- CPU world prepare is about 0.9 to 1.3 ms in the fresh run, so the 2.2 ms is mostly GPU command time.
- Steady `cpu_timing.log` shows mostly TLAS updates, not full builds.
- TLAS instance data is rebuilt into a new host-visible buffer every frame.
- Chunk BLAS currently uses `VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR`.

## Leads

### Split AS Timing

Do not optimize this block blindly. First split:

- TLAS build
- TLAS update
- chunk BLAS work
- entity BLAS work
- SBT setup
- metadata upload
- mega-BLAS work

### Persistent TLAS Instance Buffer

Current path creates a fresh host-visible instance buffer in `as.cpp` every frame. Replace with a persistent grow-only or ring-buffered allocation sized by capacity.

Expected benefit:

- lower allocation churn
- clearer GPU timeline
- easier future partial update logic

### Static/Dynamic Separation

Separate stable chunk geometry from frequently changing entities or animated geometry where practical:

- static chunk TLAS/BLAS path
- dynamic entity TLAS path
- composed top-level traversal strategy if supported cleanly

### Build Flags

A/B test build flags:

- static chunks: `PREFER_FAST_TRACE`, optionally compaction
- dynamic entities: keep fast build or update based on measured deformation/update behavior
- TLAS: compare update versus rebuild at current instance counts

### Instance Count Reduction

Investigate:

- chunk instance merging
- mega-BLAS only after async/incremental build is safe
- culling and lower-frequency updates for distant/stable instances
- SBT caching by unchanged hit layout/signature

## Tests

- Compare steady scene with 4k-ish output and internal 1920x1080 trace.
- Measure `RT.WP.TLASUpdate` and `RT.WP.TLASBuild` separately.
- Track instance count, build count, update count, scratch size, AS memory, and stutters.
- Confirm no frame references freed BLAS/buffers before GPU completion.

## Exit Criteria

- `RT.BLAS_TLAS` is broken into named costs.
- At least one measured reduction is landed or a clear reason exists why the 2.2 ms is unavoidable for the scene.

