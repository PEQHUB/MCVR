# External Research And Best Practices

Use primary sources for technical decisions.

## Vulkan

- [Vulkan Ray Tracing Guide](https://docs.vulkan.org/guide/latest/extensions/ray_tracing.html)
  - Keep ray payloads, hit attributes, and callable data small.
  - Prefer device-local memory for acceleration structures.
- [Khronos Vulkan pipeline barrier performance sample](https://docs.vulkan.org/samples/latest/samples/performance/pipeline_barriers/README.html)
  - Avoid conservative barriers that serialize more than necessary.
  - Use the earliest correct source stage and latest correct destination stage.

## NVIDIA Ray Tracing

- [Best Practices for Using NVIDIA RTX Ray Tracing](https://developer.nvidia.com/blog/best-practices-for-using-nvidia-rtx-ray-tracing-updated/)
  - Prefer `PREFER_FAST_TRACE` for static BLAS.
  - Choose dynamic BLAS flags through measurement.
  - Rebuild after large deformation or topology changes.
  - Minimize any-hit area.
- [NVIDIA Ray Tracing Tips and Tricks](https://developer.nvidia.com/blog/?p=14120)
  - Batch and merge AS build/update work judiciously.
  - Use compaction for static geometry.
  - Use the right build flags for static versus dynamic geometry.
- [Creating Optimal Meshes for Ray Tracing](https://developer.nvidia.com/blog/creating-optimal-meshes-for-ray-tracing/)
  - Avoid pathological elongated/degenerate triangles.
  - Rebuild updatable structures when updates degrade traversal quality.

## SER And OMM

- [Path Tracing Optimization in Indiana Jones: SER and Live State Reductions](https://developer.nvidia.com/blog/path-tracing-optimization-in-indiana-jones-shader-execution-reordering-and-live-state-reductions)
  - SER can improve raygen coherence, but live-state cost matters.
  - Measure active threads per warp and spill/live-state effects.
- [Path Tracing Optimizations in Indiana Jones: OMM and Dynamic BLAS Compaction](https://developer.nvidia.com/blog/path-tracing-optimizations-in-indiana-jones-opacity-micromaps-and-compaction-of-dynamic-blass/)
  - OMM can reduce any-hit work for alpha-tested geometry.
  - Approximate/two-state OMM can be appropriate for indirect rays.
- [VK_EXT_ray_tracing_invocation_reorder proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_ray_tracing_invocation_reorder.html)
  - Invocation reordering is aimed at ray generation workloads and hit-object locality.

## ReSTIR And Radiance Caching

- [ReSTIR Direct Lighting](https://research.nvidia.com/labs/rtr/publication/bitterli2020spatiotemporal/)
  - Reservoir resampling can render many dynamic lights using few rays per pixel.
- [ReSTIR GI](https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing)
  - Path resampling can reduce indirect-lighting error for real-time path tracing.
- [NVIDIA RTXGI SHaRC/NRC](https://github.com/NVIDIA-RTX/RTXGI)
  - SHaRC/NRC style caches reduce cost by replacing deeper path evaluation with cache lookup.
- [NVIDIA RTXPT](https://github.com/NVIDIA-RTX/RTXPT)
  - Reference real-time path tracer using light-sampling caches, path-space layers, and guide-buffer generation.
- [AMD FSR Radiance Cache](https://gpuopen.com/manuals/fsr_sdk/techniques/radiance-cache/)
  - Radiance caching can reduce average traced path depth and share information between paths.
- [Megakernels Considered Harmful](https://research.nvidia.com/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus)
  - Large divergent GPU kernels can suffer from control-flow divergence and register pressure.

