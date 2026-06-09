# Legacy → V2 Subsystem Mapping

## C++ (MCVR)

### Core Runtime

| Legacy | V2 | Phase |
|--------|-----|-------|
| renderer.hpp / renderer.cpp | engine/app/engine_app.hpp | 3 |
| renderer.hpp (global state) | engine/app/engine_session.hpp, engine_services.hpp | 3 |
| render_framework.cpp (acquire) | engine/frame/frame_scheduler.hpp | 4 |
| render_framework.cpp (submit) | engine/frame/submission_service.hpp | 4 |
| render_framework.cpp (present) | engine/frame/present_service.hpp | 4 |
| render_framework.cpp (recreate) | engine/frame/swapchain_service.hpp | 4 |
| render_framework.cpp (overlay) | engine/frame/overlay_service.hpp | 4 |
| render_framework.cpp (GC) | engine/frame/resource_gc.hpp | 4 |

### Vulkan Wrappers (41 files)

| Legacy | V2 | Phase |
|--------|-----|-------|
| vulkan/buffer.cpp | engine/platform/vulkan/vk2_buffer.hpp | 5 |
| vulkan/image.cpp | engine/platform/vulkan/vk2_image.hpp | 5 |
| vulkan/as.cpp (BLAS/TLAS builders) | engine/rt/blas_service.hpp, tlas_service.hpp | 7 |
| vulkan/device.cpp | engine/platform/vulkan/vk2_device.hpp | 3 |
| vulkan/swapchain.cpp | engine/frame/swapchain_service.hpp | 4 |
| vulkan/pipeline.cpp | engine/rendergraph/pass_registry.hpp | 5 |
| vulkan/descriptor.cpp | engine/platform/vulkan/vk2_descriptor.hpp | 5 |

### Pipeline

| Legacy | V2 | Phase |
|--------|-----|-------|
| pipeline.cpp | engine/rendergraph/graph_builder.hpp | 5 |
| Pipeline.java (assembleDefault) | engine/rendergraph/graph_compiler.hpp | 5 |
| YAML configs (resources/modules/) | Same YAML, read by native graph compiler | 5 |

### Scene / Chunks / RT

| Legacy | V2 | Phase |
|--------|-----|-------|
| chunks.cpp (geometry upload) | engine/scene/chunk_geometry.hpp | 6 |
| chunks.cpp (BLAS build/schedule) | engine/rt/blas_service.hpp | 7 |
| chunks.cpp (compaction) | engine/rt/compaction_policy.hpp | 7 |
| chunks.hpp (ChunkBuildScheduler) | engine/rt/blas_build_scheduler.hpp | 7 |
| world.cpp | engine/scene/world_state.hpp | 6 |
| entities.cpp | engine/scene/entity_state.hpp | 6 |
| world_prepare.cpp (cache) | engine/rt/tlas_cache.hpp | 7 |
| world_prepare.cpp (TLAS build) | engine/rt/tlas_service.hpp | 7 |
| ray_tracing_module.cpp | engine/rt/rt_pipeline.hpp + engine/rt/rt_dispatch.hpp | 7 |
| textures.cpp | engine/scene/texture_service.hpp | 6 |
| lights.cpp | engine/scene/light_service.hpp | 6 |

### Feature Modules

| Legacy | V2 | Phase |
|--------|-----|-------|
| modules/world/dlss/ | engine/features/dlss_adapter.hpp | 8 |
| modules/world/fsr_upscaler/ | engine/features/fsr_adapter.hpp | 8 |
| modules/world/nrd/ | engine/features/nrd_adapter.hpp | 8 |
| modules/world/svgf/ | engine/features/svgf_adapter.hpp | 8 |
| modules/world/tone_mapping/ | engine/features/tonemapping_adapter.hpp | 8 |
| modules/world/post_render/ | engine/features/post_render_adapter.hpp | 8 |
| modules/world/cloud/ | engine/features/cloud_adapter.hpp | 8 |
| modules/world/frame_gen/ | engine/features/framegen_adapter.hpp | 8 |
| modules/world/temporal_accumulation/ | engine/features/taa_adapter.hpp | 8 |

### Bridge / JNI

| Legacy | V2 | Phase |
|--------|-----|-------|
| middleware/com_radiance_*_Options.cpp (157 methods) | generated/bridge/config_bridge.cpp | 1 |
| middleware/com_radiance_*_RendererProxy.cpp | engine/bridge/renderer_commands.hpp | 6 |
| middleware/com_radiance_*_ChunkProxy.cpp | engine/bridge/chunk_commands.hpp | 6 |
| middleware/com_radiance_*_TextureProxy.cpp | engine/bridge/texture_commands.hpp | 6 |
| middleware/com_radiance_*_Pipeline.cpp | engine/bridge/pipeline_commands.hpp | 5 |
| middleware/com_radiance_*_BufferProxy.cpp | engine/bridge/buffer_commands.hpp | 6 |
| Other 10 middleware files | engine/bridge/* (grouped by domain) | 6-9 |

### Diagnostics

| Legacy | V2 | Phase |
|--------|-----|-------|
| aftermath_integration.cpp | engine/diagnostics/aftermath.hpp | 2 |
| gpu_diagnostics.cpp | engine/diagnostics/gpu_markers.hpp | 2 |
| crash_ring (custom) | engine/diagnostics/crash_ring.hpp | 2 |
| Ad-hoc std::cout/cerr | engine::log facade | 2 |

## Java (Radiance)

| Legacy | V2 | Phase |
|--------|-----|-------|
| option/Options.java (5247 lines) | v2/config/GeneratedConfig.java (~200 lines) | 1 |
| pipeline/Pipeline.java (assembleDefault) | v2/pipeline/PipelineEditor.java (config only) | 5 |
| proxy/vulkan/RendererProxy.java | v2/bridge/RendererBridge.java | 6 |
| proxy/vulkan/ChunkProxy.java | v2/bridge/ChunkBridge.java | 6 |
| proxy/vulkan/TextureProxy.java | v2/bridge/TextureBridge.java | 6 |
| proxy/world/* | v2/extract/* | 9 |
| gui/* | v2/ui/* (schema-driven) | 9 |
| mixins/vulkan_render_integration/ (52) | v2/extract/* (thin hooks only) | 9 |
| mixins/vanilla_resource_tracker/ (17) | Stays (minimal, correct scope) | — |

## Shader Code

Shaders (147 files, 18,587 lines) are NOT rewritten. They stay in `src/shader/`.
The v2 render graph will dispatch the same shaders with the same descriptor layouts.
Only change: descriptor set allocation moves from manual to graph-managed.
