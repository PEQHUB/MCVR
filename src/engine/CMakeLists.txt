# V2 Engine — gated behind MCVR_ENABLE_ENGINE_V2

set(ENGINE_SOURCES
    app/engine_app.cpp
    app/engine_session.cpp
    app/engine_services.cpp
    bridge/bridge_service.cpp
    config/config_service.cpp
    diagnostics/boot_trace.cpp
    diagnostics/diag_flags.cpp
    diagnostics/log.cpp
    diagnostics/metrics_service.cpp
    frame/frame_scheduler.cpp
    frame/composite_pass.cpp
    frame/offscreen_target.cpp
    frame/resource_gc.cpp
    frame/screenshot_capture.cpp
    platform/vulkan/vk2_buffer.cpp
    platform/vulkan/vk2_descriptor.cpp
    platform/vulkan/vk2_device.cpp
    platform/vulkan/vk2_image.cpp
    platform/vulkan/vk2_pipeline.cpp
    platform/vulkan/vk2_probe.cpp
    platform/vulkan/vk2_rt_pipeline.cpp
    platform/vulkan/vk2_shader.cpp
    platform/vulkan/vk2_swapchain.cpp
    rendergraph/barrier_planner.cpp
    rendergraph/graph_builder.cpp
    rendergraph/graph_resource_pool.cpp
    scene/blas_service.cpp
    scene/chunk_registry.cpp
    scene/entity_blas_service.cpp
    scene/entity_registry.cpp
    scene/gpu_upload_service.cpp
    scene/scene_resource_service.cpp
    scene/scene_service.cpp
    scene/texture_service.cpp
    scene/tlas_service.cpp
    diagnostics/replay/replay_recorder.cpp
    features/postprocess_adapter.cpp
    features/atmosphere_adapter.cpp
    features/raytracing_adapter.cpp
    features/rt_descriptor_layout.cpp
    features/dlss_adapter.cpp
    features/temporal_denoiser_adapter.cpp
    features/test_compute_adapter.cpp
    features/tonemapping_adapter.cpp
    features/color_to_depth_adapter.cpp
    features/post_render_adapter.cpp
    features/starfield_adapter.cpp
    features/cloud_adapter.cpp
    features/frame_gen_adapter.cpp
    ${CMAKE_SOURCE_DIR}/src/generated/bridge/config_bridge.cpp
)

add_library(engine STATIC ${ENGINE_SOURCES})

target_include_directories(engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/extern/spdlog/include
    ${CMAKE_SOURCE_DIR}/src/generated/config
    ${CMAKE_SOURCE_DIR}/extern/vma/include
    ${CMAKE_SOURCE_DIR}/extern/vulkan_headers/include
    ${CMAKE_SOURCE_DIR}/extern/volk
    ${CMAKE_SOURCE_DIR}/extern/glm
    ${CMAKE_SOURCE_DIR}/extern/glfw/include
    ${CMAKE_SOURCE_DIR}/extern/stb
    ${CMAKE_SOURCE_DIR}/extern/DLSS/include
)

# Link NVIDIA NGX (DLSS SDK) — required for DlssAdapter
# Link Streamline — required for DLSS-G FrameGenAdapter and Reflex
target_link_libraries(engine PUBLIC ngx streamline)

target_compile_definitions(engine PUBLIC
    VK_NO_PROTOTYPES       # Required for volk meta-loader
)
# spdlog: header-only mode — do NOT define SPDLOG_COMPILED_LIB
# (any definition, even =0, triggers compiled-lib mode)

if (MSVC)
    target_compile_options(engine PRIVATE /utf-8)
endif()

# JNI headers needed for config_bridge.cpp
find_package(JNI REQUIRED)
target_include_directories(engine PRIVATE ${JNI_INCLUDE_DIRS})

# Ensure codegen runs before engine builds
add_dependencies(engine generate_config)
