# V2 Engine — gated behind MCVR_ENABLE_ENGINE_V2

set(ENGINE_SOURCES
    app/engine_app.cpp
    app/engine_session.cpp
    app/engine_services.cpp
    bridge/bridge_service.cpp
    config/config_service.cpp
    diagnostics/log.cpp
    frame/frame_scheduler.cpp
    frame/resource_gc.cpp
    platform/vulkan/vk2_buffer.cpp
    platform/vulkan/vk2_image.cpp
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
)

target_compile_definitions(engine PUBLIC
    SPDLOG_COMPILED_LIB=0  # header-only mode
)

if (MSVC)
    target_compile_options(engine PRIVATE /utf-8)
endif()

# JNI headers needed for config_bridge.cpp
find_package(JNI REQUIRED)
target_include_directories(engine PRIVATE ${JNI_INCLUDE_DIRS})

# Ensure codegen runs before engine builds
add_dependencies(engine generate_config)
