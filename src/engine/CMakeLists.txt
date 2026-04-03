# V2 Engine — gated behind MCVR_ENABLE_ENGINE_V2
# Currently: diagnostics (spdlog logger) only

set(ENGINE_SOURCES
    diagnostics/log.cpp
)

add_library(engine_diagnostics STATIC ${ENGINE_SOURCES})

target_include_directories(engine_diagnostics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/extern/spdlog/include
)

target_compile_definitions(engine_diagnostics PUBLIC
    SPDLOG_COMPILED_LIB=0  # header-only mode
)

if (MSVC)
    target_compile_options(engine_diagnostics PRIVATE /utf-8)
endif()

# Ensure codegen runs before engine builds (for config headers)
add_dependencies(engine_diagnostics generate_config)
