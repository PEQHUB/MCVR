add_subdirectory(core)
add_subdirectory(shader)

if (MCVR_ENABLE_ENGINE_V2)
    add_subdirectory(engine)
endif()