if(WIN32)
    add_definitions(-DNOMINMAX)
    set(GLSLVALIDATOR glslangValidator.exe) 
else()
    set(GLSLVALIDATOR glslangValidator)
endif(WIN32)

file(GLOB_RECURSE GLSL_SHADERS CONFIGURE_DEPENDS "*.vert" "*.frag" "*.rgen" "*.rmiss" "*.rchit" "*.rahit" "*.comp")
file(GLOB_RECURSE GLSL_DEPENDS CONFIGURE_DEPENDS "../common/*.hpp" "../common/*.glsl" "util/*.glsl")

set(SPIRV_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SPIRV_OUTPUT_PATH})

set(SPIRV_OUTPUT_FILES "")

set(SHADER_FLAGS "$<$<CONFIG:Debug>:-g>" "$<$<CONFIG:Debug>:-Od>")

foreach(GLSL_SHADER ${GLSL_SHADERS})
    file(RELATIVE_PATH REL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${GLSL_SHADER}")
    get_filename_component(REL_DIR "${REL_PATH}" DIRECTORY)
    get_filename_component(FILE_NAME "${GLSL_SHADER}" NAME_WE)
    get_filename_component(FILE_EXT "${GLSL_SHADER}" LAST_EXT)
    string(REPLACE "." "" FILE_TYPE "${FILE_EXT}")
    
    set(OUTPUT_DIR "${SPIRV_OUTPUT_PATH}/${REL_DIR}")
    set(SPIRV_FILE "${OUTPUT_DIR}/${FILE_NAME}_${FILE_TYPE}.spv")
    
    add_custom_command(
        OUTPUT "${SPIRV_FILE}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUTPUT_DIR}"
        COMMAND ${GLSLVALIDATOR} ${SHADER_FLAGS} -V "${GLSL_SHADER}" -o "${SPIRV_FILE}" --target-env vulkan1.4 -I${PROJECT_SOURCE_DIR}/src -I${CMAKE_CURRENT_SOURCE_DIR}/util -I${PROJECT_SOURCE_DIR}/extern/FidelityFX-SDK/sdk/include/FidelityFX/gpu
        DEPENDS "${GLSL_SHADER}" ${GLSL_DEPENDS}
        VERBATIM
        COMMAND_EXPAND_LISTS
    )
    
    list(APPEND SPIRV_OUTPUT_FILES "${SPIRV_FILE}")
endforeach()

add_custom_target(shaders ALL
    DEPENDS ${SPIRV_OUTPUT_FILES} ${GLSL_DEPENDS}
)

install(
    DIRECTORY ${SPIRV_OUTPUT_PATH}/
    DESTINATION ${MCVR_INSTALL_BIN_DIR}/res
    FILES_MATCHING PATTERN "*.spv"
)

install(
    DIRECTORY ${SPIRV_OUTPUT_PATH}/
    DESTINATION ${MCVR_INSTALL_LIB_DIR}/shaders
    FILES_MATCHING PATTERN "*.spv"
)
