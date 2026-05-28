if(WIN32)
    add_definitions(-DNOMINMAX)
    set(GLSLVALIDATOR glslangValidator.exe) 
else()
    set(GLSLVALIDATOR glslangValidator)
endif(WIN32)

file(GLOB_RECURSE GLSL_SHADERS CONFIGURE_DEPENDS "*.vert" "*.frag" "*.rgen" "*.rmiss" "*.rchit" "*.rahit" "*.rint" "*.comp")
file(GLOB_RECURSE GLSL_DEPENDS CONFIGURE_DEPENDS "../common/*.hpp" "../common/*.glsl" "util/*.glsl")

set(SPIRV_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SPIRV_OUTPUT_PATH})

set(SPIRV_OUTPUT_FILES "")

set(SHADER_FLAGS "$<$<CONFIG:Debug>:-g>" "$<$<CONFIG:Debug>:-Od>")

# Nsight Graphics shader debug info (off by default — 3x SPIR-V size, may regress RT performance)
# Enable with: cmake -DMCVR_SHADER_DEBUG_INFO=ON ...
if(MCVR_SHADER_DEBUG_INFO)
    list(APPEND SHADER_FLAGS "-gVS")
    message(STATUS "MCVR shader debug info for Nsight: ON (-gVS, Release optimizations preserved)")
else()
    message(STATUS "MCVR shader debug info for Nsight: OFF")
endif()
if (MCVR_ENABLE_SHARC)
    list(APPEND SHADER_FLAGS "-DSHARC_COMPILE_ENABLE=1")
endif()

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
        COMMAND ${GLSLVALIDATOR} ${SHADER_FLAGS} -V "${GLSL_SHADER}" -o "${SPIRV_FILE}" --target-env vulkan1.4 -I${PROJECT_SOURCE_DIR}/src -I${CMAKE_CURRENT_SOURCE_DIR}/util -I${PROJECT_SOURCE_DIR}/extern/FidelityFX-SDK/sdk/include/FidelityFX/gpu -I${PROJECT_SOURCE_DIR}/extern/sharc
        DEPENDS "${GLSL_SHADER}" ${GLSL_DEPENDS}
        VERBATIM
        COMMAND_EXPAND_LISTS
    )
    
    list(APPEND SPIRV_OUTPUT_FILES "${SPIRV_FILE}")
endforeach()

set(DISPLACEMENT_VARIANT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/world/ray_tracing/world_solid_transparent.rchit")
set(DISPLACEMENT_VARIANT_OUT "${SPIRV_OUTPUT_PATH}/world/ray_tracing/world_solid_transparent_no_displacement_rchit.spv")
add_custom_command(
    OUTPUT "${DISPLACEMENT_VARIANT_OUT}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SPIRV_OUTPUT_PATH}/world/ray_tracing"
    COMMAND ${GLSLVALIDATOR} ${SHADER_FLAGS} -DRARSER_SHADER_DISPLACEMENT=0 -V "${DISPLACEMENT_VARIANT_SRC}" -o "${DISPLACEMENT_VARIANT_OUT}" --target-env vulkan1.4 -I${PROJECT_SOURCE_DIR}/src -I${CMAKE_CURRENT_SOURCE_DIR}/util -I${PROJECT_SOURCE_DIR}/extern/FidelityFX-SDK/sdk/include/FidelityFX/gpu -I${PROJECT_SOURCE_DIR}/extern/sharc
    DEPENDS "${DISPLACEMENT_VARIANT_SRC}" ${GLSL_DEPENDS}
    VERBATIM
    COMMAND_EXPAND_LISTS
)
list(APPEND SPIRV_OUTPUT_FILES "${DISPLACEMENT_VARIANT_OUT}")

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

if(WIN32 AND MCVR_SHADER_DEBUG_INFO)
    find_program(SPIRV_DIS spirv-dis.exe HINTS "$ENV{VULKAN_SDK}/Bin")
    if(SPIRV_DIS)
        add_custom_target(verify_nsight_shaders
            COMMAND powershell.exe -ExecutionPolicy Bypass -File "${PROJECT_SOURCE_DIR}/tools/verify_nsight_shaders.ps1" -ShaderDir "${SPIRV_OUTPUT_PATH}" -SpirvDis "${SPIRV_DIS}"
            DEPENDS shaders
            COMMENT "Verifying Nsight shader debug info in ${SPIRV_OUTPUT_PATH}"
            VERBATIM
        )
    else()
        message(WARNING "spirv-dis.exe was not found; verify_nsight_shaders target will not be available")
    endif()
endif()
