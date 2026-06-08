if(WIN32)
    add_definitions(-DNOMINMAX)
    set(GLSLVALIDATOR glslangValidator.exe) 
else()
    set(GLSLVALIDATOR glslangValidator)
endif(WIN32)

file(GLOB_RECURSE GLSL_SHADERS CONFIGURE_DEPENDS "*.vert" "*.frag" "*.rgen" "*.rmiss" "*.rchit" "*.rahit" "*.rint" "*.comp")
foreach(GLSL_SHADER IN LISTS GLSL_SHADERS)
    file(RELATIVE_PATH SHADER_REL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${GLSL_SHADER}")
    if(SHADER_REL_PATH MATCHES "^world/cloud/")
        list(REMOVE_ITEM GLSL_SHADERS "${GLSL_SHADER}")
    endif()
    if(SHADER_REL_PATH MATCHES "^world/ray_tracing/internal/")
        list(REMOVE_ITEM GLSL_SHADERS "${GLSL_SHADER}")
    endif()
endforeach()
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
if (MCVR_ENABLE_SHARC AND MCVR_ENABLE_SHARC_QUERY_PASS)
    list(APPEND SHADER_FLAGS "-DSHARC_QUERY_PASS_ENABLE=1")
    message(STATUS "SHARC isolated query pass: ON")
else()
    message(STATUS "SHARC isolated query pass: OFF")
endif()
if (MCVR_ENABLE_SHARC AND MCVR_SHARC_MAIN_TRACE_QUERY_MODE_ID GREATER 0)
    list(APPEND SHADER_FLAGS "-DSHARC_MAIN_TRACE_ENABLE=1")
    list(APPEND SHADER_FLAGS "-DSHARC_MAIN_TRACE_QUERY_MODE=${MCVR_SHARC_MAIN_TRACE_QUERY_MODE_ID}")
    message(STATUS "SHARC main-trace cache query: ${MCVR_SHARC_MAIN_TRACE_QUERY_MODE_LOWER} (${MCVR_SHARC_MAIN_TRACE_QUERY_MODE_ID})")
else()
    message(STATUS "SHARC main-trace cache query: OFF")
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
    FILES_MATCHING
        PATTERN "*.spv"
        PATTERN "world/cloud/*" EXCLUDE
)

install(
    DIRECTORY ${SPIRV_OUTPUT_PATH}/
    DESTINATION ${MCVR_INSTALL_LIB_DIR}/shaders
    FILES_MATCHING
        PATTERN "*.spv"
        PATTERN "world/cloud/*" EXCLUDE
)

install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/util/"
    DESTINATION "${MCVR_INSTALL_BIN_DIR}/res/util"
    FILES_MATCHING PATTERN "*.glsl"
)

install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/util/"
    DESTINATION "${MCVR_INSTALL_LIB_DIR}/shaders/util"
    FILES_MATCHING PATTERN "*.glsl"
)

set(ZIP_INSTALL_DEST "${MCVR_INSTALL_LIB_DIR}/shaders/world/ray_tracing")
set(RT_INTERNAL_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/world/ray_tracing/internal")
set(RT_UTIL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/util")
set(RT_COMMON_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../common")
set(RT_SHARC_SRC_DIR "${PROJECT_SOURCE_DIR}/extern/sharc")
set(RT_ZIP_STAGING_DIR "${CMAKE_CURRENT_BINARY_DIR}/ray_tracing_internal_zip")
set(RT_LEGACY_SHADER_PACK_ZIPS internal.zip restir-di.zip advanced.zip)

foreach(RT_PACK_NAME ${MCVR_BUILT_IN_SHADER_PACKS})
    if(NOT EXISTS "${RT_INTERNAL_ROOT_DIR}/${RT_PACK_NAME}/configs.json")
        message(FATAL_ERROR "Built-in shader pack is missing configs.json: ${RT_INTERNAL_ROOT_DIR}/${RT_PACK_NAME}")
    endif()
endforeach()
message(STATUS "Installing ray tracing built-in shader packs: ${MCVR_BUILT_IN_SHADER_PACKS}")

install(CODE "
    file(MAKE_DIRECTORY \"${ZIP_INSTALL_DEST}\")
    foreach(RT_LEGACY_SHADER_PACK_ZIP ${RT_LEGACY_SHADER_PACK_ZIPS})
        file(REMOVE \"${ZIP_INSTALL_DEST}/\${RT_LEGACY_SHADER_PACK_ZIP}\")
    endforeach()

    foreach(RT_PACK_NAME ${MCVR_BUILT_IN_SHADER_PACKS})
        set(RT_PACK_SRC_DIR \"${RT_INTERNAL_ROOT_DIR}/\${RT_PACK_NAME}\")
        set(RT_PACK_STAGING_DIR \"${RT_ZIP_STAGING_DIR}/\${RT_PACK_NAME}\")
        set(RT_PACK_OUTPUT \"${ZIP_INSTALL_DEST}/\${RT_PACK_NAME}.zip\")

        file(REMOVE_RECURSE \"\${RT_PACK_STAGING_DIR}\")
        file(MAKE_DIRECTORY \"\${RT_PACK_STAGING_DIR}\")
        file(COPY \"\${RT_PACK_SRC_DIR}/\" DESTINATION \"\${RT_PACK_STAGING_DIR}\")
        file(COPY \"${RT_UTIL_SRC_DIR}/\" DESTINATION \"\${RT_PACK_STAGING_DIR}/util\")
        file(COPY \"${RT_COMMON_SRC_DIR}/\" DESTINATION \"\${RT_PACK_STAGING_DIR}/common\")
        file(COPY \"${RT_SHARC_SRC_DIR}/\" DESTINATION \"\${RT_PACK_STAGING_DIR}/extern/sharc/include\")
        file(REMOVE \"\${RT_PACK_OUTPUT}\")

        execute_process(
            COMMAND \"${CMAKE_COMMAND}\" -E tar \"cf\" \"\${RT_PACK_OUTPUT}\" --format=zip .
            WORKING_DIRECTORY \"\${RT_PACK_STAGING_DIR}\"
            RESULT_VARIABLE ZIP_RESULT
            OUTPUT_VARIABLE ZIP_STDOUT
            ERROR_VARIABLE ZIP_STDERR
        )

        if(NOT ZIP_RESULT EQUAL 0)
            message(FATAL_ERROR \"Failed to create \${RT_PACK_NAME}.zip (code=\${ZIP_RESULT})\\nstdout: \${ZIP_STDOUT}\\nstderr: \${ZIP_STDERR}\")
        endif()
    endforeach()
")

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
