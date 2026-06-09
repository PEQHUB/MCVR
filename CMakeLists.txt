cmake_minimum_required(VERSION 3.15)

project(minecraft-vulkan-renderer LANGUAGES C CXX)

if(NOT DEFINED JAVA_PROJECT_ROOT_DIR OR "${JAVA_PROJECT_ROOT_DIR}" STREQUAL "")
    message(FATAL_ERROR "JAVA_PROJECT_ROOT_DIR is required. Pass -DJAVA_PROJECT_ROOT_DIR=/path/to/root/of/the/java/project/")
else()
    # Normalize path to use forward slashes
    file(TO_CMAKE_PATH "${JAVA_PROJECT_ROOT_DIR}" JAVA_PROJECT_ROOT_DIR)
    message(STATUS "JAVA_PROJECT_ROOT_DIR: ${JAVA_PROJECT_ROOT_DIR}")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 99)
if (MSVC)
    # Generate PDB in Release for crash stack traces (no perf impact — debug info is in separate .pdb)
    add_compile_options($<$<CONFIG:Release>:/Zi>)
    add_link_options($<$<CONFIG:Release>:/DEBUG> $<$<CONFIG:Release>:/OPT:REF> $<$<CONFIG:Release>:/OPT:ICF>)
    add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)
else()
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g -O0")
        add_compile_definitions(DEBUG)
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O2")
    endif()
endif()

set(MCVR_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src ${JAVA_PROJECT_ROOT_DIR}/src/main/native/include/)
message(STATUS "Include dir: ${MCVR_INCLUDE_DIR}")

set(MCVR_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
message(STATUS "Install bin dir: ${MCVR_INSTALL_BIN_DIR}")

set(MCVR_INSTALL_LIB_DIR ${JAVA_PROJECT_ROOT_DIR}/src/main/resources/)
message(STATUS "Install lib dir: ${MCVR_INSTALL_LIB_DIR}")

set(MCVR_BUILT_IN_SHADER_PACKS vanilla-pt CACHE STRING "Built-in ray tracing shader packs to zip and install")

include_directories(${MCVR_INCLUDE_DIR})

# GLFW
add_subdirectory(extern/glfw EXCLUDE_FROM_ALL)
set(GLFW_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/glfw/include)

# Volk
set(VOLK_BUILD_TESTS OFF CACHE BOOL "Disable tests")
set(VOLK_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples")
add_subdirectory(extern/volk EXCLUDE_FROM_ALL)
set(VOLK_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/volk)

# VMA
add_subdirectory(extern/vma EXCLUDE_FROM_ALL)
set(VMA_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/vma/include)

# GLM
set(GLM_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/glm)

# VK.XML
set(VK_XML_PATH ${PROJECT_SOURCE_DIR}/extern/vulkan_headers/registry/vk.xml)

# STB
set(STB_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/stb)

# Runtime shader-pack support (upstream RT executor)
add_library(tinyexpr STATIC extern/tinyexpr/tinyexpr.c)
set_target_properties(tinyexpr PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(TINYEXPR_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/tinyexpr)

# DLSS
add_subdirectory(extern/DLSS EXCLUDE_FROM_ALL)

# Streamline SDK (Reflex, DLSS-G — dynamic loading, no .lib)
add_subdirectory(extern/streamline EXCLUDE_FROM_ALL)

# Install Streamline DLLs alongside core.dll and into JAR resources
file(GLOB __SL_DLLS "${PROJECT_SOURCE_DIR}/extern/streamline/bin/*.dll")
install(FILES ${__SL_DLLS} DESTINATION ${MCVR_INSTALL_BIN_DIR})
install(FILES ${__SL_DLLS} DESTINATION ${MCVR_INSTALL_LIB_DIR})

# Install nvngx_dlssg.dll (DLSS-G NGX snippet) into JAR resources — small enough to bundle (7.5MB)
set(__DLSSG_NGX "${PROJECT_SOURCE_DIR}/extern/DLSS/lib/Windows_x86_64/rel/nvngx_dlssg.dll")
if(EXISTS ${__DLSSG_NGX})
    install(FILES ${__DLSSG_NGX} DESTINATION ${MCVR_INSTALL_BIN_DIR})
    install(FILES ${__DLSSG_NGX} DESTINATION ${MCVR_INSTALL_LIB_DIR})
endif()

# JNI
find_package(JNI REQUIRED)
message(STATUS "${JNI_INCLUDE_DIRS}")
message(STATUS "${JNI_LIBRARIES}")

option(USE_AMD "Enable AMD specific optimizations/features" ON)
if (USE_AMD)
    message(STATUS "AMD Support Enabled")
else()
    message(STATUS "AMD Support Disabled")
endif()

# FidelityFX Upscaler (FSR3 Upscaler only)
option(MCVR_ENABLE_FFX_UPSCALER "Enable FidelityFX Upscaler support" ON)

if (MCVR_ENABLE_FFX_UPSCALER)
    message(STATUS "FidelityFX Upscaler Enabled")

    set(FFX_API_BACKEND VK_X64 CACHE STRING "FidelityFX API backend" FORCE)
    
    # Disable everything except FSR3 Upscaler
    set(FFX_DENOISER OFF CACHE BOOL "Build FFX Denoiser" FORCE)
    set(FFX_ALL OFF CACHE BOOL "Build all FFX components" FORCE)
    set(FFX_SSSR OFF CACHE BOOL "Build FFX SSSR" FORCE)
    set(FFX_FSR3 OFF CACHE BOOL "Build FFX FSR3" FORCE) # Full FSR3 (FrameGen) disabled
    set(FFX_FSR3UPSCALER ON CACHE BOOL "Build FFX FSR3 Upscaler" FORCE)
    set(FFX_FI OFF CACHE BOOL "Build FFX Frame Interpolation" FORCE)
    set(FFX_OF OFF CACHE BOOL "Build FFX Optical Flow" FORCE)
    
    set(FFX_AUTO_COMPILE_SHADERS ON CACHE BOOL "Compile shaders automatically as a prebuild step." FORCE)

    # Use ffx-api which includes sdk internally, providing the necessary API symbols (ffxQuery)
    set(FFX_BIN_PATH ${CMAKE_BINARY_DIR}/ffx_sdk CACHE STRING "FFX binary output path" FORCE)
    add_subdirectory(extern/FidelityFX-SDK/ffx-api EXCLUDE_FROM_ALL)

    if (TARGET ffx_backend_vk_x64)
        target_include_directories(ffx_backend_vk_x64 PRIVATE ${VOLK_INCLUDE_DIR})
    endif()
    
    # Note: ffx_denoiser_x64 target will NOT be created, so don't link against it.
else()
    message(STATUS "FidelityFX Upscaler Disabled")
endif()

# NRD Denoiser
option(MCVR_ENABLE_NRD "Enable NRD denoiser support" ON)

if (MCVR_ENABLE_NRD)
    message(STATUS "NRD Denoiser Enabled")

    set(NRD_STATIC_LIBRARY ON CACHE BOOL "Build NRD as a static library" FORCE)
    set(NRD_DISABLE_SHADER_COMPILATION ON CACHE BOOL "Disable NRD shader compilation" FORCE)
    set(NRD_SHADERS_PATH "${PROJECT_SOURCE_DIR}/extern/nrd/_Shaders" CACHE STRING "NRD shader headers path" FORCE)
    set(NRD_NORMAL_ENCODING "2" CACHE STRING "Normal encoding variant (0-4, matches nrd::NormalEncoding)" FORCE)
    set(NRD_ROUGHNESS_ENCODING "1" CACHE STRING "Roughness encoding variant (0-2, matches nrd::RoughnessEncoding)" FORCE)

    add_subdirectory(extern/nrd EXCLUDE_FROM_ALL)

    set(NRD_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/nrd/Include)

    if (TARGET NRD)
        target_include_directories(NRD PRIVATE "${NRD_SHADERS_PATH}")
    endif()
else()
    message(STATUS "NRD Denoiser Disabled")
endif()

# Runtime shader-pack support. Zip packs are not the RadSER production path yet,
# but upstream's loader is built around minizip, so keep it linked while bringing
# up the pass executor.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
set(MZ_BUILD_TESTS OFF CACHE BOOL "Disable minizip tests" FORCE)
set(MZ_BUILD_UNIT_TESTS OFF CACHE BOOL "Disable minizip unit tests" FORCE)
set(MZ_FETCH_ZLIB ON CACHE BOOL "Fetch zlib automatically" FORCE)
set(MZ_ZSTD OFF CACHE BOOL "Disable zstd in minizip-ng due to upstream target mismatch" FORCE)
add_subdirectory(extern/minizip-ng EXCLUDE_FROM_ALL)
set_target_properties(minizip PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(extern/json EXCLUDE_FROM_ALL)

option(MCVR_ENABLE_SHARC "Enable SHARC hash grid radiance cache build support" ON)
option(MCVR_ENABLE_SHARC_BUFFERS "Enable SHARC hash grid buffer allocation/init. Off by default until BLAS streaming lifetime is stable." OFF)
option(MCVR_ENABLE_SHARC_UPDATE_PASS "Enable SHARC sparse update RT pipeline creation/dispatch. Off by default until explicitly profiling SHARC." OFF)
option(MCVR_ENABLE_SHARC_RESOLVE_PASS "Enable SHARC resolve compute pipeline creation/dispatch. Off by default until explicitly profiling SHARC." OFF)
option(MCVR_ENABLE_SHARC_MAIN_TRACE_QUERY "Compile the small read-only SHARC cache query into the main world RT raygen. Off by default because current NVIDIA drivers crash compiling this path into the main RT pipeline." OFF)
option(MCVR_ENABLE_SHARC_QUERY_PASS "Compile the isolated SHARC candidate/query compute pass. Runtime default stays Off; this avoids SHARC hash/BDA reads in world.rgen." OFF)
set(MCVR_SHARC_MAIN_TRACE_QUERY_MODE "off" CACHE STRING "SHARC main trace query compile mode: off, stub, math, hash, or radiance")
set_property(CACHE MCVR_SHARC_MAIN_TRACE_QUERY_MODE PROPERTY STRINGS off stub math hash radiance)
string(TOLOWER "${MCVR_SHARC_MAIN_TRACE_QUERY_MODE}" MCVR_SHARC_MAIN_TRACE_QUERY_MODE_LOWER)
if(MCVR_ENABLE_SHARC_MAIN_TRACE_QUERY AND MCVR_SHARC_MAIN_TRACE_QUERY_MODE_LOWER STREQUAL "off")
    set(MCVR_SHARC_MAIN_TRACE_QUERY_MODE_LOWER "radiance")
endif()
if(MCVR_SHARC_MAIN_TRACE_QUERY_MODE_LOWER STREQUAL "off")
    set(MCVR_SHARC_MAIN_TRACE_QUERY_MODE_ID 0)
elseif(MCVR_SHARC_MAIN_TRACE_QUERY_MODE_LOWER STREQUAL "stub")
    set(MCVR_SHARC_MAIN_TRACE_QUERY_MODE_ID 1)
elseif(MCVR_SHARC_MAIN_TRACE_QUERY_MODE_LOWER STREQUAL "math")
    set(MCVR_SHARC_MAIN_TRACE_QUERY_MODE_ID 2)
elseif(MCVR_SHARC_MAIN_TRACE_QUERY_MODE_LOWER STREQUAL "hash")
    set(MCVR_SHARC_MAIN_TRACE_QUERY_MODE_ID 3)
elseif(MCVR_SHARC_MAIN_TRACE_QUERY_MODE_LOWER STREQUAL "radiance")
    set(MCVR_SHARC_MAIN_TRACE_QUERY_MODE_ID 4)
else()
    message(FATAL_ERROR "Invalid MCVR_SHARC_MAIN_TRACE_QUERY_MODE='${MCVR_SHARC_MAIN_TRACE_QUERY_MODE}'. Expected off, stub, math, hash, or radiance.")
endif()

option(MCVR_SHADER_DEBUG_INFO "Emit NonSemantic Vulkan debug info with source (-gVS) for Nsight Graphics profiling. Off by default: 3x SPIR-V size, may regress RT pipeline performance." OFF)

# --- Config codegen (schema → C++ structs + JNI bridge) ---
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(CONFIG_SCHEMA "${CMAKE_SOURCE_DIR}/schema/engine_config.schema.json")
set(CONFIG_CODEGEN "${CMAKE_SOURCE_DIR}/tools/codegen/gen_config.py")
set(GENERATED_CPP_DIR "${CMAKE_SOURCE_DIR}/src/generated/config")
set(GENERATED_BRIDGE_DIR "${CMAKE_SOURCE_DIR}/src/generated/bridge")
set(GENERATED_JAVA_DIR "${JAVA_PROJECT_ROOT_DIR}/src/generated/java/com/radiance/v2/config")

add_custom_command(
    OUTPUT
        "${GENERATED_CPP_DIR}/engine_config.hpp"
        "${GENERATED_CPP_DIR}/config_defaults.hpp"
        "${GENERATED_BRIDGE_DIR}/config_bridge.cpp"
    COMMAND ${Python3_EXECUTABLE} "${CONFIG_CODEGEN}" "${CONFIG_SCHEMA}"
        --cpp-out "${GENERATED_CPP_DIR}"
        --java-out "${GENERATED_JAVA_DIR}"
        --bridge-out "${GENERATED_BRIDGE_DIR}"
    DEPENDS "${CONFIG_SCHEMA}" "${CONFIG_CODEGEN}"
    COMMENT "Generating config from engine_config.schema.json"
)
add_custom_target(generate_config DEPENDS
    "${GENERATED_CPP_DIR}/engine_config.hpp"
    "${GENERATED_CPP_DIR}/config_defaults.hpp"
    "${GENERATED_BRIDGE_DIR}/config_bridge.cpp"
)

# --- V2 engine (behind build flag, default OFF) ---
option(MCVR_ENABLE_ENGINE_V2 "Enable v2 engine subsystems" OFF)

# --- Build info generation (Phase 0: Texture Loader v4) ---
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE MCVR_REPO_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE MCVR_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
execute_process(
    COMMAND git diff --quiet
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE MCVR_DIRTY_RESULT
    ERROR_QUIET
)
if(MCVR_DIRTY_RESULT EQUAL 0)
    set(MCVR_DIRTY 0)
else()
    set(MCVR_DIRTY 1)
endif()
string(TIMESTAMP MCVR_BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%SZ" UTC)
set(MCVR_DLL_SHA256 "unknown")

configure_file(
    ${CMAKE_SOURCE_DIR}/src/core/build/build_info.hpp.in
    ${CMAKE_SOURCE_DIR}/src/core/build/build_info.hpp
    @ONLY
)

add_subdirectory(src)
