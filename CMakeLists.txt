cmake_minimum_required(VERSION 3.15)

project(minecraft-vulkan-renderer)

if(NOT DEFINED JAVA_PROJECT_ROOT_DIR OR "${JAVA_PROJECT_ROOT_DIR}" STREQUAL "")
    message(FATAL_ERROR "JAVA_PROJECT_ROOT_DIR is required. Pass -DJAVA_PROJECT_ROOT_DIR=/path/to/root/of/the/java/project/")
else()
    # Normalize path to use forward slashes
    file(TO_CMAKE_PATH "${JAVA_PROJECT_ROOT_DIR}" JAVA_PROJECT_ROOT_DIR)
    message(STATUS "JAVA_PROJECT_ROOT_DIR: ${JAVA_PROJECT_ROOT_DIR}")
endif()

set(CMAKE_CXX_STANDARD 23)
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

# OMM SDK (Opacity Micro-Map) - CPU baker only
# We bypass the OMM top-level CMake and add only the library + its deps,
# providing glm/stb from our existing project to avoid target conflicts.
option(MCVR_ENABLE_SHARC "Enable SHARC hash grid radiance cache build support" ON)
option(MCVR_ENABLE_SHARC_BUFFERS "Enable SHARC hash grid buffer allocation/init. Off by default until BLAS streaming lifetime is stable." OFF)
option(MCVR_ENABLE_SHARC_UPDATE_PASS "Enable SHARC sparse update RT pipeline creation/dispatch. Off by default until explicitly profiling SHARC." OFF)
option(MCVR_ENABLE_SHARC_RESOLVE_PASS "Enable SHARC resolve compute pipeline creation/dispatch. Off by default until explicitly profiling SHARC." OFF)
option(MCVR_ENABLE_SHARC_MAIN_TRACE_QUERY "Compile the small read-only SHARC cache query into the main world RT raygen. Off by default because current NVIDIA drivers crash compiling this path into the main RT pipeline." OFF)
option(MCVR_ENABLE_SHARC_QUERY_PASS "Compile the isolated SHARC candidate/query compute pass. Runtime default stays Off; this avoids SHARC hash/BDA reads in world.rgen." OFF)
option(MCVR_ENABLE_DIRECT_LIGHT_PIPELINE "Compile experimental upstream-style split direct-light pipeline scaffolding. Runtime default stays Legacy." OFF)
option(MCVR_ENABLE_RTXDI "Compile experimental RTXDI 3.0 direct-light backend scaffolding. Requires MCVR_ENABLE_DIRECT_LIGHT_PIPELINE and remains runtime-disabled by default." OFF)
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

option(MCVR_ENABLE_OMM "Enable Opacity Micro-Map SDK for AHS elimination" ON)
option(MCVR_SHADER_DEBUG_INFO "Emit NonSemantic Vulkan debug info with source (-gVS) for Nsight Graphics profiling. Off by default: 3x SPIR-V size, may regress RT pipeline performance." OFF)

if (MCVR_ENABLE_OMM)
    message(STATUS "OMM SDK Enabled")

    # glm target (OMM expects a 'glm' CMake target; we provide our existing headers)
    if(NOT TARGET glm)
        add_library(glm INTERFACE)
        target_include_directories(glm INTERFACE ${GLM_INCLUDE_DIR})
    endif()

    # stb_lib target (OMM expects 'stb_lib' for stb_image.h / stb_image_write.h)
    if(NOT TARGET stb_lib)
        add_library(stb_lib INTERFACE)
        target_include_directories(stb_lib INTERFACE ${STB_INCLUDE_DIR})
    endif()

    # lz4 (from OMM's external, static only)
    set(LZ4_BUILD_CLI OFF CACHE BOOL "" FORCE)
    set(LZ4_BUILD_LEGACY_LZ4C OFF CACHE BOOL "" FORCE)
    add_subdirectory(extern/omm/external/lz4/build/cmake EXCLUDE_FROM_ALL)

    # xxHash (from OMM's external)
    set(XXHASH_BUILD_XXHSUM OFF CACHE BOOL "" FORCE)
    add_subdirectory(extern/omm/external/xxHash/build/cmake EXCLUDE_FROM_ALL)

    # OMM library - static, CPU-only (no precompiled shaders)
    set(OMM_LIB_TARGET_NAME omm-lib)
    set(OMM_PROJECT_FOLDER "External/OMM")
    set(OMM_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
    set(OMM_ENABLE_PRECOMPILED_SHADERS_DXIL OFF CACHE BOOL "" FORCE)
    set(OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV OFF CACHE BOOL "" FORCE)
    set(OMM_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
    set(OMM_LIB_INSTALL OFF CACHE BOOL "" FORCE)
    # OMM lib uses PROJECT_VERSION which requires a version-enabled project scope
    set(PROJECT_VERSION "1.0.0")
    add_subdirectory(extern/omm/libraries/omm-lib EXCLUDE_FROM_ALL)
    unset(PROJECT_VERSION)
    # Remove stb_lib.cpp (duplicate STB_IMAGE_IMPLEMENTATION) and debug_impl.cpp
    # (debug visualization, not needed) from omm-lib sources.
    get_target_property(_omm_sources omm-lib SOURCES)
    list(FILTER _omm_sources EXCLUDE REGEX "(stb_lib|debug_impl)\\.cpp$")
    set_target_properties(omm-lib PROPERTIES SOURCES "${_omm_sources}")
else()
    message(STATUS "OMM SDK Disabled")
endif()

add_subdirectory(src)
