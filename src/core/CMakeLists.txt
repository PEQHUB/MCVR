file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS "*.cpp")

add_library(core SHARED ${SOURCE_FILES})
target_link_libraries(core PUBLIC ngx ${JNI_LIBRARIES} VulkanMemoryAllocator volk::volk glfw streamline minizip tinyexpr)
target_include_directories(core PRIVATE ${PROJECT_SOURCE_DIR}/extern/json/include ${PROJECT_SOURCE_DIR}/extern/minizip-ng)

if(DEFINED ENV{VULKAN_SDK} AND NOT "$ENV{VULKAN_SDK}" STREQUAL "")
    set(MCVR_VULKAN_SDK "$ENV{VULKAN_SDK}")
else()
    set(MCVR_VULKAN_SDK "C:/VulkanSDK/1.4.341.1")
endif()

find_path(SHADERC_INCLUDE_DIR shaderc/shaderc.hpp
    HINTS
        "${MCVR_VULKAN_SDK}/Include"
        "${MCVR_VULKAN_SDK}/include"
        "${MCVR_VULKAN_SDK}/x86_64/include"
)
find_library(SHADERC_LIBRARY
    NAMES shaderc_combined shaderc_shared shaderc
    HINTS
        "${MCVR_VULKAN_SDK}/Lib"
        "${MCVR_VULKAN_SDK}/lib"
        "${MCVR_VULKAN_SDK}/x86_64/lib"
)

if(NOT SHADERC_INCLUDE_DIR OR NOT SHADERC_LIBRARY)
    message(FATAL_ERROR "shaderc was not found. Set VULKAN_SDK or install shaderc.")
endif()

target_include_directories(core PUBLIC ${SHADERC_INCLUDE_DIR} ${TINYEXPR_INCLUDE_DIR})
target_link_libraries(core PUBLIC ${SHADERC_LIBRARY})
# NVIDIA Aftermath SDK for GPU crash dump collection (headers only — DLL loaded at runtime)
set(AFTERMATH_SDK_DIR "${CMAKE_SOURCE_DIR}/extern/aftermath")
target_include_directories(core PUBLIC ${GENTOOL_OUTPUT_DIR} ${VOLK_INCLUDE_DIR} ${GLFW_INCLUDE_DIR} ${VMA_INCLUDE_DIR} ${STB_INCLUDE_DIR} ${JNI_INCLUDE_DIRS} ${GLM_INCLUDE_DIR} ${AFTERMATH_SDK_DIR}/include)
target_compile_definitions(core PUBLIC VK_NO_PROTOTYPES CORE_LIB)
if (USE_AMD)
    target_compile_definitions(core PUBLIC USE_AMD)
endif ()
if (MCVR_ENABLE_FFX_UPSCALER)
    # amd_fidelityfx_vk provides the main FFX API (ffxCreateContext, ffxQuery, etc.)
    target_link_libraries(core PUBLIC amd_fidelityfx_vk ffx_backend_vk_x64 ffx_fsr3upscaler_x64)
    target_compile_definitions(core PUBLIC MCVR_ENABLE_FFX_UPSCALER)
endif ()
if (MCVR_ENABLE_SHARC)
    target_compile_definitions(core PUBLIC MCVR_ENABLE_SHARC)
endif ()
if (MCVR_ENABLE_SHARC AND MCVR_ENABLE_SHARC_BUFFERS)
    target_compile_definitions(core PUBLIC MCVR_ENABLE_SHARC_BUFFERS)
endif ()
if (MCVR_ENABLE_SHARC AND MCVR_ENABLE_SHARC_UPDATE_PASS)
    target_compile_definitions(core PUBLIC MCVR_ENABLE_SHARC_UPDATE_PASS)
endif ()
if (MCVR_ENABLE_SHARC AND MCVR_ENABLE_SHARC_RESOLVE_PASS)
    target_compile_definitions(core PUBLIC MCVR_ENABLE_SHARC_RESOLVE_PASS)
endif ()
if (MCVR_ENABLE_SHARC AND MCVR_ENABLE_SHARC_QUERY_PASS)
    target_compile_definitions(core PUBLIC MCVR_ENABLE_SHARC_QUERY_PASS)
endif ()
target_compile_definitions(core PUBLIC MCVR_SHARC_MAIN_TRACE_QUERY_MODE=${MCVR_SHARC_MAIN_TRACE_QUERY_MODE_ID})
if (MCVR_ENABLE_SHARC AND MCVR_SHARC_MAIN_TRACE_QUERY_MODE_ID GREATER 0)
    target_compile_definitions(core PUBLIC MCVR_ENABLE_SHARC_MAIN_TRACE_QUERY)
endif ()
if (MCVR_ENABLE_NRD)
    target_link_libraries(core PUBLIC NRD)
    target_include_directories(core PUBLIC ${NRD_INCLUDE_DIR})
    target_compile_definitions(core PUBLIC NRD_STATIC_LIBRARY=1)
    target_compile_definitions(core PUBLIC MCVR_ENABLE_NRD)
endif ()
if (MSVC)
    target_compile_definitions(core PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(core PRIVATE /utf-8)
    target_link_libraries(core PRIVATE d3d11 dxgi dcomp dwmapi)
endif ()

set_target_properties(core PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)

install(TARGETS core DESTINATION ${MCVR_INSTALL_LIB_DIR})
install(TARGETS core DESTINATION ${MCVR_INSTALL_BIN_DIR})
