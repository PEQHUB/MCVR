# Streamline SDK — headers-only target (all functions loaded dynamically at runtime)
add_library(streamline INTERFACE)
target_include_directories(streamline INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Install DLLs alongside core.dll (same pattern as DLSS snippet DLLs)
file(GLOB __SL_DLLS "${CMAKE_CURRENT_SOURCE_DIR}/bin/*.dll")
message(STATUS "Streamline DLLs: ${__SL_DLLS}")
install(FILES ${__SL_DLLS} DESTINATION ${MCVR_INSTALL_BIN_DIR})
