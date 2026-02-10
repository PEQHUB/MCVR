#include "core/all_extern.hpp"
#include <cstdlib>
#include <iostream>

#if defined(CORE_LIB)
PFN_glfwInit p_glfwInit = nullptr;
PFN_glfwTerminate p_glfwTerminate = nullptr;
PFN_glfwGetWindowSize p_glfwGetWindowSize = nullptr;
PFN_glfwCreateWindowSurface p_glfwCreateWindowSurface = nullptr;
PFN_glfwGetRequiredInstanceExtensions p_glfwGetRequiredInstanceExtensions = nullptr;
PFN_glfwSetWindowTitle p_glfwSetWindowTitle = nullptr;
PFN_glfwSetFramebufferSizeCallback p_glfwSetFramebufferSizeCallback = nullptr;
PFN_glfwGetFramebufferSize p_glfwGetFramebufferSize = nullptr;
PFN_glfwWaitEvents p_glfwWaitEvents = nullptr;
#endif