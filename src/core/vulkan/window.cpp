#include "core/vulkan/window.hpp"

#include "core/vulkan/instance.hpp"

#include <iostream>

bool vk::Window::framebufferResized = false;

vk::Window::Window(std::shared_ptr<Instance> instance, uint32_t width, uint32_t height)
    : instance_(instance), width_(width), height_(height) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // TODO: enable this
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window_ = glfwCreateWindow(width_, height_, "Vulkan Window", nullptr, nullptr);
    if (!window_) {
        std::cerr << "Cannot create glfw window!" << std::endl;
        GLFW_Terminate();
        exit(EXIT_FAILURE);
    }

    VkResult result = GLFW_CreateWindowSurface(instance_->vkInstance(), window_, nullptr, &surface_);
    if (result != VK_SUCCESS) {
        std::cerr << "Cannot create vulkan window surface!" << std::endl;
        GLFW_Terminate();
        exit(EXIT_FAILURE);
    }
}

vk::Window::Window(std::shared_ptr<Instance> instance, GLFWwindow *window_) : instance_(instance), window_(window_) {
    GLFW_GetWindowSize(window_, reinterpret_cast<int *>(&width_), reinterpret_cast<int *>(&height_));
    VkResult result = GLFW_CreateWindowSurface(instance_->vkInstance(), window_, nullptr, &surface_);
    if (result != VK_SUCCESS) {
        std::cerr << "Cannot create vulkan window surface!" << std::endl;
        GLFW_Terminate();
        exit(EXIT_FAILURE);
    }
}

vk::Window::~Window() {
    vkDestroySurfaceKHR(instance_->vkInstance(), surface_, nullptr);

#ifdef DEBUG
    std::cout << "[Window] window deconstructed" << std::endl;
#endif
}

uint32_t vk::Window::width() {
    return width_;
}

uint32_t vk::Window::height() {
    return height_;
}

GLFWwindow *vk::Window::window() {
    return window_;
}

VkSurfaceKHR &vk::Window::vkSurface() {
    return surface_;
}
