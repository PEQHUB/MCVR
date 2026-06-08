#pragma once

#include "core/all_extern.hpp"

namespace vk {
class Instance;

class Window : public SharedObject<Window> {
    friend class SharedObject<Window>;

  public:
    Window(std::shared_ptr<Instance> instance, uint32_t width, uint32_t height);
    Window(std::shared_ptr<Instance> instance, GLFWwindow *window_);
    ~Window();

    uint32_t width();
    uint32_t height();
    GLFWwindow *window();
    VkSurfaceKHR &vkSurface();

    static bool framebufferResized;
    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

  private:
    std::shared_ptr<Instance> instance_;

    uint32_t width_;
    uint32_t height_;

    GLFWwindow *window_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
};
} // namespace vk