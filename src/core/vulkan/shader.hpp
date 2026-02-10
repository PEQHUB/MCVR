#pragma once

#include "core/all_extern.hpp"

#include <string>

namespace vk {
class Device;

class Shader : public SharedObject<Shader> {
  public:
    Shader(std::shared_ptr<Device> device, std::string filePath);
    ~Shader();

    VkShaderModule &vkShaderModule();

  private:
    std::shared_ptr<Device> device_;

    std::string filePath_;
    VkShaderModule shader_;
};
}; // namespace vk