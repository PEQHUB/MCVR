#pragma once

#include "core/all_extern.hpp"

namespace vk {
class Instance : public SharedObject<Instance> {
  public:
    Instance();
    ~Instance();

    VkInstance &vkInstance();

  private:
    VkInstance instance_;
    // VkDebugReportCallbackEXT callback_ = VK_NULL_HANDLE;
};
} // namespace vk