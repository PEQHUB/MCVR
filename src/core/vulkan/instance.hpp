#pragma once

#include "core/all_extern.hpp"

namespace vk {
class Instance : public SharedObject<Instance> {
  public:
    static void setForceValidation(bool enable);

    Instance();
    ~Instance();

    VkInstance &vkInstance();

  private:
    VkInstance instance_;
    // VkDebugReportCallbackEXT callback_ = VK_NULL_HANDLE;
};
} // namespace vk