#pragma once

#include "core/all_extern.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

namespace vk {
namespace DebugUtils {

bool enabled();
bool detailEnabled();
void setEnabled(bool enabled);

template <typename Handle>
uint64_t objectHandle(Handle handle) {
    if constexpr (std::is_pointer_v<Handle>) {
        return reinterpret_cast<uint64_t>(handle);
    } else {
        return static_cast<uint64_t>(handle);
    }
}

template <typename Handle>
void setObjectName(VkDevice device, VkObjectType type, Handle handle, std::string_view name) {
    if (!enabled() || !vkSetDebugUtilsObjectNameEXT || device == VK_NULL_HANDLE || name.empty()) return;

    std::string stableName{name};
    VkDebugUtilsObjectNameInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    info.objectType = type;
    info.objectHandle = objectHandle(handle);
    info.pObjectName = stableName.c_str();
    vkSetDebugUtilsObjectNameEXT(device, &info);
}

void beginCommandLabel(VkCommandBuffer commandBuffer,
                       const char* name,
                       float r = 0.2f,
                       float g = 0.8f,
                       float b = 0.2f);
void endCommandLabel(VkCommandBuffer commandBuffer);
void beginQueueLabel(VkQueue queue,
                     const char* name,
                     float r = 0.2f,
                     float g = 0.8f,
                     float b = 0.2f);
void endQueueLabel(VkQueue queue);
void setCurrentThreadName(std::string_view name);

class ScopedCommandLabel {
  public:
    ScopedCommandLabel(VkCommandBuffer commandBuffer,
                       const char* name,
                       float r = 0.2f,
                       float g = 0.8f,
                       float b = 0.2f)
        : commandBuffer_(commandBuffer),
          active_(enabled() && commandBuffer != VK_NULL_HANDLE && vkCmdBeginDebugUtilsLabelEXT &&
                  vkCmdEndDebugUtilsLabelEXT) {
        if (active_) beginCommandLabel(commandBuffer_, name, r, g, b);
    }

    ~ScopedCommandLabel() {
        if (active_) endCommandLabel(commandBuffer_);
    }

  private:
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
    bool active_ = false;
};

class ScopedQueueLabel {
  public:
    ScopedQueueLabel(VkQueue queue, const char* name, float r = 0.2f, float g = 0.8f, float b = 0.2f)
        : queue_(queue), active_(enabled() && queue != VK_NULL_HANDLE && vkQueueBeginDebugUtilsLabelEXT &&
                                 vkQueueEndDebugUtilsLabelEXT) {
        if (active_) beginQueueLabel(queue_, name, r, g, b);
    }

    ~ScopedQueueLabel() {
        if (active_) endQueueLabel(queue_);
    }

  private:
    VkQueue queue_ = VK_NULL_HANDLE;
    bool active_ = false;
};

} // namespace DebugUtils
} // namespace vk
