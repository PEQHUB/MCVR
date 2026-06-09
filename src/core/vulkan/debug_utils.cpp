#include "core/vulkan/debug_utils.hpp"

#include <atomic>
#include <cstdlib>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {
bool envFlagEnabled(const char* name) {
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return false;
    return env[0] == '1' || env[0] == 't' || env[0] == 'T' ||
           env[0] == 'y' || env[0] == 'Y' || env[0] == 'o' || env[0] == 'O';
}

const bool kEnvForceLabels = envFlagEnabled("RADIANCE_VK_LABELS");
const bool kEnvDetailedLabels = envFlagEnabled("RADIANCE_VK_LABEL_DETAIL");
std::atomic_bool g_enabled{kEnvForceLabels};
}

bool vk::DebugUtils::enabled() {
    return g_enabled.load(std::memory_order_relaxed) || kEnvForceLabels;
}

bool vk::DebugUtils::detailEnabled() {
    return enabled() && kEnvDetailedLabels;
}

void vk::DebugUtils::setEnabled(bool enabled) {
    g_enabled.store(enabled || kEnvForceLabels, std::memory_order_relaxed);
}

void vk::DebugUtils::beginCommandLabel(VkCommandBuffer commandBuffer,
                                       const char* name,
                                       float r,
                                       float g,
                                       float b) {
    if (!enabled() || !vkCmdBeginDebugUtilsLabelEXT || commandBuffer == VK_NULL_HANDLE) return;
    VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = name;
    label.color[0] = r;
    label.color[1] = g;
    label.color[2] = b;
    label.color[3] = 1.0f;
    vkCmdBeginDebugUtilsLabelEXT(commandBuffer, &label);
}

void vk::DebugUtils::endCommandLabel(VkCommandBuffer commandBuffer) {
    if (!enabled() || !vkCmdEndDebugUtilsLabelEXT || commandBuffer == VK_NULL_HANDLE) return;
    vkCmdEndDebugUtilsLabelEXT(commandBuffer);
}

void vk::DebugUtils::beginQueueLabel(VkQueue queue,
                                     const char* name,
                                     float r,
                                     float g,
                                     float b) {
    if (!enabled() || !vkQueueBeginDebugUtilsLabelEXT || queue == VK_NULL_HANDLE) return;
    VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = name;
    label.color[0] = r;
    label.color[1] = g;
    label.color[2] = b;
    label.color[3] = 1.0f;
    vkQueueBeginDebugUtilsLabelEXT(queue, &label);
}

void vk::DebugUtils::endQueueLabel(VkQueue queue) {
    if (!enabled() || !vkQueueEndDebugUtilsLabelEXT || queue == VK_NULL_HANDLE) return;
    vkQueueEndDebugUtilsLabelEXT(queue);
}

void vk::DebugUtils::setCurrentThreadName(std::string_view name) {
    if (name.empty()) return;
#ifdef _WIN32
    std::wstring wideName(name.begin(), name.end());
    SetThreadDescription(GetCurrentThread(), wideName.c_str());
#else
    (void)name;
#endif
}
