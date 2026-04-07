#include "vk2_shader.hpp"

#include <volk.h>
#include <fstream>

namespace engine::vk2 {

ShaderModule::~ShaderModule() { destroy(); }

ShaderModule::ShaderModule(ShaderModule&& other) noexcept
    : device_(other.device_), module_(other.module_) {
    other.device_ = VK_NULL_HANDLE;
    other.module_ = VK_NULL_HANDLE;
}

ShaderModule& ShaderModule::operator=(ShaderModule&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = other.device_;
        module_ = other.module_;
        other.device_ = VK_NULL_HANDLE;
        other.module_ = VK_NULL_HANDLE;
    }
    return *this;
}

Result<ShaderModule> ShaderModule::fromFile(VkDevice device, const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        return makeError(VK_ERROR_UNKNOWN, "Cannot open shader file: " + path);
    }

    size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> bytes(size);
    file.seekg(0, std::ios::beg);
    file.read(bytes.data(), static_cast<std::streamsize>(size));
    file.close();

    return fromMemory(device, reinterpret_cast<const uint32_t*>(bytes.data()), size);
}

Result<ShaderModule> ShaderModule::fromMemory(VkDevice device, const uint32_t* code, size_t sizeBytes) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = sizeBytes;
    ci.pCode = code;

    VkShaderModule module = VK_NULL_HANDLE;
    VkResult r = vkCreateShaderModule(device, &ci, nullptr, &module);
    if (r != VK_SUCCESS) {
        return makeError(r, "vkCreateShaderModule failed");
    }

    ShaderModule sm;
    sm.device_ = device;
    sm.module_ = module;
    return sm;
}

void ShaderModule::destroy() {
    if (module_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_, module_, nullptr);
    }
    module_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}

} // namespace engine::vk2
