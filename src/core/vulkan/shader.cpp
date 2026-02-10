#include "core/vulkan/shader.hpp"

#include "core/vulkan/device.hpp"

#include <fstream>
#include <iostream>
#include <vector>

std::ostream &shaderCout() {
    return std::cout << "[Shader] ";
}

std::ostream &shaderCerr() {
    return std::cerr << "[Shader] ";
}

vk::Shader::Shader(std::shared_ptr<Device> device, std::string filePath) : device_(device), filePath_(filePath) {
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        shaderCerr() << "Cannot open file: " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<char> fileBytes(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(fileBytes.data(), fileBytes.size());
    file.close();

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = fileBytes.size();
    createInfo.pCode = (uint32_t *)fileBytes.data();

    if (vkCreateShaderModule(device->vkDevice(), &createInfo, nullptr, &shader_) != VK_SUCCESS) {
        shaderCerr() << "failed to create shader module for " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }

#ifdef DEBUG
    shaderCout() << "created shader module for " << filePath << std::endl;
#endif
}

vk::Shader::~Shader() {
    vkDestroyShaderModule(device_->vkDevice(), shader_, nullptr);
}

VkShaderModule &vk::Shader::vkShaderModule() {
    return shader_;
}