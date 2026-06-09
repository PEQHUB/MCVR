#pragma once

// Ownership: Caller (unique ownership via RAII, movable).
// Thread: Create on main thread. Module handle is immutable after creation.
// Dependencies: VkDevice.

#include "vk2_result.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::vk2 {

class ShaderModule {
public:
    ~ShaderModule();

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;
    ShaderModule(ShaderModule&& other) noexcept;
    ShaderModule& operator=(ShaderModule&& other) noexcept;

    // Load SPIR-V from file path.
    static Result<ShaderModule> fromFile(VkDevice device, const std::string& path);

    // Load SPIR-V from memory.
    static Result<ShaderModule> fromMemory(VkDevice device, const uint32_t* code, size_t sizeBytes);

    ShaderModule() = default;

    VkShaderModule handle() const { return module_; }

private:

    VkDevice device_ = VK_NULL_HANDLE;
    VkShaderModule module_ = VK_NULL_HANDLE;

    void destroy();
};

} // namespace engine::vk2
