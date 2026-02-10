#define VMA_IMPLEMENTATION
#include "core/vulkan/vma.hpp"

#include "core/vulkan/device.hpp"
#include "core/vulkan/instance.hpp"
#include "core/vulkan/physical_device.hpp"

#include <iostream>

std::ostream &vmaTableCout() {
    return std::cout << "[VMA] ";
}

std::ostream &vmaTableCerr() {
    return std::cerr << "[VMA] ";
}

vk::VMA::VMA(std::shared_ptr<Instance> instance,
             std::shared_ptr<PhysicalDevice> physicalDevice,
             std::shared_ptr<Device> device) {
    VmaAllocatorCreateInfo allocatorCreateInfo{};
    allocatorCreateInfo.physicalDevice = physicalDevice->vkPhysicalDevice();
    allocatorCreateInfo.device = device->vkDevice();
    allocatorCreateInfo.instance = instance->vkInstance();
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_4;
    allocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    if (vmaImportVulkanFunctionsFromVolk(&allocatorCreateInfo, &vulkanFunctions_)) {
        vmaTableCerr() << "failed to create vulkan function from volk" << std::endl;
        exit(EXIT_FAILURE);
    } else {
#ifdef DEBUG
        vmaTableCout() << "created vulkan function from volk" << std::endl;
#endif
    }
    allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions_;

    if (vmaCreateAllocator(&allocatorCreateInfo, &allocator_)) {
        vmaTableCerr() << "failed to create VMA" << std::endl;
        exit(EXIT_FAILURE);
    } else {
#ifdef DEBUG
        vmaTableCout() << "created VMA" << std::endl;
#endif
    }
}

vk::VMA::~VMA() {
#ifdef DEBUG
    vmaTableCout() << "VMA deconstructed" << std::endl;
#endif
    vmaDestroyAllocator(allocator_);
}

VmaAllocator &vk::VMA::allocator() {
    return allocator_;
}
