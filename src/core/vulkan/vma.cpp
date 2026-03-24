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

vk::VMA::VmaStatsSnapshot vk::VMA::getStats() const {
    VmaStatsSnapshot snapshot{};

    VmaTotalStatistics stats{};
    vmaCalculateStatistics(allocator_, &stats);
    snapshot.totalAllocBytes = stats.total.statistics.blockBytes;
    snapshot.totalUsedBytes = stats.total.statistics.allocationBytes;
    snapshot.allocationCount = stats.total.statistics.allocationCount;
    snapshot.blockCount = stats.total.statistics.blockCount;

    VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
    vmaGetHeapBudgets(allocator_, budgets);
    for (uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; ++i) {
        snapshot.budgetBytes += budgets[i].budget;
        snapshot.budgetUsageBytes += budgets[i].usage;
    }

    return snapshot;
}
