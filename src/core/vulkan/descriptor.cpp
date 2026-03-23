#include "core/vulkan/descriptor.hpp"

#include "core/vulkan/as.hpp"
#include "core/vulkan/buffer.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/image.hpp"

#include <iostream>
#include <map>

std::ostream &descriptorTableCout() {
    return std::cout << "[DescriptorTable] ";
}

std::ostream &descriptorTableCerr() {
    return std::cerr << "[DescriptorTable] ";
}

vk::DescriptorTable::DescriptorTable(std::shared_ptr<Device> device,
                                     VkDescriptorPool descriptorPool,
                                     std::vector<VkDescriptorSetLayout> tableLayout,
                                     std::vector<VkDescriptorSet> table,
                                     std::vector<std::vector<VkDescriptorType>> tableTypes,
                                     std::vector<VkPushConstantRange> pushConstantRanges)
    : device_(device),
      descriptorPool_(descriptorPool),
      tableLayout_(tableLayout),
      table_(table),
      tableTypes_(tableTypes),
      pushConstantRanges_(pushConstantRanges) {
    VkPipelineLayoutCreateInfo layoutCreateInfo = {};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCreateInfo.setLayoutCount = tableLayout_.size();
    layoutCreateInfo.pSetLayouts = tableLayout_.data();
    layoutCreateInfo.pushConstantRangeCount = pushConstantRanges_.size();
    layoutCreateInfo.pPushConstantRanges = pushConstantRanges_.data();

    if (vkCreatePipelineLayout(device->vkDevice(), &layoutCreateInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
        descriptorTableCerr() << "failed to create pipeline layout" << std::endl;
        exit(EXIT_FAILURE);
    } else {
#ifdef DEBUG
        descriptorTableCout() << "created pipeline layout" << std::endl;
#endif
    }
}

std::shared_ptr<vk::DescriptorTable> vk::DescriptorTable::bindImage(
    std::shared_ptr<Image> image, VkImageLayout layout, uint32_t set, uint32_t binding, uint32_t viewIndex) {
    VkDescriptorImageInfo descriptorImageInfo{};
    descriptorImageInfo.imageView = image->vkImageView(viewIndex);
    descriptorImageInfo.imageLayout = layout;

    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = table_[set];
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = tableTypes_[set][binding];
    writeDescriptorSet.pImageInfo = &descriptorImageInfo;
    writeDescriptorSet.dstBinding = binding;

    vkUpdateDescriptorSets(device_->vkDevice(), 1, &writeDescriptorSet, 0, nullptr);

    return shared_from_this();
}

std::shared_ptr<vk::DescriptorTable> vk::DescriptorTable::bindSamplerImage(std::shared_ptr<Sampler> sampler,
                                                                           std::shared_ptr<Image> image,
                                                                           VkImageLayout layout,
                                                                           uint32_t set,
                                                                           uint32_t binding,
                                                                           uint32_t index,
                                                                           uint32_t viewIndex) {
    VkDescriptorImageInfo descriptorImageInfo{};
    descriptorImageInfo.sampler = sampler->vkSamper();
    descriptorImageInfo.imageView = image->vkImageView(viewIndex);
    descriptorImageInfo.imageLayout = layout;

    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = table_[set];
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = tableTypes_[set][binding];
    writeDescriptorSet.pImageInfo = &descriptorImageInfo;
    writeDescriptorSet.dstBinding = binding;
    writeDescriptorSet.dstArrayElement = index;

    vkUpdateDescriptorSets(device_->vkDevice(), 1, &writeDescriptorSet, 0, nullptr);

    return shared_from_this();
}

std::shared_ptr<vk::DescriptorTable> vk::DescriptorTable::bindImageForShader(std::shared_ptr<Image> image,
                                                                             uint32_t set,
                                                                             uint32_t binding,
                                                                             uint32_t viewIndex) {
    return bindImage(image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, set, binding, viewIndex);
}

std::shared_ptr<vk::DescriptorTable> vk::DescriptorTable::bindSamplerImageForShader(std::shared_ptr<Sampler> sampler,
                                                                                    std::shared_ptr<Image> image,
                                                                                    uint32_t set,
                                                                                    uint32_t binding,
                                                                                    uint32_t viewIndex) {
    return bindSamplerImage(sampler, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, set, binding, 0, viewIndex);
}

std::shared_ptr<vk::DescriptorTable>
vk::DescriptorTable::bindBuffer(std::shared_ptr<Buffer> buffer, uint32_t set, uint32_t binding) {
    return bindBuffer(buffer, set, binding, 0);
}

std::shared_ptr<vk::DescriptorTable>
vk::DescriptorTable::bindBuffer(std::shared_ptr<Buffer> buffer, uint32_t set, uint32_t binding, uint32_t index) {
    VkDescriptorBufferInfo descriptorBufferInfo{};
    descriptorBufferInfo.buffer = buffer->vkBuffer();
    descriptorBufferInfo.offset = 0;
    descriptorBufferInfo.range = buffer->size();

    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = table_[set];
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = tableTypes_[set][binding];
    writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
    writeDescriptorSet.dstBinding = binding;
    writeDescriptorSet.dstArrayElement = index;

    vkUpdateDescriptorSets(device_->vkDevice(), 1, &writeDescriptorSet, 0, nullptr);

    return shared_from_this();
}

std::shared_ptr<vk::DescriptorTable>
vk::DescriptorTable::bindBuffers(std::vector<std::shared_ptr<Buffer>> buffers, uint32_t set, uint32_t binding) {
    std::vector<VkDescriptorBufferInfo> descriptorBufferInfos;
    for (int i = 0; i < buffers.size(); i++) {
        VkDescriptorBufferInfo descriptorBufferInfo{};
        descriptorBufferInfo.buffer = buffers[i]->vkBuffer();
        descriptorBufferInfo.offset = 0;
        descriptorBufferInfo.range = buffers[i]->size();
        descriptorBufferInfos.push_back(descriptorBufferInfo);
    }

    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = table_[set];
    writeDescriptorSet.descriptorCount = buffers.size();
    writeDescriptorSet.descriptorType = tableTypes_[set][binding];
    writeDescriptorSet.pBufferInfo = descriptorBufferInfos.data();
    writeDescriptorSet.dstBinding = binding;

    vkUpdateDescriptorSets(device_->vkDevice(), 1, &writeDescriptorSet, 0, nullptr);

    return shared_from_this();
}

std::shared_ptr<vk::DescriptorTable>
vk::DescriptorTable::bindAS(std::shared_ptr<TLAS> tlas, uint32_t set, uint32_t binding) {
    VkWriteDescriptorSetAccelerationStructureKHR asWriteInfo{};
    asWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWriteInfo.accelerationStructureCount = 1;
    asWriteInfo.pAccelerationStructures = &tlas->tlas();

    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.pNext = &asWriteInfo;
    writeDescriptorSet.dstSet = table_[set];
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = tableTypes_[set][binding];
    writeDescriptorSet.dstBinding = binding;

    vkUpdateDescriptorSets(device_->vkDevice(), 1, &writeDescriptorSet, 0, nullptr);

    return shared_from_this();
}

uint32_t vk::DescriptorTable::setCount() {
    return table_.size();
}

std::vector<VkDescriptorSet> &vk::DescriptorTable::descriptorSet() {
    return table_;
}

std::vector<VkDescriptorSetLayout> &vk::DescriptorTable::descriptorSetLayout() {
    return tableLayout_;
}

VkPipelineLayout &vk::DescriptorTable::vkPipelineLayout() {
    return pipelineLayout_;
}

vk::DescriptorTableBuilder::DescriptorLayoutSetBindingBuilder::DescriptorLayoutSetBindingBuilder(
    vk::DescriptorTableBuilder::DescriptorLayoutSetBuilder &parent)
    : parent(parent), bindings() {}

vk::DescriptorTableBuilder::DescriptorLayoutSetBindingBuilder &
vk::DescriptorTableBuilder::DescriptorLayoutSetBindingBuilder::defineDescriptorLayoutSetBinding(
    VkDescriptorSetLayoutBinding binding) {
    bindings.push_back(binding);
    types.push_back(binding.descriptorType);
    return *this;
}

vk::DescriptorTable::~DescriptorTable() {
    for (VkDescriptorSetLayout layout : tableLayout_) {
        vkDestroyDescriptorSetLayout(device_->vkDevice(), layout, nullptr);
    }
    vkDestroyPipelineLayout(device_->vkDevice(), pipelineLayout_, nullptr);
    vkDestroyDescriptorPool(device_->vkDevice(), descriptorPool_, nullptr);

#ifdef DEBUG
    descriptorTableCout() << "descriptor set deconstructed" << std::endl;
#endif
}

vk::DescriptorTableBuilder::DescriptorLayoutSetBuilder &
vk::DescriptorTableBuilder::DescriptorLayoutSetBindingBuilder::endDescriptorLayoutSetBinding() {
    return parent;
}

vk::DescriptorTableBuilder::DescriptorLayoutSetBuilder::DescriptorLayoutSetBuilder(DescriptorTableBuilder &parent)
    : parent(parent), setBindingBuilders() {}

vk::DescriptorTableBuilder::DescriptorLayoutSetBindingBuilder &
vk::DescriptorTableBuilder::DescriptorLayoutSetBuilder::beginDescriptorLayoutSetBinding() {
    return setBindingBuilders.emplace_back(*this);
}

vk::DescriptorTableBuilder &vk::DescriptorTableBuilder::DescriptorLayoutSetBuilder::endDescriptorLayoutSet() {
    return parent;
}

vk::DescriptorTableBuilder::DescriptorTableBuilder() : setBuilders(*this) {}

vk::DescriptorTableBuilder::DescriptorLayoutSetBuilder &vk::DescriptorTableBuilder::beginDescriptorLayoutSet() {
    return setBuilders;
}

vk::DescriptorTableBuilder &vk::DescriptorTableBuilder::definePushConstant(VkPushConstantRange pushConstantRange) {
    pushConstantRanges.push_back(pushConstantRange);
    return *this;
}

std::shared_ptr<vk::DescriptorTable> vk::DescriptorTableBuilder::build(std::shared_ptr<Device> device) {
    // create layout for each set
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    std::map<VkDescriptorType, uint32_t> descriptorTypeCount;

    for (vk::DescriptorTableBuilder::DescriptorLayoutSetBindingBuilder &setBindingBuilder :
         setBuilders.setBindingBuilders) {
        VkDescriptorSetLayout &descriptorSetLayout = descriptorSetLayouts.emplace_back();

        std::vector<VkDescriptorBindingFlags> bindingFlags(static_cast<uint32_t>(setBindingBuilder.bindings.size()),
                                                           VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                                               VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
                                                               VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

        VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
        bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        bindingFlagsInfo.bindingCount = setBindingBuilder.bindings.size();
        bindingFlagsInfo.pBindingFlags = bindingFlags.data();

        VkDescriptorSetLayoutCreateInfo descriptorLayoutCreateInfo = {};
        descriptorLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorLayoutCreateInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        descriptorLayoutCreateInfo.pNext = &bindingFlagsInfo;
        descriptorLayoutCreateInfo.bindingCount = setBindingBuilder.bindings.size();
        descriptorLayoutCreateInfo.pBindings = setBindingBuilder.bindings.data();

        if (vkCreateDescriptorSetLayout(device->vkDevice(), &descriptorLayoutCreateInfo, nullptr,
                                        &descriptorSetLayout) != VK_SUCCESS) {
            descriptorTableCerr() << "failed to create descriptor layout" << std::endl;
            exit(EXIT_FAILURE);
        } else {
#ifdef DEBUG
            descriptorTableCout() << "created descriptor layout" << std::endl;
#endif
        }

        for (VkDescriptorSetLayoutBinding &binding : setBindingBuilder.bindings) {
            auto it = descriptorTypeCount.find(binding.descriptorType);
            if (it != descriptorTypeCount.end()) {
                it->second += binding.descriptorCount;
            } else {
                descriptorTypeCount.emplace(binding.descriptorType, binding.descriptorCount);
            }
        }
    }

    // create descriptor pool according to the bindings
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto &[type, cnt] : descriptorTypeCount) {
        poolSizes.push_back({
            .type = type,
            .descriptorCount = cnt,
        });
    }

    VkDescriptorPoolCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    createInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    createInfo.poolSizeCount = poolSizes.size();
    createInfo.pPoolSizes = poolSizes.data();
    createInfo.maxSets = descriptorSetLayouts.size();

    if (vkCreateDescriptorPool(device->vkDevice(), &createInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        descriptorTableCerr() << "failed to create descriptor pool" << std::endl;
        exit(EXIT_FAILURE);
    } else {
#ifdef DEBUG
        descriptorTableCout() << "created descriptor pool" << std::endl;
#endif
    }

    // alloc descriptors from descriptor pool
    std::vector<VkDescriptorSet> descriptorSets(descriptorSetLayouts.size());
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = descriptorSetLayouts.size();
    allocInfo.pSetLayouts = descriptorSetLayouts.data();

    if (vkAllocateDescriptorSets(device->vkDevice(), &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        descriptorTableCerr() << "failed to create descriptor set" << std::endl;
        exit(1);
    } else {
#ifdef DEBUG
        descriptorTableCout() << "created descriptor set" << std::endl;
#endif
    }

    std::vector<std::vector<VkDescriptorType>> descriptorTypes;
    for (int i = 0; i < setBuilders.setBindingBuilders.size(); i++) {
        const auto &bindings = setBuilders.setBindingBuilders[i].bindings;
        uint32_t maxBinding = 0;
        for (const auto &binding : bindings) {
            if (binding.binding > maxBinding) { maxBinding = binding.binding; }
        }
        std::vector<VkDescriptorType> &types = descriptorTypes.emplace_back();
        types.resize(maxBinding + 1, VK_DESCRIPTOR_TYPE_MAX_ENUM);
        for (const auto &binding : bindings) { types[binding.binding] = binding.descriptorType; }
    }

    return std::make_shared<vk::DescriptorTable>(device, descriptorPool, descriptorSetLayouts, descriptorSets,
                                                 descriptorTypes, pushConstantRanges);
}
