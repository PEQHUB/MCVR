#pragma once

#include "core/all_extern.hpp"

#include <vector>

namespace vk {
class Device;
class DescriptorTableBuilder;
class Buffer;
class Image;
class Sampler;
class TLAS;

class DescriptorPool {};

class DescriptorTable : public SharedObject<DescriptorTable> {
    friend DescriptorTableBuilder;

  public:
    DescriptorTable(std::shared_ptr<Device> device,
                    VkDescriptorPool descriptorPool,
                    std::vector<VkDescriptorSetLayout> tableLayout,
                    std::vector<VkDescriptorSet> table,
                    std::vector<std::vector<VkDescriptorType>> tableTypes,
                    std::vector<VkPushConstantRange> pushConstantRanges);
    ~DescriptorTable();

    uint32_t setCount();
    std::vector<VkDescriptorSet> &descriptorSet();
    std::vector<VkDescriptorSetLayout> &descriptorSetLayout();
    VkPipelineLayout &vkPipelineLayout();

    std::shared_ptr<DescriptorTable> bindImage(
        std::shared_ptr<Image> image, VkImageLayout layout, uint32_t set, uint32_t binding, uint32_t viewIndex = 0);
    std::shared_ptr<DescriptorTable> bindSamplerImage(std::shared_ptr<Sampler> sampler,
                                                      std::shared_ptr<Image> image,
                                                      VkImageLayout layout,
                                                      uint32_t set,
                                                      uint32_t binding,
                                                      uint32_t index,
                                                      uint32_t viewIndex = 0);
    std::shared_ptr<DescriptorTable>
    bindImageForShader(std::shared_ptr<Image> image, uint32_t set, uint32_t binding, uint32_t viewIndex = 0);
    std::shared_ptr<DescriptorTable> bindSamplerImageForShader(std::shared_ptr<Sampler> sampler,
                                                               std::shared_ptr<Image> image,
                                                               uint32_t set,
                                                               uint32_t binding,
                                                               uint32_t viewIndex = 0);

    std::shared_ptr<DescriptorTable> bindBuffer(std::shared_ptr<Buffer> buffer, uint32_t set, uint32_t binding);
    std::shared_ptr<DescriptorTable>
    bindBuffer(std::shared_ptr<Buffer> buffer, uint32_t set, uint32_t binding, uint32_t index);
    std::shared_ptr<DescriptorTable>
    bindBuffers(std::vector<std::shared_ptr<Buffer>> buffers, uint32_t set, uint32_t binding);

    std::shared_ptr<DescriptorTable> bindAS(std::shared_ptr<TLAS> buffer, uint32_t set, uint32_t binding);

  private:
    std::shared_ptr<Device> device_;

    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayout> tableLayout_;
    std::vector<VkDescriptorSet> table_;
    std::vector<std::vector<VkDescriptorType>> tableTypes_;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    std::vector<VkPushConstantRange> pushConstantRanges_;
};

class DescriptorTableBuilder {
    friend DescriptorTable;

  public:
    struct DescriptorLayoutSetBuilder;

    struct DescriptorLayoutSetBindingBuilder {
        DescriptorLayoutSetBuilder &parent;
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        std::vector<VkDescriptorType> types;

        DescriptorLayoutSetBindingBuilder(DescriptorLayoutSetBuilder &parent);

        DescriptorLayoutSetBindingBuilder &defineDescriptorLayoutSetBinding(VkDescriptorSetLayoutBinding binding);
        DescriptorLayoutSetBuilder &endDescriptorLayoutSetBinding();
    };

    struct DescriptorLayoutSetBuilder {
        DescriptorTableBuilder &parent;
        std::vector<DescriptorLayoutSetBindingBuilder> setBindingBuilders;

        DescriptorLayoutSetBuilder(DescriptorTableBuilder &parent);

        DescriptorLayoutSetBindingBuilder &beginDescriptorLayoutSetBinding();
        DescriptorTableBuilder &endDescriptorLayoutSet();
    };

  public:
    DescriptorTableBuilder();

    DescriptorLayoutSetBuilder &beginDescriptorLayoutSet();
    DescriptorTableBuilder &definePushConstant(VkPushConstantRange pushConstantRange);
    std::shared_ptr<DescriptorTable> build(std::shared_ptr<Device> device);

  private:
    DescriptorLayoutSetBuilder setBuilders;

    std::vector<VkPushConstantRange> pushConstantRanges;
};
}; // namespace vk