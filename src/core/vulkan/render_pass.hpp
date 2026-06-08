#pragma once

#include "core/all_extern.hpp"

#include <vector>

namespace vk {
class Device;

class RenderPassBuilder;

class RenderPass : public SharedObject<RenderPass> {
    friend RenderPassBuilder;

  public:
    RenderPass(std::shared_ptr<Device> device, VkRenderPass renderpass);
    ~RenderPass();

    VkRenderPass &vkRenderPass();

  private:
    std::shared_ptr<Device> device_;

    VkRenderPass renderPass_ = VK_NULL_HANDLE;
};

class RenderPassBuilder {
  public:
    struct AttachmentDescriptionBuilder {
        RenderPassBuilder &parent;
        std::vector<VkAttachmentDescription> attachmentDescriptions;

        AttachmentDescriptionBuilder(RenderPassBuilder &parent);

        AttachmentDescriptionBuilder &defineAttachmentDescription(VkAttachmentDescription attachmentDescription);

        RenderPassBuilder &endAttachmentDescription();
    };

    struct AttachmentReferenceBuilder {
        RenderPassBuilder &parent;
        std::vector<VkAttachmentReference> attachmentReferences;

        AttachmentReferenceBuilder(RenderPassBuilder &parent);

        AttachmentReferenceBuilder &defineAttachmentReference(VkAttachmentReference attachmentReference);

        RenderPassBuilder &endAttachmentReference();
    };

    struct SubpassDescription {
        VkSubpassDescriptionFlags flags;
        VkPipelineBindPoint pipelineBindPoint;
        std::vector<uint32_t> inputAttachmentIndices;
        std::vector<uint32_t> colorAttachmentIndices;
        uint32_t resolveAttachmentIndex = static_cast<uint32_t>(-1);
        uint32_t depthStencilAttachmentIndex = static_cast<uint32_t>(-1);
        std::vector<uint32_t> preserveAttachmentIndices;
    };

    struct SubpassDescriptionBuilder {
        RenderPassBuilder &parent;
        std::vector<VkSubpassDescription> subpassDescriptions;
        std::vector<std::vector<VkAttachmentReference>> inputAttachmentReferences;
        std::vector<std::vector<VkAttachmentReference>> colorAttachmentReferences;
        std::vector<std::vector<uint32_t>> preserveAttachmentIndices;

        SubpassDescriptionBuilder(RenderPassBuilder &parent);

        SubpassDescriptionBuilder &defineSubpassDescription(SubpassDescription subpassDescription);

        RenderPassBuilder &endSubpassDescription();
    };

    struct SubpassDependencyBuilder {
        RenderPassBuilder &parent;
        std::vector<VkSubpassDependency> subpassDependencies;

        SubpassDependencyBuilder(RenderPassBuilder &parent);

        SubpassDependencyBuilder &defineSubpassDependency(VkSubpassDependency subpassDependency);

        RenderPassBuilder &endSubpassDependency();
    };

  public:
    RenderPassBuilder();

    AttachmentDescriptionBuilder &beginAttachmentDescription();
    AttachmentReferenceBuilder &beginAttachmentReference();
    SubpassDescriptionBuilder &beginSubpassDescription();
    SubpassDependencyBuilder &beginSubpassDependency();
    std::shared_ptr<RenderPass> build(std::shared_ptr<Device> device);

  private:
    AttachmentDescriptionBuilder attachmentDescriptionBuilder_;
    AttachmentReferenceBuilder attachmentReferenceBuilder_;
    SubpassDescriptionBuilder subpassDescriptionBuilder_;
    SubpassDependencyBuilder subpassDependencyBuilder_;
};
}; // namespace vk