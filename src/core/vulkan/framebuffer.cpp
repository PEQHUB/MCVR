#include "core/vulkan/framebuffer.hpp"

#include "core/vulkan/device.hpp"
#include "core/vulkan/image.hpp"
#include "core/vulkan/render_pass.hpp"

#include <iostream>

std::ostream &framebufferCout() {
    return std::cout << "[Framebuffer] ";
}

std::ostream &framebufferCerr() {
    return std::cerr << "[Framebuffer] ";
}

vk::Framebuffer::Framebuffer(std::shared_ptr<Device> device, VkFramebuffer framebuffer)
    : device_(device), framebuffer_(framebuffer) {
#ifdef DEBUG
    framebufferCout() << "framebuffer created" << std::endl;
#endif
}

vk::Framebuffer::~Framebuffer() {
    vkDestroyFramebuffer(device_->vkDevice(), framebuffer_, nullptr);

#ifdef DEBUG
    framebufferCout() << "framebuffer deconstructed" << std::endl;
#endif
}

VkFramebuffer &vk::Framebuffer::vkFramebuffer() {
    return framebuffer_;
}

vk::FramebufferBuilder::FramebufferAttachmentsBuilder::FramebufferAttachmentsBuilder(FramebufferBuilder &parent)
    : parent(parent) {}

vk::FramebufferBuilder::FramebufferAttachmentsBuilder &
vk::FramebufferBuilder::FramebufferAttachmentsBuilder::defineAttachment(std::shared_ptr<Image> image, int viewIndex) {
    attachments.push_back(image->vkImageView(viewIndex));

    if (width == -1) {
        width = image->width();
    } else {
        if (width != image->width()) {
            framebufferCerr() << "width is not consist" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    if (height == -1) {
        height = image->height();
    } else {
        if (height != image->height()) {
            framebufferCerr() << "height is not consist" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return *this;
}

vk::FramebufferBuilder &vk::FramebufferBuilder::FramebufferAttachmentsBuilder::endAttachment() {
    return parent;
}

vk::FramebufferBuilder::FramebufferBuilder() : framebufferAttachmentsBuilder_(*this) {}

vk::FramebufferBuilder::FramebufferAttachmentsBuilder &vk::FramebufferBuilder::beginAttachment() {
    return framebufferAttachmentsBuilder_;
}

std::shared_ptr<vk::Framebuffer> vk::FramebufferBuilder::build(std::shared_ptr<Device> device,
                                                               std::shared_ptr<RenderPass> renderPass) {
    VkFramebufferCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    createInfo.renderPass = renderPass->vkRenderPass();
    createInfo.attachmentCount = framebufferAttachmentsBuilder_.attachments.size();
    createInfo.pAttachments = framebufferAttachmentsBuilder_.attachments.data();
    createInfo.width = framebufferAttachmentsBuilder_.width;
    createInfo.height = framebufferAttachmentsBuilder_.height;
    createInfo.layers = 1;

    VkFramebuffer framebuffer;
    if (vkCreateFramebuffer(device->vkDevice(), &createInfo, nullptr, &framebuffer) != VK_SUCCESS) {
        framebufferCerr() << "failed to create framebuffer" << std::endl;
        exit(EXIT_FAILURE);
    }

    return std::make_shared<Framebuffer>(device, framebuffer);
}

std::shared_ptr<vk::Framebuffer> vk::FramebufferBuilder::build(std::shared_ptr<Device> device,
                                                               std::shared_ptr<RenderPass> renderPass,
                                                               uint32_t width,
                                                               uint32_t height) {
    VkFramebufferCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    createInfo.renderPass = renderPass->vkRenderPass();
    createInfo.attachmentCount = 0;
    createInfo.pAttachments = nullptr;
    createInfo.width = width;
    createInfo.height = height;
    createInfo.layers = 1;

    VkFramebuffer framebuffer;
    if (vkCreateFramebuffer(device->vkDevice(), &createInfo, nullptr, &framebuffer) != VK_SUCCESS) {
        framebufferCerr() << "failed to create framebuffer" << std::endl;
        exit(EXIT_FAILURE);
    }

    return std::make_shared<Framebuffer>(device, framebuffer);
}