#pragma once

#include "core/all_extern.hpp"

#include <vector>

namespace vk {
class Image;
class RenderPass;
class Device;

class FramebufferBuilder;

class Framebuffer : public SharedObject<Framebuffer> {
    friend FramebufferBuilder;

  public:
    Framebuffer(std::shared_ptr<Device> device, VkFramebuffer framebuffer);
    ~Framebuffer();

    VkFramebuffer &vkFramebuffer();

  private:
    std::shared_ptr<Device> device_;

    VkFramebuffer framebuffer_;
};


class FramebufferBuilder {
  public:
    struct FramebufferAttachmentsBuilder {
        FramebufferBuilder &parent;
        std::vector<VkImageView> attachments;
        uint32_t width = -1;
        uint32_t height = -1;

        FramebufferAttachmentsBuilder(FramebufferBuilder &parent);

        FramebufferAttachmentsBuilder &defineAttachment(std::shared_ptr<Image> image, int viewIndex = 0);

        FramebufferBuilder &endAttachment();
    };

  public:
    FramebufferBuilder();

    FramebufferAttachmentsBuilder &beginAttachment();
    std::shared_ptr<Framebuffer> build(std::shared_ptr<Device> devide, std::shared_ptr<RenderPass> renderPass);
    std::shared_ptr<Framebuffer>
    build(std::shared_ptr<Device> devide, std::shared_ptr<RenderPass> renderPass, uint32_t width, uint32_t height);

  private:
    FramebufferAttachmentsBuilder framebufferAttachmentsBuilder_;
};
}; // namespace vk