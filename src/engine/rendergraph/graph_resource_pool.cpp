#include "graph_resource_pool.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "frame/resource_gc.hpp"
#include "diagnostics/log.hpp"

namespace engine {

vk2::Result<void> GraphResourcePool::allocate(vk2::DeviceService& device,
                                              const CompiledGraph& graph,
                                              uint32_t renderWidth,
                                              uint32_t renderHeight) {
    renderWidth_ = renderWidth;
    renderHeight_ = renderHeight;

    uint32_t count = graph.resourceCount();
    images_.clear();
    images_.reserve(count);
    imageHandles_.resize(count);
    imageViews_.resize(count);

    for (uint32_t i = 0; i < count; ++i) {
        const auto& desc = graph.resources[i];

        uint32_t w = desc.fixedWidth  ? desc.fixedWidth  : static_cast<uint32_t>(renderWidth  * desc.widthScale);
        uint32_t h = desc.fixedHeight ? desc.fixedHeight : static_cast<uint32_t>(renderHeight * desc.heightScale);

        VkImageUsageFlags usage = deriveUsage(graph, i) | desc.extraUsage;

        vk2::Image::Desc imgDesc{};
        imgDesc.width = w;
        imgDesc.height = h;
        imgDesc.format = desc.format;
        imgDesc.usage = usage;

        auto result = vk2::Image::create(device.device(), device.vma(), imgDesc);
        if (!result) {
            return vk2::makeError(result.error().vkResult,
                                  "GraphResourcePool: failed to allocate '" + desc.name + "'");
        }

        auto& img = images_.emplace_back(std::move(result.value()));
        imageHandles_[i] = img.handle();
        imageViews_[i] = img.defaultView();
    }

    log::info("graph-pool", "Allocated " + std::to_string(count) + " resources at " +
              std::to_string(renderWidth) + "x" + std::to_string(renderHeight));
    return {};
}

vk2::Result<void> GraphResourcePool::recreate(vk2::DeviceService& device,
                                              const CompiledGraph& graph,
                                              uint32_t renderWidth,
                                              uint32_t renderHeight) {
    if (renderWidth == renderWidth_ && renderHeight == renderHeight_) return {};
    releaseImmediate();
    return allocate(device, graph, renderWidth, renderHeight);
}

void GraphResourcePool::release(ResourceGC& gc) {
    // Move entire image vector into a shared_ptr so the GC destructor releases them
    auto imgs = std::make_shared<std::vector<vk2::Image>>(std::move(images_));
    gc.defer([imgs]() { /* shared_ptr destructor destroys images */ });
    images_.clear();
    imageHandles_.clear();
    imageViews_.clear();
    renderWidth_ = 0;
    renderHeight_ = 0;
}

void GraphResourcePool::releaseImmediate() {
    images_.clear();
    imageHandles_.clear();
    imageViews_.clear();
    renderWidth_ = 0;
    renderHeight_ = 0;
}

ResolvedResources GraphResourcePool::resolved() const {
    ResolvedResources r{};
    r.images = const_cast<VkImage*>(imageHandles_.data());
    r.views = const_cast<VkImageView*>(imageViews_.data());
    r.count = static_cast<uint32_t>(images_.size());
    return r;
}

VkImageUsageFlags GraphResourcePool::deriveUsage(const CompiledGraph& graph, uint32_t resourceIndex) {
    // Base usage: all graph resources need transfer for clear/blit and sampling
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                            | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                            | VK_IMAGE_USAGE_SAMPLED_BIT;

    for (const auto& pass : graph.passes) {
        // If this resource is an output of any pass, it needs storage or color attachment
        for (const auto& out : pass.resources.outputs) {
            if (out.index == resourceIndex) {
                usage |= VK_IMAGE_USAGE_STORAGE_BIT;
                usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            }
        }
        // Inputs need sampled (already included) and storage for compute reads
        for (const auto& in : pass.resources.inputs) {
            if (in.index == resourceIndex) {
                usage |= VK_IMAGE_USAGE_STORAGE_BIT;
            }
        }
    }

    return usage;
}

} // namespace engine
