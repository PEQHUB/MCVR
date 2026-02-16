#include "core/render/render_framework.hpp"

#include "common/shared.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/entities.hpp"
#include "core/render/hdr_composite_pass.hpp"
#include "core/render/modules/ui_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/renderer.hpp"
#include "core/render/textures.hpp"
#include "core/render/world.hpp"

#include <iostream>
#include <random>

std::ostream &renderFrameworkCout() {
    return std::cout << "[Render Framework] ";
}

std::ostream &renderFrameworkCerr() {
    return std::cerr << "[Render Framework] ";
}

FrameworkContext::FrameworkContext(std::shared_ptr<Framework> framework, uint32_t frameIndex)
    : framework(framework),
      frameIndex(frameIndex),
      instance(framework->instance_),
      window(framework->window_),
      physicalDevice(framework->physicalDevice_),
      device(framework->device_),
      vma(framework->vma_),
      swapchain(framework->swapchain_),
      swapchainImage(framework->swapchain_->swapchainImages()[frameIndex]),
      commandPool(framework->mainCommandPool_),
      commandProcessedSemaphore(framework->commandProcessedSemaphores_[frameIndex]),
      commandFinishedFence(framework->commandFinishedFences_[frameIndex]),
      uploadCommandBuffer(framework->uploadCommandBuffers_[frameIndex]),
      overlayCommandBuffer(framework->overlayCommandBuffers_[frameIndex]),
      worldCommandBuffer(framework->worldCommandBuffers_[frameIndex]),
      fuseCommandBuffer(framework->fuseCommandBuffers_[frameIndex]) {}

FrameworkContext::~FrameworkContext() {
#ifdef DEBUG
    std::cout << "[Context] context deconstructed" << std::endl;
#endif
}

void FrameworkContext::fuseFinal() {
    auto f = framework.lock();

    if (!f->isRunning()) return;

    auto mainQueueIndex = physicalDevice->mainQueueIndex();
    auto pipelineContext = f->pipeline_->acquirePipelineContext(shared_from_this());

    bool hdrOutputActive = Renderer::options.hdrEnabled && swapchain->isHDR();
    bool hasWorldOutput = pipelineContext->worldPipelineContext && pipelineContext->worldPipelineContext->outputImage;
    auto worldOutput = hasWorldOutput ? pipelineContext->worldPipelineContext->outputImage : nullptr;
    auto overlayOutput = pipelineContext->uiModuleContext ? pipelineContext->uiModuleContext->overlayDrawColorImage : nullptr;
    bool canComposite = f->pipeline_->hdrCompositePass() && overlayOutput;

    if (hdrOutputActive && canComposite) {
        // ═══════════ HDR path: composite shader ═══════════
        // HdrCompositePass::record() handles ALL image transitions internally:
        //   world output  → SHADER_READ_ONLY_OPTIMAL → rest layout
        //   overlay       → SHADER_READ_ONLY_OPTIMAL → rest layout
        //   swapchain     → COLOR_ATTACHMENT_OPTIMAL  → PRESENT_SRC_KHR
        f->pipeline_->hdrCompositePass()->record(
            fuseCommandBuffer,
            frameIndex,
            HdrCompositePass::OutputMode::Hdr10,
            Renderer::options.hdrUiBrightnessNits,
            worldOutput,
            overlayOutput,
            swapchainImage,
            mainQueueIndex);

    } else if (!hdrOutputActive && canComposite) {
        // ═══════════ SDR path: composite shader ═══════════
        // Output is sRGB-encoded into the UNORM swapchain (SRGB_NONLINEAR colorspace).
        f->pipeline_->hdrCompositePass()->record(
            fuseCommandBuffer,
            frameIndex,
            HdrCompositePass::OutputMode::Sdr,
            0.0f,
            worldOutput,
            overlayOutput,
            swapchainImage,
            mainQueueIndex);

    } else {
        // ═══════════ SDR fallback: copy overlay → swapchain ═══════════

        if (!overlayOutput) {
            return;
        }

        fuseCommandBuffer->barriersBufferImage(
            {}, {
                    {
                        .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .oldLayout = overlayOutput->imageLayout(),
                        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        .srcQueueFamilyIndex = mainQueueIndex,
                        .dstQueueFamilyIndex = mainQueueIndex,
                        .image = overlayOutput,
                        .subresourceRange = vk::wholeColorSubresourceRange,
                    },
                    {
                        .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .oldLayout = swapchainImage->imageLayout(),
                        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        .srcQueueFamilyIndex = mainQueueIndex,
                        .dstQueueFamilyIndex = mainQueueIndex,
                        .image = swapchainImage,
                        .subresourceRange = vk::wholeColorSubresourceRange,
                    },
                });

        overlayOutput->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        swapchainImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        // Cross-platform: swapchain may be B8G8R8A8 (or other non-RGBA8 layouts)
        // while overlay is R8G8B8A8_SRGB. Use vkCmdBlitImage whenever formats are
        // not directly RGBA-compatible to avoid channel/alpha mismatches.
        VkFormat swapFormat = swapchainImage->vkFormat();
        bool needBlit = (swapFormat != VK_FORMAT_R8G8B8A8_UNORM &&
                         swapFormat != VK_FORMAT_R8G8B8A8_SRGB);

        if (needBlit) {
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {(int32_t)overlayOutput->width(),
                                  (int32_t)overlayOutput->height(), 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {(int32_t)swapchainImage->width(),
                                  (int32_t)swapchainImage->height(), 1};

            vkCmdBlitImage(fuseCommandBuffer->vkCommandBuffer(),
                           overlayOutput->vkImage(),
                           overlayOutput->imageLayout(),
                           swapchainImage->vkImage(),
                           swapchainImage->imageLayout(),
                           1, &blit, VK_FILTER_NEAREST);
        } else {
            // RGBA-compatible swapchain: raw byte copy preserves gamma-encoded values.
            VkImageCopy imageCopy{};
            imageCopy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageCopy.srcSubresource.mipLevel = 0;
            imageCopy.srcSubresource.baseArrayLayer = 0;
            imageCopy.srcSubresource.layerCount = 1;
            imageCopy.srcOffset = {0, 0, 0};
            imageCopy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageCopy.dstSubresource.mipLevel = 0;
            imageCopy.dstSubresource.baseArrayLayer = 0;
            imageCopy.dstSubresource.layerCount = 1;
            imageCopy.dstOffset = {0, 0, 0};
            imageCopy.extent = {overlayOutput->width(),
                                overlayOutput->height(), 1};

            vkCmdCopyImage(fuseCommandBuffer->vkCommandBuffer(),
                           overlayOutput->vkImage(),
                           overlayOutput->imageLayout(), swapchainImage->vkImage(),
                           swapchainImage->imageLayout(), 1, &imageCopy);
        }

        fuseCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = overlayOutput->imageLayout(),
#ifdef USE_AMD
                     .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                     .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = overlayOutput,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                  },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = swapchainImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = swapchainImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 }});

#ifdef USE_AMD
        overlayOutput->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        overlayOutput->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
        swapchainImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }
}

Framework::Framework() {}

void Framework::init(GLFWwindow *window) {
    instance_ = vk::Instance::create();
    window_ = vk::Window::create(instance_, window);
    physicalDevice_ = vk::PhysicalDevice::create(instance_, window_);
    device_ = vk::Device::create(instance_, window_, physicalDevice_);
    vma_ = vk::VMA::create(instance_, physicalDevice_, device_);
    swapchain_ = vk::Swapchain::create(physicalDevice_, device_, window_);
    mainCommandPool_ = vk::CommandPool::create(physicalDevice_, device_);
    asyncCommandPool_ = vk::CommandPool::create(physicalDevice_, device_, physicalDevice_->secondaryQueueIndex());
    gc_ = GarbageCollector::create(shared_from_this());

    uint32_t imageCount = swapchain_->imageCount();

    // create command buffer for each context
    for (int i = 0; i < imageCount; i++) {
        uploadCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
        overlayCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
        worldCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
        fuseCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
    }
    worldAsyncCommandBuffer_ = vk::CommandBuffer::create(device_, asyncCommandPool_);

    for (int i = 0; i < imageCount; i++) { commandFinishedFences_.push_back(vk::Fence::create(device_, true)); }

    for (int i = 0; i < imageCount; i++) { commandProcessedSemaphores_.push_back(vk::Semaphore::create(device_)); }

    for (int i = 0; i < imageCount; i++) { contexts_.push_back(FrameworkContext::create(shared_from_this(), i)); }

    pipeline_ = Pipeline::create(shared_from_this());
}

Framework::~Framework() {
#ifdef DEBUG
    std::cout << "[Framework] framework deconstructed" << std::endl;
#endif
}

void Framework::acquireContext() {
    if (!running_) return;

    std::shared_ptr<FrameworkContext> lastContext;
    if (currentContext_) lastContext = currentContext_;
    VkResult result;

    std::shared_ptr<vk::Semaphore> imageAcquiredSemaphore = acquireSemaphore();
    uint32_t imageIndex;
    result = vkAcquireNextImageKHR(device_->vkDevice(), swapchain_->vkSwapchain(), UINT64_MAX,
                                   imageAcquiredSemaphore->vkSemaphore(), VK_NULL_HANDLE, &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recycleSemaphore(imageAcquiredSemaphore);
        recreate();
        return;
    } else if (result != VK_SUCCESS) {
        std::cerr << "Cannot acquire images from swapchain" << std::endl;
        recycleSemaphore(imageAcquiredSemaphore);
        waitDeviceIdle();
        exit(EXIT_FAILURE);
    }

    std::shared_ptr<vk::Fence> fence = contexts_[imageIndex]->commandFinishedFence;
    result = vkWaitForFences(device_->vkDevice(), 1, &fence->vkFence(), true, UINT64_MAX);
    if (result != VK_SUCCESS) {
        std::cout << "vkWaitForFences failed with error: " << std::dec << result << std::endl;
        waitDeviceIdle();
        exit(EXIT_FAILURE);
    }
    currentContextIndex_ = imageIndex;
    currentContext_ = contexts_[imageIndex];
    indexHistory_.push(imageIndex);
    if (indexHistory_.size() > swapchain_->imageCount()) indexHistory_.pop();

    if (currentContext_->imageAcquiredSemaphore != VK_NULL_HANDLE) {
        recycleSemaphore(currentContext_->imageAcquiredSemaphore);
        currentContext_->imageAcquiredSemaphore = VK_NULL_HANDLE;
    }
    currentContext_->imageAcquiredSemaphore = imageAcquiredSemaphore;

    currentContext_->uploadCommandBuffer->begin();
    currentContext_->worldCommandBuffer->begin();
    currentContext_->overlayCommandBuffer->begin();
    currentContext_->fuseCommandBuffer->begin();

    auto pipelineContext = pipeline_->acquirePipelineContext(currentContext_);
    std::shared_ptr<UIModuleContext> lastUIContext =
        lastContext == nullptr ? nullptr : pipeline_->acquirePipelineContext(lastContext)->uiModuleContext;

    pipelineContext->uiModuleContext->begin(lastUIContext);

    gc_->clear();
    Renderer::instance().buffers()->resetFrame();
    Renderer::instance().textures()->resetFrame();
    Renderer::instance().world()->resetFrame();
    Renderer::instance().world()->chunks()->resetFrame();
    Renderer::instance().world()->entities()->resetFrame();

    static int frames = 0;
    static auto lastTime = std::chrono::high_resolution_clock::now();

    frames++;
    auto currentTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = currentTime - lastTime;

    if (elapsed.count() >= 1.0) {
        std::stringstream ss;
        ss << "FPS: " << frames;

        // GLFW_SetWindowTitle(window_->window(), ss.str().c_str());

        frames = 0;
        lastTime = currentTime;
    }
}

void Framework::submitCommand() {
    if (!running_) return;

    Renderer::instance().framework()->safeAcquireCurrentContext(); // ensure context is non nullptr

    Renderer::instance().textures()->performQueuedUpload();
    Renderer::instance().buffers()->performQueuedUpload();
    Renderer::instance().buffers()->buildAndUploadOverlayUniformBuffer();

    auto pipelineContext = pipeline_->acquirePipelineContext(currentContext_);
    if (Renderer::instance().world()->shouldRender()) pipelineContext->worldPipelineContext->render();
    pipelineContext->uiModuleContext->end();

    currentContext_->fuseFinal();

    currentContext_->uploadCommandBuffer->end();
    currentContext_->worldCommandBuffer->end();
    currentContext_->overlayCommandBuffer->end();
    currentContext_->fuseCommandBuffer->end();

    std::vector<VkSemaphore> waitSemaphores = {currentContext_->imageAcquiredSemaphore->vkSemaphore()};
    std::vector<VkPipelineStageFlags> waitStageMasks = {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
    std::vector<VkSemaphore> signalSemaphores = {currentContext_->commandProcessedSemaphore->vkSemaphore()};
    std::vector<VkCommandBuffer> commandbuffers = {
        currentContext_->uploadCommandBuffer->vkCommandBuffer(),
        currentContext_->worldCommandBuffer->vkCommandBuffer(),
        currentContext_->overlayCommandBuffer->vkCommandBuffer(),
        currentContext_->fuseCommandBuffer->vkCommandBuffer(),
    };

    VkSubmitInfo vkSubmitInfo = {};
    vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo.waitSemaphoreCount = waitSemaphores.size();
    vkSubmitInfo.pWaitSemaphores = waitSemaphores.data();
    vkSubmitInfo.pWaitDstStageMask = waitStageMasks.data();
    vkSubmitInfo.commandBufferCount = commandbuffers.size();
    vkSubmitInfo.pCommandBuffers = commandbuffers.data();
    vkSubmitInfo.signalSemaphoreCount = signalSemaphores.size();
    vkSubmitInfo.pSignalSemaphores = signalSemaphores.data();

    std::shared_ptr<vk::Fence> fence = currentContext_->commandFinishedFence;
    vkResetFences(device_->vkDevice(), 1, &fence->vkFence());
    vkQueueSubmit(device_->mainVkQueue(), 1, &vkSubmitInfo, fence->vkFence());
}

void Framework::present() {
    if (!running_) return;

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &currentContext_->commandProcessedSemaphore->vkSemaphore();

    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain_->vkSwapchain();
    presentInfo.pImageIndices = &currentContext_->frameIndex;

    VkResult result = vkQueuePresentKHR(device_->mainVkQueue(), &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || vk::Window::framebufferResized ||
        Renderer::options.needRecreate || pipeline_->needRecreate) {
        recreate();
        return;
    } else if (result != VK_SUCCESS) {
        std::cerr << "failed to submit present command buffer" << std::endl;
        waitDeviceIdle();
        exit(EXIT_FAILURE);
    }
}

void Framework::recreate() {
    if (!running_) return;

    std::unique_lock<std::recursive_mutex> lck(Renderer::instance().framework()->recreateMtx());

    Renderer::options.needRecreate = false;
    vk::Window::framebufferResized = false;
    pipeline_->needRecreate = false;

    waitRenderQueueIdle();

    int width = 0, height = 0;
    GLFW_GetFramebufferSize(window_->window(), &width, &height);
    while (width == 0 || height == 0) {
        GLFW_GetFramebufferSize(window_->window(), &width, &height);
        GLFW_WaitEvents();
    }

    currentContextIndex_ = 0;
    currentContext_ = nullptr;
    contexts_.clear();

    uploadCommandBuffers_.clear();
    overlayCommandBuffers_.clear();
    worldCommandBuffers_.clear();
    fuseCommandBuffers_.clear();
    commandFinishedFences_.clear();
    commandProcessedSemaphores_.clear();

    swapchain_->reconstruct();

    uint32_t size = swapchain_->imageCount();

    // create command buffer for each context
    for (int i = 0; i < size; i++) {
        uploadCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
        overlayCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
        worldCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
        fuseCommandBuffers_.emplace_back(vk::CommandBuffer::create(device_, mainCommandPool_));
    }

    // create fence for each context
    for (int i = 0; i < size; i++) { commandFinishedFences_.push_back(vk::Fence::create(device_, true)); }

    // create semaphore for each context for command procssed
    for (int i = 0; i < size; i++) { commandProcessedSemaphores_.push_back(vk::Semaphore::create(device_)); }

    for (int i = 0; i < size; i++) { contexts_.push_back(FrameworkContext::create(shared_from_this(), i)); }

    pipeline_->recreate(shared_from_this());

    Renderer::instance().textures()->bindAllTextures();
}

void Framework::waitDeviceIdle() {
    vkDeviceWaitIdle(device_->vkDevice());
}

void Framework::waitRenderQueueIdle() {
    vkQueueWaitIdle(device_->mainVkQueue());
}

void Framework::waitBackendQueueIdle() {
    vkQueueWaitIdle(device_->secondaryQueue());
}

void Framework::close() {
    if (running_) { pipeline_->close(); }
    running_ = false;
}

bool Framework::isRunning() {
    return running_;
}

void Framework::takeScreenshot(bool withUI, int width, int height, int channel, void *dstPointer) {
    if (indexHistory_.empty()) return;

    uint32_t targetIndex = indexHistory_.front();
    auto context = contexts_[targetIndex];
    std::shared_ptr<vk::Fence> fence = context->commandFinishedFence;
    VkResult result = vkWaitForFences(device_->vkDevice(), 1, &fence->vkFence(), true, UINT64_MAX);
    if (result != VK_SUCCESS) {
        std::cout << "vkWaitForFences failed with error for screenshot: " << std::dec << result << std::endl;
        waitDeviceIdle();
        exit(EXIT_FAILURE);
    }

    std::shared_ptr<vk::HostVisibleBuffer> dstBuffer;
    std::shared_ptr<vk::Image> srcImage;
    std::shared_ptr<vk::DeviceLocalImage> srcDeviceImage;
    std::shared_ptr<vk::SwapchainImage> srcSwapchainImage;

    auto pipelineContext = pipeline_->acquirePipelineContext(context);

    if (withUI && !swapchain_->supportsTransferSrc()) {
        // Swapchain images may not support TRANSFER_SRC on some surfaces.
        // Fall back to the world output (no UI) rather than issuing an invalid copy.
        withUI = false;
    }

    if (withUI) {
        // With-UI screenshot should capture the final composited frame.
        // SDR and HDR now both composite into the swapchain in fuseFinal().
        srcSwapchainImage = context->swapchainImage;
        srcImage = srcSwapchainImage;
    } else {
        if (pipelineContext->worldPipelineContext == nullptr ||
            pipelineContext->worldPipelineContext->outputImage == nullptr) {
            return;
        }
        srcDeviceImage = pipelineContext->worldPipelineContext->outputImage;
        srcImage = srcDeviceImage;
    }

    if (srcImage == nullptr) return;
    if (static_cast<uint32_t>(width) != srcImage->width() || static_cast<uint32_t>(height) != srcImage->height()) return;

    auto isRawMemcpyOk = [](VkFormat format) {
        // Only formats that are already RGBA in memory can be memcpy'd.
        // Swapchain may be BGRA on some platforms.
        return format == VK_FORMAT_R8G8B8A8_UNORM ||
               format == VK_FORMAT_R8G8B8A8_SRGB;
    };

    bool needFormatConversion = !isRawMemcpyOk(srcImage->vkFormat());
    if (channel != 4) return;

    uint32_t rawBufferSize = srcImage->width() * srcImage->height() * srcImage->layer() * vk::formatToByte(srcImage->vkFormat());
    dstBuffer = vk::HostVisibleBuffer::create(vma_, device_, rawBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    VkImageLayout initialLayout = srcSwapchainImage ? srcSwapchainImage->imageLayout() : srcDeviceImage->imageLayout();
    auto mainQueueIndex = physicalDevice_->mainQueueIndex();

    std::shared_ptr<vk::CommandBuffer> oneTimeBuffer = vk::CommandBuffer::create(device_, mainCommandPool_);
    oneTimeBuffer->begin();

    oneTimeBuffer->barriersBufferImage(
        {}, {{
                .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout = initialLayout,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image = srcImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            }});

    VkBufferImageCopy bufferImageCopy{};
    bufferImageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bufferImageCopy.imageSubresource.mipLevel = 0;
    bufferImageCopy.imageSubresource.baseArrayLayer = 0;
    bufferImageCopy.imageSubresource.layerCount = 1;
    bufferImageCopy.imageExtent.width = srcImage->width();
    bufferImageCopy.imageExtent.height = srcImage->height();
    bufferImageCopy.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(oneTimeBuffer->vkCommandBuffer(), srcImage->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           dstBuffer->vkBuffer(), 1, &bufferImageCopy);

    oneTimeBuffer->barriersBufferImage(
        {}, {{
                .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .newLayout = initialLayout,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image = srcImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            }});

    oneTimeBuffer->end();

    VkSubmitInfo vkSubmitInfo = {};
    vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo.waitSemaphoreCount = 0;
    vkSubmitInfo.commandBufferCount = 1;
    vkSubmitInfo.pCommandBuffers = &oneTimeBuffer->vkCommandBuffer();
    vkSubmitInfo.signalSemaphoreCount = 0;
    std::shared_ptr<vk::Fence> oneTimeFence = vk::Fence::create(device_);

    vkQueueSubmit(device_->mainVkQueue(), 1, &vkSubmitInfo, oneTimeFence->vkFence());
    result = vkWaitForFences(device_->vkDevice(), 1, &oneTimeFence->vkFence(), true, UINT64_MAX);
    if (result != VK_SUCCESS) {
        std::cout << "vkWaitForFences failed with error for screenshot: " << std::dec << result << std::endl;
        waitDeviceIdle();
        exit(EXIT_FAILURE);
    }

    if (!needFormatConversion) {
        std::memcpy(dstPointer, dstBuffer->mappedPtr(), width * height * channel);
        return;
    }

    auto *srcPixels = reinterpret_cast<const uint32_t *>(dstBuffer->mappedPtr());
    auto *dstPixels = reinterpret_cast<uint8_t *>(dstPointer);
    uint32_t pixelCount = static_cast<uint32_t>(width * height);
    VkFormat format = srcImage->vkFormat();

    if (format == VK_FORMAT_A2B10G10R10_UNORM_PACK32) {
        for (uint32_t i = 0; i < pixelCount; i++) {
            uint32_t p = srcPixels[i];
            uint32_t r10 = (p >> 0) & 0x3FF;
            uint32_t g10 = (p >> 10) & 0x3FF;
            uint32_t b10 = (p >> 20) & 0x3FF;
            uint32_t a2 = (p >> 30) & 0x03;

            dstPixels[i * 4 + 0] = static_cast<uint8_t>((r10 * 255 + 511) / 1023);
            dstPixels[i * 4 + 1] = static_cast<uint8_t>((g10 * 255 + 511) / 1023);
            dstPixels[i * 4 + 2] = static_cast<uint8_t>((b10 * 255 + 511) / 1023);
            dstPixels[i * 4 + 3] = static_cast<uint8_t>(a2 * 85);
        }
    } else if (format == VK_FORMAT_A2R10G10B10_UNORM_PACK32) {
        for (uint32_t i = 0; i < pixelCount; i++) {
            uint32_t p = srcPixels[i];
            uint32_t b10 = (p >> 0) & 0x3FF;
            uint32_t g10 = (p >> 10) & 0x3FF;
            uint32_t r10 = (p >> 20) & 0x3FF;
            uint32_t a2 = (p >> 30) & 0x03;

            dstPixels[i * 4 + 0] = static_cast<uint8_t>((r10 * 255 + 511) / 1023);
            dstPixels[i * 4 + 1] = static_cast<uint8_t>((g10 * 255 + 511) / 1023);
            dstPixels[i * 4 + 2] = static_cast<uint8_t>((b10 * 255 + 511) / 1023);
            dstPixels[i * 4 + 3] = static_cast<uint8_t>(a2 * 85);
        }
    } else if (format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB) {
        // Swizzle BGRA → RGBA.
        for (uint32_t i = 0; i < pixelCount; i++) {
            uint32_t p = srcPixels[i];
            uint8_t b = static_cast<uint8_t>((p >> 0) & 0xFF);
            uint8_t g = static_cast<uint8_t>((p >> 8) & 0xFF);
            uint8_t r = static_cast<uint8_t>((p >> 16) & 0xFF);
            uint8_t a = static_cast<uint8_t>((p >> 24) & 0xFF);
            dstPixels[i * 4 + 0] = r;
            dstPixels[i * 4 + 1] = g;
            dstPixels[i * 4 + 2] = b;
            dstPixels[i * 4 + 3] = a;
        }
    } else {
        // Fallback for unsupported packed HDR formats.
        std::memset(dstPointer, 0, width * height * channel);
    }
}

VkFormat Framework::takeScreenshotRawHdrPacked(bool withUI,
                                               int width,
                                               int height,
                                               void *dstPointer,
                                               int dstByteSize) {
    if (dstPointer == nullptr || dstByteSize <= 0) return VK_FORMAT_UNDEFINED;
    if (indexHistory_.empty()) return VK_FORMAT_UNDEFINED;

    uint32_t targetIndex = indexHistory_.front();
    auto context = contexts_[targetIndex];
    std::shared_ptr<vk::Fence> fence = context->commandFinishedFence;
    VkResult result = vkWaitForFences(device_->vkDevice(), 1, &fence->vkFence(), true, UINT64_MAX);
    if (result != VK_SUCCESS) {
        std::cout << "vkWaitForFences failed with error for screenshot: " << std::dec << result << std::endl;
        waitDeviceIdle();
        exit(EXIT_FAILURE);
    }

    std::shared_ptr<vk::HostVisibleBuffer> dstBuffer;
    std::shared_ptr<vk::Image> srcImage;
    std::shared_ptr<vk::DeviceLocalImage> srcDeviceImage;
    std::shared_ptr<vk::SwapchainImage> srcSwapchainImage;

    auto pipelineContext = pipeline_->acquirePipelineContext(context);

    if (withUI) {
        if (Renderer::options.hdrEnabled && swapchain_->isHDR()) {
            if (!swapchain_->supportsTransferSrc()) {
                return VK_FORMAT_UNDEFINED;
            }
            srcSwapchainImage = context->swapchainImage;
            srcImage = srcSwapchainImage;
        } else {
            srcDeviceImage = pipelineContext->uiModuleContext->overlayDrawColorImage;
            srcImage = srcDeviceImage;
        }
    } else {
        if (pipelineContext->worldPipelineContext == nullptr ||
            pipelineContext->worldPipelineContext->outputImage == nullptr) {
            return VK_FORMAT_UNDEFINED;
        }
        srcDeviceImage = pipelineContext->worldPipelineContext->outputImage;
        srcImage = srcDeviceImage;
    }

    if (srcImage == nullptr) return VK_FORMAT_UNDEFINED;
    if (static_cast<uint32_t>(width) != srcImage->width() || static_cast<uint32_t>(height) != srcImage->height()) {
        return VK_FORMAT_UNDEFINED;
    }

    VkFormat format = srcImage->vkFormat();
    if (format != VK_FORMAT_A2B10G10R10_UNORM_PACK32 && format != VK_FORMAT_A2R10G10B10_UNORM_PACK32) {
        return VK_FORMAT_UNDEFINED;
    }

    uint64_t rawBufferSize64 = static_cast<uint64_t>(srcImage->width()) * srcImage->height() * srcImage->layer() *
                               vk::formatToByte(srcImage->vkFormat());
    if (rawBufferSize64 > static_cast<uint64_t>(dstByteSize)) {
        return VK_FORMAT_UNDEFINED;
    }
    size_t rawBufferSize = static_cast<size_t>(rawBufferSize64);

    dstBuffer = vk::HostVisibleBuffer::create(vma_, device_, rawBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    VkImageLayout initialLayout = srcSwapchainImage ? srcSwapchainImage->imageLayout() : srcDeviceImage->imageLayout();
    auto mainQueueIndex = physicalDevice_->mainQueueIndex();

    std::shared_ptr<vk::CommandBuffer> oneTimeBuffer = vk::CommandBuffer::create(device_, mainCommandPool_);
    oneTimeBuffer->begin();

    oneTimeBuffer->barriersBufferImage(
        {}, {{
                .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout = initialLayout,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image = srcImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            }});

    VkBufferImageCopy bufferImageCopy{};
    bufferImageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bufferImageCopy.imageSubresource.mipLevel = 0;
    bufferImageCopy.imageSubresource.baseArrayLayer = 0;
    bufferImageCopy.imageSubresource.layerCount = 1;
    bufferImageCopy.imageExtent.width = srcImage->width();
    bufferImageCopy.imageExtent.height = srcImage->height();
    bufferImageCopy.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(oneTimeBuffer->vkCommandBuffer(), srcImage->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           dstBuffer->vkBuffer(), 1, &bufferImageCopy);

    oneTimeBuffer->barriersBufferImage(
        {}, {{
                .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .newLayout = initialLayout,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image = srcImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            }});

    oneTimeBuffer->end();

    VkSubmitInfo vkSubmitInfo = {};
    vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo.waitSemaphoreCount = 0;
    vkSubmitInfo.commandBufferCount = 1;
    vkSubmitInfo.pCommandBuffers = &oneTimeBuffer->vkCommandBuffer();
    vkSubmitInfo.signalSemaphoreCount = 0;
    std::shared_ptr<vk::Fence> oneTimeFence = vk::Fence::create(device_);

    vkQueueSubmit(device_->mainVkQueue(), 1, &vkSubmitInfo, oneTimeFence->vkFence());
    result = vkWaitForFences(device_->vkDevice(), 1, &oneTimeFence->vkFence(), true, UINT64_MAX);
    if (result != VK_SUCCESS) {
        std::cout << "vkWaitForFences failed with error for screenshot: " << std::dec << result << std::endl;
        waitDeviceIdle();
        exit(EXIT_FAILURE);
    }

    std::memcpy(dstPointer, dstBuffer->mappedPtr(), rawBufferSize);
    return format;
}

std::recursive_mutex &Framework::recreateMtx() {
    return recreateMtx_;
}

std::shared_ptr<vk::Instance> Framework::instance() {
    return instance_;
}

std::shared_ptr<vk::Window> Framework::window() {
    return window_;
}

std::shared_ptr<vk::PhysicalDevice> Framework::physicalDevice() {
    return physicalDevice_;
}

std::shared_ptr<vk::Device> Framework::device() {
    return device_;
}

std::shared_ptr<vk::VMA> Framework::vma() {
    return vma_;
}

std::shared_ptr<vk::Swapchain> Framework::swapchain() {
    return swapchain_;
}

std::shared_ptr<vk::CommandPool> Framework::mainCommandPool() {
    return mainCommandPool_;
}

std::shared_ptr<vk::CommandPool> Framework::asyncCommandPool() {
    return asyncCommandPool_;
}

std::shared_ptr<vk::CommandBuffer> Framework::worldAsyncCommandBuffer() {
    return worldAsyncCommandBuffer_;
}

std::vector<std::shared_ptr<vk::Semaphore>> &Framework::commandProcessedSemaphores() {
    return commandProcessedSemaphores_;
}

std::vector<std::shared_ptr<vk::Fence>> &Framework::commandFinishedFences() {
    return commandFinishedFences_;
}

std::vector<std::shared_ptr<FrameworkContext>> &Framework::contexts() {
    return contexts_;
}

std::shared_ptr<FrameworkContext> Framework::safeAcquireCurrentContext() {
    std::unique_lock<std::recursive_mutex> lck(recreateMtx_);
    // for continous window operation, currentContext_ will always be reset, busy waiting
    while (currentContext_ == nullptr) {
        // ensure currentContext_ is not nullptr after seapchain recreation
        acquireContext();
    }
    return currentContext_;
}

std::shared_ptr<Pipeline> Framework::pipeline() {
    return pipeline_;
}

GarbageCollector &Framework::gc() {
    return *gc_;
}

std::shared_ptr<vk::Semaphore> Framework::acquireSemaphore() {
    std::shared_ptr<vk::Semaphore> semaphore;
    if (recycledImageAcquiredSemaphores_.empty()) {
        semaphore = vk::Semaphore::create(device_);
    } else {
        semaphore = recycledImageAcquiredSemaphores_.front();
        recycledImageAcquiredSemaphores_.pop();
    }
    return semaphore;
}

void Framework::recycleSemaphore(std::shared_ptr<vk::Semaphore> semaphore) {
    recycledImageAcquiredSemaphores_.push(semaphore);
}

GarbageCollector::GarbageCollector(std::shared_ptr<Framework> framework) : framework_(framework) {
    collectors_.resize(framework->swapchain_->imageCount());
}

void GarbageCollector::clear() {
    index_ = (index_ + 1) % collectors_.size();

    auto framework = framework_.lock();
    collectors_[index_].clear();
}
