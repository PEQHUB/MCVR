#include "core/render/ui_render_context.hpp"

#include "core/render/buffers.hpp"
#include "core/render/overlay_compositor.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstdarg>

static std::ostream &uiCtxCout() { return std::cout << "[UIRenderContext] "; }
static std::ostream &uiCtxCerr() { return std::cerr << "[UIRenderContext] "; }

void UIRenderContext::diagLog(const char *fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (diagFile_.is_open()) {
        SYSTEMTIME st;
        GetLocalTime(&st);
        char ts[32];
        snprintf(ts, sizeof(ts), "%02d:%02d:%02d.%03d",
                 st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
        diagFile_ << ts << " " << buf << std::endl;
        diagFile_.flush();
    }
}

UIRenderContext::~UIRenderContext() {
    destroy();
}

bool UIRenderContext::init(std::shared_ptr<Framework> framework, UIModule *uiModule) {
    framework_ = framework;
    device_ = framework->device();
    uiModule_ = uiModule;

    auto swapExtent = framework->swapchain()->vkExtent();
    width_ = swapExtent.width;
    height_ = swapExtent.height;

    // Open diagnostic log
    auto logDir = Renderer::folderPath / "logs";
    std::filesystem::create_directories(logDir);
    diagFile_.open((logDir / "ui_render_context_diag.log").string(), std::ios::trunc);

    createResources();

    diagLog("init: %ux%u", width_, height_);
    uiCtxCout() << "initialized " << width_ << "x" << height_ << std::endl;
    return true;
}

void UIRenderContext::destroy() {
    destroyResources();
    device_.reset();
    framework_.reset();
    uiModule_ = nullptr;
    if (diagFile_.is_open()) diagFile_.close();
}

void UIRenderContext::createResources() {
    auto fw = framework_.lock();
    if (!fw || !device_) return;

    auto physicalDevice = fw->physicalDevice();

    // Command pool on main queue family (same family as secondary on NVIDIA,
    // but using main ensures Streamline interposer recognizes it)
    commandPool_ = vk::CommandPool::create(physicalDevice, device_,
                                           physicalDevice->mainQueueIndex());

    // Own render pass — identical to UIModule's but with CLEAR depth (not LOAD).
    // UIModule uses LOAD because it carries depth across draw calls within a frame.
    // UIRenderContext's depth images start undefined each frame, so CLEAR is required.
    renderPass_ = vk::RenderPassBuilder{}
                      .beginAttachmentDescription()
                      .defineAttachmentDescription(VkAttachmentDescription{
                          .format = VK_FORMAT_R8G8B8A8_SRGB,
                          .samples = VK_SAMPLE_COUNT_1_BIT,
                          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                          .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
#ifdef USE_AMD
                          .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                          .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                          .initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                          .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                      })
                      .defineAttachmentDescription(VkAttachmentDescription{
                          .format = VK_FORMAT_D32_SFLOAT,
                          .samples = VK_SAMPLE_COUNT_1_BIT,
                          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                          .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
                          .initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                          .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                      })
                      .endAttachmentDescription()
                      .beginAttachmentReference()
                      .defineAttachmentReference({
                          .attachment = 0,
                          .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      })
                      .defineAttachmentReference({
                          .attachment = 1,
                          .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                      })
                      .endAttachmentReference()
                      .beginSubpassDescription()
                      .defineSubpassDescription({
                          .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                          .colorAttachmentIndices = {0},
                          .depthStencilAttachmentIndex = 1,
                      })
                      .endSubpassDescription()
                      .build(device_);

    for (int i = 0; i < kBufferCount; i++) {
        commandBuffers_[i] = vk::CommandBuffer::create(device_, commandPool_);

        // Color image — same format as UIModule's overlay draw images
        colorImages_[i] = vk::DeviceLocalImage::create(
            device_, fw->vma(), false, width_, height_, 1,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

        colorImages_[i]->imageLayout() = VK_IMAGE_LAYOUT_UNDEFINED;

        // Depth image
        depthImages_[i] = vk::DeviceLocalImage::create(
            device_, fw->vma(), false, width_, height_, 1,
            VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        depthImages_[i]->imageLayout() = VK_IMAGE_LAYOUT_UNDEFINED;

        // Framebuffer using our own render pass (CLEAR depth, not LOAD)
        framebuffers_[i] = vk::FramebufferBuilder{}
                               .beginAttachment()
                               .defineAttachment(colorImages_[i])
                               .defineAttachment(depthImages_[i])
                               .endAttachment()
                               .build(device_, renderPass_);

        // Own descriptor table (same layout as UIModule's — bindless texture array)
        descriptorTables_[i] = vk::DescriptorTableBuilder{}
                                   .beginDescriptorLayoutSet()  // set 0
                                   .beginDescriptorLayoutSetBinding()
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 0,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .descriptorCount = 4096,
                                       .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 1,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   })
                                   .endDescriptorLayoutSetBinding()
                                   .endDescriptorLayoutSet()
                                   .beginDescriptorLayoutSet()  // set 1
                                   .beginDescriptorLayoutSetBinding()
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 0,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 1,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   })
                                   .endDescriptorLayoutSetBinding()
                                   .endDescriptorLayoutSet()
                                   .definePushConstant(VkPushConstantRange{
                                       .stageFlags = VK_SHADER_STAGE_ALL,
                                       .offset = 0,
                                       .size = sizeof(int),
                                   })
                                   .build(device_);

        colorImageSamplers_[i] = vk::Sampler::create(
            device_, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);

        // Bind the color image to set 0, binding 1 (frame sampler) in descriptor table
        descriptorTables_[i]->bindSamplerImage(
            colorImageSamplers_[i], colorImages_[i],
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1, 0);

        // Fence — starts signaled so first beginFrame() wait succeeds
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(device_->vkDevice(), &fenceInfo, nullptr, &fences_[i]);
    }

    diagLog("createResources: %d buffers, %ux%u", kBufferCount, width_, height_);
}

void UIRenderContext::destroyResources() {
    if (!device_) return;

    // Wait for any in-flight work
    for (int i = 0; i < kBufferCount; i++) {
        if (fences_[i] != VK_NULL_HANDLE && !fenceSignaled_[i]) {
            VkResult r = vkWaitForFences(device_->vkDevice(), 1, &fences_[i], VK_TRUE, 500000000ULL); // 500ms
            if (r == VK_TIMEOUT) {
                diagLog("destroyResources: fence[%d] timeout, continuing", i);
            }
            fenceSignaled_[i] = true;
        }
    }

    for (int i = 0; i < kBufferCount; i++) {
        commandBuffers_[i].reset();
        colorImages_[i].reset();
        depthImages_[i].reset();
        framebuffers_[i].reset();
        descriptorTables_[i].reset();
        colorImageSamplers_[i].reset();

        if (fences_[i] != VK_NULL_HANDLE) {
            vkDestroyFence(device_->vkDevice(), fences_[i], nullptr);
            fences_[i] = VK_NULL_HANDLE;
        }
    }

    commandPool_.reset();
    renderPass_.reset();
    frameActive_ = false;
    renderPassActive_ = false;
    // Note: firstFramePresented_ is NOT reset here. After swapchain recreate,
    // the UIThread will present again immediately. Keeping it true prevents the
    // render thread from briefly re-enabling overlay draws (which would double the UI).
}

// ── Frame lifecycle ──────────────────────────────────────────────────────

void UIRenderContext::beginFrame() {
    // Check pause — if requested, acknowledge and wait for resume
    if (pauseRequested_.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lk(pauseMtx_);
        pauseAcknowledged_ = true;
        pauseCv_.notify_all();
        pauseCv_.wait(lk, [this] { return !pauseRequested_.load(std::memory_order_acquire); });
    }

    auto fw = framework_.lock();
    if (!fw) return;

    int idx = currentIndex_;

    // Wait for this buffer's previous submission to complete (only if actually submitted)
    if (!fenceSignaled_[idx]) {
        vkWaitForFences(device_->vkDevice(), 1, &fences_[idx], VK_TRUE, UINT64_MAX);
    }
    // Don't reset fence here — reset immediately before vkQueueSubmit in submitAndPresent
    // to guarantee every reset fence is submitted (prevents hang on pause path).

    // Mark that a thread is actively driving the loop
    loopActive_.store(true, std::memory_order_release);

    vkResetCommandBuffer(commandBuffers_[idx]->vkCommandBuffer(), 0);
    commandBuffers_[idx]->begin();

    auto mainQueueIndex = fw->physicalDevice()->mainQueueIndex();

    // Transition color image to render pass initialLayout
    commandBuffers_[idx]->barriersBufferImage(
        {}, {
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    .oldLayout = colorImages_[idx]->imageLayout(),
#ifdef USE_AMD
                    .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                    .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = colorImages_[idx],
                    .subresourceRange = vk::wholeColorSubresourceRange,
                },
                {
                    .srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                    .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    .oldLayout = depthImages_[idx]->imageLayout(),
                    .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = depthImages_[idx],
                    .subresourceRange = vk::wholeDepthSubresourceRange,
                },
            });

#ifdef USE_AMD
    colorImages_[idx]->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    colorImages_[idx]->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
    depthImages_[idx]->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Begin render pass with CLEAR for both color and depth
    commandBuffers_[idx]->beginRenderPass({
        .renderPass = renderPass_,
        .framebuffer = framebuffers_[idx],
        .renderAreaExtent = {width_, height_},
        .clearValues = {{.color = {clearColor_[0], clearColor_[1], clearColor_[2], clearColor_[3]}},
                        {.depthStencil = {.depth = 1.0f}}},
    });

    // Bind descriptor table for draw calls
    commandBuffers_[idx]->bindDescriptorTable(descriptorTables_[idx], VK_PIPELINE_BIND_POINT_GRAPHICS);

    frameActive_ = true;
    renderPassActive_ = true;
}

void UIRenderContext::endFrame() {
    if (!frameActive_) return;

    int idx = currentIndex_;

    if (renderPassActive_) {
        commandBuffers_[idx]->endRenderPass();
        renderPassActive_ = false;
#ifdef USE_AMD
        colorImages_[idx]->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        colorImages_[idx]->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
    }

    // Premultiply alpha for DComp compositing
    auto *compositor = overlayCompositor_.load(std::memory_order_acquire);
    if (compositor && compositor->isActive()) {
        auto fw = framework_.lock();
        if (fw) {
            compositor->recordPremultiply(commandBuffers_[idx], colorImages_[idx],
                                          fw->physicalDevice()->mainQueueIndex());
        }
    }

    commandBuffers_[idx]->end();
}

void UIRenderContext::submitAndPresent() {
    if (!frameActive_) return;

    // Mid-frame pause check — if pause was requested while we were in Java-land,
    // acknowledge it now before submitting GPU work.
    if (pauseRequested_.load(std::memory_order_acquire)) {
        // Don't submit — just end the frame and acknowledge the pause.
        // The command buffer was already ended in endFrame(), so we just skip submit.
        // Mark fence as signaled so next beginFrame() won't wait on an unsubmitted fence.
        int skipIdx = currentIndex_;
        fenceSignaled_[skipIdx] = true;
        frameActive_ = false;
        currentIndex_ = (currentIndex_ + 1) % kBufferCount;
        diagLog("submitAndPresent: skipped — pause requested mid-frame");

        std::unique_lock<std::mutex> lk(pauseMtx_);
        pauseAcknowledged_ = true;
        pauseCv_.notify_all();
        pauseCv_.wait(lk, [this] { return !pauseRequested_.load(std::memory_order_acquire); });
        return;
    }

    int idx = currentIndex_;
    auto fw = framework_.lock();
    if (!fw) return;

    // Submit to secondary queue with fence
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers_[idx]->vkCommandBuffer();

    // Reset fence immediately before submit — guarantees every reset fence gets submitted.
    // This prevents the hang where beginFrame() waits on a fence that was reset but never submitted.
    vkResetFences(device_->vkDevice(), 1, &fences_[idx]);
    fenceSignaled_[idx] = false;

    // Submit to main queue with mutex (shared with render thread)
    VkResult result;
    {
        std::lock_guard<std::mutex> qLock(device_->queueMutex());
        result = vkQueueSubmit(device_->mainVkQueue(), 1, &submitInfo, fences_[idx]);
    }
    if (result != VK_SUCCESS) {
        uiCtxCerr() << "submit failed: " << result << " idx=" << idx << std::endl;
        diagLog("submit failed: %d idx=%d", (int)result, idx);
        fenceSignaled_[idx] = true;  // submit failed — fence was never actually submitted
        frameActive_ = false;
        currentIndex_ = (currentIndex_ + 1) % kBufferCount;
        return;
    }

    // Wait for GPU to finish
    VkResult waitResult = vkWaitForFences(device_->vkDevice(), 1, &fences_[idx], VK_TRUE, 100000000ULL); // 100ms timeout
    fenceSignaled_[idx] = (waitResult == VK_SUCCESS || waitResult == VK_TIMEOUT);
    if (waitResult == VK_ERROR_DEVICE_LOST) {
        diagLog("device lost during fence wait — stopping UI loop");
        uiCtxCerr() << "device lost, UI thread will idle" << std::endl;
        frameActive_ = false;
        currentIndex_ = (currentIndex_ + 1) % kBufferCount;
        return;
    }

    // Present via DComp overlay + pace to display rate
    auto *compositor = overlayCompositor_.load(std::memory_order_acquire);
    if (compositor && compositor->isActive()) {
        compositor->present();
        compositor->waitForDisplayReady(8);
    } else {
        Sleep(8);
    }

    if (!firstFramePresented_.load(std::memory_order_acquire)) {
        firstFramePresented_.store(true, std::memory_order_release);
        diagLog("first frame presented");
    }

    frameActive_ = false;

    // Advance to next buffer
    currentIndex_ = (currentIndex_ + 1) % kBufferCount;
}

// ── Draw commands ────────────────────────────────────────────────────────

void UIRenderContext::drawIndexed(std::shared_ptr<vk::DeviceLocalBuffer> vertexBuffer,
                                   std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer,
                                   OverlayDrawPipelineType pipelineType,
                                   uint32_t indexCount,
                                   VkIndexType indexType) {
    if (!frameActive_ || !renderPassActive_) return;

    int idx = currentIndex_;
    auto cmd = commandBuffers_[idx];

    // Bind pipeline
    vkCmdBindPipeline(cmd->vkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                      uiModule_->overlayDrawPipeline(pipelineType)->vkPipeline());

    // Push draw ID
    auto pipelineLayout = descriptorTables_[idx]->vkPipelineLayout();
    int drawID = Renderer::instance().buffers()->getDrawID();
    vkCmdPushConstants(cmd->vkCommandBuffer(), pipelineLayout, VK_SHADER_STAGE_ALL, 0,
                       sizeof(int), &drawID);

    cmd->bindVertexBuffers(vertexBuffer)
        ->bindIndexBuffer(indexBuffer, indexType)
        ->drawIndexed(indexCount, 1);
}

// ── Dynamic state ────────────────────────────────────────────────────────

void UIRenderContext::setViewport(int x, int y, int w, int h) {
    viewport_ = {(float)x, (float)y, (float)w, (float)h, 0.0f, 1.0f};
}

void UIRenderContext::setScissor(int x, int y, int w, int h) {
    scissor_ = {{x, y}, {(uint32_t)w, (uint32_t)h}};
}

void UIRenderContext::setScissorEnabled(bool enabled) {
    scissorEnabled_ = enabled;
}

void UIRenderContext::clearColor(float r, float g, float b, float a) {
    clearColor_ = {r, g, b, a};
}

void UIRenderContext::syncDynamicState() {
    if (!frameActive_) return;

    int idx = currentIndex_;
    auto vkCmd = commandBuffers_[idx]->vkCommandBuffer();

    // Viewport
    vkCmdSetViewport(vkCmd, 0, 1, &viewport_);

    // Scissor
    if (scissorEnabled_) {
        vkCmdSetScissor(vkCmd, 0, 1, &scissor_);
    } else {
        VkRect2D full = {{0, 0}, {width_, height_}};
        vkCmdSetScissor(vkCmd, 0, 1, &full);
    }

    // Depth/stencil
    vkCmdSetDepthTestEnable(vkCmd, depthTestEnable_);
    vkCmdSetDepthWriteEnable(vkCmd, depthWriteEnable_);
    vkCmdSetDepthCompareOp(vkCmd, depthCompareOp_);
    vkCmdSetStencilTestEnable(vkCmd, stencilTestEnable_);
    vkCmdSetStencilOp(vkCmd, VK_STENCIL_FACE_FRONT_BIT, failOp_[0], passOp_[0], depthFailOp_[0], stencilCompareOp_[0]);
    vkCmdSetStencilOp(vkCmd, VK_STENCIL_FACE_BACK_BIT, failOp_[1], passOp_[1], depthFailOp_[1], stencilCompareOp_[1]);
    vkCmdSetStencilReference(vkCmd, VK_STENCIL_FACE_FRONT_BIT, stencilReference_[0]);
    vkCmdSetStencilReference(vkCmd, VK_STENCIL_FACE_BACK_BIT, stencilReference_[1]);
    vkCmdSetStencilCompareMask(vkCmd, VK_STENCIL_FACE_FRONT_BIT, stencilCompareMask_[0]);
    vkCmdSetStencilCompareMask(vkCmd, VK_STENCIL_FACE_BACK_BIT, stencilCompareMask_[1]);
    vkCmdSetStencilWriteMask(vkCmd, VK_STENCIL_FACE_FRONT_BIT, stencilWriteMask_[0]);
    vkCmdSetStencilWriteMask(vkCmd, VK_STENCIL_FACE_BACK_BIT, stencilWriteMask_[1]);

    // Rasterization
    vkCmdSetCullMode(vkCmd, cullMode_);
    vkCmdSetFrontFace(vkCmd, frontFace_);
    vkCmdSetLineWidth(vkCmd, lineWidth_);
    vkCmdSetDepthBiasEnable(vkCmd, depthBiasEnable_);
    if (depthBiasEnable_) {
        vkCmdSetDepthBias(vkCmd, depthBiasConstantFactor_, 0.0f, depthBiasSlopeFactor_);
    }

    // Blending (EXT dynamic state)
    vkCmdSetColorBlendEnableEXT(vkCmd, 0, 1, &blendEnabled_);
    vkCmdSetColorBlendEquationEXT(vkCmd, 0, 1, &colorBlendEquation_);
    VkColorComponentFlags writeMask = colorWriteMask_;
    vkCmdSetColorWriteMaskEXT(vkCmd, 0, 1, &writeMask);
    VkBool32 logicEnable = colorLogicOpEnable_;
    vkCmdSetLogicOpEnableEXT(vkCmd, logicEnable);
    if (colorLogicOpEnable_) {
        vkCmdSetLogicOpEXT(vkCmd, colorLogicOp_);
    }
    vkCmdSetBlendConstants(vkCmd, blendConstants_.data());
}

// ── State setters ────────────────────────────────────────────────────────

void UIRenderContext::setBlendEnable(bool enable) {
    blendEnabled_ = enable ? VK_TRUE : VK_FALSE;
}

void UIRenderContext::setBlendFuncSeparate(int srcColor, int srcAlpha, int dstColor, int dstAlpha) {
    colorBlendEquation_.srcColorBlendFactor = static_cast<VkBlendFactor>(srcColor);
    colorBlendEquation_.srcAlphaBlendFactor = static_cast<VkBlendFactor>(srcAlpha);
    colorBlendEquation_.dstColorBlendFactor = static_cast<VkBlendFactor>(dstColor);
    colorBlendEquation_.dstAlphaBlendFactor = static_cast<VkBlendFactor>(dstAlpha);
}

void UIRenderContext::setBlendOpSeparate(int colorOp, int alphaOp) {
    colorBlendEquation_.colorBlendOp = static_cast<VkBlendOp>(colorOp);
    colorBlendEquation_.alphaBlendOp = static_cast<VkBlendOp>(alphaOp);
}

void UIRenderContext::setColorWriteMask(int mask) {
    colorWriteMask_ = static_cast<VkColorComponentFlags>(mask);
}

void UIRenderContext::setColorLogicOpEnable(bool enable) {
    colorLogicOpEnable_ = enable;
}

void UIRenderContext::setColorLogicOp(int op) {
    colorLogicOp_ = static_cast<VkLogicOp>(op);
}

void UIRenderContext::setBlendConstants(float c0, float c1, float c2, float c3) {
    blendConstants_ = {c0, c1, c2, c3};
}

void UIRenderContext::setDepthTestEnable(bool enable) {
    depthTestEnable_ = enable;
}

void UIRenderContext::setDepthWriteEnable(bool enable) {
    depthWriteEnable_ = enable;
}

void UIRenderContext::setDepthCompareOp(int op) {
    depthCompareOp_ = static_cast<VkCompareOp>(op);
}

void UIRenderContext::setStencilTestEnable(bool enable) {
    stencilTestEnable_ = enable;
}

void UIRenderContext::setStencilFrontFunc(int compareOp, int reference, int compareMask) {
    stencilCompareOp_[0] = static_cast<VkCompareOp>(compareOp);
    stencilReference_[0] = reference;
    stencilCompareMask_[0] = compareMask;
}

void UIRenderContext::setStencilBackFunc(int compareOp, int reference, int compareMask) {
    stencilCompareOp_[1] = static_cast<VkCompareOp>(compareOp);
    stencilReference_[1] = reference;
    stencilCompareMask_[1] = compareMask;
}

void UIRenderContext::setStencilFrontOp(int failOp, int depthFailOp, int passOp) {
    failOp_[0] = static_cast<VkStencilOp>(failOp);
    depthFailOp_[0] = static_cast<VkStencilOp>(depthFailOp);
    passOp_[0] = static_cast<VkStencilOp>(passOp);
}

void UIRenderContext::setStencilBackOp(int failOp, int depthFailOp, int passOp) {
    failOp_[1] = static_cast<VkStencilOp>(failOp);
    depthFailOp_[1] = static_cast<VkStencilOp>(depthFailOp);
    passOp_[1] = static_cast<VkStencilOp>(passOp);
}

void UIRenderContext::setStencilFrontWriteMask(int writeMask) {
    stencilWriteMask_[0] = writeMask;
}

void UIRenderContext::setStencilBackWriteMask(int writeMask) {
    stencilWriteMask_[1] = writeMask;
}

void UIRenderContext::setCullMode(int cullMode) {
    cullMode_ = static_cast<VkCullModeFlags>(cullMode);
}

void UIRenderContext::setFrontFace(int frontFace) {
    frontFace_ = static_cast<VkFrontFace>(frontFace);
}

void UIRenderContext::setPolygonMode(int polygonMode) {
    polygonMode_ = static_cast<VkPolygonMode>(polygonMode);
}

void UIRenderContext::setDepthBiasEnable(int /*polygonMode*/, bool enable) {
    depthBiasEnable_ = enable;
}

void UIRenderContext::setDepthBias(float slopeFactor, float constantFactor) {
    depthBiasSlopeFactor_ = slopeFactor;
    depthBiasConstantFactor_ = constantFactor;
}

void UIRenderContext::setLineWidth(float lineWidth) {
    lineWidth_ = lineWidth;
}

// ── Texture binding ──────────────────────────────────────────────────────

void UIRenderContext::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                                   std::shared_ptr<vk::DeviceLocalImage> image,
                                   int index) {
    for (int i = 0; i < kBufferCount; i++) {
        descriptorTables_[i]->bindSamplerImage(
            sampler, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 0, index);
    }
}

// ── Pause/resume ─────────────────────────────────────────────────────────

void UIRenderContext::pause() {
    if (paused_) return;

    // If no thread is driving the frame loop (beginFrame hasn't been called),
    // there's nothing to synchronize — just mark as paused immediately.
    if (!loopActive_.load(std::memory_order_acquire)) {
        paused_ = true;
        diagLog("paused (no active loop)");
        return;
    }

    pauseAcknowledged_ = false;
    pauseRequested_ = true;

    // Wait for the UI thread to acknowledge the pause (with timeout)
    std::unique_lock<std::mutex> lk(pauseMtx_);
    bool acked = pauseCv_.wait_for(lk, std::chrono::seconds(3),
                                    [this] { return pauseAcknowledged_.load(); });

    if (!acked) {
        // UIThread is alive but busy. Wait indefinitely — proceeding without
        // acknowledgment would destroy resources UIThread is using (use-after-free).
        // The Java isPauseRequested() check ensures UIThread responds within one
        // frame cycle (~30ms). If it takes longer, it's a lag spike, not a deadlock.
        diagLog("pause timeout (3s) — UIThread busy, waiting indefinitely...");
        uiCtxCout() << "pause timeout — waiting for UIThread..." << std::endl;
        pauseCv_.wait(lk, [this] { return pauseAcknowledged_.load(); });
        diagLog("pause acknowledged (after extended wait)");
    }

    paused_ = true;
    uiCtxCout() << "paused (acknowledged)" << std::endl;
}

void UIRenderContext::resume() {
    if (!paused_) return;

    paused_ = false;
    pauseRequested_ = false;
    // loopActive_ stays as-is — the thread will set it on next beginFrame
    pauseCv_.notify_all();

    diagLog("resumed");
    uiCtxCout() << "resumed" << std::endl;
}

void UIRenderContext::onSwapchainRecreate() {
    auto fw = framework_.lock();
    if (!fw) return;

    auto swapExtent = fw->swapchain()->vkExtent();
    width_ = swapExtent.width;
    height_ = swapExtent.height;

    destroyResources();
    createResources();

    diagLog("onSwapchainRecreate: %ux%u", width_, height_);
}

void UIRenderContext::setOverlayCompositor(OverlayCompositor *compositor) {
    overlayCompositor_.store(compositor, std::memory_order_release);
}
