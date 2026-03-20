#include "core/render/present_thread.hpp"
#include "core/render/hdr_composite_pass.hpp"
#include "core/render/overlay_compositor.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdarg>

// ── Diagnostic file logger (writes to same dir as overlay_diag.log) ──
static std::ofstream sPtDiag;
static void ptDiag(const char *fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (sPtDiag.is_open()) {
        SYSTEMTIME st;
        GetLocalTime(&st);
        char ts[32];
        snprintf(ts, sizeof(ts), "%02d:%02d:%02d.%03d",
                 st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
        sPtDiag << ts << " " << buf << std::endl;
        sPtDiag.flush();
    }
}

static std::ostream &ptCout() { return std::cout << "[PresentThread] "; }
static std::ostream &ptCerr() { return std::cerr << "[PresentThread] "; }

PresentThread::~PresentThread() {
    stop();
    destroyVulkanResources();
}

void PresentThread::init(std::shared_ptr<Framework> framework, FrameSlotRing *ring) {
    framework_ = framework;
    ring_ = ring;
    device_ = framework->device();
    createVulkanResources();

    auto logDir = Renderer::folderPath / "logs";
    std::filesystem::create_directories(logDir);
    sPtDiag.open((logDir / "present_thread_diag.log").string(), std::ios::trunc);
    ptDiag("init complete");

    ptCout() << "initialized" << std::endl;
}

void PresentThread::createVulkanResources() {
    auto fw = framework_.lock();
    if (!fw || !device_) return;

    auto physicalDevice = fw->physicalDevice();
    auto swapchain = fw->swapchain();

    // Command pool for the secondary (present) queue family
    commandPool_ = vk::CommandPool::create(physicalDevice, device_,
                                           physicalDevice->secondaryQueueIndex());

    // One command buffer per swapchain image
    uint32_t imageCount = swapchain->imageCount();
    commandBuffers_.clear();
    for (uint32_t i = 0; i < imageCount; i++) {
        commandBuffers_.push_back(vk::CommandBuffer::create(device_, commandPool_));
    }

    // Fence for composite GPU completion (starts signaled so first wait succeeds)
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(device_->vkDevice(), &fenceInfo, nullptr, &compositeFence_);

    // Semaphores: acquire -> composite -> present ordering
    VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(device_->vkDevice(), &semInfo, nullptr, &acquireSemaphore_);
    vkCreateSemaphore(device_->vkDevice(), &semInfo, nullptr, &compositeSemaphore_);
}

void PresentThread::destroyVulkanResources() {
    if (!device_) return;

    commandBuffers_.clear();
    commandPool_.reset();

    if (compositeFence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_->vkDevice(), compositeFence_, nullptr);
        compositeFence_ = VK_NULL_HANDLE;
    }
    if (acquireSemaphore_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(device_->vkDevice(), acquireSemaphore_, nullptr);
        acquireSemaphore_ = VK_NULL_HANDLE;
    }
    if (compositeSemaphore_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(device_->vkDevice(), compositeSemaphore_, nullptr);
        compositeSemaphore_ = VK_NULL_HANDLE;
    }
}

void PresentThread::start() {
    if (running_) return;
    running_ = true;
    paused_ = false;
    thread_ = std::thread(&PresentThread::threadFunc, this);
    ptCout() << "started" << std::endl;
}

void PresentThread::pause() {
    if (!running_ || paused_) return;
    pauseAcknowledged_ = false;
    paused_ = true;
    // Wait until the thread actually enters its pause wait (so it is NOT holding
    // recreateMtx_ inside a blocking call like AcquireSync or vkWaitForFences).
    {
        std::unique_lock<std::mutex> lk(pauseMtx_);
        pauseCv_.wait(lk, [this] { return pauseAcknowledged_.load() || !running_; });
    }
    ptCout() << "paused (acknowledged)" << std::endl;
}

void PresentThread::resume() {
    if (!running_ || !paused_) return;
    paused_ = false;
    pauseCv_.notify_one();
    ptCout() << "resumed" << std::endl;
}

void PresentThread::stop() {
    if (!running_) return;
    running_ = false;
    paused_ = false;
    pauseCv_.notify_all();  // wake both pause() caller and thread's pause wait
    // Wake up waitAndConsume with a dummy publish
    if (ring_) {
        ring_->publish(nullptr, nullptr, VK_NULL_HANDLE, false, 0.0f);
    }
    if (thread_.joinable()) thread_.join();
    ptCout() << "stopped" << std::endl;
}

void PresentThread::onSwapchainRecreate() {
    // Called while thread is paused. Recreate Vulkan resources for new swapchain.
    destroyVulkanResources();
    createVulkanResources();
}

void PresentThread::setOverlayMode(bool enabled, OverlayCompositor *compositor) {
    // Store compositor before enabling the flag (reader checks flag first, then deref).
    // When disabling, clear flag first to prevent use-after-free.
    if (enabled) {
        overlayCompositor_.store(compositor);
        overlayMode_.store(true);
    } else {
        overlayMode_.store(false);
        overlayCompositor_.store(nullptr);
    }
    if (enabled) {
        ptCout() << "overlay mode enabled" << std::endl;
    } else {
        ptCout() << "overlay mode disabled" << std::endl;
    }
}

void PresentThread::threadFunc() {
    while (running_) {
        // Check pause -- signal acknowledgment then block until resumed
        if (paused_.load()) {
            std::unique_lock<std::mutex> lk(pauseMtx_);
            pauseAcknowledged_ = true;
            pauseCv_.notify_all();  // wake pause() caller
            pauseCv_.wait(lk, [this] { return !paused_ || !running_; });
        }
        if (!running_) break;

        // ═══════════ Overlay mode: render-coupled ═══════════
        // One present per render frame — no free-running re-presentation.
        // The render thread publishes to the ring after GPU submit; we wait
        // for the slot, wait for the GPU fence, then present once.
        auto *compositor = overlayCompositor_.load();
        if (overlayMode_.load() && compositor) {
            // Block until render thread publishes a frame (16ms timeout to check pause/stop)
            FrameSlot *slot = ring_->waitAndConsume(std::chrono::milliseconds(16));
            if (!slot || !running_ || paused_.load()) continue;
            if (slot->renderDoneFence == VK_NULL_HANDLE) continue;  // shutdown sentinel

            // Wait for GPU to finish rendering this frame
            vkWaitForFences(device_->vkDevice(), 1,
                            &slot->renderDoneFence, VK_TRUE, UINT64_MAX);

            // Present the premultiplied overlay via D3D11/DComp
            compositor->present();

            // Error recovery: if compositor deactivated (DXGI device lost),
            // trigger swapchain recreate to rebuild it.
            if (!compositor->isActive()) {
                Renderer::options.needRecreate = true;
            }
            continue;
        }

        // ═══════════ Vulkan swapchain mode: render-rate coupled ═══════════
        FrameSlot *slot = ring_->waitAndConsume();
        if (!slot || !running_) continue;
        if (!slot->worldImage && !slot->overlayImage) continue;

        auto fw = framework_.lock();
        if (!fw || !fw->isRunning()) continue;

        std::unique_lock<std::recursive_mutex> recreateLk(fw->recreateMtx());
        if (!running_ || paused_) continue;

        auto swapchain = fw->swapchain();
        if (!swapchain) continue;

        // CPU-wait for render GPU to complete
        if (slot->renderDoneFence != VK_NULL_HANDLE) {
            vkWaitForFences(device_->vkDevice(), 1, &slot->renderDoneFence, VK_TRUE, UINT64_MAX);
        }

        // Acquire swapchain image
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(
            device_->vkDevice(), swapchain->vkSwapchain(),
            UINT64_MAX, acquireSemaphore_, VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            Renderer::options.needRecreate = true;
            continue;
        }
        if (result != VK_SUCCESS) continue;

        // Wait for previous composite to finish
        vkWaitForFences(device_->vkDevice(), 1, &compositeFence_, VK_TRUE, UINT64_MAX);
        vkResetFences(device_->vkDevice(), 1, &compositeFence_);

        if (imageIndex >= commandBuffers_.size()) continue;

        // Record composite pass
        auto cmd = commandBuffers_[imageIndex];
        cmd->begin();

        auto swapchainImage = swapchain->swapchainImages()[imageIndex];
        auto compositePass = fw->pipeline()->hdrCompositePass();

        if (compositePass) {
            auto mode = slot->hdrOutput
                            ? HdrCompositePass::OutputMode::Hdr10
                            : HdrCompositePass::OutputMode::Sdr;
            compositePass->record(
                cmd, imageIndex, mode,
                slot->hdrUiBrightnessNits,
                slot->worldImage, slot->overlayImage,
                swapchainImage,
                fw->physicalDevice()->mainQueueIndex());
        }

        cmd->end();

        // Submit composite -- wait on acquire semaphore, signal composite semaphore
        VkSemaphore waitSems[] = {acquireSemaphore_};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore signalSems[] = {compositeSemaphore_};

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSems;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd->vkCommandBuffer();
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSems;

        // Submit to secondary (present) queue
        // NOTE: assumes secondary queue supports presentation (same family as main on NVIDIA).
        // TODO: add queue family presentation capability check for multi-family GPUs.
        VkResult submitResult = vkQueueSubmit(
            device_->secondaryQueue(), 1, &submitInfo, compositeFence_);
        if (submitResult != VK_SUCCESS) {
            ptCerr() << "composite submit failed: " << submitResult << std::endl;
            continue;
        }

        // Present
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSems;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain->vkSwapchain();
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(device_->secondaryQueue(), &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            Renderer::options.needRecreate = true;
        }
    }
}
