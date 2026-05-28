#include "core/render/render_framework.hpp"

#include "common/shared.hpp"
#include "core/render/gpu_diagnostics.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/entities.hpp"
#include "core/render/hdr_composite_pass.hpp"
#include "core/render/modules/ui_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/renderer.hpp"
#include "core/render/streamline_context.hpp"
#include "core/render/textures.hpp"
#include "core/render/world.hpp"

#include "core/render/crash_ring_buffer.hpp"
#include "core/render/radiance_logger.hpp"
#include "core/vulkan/vma.hpp"
#include "core/render/modules/world/frame_gen/frame_gen_manager.hpp"
#include "core/render/frame_slot_ring.hpp"
#include "core/render/overlay_compositor.hpp"
#include "core/render/present_thread.hpp"
#include "core/vulkan/debug_utils.hpp"

#include <iostream>
#include <random>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cstdarg>
#include <thread>

// ── Diagnostic file logger (same pattern as overlay_diag.log) ──
static std::ofstream sRenderDiag;

// ── Full-frame profiling (logs every 120 frames to frame_timing.log) ──
// Fence timeout: prevents infinite hang on GPU TDR/driver crash.
static constexpr uint64_t kFenceTimeoutNs = 5000000000ULL; // 5 seconds

namespace FrameTiming {
    using Clock = std::chrono::steady_clock;
    static Clock::time_point frameStart;
    static float accReflexSleep = 0, accAcquireImage = 0, accFenceWait = 0;
    static float accJavaGap = 0; // time between acquireContext return and submitCommand call
    static float accUpload = 0, accTexFlush = 0, accWorldRender = 0, accUIEnd = 0;
    static float accFuse = 0, accCmdEnd = 0, accQueueSubmit = 0;
    static float accPresent = 0, accFrameTotal = 0;
	static float accWallClock = 0; // frame-to-frame wall-clock time (what RTSS measures)
static float accLimiterSleep = 0; // native FPS limiter sleep time
    static uint32_t frameCount = 0;
    static Clock::time_point acquireEnd; // set at end of acquireContext
    static Clock::time_point lastFrameEnd;
    static bool hasLastFrame = false;

	// Per-frame CSV accumulators (written every frame when perFrameTiming is on)
	static float pfReflexSleep = 0, pfAcquireImage = 0, pfFenceWait = 0;
	static float pfJavaGap = 0;
	static float pfUpload = 0, pfTexFlush = 0, pfWorldRender = 0, pfUIEnd = 0;
	static float pfFuse = 0, pfCmdEnd = 0, pfQueueSubmit = 0;
	static float pfPresent = 0, pfFrameTotal = 0;
	static float pfWallClock = 0;
static float pfLimiterSleep = 0;
	static float pfGpu = 0;
static uint64_t pfUploadBytes = 0, pfUploadRegions = 0;
	static uint64_t pfFrameIndex = 0;
	static std::ofstream sJavaStutterLog;
	static bool sJavaStutterHeaderWritten = false;
	static float sJavaStutterMedian = 0;
	static constexpr size_t JAVA_STUFFER_RING_SIZE = 60;
	static float sJavaStutterRing[JAVA_STUFFER_RING_SIZE] = {};
	static size_t sJavaStutterRingIdx = 0;
	static size_t sJavaStutterRingCount = 0;
	static bool pfHeaderWritten = false;

    static float ms(Clock::time_point t0) {
        return std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
    }

	static bool perFrameTimingEnabled = true;  // Toggle via Options or config
static void logPerFrame() {
    if (!perFrameTimingEnabled) return;
		
		std::ofstream f("C:/RadSER/results/per_frame_timing.log", std::ios::app);
		if (!f.is_open()) return;
		if (!pfHeaderWritten) {
			f << "frame,reflexSleep,acquireImg,fenceWait,JAVA,upload,texFlush,worldRender,uiEnd,fuse,cmdEnd,queueSubmit,present,TOTAL,wallClock,gpu,limiterSleep,uploadBytes,uploadRegions\n";
			pfHeaderWritten = true;
		}
		float total = pfReflexSleep + pfAcquireImage + pfFenceWait + pfJavaGap +
			pfUpload + pfTexFlush + pfWorldRender + pfUIEnd + pfFuse +
			pfCmdEnd + pfQueueSubmit + pfPresent;
		f << pfFrameIndex << ","
			<< pfReflexSleep << ","
			<< pfAcquireImage << ","
			<< pfFenceWait << ","
			<< pfJavaGap << ","
			<< pfUpload << ","
			<< pfTexFlush << ","
			<< pfWorldRender << ","
			<< pfUIEnd << ","
			<< pfFuse << ","
			<< pfCmdEnd << ","
			<< pfQueueSubmit << ","
			<< pfPresent << ","
			<< total << ","
			<< pfWallClock << ","
			<< pfGpu << "," << pfLimiterSleep << "," << pfUploadBytes << "," << pfUploadRegions << "\n";
		pfFrameIndex++;

		// ── JAVA spike detection ──
		// Track JAVA time (acquireContext→submitCommand gap) and log spikes
		// that exceed 2x the rolling median. This catches texture-pack tick work.
		{
			sJavaStutterRing[sJavaStutterRingIdx] = pfJavaGap;
			sJavaStutterRingIdx = (sJavaStutterRingIdx + 1) % JAVA_STUFFER_RING_SIZE;
			sJavaStutterRingCount = std::min(sJavaStutterRingCount + 1, JAVA_STUFFER_RING_SIZE);
			// Compute rolling median
			if (sJavaStutterRingCount >= 10) {
				float sorted[JAVA_STUFFER_RING_SIZE];
				size_t n = sJavaStutterRingCount;
				for (size_t i = 0; i < n; i++) sorted[i] = sJavaStutterRing[i];
				std::sort(sorted, sorted + n);
				sJavaStutterMedian = sorted[n / 2];
				// Log spike if JAVA > 2x median and > 1ms
				if (pfJavaGap > std::max(sJavaStutterMedian * 2.0f, 1.0f) && pfJavaGap > 0.5f) {
					if (!sJavaStutterLog.is_open()) {
						sJavaStutterLog.open("C:/RadSER/results/java_stutter.log", std::ios::app);
						if (!sJavaStutterHeaderWritten) {
							sJavaStutterLog << "frame,javaMs,medianMs,ratio,texFlushMs,limiterMs" << std::endl;
							sJavaStutterHeaderWritten = true;
						}
					}
					sJavaStutterLog << pfFrameIndex << "," << pfJavaGap << "," 
						<< sJavaStutterMedian << "," << (pfJavaGap / std::max(sJavaStutterMedian, 0.001f))
						<< "," << pfTexFlush << "," << pfLimiterSleep << std::endl;
				}
			}
		}
		pfReflexSleep = pfAcquireImage = pfFenceWait = 0;
		pfJavaGap = 0;
		accTexFlush += pfTexFlush;
	pfUpload = pfTexFlush = pfWorldRender = pfUIEnd = 0;
		pfFuse = pfCmdEnd = pfQueueSubmit = 0;
		pfPresent = pfFrameTotal = pfWallClock = pfGpu = pfLimiterSleep = 0; pfUploadBytes = pfUploadRegions = 0;
	}

    static void log() {
		logPerFrame();
        if (++frameCount < 120) return;
        float n = static_cast<float>(frameCount);
        float total = (accReflexSleep + accAcquireImage + accJavaGap + accUpload +
                       accWorldRender + accUIEnd + accFuse + accCmdEnd +
                       accQueueSubmit + accPresent) / n;
        std::ofstream f("C:/RadSER/results/frame_timing.log", std::ios::app);
        if (f.is_open()) {
            f << "[FRAME] reflexSleep=" << (accReflexSleep / n)
              << " acquireImg=" << (accAcquireImage / n)
              << " fenceWait=" << (accFenceWait / n)
              << " JAVA=" << (accJavaGap / n)
              << " upload=" << (accUpload / n)
              << " worldRender=" << (accWorldRender / n)
              << " uiEnd=" << (accUIEnd / n)
              << " fuse=" << (accFuse / n)
              << " cmdEnd=" << (accCmdEnd / n)
              << " queueSubmit=" << (accQueueSubmit / n)
              << " present=" << (accPresent / n)
              << " TOTAL=" << total
              << " wallClock=" << accWallClock / n
			<< " gpu=" << accFrameTotal / n
<< " limiterSleep=" << accLimiterSleep / n << "\n";
        }
        std::cout << "[FRAME] JAVA=" << (accJavaGap / n)
                  << " worldRender=" << (accWorldRender / n)
                  << " fenceWait=" << (accFenceWait / n)
                  << " present=" << (accPresent / n)
                  << " TOTAL=" << total << "ms"
<< " limiterSleep=" << (accLimiterSleep / n) << "ms" << " maxFps=" << Renderer::options.maxFps << std::endl;
        accReflexSleep = accAcquireImage = accFenceWait = 0;
        accJavaGap = 0;
        accUpload = accTexFlush = accWorldRender = accUIEnd = 0;
        accFuse = accCmdEnd = accQueueSubmit = 0;
        accPresent = accFrameTotal = accWallClock = accLimiterSleep = 0;
        frameCount = 0;
    }
}

void FrameTiming_addTexFlush(float ms) {
	FrameTiming::pfTexFlush += ms;
}

void renderDiag(const char *fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (sRenderDiag.is_open()) {
        SYSTEMTIME st;
        GetLocalTime(&st);
        char ts[32];
        snprintf(ts, sizeof(ts), "%02d:%02d:%02d.%03d",
                 st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
        sRenderDiag << ts << " " << buf << std::endl;
        sRenderDiag.flush();
    }
}

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
    if (!f || !f->isRunning()) return;

    auto mainQueueIndex = physicalDevice->mainQueueIndex();
    auto pipelineContext = f->pipeline_->acquirePipelineContext(shared_from_this());

    bool hdrOutputActive = Renderer::options.hdrEnabled && swapchain->isHDR();
    bool hasWorldOutput = pipelineContext->worldPipelineContext && pipelineContext->worldPipelineContext->outputImage;
    auto worldOutput = hasWorldOutput ? pipelineContext->worldPipelineContext->outputImage : nullptr;

    auto overlayOutput = pipelineContext->uiModuleContext ? pipelineContext->uiModuleContext->overlayDrawColorImage : nullptr;

    // When overlay compositor is active, composite world-only to swapchain
    // (UI composites on its own surface). Currently disabled (wantOverlay=false).
    auto overlayCompositor = f->overlayCompositor_.get();
    bool overlayActive = FrameGenManager::isActive() &&
                         overlayCompositor && overlayCompositor->isActive();
    auto compositeOverlay = overlayActive ? nullptr : overlayOutput;

    bool canComposite = f->pipeline_->hdrCompositePass() && (compositeOverlay || overlayActive);

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
            compositeOverlay,
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
            compositeOverlay,
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

    // Record premultiply pass when overlay compositor is active.
    if (overlayActive && overlayOutput) {
        overlayCompositor->recordPremultiply(fuseCommandBuffer, overlayOutput, mainQueueIndex);
    }
}

Framework::Framework() {}

void Framework::init(GLFWwindow *window) {
    instance_ = vk::Instance::create();
    window_ = vk::Window::create(instance_, window);
    physicalDevice_ = vk::PhysicalDevice::create(instance_, window_);
    device_ = vk::Device::create(instance_, window_, physicalDevice_);
    device_->createTimelineSemaphores();
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

    // Initialize DLSS-G frame generation
    FrameGenManager::init();

    // Initialize GPU profiler
    Renderer::gpuProfiler.init(device_, physicalDevice_, 16, imageCount);

    // Initialize decoupled presentation (PresentThread + FrameSlotRing)
    frameSlotRing_ = std::make_unique<FrameSlotRing>();
    presentThread_ = std::make_unique<PresentThread>();
    presentThread_->init(shared_from_this(), frameSlotRing_.get());

    // Open render diagnostic log
    {
        auto logDir = Renderer::folderPath / "logs";
        std::filesystem::create_directories(logDir);
        sRenderDiag.open((logDir / "render_diag.log").string(), std::ios::trunc);
        renderDiag("init complete, contexts=%u", (unsigned)contexts_.size());
    }

    // Decoupled present is initialized but NOT started by default.
    // The secondary queue may not support graphics/presentation on all GPUs.
    // TODO: Add queue family capability check before enabling.
    // To enable: call enableDecoupledPresent() after verifying queue support.
}

Framework::~Framework() {
#ifdef DEBUG
    std::cout << "[Framework] framework deconstructed" << std::endl;
#endif
}

void Framework::acquireContext() {
    if (!running_) return;

    // ── Full-frame profiling: measure frame-to-frame time ──
    auto ftNow = FrameTiming::Clock::now();
    if (FrameTiming::hasLastFrame) {
        FrameTiming::accFrameTotal += std::chrono::duration<float, std::milli>(
            ftNow - FrameTiming::lastFrameEnd).count();
        FrameTiming::pfFrameTotal += std::chrono::duration<float, std::milli>(
            ftNow - FrameTiming::lastFrameEnd).count();
    }
    FrameTiming::frameStart = ftNow;
    FrameTiming::hasLastFrame = true;

    g_crashRing.advanceFrame();
    GpuDiag::setEnabled(Renderer::options.gpuDiagnostics);
    renderDiag("acquireContext frame=%llu decoupled=%d", g_crashRing.frameCount(), (int)decoupledPresent_);

    if (RadianceLogger::isEnabled()) {
        RadianceLogger::log("Frame", "INFO", "frame=%llu begin", g_crashRing.frameCount());
    }

    // Streamline: advance frame token and sleep at the very top of the frame.
    // Per NVIDIA QA checklist: "slReflexSleep is called regardless of Reflex Low Latency mode state."
    // SL handles the mode internally — always call sleep when Reflex is available.
    //
    // Skip Reflex sleep ONLY when both conditions are true:
    //   1. Not rendering world (menu/loading) — UI-only frames are trivially fast
    //   2. DLSS-G is not active — Reflex sleep coordinates with FG's frame cadence;
    //      skipping it while FG is on would overwhelm the SL interposer with 300+ FPS
    //      of untagged presents. Once FG activates (world entered), always respect sleep.
    auto ftReflex0 = FrameTiming::Clock::now();
    bool reflexSlept = false;
    if (StreamlineContext::isAvailable()) {
        StreamlineContext::advanceFrame();
        bool shouldSleep = Renderer::instance().world()->shouldRender() ||
            FrameGenManager::isActive();
        if (StreamlineContext::isReflexAvailable() && shouldSleep) {
            StreamlineContext::reflexSleep();
            reflexSlept = true;
        }
    }
    FrameTiming::accReflexSleep += FrameTiming::ms(ftReflex0);
    FrameTiming::pfReflexSleep += FrameTiming::ms(ftReflex0);

    // ── Native FPS limiter ──
    // When Reflex sleep was NOT called (menu/loading screens, or Reflex disabled),
    // and VSync is off (IMMEDIATE present mode), the frame loop runs uncapped.
    // This causes menu stutter because texture animation ticks fire every frame
    // and Java-side work between acquire/submit becomes the dominant cost.
    // The native limiter sleeps to hit the target frame interval from Options.maxFps.
    // 0 = unlimited (no sleep). When FG is active, FG handles pacing internally.
    auto ftLimiter0 = FrameTiming::Clock::now();
    if (!reflexSlept && !FrameGenManager::isActive() && !Renderer::options.vsync) {
        uint32_t maxFps = Renderer::options.maxFps;
        if (maxFps > 0 && maxFps < 1000000 && FrameTiming::hasLastFrame) {
            float targetMs = 1000.0f / static_cast<float>(maxFps);
            float elapsedMs = std::chrono::duration<float, std::milli>(
                FrameTiming::Clock::now() - FrameTiming::lastFrameEnd).count();
            if (elapsedMs < targetMs) {
                float sleepMs = targetMs - elapsedMs;
                // Spin-wait for the last ~0.5ms for accuracy, sleep for the rest
                if (sleepMs > 1.0f) {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(static_cast<int64_t>((sleepMs - 0.5f) * 1000.0f)));
                }
                // Spin-wait the remainder
                auto deadline = FrameTiming::lastFrameEnd +
                    std::chrono::duration<float, std::milli>(targetMs);
                while (FrameTiming::Clock::now() < deadline) {
                    std::this_thread::yield();
                }
            }
        }
    }
    float limiterMs = FrameTiming::ms(ftLimiter0);
    FrameTiming::accLimiterSleep += limiterMs;
    FrameTiming::pfLimiterSleep += limiterMs;

    std::shared_ptr<FrameworkContext> lastContext;
    if (currentContext_) lastContext = currentContext_;

    uint32_t imageIndex;
    std::shared_ptr<vk::Semaphore> imageAcquiredSemaphore;

    auto ftAcquire0 = FrameTiming::Clock::now();
    if (decoupledPresent_) {
        // Decoupled: round-robin through contexts without swapchain acquire.
        // PresentThread owns swapchain acquire/present.
        decoupledFrameIndex_ = (decoupledFrameIndex_ + 1) % static_cast<uint32_t>(contexts_.size());
        imageIndex = decoupledFrameIndex_;

        std::shared_ptr<vk::Fence> fence = contexts_[imageIndex]->commandFinishedFence;
        g_crashRing.record("waitFence");
        auto ftFence0 = FrameTiming::Clock::now();
        VkResult fenceResult = vkWaitForFences(device_->vkDevice(), 1, &fence->vkFence(), true, kFenceTimeoutNs);
        FrameTiming::accFenceWait += FrameTiming::ms(ftFence0);
        FrameTiming::pfFenceWait += FrameTiming::ms(ftFence0);
        if (fenceResult == VK_TIMEOUT) {
            renderDiag("vkWaitForFences timed out (5s) — GPU may be hung, attempting recreate");
                std::cerr << "[FRAMEWORK] vkWaitForFences timed out (5s) — GPU may be hung" << std::endl;
            Renderer::options.needRecreate = true;
            return;
        }
        if (fenceResult == VK_TIMEOUT) {
            renderDiag("vkWaitForFences timed out (5s) — GPU may be hung, attempting recreate");
            std::cerr << "[FRAMEWORK] vkWaitForFences timed out (5s) — GPU may be hung" << std::endl;
            Renderer::options.needRecreate = true;
            return;
        }
        if (fenceResult != VK_SUCCESS) {
            waitDeviceIdle();
            crashExitWithQueue(fenceResult, "vkWaitForFences failed (decoupled)", device_->mainVkQueue());
        }
    } else {
        // Standard: acquire swapchain image
        imageAcquiredSemaphore = acquireSemaphore();
        g_crashRing.record("acquireImage");
        renderDiag("  vkAcquireNextImageKHR...");
        VkResult result = vkAcquireNextImageKHR(device_->vkDevice(), swapchain_->vkSwapchain(), UINT64_MAX,
                                       imageAcquiredSemaphore->vkSemaphore(), VK_NULL_HANDLE, &imageIndex);
        renderDiag("  vkAcquireNextImageKHR -> %d idx=%u", result, imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recycleSemaphore(imageAcquiredSemaphore);
            recreate();
            return;
        } else if (result != VK_SUCCESS) {
            recycleSemaphore(imageAcquiredSemaphore);
            waitDeviceIdle();
            crashExit(result, "vkAcquireNextImageKHR failed");
        }

        std::shared_ptr<vk::Fence> fence = contexts_[imageIndex]->commandFinishedFence;
        g_crashRing.record("waitFence");
        auto ftFence0 = FrameTiming::Clock::now();
        VkResult fenceResult = vkWaitForFences(device_->vkDevice(), 1, &fence->vkFence(), true, kFenceTimeoutNs);
        FrameTiming::accFenceWait += FrameTiming::ms(ftFence0);
        FrameTiming::pfFenceWait += FrameTiming::ms(ftFence0);
        if (fenceResult == VK_TIMEOUT) {
            renderDiag("vkWaitForFences timed out (5s) — GPU may be hung, attempting recreate");
            std::cerr << "[FRAMEWORK] vkWaitForFences timed out" << std::endl;
            Renderer::options.needRecreate = true;
            return;
        }
        if (fenceResult != VK_SUCCESS) {
            waitDeviceIdle();
            crashExitWithQueue(fenceResult, "vkWaitForFences failed", device_->mainVkQueue());
        }
    }
    FrameTiming::accAcquireImage += FrameTiming::ms(ftAcquire0);
    FrameTiming::pfAcquireImage += FrameTiming::ms(ftAcquire0);

    currentContextIndex_ = imageIndex;
    currentContext_ = contexts_[imageIndex];
    indexHistory_.push(imageIndex);

    // Read GPU profiler results from this frame's previous submission
    Renderer::gpuProfiler.readResults(imageIndex);
    if (Renderer::gpuProfiler.isEnabled()) {
        FrameTiming::accFrameTotal += Renderer::gpuProfiler.getTotalGpuMs();
        FrameTiming::pfGpu += Renderer::gpuProfiler.getTotalGpuMs();
    }
    if (indexHistory_.size() > swapchain_->imageCount()) indexHistory_.pop();

    if (currentContext_->imageAcquiredSemaphore != VK_NULL_HANDLE) {
        recycleSemaphore(currentContext_->imageAcquiredSemaphore);
        currentContext_->imageAcquiredSemaphore = VK_NULL_HANDLE;
    }
    if (imageAcquiredSemaphore) {
        currentContext_->imageAcquiredSemaphore = imageAcquiredSemaphore;
    }

    // PCL: mark simulation start AFTER blocking sync waits (acquire + fence) complete.
    // Placing it before would inflate simulation time with driver stalls,
    // giving Reflex inaccurate timing data for its sleep calculations.
#ifdef _WIN32
    StreamlineContext::pclSetMarker(sl::PCLMarker::eSimulationStart);
#endif

    currentContext_->uploadCommandBuffer->begin();
    currentContext_->worldCommandBuffer->begin();
    currentContext_->overlayCommandBuffer->begin();
    currentContext_->fuseCommandBuffer->begin();

    auto pipelineContext = pipeline_->acquirePipelineContext(currentContext_);
    std::shared_ptr<UIModuleContext> lastUIContext =
        lastContext == nullptr ? nullptr : pipeline_->acquirePipelineContext(lastContext)->uiModuleContext;

    pipelineContext->uiModuleContext->begin(lastUIContext);

    // Cross-queue GC sync: wait for ALL secondary queue (BLAS thread) work to complete
    // before freeing resources. Without this, GC can free BLAS backing buffers while the
    // secondary queue's AS build shader still references them (WRITE_AFTER_DESTROY crash).
    {
        auto blasSem = device_->blasSemaphore();
        auto world = Renderer::instance().world();
        if (blasSem && world && world->chunks() && world->chunks()->chunkBuildScheduler()) {
            uint64_t lastSubmitted = world->chunks()->chunkBuildScheduler()->lastSubmittedTimelineValue();
            if (lastSubmitted > 0) {
                blasSem->waitValue(lastSubmitted);
            }
        }
    }

    gc_->clear();
    Renderer::instance().buffers()->resetFrame();
    // Capture per-frame upload diagnostics before resetFrame clears them
      FrameTiming::pfUploadBytes += Renderer::instance().textures()->uploadBytes();
      FrameTiming::pfUploadRegions += Renderer::instance().textures()->uploadRegions();
      Renderer::instance().textures()->resetFrame();
    Renderer::instance().world()->resetFrame();
    Renderer::instance().world()->chunks()->resetFrame();
    Renderer::instance().world()->entities()->resetFrame();

    FrameTiming::acquireEnd = FrameTiming::Clock::now();

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

    // Measure Java-side work between acquireContext() return and submitCommand() call
    FrameTiming::accJavaGap += std::chrono::duration<float, std::milli>(
        FrameTiming::Clock::now() - FrameTiming::acquireEnd).count();
    FrameTiming::pfJavaGap += std::chrono::duration<float, std::milli>(
        FrameTiming::Clock::now() - FrameTiming::acquireEnd).count();

    renderDiag("submitCommand shouldRender=%d overlayActive=%d",
            (int)Renderer::instance().world()->shouldRender(),
            (int)(overlayCompositor_ && overlayCompositor_->isActive()));
  // Log VRAM state on every shouldRender=1 transition (first real render frame)
  static bool sLoggedFirstRender = false;
  bool shouldRender = Renderer::instance().world()->shouldRender();
  if (shouldRender && !sLoggedFirstRender) {
    sLoggedFirstRender = true;
    auto vma = vma_;
    if (vma) {
      VmaTotalStatistics vts{};
      vmaCalculateStatistics(vma->allocator(), &vts);
      VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
      vmaGetHeapBudgets(vma->allocator(), budgets);
      uint64_t totalBudget = 0, totalUsage = 0;
      for (uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; ++i) { totalBudget += budgets[i].budget; totalUsage += budgets[i].usage; }
      float usedGB = (float)totalUsage / (1024.f * 1024.f * 1024.f);
      float budgetGB = (float)totalBudget / (1024.f * 1024.f * 1024.f);
      float allocGB = (float)vts.total.statistics.blockBytes / (1024.f * 1024.f * 1024.f);
      std::cerr << "[Radiance] FIRST RENDER VRAM: usage=" << usedGB << "GB budget=" << budgetGB
                << "GB allocated=" << allocGB << "GB allocs=" << vts.total.statistics.allocationCount
                << " blocks=" << vts.total.statistics.blockCount << std::endl;
    }
  }
#ifdef _WIN32
    StreamlineContext::pclSetMarker(sl::PCLMarker::eSimulationEnd);
#endif

    Renderer::instance().framework()->safeAcquireCurrentContext(); // ensure context is non nullptr

    // Granular GPU checkpoints on the upload command buffer for crash diagnosis
    auto ftUpload0 = FrameTiming::Clock::now();
    currentContext_->uploadCommandBuffer->beginLabel("Upload", 0.1f, 0.55f, 1.0f);
    currentContext_->uploadCommandBuffer->beginLabel("Upload:Textures", 0.1f, 0.6f, 1.0f);
    GpuDiag::checkpoint(currentContext_->uploadCommandBuffer->vkCommandBuffer(), GpuDiag::UPLOAD_TEX_BEGIN);
    Renderer::instance().textures()->performQueuedUpload();
    GpuDiag::checkpoint(currentContext_->uploadCommandBuffer->vkCommandBuffer(), GpuDiag::UPLOAD_TEX_END);
    currentContext_->uploadCommandBuffer->endLabel();
    currentContext_->uploadCommandBuffer->beginLabel("Upload:Buffers", 0.1f, 0.75f, 1.0f);
    GpuDiag::checkpoint(currentContext_->uploadCommandBuffer->vkCommandBuffer(), GpuDiag::UPLOAD_BUF_BEGIN);
    Renderer::instance().buffers()->performQueuedUpload();
    GpuDiag::checkpoint(currentContext_->uploadCommandBuffer->vkCommandBuffer(), GpuDiag::UPLOAD_BUF_END);
    Renderer::instance().buffers()->buildAndUploadOverlayUniformBuffer();
    currentContext_->uploadCommandBuffer->endLabel();
    currentContext_->uploadCommandBuffer->endLabel();
    FrameTiming::accUpload += FrameTiming::ms(ftUpload0);
    FrameTiming::pfUpload += FrameTiming::ms(ftUpload0);

    auto ftWorld0 = FrameTiming::Clock::now();
    auto pipelineContext = pipeline_->acquirePipelineContext(currentContext_);
    if (Renderer::instance().world()->shouldRender() && pipelineContext->worldPipelineContext) {
        Renderer::gpuProfiler.beginFrame(
            currentContext_->worldCommandBuffer->vkCommandBuffer(), currentContextIndex_);
        try {
            pipelineContext->worldPipelineContext->render();
        } catch (const std::exception& e) {
            renderDiag("FATAL: WorldPipelineContext::render() threw: %s", e.what());
            renderDiag("FATAL: render() threw: %s", e.what());
            throw; // re-throw so JVM still sees the failure
        } catch (...) {
            renderDiag("FATAL: WorldPipelineContext::render() threw unknown exception");
            renderDiag("FATAL: render() threw unknown exception");
            throw;
        }
        Renderer::gpuProfiler.endFrame(
            currentContext_->worldCommandBuffer->vkCommandBuffer());
    }
    FrameTiming::accWorldRender += FrameTiming::ms(ftWorld0);
    FrameTiming::pfWorldRender += FrameTiming::ms(ftWorld0);

    auto ftUI0 = FrameTiming::Clock::now();
    currentContext_->overlayCommandBuffer->beginLabel("UI", 0.7f, 0.4f, 1.0f);
    pipelineContext->uiModuleContext->end();
    currentContext_->overlayCommandBuffer->endLabel();
    FrameTiming::accUIEnd += FrameTiming::ms(ftUI0);
    FrameTiming::pfUIEnd += FrameTiming::ms(ftUI0);

    // Composite world+UI to swapchain (standard mode only).
    // In decoupled mode, PresentThread handles compositing.
    auto ftFuse0 = FrameTiming::Clock::now();
    if (!decoupledPresent_) {
        currentContext_->fuseCommandBuffer->beginLabel("Fuse:WorldUIToSwapchain", 0.1f, 0.9f, 0.35f);
        currentContext_->fuseFinal();
        currentContext_->fuseCommandBuffer->endLabel();
    }

    // Tag resources for DLSS-G frame generation (after composite).
    // Only HUDless is tagged — DLSS-G diffs it against the backbuffer to find UI.
    {
        auto worldOutput = pipelineContext->worldPipelineContext
                             ? pipelineContext->worldPipelineContext->outputImage : nullptr;
        FrameGenManager::tagFrame(currentContext_, worldOutput);
    }
    FrameTiming::accFuse += FrameTiming::ms(ftFuse0);
    FrameTiming::pfFuse += FrameTiming::ms(ftFuse0);

    auto ftCmdEnd0 = FrameTiming::Clock::now();
    currentContext_->uploadCommandBuffer->end();
    currentContext_->worldCommandBuffer->end();
    currentContext_->overlayCommandBuffer->end();
    currentContext_->fuseCommandBuffer->end();
    FrameTiming::accCmdEnd += FrameTiming::ms(ftCmdEnd0);
    FrameTiming::pfCmdEnd += FrameTiming::ms(ftCmdEnd0);

    // Standard mode: wait on acquire semaphore, signal processed semaphore.
    // Decoupled mode: no semaphores — fence handles sync with PresentThread.
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkPipelineStageFlags> waitStageMasks;
    std::vector<uint64_t> waitValues;
    std::vector<VkSemaphore> signalSemaphores;
    std::vector<uint64_t> signalValues;
    if (!decoupledPresent_) {
        waitSemaphores.push_back(currentContext_->imageAcquiredSemaphore->vkSemaphore());
        waitStageMasks.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        waitValues.push_back(0);  // binary semaphore — value ignored
        signalSemaphores.push_back(currentContext_->commandProcessedSemaphore->vkSemaphore());
        signalValues.push_back(0);  // binary semaphore — value ignored
    }

    // Cross-queue sync: make the main queue wait on the BLAS timeline semaphore.
    // The BLAS thread builds chunk BLASes on the secondary (async compute) queue.
    // CPU-side timeline polling confirms completion, but the Vulkan spec requires a
    // semaphore wait to establish GPU memory visibility between queue families.
    // The value is already reached, so the GPU wait is a no-op — but it ensures
    // the main queue engine sees all secondary queue writes (BLAS BVH data).
    auto blasSem = device_->blasSemaphore();
    if (blasSem) {
        uint64_t blasTimelineVal = blasSem->getValue();
        if (blasTimelineVal > 0) {
            waitSemaphores.push_back(blasSem->vkSemaphore());
            waitStageMasks.push_back(VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
            waitValues.push_back(blasTimelineVal);
        }
    }

    std::vector<VkCommandBuffer> commandbuffers = {
        currentContext_->uploadCommandBuffer->vkCommandBuffer(),
        currentContext_->worldCommandBuffer->vkCommandBuffer(),
        currentContext_->overlayCommandBuffer->vkCommandBuffer(),
        currentContext_->fuseCommandBuffer->vkCommandBuffer(),
    };

    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.waitSemaphoreValueCount = static_cast<uint32_t>(waitValues.size());
    timelineInfo.pWaitSemaphoreValues = waitValues.data();
    timelineInfo.signalSemaphoreValueCount = static_cast<uint32_t>(signalValues.size());
    timelineInfo.pSignalSemaphoreValues = signalValues.data();

    VkSubmitInfo vkSubmitInfo = {};
    vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo.pNext = &timelineInfo;
    vkSubmitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
    vkSubmitInfo.pWaitSemaphores = waitSemaphores.data();
    vkSubmitInfo.pWaitDstStageMask = waitStageMasks.data();
    vkSubmitInfo.commandBufferCount = static_cast<uint32_t>(commandbuffers.size());
    vkSubmitInfo.pCommandBuffers = commandbuffers.data();
    vkSubmitInfo.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size());
    vkSubmitInfo.pSignalSemaphores = signalSemaphores.data();

    std::shared_ptr<vk::Fence> fence = currentContext_->commandFinishedFence;
    g_crashRing.record("resetFence");
    VkResult resetResult = vkResetFences(device_->vkDevice(), 1, &fence->vkFence());
    if (resetResult != VK_SUCCESS) {
        waitDeviceIdle();
        crashExit(resetResult, "vkResetFences failed");
    }

    // PCL: bracket the GPU submit
#ifdef _WIN32
    StreamlineContext::pclSetMarker(sl::PCLMarker::eRenderSubmitStart);
#endif
    g_crashRing.record("queueSubmit");
    auto ftSubmit0 = FrameTiming::Clock::now();
    VkResult submitResult;
    {
        std::lock_guard<std::mutex> qLock(device_->queueMutex());
        vk::DebugUtils::ScopedQueueLabel queueLabel(
            device_->mainVkQueue(), "Radiance QueueSubmit: Frame", 0.3f, 0.8f, 0.3f);
        submitResult = vkQueueSubmit(device_->mainVkQueue(), 1, &vkSubmitInfo, fence->vkFence());
    }
    FrameTiming::accQueueSubmit += FrameTiming::ms(ftSubmit0);
    FrameTiming::pfQueueSubmit += FrameTiming::ms(ftSubmit0);
#ifdef _WIN32
    StreamlineContext::pclSetMarker(sl::PCLMarker::eRenderSubmitEnd);
#endif
    if (submitResult != VK_SUCCESS) {
        waitDeviceIdle();
        crashExitWithQueue(submitResult, "vkQueueSubmit failed", device_->mainVkQueue());
    }

    // Overlay present on render thread: wait for GPU, then D3D11 copy + DXGI present.
    bool overlayActive = FrameGenManager::isActive() &&
                         overlayCompositor_ && overlayCompositor_->isActive();
    if (overlayActive) {
        VkResult waitResult = vkWaitForFences(device_->vkDevice(), 1, &fence->vkFence(), true, UINT64_MAX);
        if (waitResult != VK_SUCCESS) {
            waitDeviceIdle();
            crashExitWithQueue(waitResult, "vkWaitForFences failed (overlay present)", device_->mainVkQueue());
        }
        overlayCompositor_->present();
        if (!overlayCompositor_->isActive()) {
            Renderer::options.needRecreate = true; // DXGI device lost recovery
            renderDiag("  overlay lost during present, triggering recreate");
        }
        renderDiag("  overlay present on render thread");
    }

    // Publish to FrameSlotRing for PresentThread (decoupled mode only)
    if (decoupledPresent_ && frameSlotRing_) {
        auto worldOut = pipelineContext->worldPipelineContext
                          ? pipelineContext->worldPipelineContext->outputImage : nullptr;
        auto overlayOut = pipelineContext->uiModuleContext
                            ? pipelineContext->uiModuleContext->overlayDrawColorImage : nullptr;
        bool hdr = Renderer::options.hdrEnabled && swapchain_->isHDR();
        frameSlotRing_->publish(
            worldOut, overlayOut,
            currentContext_->commandFinishedFence->vkFence(),
            hdr, Renderer::options.hdrUiBrightnessNits);
        renderDiag("  published to ring (world=%p overlay=%p)", worldOut.get(), overlayOut.get());
    }
    renderDiag("submitCommand done");
}

void Framework::present() {
    if (!running_) return;
    auto ftPresent0 = FrameTiming::Clock::now();
    renderDiag("present decoupled=%d needRecreate=%d", (int)decoupledPresent_, (int)Renderer::options.needRecreate);

    // Decoupled mode: PresentThread handles presenting.
    // Still check for recreation triggers.
    if (decoupledPresent_) {
        if (vk::Window::framebufferResized || Renderer::options.needRecreate || pipeline_->needRecreate) {
            recreate();
        }
        FrameTiming::accPresent += FrameTiming::ms(ftPresent0);
        FrameTiming::pfPresent += FrameTiming::ms(ftPresent0);
        FrameTiming::lastFrameEnd = FrameTiming::Clock::now();
        FrameTiming::log();
        return;
    }

    VkSemaphore waitSem = currentContext_->commandProcessedSemaphore->vkSemaphore();
    VkSwapchainKHR swapchainHandle = swapchain_->vkSwapchain();

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &waitSem;

    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchainHandle;
    presentInfo.pImageIndices = &currentContext_->frameIndex;

    // PCL: bracket the present call
#ifdef _WIN32
    StreamlineContext::pclSetMarker(sl::PCLMarker::ePresentStart);
#endif
    g_crashRing.record("present");
    renderDiag("  vkQueuePresentKHR...");
    VkResult result;
    {
        vk::DebugUtils::ScopedQueueLabel queueLabel(
            device_->mainVkQueue(), "Radiance Present", 0.2f, 0.6f, 1.0f);
        result = vkQueuePresentKHR(device_->mainVkQueue(), &presentInfo);
    }
    renderDiag("  vkQueuePresentKHR -> %d", result);
#ifdef _WIN32
    StreamlineContext::pclSetMarker(sl::PCLMarker::ePresentEnd);
#endif

    if (result == VK_ERROR_OUT_OF_DATE_KHR || vk::Window::framebufferResized ||
        Renderer::options.needRecreate || pipeline_->needRecreate) {
        renderDiag("  triggering recreate (result=%d needRecreate=%d)", result, (int)Renderer::options.needRecreate);
        FrameTiming::accPresent += FrameTiming::ms(ftPresent0);
        FrameTiming::pfPresent += FrameTiming::ms(ftPresent0);
        FrameTiming::lastFrameEnd = FrameTiming::Clock::now();
        FrameTiming::log();
        recreate();
        return;
    }
    // VK_SUBOPTIMAL_KHR — image was presented successfully, just not optimal for
    // the current surface properties. No need to recreate; unnecessary recreates
    // cause frame stutters.
    if (result == VK_SUBOPTIMAL_KHR) {
        static bool loggedOnce = false;
        if (!loggedOnce) {
            renderDiag("  VK_SUBOPTIMAL_KHR from vkQueuePresentKHR — continuing without recreate");
            loggedOnce = true;
        }
    } else if (result != VK_SUCCESS) {
        waitDeviceIdle();
        crashExitWithQueue(result, "vkQueuePresentKHR failed", device_->mainVkQueue());
    }
    FrameTiming::accPresent += FrameTiming::ms(ftPresent0);
    FrameTiming::pfPresent += FrameTiming::ms(ftPresent0);
    FrameTiming::lastFrameEnd = FrameTiming::Clock::now();
    FrameTiming::log();
}

void Framework::recreate() {
    if (!running_) return;
    renderDiag("recreate() START fgEnabled=%d ptRunning=%d ptPaused=%d",
               (int)Renderer::options.frameGenEnabled,
               presentThread_ ? (int)presentThread_->isRunning() : -1,
               presentThread_ ? (int)presentThread_->isPaused() : -1);

    // Pause present thread BEFORE locking recreateMtx_ — pause() is synchronous and
    // waits for the thread to acknowledge, but the thread may be holding recreateMtx_
    // inside its loop body. Pausing first ensures the thread releases recreateMtx_
    // and enters its pause wait before we try to acquire the lock.
    if (presentThread_ && presentThread_->isRunning() && !presentThread_->isPaused()) {
        renderDiag("  pausing PresentThread...");
        presentThread_->pause();
        renderDiag("  PresentThread paused");
    }

    renderDiag("  locking recreateMtx_...");
    std::unique_lock<std::recursive_mutex> lck(Renderer::instance().framework()->recreateMtx());
    renderDiag("  recreateMtx_ locked");

    // Check framebuffer size BEFORE any teardown — if 0x0 (minimized), defer.
    // Must happen before waitRenderQueueIdle/FrameGenManager teardown to avoid
    // leaving the pipeline in a half-torn-down state.
    int width = 0, height = 0;
    GLFW_GetFramebufferSize(window_->window(), &width, &height);
    renderDiag("  framebufferSize: %dx%d", width, height);
    if (width == 0 || height == 0) {
        // Window minimized — block render thread until window is restored.
        renderDiag("  waiting for non-zero framebuffer...");
        while (width == 0 || height == 0) {
            GLFW_PollEvents();
            GLFW_GetFramebufferSize(window_->window(), &width, &height);
            if (width == 0 || height == 0) Sleep(16);
        }
        renderDiag("  framebuffer restored: %dx%d", width, height);
    }

    Renderer::options.needRecreate = false;
    vk::Window::framebufferResized = false;
    pipeline_->needRecreate = false;

    // Pause BLAS thread FIRST to prevent new secondary queue submits during recreate.
    // This closes the race window between queue idle and context destruction.
    renderDiag("  pausing BLAS thread...");
    {
        auto world = Renderer::instance().world();
        if (world && world->chunks() && world->chunks()->chunkBuildScheduler()) {
            world->chunks()->chunkBuildScheduler()->pause();
        }
    }
    renderDiag("  BLAS thread paused, waiting device idle...");
    waitDeviceIdle();
    renderDiag("  device idle");

    // Notify frame gen manager before swapchain teardown
    renderDiag("  FrameGenManager::beforeSwapchainRecreate...");
    FrameGenManager::beforeSwapchainRecreate();
    renderDiag("  FrameGenManager done");

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

    // Notify frame gen manager after swapchain recreation
    FrameGenManager::afterSwapchainRecreate();

    // Recreate present thread resources for new swapchain
    if (presentThread_) {
        presentThread_->onSwapchainRecreate();
    }
    if (frameSlotRing_) {
        frameSlotRing_->reset();
    }

    // Overlay compositor lifecycle: create when FG enabled, destroy when disabled.
    // DComp overlay is NOT used with DLSS-G — UI goes through kBufferTypeUIColorAndAlpha
    // tagging in the swapchain (NVIDIA official path). Overlay compositor forces composed
    // flip which breaks DLSS-G frame pacing. Only create overlay when FG is OFF and
    // decoupled UI at display rate is desired (future MPO path).
    bool wantOverlay = false;  // disabled — DLSS-G uses render-thread UI tagging
    renderDiag("  overlay: want=%d exists=%d compositorActive=%d", (int)wantOverlay,
               (int)(overlayCompositor_ != nullptr),
               (int)(overlayCompositor_ && overlayCompositor_->isActive()));
    renderFrameworkCout() << "recreate: frameGenEnabled=" << wantOverlay
                         << " overlayCompositor=" << (overlayCompositor_ ? "exists" : "null")
                         << " active=" << (overlayCompositor_ && overlayCompositor_->isActive() ? "true" : "false")
                         << std::endl;
    if (wantOverlay && !overlayCompositor_) {
        renderDiag("  creating overlay compositor...");
        renderFrameworkCout() << "creating overlay compositor..." << std::endl;
        overlayCompositor_ = std::make_unique<OverlayCompositor>();
        if (!overlayCompositor_->init(shared_from_this())) {
            renderFrameworkCerr() << "overlay compositor init failed, falling back" << std::endl;
            overlayCompositor_.reset();
            renderDiag("  overlay compositor init FAILED");
        } else {
            renderFrameworkCout() << "overlay compositor init succeeded" << std::endl;
            renderDiag("  overlay compositor init SUCCESS");
        }
    } else if (wantOverlay && overlayCompositor_ && overlayCompositor_->isActive()) {
        overlayCompositor_->resize(swapchain_->vkExtent().width, swapchain_->vkExtent().height);
    } else if (wantOverlay && overlayCompositor_ && !overlayCompositor_->isActive()) {
        // Overlay exists but became inactive (DXGI device lost, resize failure, etc.)
        renderDiag("  overlay inactive, recreating...");
        overlayCompositor_->destroy();
        overlayCompositor_.reset();
        overlayCompositor_ = std::make_unique<OverlayCompositor>();
        if (!overlayCompositor_->init(shared_from_this())) {
            overlayCompositor_.reset();
            renderDiag("  overlay recreate FAILED");
        } else {
            renderDiag("  overlay recreate SUCCESS");
        }
    } else if (!wantOverlay && overlayCompositor_) {
        overlayCompositor_->destroy();
        overlayCompositor_.reset();
    }

    // Overlay present is handled on the render thread (submitCommand), not PresentThread.
    // PresentThread is only for future decoupled present mode.
    if (presentThread_ && !decoupledPresent_) {
        presentThread_->setOverlayMode(false, nullptr);
    }

    minimizedDefer_ = false;
    renderDiag("recreate() END");

    // Disable decoupled present during recreate. Decoupled present (Phase 2a)
    // will be enabled explicitly via JNI once the overlay window is set up.
    if (decoupledPresent_) {
        disableDecoupledPresent();
    }

    Renderer::instance().textures()->bindAllTextures();

    // Resume BLAS thread after recreate is complete
    {
        auto world = Renderer::instance().world();
        if (world && world->chunks() && world->chunks()->chunkBuildScheduler()) {
            world->chunks()->chunkBuildScheduler()->resume();
        }
    }
    renderDiag("  BLAS thread resumed");
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
    // Destroy overlay compositor before GPU teardown
    overlayCompositor_.reset();

    // Stop present thread before GPU teardown
    if (presentThread_) {
        presentThread_->stop();
        presentThread_.reset();
    }
    frameSlotRing_.reset();

    if (running_) { pipeline_->close(); }
    running_ = false;
    // Shutdown Streamline before Vulkan device destruction
    StreamlineContext::shutdown();
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
        waitDeviceIdle();
        crashExit(result, "vkWaitForFences failed (screenshot)");
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
        waitDeviceIdle();
        crashExit(result, "vkWaitForFences failed (screenshot)");
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
        waitDeviceIdle();
        crashExit(result, "vkWaitForFences failed (screenshot)");
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
        waitDeviceIdle();
        crashExit(result, "vkWaitForFences failed (screenshot)");
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
    while (currentContext_ == nullptr) {
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

void Framework::enableDecoupledPresent() {
    if (decoupledPresent_) return;
    decoupledPresent_ = true;
    decoupledFrameIndex_ = 0;
    if (presentThread_) {
        if (!presentThread_->isRunning()) {
            presentThread_->start();
        } else if (presentThread_->isPaused()) {
            presentThread_->resume();
        }
    }
}

void Framework::disableDecoupledPresent() {
    if (!decoupledPresent_) return;
    if (presentThread_ && presentThread_->isRunning() && !presentThread_->isPaused()) {
        presentThread_->pause();
    }
    decoupledPresent_ = false;
}

bool Framework::isDecoupledPresent() const {
    return decoupledPresent_;
}

bool Framework::isOverlayCompositorActive() const {
    return overlayCompositor_ && overlayCompositor_->isActive();
}

OverlayCompositor *Framework::overlayCompositor() const {
    return overlayCompositor_.get();
}

GarbageCollector::GarbageCollector(std::shared_ptr<Framework> framework) : framework_(framework) {
    // Use more slots than swapchain imageCount to give GPU work on the secondary
    // queue (chunk BLAS builds) time to complete before resources are freed.
    // With imageCount=2-3, resources can be destroyed while the GPU still references
    // them from in-flight BLAS/TLAS builds, causing WRITE_AFTER_DESTROY at address 0x0.
    uint32_t gcSlots = std::max(framework->swapchain_->imageCount() * 3, 32u);
    collectors_.resize(gcSlots);
}

void GarbageCollector::clear() {
    index_ = (index_ + 1) % collectors_.size();

    auto framework = framework_.lock();
    collectors_[index_].clear();
}
