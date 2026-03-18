#include "fsr_frame_gen_manager.hpp"
#include "core/render/renderer.hpp"
#include "core/render/modules/world/fsr_upscaler/fsr_setup.hpp"
#include <iostream>

#ifdef MCVR_ENABLE_FFX_UPSCALER
#   ifdef _WIN32
#       include <windows.h>
#       define FFX_API_ENTRY
#       pragma warning(push)
#       pragma warning(disable : 4005)
#   endif

#   include <ffx_api/ffx_api.hpp>
#   include <ffx_api/ffx_framegeneration.hpp>
#   include <ffx_api/vk/ffx_api_vk.hpp>

#   ifdef _WIN32
#       pragma warning(pop)
#   endif

#   include <volk.h>
#endif

static auto& fgCout() { return std::cout << "[FSR-FG] "; }
static auto& fgCerr() { return std::cerr << "[FSR-FG] "; }

// Static members
bool FsrFrameGenManager::initialized_ = false;
bool FsrFrameGenManager::active_ = false;
bool FsrFrameGenManager::pendingEnable_ = false;
uint64_t FsrFrameGenManager::frameId_ = 0;

std::shared_ptr<vk::Device> FsrFrameGenManager::device_;
std::shared_ptr<vk::PhysicalDevice> FsrFrameGenManager::physDevice_;

PFN_vkQueuePresentKHR       FsrFrameGenManager::proxyPresent_ = nullptr;
PFN_vkAcquireNextImageKHR   FsrFrameGenManager::proxyAcquire_ = nullptr;
PFN_vkGetSwapchainImagesKHR FsrFrameGenManager::proxyGetImages_ = nullptr;

void* FsrFrameGenManager::fgContext_ = nullptr;
void* FsrFrameGenManager::fgSwapChainContext_ = nullptr;
bool FsrFrameGenManager::fgContextCreated_ = false;
bool FsrFrameGenManager::fgSwapChainCreated_ = false;

bool FsrFrameGenManager::init(std::shared_ptr<vk::Device> device,
                               std::shared_ptr<vk::PhysicalDevice> physDevice) {
#ifndef MCVR_ENABLE_FFX_UPSCALER
    (void)device; (void)physDevice;
    return false;
#else
    device_ = device;
    physDevice_ = physDevice;
    initialized_ = true;
    fgCout() << "initialized (FSR 3.1.4 frame generation available)" << std::endl;
    return true;
#endif
}

bool FsrFrameGenManager::isActive() {
    return active_ || pendingEnable_;
}

void FsrFrameGenManager::prepare(VkCommandBuffer cmdBuffer,
                                  VkImage depthImage, VkFormat depthFormat,
                                  uint32_t depthWidth, uint32_t depthHeight,
                                  VkImage mvImage, VkFormat mvFormat,
                                  uint32_t mvWidth, uint32_t mvHeight,
                                  float jitterX, float jitterY,
                                  float mvScaleX, float mvScaleY,
                                  float frameTimeDelta,
                                  float cameraNear, float cameraFar,
                                  float cameraFovVertical,
                                  const float cameraPos[3],
                                  const float cameraUp[3],
                                  const float cameraRight[3],
                                  const float cameraForward[3]) {
#ifdef MCVR_ENABLE_FFX_UPSCALER
    if (!active_ || !fgContextCreated_) return;
    // Skip during offline accumulation
    if (Renderer::options.offlineState == 2) return;

    auto& ctx = reinterpret_cast<ffx::Context&>(fgContext_);

    // Prepare dispatch: dilate depth + MVs at render resolution
    ffx::DispatchDescFrameGenerationPrepare prepareDesc{};
    prepareDesc.frameID = frameId_;
    prepareDesc.commandList = cmdBuffer;
    prepareDesc.renderSize = {depthWidth, depthHeight};
    prepareDesc.jitterOffset = {jitterX, jitterY};
    prepareDesc.motionVectorScale = {mvScaleX, mvScaleY};
    prepareDesc.frameTimeDelta = frameTimeDelta;
    prepareDesc.cameraNear = cameraNear;
    prepareDesc.cameraFar = cameraFar;
    prepareDesc.cameraFovAngleVertical = cameraFovVertical;
    prepareDesc.viewSpaceToMetersFactor = 1.0f;  // Minecraft: 1 block = 1 meter

    // Depth resource
    prepareDesc.depth.resource = depthImage;
    prepareDesc.depth.state = VK_IMAGE_LAYOUT_GENERAL;
    prepareDesc.depth.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    prepareDesc.depth.description.format = mcvr::fsr::vkToFfxFormat(depthFormat);
    prepareDesc.depth.description.width = depthWidth;
    prepareDesc.depth.description.height = depthHeight;
    prepareDesc.depth.description.depth = 1;
    prepareDesc.depth.description.mipCount = 1;

    // Motion vector resource
    prepareDesc.motionVectors.resource = mvImage;
    prepareDesc.motionVectors.state = VK_IMAGE_LAYOUT_GENERAL;
    prepareDesc.motionVectors.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    prepareDesc.motionVectors.description.format = mcvr::fsr::vkToFfxFormat(mvFormat);
    prepareDesc.motionVectors.description.width = mvWidth;
    prepareDesc.motionVectors.description.height = mvHeight;
    prepareDesc.motionVectors.description.depth = 1;
    prepareDesc.motionVectors.description.mipCount = 1;

    // Camera info (required for FSR 3.1.4+ quality)
    ffx::DispatchDescFrameGenerationPrepareCameraInfo cameraInfo{};
    cameraInfo.cameraPosition[0] = cameraPos[0];
    cameraInfo.cameraPosition[1] = cameraPos[1];
    cameraInfo.cameraPosition[2] = cameraPos[2];
    cameraInfo.cameraUp[0] = cameraUp[0];
    cameraInfo.cameraUp[1] = cameraUp[1];
    cameraInfo.cameraUp[2] = cameraUp[2];
    cameraInfo.cameraRight[0] = cameraRight[0];
    cameraInfo.cameraRight[1] = cameraRight[1];
    cameraInfo.cameraRight[2] = cameraRight[2];
    cameraInfo.cameraForward[0] = cameraForward[0];
    cameraInfo.cameraForward[1] = cameraForward[1];
    cameraInfo.cameraForward[2] = cameraForward[2];

    auto result = ffx::Dispatch(ctx, prepareDesc, cameraInfo);
    if (result != ffx::ReturnCode::Ok) {
        fgCerr() << "Prepare dispatch failed: " << static_cast<uint32_t>(result) << std::endl;
    }
#endif
}

void FsrFrameGenManager::configureFrame(std::shared_ptr<FrameworkContext> context,
                                         VkImage hudlessImage, VkFormat hudlessFormat,
                                         uint32_t hudlessWidth, uint32_t hudlessHeight,
                                         VkImage uiImage, VkFormat uiFormat,
                                         uint32_t uiWidth, uint32_t uiHeight) {
#ifdef MCVR_ENABLE_FFX_UPSCALER
    if (!active_ || !fgContextCreated_) return;
    if (Renderer::options.offlineState == 2) return;

    auto& ctx = reinterpret_cast<ffx::Context&>(fgContext_);
    auto& scCtx = reinterpret_cast<ffx::Context&>(fgSwapChainContext_);

    // Configure frame generation for this frame
    ffx::ConfigureDescFrameGeneration configDesc{};
    configDesc.frameGenerationEnabled = true;
    configDesc.allowAsyncWorkloads = false;  // conservative: single queue for now
    configDesc.frameID = frameId_;
    configDesc.flags = 0;

    // HUD-less color (post-tonemap world output, pre-UI)
    if (hudlessImage != VK_NULL_HANDLE) {
        configDesc.HUDLessColor.resource = hudlessImage;
        configDesc.HUDLessColor.state = VK_IMAGE_LAYOUT_GENERAL;
        configDesc.HUDLessColor.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
        configDesc.HUDLessColor.description.format = mcvr::fsr::vkToFfxFormat(hudlessFormat);
        configDesc.HUDLessColor.description.width = hudlessWidth;
        configDesc.HUDLessColor.description.height = hudlessHeight;
        configDesc.HUDLessColor.description.depth = 1;
        configDesc.HUDLessColor.description.mipCount = 1;
    }

    // Point swapChain to the proxy context
    configDesc.swapChain = &scCtx;

    auto result = ffx::Configure(ctx, configDesc);
    if (result != ffx::ReturnCode::Ok) {
        fgCerr() << "Configure failed: " << static_cast<uint32_t>(result) << std::endl;
    }

    // Register UI resource with proxy swapchain
    if (uiImage != VK_NULL_HANDLE) {
        ffxConfigureDescFrameGenerationSwapChainRegisterUiResourceVK uiDesc{};
        uiDesc.header.type = FFX_API_CONFIGURE_DESC_TYPE_FGSWAPCHAIN_REGISTERUIRESOURCE_VK;
        uiDesc.uiResource.resource = uiImage;
        uiDesc.uiResource.state = VK_IMAGE_LAYOUT_GENERAL;
        uiDesc.uiResource.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
        uiDesc.uiResource.description.format = mcvr::fsr::vkToFfxFormat(uiFormat);
        uiDesc.uiResource.description.width = uiWidth;
        uiDesc.uiResource.description.height = uiHeight;
        uiDesc.uiResource.description.depth = 1;
        uiDesc.uiResource.description.mipCount = 1;
        uiDesc.flags = 0;

        ffxConfigure(reinterpret_cast<ffxContext*>(&scCtx), &uiDesc.header);
    }

    frameId_++;
#endif
}

VkResult FsrFrameGenManager::present(VkQueue queue, const VkPresentInfoKHR* presentInfo) {
    if (proxyPresent_) {
        return proxyPresent_(queue, presentInfo);
    }
    return vkQueuePresentKHR(queue, presentInfo);
}

void FsrFrameGenManager::beforeSwapchainRecreate() {
#ifdef MCVR_ENABLE_FFX_UPSCALER
    if (!initialized_) return;

    // Disable FG before swapchain teardown
    if (fgContextCreated_) {
        auto& ctx = reinterpret_cast<ffx::Context&>(fgContext_);
        ffx::ConfigureDescFrameGeneration configDesc{};
        configDesc.frameGenerationEnabled = false;
        ffx::Configure(ctx, configDesc);
    }

    // Destroy proxy swapchain context
    if (fgSwapChainCreated_) {
        ffxDestroyContext(reinterpret_cast<ffxContext*>(&fgSwapChainContext_), nullptr);
        fgSwapChainCreated_ = false;
        proxyPresent_ = nullptr;
        proxyAcquire_ = nullptr;
        proxyGetImages_ = nullptr;
    }

    // Destroy FG context
    if (fgContextCreated_) {
        ffxDestroyContext(reinterpret_cast<ffxContext*>(&fgContext_), nullptr);
        fgContextCreated_ = false;
    }

    active_ = false;
    pendingEnable_ = false;
#endif
}

void FsrFrameGenManager::afterSwapchainRecreate(std::shared_ptr<vk::Swapchain> swapchain,
                                                  std::shared_ptr<vk::Device> device,
                                                  std::shared_ptr<vk::PhysicalDevice> physDevice) {
#ifdef MCVR_ENABLE_FFX_UPSCALER
    if (!initialized_) return;

    bool wantActive = (Renderer::options.frameGenBackend == 2 && Renderer::options.frameGenMode != 0);
    if (!wantActive) return;

    device_ = device;
    physDevice_ = physDevice;

    // --- Create FG context ---
    ffx::CreateBackendVKDesc backendDesc{};
    backendDesc.vkDevice = device_->vkDevice();
    backendDesc.vkPhysicalDevice = physDevice_->vkPhysicalDevice();
    backendDesc.vkDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        mcvr::fsr::customVkGetDeviceProcAddr);

    ffx::CreateContextDescFrameGeneration fgCreateDesc{};
    fgCreateDesc.displaySize = {swapchain->width(), swapchain->height()};
    fgCreateDesc.maxRenderSize = {swapchain->width(), swapchain->height()};  // TODO: actual render res
    fgCreateDesc.backBufferFormat = mcvr::fsr::vkToFfxFormat(swapchain->imageFormat());
    fgCreateDesc.flags = FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED
                       | FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE
                       | FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE;

    auto& ctx = reinterpret_cast<ffx::Context&>(fgContext_);
    auto result = ffx::CreateContext(ctx, nullptr, fgCreateDesc, backendDesc);
    if (result != ffx::ReturnCode::Ok) {
        fgCerr() << "Failed to create FG context: " << static_cast<uint32_t>(result) << std::endl;
        return;
    }
    fgContextCreated_ = true;
    fgCout() << "FG context created" << std::endl;

    // --- Create proxy swapchain ---
    ffxCreateContextDescFrameGenerationSwapChainVK scDesc{};
    scDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
    scDesc.physicalDevice = physDevice_->vkPhysicalDevice();
    scDesc.device = device_->vkDevice();
    scDesc.swapchain = &swapchain->vkSwapchain();
    scDesc.allocator = nullptr;
    scDesc.createInfo = swapchain->lastCreateInfo();

    // Queue setup
    scDesc.gameQueue.queue = device_->mainVkQueue();
    scDesc.gameQueue.familyIndex = physDevice_->mainQueueIndex();
    scDesc.gameQueue.submitFunc = nullptr;

    scDesc.asyncComputeQueue.queue = device_->secondaryQueue();
    scDesc.asyncComputeQueue.familyIndex = physDevice_->secondaryQueueIndex();
    scDesc.asyncComputeQueue.submitFunc = nullptr;

    scDesc.presentQueue.queue = device_->presentVkQueue();
    scDesc.presentQueue.familyIndex = physDevice_->presentQueueIndex();
    scDesc.presentQueue.submitFunc = nullptr;

    scDesc.imageAcquireQueue.queue = device_->secondaryQueue();
    scDesc.imageAcquireQueue.familyIndex = physDevice_->secondaryQueueIndex();
    scDesc.imageAcquireQueue.submitFunc = nullptr;

    auto& scCtx = reinterpret_cast<ffx::Context&>(fgSwapChainContext_);

    // Chain the backend desc for the swapchain context creation
    ffx::CreateBackendVKDesc scBackendDesc{};
    scBackendDesc.vkDevice = device_->vkDevice();
    scBackendDesc.vkPhysicalDevice = physDevice_->vkPhysicalDevice();
    scBackendDesc.vkDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        mcvr::fsr::customVkGetDeviceProcAddr);

    auto scResult = ffxCreateContext(reinterpret_cast<ffxContext*>(&scCtx),
                                     &scDesc.header, nullptr);
    if (scResult != FFX_API_RETURN_OK) {
        fgCerr() << "Failed to create proxy swapchain: " << scResult << std::endl;
        // Clean up FG context
        ffxDestroyContext(reinterpret_cast<ffxContext*>(&fgContext_), nullptr);
        fgContextCreated_ = false;
        return;
    }
    fgSwapChainCreated_ = true;

    // Query replacement function pointers
    ffxQueryDescSwapchainReplacementFunctionsVK funcQuery{};
    funcQuery.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
    auto qResult = ffxQuery(reinterpret_cast<ffxContext*>(&scCtx), &funcQuery.header);
    if (qResult == FFX_API_RETURN_OK) {
        proxyPresent_ = funcQuery.pOutQueuePresentKHR;
        proxyAcquire_ = funcQuery.pOutAcquireNextImageKHR;
        proxyGetImages_ = funcQuery.pOutGetSwapchainImagesKHR;
    } else {
        fgCerr() << "Failed to query replacement functions: " << qResult << std::endl;
    }

    active_ = true;
    frameId_ = 0;
    fgCout() << "proxy swapchain created, frame generation active" << std::endl;
#endif
}

void FsrFrameGenManager::shutdown() {
    beforeSwapchainRecreate();
    initialized_ = false;
    device_.reset();
    physDevice_.reset();
}

PFN_vkAcquireNextImageKHR FsrFrameGenManager::getAcquireNextImage() {
    return proxyAcquire_;
}

PFN_vkGetSwapchainImagesKHR FsrFrameGenManager::getGetSwapchainImages() {
    return proxyGetImages_;
}
