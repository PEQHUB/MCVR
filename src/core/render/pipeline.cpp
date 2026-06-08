#include "core/render/pipeline.hpp"

#include "core/render/gpu_diagnostics.hpp"
#include "core/render/crash_ring_buffer.hpp"
#include "core/render/hdr_composite_pass.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include "core/render/modules/ui_module.hpp"
#include "core/render/modules/world/dlss/dlss_module.hpp"

#include "core/render/modules/world/post_render/post_render_module.hpp"
#include "core/render/modules/world/ray_tracing/ray_tracing_module.hpp"
#include "core/render/modules/world/svgf/svgf_module.hpp"
#include "core/render/modules/world/temporal_accumulation/temporal_accumulation_module.hpp"
#include "core/render/modules/world/tone_mapping/tone_mapping_module.hpp"

#include "core/render/gpu_profiler.hpp"
#include "core/render/radiance_logger.hpp"

#include <cstdlib>
#include <set>

WorldPipelineBlueprint::WorldPipelineBlueprint(WorldPipelineBuildParams *params) {
    std::set<uint32_t> imageIndices;
    auto framework = Renderer::instance().framework();
    auto pipeline = framework->pipeline();

    for (int i = 0; i < params->moduleCount; i++) {
        std::string moduleName = params->moduleNames[i];
        moduleNames_.emplace_back(moduleName);
        auto &moduleInputIndices = modulesInputIndices_.emplace_back();
        auto &moduleOutputIndices = modulesOutputIndices_.emplace_back();

        attributeCounts_.emplace_back(params->attributeCounts[i]);
        auto &attributeKV = attributeKVs_.emplace_back();
        for (int j = 0; j < params->attributeCounts[i]; j++) {
            attributeKV.push_back(std::string{params->attributeKVs[i][2 * j + 0]});
            attributeKV.push_back(std::string{params->attributeKVs[i][2 * j + 1]});
        }

        auto [inputImageNum, outputImageNum] = Pipeline::worldModuleInOutImageNums[moduleName];
        for (int j = 0; j < inputImageNum; j++) {
            moduleInputIndices.emplace_back(params->inputIndices[i][j]);
            imageIndices.insert(params->inputIndices[i][j]);
        }
        for (int j = 0; j < outputImageNum; j++) {
            moduleOutputIndices.emplace_back(params->outputIndices[i][j]);
            imageIndices.insert(params->outputIndices[i][j]);
        }
    }

    int cnt = 0;
    for (auto i : imageIndices) {
        if (i != cnt) { throw std::runtime_error("Indices are not continous!"); }
        cnt++;
        imageFormats_.push_back(static_cast<VkFormat>(params->imageFormats[i]));
    }
}

WorldPipeline::WorldPipeline() {}

void WorldPipeline::init(std::shared_ptr<Framework> framework, std::shared_ptr<Pipeline> pipeline) {
    auto blueprint = pipeline->worldPipelineBlueprint();
    uint32_t frameNum = framework->swapchain()->imageCount();

    worldModules_.resize(blueprint->moduleNames_.size());
    moduleNames_ = blueprint->moduleNames_;
    sharedImages_.resize(frameNum,
                         std::vector<std::shared_ptr<vk::DeviceLocalImage>>(blueprint->imageFormats_.size(), nullptr));
    contexts_.resize(frameNum);

    // Determine initial render resolution from upscaler quality (if present)
 // Determine initial render resolution from DLSS quality mode (if present)
 VkExtent2D extent = framework->swapchain()->vkExtent();
 uint32_t renderWidth = extent.width;
 uint32_t renderHeight = extent.height;
 size_t upscalerIndex = std::numeric_limits<size_t>::max();
 for (size_t i = 0; i < blueprint->moduleNames_.size(); i++) {
 if (blueprint->moduleNames_[i] != DLSSModule::NAME) continue;
 upscalerIndex = i;
 // DLSS-RR queries its own render resolution from NVIDIA API during build().
 // The render resolution calculation is handled internally by DLSSModule
 // based on the attribute mode (performance/balanced/quality/dlaa).
 break;
 }

    for (int frameIndex = 0; frameIndex < frameNum; frameIndex++) {
        // Keep the primary output at display resolution
        sharedImages_[frameIndex][0] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, extent.width, extent.height, 1, blueprint->imageFormats_[0],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        );
    }

    // Pre-create outputs for modules before upscaler at render resolution
    if (upscalerIndex != std::numeric_limits<size_t>::max() &&
        (renderWidth != extent.width || renderHeight != extent.height)) {
        std::set<uint32_t> renderIndices;
        for (size_t i = 0; i < upscalerIndex; i++) {
            for (uint32_t idx : blueprint->modulesOutputIndices_[i]) renderIndices.insert(idx);
        }

        for (int frameIndex = 0; frameIndex < frameNum; frameIndex++) {
            for (uint32_t idx : renderIndices) {
                if (sharedImages_[frameIndex][idx] != nullptr) continue;
                sharedImages_[frameIndex][idx] = vk::DeviceLocalImage::create(
                    framework->device(), framework->vma(), false, renderWidth, renderHeight, 1,
                    blueprint->imageFormats_[idx],
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
#ifdef USE_AMD
                        | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
#endif
                );
            }
        }
    }

    for (int i = blueprint->moduleNames_.size() - 1; i >= 0; i--) {
        worldModules_[i] = Pipeline::worldModuleConstructors[blueprint->moduleNames_[i]](framework, shared_from_this());

        auto &moduleInputIndices = blueprint->modulesInputIndices_[i];
        auto &moduleOutputIndices = blueprint->modulesOutputIndices_[i];

        for (int frameIndex = 0; frameIndex < frameNum; frameIndex++) {
            { // output
                std::vector<std::shared_ptr<vk::DeviceLocalImage>> outputImages;
                std::vector<VkFormat> outputFormats;
                for (int j = 0; j < moduleOutputIndices.size(); j++) {
                    outputImages.push_back(sharedImages_[frameIndex][moduleOutputIndices[j]]);
                    outputFormats.push_back(blueprint->imageFormats_[moduleOutputIndices[j]]);
                }
                bool result = worldModules_[i]->setOrCreateOutputImages(outputImages, outputFormats, frameIndex);
                if (!result) {
                    std::cout << blueprint->moduleNames_[i] << std::endl;
                    throw std::runtime_error("Output image not set properly");
                }
                for (int j = 0; j < moduleOutputIndices.size(); j++) {
                    sharedImages_[frameIndex][moduleOutputIndices[j]] = outputImages[j];
                }
            }

            { // input
                std::vector<std::shared_ptr<vk::DeviceLocalImage>> inputImages;
                std::vector<VkFormat> inputFormats;
                for (int j = 0; j < moduleInputIndices.size(); j++) {
                    inputImages.push_back(sharedImages_[frameIndex][moduleInputIndices[j]]);
                    inputFormats.push_back(blueprint->imageFormats_[moduleInputIndices[j]]);
                }
                bool result = worldModules_[i]->setOrCreateInputImages(inputImages, inputFormats, frameIndex);
                if (!result) throw std::runtime_error("Input image not set properly");
                for (int j = 0; j < moduleInputIndices.size(); j++) {
                    sharedImages_[frameIndex][moduleInputIndices[j]] = inputImages[j];
                }
            }
        }

        worldModules_[i]->setAttributes(blueprint->attributeCounts_[i], blueprint->attributeKVs_[i]);

        worldModules_[i]->build();
    }

    for (int i = 0; i < framework->swapchain()->imageCount(); i++) {
        contexts_[i] = WorldPipelineContext::create(framework->contexts()[i], shared_from_this());
    }
}

std::vector<std::shared_ptr<WorldModule>> &WorldPipeline::worldModules() {
    return worldModules_;
}

std::vector<std::shared_ptr<WorldPipelineContext>> &WorldPipeline::contexts() {
    return contexts_;
}

void WorldPipeline::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                                std::shared_ptr<vk::DeviceLocalImage> image,
                                int index) {
    for (int i = 0; i < worldModules_.size(); i++) { worldModules_[i]->bindTexture(sampler, image, index); }
}

WorldPipelineContext::WorldPipelineContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                           std::shared_ptr<WorldPipeline> worldPipeline)
    : frameworkContext(frameworkContext),
      worldPipeline(worldPipeline),
      outputImage(worldPipeline->sharedImages_[frameworkContext->frameIndex][0]) {
    auto &worldModules = worldPipeline->worldModules();
    for (int i = 0; i < worldModules.size(); i++) {
        worldModuleContexts.push_back(worldModules[i]->contexts()[frameworkContext->frameIndex]);
    }
}

void WorldPipelineContext::render() {
    auto context = frameworkContext.lock();
    if (!context) return;
    auto framework = context->framework.lock();
    if (!framework) return;
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    // Preflight: ensure output image has a valid initial layout (avoid UNDEFINED on AMD)
    if (outputImage && outputImage->imageLayout() == VK_IMAGE_LAYOUT_UNDEFINED) {
        VkImageLayout targetLayout =
#ifdef USE_AMD
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
        VkPipelineStageFlags2 dstStage =
#ifdef USE_AMD
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
#else
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
#endif
        VkAccessFlags2 dstAccess =
#ifdef USE_AMD
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
#else
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
#endif
        worldCommandBuffer->barriersBufferImage({}, {{
                                                        .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                                        .srcAccessMask = 0,
                                                        .dstStageMask = dstStage,
                                                        .dstAccessMask = dstAccess,
                                                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                                        .newLayout = targetLayout,
                                                        .srcQueueFamilyIndex = mainQueueIndex,
                                                        .dstQueueFamilyIndex = mainQueueIndex,
                                                        .image = outputImage,
                                                        .subresourceRange = vk::wholeColorSubresourceRange,
                                                    }});
        outputImage->imageLayout() = targetLayout;
    }

    GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::FRAME_BEGIN);

    // Short human-readable names for Nsight labels + GPU profiler
    static const std::map<std::string, std::string> moduleShortNames = {
        {"render_pipeline.module.ray_tracing.name", "RayTracing"},
        {"render_pipeline.module.dlss.name", "DLSS-RR"},
        {"render_pipeline.module.tone_mapping.name", "ToneMapping"},
        {"render_pipeline.module.post_render.name", "PostRender"},
        {"render_pipeline.module.temporal_accumulation.name", "TAA"},
        {"SVGF", "SVGF"},
    };

    auto& profiler = Renderer::gpuProfiler;
    bool profiling = profiler.isEnabled();
    VkCommandBuffer rawCmd = profiling ? worldCommandBuffer->vkCommandBuffer() : VK_NULL_HANDLE;
    auto wp = worldPipeline.lock();

    for (int i = 0; i < worldModuleContexts.size(); i++) {
    // Resolve module name
    std::string name = "Module_" + std::to_string(i);
    if (wp && i < wp->moduleNames_.size()) {
      auto it = moduleShortNames.find(wp->moduleNames_[i]);
      name = (it != moduleShortNames.end()) ? it->second : wp->moduleNames_[i];
    }

    bool labelOpen = false;
    bool profilerOpen = false;
    try {
        if (!worldModuleContexts[i]) {
            throw std::runtime_error("null world module context");
        }

        // Vulkan debug label (visible in Nsight Systems/Graphics)
        worldCommandBuffer->beginLabel(name.c_str());
        labelOpen = true;

        // Native GPU profiler timestamp
        if (profiling) {
            profiler.beginModule(rawCmd, name);
            profilerOpen = true;
        }

        std::string crashTag = "wm:" + name;
        renderDiag("world module begin i=%d name=%s", i, name.c_str());
        g_crashRing.record(crashTag.c_str());

        worldModuleContexts[i]->render();

        renderDiag("world module end i=%d name=%s", i, name.c_str());

        if (profilerOpen) {
            profiler.endModule(rawCmd);
            profilerOpen = false;
        }
        worldCommandBuffer->endLabel();
        labelOpen = false;
    } catch (const std::exception& e) {
        renderDiag("FATAL: world module %d (%s) threw: %s", i, name.c_str(), e.what());
        try {
            if (profilerOpen) profiler.endModule(rawCmd);
        } catch (...) {}
        try {
            if (labelOpen) worldCommandBuffer->endLabel();
        } catch (...) {}
        throw;
    } catch (...) {
        renderDiag("FATAL: world module %d (%s) threw unknown exception", i, name.c_str());
        try {
            if (profilerOpen) profiler.endModule(rawCmd);
        } catch (...) {}
        try {
            if (labelOpen) worldCommandBuffer->endLabel();
        } catch (...) {}
        throw;
    }
	}

	GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::FRAME_COMPLETE);
    worldCommandBuffer->barriersBufferImage(
        {}, {{
                .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout = outputImage->imageLayout(),
#ifdef USE_AMD
                .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image = outputImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            }});

#ifdef USE_AMD
    outputImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    outputImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
}

std::map<std::string,
         std::function<std::shared_ptr<WorldModule>(std::shared_ptr<Framework>, std::shared_ptr<WorldPipeline>)>>
    Pipeline::worldModuleConstructors{};
std::map<std::string, std::pair<uint32_t, uint32_t>> Pipeline::worldModuleInOutImageNums{};
std::map<std::string, std::function<void()>> Pipeline::worldModuleStaticPreCloser{};

void Pipeline::collectWorldModules() {
    worldModuleConstructors.insert(std::make_pair(
        RayTracingModule::NAME, [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
            return RayTracingModule::create(framework, worldPipeline);
        }));
    worldModuleInOutImageNums.insert(std::make_pair(
      RayTracingModule::NAME, std::make_pair(RayTracingModule::inputImageNum, RayTracingModule::outputImageNum)));

  // NRD removed — NVIDIA-only hard fork uses DLSS-RR's built-in denoiser

  // SVGF not working well, leave it here for future use
  // worldModuleConstructors.insert(std::make_pair(
  //     SvgfModule::NAME, [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
  //       return SvgfModule::create(framework, worldPipeline); }));
  // worldModuleInOutImageNums.insert(
  //     std::make_pair(SvgfModule::NAME, std::make_pair(SvgfModule::inputImageNum, SvgfModule::outputImageNum)));

  worldModuleConstructors.insert(
      std::make_pair(TemporalAccumulationModule::NAME,
      [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
        return TemporalAccumulationModule::create(framework, worldPipeline);
      }));
  worldModuleInOutImageNums.insert(
      std::make_pair(TemporalAccumulationModule::NAME, std::make_pair(TemporalAccumulationModule::inputImageNum,
      TemporalAccumulationModule::outputImageNum)));

  // FSR upscaler removed — NVIDIA-only hard fork uses DLSS-RR

  worldModuleConstructors.insert(std::make_pair(
      ToneMappingModule::NAME,
      [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
        return ToneMappingModule::create(framework, worldPipeline);
      }));
  worldModuleInOutImageNums.insert(std::make_pair(
      ToneMappingModule::NAME, std::make_pair(ToneMappingModule::inputImageNum, ToneMappingModule::outputImageNum)));

  bool result = DLSSModule::initNGXContext();
  if (result) {
    worldModuleConstructors.insert(std::make_pair(
        DLSSModule::NAME, [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
          return DLSSModule::create(framework, worldPipeline);
        }));
    worldModuleInOutImageNums.insert(
        std::make_pair(DLSSModule::NAME, std::make_pair(DLSSModule::inputImageNum, DLSSModule::outputImageNum)));
    worldModuleStaticPreCloser.insert(std::make_pair(DLSSModule::NAME, DLSSModule::deinitNGXContext));
  }

  worldModuleConstructors.insert(
      std::make_pair(PostRenderModule::NAME,
      [](std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
        return PostRenderModule::create(framework, worldPipeline);
      }));
  worldModuleInOutImageNums.insert(std::make_pair(
      PostRenderModule::NAME, std::make_pair(PostRenderModule::inputImageNum, PostRenderModule::outputImageNum)));

  // The fork-native volumetric cloud compute module is intentionally not registered.
  // Clouds are handled shader-side in the ray tracing path for the visual-only lane.

  // TODO: invoke extension's collection
}
void Pipeline::recollectWorldModules() {
    // Shut down any existing NGX context BEFORE re-collecting modules.
    // NVSDK_NGX_VULKAN_Init() is a process-level call — Shutdown1() must be called
    // before it can be successfully re-initialized (e.g. after user drops in DLSS DLLs).
    // Without this, the second call to initNGXContext() fails silently on double-init.
    DLSSModule::deinitNGXContext();

    // Clear all three maps so re-insertion works correctly, then re-run full collection.
    worldModuleConstructors.clear();
    worldModuleInOutImageNums.clear();
    worldModuleStaticPreCloser.clear();
    collectWorldModules();
}

Pipeline::Pipeline() {}

Pipeline::~Pipeline() {
#ifdef DEBUG
    std::cout << "Pipeline deconstruct" << std::endl;
#endif
}

void Pipeline::init(std::shared_ptr<Framework> framework) {
    framework_ = framework;
    uiModule_ = UIModule::create(framework);
    contexts_ = std::make_shared<std::vector<std::shared_ptr<PipelineContext>>>();

    // World/UI composite pass (used for both SDR and HDR output)
    hdrCompositePass_ = HdrCompositePass::create();
    hdrCompositePass_->init(framework);

    uint32_t size = framework->swapchain()->imageCount();
    contexts_->resize(size);

    for (int i = 0; i < size; i++) {
        contexts_->at(i) = PipelineContext::create(framework->contexts()[i], shared_from_this());
    }
}

void Pipeline::buildWorldPipelineBlueprint(WorldPipelineBuildParams *params) {
    worldPipelineBlueprint_ = WorldPipelineBlueprint::create(params);
    needRecreate = true;
}

void Pipeline::recreate(std::shared_ptr<Framework> framework) {
    uiModule_.reset();
    uiModule_ = UIModule::create(framework);

    if (worldPipeline_ != nullptr)
        for (auto &module : worldPipeline_->worldModules()) { module->preClose(); }
    worldPipeline_.reset();
    worldPipeline_ =
        worldPipelineBlueprint_ == nullptr ? nullptr : WorldPipeline::create(framework, shared_from_this());

    // World/UI composite pass (used for both SDR and HDR output)
    if (hdrCompositePass_) {
        hdrCompositePass_->recreate(framework);
    } else {
        hdrCompositePass_ = HdrCompositePass::create();
        hdrCompositePass_->init(framework);
    }

    contexts_.reset();
    contexts_ = std::make_shared<std::vector<std::shared_ptr<PipelineContext>>>();

    uint32_t size = framework->swapchain()->imageCount();
    contexts_->resize(size);

    for (int i = 0; i < size; i++) {
        contexts_->at(i) = PipelineContext::create(framework->contexts()[i], shared_from_this());
    }
}

void Pipeline::close() {
    for (auto &module : worldPipeline_->worldModules()) { module->preClose(); }

    for (auto &destructor : worldModuleStaticPreCloser) { destructor.second(); }
}

std::shared_ptr<PipelineContext> Pipeline::acquirePipelineContext(std::shared_ptr<FrameworkContext> context) {
    return contexts_->at(context->frameIndex);
}

std::vector<std::shared_ptr<PipelineContext>> &Pipeline::contexts() {
    return *contexts_;
}

void Pipeline::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                           std::shared_ptr<vk::DeviceLocalImage> image,
                           int index) {
    if (worldPipeline_ != nullptr) worldPipeline_->bindTexture(sampler, image, index);
    uiModule_->bindTexture(sampler, image, index);
}

std::shared_ptr<UIModule> Pipeline::uiModule() {
    return uiModule_;
}

std::shared_ptr<WorldPipeline> Pipeline::worldPipeline() {
    return worldPipeline_;
}

std::shared_ptr<HdrCompositePass> Pipeline::hdrCompositePass() {
    return hdrCompositePass_;
}

std::shared_ptr<WorldPipelineBlueprint> Pipeline::worldPipelineBlueprint() {
    return worldPipelineBlueprint_;
}

PipelineContext::PipelineContext(std::shared_ptr<FrameworkContext> frameworkContext, std::shared_ptr<Pipeline> pipeline)
    : frameworkContext(frameworkContext),
      uiModuleContext(pipeline->uiModule()->contexts()[frameworkContext->frameIndex]),
      worldPipelineContext(pipeline->worldPipeline() == nullptr ?
                               nullptr :
                               pipeline->worldPipeline()->contexts()[frameworkContext->frameIndex]) {}

void PipelineContext::fuseWorld() {
    auto context = frameworkContext.lock();
    if (!context) return;
    auto framework = context->framework.lock();
    if (!framework || !framework->isRunning()) return;

    uiModuleContext->end();
}
