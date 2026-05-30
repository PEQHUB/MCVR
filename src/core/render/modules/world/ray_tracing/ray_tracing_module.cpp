#include "core/render/modules/world/ray_tracing/ray_tracing_module.hpp"
#include <chrono>

#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/gpu_diagnostics.hpp"
#include "core/render/lights.hpp"
#include "core/render/modules/world/ray_tracing/submodules/atmosphere.hpp"
#include "core/render/modules/world/ray_tracing/submodules/world_prepare.hpp"
#include "core/render/modules/world/shader_pack/shader_pack.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/crash_ring_buffer.hpp"
#include "core/render/radiance_logger.hpp"
#include "core/render/renderer.hpp"
#include "core/render/textures.hpp"
#include "core/render/modules/world/svgf/blue_noise.hpp"
#include "core/vulkan/debug_utils.hpp"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <unordered_set>

namespace {
constexpr int kSharcMinCapacityExponent = 18;
constexpr int kSharcMaxCapacityExponent = 24;
constexpr uint32_t kSharcDispatchStableFrameThreshold = 60;
constexpr uint32_t kSharcMainTraceMaxWarmupFrames = 8;
constexpr uint32_t kSharcQueryCounterCount = 8;
constexpr uint32_t kDirectLightCounterCount = 16;
constexpr uint32_t kRtDebugDisableDirectLighting = 16;
constexpr bool kForceDisableShaderDisplacementForGpuFaultIsolation = true;
constexpr bool kAllowUpstreamRtExecutorDispatch = false;
constexpr bool kAllowDirectLightScaffoldDispatch = false;

bool shaderDisplacementRuntimeAllowed() {
    return !kForceDisableShaderDisplacementForGpuFaultIsolation;
}

int effectiveSharcCapacityExponent() {
    return std::clamp(Renderer::options.sharcCapacityExponent,
                      kSharcMinCapacityExponent,
                      kSharcMaxCapacityExponent);
}

uint32_t effectiveSharcCapacity() {
    return 1u << static_cast<uint32_t>(effectiveSharcCapacityExponent());
}

int effectiveSharcDownscale() {
    return std::clamp(Renderer::options.sharcDownscale, 1, 8);
}

int effectiveSharcUpdateBlockSize() {
    return std::clamp(Renderer::options.sharcUpdateBlockSize, 2, 8);
}

int effectiveSharcUpdateBounces() {
    return std::clamp(Renderer::options.sharcUpdateBounces, 2, 8);
}

uint32_t effectiveSharcMainTraceWarmupFrames() {
    return std::min<uint32_t>(
        kSharcMainTraceMaxWarmupFrames,
        static_cast<uint32_t>(std::clamp(Renderer::options.sharcAccumulationFrames, 1, 256)));
}

int effectiveSharcQueryMode() {
    return std::clamp(Renderer::options.sharcQueryMode, 0, 2);
}

uint32_t effectiveDirectLightBackend() {
    return std::clamp<uint32_t>(Renderer::options.directLightBackend, 0, 2);
}

bool upstreamRtPackFileExists(const char* relativePath) {
    return std::filesystem::exists(
        Renderer::folderPath / "shaders/world/ray_tracing/internal/advanced" / relativePath);
}

uint32_t upstreamRtPassReadinessMask() {
    uint32_t mask = 0;
    const char* passFiles[] = {
        "primary/primary.rgen",
        "generate_initial_samples/generate_initial_samples.comp",
        "visibility/visibility.rgen",
        "temporal_reuse/temporal_reuse.comp",
        "spatial_reuse/spatial_reuse.comp",
        "direct_light/direct_light.rgen",
        "world/world.rgen",
    };
    for (uint32_t i = 0; i < static_cast<uint32_t>(sizeof(passFiles) / sizeof(passFiles[0])); ++i) {
        if (upstreamRtPackFileExists(passFiles[i])) {
            mask |= (1u << i);
        }
    }
    return mask;
}

bool upstreamRtPackReady() {
    constexpr uint32_t kRequiredPassMask = 0x7fu;
    return upstreamRtPackFileExists("configs.json") &&
        (upstreamRtPassReadinessMask() & kRequiredPassMask) == kRequiredPassMask;
}

bool upstreamDirectLightExecutorPassIncluded(const std::string& name) {
    return name == "primary" ||
        name == "precompute_light_neighborhoods" ||
        name == "generate_initial_samples" ||
        name == "visibility" ||
        name == "temporal_reuse" ||
        name == "spatial_reuse" ||
        name == "direct_light" ||
        name == "final_compose";
}

class ScopedGpuProfile {
public:
    ScopedGpuProfile(VkCommandBuffer cmd, const char* name) : cmd_(cmd) {
        if (cmd_ != VK_NULL_HANDLE && Renderer::gpuProfiler.isEnabled()) {
            Renderer::gpuProfiler.beginModule(cmd_, name);
            open_ = true;
        }
    }

    ~ScopedGpuProfile() {
        close();
    }

    void close() {
        if (open_) {
            Renderer::gpuProfiler.endModule(cmd_);
            open_ = false;
        }
    }

private:
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    bool open_ = false;
};
}

RayTracingModule::RayTracingModule() {}

std::string RayTracingModule::diagnosticFeatureTruth() const {
    std::ostringstream out;
    const bool accumulating = Renderer::options.offlineState == 2;
    const uint32_t directLightBackend = effectiveDirectLightBackend();
#ifdef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
    constexpr bool directLightPipelineCompiled = true;
#else
    constexpr bool directLightPipelineCompiled = false;
#endif
#ifdef MCVR_ENABLE_RTXDI
    constexpr bool rtxdiCompiled = true;
#else
    constexpr bool rtxdiCompiled = false;
#endif
    const bool directLightPipelinePossible = directLightPipelineCompiled && directLightBackend != 0;
    const bool rtxdiBackendPossible = directLightPipelineCompiled && rtxdiCompiled && directLightBackend == 2;
    const uint32_t upstreamRtPassMask = upstreamRtPassReadinessMask();
    const bool upstreamRtReady = upstreamRtPackReady();
#ifdef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
    bool directLightResourcesReady = false;
    uint32_t directLightReservoirWidth = 0;
    uint32_t directLightReservoirHeight = 0;
    uint64_t directLightReservoirPixels = 0;
    if (!directLightReservoirPingImages_.empty()) {
        for (const auto& image : directLightReservoirPingImages_) {
            if (image) {
                directLightResourcesReady = true;
                directLightReservoirWidth = image->width();
                directLightReservoirHeight = image->height();
                directLightReservoirPixels =
                    static_cast<uint64_t>(directLightReservoirWidth) * directLightReservoirHeight;
                break;
            }
        }
    }
    int directLightLightCount = 0;
    if (worldPrepare_) {
        for (const auto& ctx : worldPrepare_->contexts_) {
            if (ctx) {
                directLightLightCount = std::max(directLightLightCount, ctx->areaLightCount);
            }
        }
    }
    const bool upstreamExecutorReady =
        directLightBackend == 1 &&
        directLightUpstreamPackRuntimeReady_ &&
        directLightUpstreamRuntimeResourcesReady_ &&
        directLightUpstreamPassRuntimeReady_ &&
        directLightUpstreamShaderCompileReady_ &&
        directLightUpstreamPipelineReady_ &&
        directLightUpstreamSbtReady_;
    const bool upstreamExecutorActive = kAllowUpstreamRtExecutorDispatch && upstreamExecutorReady;
    const bool directLightScaffoldActive = kAllowDirectLightScaffoldDispatch && directLightBackend == 2;
#else
    constexpr bool directLightResourcesReady = false;
    constexpr uint32_t directLightReservoirWidth = 0;
    constexpr uint32_t directLightReservoirHeight = 0;
    constexpr uint64_t directLightReservoirPixels = 0;
    constexpr int directLightLightCount = 0;
    constexpr bool upstreamExecutorReady = false;
    constexpr bool upstreamExecutorActive = false;
    constexpr bool directLightScaffoldActive = false;
#endif

    out << "directLightBackend:" << directLightBackend
        << ",directLightPipelineCompiled:" << (directLightPipelineCompiled ? 1 : 0)
        << ",rtxdiCompiled:" << (rtxdiCompiled ? 1 : 0)
        << ",directLightPipelinePossible:" << (directLightPipelinePossible ? 1 : 0)
        << ",rtxdiBackendPossible:" << (rtxdiBackendPossible ? 1 : 0)
        << ",directLightPipelineActive:" << ((upstreamExecutorActive || directLightScaffoldActive) ? 1 : 0)
        << ",directLightVisualOverrideActive:0"
        << ",directLightScaffoldVisualSubstituteActive:0"
        << ",directLightRuntimeCompilerReady:1"
        << ",directLightUpstreamExecutionActive:" << (upstreamExecutorActive ? 1 : 0)
        << ",directLightUpstreamExecutorReady:" << (upstreamExecutorReady ? 1 : 0)
#ifdef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
        << ",directLightUpstreamExecutorBlockedNoPassRuntime:" << (directLightUpstreamPassRuntimeReady_ ? 0 : 1)
        << ",directLightUpstreamExecutorBlockedNoRuntimeDescriptors:" << (directLightUpstreamRuntimeResourcesReady_ ? 0 : 1)
        << ",directLightUpstreamPackRuntimeReady:" << (directLightUpstreamPackRuntimeReady_ ? 1 : 0)
        << ",directLightUpstreamRuntimeResourcesReady:" << (directLightUpstreamRuntimeResourcesReady_ ? 1 : 0)
        << ",directLightUpstreamPassRuntimeReady:" << (directLightUpstreamPassRuntimeReady_ ? 1 : 0)
        << ",directLightUpstreamShaderCompileReady:" << (directLightUpstreamShaderCompileReady_ ? 1 : 0)
        << ",directLightUpstreamPipelineReady:" << (directLightUpstreamPipelineReady_ ? 1 : 0)
        << ",directLightUpstreamSbtReady:" << (directLightUpstreamSbtReady_ ? 1 : 0)
#else
        << ",directLightUpstreamExecutorBlockedNoPassRuntime:1"
        << ",directLightUpstreamExecutorBlockedNoRuntimeDescriptors:1"
        << ",directLightUpstreamPackRuntimeReady:0"
        << ",directLightUpstreamRuntimeResourcesReady:0"
        << ",directLightUpstreamPassRuntimeReady:0"
        << ",directLightUpstreamShaderCompileReady:0"
        << ",directLightUpstreamPipelineReady:0"
        << ",directLightUpstreamSbtReady:0"
#endif
        << ",directLightPerfComparisonValid:" << (upstreamExecutorReady ? 1 : 0)
        << ",directLightUpstreamPackReady:" << (upstreamRtReady ? 1 : 0)
        << ",directLightUpstreamPassMask:" << upstreamRtPassMask
        << ",directLightUpstreamPrimaryReady:" << ((upstreamRtPassMask & (1u << 0)) ? 1 : 0)
        << ",directLightUpstreamInitialReady:" << ((upstreamRtPassMask & (1u << 1)) ? 1 : 0)
        << ",directLightUpstreamVisibilityReady:" << ((upstreamRtPassMask & (1u << 2)) ? 1 : 0)
        << ",directLightUpstreamTemporalReady:" << ((upstreamRtPassMask & (1u << 3)) ? 1 : 0)
        << ",directLightUpstreamSpatialReady:" << ((upstreamRtPassMask & (1u << 4)) ? 1 : 0)
        << ",directLightUpstreamDirectLightReady:" << ((upstreamRtPassMask & (1u << 5)) ? 1 : 0)
        << ",directLightUpstreamWorldReady:" << ((upstreamRtPassMask & (1u << 6)) ? 1 : 0)
        << ",directLightResourcesReady:" << (directLightResourcesReady ? 1 : 0)
        << ",directLightReservoirWidth:" << directLightReservoirWidth
        << ",directLightReservoirHeight:" << directLightReservoirHeight
        << ",directLightReservoirPixels:" << directLightReservoirPixels
        << ",directLightLightCount:" << directLightLightCount
        << ",directLightEmissiveChunks:0"
#ifdef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
        << ",directLightPrimaryPipelineReady:" << (directLightPrimaryPipeline_ != VK_NULL_HANDLE ? 1 : 0)
        << ",directLightInitialPipelineReady:" << (directLightInitialPipeline_ != VK_NULL_HANDLE ? 1 : 0)
        << ",directLightUtilityPipelineReady:" << (directLightUtilityPipeline_ != VK_NULL_HANDLE ? 1 : 0)
        << ",directLightExternalized:0"
        << ",directLightPrimarySurfacePixels:" << directLightLastCounters_[0]
        << ",directLightValidReservoirs:" << directLightLastCounters_[1]
        << ",directLightTemporalReused:" << directLightLastCounters_[2]
        << ",directLightSpatialTaps:" << directLightLastCounters_[3]
        << ",directLightVisibilityRays:" << directLightLastCounters_[4]
        << ",directLightDepthHits:" << directLightLastCounters_[5]
        << ",directLightShadeValidPixels:" << directLightLastCounters_[6]
        << ",directLightDirectionalCandidates:" << directLightLastCounters_[7]
        << ",directLightDirectionalVisibilityRays:" << directLightLastCounters_[8]
        << ",directLightDirectionalUnoccluded:" << directLightLastCounters_[9]
        << ",directLightAreaTilePixels:" << directLightLastCounters_[10]
        << ",directLightAreaCandidates:" << directLightLastCounters_[11]
        << ",directLightAreaVisibilityRays:" << directLightLastCounters_[12]
        << ",directLightAreaUnoccluded:" << directLightLastCounters_[13]
        << ",directLightOutputPixels:" << directLightLastCounters_[14]
        << ",directLightNonzeroOutputPixels:" << directLightLastCounters_[15]
#else
        << ",directLightPrimaryPipelineReady:0"
        << ",directLightInitialPipelineReady:0"
        << ",directLightUtilityPipelineReady:0"
        << ",directLightExternalized:0"
        << ",directLightPrimarySurfacePixels:0"
        << ",directLightValidReservoirs:0"
        << ",directLightTemporalReused:0"
        << ",directLightSpatialTaps:0"
        << ",directLightVisibilityRays:0"
        << ",directLightDepthHits:0"
        << ",directLightShadeValidPixels:0"
        << ",directLightDirectionalCandidates:0"
        << ",directLightDirectionalVisibilityRays:0"
        << ",directLightDirectionalUnoccluded:0"
        << ",directLightAreaTilePixels:0"
        << ",directLightAreaCandidates:0"
        << ",directLightAreaVisibilityRays:0"
        << ",directLightAreaUnoccluded:0"
        << ",directLightOutputPixels:0"
        << ",directLightNonzeroOutputPixels:0"
#endif
        << ",sharcOption:" << (Renderer::options.sharcEnabled ? 1 : 0)
        << ",offlineAccumulating:" << (accumulating ? 1 : 0);

#ifdef MCVR_ENABLE_SHARC
    constexpr bool sharcCompiled = true;
#ifdef MCVR_ENABLE_SHARC_BUFFERS
    constexpr bool sharcBuffersCompiled = true;
#else
    constexpr bool sharcBuffersCompiled = false;
#endif
#ifdef MCVR_ENABLE_SHARC_UPDATE_PASS
    constexpr bool sharcUpdateCompiled = true;
#else
    constexpr bool sharcUpdateCompiled = false;
#endif
#ifdef MCVR_ENABLE_SHARC_RESOLVE_PASS
    constexpr bool sharcResolveCompiled = true;
#else
    constexpr bool sharcResolveCompiled = false;
#endif
#ifdef MCVR_ENABLE_SHARC_QUERY_PASS
    constexpr bool sharcQueryPassCompiled = true;
#else
    constexpr bool sharcQueryPassCompiled = false;
#endif
#ifdef MCVR_ENABLE_SHARC_MAIN_TRACE_QUERY
    constexpr bool sharcMainTraceCompiled = true;
#else
    constexpr bool sharcMainTraceCompiled = false;
#endif
    const bool buffersAllocated = sharcHashEntries_ && sharcAccumulation_ && sharcResolved_;
    size_t sharcSbtReady = 0;
    for (const auto& sbt : sharcUpdateSbts_) {
        if (sbt) sharcSbtReady++;
    }
    const bool updatePipelineReady = sharcUpdatePipeline_ != nullptr && sharcSbtReady > 0;
    const bool resolvePipelineReady = sharcResolvePipeline_ != VK_NULL_HANDLE;
    const bool queryPipelineReady = sharcQueryPipeline_ != VK_NULL_HANDLE;
    const uint32_t mainTraceWarmupFrames = effectiveSharcMainTraceWarmupFrames();
    const bool mainTraceQueryPossible = sharcMainTraceCompiled && Renderer::options.sharcEnabled && !accumulating
        && buffersAllocated && sharcBuffersInitialized_ && sharcDispatchReady_;
    const bool mainTraceQueryActive = mainTraceQueryPossible && sharcFrameIndex_ >= mainTraceWarmupFrames;
    const bool updateResolvePossible = Renderer::options.sharcEnabled && !accumulating && buffersAllocated
        && (sharcUpdateCompiled ? updatePipelineReady : true)
        && (sharcResolveCompiled ? resolvePipelineReady : true);
    const bool queryPassPossible = sharcQueryPassCompiled && Renderer::options.sharcEnabled && !accumulating
        && effectiveSharcQueryMode() > 0 && buffersAllocated && sharcBuffersInitialized_
        && sharcDispatchReady_ && queryPipelineReady;
#else
    constexpr bool sharcCompiled = false;
    constexpr bool sharcBuffersCompiled = false;
    constexpr bool sharcUpdateCompiled = false;
    constexpr bool sharcResolveCompiled = false;
    constexpr bool sharcQueryPassCompiled = false;
    constexpr bool sharcMainTraceCompiled = false;
    constexpr bool buffersAllocated = false;
    constexpr size_t sharcSbtReady = 0;
    constexpr bool updatePipelineReady = false;
    constexpr bool resolvePipelineReady = false;
    constexpr bool queryPipelineReady = false;
    constexpr uint32_t mainTraceWarmupFrames = 0;
    constexpr bool mainTraceQueryPossible = false;
    constexpr bool mainTraceQueryActive = false;
    constexpr bool updateResolvePossible = false;
    constexpr bool queryPassPossible = false;
#endif

    out << ",sharcCompiled:" << (sharcCompiled ? 1 : 0)
        << ",sharcBuffersCompiled:" << (sharcBuffersCompiled ? 1 : 0)
        << ",sharcUpdateCompiled:" << (sharcUpdateCompiled ? 1 : 0)
        << ",sharcResolveCompiled:" << (sharcResolveCompiled ? 1 : 0)
        << ",sharcQueryPassCompiled:" << (sharcQueryPassCompiled ? 1 : 0)
        << ",sharcMainTraceCompiled:" << (sharcMainTraceCompiled ? 1 : 0)
        << ",sharcQueryMode:" << effectiveSharcQueryMode()
        << ",sharcMainTraceQueryMode:" << MCVR_SHARC_MAIN_TRACE_QUERY_MODE
        << ",sharcBuffersAllocated:" << (buffersAllocated ? 1 : 0)
#ifdef MCVR_ENABLE_SHARC
        << ",sharcBuffersInitialized:" << (sharcBuffersInitialized_ ? 1 : 0)
        << ",sharcResizePending:" << (sharcResizePending_ ? 1 : 0)
#else
        << ",sharcBuffersInitialized:0"
        << ",sharcResizePending:0"
#endif
        << ",sharcUpdatePipeline:" << (updatePipelineReady ? 1 : 0)
        << ",sharcResolvePipeline:" << (resolvePipelineReady ? 1 : 0)
        << ",sharcQueryPipeline:" << (queryPipelineReady ? 1 : 0)
        << ",sharcMainTraceQueryPossible:" << (mainTraceQueryPossible ? 1 : 0)
        << ",sharcMainTraceQueryActive:" << (mainTraceQueryActive ? 1 : 0)
        << ",sharcQueryPassPossible:" << (queryPassPossible ? 1 : 0)
        << ",sharcMainTraceWarmupFrames:" << mainTraceWarmupFrames
        << ",sharcUpdateResolvePossible:" << (updateResolvePossible ? 1 : 0)
#ifdef MCVR_ENABLE_SHARC
        << ",sharcUpdateSbtReady:" << sharcSbtReady
        << ",sharcUpdateSbtTotal:" << sharcUpdateSbts_.size()
        << ",sharcRequestedCapacityExponent:" << Renderer::options.sharcCapacityExponent
        << ",sharcEffectiveCapacityExponent:" << effectiveSharcCapacityExponent()
        << ",sharcRequestedUpdateBounces:" << Renderer::options.sharcUpdateBounces
        << ",sharcEffectiveUpdateBounces:" << effectiveSharcUpdateBounces()
        << ",sharcStreamStableFrames:" << sharcStreamStableFrames_
        << ",sharcDispatchReady:" << (sharcDispatchReady_ ? 1 : 0)
        << ",sharcCapacity:" << sharcCapacity_
        << ",sharcFrameIndex:" << sharcFrameIndex_
        << ",sharcCandidates:" << sharcQueryLastCounters_[0]
        << ",sharcActiveTerminated:" << sharcQueryLastCounters_[1]
        << ",sharcQueryAttempts:" << sharcQueryLastCounters_[2]
        << ",sharcQueryHits:" << sharcQueryLastCounters_[3]
        << ",sharcQueryMisses:" << sharcQueryLastCounters_[4]
        << ",sharcQueryInvalid:" << sharcQueryLastCounters_[5]
        << ",sharcQueryVoxelSkips:" << sharcQueryLastCounters_[6]
        << ",sharcQueryWrites:" << sharcQueryLastCounters_[7];
#else
        << ",sharcUpdateSbtReady:0"
        << ",sharcUpdateSbtTotal:0"
        << ",sharcCapacity:0"
        << ",sharcFrameIndex:0"
        << ",sharcCandidates:0"
        << ",sharcActiveTerminated:0"
        << ",sharcQueryAttempts:0"
        << ",sharcQueryHits:0"
        << ",sharcQueryMisses:0"
        << ",sharcQueryInvalid:0"
        << ",sharcQueryVoxelSkips:0"
        << ",sharcQueryWrites:0";
#endif

    return out.str();
}

void RayTracingModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();

    hdrNoisyOutputImages_.resize(size);
    diffuseAlbedoImages_.resize(size);
    specularAlbedoImages_.resize(size);
    normalRoughnessImages_.resize(size);
    motionVectorImages_.resize(size);
    linearDepthImages_.resize(size);
    specularHitDepthImages_.resize(size);
    firstHitDepthImages_.resize(size);
    firstHitDiffuseDirectLightImages_.resize(size);
    firstHitDiffuseIndirectLightImages_.resize(size);
    firstHitSpecularImages_.resize(size);
    firstHitClearImages_.resize(size);
    firstHitBaseEmissionImages_.resize(size);
    directLightDepthImages_.resize(size);
    diffuseRayDirHitDistImages_.resize(size);
    specularRayDirHitDistImages_.resize(size);
    reflectionMvImages_.resize(size);
    animatedTexMaskImages_.resize(size);
    particleMaskImages_.resize(size);
    biasMaskImages_.resize(size);
    rtHitDistImages_.resize(size);
    motionVectors3DImages_.resize(size);
    gbufferMetallicImages_.resize(size);
    gbufferShadingModelIdImages_.resize(size);
    gbufferMaterialIdImages_.resize(size);
    positionViewSpaceImages_.resize(size);
    transparencyLayerImages_.resize(size);
    transparencyLayerOpacityImages_.resize(size);
    transparencyLayerMvecsImages_.resize(size);
    upstreamFogImages_.resize(size);
    upstreamFirstHitRefractionImages_.resize(size);
    sharcCandidatePosHitTImages_.resize(size);
    sharcCandidateNormalRoughnessImages_.resize(size);
    sharcCandidateThroughputImages_.resize(size);
    sharcCandidatePrefixRadianceFlagsImages_.resize(size);
    sharcQueryCounterBuffers_.resize(size);
#ifdef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
    directLightPrimarySurfaceImages_.resize(size);
    directLightReservoirPingImages_.resize(size);
    directLightReservoirPongImages_.resize(size);
    directLightOutputImages_.resize(size);
    directLightCounterBuffers_.resize(size);
#endif

    atmosphere_ = Atmosphere::create(framework, shared_from_this());
    worldPrepare_ = WorldPrepare::create(framework, shared_from_this());
}

bool RayTracingModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                              std::vector<VkFormat> &formats,
                                              uint32_t frameIndex) {
    return true;
}

bool RayTracingModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                               std::vector<VkFormat> &formats,
                                               uint32_t frameIndex) {
    uint32_t width, height;
    bool set = false;
    for (auto &image : images) {
        if (image != nullptr) {
            if (!set) {
                width = image->width();
                height = image->height();
                set = true;
            } else {
                if (image->width() != width || image->height() != height) { return false; }
            }
        }
    }

    if (!set) { return false; }

    auto framework = framework_.lock();
    if (!framework) return false;
    for (int i = 0; i < images.size(); i++) {
        if (images[i] == nullptr) {
            images[i] = vk::DeviceLocalImage::create(
                framework->device(), framework->vma(), false, width, height, 1, formats[i],
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        }
    }

    hdrNoisyOutputImages_[frameIndex] = images[0];
    diffuseAlbedoImages_[frameIndex] = images[1];
    specularAlbedoImages_[frameIndex] = images[2];
    normalRoughnessImages_[frameIndex] = images[3];
    motionVectorImages_[frameIndex] = images[4];
    linearDepthImages_[frameIndex] = images[5];
    specularHitDepthImages_[frameIndex] = images[6];
    firstHitDepthImages_[frameIndex] = images[7];
    firstHitDiffuseDirectLightImages_[frameIndex] = images[8];
    firstHitDiffuseIndirectLightImages_[frameIndex] = images[9];
    firstHitSpecularImages_[frameIndex] = images[10];
    firstHitClearImages_[frameIndex] = images[11];
    firstHitBaseEmissionImages_[frameIndex] = images[12];
    directLightDepthImages_[frameIndex] = images[13];
    diffuseRayDirHitDistImages_[frameIndex] = images[14];
    specularRayDirHitDistImages_[frameIndex] = images[15];
    reflectionMvImages_[frameIndex] = images[16];
    animatedTexMaskImages_[frameIndex] = images[17];
    particleMaskImages_[frameIndex] = images[18];
    biasMaskImages_[frameIndex] = images[19];
    rtHitDistImages_[frameIndex] = images[20];
    motionVectors3DImages_[frameIndex] = images[21];
    gbufferMetallicImages_[frameIndex] = images[22];
    gbufferShadingModelIdImages_[frameIndex] = images[23];
    gbufferMaterialIdImages_[frameIndex] = images[24];
    positionViewSpaceImages_[frameIndex] = images[25];
    transparencyLayerImages_[frameIndex] = images[26];
    transparencyLayerOpacityImages_[frameIndex] = images[27];
    transparencyLayerMvecsImages_[frameIndex] = images[28];

#ifdef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
    auto ensureUpstreamImage =
        [&](std::vector<std::shared_ptr<vk::DeviceLocalImage>> &target, VkFormat format) {
            auto &img = target[frameIndex];
            if (!img || img->width() != width || img->height() != height) {
                img = vk::DeviceLocalImage::create(
                    framework->device(), framework->vma(), false, width, height, 1, format,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
            }
        };
    ensureUpstreamImage(upstreamFogImages_, VK_FORMAT_R16G16B16A16_SFLOAT);
    ensureUpstreamImage(upstreamFirstHitRefractionImages_, VK_FORMAT_R16G16B16A16_SFLOAT);
#endif

    // Publish depth and motion vectors for frame generation resource tagging
    if (Renderer::frameGenDepthImages.size() <= frameIndex) {
        Renderer::frameGenDepthImages.resize(frameIndex + 1);
        Renderer::frameGenMotionVectorImages.resize(frameIndex + 1);
    }
    Renderer::frameGenDepthImages[frameIndex] = linearDepthImages_[frameIndex];
    Renderer::frameGenMotionVectorImages[frameIndex] = motionVectorImages_[frameIndex];

    // Create reservoir images for ReSTIR DI (only once, shared across frames)
    if (Renderer::options.restirEnabled) {
        if (!reservoirImages_[0]) {
            for (int r = 0; r < 2; r++) {
                reservoirImages_[r] = vk::DeviceLocalImage::create(
                    framework->device(), framework->vma(), false, width, height, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            }
        }

        // Create bounce ReSTIR DI reservoir images (per-bounce temporal reuse, bounces 1-3)
        if (!bounceReservoirImages_[0]) {
            for (int b = 0; b < 3; b++) {
                bounceReservoirImages_[b] = vk::DeviceLocalImage::create(
                    framework->device(), framework->vma(), false, width, height, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            }
        }
    }

#ifdef MCVR_ENABLE_SHARC_QUERY_PASS
    auto ensureCandidateImage =
        [&](std::vector<std::shared_ptr<vk::DeviceLocalImage>> &target) {
            auto &img = target[frameIndex];
            if (!img || img->width() != width || img->height() != height) {
                img = vk::DeviceLocalImage::create(
                    framework->device(), framework->vma(), false, width, height, 1,
                    VK_FORMAT_R32G32B32A32_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            }
        };
    ensureCandidateImage(sharcCandidatePosHitTImages_);
    ensureCandidateImage(sharcCandidateNormalRoughnessImages_);
    ensureCandidateImage(sharcCandidateThroughputImages_);
    ensureCandidateImage(sharcCandidatePrefixRadianceFlagsImages_);
    if (!sharcQueryCounterBuffers_[frameIndex]) {
        sharcQueryCounterBuffers_[frameIndex] = vk::HostVisibleBuffer::create(
            framework->vma(), framework->device(), kSharcQueryCounterCount * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }
#endif

    return true;
}

void RayTracingModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    auto parseBool = [](const std::string &value) {
        return value == "1" || value == "true" || value == "True" || value == "TRUE";
    };

    for (int i = 0; i < attributeCount; i++) {
        const std::string &key = attributeKVs[2 * i];
        const std::string &value = attributeKVs[2 * i + 1];

        if (key == "render_pipeline.module.dlss.attribute.num_ray_bounces") {
            numRayBounces_ = std::stoi(value);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.use_jitter") {
            useJitter_ = parseBool(value);
            Renderer::instance().buffers()->setUseJitter(useJitter_);
        }
    }
}

void RayTracingModule::build() {
    atmosphere_->build();
    worldPrepare_->build();

    auto framework = framework_.lock();
    if (!framework) return;
    auto worldPipeline = worldPipeline_.lock();
    if (!worldPipeline) return;
    uint32_t size = framework->swapchain()->imageCount();

    contexts_.resize(size);

    initDescriptorTables();
    initEnergyLUT();
    initImages();
    initPipeline();
    initSBT();
    initSpatialPipeline();
    initClusterPipeline();
#ifdef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
    initDirectLightPipeline();
    initUpstreamDirectLightRuntime();
    initUpstreamDirectLightDescriptorTables();
#endif
#if defined(MCVR_ENABLE_SHARC) && defined(MCVR_ENABLE_SHARC_BUFFERS)
    sharcCapacity_ = effectiveSharcCapacity();
    if (Renderer::options.sharcEnabled) {
        initSharcBuffers();
#ifdef MCVR_ENABLE_SHARC_UPDATE_PASS
        initSharcUpdatePipeline();
#endif
#ifdef MCVR_ENABLE_SHARC_RESOLVE_PASS
        initSharcResolvePipeline();
#endif
#ifdef MCVR_ENABLE_SHARC_QUERY_PASS
        initSharcQueryPipeline();
#endif
    }
#endif
    initAccumulationPipeline();

    // Initialize blue noise buffers (Owen-scrambled Sobol + spatial scrambling tile)
    // Upload immediately via important-upload path (synchronous staging→device before first RT dispatch)
    blueNoise_ = std::make_shared<BlueNoise>(framework->device(), framework->vma());

    for (int i = 0; i < size; i++) {
        contexts_[i] =
            RayTracingModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i], shared_from_this());

        // set rayTracingModuleContext of sub-modules, order is important
        atmosphere_->contexts_[i]->rayTracingModuleContext =
            std::static_pointer_cast<RayTracingModuleContext>(contexts_[i]);
        worldPrepare_->contexts_[i]->rayTracingModuleContext =
            std::static_pointer_cast<RayTracingModuleContext>(contexts_[i]);
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &RayTracingModule::contexts() {
    return contexts_;
}

void RayTracingModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                                   std::shared_ptr<vk::DeviceLocalImage> image,
                                   int index) {
    auto framework = framework_.lock();
    if (!framework) return;

    uint32_t size = framework->swapchain()->imageCount();
    for (int i = 0; i < size; i++) {
        if (rayTracingDescriptorTables_[i] != nullptr)
            rayTracingDescriptorTables_[i]->bindSamplerImage(sampler, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                             0, 0, index);
#ifdef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
        if (i < static_cast<int>(directLightUpstreamDescriptorTables_.size()) &&
            directLightUpstreamDescriptorTables_[i] != nullptr) {
            directLightUpstreamDescriptorTables_[i]->bindSamplerImage(
                sampler, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 0, index);
        }
#endif
    }
}

void RayTracingModule::initAccumulationPipeline() {
    auto framework = framework_.lock();
    if (!framework) return;
    auto device = framework->device();
    VkDevice dev = device->vkDevice();
    uint32_t size = framework->swapchain()->imageCount();

    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    accumShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/accumulate_comp.spv").string());
    if (!accumShader_) {
        std::cerr << "[Offline] Failed to load accumulate_comp.spv — accumulation disabled" << std::endl;
        return;
    }

    // Create RGBA32F accumulation buffer at render resolution
    uint32_t w = hdrNoisyOutputImages_[0]->width();
    uint32_t h = hdrNoisyOutputImages_[0]->height();
    Renderer::accumBufferImage = vk::DeviceLocalImage::create(
        device, framework->vma(), false, w, h, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    // Transition from UNDEFINED to GENERAL (required before first storage access)
    Renderer::accumBufferImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    // Descriptor set layout: 2 storage images (accumBuffer, noisyInput)
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &Renderer::accumDescSetLayout);

    // Pipeline layout with push constant (1 int = 4 bytes)
    VkPushConstantRange pushRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &Renderer::accumDescSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(dev, &pipelineLayoutInfo, nullptr, &Renderer::accumPipelineLayout);

    // Compute pipeline
    VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = accumShader_->vkShaderModule();
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = Renderer::accumPipelineLayout;
    vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &Renderer::accumPipeline);

    // Descriptor pool
    VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * size}};
    VkDescriptorPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = size;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    vkCreateDescriptorPool(dev, &poolInfo, nullptr, &Renderer::accumDescPool);

    // Allocate descriptor sets
    std::vector<VkDescriptorSetLayout> layouts(size, Renderer::accumDescSetLayout);
    VkDescriptorSetAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = Renderer::accumDescPool;
    allocInfo.descriptorSetCount = size;
    allocInfo.pSetLayouts = layouts.data();
    Renderer::accumDescSets.resize(size);
    vkAllocateDescriptorSets(dev, &allocInfo, Renderer::accumDescSets.data());

    Renderer::accumPipelineReady = true;
    std::cout << "[Offline] Accumulation pipeline initialized (" << w << "x" << h << " RGBA32F)" << std::endl;

    // --- Emission compose pipeline (DLSS-RR: subtract emission before, add after) ---
    auto emissionShader = vk::Shader::create(device, (shaderPath / "world/ray_tracing/emission_compose_comp.spv").string());
    if (!emissionShader) {
        std::cerr << "[Offline] Failed to load emission_compose_comp.spv — emission preservation disabled" << std::endl;
        return;
    }

    // Same descriptor set layout: 2 storage images (target, emission)
    VkDescriptorSetLayoutCreateInfo ecLayoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ecLayoutInfo.bindingCount = (uint32_t)bindings.size();
    ecLayoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(dev, &ecLayoutInfo, nullptr, &Renderer::emissionComposeDescSetLayout);

    // Pipeline layout with push constant: float preExposure + int mode = 8 bytes
    VkPushConstantRange ecPushRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo ecPipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    ecPipelineLayoutInfo.setLayoutCount = 1;
    ecPipelineLayoutInfo.pSetLayouts = &Renderer::emissionComposeDescSetLayout;
    ecPipelineLayoutInfo.pushConstantRangeCount = 1;
    ecPipelineLayoutInfo.pPushConstantRanges = &ecPushRange;
    vkCreatePipelineLayout(dev, &ecPipelineLayoutInfo, nullptr, &Renderer::emissionComposePipelineLayout);

    VkComputePipelineCreateInfo ecPipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ecPipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ecPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ecPipelineInfo.stage.module = emissionShader->vkShaderModule();
    ecPipelineInfo.stage.pName = "main";
    ecPipelineInfo.layout = Renderer::emissionComposePipelineLayout;
    vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ecPipelineInfo, nullptr, &Renderer::emissionComposePipeline);

    VkDescriptorPoolSize ecPoolSizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * size}};
    VkDescriptorPoolCreateInfo ecPoolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ecPoolInfo.maxSets = size;
    ecPoolInfo.poolSizeCount = 1;
    ecPoolInfo.pPoolSizes = ecPoolSizes;
    vkCreateDescriptorPool(dev, &ecPoolInfo, nullptr, &Renderer::emissionComposeDescPool);

    std::vector<VkDescriptorSetLayout> ecLayouts(size, Renderer::emissionComposeDescSetLayout);
    VkDescriptorSetAllocateInfo ecAllocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ecAllocInfo.descriptorPool = Renderer::emissionComposeDescPool;
    ecAllocInfo.descriptorSetCount = size;
    ecAllocInfo.pSetLayouts = ecLayouts.data();
    Renderer::emissionComposeDescSets.resize(size);
    vkAllocateDescriptorSets(dev, &ecAllocInfo, Renderer::emissionComposeDescSets.data());

    Renderer::emissionComposePipelineReady = true;
    std::cout << "[Offline] Emission compose pipeline initialized" << std::endl;
}

void RayTracingModule::preClose() {
    auto framework = framework_.lock();
    if (framework) {
        VkDevice dev = framework->device()->vkDevice();
        if (spatialPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(dev, spatialPipeline_, nullptr);
        if (spatialPipelineLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, spatialPipelineLayout_, nullptr);
        if (spatialDescSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, spatialDescSetLayout_, nullptr);
        if (spatialDescPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, spatialDescPool_, nullptr);
        if (clusterPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(dev, clusterPipeline_, nullptr);
        if (clusterPipelineLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, clusterPipelineLayout_, nullptr);
        if (clusterDescSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, clusterDescSetLayout_, nullptr);
        if (clusterDescPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, clusterDescPool_, nullptr);
#ifdef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
        if (directLightPrimaryPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(dev, directLightPrimaryPipeline_, nullptr);
        if (directLightPrimaryPipelineLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, directLightPrimaryPipelineLayout_, nullptr);
        if (directLightPrimaryDescSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, directLightPrimaryDescSetLayout_, nullptr);
        if (directLightPrimaryDescPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, directLightPrimaryDescPool_, nullptr);
        if (directLightInitialPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(dev, directLightInitialPipeline_, nullptr);
        if (directLightInitialPipelineLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, directLightInitialPipelineLayout_, nullptr);
        if (directLightInitialDescSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, directLightInitialDescSetLayout_, nullptr);
        if (directLightInitialDescPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, directLightInitialDescPool_, nullptr);
        if (directLightUtilityPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(dev, directLightUtilityPipeline_, nullptr);
        if (directLightUtilityPipelineLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, directLightUtilityPipelineLayout_, nullptr);
        if (directLightUtilityDescSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, directLightUtilityDescSetLayout_, nullptr);
        if (directLightUtilityDescPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, directLightUtilityDescPool_, nullptr);
#endif
        if (sharcResolvePipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(dev, sharcResolvePipeline_, nullptr);
        if (sharcResolvePipelineLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, sharcResolvePipelineLayout_, nullptr);
        if (sharcQueryPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(dev, sharcQueryPipeline_, nullptr);
        if (sharcQueryPipelineLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, sharcQueryPipelineLayout_, nullptr);
        if (sharcQueryDescSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, sharcQueryDescSetLayout_, nullptr);
        if (sharcQueryDescPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, sharcQueryDescPool_, nullptr);
        if (Renderer::accumPipeline != VK_NULL_HANDLE) vkDestroyPipeline(dev, Renderer::accumPipeline, nullptr);
        if (Renderer::accumPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, Renderer::accumPipelineLayout, nullptr);
        if (Renderer::accumDescSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, Renderer::accumDescSetLayout, nullptr);
        if (Renderer::accumDescPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, Renderer::accumDescPool, nullptr);
        Renderer::accumPipelineReady = false;
        if (Renderer::emissionComposePipeline != VK_NULL_HANDLE) vkDestroyPipeline(dev, Renderer::emissionComposePipeline, nullptr);
        if (Renderer::emissionComposePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, Renderer::emissionComposePipelineLayout, nullptr);
        if (Renderer::emissionComposeDescSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, Renderer::emissionComposeDescSetLayout, nullptr);
        if (Renderer::emissionComposeDescPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, Renderer::emissionComposeDescPool, nullptr);
        Renderer::emissionComposePipelineReady = false;
    }
}

void RayTracingModule::initDescriptorTables() {
    auto framework = framework_.lock();
    if (!framework) return;

    uint32_t size = framework->swapchain()->imageCount();
    rayTracingDescriptorTables_.resize(size);

    for (int i = 0; i < size; i++) {
        rayTracingDescriptorTables_[i] =
            vk::DescriptorTableBuilder{}
                .beginDescriptorLayoutSet() // set 0
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 4096, // bindless texture array
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1, // world atmosphere LUT
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 2, // world atmosphere cube map
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 3, // block sprite albedo sampler2DArray
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 4, // block sprite specular sampler2DArray
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 5, // block sprite normal sampler2DArray
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .beginDescriptorLayoutSet() // set 1
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0, // binding 0: TLAS(s)
                    .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1, // binding 1: blasOffsets
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 2, // binding 2: vertex buffer addrs
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 3, // binding 3: index buffer addrs
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 4, // binding 4: last vertex buffer addrs
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 5, // binding 5: last index buffer addrs
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 6, // binding 6: last obj to world mat
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 7, // binding 7: texture mapping
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 8, // binding 8: area light SSBO
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 9, // binding 9: tile light buffer (light clustering)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 10, // binding 10: Blender PBR texture mapping (cold path)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 11, // binding 11: Material class mapping (unified material system)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 13, // binding 13: SpriteRegistry SSBO (texture array metadata)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .beginDescriptorLayoutSet() // set 2
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0, // binding 0: current world ubo
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1, // binding 1: last world ubo
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 2, // binding 2: sky ubo
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_VERTEX_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 4, // binding 4: energy compensation LUT (multi-scatter GGX + EON diffuse)
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 9, // Blue noise Sobol (moved to set 2 to avoid set 1 binding count issue)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 10, // Blue noise scrambling tile
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .beginDescriptorLayoutSet() // set 3
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0, // binding 0: hdrNoisyImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1, // binding 1: diffuseAlbedoImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 2, // binding 2: specularAlbedoImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 3, // binding 3: normalRoughnessImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 4, // binding 4: motionVectorImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 5, // binding 5: linearDepthImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 6, // binding 6: specularHitDepth
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 7, // binding 7: firstHitDepthImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 8, // binding 8: firstHitDiffuseDirectLightImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 9, // binding 9: firstHitDiffuseIndirectLightImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 10, // binding 10: firstHitSpecularImage;
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 11, // binding 11: firstHitClearImage;
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 12, // binding 12: firstHitBaseEmissionImage;
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 13, // binding 13: directLightDepthImage;
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 14, // binding 14: reservoirCurrentImage (ReSTIR DI write)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 15, // binding 15: reservoirPreviousImage (ReSTIR DI read)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 16, // binding 16: diffuseRayDirHitDistImage (DLSS-RR guide)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 17, // binding 17: specularRayDirHitDistImage (DLSS-RR guide)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 18, // binding 18: bounceReservoirImage1 (bounce 1 ReSTIR DI)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 19, // binding 19: bounceReservoirImage2 (bounce 2 ReSTIR DI)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 20, // binding 20: bounceReservoirImage3 (bounce 3 ReSTIR DI)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 21, // binding 21: reflectionMvImage (DLSS-RR specular reflection MVs)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 22, // binding 22: animTexMaskImage (DLSS-RR animated texture mask)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 23, // binding 23: particleMaskImage (DLSS-RR transparent/refractive surface mask)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 24, // binding 24: biasMaskImage (DLSS-RR bias current color mask)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 25, // binding 25: rtHitDistImage (DLSS-RR per-pixel noise level hint)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 26, // binding 26: motionVectors3DImage (DLSS-RR world-space 3D velocity)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 27, // binding 27: gbufferMetallicImage (GBuffer metallic factor)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 28, // binding 28: gbufferShadingModelIdImage (GBuffer shading model ID)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 29, // binding 29: gbufferMaterialIdImage (GBuffer material ID)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 30, // binding 30: positionViewSpaceImage (view-space hit position)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 31, // binding 31: transparencyLayerImage (DLSS-RR stable planes)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 32, // binding 32: transparencyLayerOpacityImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 33, // binding 33: transparencyLayerMvecsImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 34, // binding 34: SHARC candidate position + hitT
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 35, // binding 35: SHARC candidate normal + roughness
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 36, // binding 36: SHARC candidate throughput
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 37, // binding 37: SHARC candidate prefix radiance + flags
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .definePushConstant({
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                    .offset = 0,
                    .size = sizeof(RayTracingPushConstant),
                })
                .build(framework->device());
    }
}

// =============================================================================
// Energy Compensation LUT bake (Kulla-Conty GGX + EON FON)
// 64x64 RGBA16F: R=GGX_E, G=FON_E, B=GGX_Eavg, A=FON_Eavg
// Runs once at init. ~2ms on modern CPUs.
// =============================================================================

namespace {

// CPU versions of GGX utility functions (mirroring shader code)
struct Vec3 { float x, y, z; };

Vec3 normalize3(Vec3 v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (len < 1e-8f) return {0, 0, 1};
    return {v.x/len, v.y/len, v.z/len};
}

Vec3 sampleGGXVNDF_CPU(Vec3 V, float ax, float ay, float r1, float r2) {
    Vec3 Vh = normalize3({ax * V.x, ay * V.y, V.z});
    float lensq = Vh.x*Vh.x + Vh.y*Vh.y;
    Vec3 T1 = lensq > 0 ? Vec3{-Vh.y / sqrtf(lensq), Vh.x / sqrtf(lensq), 0}
                         : Vec3{1, 0, 0};
    Vec3 T2 = {Vh.y*T1.z - Vh.z*T1.y, Vh.z*T1.x - Vh.x*T1.z, Vh.x*T1.y - Vh.y*T1.x};
    float r = sqrtf(r1);
    float phi = 6.28318530718f * r2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1*t1)) + s * t2;
    float nz = sqrtf(fmaxf(0.0f, 1.0f - t1*t1 - t2*t2));
    Vec3 Nh = {t1*T1.x + t2*T2.x + nz*Vh.x,
               t1*T1.y + t2*T2.y + nz*Vh.y,
               t1*T1.z + t2*T2.z + nz*Vh.z};
    return normalize3({ax * Nh.x, ay * Nh.y, fmaxf(0.0f, Nh.z)});
}

float smithG1_CPU(float NdotV, float alpha) {
    float a2 = alpha * alpha;
    float c = NdotV;
    return (2.0f * c) / (c + sqrtf(a2 + c*c - a2*c*c));
}

// Simple xorshift32 RNG
uint32_t xorshift32(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float randFloat(uint32_t &state) {
    return (float)(xorshift32(state) & 0xFFFFFF) / (float)0xFFFFFF;
}

// Integrate GGX directional albedo E(mu, alpha) for F0=1 via VNDF importance sampling
float bakeGGXDirectionalAlbedo(float NdotV, float alpha, int numSamples) {
    NdotV = fmaxf(NdotV, 1e-4f);
    alpha = fmaxf(alpha, 1e-4f);
    Vec3 V = {sqrtf(1.0f - NdotV*NdotV), 0.0f, NdotV};
    float sum = 0.0f;
    uint32_t seed = (uint32_t)(NdotV * 12345.0f + alpha * 67890.0f + 1);
    for (int i = 0; i < numSamples; i++) {
        float r1 = randFloat(seed);
        float r2 = randFloat(seed);
        Vec3 H = sampleGGXVNDF_CPU(V, alpha, alpha, r1, r2);
        // Reflect V around H
        float VdotH = V.x*H.x + V.y*H.y + V.z*H.z;
        Vec3 L = {2.0f*VdotH*H.x - V.x, 2.0f*VdotH*H.y - V.y, 2.0f*VdotH*H.z - V.z};
        if (L.z > 0.0f) {
            // With VNDF sampling + separable Smith, weight = G1(L)
            sum += smithG1_CPU(L.z, alpha);
        }
    }
    return fminf(fmaxf(sum / (float)numSamples, 0.0f), 1.0f);
}

// Integrate FON directional albedo E_F(mu, r) via cosine-weighted hemisphere sampling
float bakeFONDirectionalAlbedo(float mu_o, float r, int numSamples) {
    mu_o = fmaxf(mu_o, 1e-4f);
    float sinTheta_o = sqrtf(1.0f - mu_o*mu_o);
    // FON coefficients
    float A_F = 1.0f / (1.0f + (0.5f - 2.0f / (3.0f * 3.14159265f)) * r);
    float B_F = r * A_F;
    float sum = 0.0f;
    uint32_t seed = (uint32_t)(mu_o * 54321.0f + r * 98765.0f + 7);
    for (int i = 0; i < numSamples; i++) {
        float r1 = randFloat(seed);
        float r2 = randFloat(seed);
        // Cosine-weighted hemisphere sample
        float sqrtR1 = sqrtf(r1);
        float phi = 6.28318530718f * r2;
        float Lx = sqrtR1 * cosf(phi);
        float Ly = sqrtR1 * sinf(phi);
        float Lz = sqrtf(fmaxf(0.0f, 1.0f - r1));
        float mu_i = Lz;
        // s = dot(L, V) - mu_i * mu_o  (V is in the xz plane)
        float VdotL = Lx * sinTheta_o + Lz * mu_o;
        float s = VdotL - mu_i * mu_o;
        float t = (s > 0.0f) ? fmaxf(mu_i, mu_o) : 1.0f;
        float f_FON = (1.0f / 3.14159265f) * (A_F + B_F * s / t);
        // Cosine-weighted sampling: weight = f_FON * pi (PDF = cos/pi, integrand = f*cos)
        sum += f_FON * 3.14159265f;
    }
    return fminf(fmaxf(sum / (float)numSamples, 0.0f), 1.0f);
}

} // anonymous namespace

void RayTracingModule::initEnergyLUT() {
    auto framework = framework_.lock();
    if (!framework) return;

    constexpr int LUT_SIZE = 64;
    constexpr int GGX_SAMPLES = 1024;
    constexpr int FON_SAMPLES = 512;

    // Bake LUT on CPU
    // Layout: RGBA16F, R=GGX_E(NdotV,alpha), G=FON_E(NdotV,r), B=GGX_Eavg, A=FON_Eavg
    // UV: u=NdotV [0,1], v=perceptual roughness r [0,1]; alpha = r*r
    std::vector<uint16_t> lutData(LUT_SIZE * LUT_SIZE * 4); // RGBA16F = 4 x fp16 per texel

    // First pass: compute E values per texel
    std::vector<float> ggxE(LUT_SIZE * LUT_SIZE);
    std::vector<float> fonE(LUT_SIZE * LUT_SIZE);

    for (int y = 0; y < LUT_SIZE; y++) {
        float r = ((float)y + 0.5f) / (float)LUT_SIZE; // perceptual roughness
        float alpha = r * r; // GGX alpha
        for (int x = 0; x < LUT_SIZE; x++) {
            float NdotV = ((float)x + 0.5f) / (float)LUT_SIZE;
            int idx = y * LUT_SIZE + x;
            ggxE[idx] = bakeGGXDirectionalAlbedo(NdotV, alpha, GGX_SAMPLES);
            fonE[idx] = bakeFONDirectionalAlbedo(NdotV, r, FON_SAMPLES);
        }
    }

    // Second pass: compute E_avg per roughness row (cosine-weighted average over NdotV)
    std::vector<float> ggxEavg(LUT_SIZE);
    std::vector<float> fonEavg(LUT_SIZE);

    for (int y = 0; y < LUT_SIZE; y++) {
        float sumGGX = 0.0f, sumFON = 0.0f, sumWeight = 0.0f;
        for (int x = 0; x < LUT_SIZE; x++) {
            float NdotV = ((float)x + 0.5f) / (float)LUT_SIZE;
            float weight = NdotV; // cosine weight (mu * dmu, uniform spacing)
            int idx = y * LUT_SIZE + x;
            sumGGX += ggxE[idx] * weight;
            sumFON += fonE[idx] * weight;
            sumWeight += weight;
        }
        ggxEavg[y] = sumWeight > 0 ? sumGGX / sumWeight : 0.5f;
        fonEavg[y] = sumWeight > 0 ? sumFON / sumWeight : 0.5f;
    }

    // Convert to fp16 and pack into RGBA16F
    auto toFP16 = [](float f) -> uint16_t {
        // IEEE 754 float32 to float16 conversion
        uint32_t bits;
        memcpy(&bits, &f, 4);
        uint32_t sign = (bits >> 16) & 0x8000;
        int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = bits & 0x7FFFFF;
        if (exponent <= 0) return (uint16_t)sign; // underflow to 0
        if (exponent >= 31) return (uint16_t)(sign | 0x7C00); // overflow to inf
        return (uint16_t)(sign | (exponent << 10) | (mantissa >> 13));
    };

    for (int y = 0; y < LUT_SIZE; y++) {
        for (int x = 0; x < LUT_SIZE; x++) {
            int idx = y * LUT_SIZE + x;
            int pixelBase = idx * 4;
            lutData[pixelBase + 0] = toFP16(ggxE[idx]);      // R: GGX E(NdotV, alpha)
            lutData[pixelBase + 1] = toFP16(fonE[idx]);       // G: FON E(NdotV, r)
            lutData[pixelBase + 2] = toFP16(ggxEavg[y]);      // B: GGX E_avg(alpha)
            lutData[pixelBase + 3] = toFP16(fonEavg[y]);      // A: FON E_avg(r)
        }
    }

    // Create GPU image
    energyLUT_ = vk::DeviceLocalImage::create(
        framework->device(), framework->vma(), true, // persistStaging
        LUT_SIZE, LUT_SIZE, 1,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    energyLUT_->uploadToStagingBuffer(lutData.data());

    // One-shot transfer: staging buffer -> GPU image
    auto cmdBuf = vk::CommandBuffer::create(framework->device(), framework->mainCommandPool());
    cmdBuf->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    cmdBuf->copyToDeviceLocalImage(energyLUT_);
    cmdBuf->end();
    cmdBuf->submitMainQueueIndividual(framework->device());
    vkQueueWaitIdle(framework->device()->mainVkQueue());

    // Create sampler (bilinear, clamp to edge)
    energyLUTSampler_ = vk::Sampler::create(
        framework->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    RadianceLogger::log("RayTracing", "INFO", "Energy compensation LUT baked (%dx%d, GGX=%d FON=%d samples)",
        LUT_SIZE, LUT_SIZE, GGX_SAMPLES, FON_SAMPLES);
}

void RayTracingModule::initImages() {
    auto framework = framework_.lock();
    if (!framework) return;

    uint32_t size = framework->swapchain()->imageCount();

    for (int i = 0; i < size; i++) {
        rayTracingDescriptorTables_[i]->bindSamplerImageForShader(atmosphere_->atmLUTImageSampler_,
                                                                  atmosphere_->atmLUTImage_, 0, 1);
        rayTracingDescriptorTables_[i]->bindSamplerImageForShader(atmosphere_->atmCubeMapImageSamplers_[i],
                                                                  atmosphere_->atmCubeMapImages_[i], 0, 2, 7);

        // Batch all RT output image bindings into a single vkUpdateDescriptorSets call
        using IB = vk::DescriptorTable::ImageBinding;
        std::vector<IB> imageBindings = {
            {hdrNoisyOutputImages_[i],                 VK_IMAGE_LAYOUT_GENERAL, 3,  0},
            {diffuseAlbedoImages_[i],                  VK_IMAGE_LAYOUT_GENERAL, 3,  1},
            {specularAlbedoImages_[i],                 VK_IMAGE_LAYOUT_GENERAL, 3,  2},
            {normalRoughnessImages_[i],                VK_IMAGE_LAYOUT_GENERAL, 3,  3},
            {motionVectorImages_[i],                   VK_IMAGE_LAYOUT_GENERAL, 3,  4},
            {linearDepthImages_[i],                    VK_IMAGE_LAYOUT_GENERAL, 3,  5},
            {specularHitDepthImages_[i],               VK_IMAGE_LAYOUT_GENERAL, 3,  6},
            {firstHitDepthImages_[i],                  VK_IMAGE_LAYOUT_GENERAL, 3,  7},
            {firstHitDiffuseDirectLightImages_[i],     VK_IMAGE_LAYOUT_GENERAL, 3,  8},
            {firstHitDiffuseIndirectLightImages_[i],   VK_IMAGE_LAYOUT_GENERAL, 3,  9},
            {firstHitSpecularImages_[i],               VK_IMAGE_LAYOUT_GENERAL, 3, 10},
            {firstHitClearImages_[i],                  VK_IMAGE_LAYOUT_GENERAL, 3, 11},
            {firstHitBaseEmissionImages_[i],           VK_IMAGE_LAYOUT_GENERAL, 3, 12},
            {directLightDepthImages_[i],               VK_IMAGE_LAYOUT_GENERAL, 3, 13},
            {diffuseRayDirHitDistImages_[i],           VK_IMAGE_LAYOUT_GENERAL, 3, 16},
            {specularRayDirHitDistImages_[i],          VK_IMAGE_LAYOUT_GENERAL, 3, 17},
            {reflectionMvImages_[i],                   VK_IMAGE_LAYOUT_GENERAL, 3, 21},
        };
        // Conditionally add optional output images
        if (animatedTexMaskImages_[i])
            imageBindings.push_back({animatedTexMaskImages_[i],       VK_IMAGE_LAYOUT_GENERAL, 3, 22});
        if (particleMaskImages_[i])
            imageBindings.push_back({particleMaskImages_[i],          VK_IMAGE_LAYOUT_GENERAL, 3, 23});
        if (biasMaskImages_[i])
            imageBindings.push_back({biasMaskImages_[i],              VK_IMAGE_LAYOUT_GENERAL, 3, 24});
        if (rtHitDistImages_[i])
            imageBindings.push_back({rtHitDistImages_[i],             VK_IMAGE_LAYOUT_GENERAL, 3, 25});
        if (motionVectors3DImages_[i])
            imageBindings.push_back({motionVectors3DImages_[i],       VK_IMAGE_LAYOUT_GENERAL, 3, 26});
        if (gbufferMetallicImages_[i])
            imageBindings.push_back({gbufferMetallicImages_[i],       VK_IMAGE_LAYOUT_GENERAL, 3, 27});
        if (gbufferShadingModelIdImages_[i])
            imageBindings.push_back({gbufferShadingModelIdImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 28});
        if (gbufferMaterialIdImages_[i])
            imageBindings.push_back({gbufferMaterialIdImages_[i],     VK_IMAGE_LAYOUT_GENERAL, 3, 29});
        if (positionViewSpaceImages_[i])
            imageBindings.push_back({positionViewSpaceImages_[i],     VK_IMAGE_LAYOUT_GENERAL, 3, 30});
        if (transparencyLayerImages_[i])
            imageBindings.push_back({transparencyLayerImages_[i],        VK_IMAGE_LAYOUT_GENERAL, 3, 31});
        if (transparencyLayerOpacityImages_[i])
            imageBindings.push_back({transparencyLayerOpacityImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 32});
        if (transparencyLayerMvecsImages_[i])
            imageBindings.push_back({transparencyLayerMvecsImages_[i],   VK_IMAGE_LAYOUT_GENERAL, 3, 33});
#ifdef MCVR_ENABLE_SHARC_QUERY_PASS
        if (sharcCandidatePosHitTImages_[i])
            imageBindings.push_back({sharcCandidatePosHitTImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 34});
        if (sharcCandidateNormalRoughnessImages_[i])
            imageBindings.push_back({sharcCandidateNormalRoughnessImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 35});
        if (sharcCandidateThroughputImages_[i])
            imageBindings.push_back({sharcCandidateThroughputImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 36});
        if (sharcCandidatePrefixRadianceFlagsImages_[i])
            imageBindings.push_back({sharcCandidatePrefixRadianceFlagsImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 37});
#endif

        // ReSTIR DI reservoir images (initial binding, rebound each frame in render)
        if (reservoirImages_[0]) {
            imageBindings.push_back({reservoirImages_[0], VK_IMAGE_LAYOUT_GENERAL, 3, 14});
            imageBindings.push_back({reservoirImages_[1], VK_IMAGE_LAYOUT_GENERAL, 3, 15});
        }

        rayTracingDescriptorTables_[i]->bindImages(imageBindings);

        // Energy compensation LUT (set 2, binding 4) — sampler image, stays individual
        if (energyLUT_ && energyLUTSampler_) {
            rayTracingDescriptorTables_[i]->bindSamplerImageForShader(energyLUTSampler_, energyLUT_, 2, 4);
        }

        // Pre-bind blue noise SSBOs to ALL descriptor tables at init time.
        // The render() path re-binds per-frame, but this ensures no descriptor
        // is left dangling at set=1 bindings 13/14 before the first render.
        if (blueNoise_) {
            using BB = vk::DescriptorTable::BufferBinding;
            rayTracingDescriptorTables_[i]->bindBufferBatch({
                BB{blueNoise_->sobolBuffer(),      2, 9},
                BB{blueNoise_->scramblingBuffer(), 2, 10},
            });
        }
    }

    // Publish emission images so tone mapping can add them back after DLSS-RR denoising
    Renderer::emissionImages = firstHitBaseEmissionImages_;
}

void RayTracingModule::initPipeline() {
    auto framework = framework_.lock();
    if (!framework) return;
    auto device = framework->device();

    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    worldRayGenShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_rgen.spv").string());
    worldRayMissShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_rmiss.spv").string());
    handRayMissShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/hand_rmiss.spv").string());
    shadowRayMissShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/shadow_rmiss.spv").string());
    pointLightShadowMissShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/point_light_shadow_rmiss.spv").string());
    shadowRayClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/shadow_rchit.spv").string());
    worldSolidTransparentClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_solid_transparent_rchit.spv").string());
    worldSolidTransparentNoDisplacementClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_solid_transparent_no_displacement_rchit.spv").string());
    const bool displacementRequested = Renderer::options.pomEnabled && worldSolidTransparentClosestHitShader_;
    const bool useShaderDisplacement = displacementRequested && shaderDisplacementRuntimeAllowed();
    auto activeWorldSolidTransparentClosestHitShader =
        useShaderDisplacement
            ? worldSolidTransparentClosestHitShader_
            : worldSolidTransparentNoDisplacementClosestHitShader_;
    if (!activeWorldSolidTransparentClosestHitShader) {
        activeWorldSolidTransparentClosestHitShader = worldSolidTransparentClosestHitShader_;
    }
    RadianceLogger::log("RayTracing", "INFO",
                        "World closest-hit variant: %s (displacementRequested=%d displacementRuntimeAllowed=%d noDisplacementShader=%d)",
                        useShaderDisplacement ? "displacement" : "no_displacement",
                        displacementRequested ? 1 : 0,
                        shaderDisplacementRuntimeAllowed() ? 1 : 0,
                        worldSolidTransparentNoDisplacementClosestHitShader_ ? 1 : 0);
    worldNoReflectClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_no_reflect_rchit.spv").string());
    worldCloudClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_cloud_rchit.spv").string());
    shadowAnyHitShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/shadow_rahit.spv").string());
    worldTransparentAnyHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_transparent_rahit.spv").string());
    worldNoReflectAnyHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_no_reflect_rahit.spv").string());
    worldCloudAnyHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_cloud_rahit.spv").string());

    boatWaterMaskClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/boat_water_mask_rchit.spv").string());
    boatWaterMaskAnyHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/boat_water_mask_rahit.spv").string());

    endPortalClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/end_portal_rchit.spv").string());
    endPortalAnyHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/end_portal_rahit.spv").string());

    endGatewayClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/end_gateway_rchit.spv").string());
    endGatewayAnyHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/end_gateway_rahit.spv").string());

    RadianceLogger::log("RayTracing", "INFO",
                        "World RT pipeline create start sharcQueryMode=%d sharcMainTraceCompiled=%d",
                        MCVR_SHARC_MAIN_TRACE_QUERY_MODE,
#ifdef MCVR_ENABLE_SHARC_MAIN_TRACE_QUERY
                        1
#else
                        0
#endif
                        );
    g_crashRing.record("RT:pipeline:create:start");
    rayTracingPipeline_ =
        vk::RayTracingPipelineBuilder{}
            .beginShaderStage()
            .defineShaderStage(worldRayGenShader_, VK_SHADER_STAGE_RAYGEN_BIT_KHR)                          // 0
            .defineShaderStage(worldRayMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                           // 1
            .defineShaderStage(handRayMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                            // 2
            .defineShaderStage(shadowRayMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                          // 3
            .defineShaderStage(activeWorldSolidTransparentClosestHitShader, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR) // 4
            .defineShaderStage(worldNoReflectClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)        // 5
            .defineShaderStage(worldCloudClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)            // 6
            .defineShaderStage(worldTransparentAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)              // 7
            .defineShaderStage(worldNoReflectAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                // 8
            .defineShaderStage(worldCloudAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                    // 9
            .defineShaderStage(shadowRayClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)             // 10
            .defineShaderStage(shadowAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                        // 11
            .defineShaderStage(boatWaterMaskClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)         // 12
            .defineShaderStage(boatWaterMaskAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                 // 13
            .defineShaderStage(endPortalClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)             // 14
            .defineShaderStage(endPortalAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                     // 15
            .defineShaderStage(endGatewayClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)            // 16
            .defineShaderStage(endGatewayAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                    // 17
            .defineShaderStage(pointLightShadowMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                   // 18
            .endShaderStage()
            .beginShaderGroup()
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR)
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR)
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 2, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR)
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 3, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR)
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 18, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR) // point light shadow miss
            // Hit groups 0-7: triangle groups (existing)
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 10, 11,
                               VK_SHADER_UNUSED_KHR) // shadow
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 4,
                               VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR) // world solid
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 4, 7,
                               VK_SHADER_UNUSED_KHR) // world transparent
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 5, 8,
                               VK_SHADER_UNUSED_KHR) // world no reflect
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 6, 9,
                               VK_SHADER_UNUSED_KHR) // world cloud
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 12, 13,
                               VK_SHADER_UNUSED_KHR) // boat water mask
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 14, 15,
                               VK_SHADER_UNUSED_KHR) // end portal
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 16, 17,
                               VK_SHADER_UNUSED_KHR) // end gateway
            .endShaderGroup()
            .definePipelineLayout(rayTracingDescriptorTables_[0])
            .build(device);
    RadianceLogger::log("RayTracing", "INFO", "World RT pipeline create done (ok=%d)",
                        rayTracingPipeline_ ? 1 : 0);
    g_crashRing.record(rayTracingPipeline_ ? "RT:pipeline:create:done" : "RT:pipeline:create:failed");
}

void RayTracingModule::initSBT() {
    auto framework = framework_.lock();
    if (!framework) return;

    // 4 miss groups, 8 triangle hit groups.
    sbts_.resize(framework->swapchain()->imageCount());
    for (int i = 0; i < framework->swapchain()->imageCount(); i++) {
        sbts_[i] = vk::SBT::create(framework->physicalDevice(), framework->device(), framework->vma(),
                                   rayTracingPipeline_, 4, 8);
    }
}

void RayTracingModule::initSpatialPipeline() {
    auto framework = framework_.lock();
    if (!framework) return;
    auto device = framework->device();
    VkDevice dev = device->vkDevice();
    uint32_t size = framework->swapchain()->imageCount();

    // Load compute shader
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    spatialShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/restir_spatial_comp.spv").string());
    if (!spatialShader_) {
        std::cerr << "[ReSTIR] Failed to load restir_spatial_comp.spv" << std::endl;
        return;
    }

    // Descriptor set layout: 4 storage images
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &spatialDescSetLayout_);

    // Pipeline layout with push constant (6 int32s = 24 bytes)
    VkPushConstantRange pushRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 24};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &spatialDescSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(dev, &pipelineLayoutInfo, nullptr, &spatialPipelineLayout_);

    // Compute pipeline
    VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = spatialShader_->vkShaderModule();
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = spatialPipelineLayout_;
    vkCreateComputePipelines(dev, device->pipelineCache(), 1, &pipelineInfo, nullptr, &spatialPipeline_);

    // Descriptor pool
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 * size},
    };
    VkDescriptorPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = size;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    vkCreateDescriptorPool(dev, &poolInfo, nullptr, &spatialDescPool_);

    // Allocate descriptor sets
    std::vector<VkDescriptorSetLayout> layouts(size, spatialDescSetLayout_);
    VkDescriptorSetAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = spatialDescPool_;
    allocInfo.descriptorSetCount = size;
    allocInfo.pSetLayouts = layouts.data();
    spatialDescSets_.resize(size);
    vkAllocateDescriptorSets(dev, &allocInfo, spatialDescSets_.data());
}

void RayTracingModule::initClusterPipeline() {
    auto framework = framework_.lock();
    if (!framework) return;
    auto device = framework->device();
    VkDevice dev = device->vkDevice();
    uint32_t size = framework->swapchain()->imageCount();

    // Load compute shader
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    clusterShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/light_clustering_comp.spv").string());
    if (!clusterShader_) {
        std::cerr << "[Clustering] Failed to load light_clustering_comp.spv" << std::endl;
        return;
    }

    // Descriptor set layout: 2 storage buffers (area lights + tile buffer)
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &clusterDescSetLayout_);

    // Pipeline layout with push constant (width, height, lightCount, maxPerTile, mat4 vpCameraRel)
    VkPushConstantRange pushRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 80}; // 16 bytes ints + 64 bytes mat4
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &clusterDescSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(dev, &pipelineLayoutInfo, nullptr, &clusterPipelineLayout_);

    // Compute pipeline
    VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = clusterShader_->vkShaderModule();
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = clusterPipelineLayout_;
    vkCreateComputePipelines(dev, device->pipelineCache(), 1, &pipelineInfo, nullptr, &clusterPipeline_);

    // Descriptor pool
    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * size};
    VkDescriptorPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = size;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(dev, &poolInfo, nullptr, &clusterDescPool_);

    // Allocate descriptor sets
    std::vector<VkDescriptorSetLayout> clusterLayouts(size, clusterDescSetLayout_);
    VkDescriptorSetAllocateInfo clusterAllocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    clusterAllocInfo.descriptorPool = clusterDescPool_;
    clusterAllocInfo.descriptorSetCount = size;
    clusterAllocInfo.pSetLayouts = clusterLayouts.data();
    clusterDescSets_.resize(size);
    vkAllocateDescriptorSets(dev, &clusterAllocInfo, clusterDescSets_.data());

    // Create tile light buffer (will be resized per-frame if needed)
    // Initial size based on common resolution
    int tilesX = (1920 + TILE_SIZE - 1) / TILE_SIZE;
    int tilesY = (1080 + TILE_SIZE - 1) / TILE_SIZE;
    int totalTiles = tilesX * tilesY;
    size_t bufferSize = totalTiles * (1 + MAX_LIGHTS_PER_TILE) * sizeof(uint32_t);
    tileLightBuffer_ = vk::DeviceLocalBuffer::create(
        framework->vma(), device, bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

void RayTracingModule::initDirectLightPipeline() {
#ifndef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
    return;
#else
    auto framework = framework_.lock();
    if (!framework) return;
    auto device = framework->device();
    VkDevice dev = device->vkDevice();
    uint32_t size = framework->swapchain()->imageCount();

    if (directLightPrimaryPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, directLightPrimaryPipeline_, nullptr);
        directLightPrimaryPipeline_ = VK_NULL_HANDLE;
    }
    if (directLightPrimaryPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, directLightPrimaryPipelineLayout_, nullptr);
        directLightPrimaryPipelineLayout_ = VK_NULL_HANDLE;
    }
    if (directLightPrimaryDescSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(dev, directLightPrimaryDescSetLayout_, nullptr);
        directLightPrimaryDescSetLayout_ = VK_NULL_HANDLE;
    }
    if (directLightPrimaryDescPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(dev, directLightPrimaryDescPool_, nullptr);
        directLightPrimaryDescPool_ = VK_NULL_HANDLE;
    }
    if (directLightInitialPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, directLightInitialPipeline_, nullptr);
        directLightInitialPipeline_ = VK_NULL_HANDLE;
    }
    if (directLightInitialPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, directLightInitialPipelineLayout_, nullptr);
        directLightInitialPipelineLayout_ = VK_NULL_HANDLE;
    }
    if (directLightInitialDescSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(dev, directLightInitialDescSetLayout_, nullptr);
        directLightInitialDescSetLayout_ = VK_NULL_HANDLE;
    }
    if (directLightInitialDescPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(dev, directLightInitialDescPool_, nullptr);
        directLightInitialDescPool_ = VK_NULL_HANDLE;
    }
    if (directLightUtilityPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, directLightUtilityPipeline_, nullptr);
        directLightUtilityPipeline_ = VK_NULL_HANDLE;
    }
    if (directLightUtilityPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, directLightUtilityPipelineLayout_, nullptr);
        directLightUtilityPipelineLayout_ = VK_NULL_HANDLE;
    }
    if (directLightUtilityDescSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(dev, directLightUtilityDescSetLayout_, nullptr);
        directLightUtilityDescSetLayout_ = VK_NULL_HANDLE;
    }
    if (directLightUtilityDescPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(dev, directLightUtilityDescPool_, nullptr);
        directLightUtilityDescPool_ = VK_NULL_HANDLE;
    }

    RadianceLogger::log("RayTracing", "INFO", "Direct-light primary pipeline create start");
    g_crashRing.record("DirectLight:primaryPipeline:create:start");
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    directLightPrimaryShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/direct_light_primary_comp.spv").string());
    if (!directLightPrimaryShader_) {
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light primary shader load failed");
        g_crashRing.record("DirectLight:primaryPipeline:create:missingShader");
        return;
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    VkResult layoutResult = vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &directLightPrimaryDescSetLayout_);
    if (layoutResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light primary descriptor layout failed result=%d",
                            static_cast<int>(layoutResult));
        g_crashRing.record("DirectLight:primaryPipeline:create:layoutFailed");
        return;
    }

    VkPushConstantRange pushRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t) * 2};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &directLightPrimaryDescSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    VkResult pipelineLayoutResult =
        vkCreatePipelineLayout(dev, &pipelineLayoutInfo, nullptr, &directLightPrimaryPipelineLayout_);
    if (pipelineLayoutResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light primary pipeline layout failed result=%d",
                            static_cast<int>(pipelineLayoutResult));
        g_crashRing.record("DirectLight:primaryPipeline:create:pipelineLayoutFailed");
        return;
    }

    VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = directLightPrimaryShader_->vkShaderModule();
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = directLightPrimaryPipelineLayout_;
    VkResult pipelineResult = vkCreateComputePipelines(dev, device->pipelineCache(), 1, &pipelineInfo, nullptr,
                                                       &directLightPrimaryPipeline_);
    if (pipelineResult != VK_SUCCESS) {
        directLightPrimaryPipeline_ = VK_NULL_HANDLE;
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light primary pipeline create done ok=0 result=%d",
                            static_cast<int>(pipelineResult));
        g_crashRing.record("DirectLight:primaryPipeline:create:failed");
        return;
    }

    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4u * size},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, size},
    };
    VkDescriptorPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = size;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    VkResult poolResult = vkCreateDescriptorPool(dev, &poolInfo, nullptr, &directLightPrimaryDescPool_);
    if (poolResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light primary descriptor pool failed result=%d",
                            static_cast<int>(poolResult));
        g_crashRing.record("DirectLight:primaryPipeline:create:poolFailed");
        return;
    }

    std::vector<VkDescriptorSetLayout> layouts(size, directLightPrimaryDescSetLayout_);
    VkDescriptorSetAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = directLightPrimaryDescPool_;
    allocInfo.descriptorSetCount = size;
    allocInfo.pSetLayouts = layouts.data();
    directLightPrimaryDescSets_.resize(size);
    VkResult allocResult = vkAllocateDescriptorSets(dev, &allocInfo, directLightPrimaryDescSets_.data());
    if (allocResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light primary descriptor alloc failed result=%d",
                            static_cast<int>(allocResult));
        g_crashRing.record("DirectLight:primaryPipeline:create:allocFailed");
        return;
    }

    RadianceLogger::log("RayTracing", "INFO", "Direct-light primary pipeline create done ok=1 sets=%u", size);
    g_crashRing.record("DirectLight:primaryPipeline:create:done");

    RadianceLogger::log("RayTracing", "INFO", "Direct-light initial pipeline create start");
    g_crashRing.record("DirectLight:initialPipeline:create:start");
    directLightInitialShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/direct_light_initial_comp.spv").string());
    if (!directLightInitialShader_) {
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light initial shader load failed");
        g_crashRing.record("DirectLight:initialPipeline:create:missingShader");
        return;
    }

    std::vector<VkDescriptorSetLayoutBinding> initialBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {8, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {11, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {12, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo initialLayoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    initialLayoutInfo.bindingCount = static_cast<uint32_t>(initialBindings.size());
    initialLayoutInfo.pBindings = initialBindings.data();
    VkResult initialLayoutResult =
        vkCreateDescriptorSetLayout(dev, &initialLayoutInfo, nullptr, &directLightInitialDescSetLayout_);
    if (initialLayoutResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light initial descriptor layout failed result=%d",
                            static_cast<int>(initialLayoutResult));
        g_crashRing.record("DirectLight:initialPipeline:create:layoutFailed");
        return;
    }

    VkPipelineLayoutCreateInfo initialPipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    initialPipelineLayoutInfo.setLayoutCount = 1;
    initialPipelineLayoutInfo.pSetLayouts = &directLightInitialDescSetLayout_;
    initialPipelineLayoutInfo.pushConstantRangeCount = 1;
    initialPipelineLayoutInfo.pPushConstantRanges = &pushRange;
    VkResult initialPipelineLayoutResult =
        vkCreatePipelineLayout(dev, &initialPipelineLayoutInfo, nullptr, &directLightInitialPipelineLayout_);
    if (initialPipelineLayoutResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light initial pipeline layout failed result=%d",
                            static_cast<int>(initialPipelineLayoutResult));
        g_crashRing.record("DirectLight:initialPipeline:create:pipelineLayoutFailed");
        return;
    }

    VkComputePipelineCreateInfo initialPipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    initialPipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    initialPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    initialPipelineInfo.stage.module = directLightInitialShader_->vkShaderModule();
    initialPipelineInfo.stage.pName = "main";
    initialPipelineInfo.layout = directLightInitialPipelineLayout_;
    VkResult initialPipelineResult = vkCreateComputePipelines(dev, device->pipelineCache(), 1, &initialPipelineInfo,
                                                              nullptr, &directLightInitialPipeline_);
    if (initialPipelineResult != VK_SUCCESS) {
        directLightInitialPipeline_ = VK_NULL_HANDLE;
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light initial pipeline create done ok=0 result=%d",
                            static_cast<int>(initialPipelineResult));
        g_crashRing.record("DirectLight:initialPipeline:create:failed");
        return;
    }

    VkDescriptorPoolSize initialPoolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7u * size},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3u * size},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2u * size},
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, size},
    };
    VkDescriptorPoolCreateInfo initialPoolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    initialPoolInfo.maxSets = size;
    initialPoolInfo.poolSizeCount = 4;
    initialPoolInfo.pPoolSizes = initialPoolSizes;
    VkResult initialPoolResult = vkCreateDescriptorPool(dev, &initialPoolInfo, nullptr, &directLightInitialDescPool_);
    if (initialPoolResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light initial descriptor pool failed result=%d",
                            static_cast<int>(initialPoolResult));
        g_crashRing.record("DirectLight:initialPipeline:create:poolFailed");
        return;
    }

    std::vector<VkDescriptorSetLayout> initialLayouts(size, directLightInitialDescSetLayout_);
    VkDescriptorSetAllocateInfo initialAllocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    initialAllocInfo.descriptorPool = directLightInitialDescPool_;
    initialAllocInfo.descriptorSetCount = size;
    initialAllocInfo.pSetLayouts = initialLayouts.data();
    directLightInitialDescSets_.resize(size);
    VkResult initialAllocResult =
        vkAllocateDescriptorSets(dev, &initialAllocInfo, directLightInitialDescSets_.data());
    if (initialAllocResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light initial descriptor alloc failed result=%d",
                            static_cast<int>(initialAllocResult));
        g_crashRing.record("DirectLight:initialPipeline:create:allocFailed");
        return;
    }

    RadianceLogger::log("RayTracing", "INFO", "Direct-light initial pipeline create done ok=1 sets=%u", size);
    g_crashRing.record("DirectLight:initialPipeline:create:done");

    RadianceLogger::log("RayTracing", "INFO", "Direct-light utility pipeline create start");
    g_crashRing.record("DirectLight:utilityPipeline:create:start");
    directLightUtilityShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/direct_light_reservoir_comp.spv").string());
    if (!directLightUtilityShader_) {
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light utility shader load failed");
        g_crashRing.record("DirectLight:utilityPipeline:create:missingShader");
        return;
    }

    std::vector<VkDescriptorSetLayoutBinding> utilityBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo utilityLayoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    utilityLayoutInfo.bindingCount = static_cast<uint32_t>(utilityBindings.size());
    utilityLayoutInfo.pBindings = utilityBindings.data();
    VkResult utilityLayoutResult =
        vkCreateDescriptorSetLayout(dev, &utilityLayoutInfo, nullptr, &directLightUtilityDescSetLayout_);
    if (utilityLayoutResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light utility descriptor layout failed result=%d",
                            static_cast<int>(utilityLayoutResult));
        g_crashRing.record("DirectLight:utilityPipeline:create:layoutFailed");
        return;
    }

    VkPushConstantRange utilityPushRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t) * 4};
    VkPipelineLayoutCreateInfo utilityPipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    utilityPipelineLayoutInfo.setLayoutCount = 1;
    utilityPipelineLayoutInfo.pSetLayouts = &directLightUtilityDescSetLayout_;
    utilityPipelineLayoutInfo.pushConstantRangeCount = 1;
    utilityPipelineLayoutInfo.pPushConstantRanges = &utilityPushRange;
    VkResult utilityPipelineLayoutResult =
        vkCreatePipelineLayout(dev, &utilityPipelineLayoutInfo, nullptr, &directLightUtilityPipelineLayout_);
    if (utilityPipelineLayoutResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light utility pipeline layout failed result=%d",
                            static_cast<int>(utilityPipelineLayoutResult));
        g_crashRing.record("DirectLight:utilityPipeline:create:pipelineLayoutFailed");
        return;
    }

    VkComputePipelineCreateInfo utilityPipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    utilityPipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    utilityPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    utilityPipelineInfo.stage.module = directLightUtilityShader_->vkShaderModule();
    utilityPipelineInfo.stage.pName = "main";
    utilityPipelineInfo.layout = directLightUtilityPipelineLayout_;
    VkResult utilityPipelineResult = vkCreateComputePipelines(dev, device->pipelineCache(), 1, &utilityPipelineInfo,
                                                              nullptr, &directLightUtilityPipeline_);
    if (utilityPipelineResult != VK_SUCCESS) {
        directLightUtilityPipeline_ = VK_NULL_HANDLE;
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light utility pipeline create done ok=0 result=%d",
                            static_cast<int>(utilityPipelineResult));
        g_crashRing.record("DirectLight:utilityPipeline:create:failed");
        return;
    }

    constexpr uint32_t kDirectLightUtilityPasses = 4;
    VkDescriptorPoolSize utilityPoolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4u * size * kDirectLightUtilityPasses},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, size * kDirectLightUtilityPasses},
    };
    VkDescriptorPoolCreateInfo utilityPoolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    utilityPoolInfo.maxSets = size * kDirectLightUtilityPasses;
    utilityPoolInfo.poolSizeCount = 2;
    utilityPoolInfo.pPoolSizes = utilityPoolSizes;
    VkResult utilityPoolResult = vkCreateDescriptorPool(dev, &utilityPoolInfo, nullptr, &directLightUtilityDescPool_);
    if (utilityPoolResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light utility descriptor pool failed result=%d",
                            static_cast<int>(utilityPoolResult));
        g_crashRing.record("DirectLight:utilityPipeline:create:poolFailed");
        return;
    }

    std::vector<VkDescriptorSetLayout> utilityLayouts(size * kDirectLightUtilityPasses, directLightUtilityDescSetLayout_);
    VkDescriptorSetAllocateInfo utilityAllocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    utilityAllocInfo.descriptorPool = directLightUtilityDescPool_;
    utilityAllocInfo.descriptorSetCount = static_cast<uint32_t>(utilityLayouts.size());
    utilityAllocInfo.pSetLayouts = utilityLayouts.data();
    directLightUtilityDescSets_.resize(utilityLayouts.size());
    VkResult utilityAllocResult =
        vkAllocateDescriptorSets(dev, &utilityAllocInfo, directLightUtilityDescSets_.data());
    if (utilityAllocResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "Direct-light utility descriptor alloc failed result=%d",
                            static_cast<int>(utilityAllocResult));
        g_crashRing.record("DirectLight:utilityPipeline:create:allocFailed");
        return;
    }

    RadianceLogger::log("RayTracing", "INFO", "Direct-light utility pipeline create done ok=1 sets=%u",
                        static_cast<uint32_t>(utilityLayouts.size()));
    g_crashRing.record("DirectLight:utilityPipeline:create:done");
#endif
}

void RayTracingModule::initUpstreamDirectLightRuntime() {
#ifndef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
    return;
#else
    directLightUpstreamPackRuntimeReady_ = false;
    directLightUpstreamRuntimeResourcesReady_ = false;
    directLightUpstreamPassRuntimeReady_ = false;
    directLightUpstreamShaderCompileReady_ = false;
    directLightUpstreamPipelineReady_ = false;
    directLightUpstreamSbtReady_ = false;
    directLightUpstreamRuntimeError_.clear();

    auto framework = framework_.lock();
    if (!framework) {
        directLightUpstreamRuntimeError_ = "framework unavailable";
        return;
    }

    try {
        directLightUpstreamShaderPack_ = std::make_shared<ShaderPack>(framework);
        ShaderPack::BuildConfig config;
        config.shaderPackPath =
            (Renderer::folderPath / "shaders/world/ray_tracing/internal/advanced").string();
        config.shouldUseSharc = false;
        config.language = "en_us";
        config.staticAttributes.emplace(
            "render_pipeline.module.ray_tracing.attribute.cloud_mode",
            "render_pipeline.module.ray_tracing.attribute.cloud_mode.vanilla");
        config.staticAttributes.emplace(
            "render_pipeline.module.ray_tracing.attribute.volumetric_light_mode",
            "render_pipeline.module.ray_tracing.attribute.volumetric_light_mode.vanilla");

        std::string error;
        if (!directLightUpstreamShaderPack_->initialize(config, error)) {
            directLightUpstreamShaderPack_.reset();
            directLightUpstreamRuntimeError_ = error;
            RadianceLogger::log("RayTracing", "ERROR",
                                "Upstream RT shader-pack runtime init failed: %s",
                                error.c_str());
            return;
        }
        directLightUpstreamPackRuntimeReady_ = true;

        const auto& pack = directLightUpstreamShaderPack_->shaderPack();
        const char* requiredPasses[] = {
            "primary",
            "precompute_light_neighborhoods",
            "generate_initial_samples",
            "visibility",
            "temporal_reuse",
            "spatial_reuse",
            "direct_light",
            "final_compose",
        };
        uint32_t foundMask = 0;
        for (const auto& passConfig : pack.passes) {
            std::string name;
            switch (passConfig.type) {
                case ShaderPackLoader::PassConfig::Type::RayTracing:
                    name = passConfig.rayTracing.name;
                    break;
                case ShaderPackLoader::PassConfig::Type::Compute:
                    name = passConfig.compute.name;
                    break;
                case ShaderPackLoader::PassConfig::Type::FullScreen:
                    name = passConfig.fullScreen.name;
                    break;
                case ShaderPackLoader::PassConfig::Type::Render:
                    name = passConfig.render.name;
                    break;
            }
            for (uint32_t i = 0; i < static_cast<uint32_t>(std::size(requiredPasses)); ++i) {
                if (name == requiredPasses[i]) {
                    foundMask |= (1u << i);
                    break;
                }
            }
        }
        directLightUpstreamPassRuntimeReady_ =
            foundMask == ((1u << static_cast<uint32_t>(std::size(requiredPasses))) - 1u);
        if (!directLightUpstreamPassRuntimeReady_) {
            directLightUpstreamRuntimeError_ =
                "missing required upstream pass runtime mask=" + std::to_string(foundMask);
            RadianceLogger::log("RayTracing", "ERROR",
                                "Upstream RT pass runtime incomplete mask=%u",
                                foundMask);
            return;
        }

        RadianceLogger::log(
            "RayTracing", "INFO",
            "Upstream RT shader-pack runtime ready passes=%zu textures=%zu buffers=%zu",
            pack.passes.size(), pack.textures.size(), pack.buffers.size());
    } catch (const std::exception& e) {
        directLightUpstreamShaderPack_.reset();
        directLightUpstreamRuntimeError_ = e.what();
        RadianceLogger::log("RayTracing", "ERROR",
                            "Upstream RT shader-pack runtime exception: %s",
                            e.what());
    } catch (...) {
        directLightUpstreamShaderPack_.reset();
        directLightUpstreamRuntimeError_ = "unknown exception";
        RadianceLogger::log("RayTracing", "ERROR",
                            "Upstream RT shader-pack runtime unknown exception");
    }
#endif
}

void RayTracingModule::refreshUpstreamDirectLightRuntime(uint32_t frameIndex) {
#ifndef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
    (void)frameIndex;
    return;
#else
    if (effectiveDirectLightBackend() != 1) { return; }
    if (!directLightUpstreamShaderPack_) {
        initUpstreamDirectLightRuntime();
    }
    if (!directLightUpstreamShaderPack_ || !directLightUpstreamPassRuntimeReady_) { return; }
    if (frameIndex >= hdrNoisyOutputImages_.size() || !hdrNoisyOutputImages_[frameIndex]) { return; }

    try {
        auto image = hdrNoisyOutputImages_[frameIndex];
        renderDiag("UpstreamRT runtime refresh begin frame=%u size=%ux%u", frameIndex, image->width(), image->height());
        g_crashRing.record("UpstreamRT:runtimeRefresh:start");
        auto expressionVariables = upstreamDirectLightExpressionVariables();
        bool hasRenderWidth = false;
        bool hasRenderHeight = false;
        for (auto &variable : expressionVariables) {
            if (variable.name == "RENDER_WIDTH") {
                variable.value = static_cast<double>(image->width());
                hasRenderWidth = true;
            } else if (variable.name == "RENDER_HEIGHT") {
                variable.value = static_cast<double>(image->height());
                hasRenderHeight = true;
            }
        }
        if (!hasRenderWidth) {
            expressionVariables.push_back({.name = "RENDER_WIDTH", .value = static_cast<double>(image->width())});
        }
        if (!hasRenderHeight) {
            expressionVariables.push_back({.name = "RENDER_HEIGHT", .value = static_cast<double>(image->height())});
        }
        g_crashRing.record("UpstreamRT:runtimeRefresh:variables");
        directLightUpstreamShaderPack_->setRuntimeResourceExpressionVariables(std::move(expressionVariables));
        g_crashRing.record("UpstreamRT:runtimeRefresh:ensure:start");
        directLightUpstreamShaderPack_->ensureRuntimeResources(image->width(), image->height());
        g_crashRing.record("UpstreamRT:runtimeRefresh:ensure:done");
        directLightUpstreamRuntimeResourcesReady_ =
            directLightUpstreamShaderPack_->runtimeResourcesReady();
        if (directLightUpstreamRuntimeResourcesReady_ &&
            frameIndex < directLightUpstreamDescriptorTables_.size() &&
            directLightUpstreamDescriptorTables_[frameIndex]) {
            g_crashRing.record("UpstreamRT:runtimeRefresh:bind:start");
            directLightUpstreamShaderPack_->bindRuntimeResources(
                directLightUpstreamDescriptorTables_[frameIndex], 5, frameIndex);
            g_crashRing.record("UpstreamRT:runtimeRefresh:bind:done");
        }
        renderDiag("UpstreamRT runtime refresh end ready=%d", directLightUpstreamRuntimeResourcesReady_ ? 1 : 0);
    } catch (const std::exception& e) {
        directLightUpstreamRuntimeResourcesReady_ = false;
        directLightUpstreamShaderCompileReady_ = false;
        directLightUpstreamRuntimeError_ = e.what();
        RadianceLogger::log("RayTracing", "ERROR",
                            "Upstream RT runtime resource refresh failed: %s",
                            e.what());
    } catch (...) {
        directLightUpstreamRuntimeResourcesReady_ = false;
        directLightUpstreamShaderCompileReady_ = false;
        directLightUpstreamRuntimeError_ = "unknown exception";
        RadianceLogger::log("RayTracing", "ERROR",
                            "Upstream RT runtime resource refresh unknown exception");
    }
#endif
}

#ifdef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
std::vector<ExpressionEvaluator::Variable> RayTracingModule::upstreamDirectLightExpressionVariables() const {
    std::vector<ExpressionEvaluator::Variable> variables;
    const uint32_t renderWidth = hdrNoisyOutputImages_.empty() || !hdrNoisyOutputImages_[0]
        ? 0u
        : hdrNoisyOutputImages_[0]->width();
    const uint32_t renderHeight = hdrNoisyOutputImages_.empty() || !hdrNoisyOutputImages_[0]
        ? 0u
        : hdrNoisyOutputImages_[0]->height();
    variables.push_back({.name = "RENDER_WIDTH", .value = static_cast<double>(renderWidth)});
    variables.push_back({.name = "RENDER_HEIGHT", .value = static_cast<double>(renderHeight)});
    auto world = Renderer::instance().world();
    auto chunks = world == nullptr ? nullptr : world->chunks();
    if (chunks != nullptr) {
        variables.push_back({.name = "CHUNK_NUM", .value = static_cast<double>(chunks->chunks().size())});
    }
    return variables;
}

double RayTracingModule::evaluateUpstreamDirectLightNumericExpression(
    const std::string &expression,
    const ShaderPack::ExecutionVariables &variables) {
    if (!directLightUpstreamShaderPack_) { return 1.0; }
    return directLightUpstreamShaderPack_->evaluateNumericExpression(
        ShaderPackLoader::Stage::RayTracing, expression, variables,
        upstreamDirectLightExpressionVariables(), true);
}

std::optional<std::reference_wrapper<ShaderPackLoader::VariableConfig>>
RayTracingModule::findUpstreamDirectLightExecutionVariableConfig(std::string_view name) {
    auto iter = directLightUpstreamExecutionVariableConfigs_.find(std::string(name));
    if (iter != directLightUpstreamExecutionVariableConfigs_.end()) { return std::ref(iter->second); }
    return std::nullopt;
}

std::optional<std::reference_wrapper<ShaderPack::RuntimeTexture>>
RayTracingModule::findUpstreamDirectLightRuntimeTexture(std::string_view name) {
    if (!directLightUpstreamShaderPack_) { return std::nullopt; }
    return directLightUpstreamShaderPack_->findRuntimeTexture(name);
}

std::optional<std::reference_wrapper<ShaderPack::RuntimeBuffer>>
RayTracingModule::findUpstreamDirectLightRuntimeBuffer(std::string_view name) {
    if (!directLightUpstreamShaderPack_) { return std::nullopt; }
    return directLightUpstreamShaderPack_->findRuntimeBuffer(name);
}

std::shared_ptr<vk::DeviceLocalImage> RayTracingModule::findUpstreamDirectLightTargetImage(
    const std::string &target,
    uint32_t frameIndex) {
    if (target == TARGET_RADIANCE) return hdrNoisyOutputImages_[frameIndex];
    if (target == TARGET_DIFFUSE_ALBEDO_METALLIC) return diffuseAlbedoImages_[frameIndex];
    if (target == TARGET_SPECULAR_ALBEDO) return specularAlbedoImages_[frameIndex];
    if (target == TARGET_NORMAL_ROUGHNESS) return normalRoughnessImages_[frameIndex];
    if (target == TARGET_MOTION_VECTOR) return motionVectorImages_[frameIndex];
    if (target == TARGET_LINEAR_DEPTH) return linearDepthImages_[frameIndex];
    if (target == TARGET_SPECULAR_HIT_DEPTH) return specularHitDepthImages_[frameIndex];
    if (target == TARGET_FIRST_HIT_DEPTH) return firstHitDepthImages_[frameIndex];
    if (target == TARGET_FIRST_HIT_DIFFUSE_DIRECT_LIGHT) return firstHitDiffuseDirectLightImages_[frameIndex];
    if (target == TARGET_FIRST_HIT_DIFFUSE_INDIRECT_LIGHT) return firstHitDiffuseIndirectLightImages_[frameIndex];
    if (target == TARGET_FIRST_HIT_SPECULAR) return firstHitSpecularImages_[frameIndex];
    if (target == TARGET_FIRST_HIT_CLEAR) return firstHitClearImages_[frameIndex];
    if (target == TARGET_FIRST_HIT_BASE_EMISSION) return firstHitBaseEmissionImages_[frameIndex];
    if (target == TARGET_FOG_IMAGE) return upstreamFogImages_[frameIndex];
    if (target == TARGET_FIRST_HIT_REFRACTION) return upstreamFirstHitRefractionImages_[frameIndex];
    if (auto runtimeTexture = findUpstreamDirectLightRuntimeTexture(target); runtimeTexture.has_value()) {
        if (runtimeTexture->get().config.imported) { return nullptr; }
        return directLightUpstreamShaderPack_->findRuntimeVKTexture(runtimeTexture->get(), frameIndex);
    }
    return nullptr;
}

void RayTracingModule::initUpstreamDirectLightDescriptorTables() {
    if (!directLightUpstreamShaderPack_ || !directLightUpstreamPackRuntimeReady_) { return; }
    auto framework = framework_.lock();
    if (!framework) { return; }
    const uint32_t size = framework->swapchain()->imageCount();
    directLightUpstreamDescriptorTables_.resize(size);
    constexpr VkShaderStageFlags allStages =
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
        VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    for (uint32_t i = 0; i < size; ++i) {
        vk::DescriptorTableBuilder builder;
        auto &set0 = builder.beginDescriptorLayoutSet();
        auto &set0Bindings = set0.beginDescriptorLayoutSetBinding();
        set0Bindings.defineDescriptorLayoutSetBinding({
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 4096,
            .stageFlags = allStages,
        });
        set0Bindings.endDescriptorLayoutSetBinding();
        set0.endDescriptorLayoutSet();

        auto &set1 = builder.beginDescriptorLayoutSet();
        auto &set1Bindings = set1.beginDescriptorLayoutSetBinding();
        set1Bindings
            .defineDescriptorLayoutSetBinding({0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                                               VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                                               nullptr})
            .defineDescriptorLayoutSetBinding({1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allStages, nullptr})
            .defineDescriptorLayoutSetBinding({2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allStages, nullptr})
            .defineDescriptorLayoutSetBinding({3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allStages, nullptr})
            .defineDescriptorLayoutSetBinding({4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allStages, nullptr})
            .defineDescriptorLayoutSetBinding({5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allStages, nullptr})
            .defineDescriptorLayoutSetBinding({6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allStages, nullptr})
            .defineDescriptorLayoutSetBinding({7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allStages, nullptr})
            .defineDescriptorLayoutSetBinding({8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allStages, nullptr})
            .defineDescriptorLayoutSetBinding({9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, allStages, nullptr});
        set1Bindings.endDescriptorLayoutSetBinding();
        set1.endDescriptorLayoutSet();

        auto &set2 = builder.beginDescriptorLayoutSet();
        auto &set2Bindings = set2.beginDescriptorLayoutSetBinding();
        set2Bindings
            .defineDescriptorLayoutSetBinding({0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, allStages, nullptr})
            .defineDescriptorLayoutSetBinding({1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, allStages, nullptr})
            .defineDescriptorLayoutSetBinding({2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, allStages, nullptr});
        set2Bindings.endDescriptorLayoutSetBinding();
        set2.endDescriptorLayoutSet();

        auto &set3 = builder.beginDescriptorLayoutSet();
        auto &set3Bindings = set3.beginDescriptorLayoutSetBinding();
        for (uint32_t binding = 0; binding < 15; ++binding) {
            set3Bindings.defineDescriptorLayoutSetBinding({
                .binding = binding,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = allStages,
            });
        }
        set3Bindings.endDescriptorLayoutSetBinding();
        set3.endDescriptorLayoutSet();

        auto &set4 = builder.beginDescriptorLayoutSet();
        auto &set4Bindings = set4.beginDescriptorLayoutSetBinding();
        for (uint32_t binding = 0; binding < 5; ++binding) {
            set4Bindings.defineDescriptorLayoutSetBinding({
                .binding = binding,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT,
            });
        }
        set4Bindings.endDescriptorLayoutSetBinding();
        set4.endDescriptorLayoutSet();

        directLightUpstreamShaderPack_->defineRuntimeResourceDescriptorSet(builder, allStages, allStages, allStages);
        directLightUpstreamShaderPack_->defineExecutionDescriptorSet(
            builder, ShaderPackLoader::Stage::RayTracing, allStages);

        directLightUpstreamDescriptorTables_[i] = builder.build(framework->device());
        auto table = directLightUpstreamDescriptorTables_[i];
        table->bindImage(hdrNoisyOutputImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 0);
        table->bindImage(diffuseAlbedoImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 1);
        table->bindImage(specularAlbedoImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 2);
        table->bindImage(normalRoughnessImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 3);
        table->bindImage(motionVectorImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 4);
        table->bindImage(linearDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 5);
        table->bindImage(specularHitDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 6);
        table->bindImage(firstHitDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 7);
        table->bindImage(firstHitDiffuseDirectLightImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 8);
        table->bindImage(firstHitDiffuseIndirectLightImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 9);
        table->bindImage(firstHitSpecularImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 10);
        table->bindImage(firstHitClearImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 11);
        table->bindImage(firstHitBaseEmissionImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 12);
        table->bindImage(upstreamFogImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 13);
        table->bindImage(upstreamFirstHitRefractionImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 14);
    }

    // Upstream descriptor tables are created lazily after Minecraft has usually
    // published its bindless texture atlas. Replay existing texture bindings so
    // upstream set 0 matches the legacy RT descriptor tables before any dispatch.
    Renderer::instance().textures()->bindAllTextures();
    RadianceLogger::log("RayTracing", "INFO",
                        "Upstream RT descriptor tables ready frames=%u texturesReplayed=1",
                        size);
}

void RayTracingModule::collectUpstreamDirectLightRayTracingRequests(
    RayTracingPass &pass,
    std::vector<ShaderPack::ShaderCreateInfo> &requests) {
    pass.missShaders.clear();
    pass.hitShaderGroups.clear();
    pass.hitGroupNameToIndex.clear();
    pass.shadowHitGroupIndex = 0;
    pass.fallbackHitGroupIndex = 0;
    pass.missGroupCount = 0;
    pass.hitGroupCount = 0;
    pass.isSharcUpdatePass = false;
    pass.querySharcEnabled = false;
    pass.missRequestCount_ = pass.config.missShaders.size();
    pass.hitRequestCount_ = 0;
    pass.rayGenRequestCount_ = 1;
    pass.sharcRequestCount_ = 0;
    pass.hitAssignments_.clear();

    const uint32_t executionSet = directLightUpstreamShaderPack_->executionSet(5u);
    for (const auto &miss : pass.config.missShaders) {
        requests.push_back({
            .path = miss.shaderPath,
            .stage = VK_SHADER_STAGE_MISS_BIT_KHR,
            .definitions = pass.config.definitions,
            .executionStage = ShaderPackLoader::Stage::RayTracing,
            .executionSet = executionSet,
        });
    }

    pass.hitShaderGroups.resize(pass.config.hitGroups.size());
    for (size_t groupIndex = 0; groupIndex < pass.config.hitGroups.size(); ++groupIndex) {
        const auto &groupConfig = pass.config.hitGroups[groupIndex];
        HitShaderGroup group;
        group.name = groupConfig.name;
        group.type = groupConfig.type;
        if (groupConfig.closestHit.has_value()) {
            requests.push_back({
                .path = *groupConfig.closestHit,
                .stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                .definitions = pass.config.definitions,
                .executionStage = ShaderPackLoader::Stage::RayTracing,
                .executionSet = executionSet,
            });
            pass.hitAssignments_.push_back({groupIndex, 0});
            pass.hitRequestCount_++;
        }
        if (groupConfig.anyHit.has_value()) {
            requests.push_back({
                .path = *groupConfig.anyHit,
                .stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                .definitions = pass.config.definitions,
                .executionStage = ShaderPackLoader::Stage::RayTracing,
                .executionSet = executionSet,
            });
            pass.hitAssignments_.push_back({groupIndex, 1});
            pass.hitRequestCount_++;
        }
        if (groupConfig.intersection.has_value()) {
            requests.push_back({
                .path = *groupConfig.intersection,
                .stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                .definitions = pass.config.definitions,
                .executionStage = ShaderPackLoader::Stage::RayTracing,
                .executionSet = executionSet,
            });
            pass.hitAssignments_.push_back({groupIndex, 2});
            pass.hitRequestCount_++;
        }
        pass.hitShaderGroups[groupIndex] = std::move(group);
        pass.hitGroupNameToIndex[pass.hitShaderGroups[groupIndex].name] = static_cast<uint32_t>(groupIndex);
        if (pass.hitShaderGroups[groupIndex].name == pass.config.defaultHitGroupName) {
            pass.fallbackHitGroupIndex = static_cast<uint32_t>(groupIndex);
        }
        if (pass.hitShaderGroups[groupIndex].name == "shadow") {
            pass.shadowHitGroupIndex = static_cast<uint32_t>(groupIndex);
        }
    }

    requests.push_back({
        .path = pass.config.rayGenShaderPath,
        .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        .definitions = pass.config.definitions,
        .executionStage = ShaderPackLoader::Stage::RayTracing,
        .executionSet = executionSet,
    });
}

void RayTracingModule::collectUpstreamDirectLightComputeRequests(
    const ComputePass &pass,
    std::vector<ShaderPack::ShaderCreateInfo> &requests) {
    requests.push_back({
        .path = pass.config.computeShaderPath,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .definitions = pass.config.definitions,
        .executionStage = ShaderPackLoader::Stage::RayTracing,
        .executionSet = directLightUpstreamShaderPack_->executionSet(5u),
    });
}

void RayTracingModule::buildUpstreamDirectLightRayTracingPass(
    RayTracingPass &pass,
    std::shared_ptr<vk::Device> device,
    const std::vector<std::shared_ptr<vk::Shader>> &compiledShaders,
    size_t &shaderOffset) {
    auto framework = framework_.lock();
    if (!framework) { return; }
    for (size_t i = 0; i < pass.missRequestCount_; ++i) {
        pass.missShaders.push_back({
            .name = pass.config.missShaders[i].name,
            .index = static_cast<uint32_t>(i),
            .shader = compiledShaders[shaderOffset + i],
        });
    }
    shaderOffset += pass.missRequestCount_;
    for (size_t i = 0; i < pass.hitRequestCount_; ++i) {
        auto &group = pass.hitShaderGroups[pass.hitAssignments_[i].groupIndex];
        switch (pass.hitAssignments_[i].shaderType) {
            case 0: group.closestHitShader = compiledShaders[shaderOffset + i]; break;
            case 1: group.anyHitShader = compiledShaders[shaderOffset + i]; break;
            case 2: group.intersectionShader = compiledShaders[shaderOffset + i]; break;
            default: break;
        }
    }
    shaderOffset += pass.hitRequestCount_;
    if (pass.hitGroupNameToIndex.find("shadow") == pass.hitGroupNameToIndex.end()) {
        pass.shadowHitGroupIndex = pass.fallbackHitGroupIndex;
    }
    pass.hitGroupCount = static_cast<uint32_t>(pass.hitShaderGroups.size());
    pass.missGroupCount = static_cast<uint32_t>(pass.missShaders.size());
    pass.rayGenQueryShader = compiledShaders[shaderOffset++];
    pass.rayGenUpdateShader = pass.rayGenQueryShader;

    vk::RayTracingPipelineBuilder builder;
    auto &stageBuilder = builder.beginShaderStage();
    uint32_t stageIndex = 0;
    stageBuilder.defineShaderStage(pass.rayGenQueryShader, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    stageIndex++;
    std::vector<uint32_t> missStageIndices;
    for (const auto &missShader : pass.missShaders) {
        missStageIndices.push_back(stageIndex);
        stageBuilder.defineShaderStage(missShader.shader, VK_SHADER_STAGE_MISS_BIT_KHR);
        stageIndex++;
    }
    struct HitStageIndices {
        VkRayTracingShaderGroupTypeKHR type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        uint32_t closestHit = VK_SHADER_UNUSED_KHR;
        uint32_t anyHit = VK_SHADER_UNUSED_KHR;
        uint32_t intersection = VK_SHADER_UNUSED_KHR;
    };
    std::vector<HitStageIndices> hitStageIndices;
    for (const auto &group : pass.hitShaderGroups) {
        HitStageIndices indices;
        indices.type = group.type;
        if (group.closestHitShader) {
            indices.closestHit = stageIndex;
            stageBuilder.defineShaderStage(group.closestHitShader, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
            stageIndex++;
        }
        if (group.anyHitShader) {
            indices.anyHit = stageIndex;
            stageBuilder.defineShaderStage(group.anyHitShader, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
            stageIndex++;
        }
        if (group.intersectionShader) {
            indices.intersection = stageIndex;
            stageBuilder.defineShaderStage(group.intersectionShader, VK_SHADER_STAGE_INTERSECTION_BIT_KHR);
            stageIndex++;
        }
        hitStageIndices.push_back(indices);
    }
    stageBuilder.endShaderStage();

    auto &groupBuilder = builder.beginShaderGroup();
    groupBuilder.defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0, VK_SHADER_UNUSED_KHR,
                                   VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
    for (uint32_t missStageIndex : missStageIndices) {
        groupBuilder.defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, missStageIndex,
                                       VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR);
    }
    for (const auto &indices : hitStageIndices) {
        groupBuilder.defineShaderGroup(indices.type, VK_SHADER_UNUSED_KHR, indices.closestHit, indices.anyHit,
                                       indices.intersection);
    }
    groupBuilder.endShaderGroup();

    pass.queryPipeline = builder.definePipelineLayout(directLightUpstreamDescriptorTables_[0]).build(device);
    pass.updatePipeline = pass.queryPipeline;
    const uint32_t frameCount = framework->swapchain()->imageCount();
    pass.querySbts.resize(frameCount);
    for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
        pass.querySbts[frameIndex] = vk::SBT::create(
            framework->physicalDevice(), device, framework->vma(),
            pass.queryPipeline, pass.missGroupCount, pass.hitGroupCount);
    }
}

void RayTracingModule::buildUpstreamDirectLightComputePass(
    ComputePass &pass,
    std::shared_ptr<vk::Device> device,
    const std::vector<std::shared_ptr<vk::Shader>> &compiledShaders,
    size_t &shaderOffset) {
    pass.computeShader = compiledShaders[shaderOffset++];
    pass.pipeline = vk::ComputePipelineBuilder{}
        .defineShader(pass.computeShader)
        .definePipelineLayout(directLightUpstreamDescriptorTables_[0])
        .build(device);
}

void RayTracingModule::uploadUpstreamDirectLightStaticSbts(std::shared_ptr<vk::Device> device) {
    auto framework = framework_.lock();
    if (!framework) { return; }
    auto commandPool = vk::CommandPool::create(framework->physicalDevice(), device);
    auto commandBuffer = vk::CommandBuffer::create(device, commandPool);
    commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    for (auto &passVariant : directLightUpstreamPasses_) {
        if (auto pass = std::get_if<std::shared_ptr<RayTracingPass>>(&passVariant)) {
            (void) pass;
        }
    }
    commandBuffer->end()->submitMainQueueIndividual(device);
    vkQueueWaitIdle(device->mainVkQueue());
}

void RayTracingModule::initUpstreamDirectLightPipelines() {
    if (!directLightUpstreamShaderPack_ || !directLightUpstreamPassRuntimeReady_ ||
        directLightUpstreamDescriptorTables_.empty() || !directLightUpstreamDescriptorTables_[0]) {
        return;
    }
    auto framework = framework_.lock();
    if (!framework) { return; }
    auto device = framework->device();
    directLightUpstreamShaderCompileReady_ = false;
    directLightUpstreamPipelineReady_ = false;
    directLightUpstreamSbtReady_ = false;
    directLightUpstreamPasses_.clear();
    directLightUpstreamPassNameToPass_.clear();
    directLightUpstreamExecutionVariableConfigs_.clear();
    directLightUpstreamGlobalVariables_.clear();
    directLightUpstreamShaderPack_->copyStageExecutionState(
        ShaderPackLoader::Stage::RayTracing,
        directLightUpstreamExecutionVariableConfigs_,
        directLightUpstreamGlobalVariables_);
    directLightUpstreamGlobalVariables_["ADV_TRANS_LUT_READY"] = "true";

    size_t executionBufferSize =
        directLightUpstreamShaderPack_->shaderPack().rayTracingExecution.variables.size() * sizeof(float);
    std::vector<ShaderPack::ShaderCreateInfo> requests;
    size_t skippedInactivePasses = 0;
    for (const auto &passConfig : directLightUpstreamShaderPack_->shaderPack().passes) {
        if (passConfig.stage != ShaderPackLoader::Stage::RayTracing) { continue; }
        switch (passConfig.type) {
            case ShaderPackLoader::PassConfig::Type::RayTracing: {
                if (!upstreamDirectLightExecutorPassIncluded(passConfig.rayTracing.name)) {
                    skippedInactivePasses++;
                    break;
                }
                auto pass = std::make_shared<RayTracingPass>();
                pass->config = passConfig.rayTracing;
                pass->querySharcEnabled = false;
                pass->isSharcUpdatePass = false;
                pass->executionBuffer = ShaderPack::createPassExecutionBuffer(
                    device, framework->vma(), executionBufferSize);
                collectUpstreamDirectLightRayTracingRequests(*pass, requests);
                directLightUpstreamPasses_.push_back(pass);
                directLightUpstreamPassNameToPass_.emplace(pass->config.name, pass);
                break;
            }
            case ShaderPackLoader::PassConfig::Type::Compute: {
                if (!upstreamDirectLightExecutorPassIncluded(passConfig.compute.name)) {
                    skippedInactivePasses++;
                    break;
                }
                auto pass = std::make_shared<ComputePass>();
                pass->config = passConfig.compute;
                pass->executionBuffer = ShaderPack::createPassExecutionBuffer(
                    device, framework->vma(), executionBufferSize);
                collectUpstreamDirectLightComputeRequests(*pass, requests);
                directLightUpstreamPasses_.push_back(pass);
                directLightUpstreamPassNameToPass_.emplace(pass->config.name, pass);
                break;
            }
            default:
                break;
        }
    }

    try {
        RadianceLogger::log("RayTracing", "INFO",
                            "Upstream RT executor compile start selectedPasses=%zu skippedPasses=%zu shaderRequests=%zu",
                            directLightUpstreamPasses_.size(), skippedInactivePasses, requests.size());
        renderDiag("UpstreamRT pipeline compile start selectedPasses=%zu skippedPasses=%zu shaderRequests=%zu",
                   directLightUpstreamPasses_.size(), skippedInactivePasses, requests.size());
        g_crashRing.record("UpstreamRT:pipeline:compile:start");
        auto shaders = directLightUpstreamShaderPack_->createShaders(device, requests);
        g_crashRing.record("UpstreamRT:pipeline:compile:done");
        renderDiag("UpstreamRT pipeline compile done shaders=%zu", shaders.size());
        directLightUpstreamShaderCompileReady_ = !shaders.empty() || requests.empty();
        size_t shaderOffset = 0;
        for (auto &passVariant : directLightUpstreamPasses_) {
            if (auto pass = std::get_if<std::shared_ptr<RayTracingPass>>(&passVariant)) {
                const std::string tag = "UpstreamRT:buildRT:" + (*pass)->config.name;
                g_crashRing.record(tag.c_str());
                renderDiag("UpstreamRT build RT pass=%s", (*pass)->config.name.c_str());
                buildUpstreamDirectLightRayTracingPass(**pass, device, shaders, shaderOffset);
                renderDiag("UpstreamRT build RT pass done=%s", (*pass)->config.name.c_str());
            } else if (auto pass = std::get_if<std::shared_ptr<ComputePass>>(&passVariant)) {
                const std::string tag = "UpstreamRT:buildC:" + (*pass)->config.name;
                g_crashRing.record(tag.c_str());
                renderDiag("UpstreamRT build compute pass=%s", (*pass)->config.name.c_str());
                buildUpstreamDirectLightComputePass(**pass, device, shaders, shaderOffset);
                renderDiag("UpstreamRT build compute pass done=%s", (*pass)->config.name.c_str());
            }
        }
        directLightUpstreamPipelineReady_ = shaderOffset == shaders.size() && !directLightUpstreamPasses_.empty();
        g_crashRing.record("UpstreamRT:pipeline:sbtUpload:start");
        renderDiag("UpstreamRT static SBT upload start");
        uploadUpstreamDirectLightStaticSbts(device);
        g_crashRing.record("UpstreamRT:pipeline:sbtUpload:done");
        renderDiag("UpstreamRT static SBT upload done");
        directLightUpstreamSbtReady_ = directLightUpstreamPipelineReady_;
        RadianceLogger::log("RayTracing", "INFO",
                            "Upstream RT executor pipelines ready passes=%zu shaders=%zu",
                            directLightUpstreamPasses_.size(), shaders.size());
    } catch (const std::exception &e) {
        directLightUpstreamRuntimeError_ = e.what();
        directLightUpstreamShaderCompileReady_ = false;
        directLightUpstreamPipelineReady_ = false;
        directLightUpstreamSbtReady_ = false;
        RadianceLogger::log("RayTracing", "ERROR", "Upstream RT executor pipeline build failed: %s", e.what());
    }
}

static const char *upstreamPassProfileName(const std::string &name) {
    if (name == "primary") return "RT.Upstream.Primary";
    if (name == "precompute_light_neighborhoods") return "RT.Upstream.PrecomputeLightNeighborhoods";
    if (name == "generate_initial_samples") return "RT.Upstream.Initial";
    if (name == "visibility") return "RT.Upstream.Visibility";
    if (name == "temporal_reuse") return "RT.Upstream.Temporal";
    if (name == "spatial_reuse") return "RT.Upstream.Spatial";
    if (name == "direct_light") return "RT.Upstream.DirectLight";
    if (name == "final_compose") return "RT.Upstream.FinalCompose";
    if (name == "volumetric_light_clear") return "RT.Upstream.VolumetricLightClear";
    if (name == "cont_reflection") return "RT.Upstream.ContinuousReflection";
    if (name == "cont_refraction") return "RT.Upstream.ContinuousRefraction";
    return "RT.Upstream.Pass";
}

void RayTracingModule::renderUpstreamDirectLightRayTracingPass(
    RayTracingPass &pass,
    RayTracingModuleContext &context,
    const ShaderPack::ExecutionVariables &variables) {
    auto frameworkContext = context.frameworkContext.lock();
    auto worldCommandBuffer = frameworkContext->worldCommandBuffer;
    const uint32_t frameIndex = frameworkContext->frameIndex;
    if (!context.worldPrepareContext || !context.worldPrepareContext->tlas ||
        frameIndex >= pass.querySbts.size() || !pass.querySbts[frameIndex] ||
        !pass.queryPipeline || !context.directLightUpstreamDescriptorTable) {
        return;
    }

    auto hitGroupIndex = [&](const char *name, uint32_t fallback) {
        auto iter = pass.hitGroupNameToIndex.find(name);
        return iter == pass.hitGroupNameToIndex.end() ? fallback : iter->second;
    };
    const uint32_t defaultHit = pass.fallbackHitGroupIndex;
    const uint32_t shadowHit = hitGroupIndex("shadow", defaultHit);
    const uint32_t cloudsHit = hitGroupIndex("clouds", defaultHit);
    const uint32_t waterHit = hitGroupIndex("water_mask", defaultHit);
    const uint32_t portalHit = hitGroupIndex("end_portal", defaultHit);
    const uint32_t gatewayHit = hitGroupIndex("end_gateway", defaultHit);
    const uint32_t noReflectHit = hitGroupIndex("world_no_reflect", defaultHit);
    std::vector<uint32_t> hitIndices;
    hitIndices.reserve(context.worldPrepareContext->lastGeometryTypes_.size());
    for (auto typeIndex : context.worldPrepareContext->lastGeometryTypes_) {
        switch (static_cast<World::GeometryTypes>(typeIndex)) {
            case World::GeometryTypes::SHADOW: hitIndices.push_back(shadowHit); break;
            case World::GeometryTypes::WORLD_NO_REFLECT: hitIndices.push_back(noReflectHit); break;
            case World::GeometryTypes::WORLD_CLOUD: hitIndices.push_back(cloudsHit); break;
            case World::GeometryTypes::BOAT_WATER_MASK: hitIndices.push_back(waterHit); break;
            case World::GeometryTypes::END_PORTAL: hitIndices.push_back(portalHit); break;
            case World::GeometryTypes::END_GATE_WAY: hitIndices.push_back(gatewayHit); break;
            default: hitIndices.push_back(defaultHit); break;
        }
    }
    if (!hitIndices.empty()) {
        pass.querySbts[frameIndex]->setupHitSBT(hitIndices);
    }

    auto evalDim = [&](const std::string &expression) {
        double value = evaluateUpstreamDirectLightNumericExpression(expression, variables);
        return static_cast<uint32_t>(std::max(1.0, std::ceil(value)));
    };
    const uint32_t traceWidth = evalDim(pass.config.width);
    const uint32_t traceHeight = evalDim(pass.config.height);
    const uint32_t traceDepth = evalDim(pass.config.depth);

    worldCommandBuffer->barriersMemory({{
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
    }});
    worldCommandBuffer->bindDescriptorTable(context.directLightUpstreamDescriptorTable,
                                            VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
        ->bindRTPipeline(pass.queryPipeline)
        ->raytracing(pass.querySbts[frameIndex], traceWidth, traceHeight, traceDepth);
}

void RayTracingModule::renderUpstreamDirectLightComputePass(
    ComputePass &pass,
    RayTracingModuleContext &context,
    const ShaderPack::ExecutionVariables &variables) {
    auto frameworkContext = context.frameworkContext.lock();
    auto worldCommandBuffer = frameworkContext->worldCommandBuffer;
    if (!pass.pipeline || !context.directLightUpstreamDescriptorTable) { return; }
    auto evalDim = [&](const std::string &expression) {
        double value = evaluateUpstreamDirectLightNumericExpression(expression, variables);
        return static_cast<uint32_t>(std::max(1.0, std::ceil(value)));
    };
    const uint32_t gx = evalDim(pass.config.gx);
    const uint32_t gy = evalDim(pass.config.gy);
    const uint32_t gz = evalDim(pass.config.gz);
    worldCommandBuffer->barriersMemory({{
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
    }});
    worldCommandBuffer->bindDescriptorTable(context.directLightUpstreamDescriptorTable,
                                            VK_PIPELINE_BIND_POINT_COMPUTE)
        ->bindComputePipeline(pass.pipeline);
    vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), gx, gy, gz);
}

bool RayTracingModule::renderUpstreamDirectLight(RayTracingModuleContext &context) {
    if (effectiveDirectLightBackend() != 1 || !directLightUpstreamShaderPack_ ||
        !directLightUpstreamPackRuntimeReady_ || !directLightUpstreamPassRuntimeReady_ ||
        !context.directLightUpstreamDescriptorTable) {
        return false;
    }
    auto frameworkContext = context.frameworkContext.lock();
    if (!frameworkContext) { return false; }
    refreshUpstreamDirectLightRuntime(frameworkContext->frameIndex);
    if (!directLightUpstreamRuntimeResourcesReady_) { return false; }
    if (!directLightUpstreamPipelineReady_ || !directLightUpstreamSbtReady_) {
        RadianceLogger::log("RayTracing", "INFO", "Upstream RT executor lazy pipeline init start");
        g_crashRing.record("UpstreamRT:pipelineLazy:start");
        initUpstreamDirectLightPipelines();
        g_crashRing.record("UpstreamRT:pipelineLazy:done");
        RadianceLogger::log("RayTracing", "INFO",
                            "Upstream RT executor lazy pipeline init done shader=%d pipeline=%d sbt=%d",
                            directLightUpstreamShaderCompileReady_ ? 1 : 0,
                            directLightUpstreamPipelineReady_ ? 1 : 0,
                            directLightUpstreamSbtReady_ ? 1 : 0);
    }
    if (!directLightUpstreamShaderCompileReady_ ||
        !directLightUpstreamPipelineReady_ ||
        !directLightUpstreamSbtReady_) {
        return false;
    }
    auto buffers = Renderer::instance().buffers();
    auto world = Renderer::instance().world();
    auto chunks = world ? world->chunks() : nullptr;
    auto table = context.directLightUpstreamDescriptorTable;
    table->bindBuffer(buffers->worldUniformBuffer(), 2, 0);
    table->bindBuffer(buffers->lastWorldUniformBuffer(), 2, 1);
    table->bindBuffer(buffers->skyUniformBuffer(), 2, 2);
    if (context.worldPrepareContext && context.worldPrepareContext->tlas) {
        table->bindAS(context.worldPrepareContext->tlas, 1, 0);
        table->bindBuffer(context.worldPrepareContext->blasOffsetsBuffer, 1, 1);
        table->bindBuffer(context.worldPrepareContext->indexBufferAddr, 1, 2);
        table->bindBuffer(context.worldPrepareContext->lastIndexBufferAddr, 1, 3);
        table->bindBuffer(context.worldPrepareContext->vertexBufferAddr, 1, 4);
        table->bindBuffer(context.worldPrepareContext->vertexBufferAddr, 1, 5);
        table->bindBuffer(context.worldPrepareContext->lastVertexBufferAddr, 1, 6);
        table->bindBuffer(buffers->textureMappingBuffer(), 1, 7);
        table->bindBuffer(context.worldPrepareContext->lastObjToWorldMat, 1, 8);
        if (chunks && chunks->chunkPackedData()) {
            table->bindBuffer(chunks->chunkPackedData(), 1, 9);
        } else {
            table->bindBuffer(context.worldPrepareContext->blasOffsetsBuffer, 1, 9);
        }
    }

    uint32_t brdfFlags = 0;
    if (Renderer::options.multiScatterGGX) { brdfFlags |= 128u; }
    if (Renderer::options.eonDiffuse) { brdfFlags |= 256u; }
    directLightUpstreamGlobalVariables_["RADIANCE_BRDF_FLAGS"] = std::to_string(brdfFlags);
    directLightUpstreamGlobalVariables_["ADV_TRANS_LUT_READY"] = "true";

    ShaderPack::ExecutionVariables variables;
    for (const auto &[name, value] : directLightUpstreamGlobalVariables_) {
        auto config = findUpstreamDirectLightExecutionVariableConfig(name);
        variables.emplace(name, ShaderPack::ExecutionVariable{
            .name = name,
            .value = value,
            .type = config.has_value() ? config->get().type : "",
        });
    }

    auto executePass = [&](const std::string &passName, ShaderPack::ExecutionVariables &passVariables) {
        auto iter = directLightUpstreamPassNameToPass_.find(passName);
        if (iter == directLightUpstreamPassNameToPass_.end()) {
            static std::unordered_set<std::string> loggedMissingPasses;
            if (loggedMissingPasses.insert(passName).second) {
                RadianceLogger::log("RayTracing", "INFO",
                                    "Upstream RT executor skipping unsupported pass %s",
                                    passName.c_str());
            }
            return;
        }
        const std::string crashBegin = "UpstreamRT:pass:" + passName + ":begin";
        g_crashRing.record(crashBegin.c_str());
        ScopedGpuProfile profile(frameworkContext->worldCommandBuffer->vkCommandBuffer(),
                                 upstreamPassProfileName(passName));
        std::visit([&](auto &pass) {
            if (pass->executionBuffer) {
                directLightUpstreamShaderPack_->uploadExecutionBuffer(
                    ShaderPackLoader::Stage::RayTracing, pass->executionBuffer, passVariables,
                    frameworkContext->worldCommandBuffer, table,
                    directLightUpstreamShaderPack_->executionSet(5u),
                    frameworkContext->framework.lock()->physicalDevice()->mainQueueIndex(),
                    VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            }
            using T = std::decay_t<decltype(pass)>;
            if constexpr (std::is_same_v<T, std::shared_ptr<RayTracingPass>>) {
                renderUpstreamDirectLightRayTracingPass(*pass, context, passVariables);
            } else if constexpr (std::is_same_v<T, std::shared_ptr<ComputePass>>) {
                renderUpstreamDirectLightComputePass(*pass, context, passVariables);
            }
        }, iter->second);
        profile.close();
        const std::string crashEnd = "UpstreamRT:pass:" + passName + ":end";
        g_crashRing.record(crashEnd.c_str());
    };

    try {
        directLightUpstreamShaderPack_->executeCommands(
            ShaderPackLoader::Stage::RayTracing,
            directLightUpstreamShaderPack_->execution(ShaderPackLoader::Stage::RayTracing).commands,
            variables,
            upstreamDirectLightExpressionVariables(),
            true,
            1u << 16,
            executePass);
        for (auto &[name, value] : directLightUpstreamGlobalVariables_) {
            auto iter = variables.find(name);
            if (iter != variables.end()) { value = iter->second.value; }
        }
        return true;
    } catch (const std::exception &e) {
        directLightUpstreamRuntimeError_ = e.what();
        RadianceLogger::log("RayTracing", "ERROR", "Upstream RT executor dispatch failed: %s", e.what());
        return false;
    }
}
#endif

void RayTracingModule::initSharcBuffers() {
    auto framework = framework_.lock();
    if (!framework) return;

    RadianceLogger::log("RayTracing", "INFO",
                        "SHARC buffers create start requestedExp=%d effectiveExp=%d capacity=%u bytes=%llu",
                        Renderer::options.sharcCapacityExponent,
                        effectiveSharcCapacityExponent(),
                        sharcCapacity_,
                        static_cast<unsigned long long>(sharcCapacity_) * 40ull);
    g_crashRing.record("SHARC:buffers:create:start");
    VkBufferUsageFlags sharcUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
        | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    sharcHashEntries_ = vk::DeviceLocalBuffer::create(
        framework->vma(), framework->device(), sharcCapacity_ * 8, sharcUsage);
    sharcAccumulation_ = vk::DeviceLocalBuffer::create(
        framework->vma(), framework->device(), sharcCapacity_ * 16, sharcUsage);
    sharcResolved_ = vk::DeviceLocalBuffer::create(
        framework->vma(), framework->device(), sharcCapacity_ * 16, sharcUsage);

    sharcBuffersInitialized_ = false; // Will zero-fill on first use
    RadianceLogger::log("RayTracing", "INFO", "SHARC buffers create done ok=%d bytes=%llu",
                        (sharcHashEntries_ && sharcAccumulation_ && sharcResolved_) ? 1 : 0,
                        static_cast<unsigned long long>(sharcCapacity_) * 40ull);
    g_crashRing.record("SHARC:buffers:create:done");
}

void RayTracingModule::initSharcUpdatePipeline() {
    auto framework = framework_.lock();
    if (!framework) return;
    auto device = framework->device();

    RadianceLogger::log("RayTracing", "INFO", "SHARC update pipeline create start");
    g_crashRing.record("SHARC:updatePipeline:create:start");
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    sharcUpdateRayGenShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/sharc_update_rgen.spv").string());
    if (!sharcUpdateRayGenShader_) {
        std::cerr << "[SHARC] Failed to load sharc_update_rgen.spv" << std::endl;
        return;
    }
    const bool displacementRequested = Renderer::options.pomEnabled && worldSolidTransparentClosestHitShader_;
    const bool useShaderDisplacement = displacementRequested && shaderDisplacementRuntimeAllowed();
    auto activeWorldSolidTransparentClosestHitShader =
        useShaderDisplacement
            ? worldSolidTransparentClosestHitShader_
            : worldSolidTransparentNoDisplacementClosestHitShader_;
    if (!activeWorldSolidTransparentClosestHitShader) {
        activeWorldSolidTransparentClosestHitShader = worldSolidTransparentClosestHitShader_;
    }
    RadianceLogger::log("RayTracing", "INFO",
                        "SHARC closest-hit variant: %s (displacementRequested=%d displacementRuntimeAllowed=%d noDisplacementShader=%d)",
                        useShaderDisplacement ? "displacement" : "no_displacement",
                        displacementRequested ? 1 : 0,
                        shaderDisplacementRuntimeAllowed() ? 1 : 0,
                        worldSolidTransparentNoDisplacementClosestHitShader_ ? 1 : 0);

    // Build update RT pipeline with same CHS/AHS/miss shaders as main pipeline
    sharcUpdatePipeline_ =
        vk::RayTracingPipelineBuilder{}
            .beginShaderStage()
            .defineShaderStage(sharcUpdateRayGenShader_, VK_SHADER_STAGE_RAYGEN_BIT_KHR)                     // 0 (update rgen)
            .defineShaderStage(worldRayMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                            // 1
            .defineShaderStage(handRayMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                             // 2
            .defineShaderStage(shadowRayMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                           // 3
            .defineShaderStage(activeWorldSolidTransparentClosestHitShader, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)  // 4
            .defineShaderStage(worldNoReflectClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)         // 5
            .defineShaderStage(worldCloudClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)             // 6
            .defineShaderStage(worldTransparentAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)               // 7
            .defineShaderStage(worldNoReflectAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                 // 8
            .defineShaderStage(worldCloudAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                     // 9
            .defineShaderStage(shadowRayClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)              // 10
            .defineShaderStage(shadowAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                         // 11
            .defineShaderStage(boatWaterMaskClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)          // 12
            .defineShaderStage(boatWaterMaskAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                  // 13
            .defineShaderStage(endPortalClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)              // 14
            .defineShaderStage(endPortalAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                      // 15
            .defineShaderStage(endGatewayClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)             // 16
            .defineShaderStage(endGatewayAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                     // 17
            .defineShaderStage(pointLightShadowMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                    // 18
            .endShaderStage()
            .beginShaderGroup()
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR)
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR)
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 2, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR)
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 3, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR)
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 18, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR) // point light shadow miss
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 10, 11,
                               VK_SHADER_UNUSED_KHR) // shadow
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 4,
                               VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR) // world solid
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 4, 7,
                               VK_SHADER_UNUSED_KHR) // world transparent
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 5, 8,
                               VK_SHADER_UNUSED_KHR) // world no reflect
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 6, 9,
                               VK_SHADER_UNUSED_KHR) // world cloud
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 12, 13,
                               VK_SHADER_UNUSED_KHR) // boat water mask
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 14, 15,
                               VK_SHADER_UNUSED_KHR) // end portal
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 16, 17,
                               VK_SHADER_UNUSED_KHR) // end gateway
            .endShaderGroup()
            .definePipelineLayout(rayTracingDescriptorTables_[0])
            .build(device);

    if (!sharcUpdatePipeline_) {
        std::cerr << "[SHARC] Failed to create update RT pipeline" << std::endl;
        RadianceLogger::log("RayTracing", "ERROR", "SHARC update pipeline create done ok=0");
        g_crashRing.record("SHARC:updatePipeline:create:failed");
        return;
    }

    // Create SBTs for the update pipeline (same structure as main: 4 miss, 8 hit groups)
    sharcUpdateSbts_.resize(framework->swapchain()->imageCount());
    for (int i = 0; i < framework->swapchain()->imageCount(); i++) {
        sharcUpdateSbts_[i] = vk::SBT::create(framework->physicalDevice(), framework->device(), framework->vma(),
                                               sharcUpdatePipeline_, 4, 8);
    }
    RadianceLogger::log("RayTracing", "INFO", "SHARC update pipeline create done ok=1 sbts=%llu",
                        static_cast<unsigned long long>(sharcUpdateSbts_.size()));
    g_crashRing.record("SHARC:updatePipeline:create:done");
}

void RayTracingModule::initSharcResolvePipeline() {
    auto framework = framework_.lock();
    if (!framework) return;
    auto device = framework->device();
    VkDevice dev = device->vkDevice();

    // Destroy existing handles before recreating (prevents resource leak on rebuild)
    RadianceLogger::log("RayTracing", "INFO", "SHARC resolve pipeline create start");
    g_crashRing.record("SHARC:resolvePipeline:create:start");
    if (sharcResolvePipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, sharcResolvePipeline_, nullptr);
        sharcResolvePipeline_ = VK_NULL_HANDLE;
    }
    if (sharcResolvePipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, sharcResolvePipelineLayout_, nullptr);
        sharcResolvePipelineLayout_ = VK_NULL_HANDLE;
    }

    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    sharcResolveShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/sharc_resolve_comp.spv").string());
    if (!sharcResolveShader_) {
        std::cerr << "[SHARC] Failed to load sharc_resolve_comp.spv" << std::endl;
        return;
    }

    // Pipeline layout with push constant only (no descriptor sets — uses BDA)
    VkPushConstantRange pushRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 76}; // 19 fields × 4 bytes (with uint64_t alignment)
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(dev, &pipelineLayoutInfo, nullptr, &sharcResolvePipelineLayout_);

    VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = sharcResolveShader_->vkShaderModule();
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = sharcResolvePipelineLayout_;
    VkResult result = vkCreateComputePipelines(dev, device->pipelineCache(), 1, &pipelineInfo, nullptr, &sharcResolvePipeline_);
    if (result != VK_SUCCESS) {
        std::cerr << "[SHARC] Failed to create resolve compute pipeline: " << result << std::endl;
        sharcResolvePipeline_ = VK_NULL_HANDLE;
    }
    RadianceLogger::log("RayTracing", result == VK_SUCCESS ? "INFO" : "ERROR",
                        "SHARC resolve pipeline create done ok=%d result=%d",
                        result == VK_SUCCESS ? 1 : 0, static_cast<int>(result));
    g_crashRing.record(result == VK_SUCCESS ? "SHARC:resolvePipeline:create:done" : "SHARC:resolvePipeline:create:failed");
}

void RayTracingModule::initSharcQueryPipeline() {
#ifndef MCVR_ENABLE_SHARC_QUERY_PASS
    return;
#else
    auto framework = framework_.lock();
    if (!framework) return;
    auto device = framework->device();
    VkDevice dev = device->vkDevice();
    uint32_t size = framework->swapchain()->imageCount();

    RadianceLogger::log("RayTracing", "INFO", "SHARC query pipeline create start");
    g_crashRing.record("SHARC:queryPipeline:create:start");

    if (sharcQueryPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, sharcQueryPipeline_, nullptr);
        sharcQueryPipeline_ = VK_NULL_HANDLE;
    }
    if (sharcQueryPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, sharcQueryPipelineLayout_, nullptr);
        sharcQueryPipelineLayout_ = VK_NULL_HANDLE;
    }
    if (sharcQueryDescSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(dev, sharcQueryDescSetLayout_, nullptr);
        sharcQueryDescSetLayout_ = VK_NULL_HANDLE;
    }
    if (sharcQueryDescPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(dev, sharcQueryDescPool_, nullptr);
        sharcQueryDescPool_ = VK_NULL_HANDLE;
    }

    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    sharcQueryShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/sharc_query_comp.spv").string());
    if (!sharcQueryShader_) {
        std::cerr << "[SHARC] Failed to load sharc_query_comp.spv" << std::endl;
        RadianceLogger::log("RayTracing", "ERROR", "SHARC query shader load failed");
        g_crashRing.record("SHARC:queryPipeline:create:missingShader");
        return;
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    VkResult layoutResult = vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &sharcQueryDescSetLayout_);
    if (layoutResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "SHARC query descriptor layout failed result=%d",
                            static_cast<int>(layoutResult));
        g_crashRing.record("SHARC:queryPipeline:create:layoutFailed");
        return;
    }

    struct SharcQueryPushConstant {
        int32_t width, height, mode, reserved0;
        uint64_t hashEntriesBDA, resolvedBDA;
        float cameraX, cameraY, cameraZ, sceneScale;
        uint32_t capacity;
        float roughnessThreshold;
        uint32_t frameIndex;
        uint32_t reserved1;
    };
    VkPushConstantRange pushRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SharcQueryPushConstant)};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &sharcQueryDescSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    VkResult pipelineLayoutResult = vkCreatePipelineLayout(dev, &pipelineLayoutInfo, nullptr, &sharcQueryPipelineLayout_);
    if (pipelineLayoutResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "SHARC query pipeline layout failed result=%d",
                            static_cast<int>(pipelineLayoutResult));
        g_crashRing.record("SHARC:queryPipeline:create:pipelineLayoutFailed");
        return;
    }

    VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = sharcQueryShader_->vkShaderModule();
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = sharcQueryPipelineLayout_;
    VkResult pipelineResult = vkCreateComputePipelines(dev, device->pipelineCache(), 1, &pipelineInfo, nullptr,
                                                       &sharcQueryPipeline_);
    if (pipelineResult != VK_SUCCESS) {
        sharcQueryPipeline_ = VK_NULL_HANDLE;
        RadianceLogger::log("RayTracing", "ERROR", "SHARC query pipeline create done ok=0 result=%d",
                            static_cast<int>(pipelineResult));
        g_crashRing.record("SHARC:queryPipeline:create:failed");
        return;
    }

    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7u * size},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, size},
    };
    VkDescriptorPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = size;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    VkResult poolResult = vkCreateDescriptorPool(dev, &poolInfo, nullptr, &sharcQueryDescPool_);
    if (poolResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "SHARC query descriptor pool failed result=%d",
                            static_cast<int>(poolResult));
        g_crashRing.record("SHARC:queryPipeline:create:poolFailed");
        return;
    }

    std::vector<VkDescriptorSetLayout> layouts(size, sharcQueryDescSetLayout_);
    VkDescriptorSetAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = sharcQueryDescPool_;
    allocInfo.descriptorSetCount = size;
    allocInfo.pSetLayouts = layouts.data();
    sharcQueryDescSets_.resize(size);
    VkResult allocResult = vkAllocateDescriptorSets(dev, &allocInfo, sharcQueryDescSets_.data());
    if (allocResult != VK_SUCCESS) {
        RadianceLogger::log("RayTracing", "ERROR", "SHARC query descriptor alloc failed result=%d",
                            static_cast<int>(allocResult));
        g_crashRing.record("SHARC:queryPipeline:create:allocFailed");
        return;
    }

    RadianceLogger::log("RayTracing", "INFO", "SHARC query pipeline create done ok=1 sets=%u", size);
    g_crashRing.record("SHARC:queryPipeline:create:done");
#endif
}

RayTracingModuleContext::RayTracingModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                                 std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                                 std::shared_ptr<RayTracingModule> rayTracingModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      rayTracingModule(rayTracingModule),
      rayTracingDescriptorTable(rayTracingModule->rayTracingDescriptorTables_[frameworkContext->frameIndex]),
      directLightUpstreamDescriptorTable(
          frameworkContext->frameIndex < rayTracingModule->directLightUpstreamDescriptorTables_.size()
              ? rayTracingModule->directLightUpstreamDescriptorTables_[frameworkContext->frameIndex]
              : nullptr),
      sbt(rayTracingModule->sbts_[frameworkContext->frameIndex]),
      sharcUpdateSbt(frameworkContext->frameIndex < rayTracingModule->sharcUpdateSbts_.size()
                     ? rayTracingModule->sharcUpdateSbts_[frameworkContext->frameIndex] : nullptr),
      hdrNoisyOutputImage(rayTracingModule->hdrNoisyOutputImages_[frameworkContext->frameIndex]),
      diffuseAlbedoImage(rayTracingModule->diffuseAlbedoImages_[frameworkContext->frameIndex]),
      specularAlbedoImage(rayTracingModule->specularAlbedoImages_[frameworkContext->frameIndex]),
      normalRoughnessImage(rayTracingModule->normalRoughnessImages_[frameworkContext->frameIndex]),
      motionVectorImage(rayTracingModule->motionVectorImages_[frameworkContext->frameIndex]),
      linearDepthImage(rayTracingModule->linearDepthImages_[frameworkContext->frameIndex]),
      specularHitDepthImage(rayTracingModule->specularHitDepthImages_[frameworkContext->frameIndex]),
      firstHitDepthImage(rayTracingModule->firstHitDepthImages_[frameworkContext->frameIndex]),
      firstHitDiffuseDirectLightImage(
          rayTracingModule->firstHitDiffuseDirectLightImages_[frameworkContext->frameIndex]),
      firstHitDiffuseIndirectLightImage(
          rayTracingModule->firstHitDiffuseIndirectLightImages_[frameworkContext->frameIndex]),
      firstHitSpecularImage(rayTracingModule->firstHitSpecularImages_[frameworkContext->frameIndex]),
      firstHitClearImage(rayTracingModule->firstHitClearImages_[frameworkContext->frameIndex]),
      firstHitBaseEmissionImage(rayTracingModule->firstHitBaseEmissionImages_[frameworkContext->frameIndex]),
      directLightDepthImage(rayTracingModule->directLightDepthImages_[frameworkContext->frameIndex]),
      diffuseRayDirHitDistImage(rayTracingModule->diffuseRayDirHitDistImages_[frameworkContext->frameIndex]),
      specularRayDirHitDistImage(rayTracingModule->specularRayDirHitDistImages_[frameworkContext->frameIndex]),
      atmosphereContext(rayTracingModule->atmosphere_->contexts_[frameworkContext->frameIndex]),
      worldPrepareContext(rayTracingModule->worldPrepare_->contexts_[frameworkContext->frameIndex]) {}

void RayTracingModuleContext::render() {
    renderDiag("RT begin");
    g_crashRing.record("RT:begin");
    auto ctx0 = frameworkContext.lock();
    VkCommandBuffer profileCmd = (ctx0 && Renderer::gpuProfiler.isEnabled())
        ? ctx0->worldCommandBuffer->vkCommandBuffer()
        : VK_NULL_HANDLE;
    if (ctx0) {
        ctx0->worldCommandBuffer->beginLabel("RT:Atmosphere", 0.3f, 0.3f, 0.9f);
    }
    {
        ScopedGpuProfile profile(profileCmd, "RT.Atmosphere");
        renderDiag("RT atmosphere begin");
        g_crashRing.record("RT:atmosphere");
        atmosphereContext->render();
        renderDiag("RT atmosphere end");
    }
    if (ctx0) ctx0->worldCommandBuffer->endLabel();

    if (ctx0) ctx0->worldCommandBuffer->beginLabel("RT:BLAS/TLAS Build", 0.9f, 0.3f, 0.3f);
    {
        ScopedGpuProfile profile(profileCmd, "RT.BLAS_TLAS");
        renderDiag("RT worldPrepare begin");
        g_crashRing.record("RT:worldPrepare");
        worldPrepareContext->render();
        renderDiag("RT worldPrepare end tlas=%d instances=%u lights=%d",
                   (int)(worldPrepareContext->tlas != nullptr),
                   worldPrepareContext->prevTlasInstanceCount_,
                   worldPrepareContext->areaLightCount);
    }
    if (ctx0) ctx0->worldCommandBuffer->endLabel();

    if (worldPrepareContext->tlas == nullptr) {
        std::cout << "tlas is nullptr" << std::endl;
        return;
    }

    auto context = frameworkContext.lock();
    if (!context) return;
    auto framework = context->framework.lock();
    if (!framework) return;
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    auto module = rayTracingModule.lock();
    if (!module) return;

#ifdef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
    const uint32_t directLightBackend = effectiveDirectLightBackend();
    const bool upstreamDirectLightActive = kAllowUpstreamRtExecutorDispatch && directLightBackend == 1;
    const bool directLightPipelineActive = kAllowDirectLightScaffoldDispatch && directLightBackend == 2;
    bool directLightExternalActive = false;
    if (directLightPipelineActive) {
        const uint32_t frameIdx = context->frameIndex;
        const uint32_t width = hdrNoisyOutputImage->width();
        const uint32_t height = hdrNoisyOutputImage->height();
        auto ensureDirectLightImage = [&](std::vector<std::shared_ptr<vk::DeviceLocalImage>>& target) {
            if (frameIdx >= target.size()) return;
            auto& image = target[frameIdx];
            if (!image || image->width() != width || image->height() != height) {
                image = vk::DeviceLocalImage::create(
                    framework->device(), framework->vma(), false, width, height, 1,
                    VK_FORMAT_R32G32B32A32_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            }
        };

        ensureDirectLightImage(module->directLightPrimarySurfaceImages_);
        ensureDirectLightImage(module->directLightReservoirPingImages_);
        ensureDirectLightImage(module->directLightReservoirPongImages_);
        ensureDirectLightImage(module->directLightOutputImages_);
        if (frameIdx < module->directLightCounterBuffers_.size()
            && !module->directLightCounterBuffers_[frameIdx]) {
            module->directLightCounterBuffers_[frameIdx] = vk::HostVisibleBuffer::create(
                framework->vma(), framework->device(), kDirectLightCounterCount * sizeof(uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        }
        module->refreshUpstreamDirectLightRuntime(frameIdx);
        // Backend 1 is reserved for upstream parity. The old compute scaffold is
        // diagnostics-only and must not replace legacy direct lighting; doing so
        // drops the real RT sun/shadow path and makes visual comparisons invalid.
        directLightExternalActive = false && directLightBackend == 1
            && module->directLightInitialPipeline_ != VK_NULL_HANDLE
            && frameIdx < module->directLightInitialDescSets_.size()
            && frameIdx < module->directLightPrimarySurfaceImages_.size()
            && frameIdx < module->directLightReservoirPingImages_.size()
            && frameIdx < module->directLightOutputImages_.size()
            && frameIdx < module->directLightCounterBuffers_.size()
            && module->directLightPrimarySurfaceImages_[frameIdx]
            && module->directLightReservoirPingImages_[frameIdx]
            && module->directLightOutputImages_[frameIdx]
            && module->directLightCounterBuffers_[frameIdx]
            && firstHitDiffuseDirectLightImage != nullptr
            && frameIdx < module->positionViewSpaceImages_.size()
            && module->positionViewSpaceImages_[frameIdx] != nullptr
            && diffuseAlbedoImage != nullptr
            && worldPrepareContext->tlas != nullptr
            && worldPrepareContext->areaLightBuffer != nullptr
            && module->tileLightBuffer_ != nullptr;
    }
#else
    constexpr uint32_t directLightBackend = 0;
    constexpr bool upstreamDirectLightActive = false;
    constexpr bool directLightPipelineActive = false;
    constexpr bool directLightExternalActive = false;
#endif

    renderDiag("RT descriptors begin");
    g_crashRing.record("RT:descriptors");
    rayTracingDescriptorTable->bindAS(worldPrepareContext->tlas, 1, 0);

    if (atmosphereContext && atmosphereContext->atmCubeMapImage && module->atmosphere_) {
        g_crashRing.record("RT:skyDescriptors");
        rayTracingDescriptorTable->bindSamplerImageForShader(
            module->atmosphere_->atmLUTImageSampler_, module->atmosphere_->atmLUTImage_, 0, 1);
        if (context->frameIndex < module->atmosphere_->atmCubeMapImageSamplers_.size()) {
            rayTracingDescriptorTable->bindSamplerImageForShader(
                module->atmosphere_->atmCubeMapImageSamplers_[context->frameIndex],
                atmosphereContext->atmCubeMapImage, 0, 2, 7);
        }
    }

    auto buffers = Renderer::instance().buffers();
    auto worldBuffer = buffers->worldUniformBuffer();

    // Batch all per-frame buffer bindings into a single vkUpdateDescriptorSets call
    using BB = vk::DescriptorTable::BufferBinding;
    std::vector<BB> bufferBindings = {
        {worldPrepareContext->blasOffsetsBuffer,     1,  1},
        {worldPrepareContext->vertexBufferAddr,      1,  2},
        {worldPrepareContext->indexBufferAddr,        1,  3},
        {worldPrepareContext->lastVertexBufferAddr,  1,  4},
        {worldPrepareContext->lastIndexBufferAddr,   1,  5},
        {worldPrepareContext->lastObjToWorldMat,     1,  6},
        {buffers->textureMappingBuffer(),            1,  7},
        {worldPrepareContext->areaLightBuffer,       1,  8},
        {worldBuffer,                                2,  0},
        {buffers->lastWorldUniformBuffer(),          2,  1},
        {buffers->skyUniformBuffer(),                2,  2},
    };
    if (module->tileLightBuffer_) {
        bufferBindings.push_back({module->tileLightBuffer_, 1, 9});
    }
    // Per-section biome colors for shader-side tinting
    if (worldPrepareContext->biomeColorBuffer) {
        bufferBindings.push_back({worldPrepareContext->biomeColorBuffer, 1, 10});
    }
    // Always bind Material Class Mapping buffer (may be real data or dummy)
    auto mcBuffer = buffers->materialClassMappingBuffer();
    if (mcBuffer) {
        bufferBindings.push_back({mcBuffer, 1, 11});
    }
    // SpriteRegistry SSBO for texture array metadata
    auto spriteRegBuffer = Renderer::textureSystem.registry().getBuffer();
    if (spriteRegBuffer) {
        bufferBindings.push_back({spriteRegBuffer, 1, 13});
    }
    // Blue noise buffers (already uploaded at init time via queueImportantWorldUpload)
    if (module->blueNoise_) {
        bufferBindings.push_back({module->blueNoise_->sobolBuffer(),      2, 9});
        bufferBindings.push_back({module->blueNoise_->scramblingBuffer(), 2, 10});
    }
    rayTracingDescriptorTable->bindBufferBatch(bufferBindings);

    // Batch per-frame image rebindings (ReSTIR reservoirs, bounce reservoirs)
    using IB = vk::DescriptorTable::ImageBinding;
    std::vector<IB> frameImageBindings;

    // ReSTIR DI: reservoir image binding
    // Binding 14 = CHS write (temporal output)
    // Binding 15 = CHS read (previous frame source)
    if (module->reservoirImages_[0]) {
        bool spatialEnabled = Renderer::options.restirEnabled && Renderer::options.restirSpatialEnabled && module->spatialPipeline_ != VK_NULL_HANDLE;
        if (spatialEnabled) {
            // Spatial path: CHS writes [0], spatial transforms [0]->[1], next CHS reads [1]
            // No self-aliasing: CHS reads [1] and writes [0]
            frameImageBindings.push_back({module->reservoirImages_[0], VK_IMAGE_LAYOUT_GENERAL, 3, 14});
            frameImageBindings.push_back({module->reservoirImages_[1], VK_IMAGE_LAYOUT_GENERAL, 3, 15});
        } else {
            // No spatial: ping-pong to prevent read-write race on same image
            uint32_t frameIdx = context->frameIndex;
            int writeIdx = frameIdx & 1;
            int readIdx  = 1 - writeIdx;
            frameImageBindings.push_back({module->reservoirImages_[writeIdx], VK_IMAGE_LAYOUT_GENERAL, 3, 14});
            frameImageBindings.push_back({module->reservoirImages_[readIdx],  VK_IMAGE_LAYOUT_GENERAL, 3, 15});
        }
    }

    // Bounce ReSTIR DI: per-bounce reservoir images (bindings 18-20)
    for (int b = 0; b < 3; b++) {
        if (module->bounceReservoirImages_[b]) {
            frameImageBindings.push_back(
                {module->bounceReservoirImages_[b], VK_IMAGE_LAYOUT_GENERAL, 3, static_cast<uint32_t>(18 + b)});
        }
    }
#ifdef MCVR_ENABLE_SHARC_QUERY_PASS
    if (context->frameIndex < module->sharcCandidatePosHitTImages_.size()
        && module->sharcCandidatePosHitTImages_[context->frameIndex]) {
        frameImageBindings.push_back({module->sharcCandidatePosHitTImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL, 3, 34});
        frameImageBindings.push_back({module->sharcCandidateNormalRoughnessImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL, 3, 35});
        frameImageBindings.push_back({module->sharcCandidateThroughputImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL, 3, 36});
        frameImageBindings.push_back({module->sharcCandidatePrefixRadianceFlagsImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL, 3, 37});
    }
#endif
    rayTracingDescriptorTable->bindImages(frameImageBindings);

    // Bind block sprite texture arrays (set 0, bindings 3-5)
    auto& texSystem = Renderer::textureSystem;
    const uint64_t textureGeneration = texSystem.generation();
    std::string textureDescriptorLabel = "RT:TexturePublishAndDescriptors gen=" +
        std::to_string(textureGeneration) + " frame=" + std::to_string(context->frameIndex);
    worldCommandBuffer->beginLabel(textureDescriptorLabel.c_str(), 0.95f, 0.35f, 0.1f);
    ScopedGpuProfile textureProfile(profileCmd, "RT.TexturePublish");
    // Flush any pending texture array uploads (staged by animation tick, executed here on render thread)
    if (texSystem.isFinalized()) {
        renderDiag("RT textureFlush begin");
        auto vma = framework->vma();
        auto device = framework->device();
        {
            auto _ft0 = std::chrono::steady_clock::now();
            texSystem.flushPendingUploads(vma, device, worldCommandBuffer, framework->gc());
            FrameTiming_addTexFlush(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - _ft0).count());
        }
        renderDiag("RT textureFlush end");
    }
    // Block albedo array
    auto& texArrayMgr = texSystem.arrayManager();
    const auto* albedoInfo = texArrayMgr.getArray(texSystem.blockAlbedoArrayId());
    const bool hasAlbedo = albedoInfo && albedoInfo->image && albedoInfo->sampler;
    const auto* specInfo = texArrayMgr.getArray(texSystem.blockSpecularArrayId());
    const bool hasSpec = specInfo && specInfo->image && specInfo->sampler;
    const auto* normInfo = texArrayMgr.getArray(texSystem.blockNormalArrayId());
    const bool hasNorm = normInfo && normInfo->image && normInfo->sampler;

    const VkImageView albedoView = hasAlbedo ? albedoInfo->image->vkImageView() : VK_NULL_HANDLE;
    const VkImageView specView = hasSpec ? specInfo->image->vkImageView() : VK_NULL_HANDLE;
    const VkImageView normView = hasNorm ? normInfo->image->vkImageView() : VK_NULL_HANDLE;

    auto bindBlockTextureArrays = [&](const std::shared_ptr<vk::DescriptorTable>& table) {
        if (!table) return;
        if (hasAlbedo) {
            table->bindSamplerImage(albedoInfo->sampler, albedoInfo->image,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 3, 0);
        }
        if (hasSpec) {
            table->bindSamplerImage(specInfo->sampler, specInfo->image,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 4, 0);
        }
        if (hasNorm) {
            table->bindSamplerImage(normInfo->sampler, normInfo->image,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 5, 0);
        }
    };

    const bool textureDescriptorGenerationChanged =
        textureGeneration != module->lastTextureDescriptorRefreshGeneration_ ||
        albedoView != module->lastAlbedoTextureView_ ||
        specView != module->lastSpecularTextureView_ ||
        normView != module->lastNormalTextureView_;

    if (textureDescriptorGenerationChanged) {
        for (auto& table : module->rayTracingDescriptorTables_) {
            bindBlockTextureArrays(table);
        }
        RadianceLogger::log(
            "TextureDescriptors", "INFO",
            "refresh-all gen=%llu frame=%u tables=%zu albedoId=%u image=0x%llx view=0x%llx sampler=0x%llx "
            "specId=%u image=0x%llx view=0x%llx sampler=0x%llx normId=%u image=0x%llx view=0x%llx sampler=0x%llx",
            static_cast<unsigned long long>(textureGeneration),
            context->frameIndex,
            module->rayTracingDescriptorTables_.size(),
            texSystem.blockAlbedoArrayId(),
            hasAlbedo ? static_cast<unsigned long long>(vk::DebugUtils::objectHandle(albedoInfo->image->vkImage())) : 0ull,
            static_cast<unsigned long long>(vk::DebugUtils::objectHandle(albedoView)),
            hasAlbedo ? static_cast<unsigned long long>(vk::DebugUtils::objectHandle(albedoInfo->sampler->vkSamper())) : 0ull,
            texSystem.blockSpecularArrayId(),
            hasSpec ? static_cast<unsigned long long>(vk::DebugUtils::objectHandle(specInfo->image->vkImage())) : 0ull,
            static_cast<unsigned long long>(vk::DebugUtils::objectHandle(specView)),
            hasSpec ? static_cast<unsigned long long>(vk::DebugUtils::objectHandle(specInfo->sampler->vkSamper())) : 0ull,
            texSystem.blockNormalArrayId(),
            hasNorm ? static_cast<unsigned long long>(vk::DebugUtils::objectHandle(normInfo->image->vkImage())) : 0ull,
            static_cast<unsigned long long>(vk::DebugUtils::objectHandle(normView)),
            hasNorm ? static_cast<unsigned long long>(vk::DebugUtils::objectHandle(normInfo->sampler->vkSamper())) : 0ull);
        module->lastTextureDescriptorRefreshGeneration_ = textureGeneration;
        module->lastAlbedoTextureView_ = albedoView;
        module->lastSpecularTextureView_ = specView;
        module->lastNormalTextureView_ = normView;
    } else {
        bindBlockTextureArrays(rayTracingDescriptorTable);
    }
    textureProfile.close();
    worldCommandBuffer->endLabel();
    renderDiag("RT descriptors end");

    bool accumulating = Renderer::options.offlineState == 2;
    bool sharcQueuesIdle = true;

#ifdef MCVR_ENABLE_SHARC
    // Lazy SHARC init: create buffers + pipelines if enabled at runtime but not yet allocated.
    // Must happen BEFORE push constant population so SHARC BDAs are available on the same frame.
#ifdef MCVR_ENABLE_SHARC_BUFFERS
    if (Renderer::options.sharcEnabled && !module->sharcHashEntries_) {
        module->sharcCapacity_ = effectiveSharcCapacity();
        module->initSharcBuffers();
#ifdef MCVR_ENABLE_SHARC_UPDATE_PASS
        module->initSharcUpdatePipeline();
#endif
#ifdef MCVR_ENABLE_SHARC_RESOLVE_PASS
        module->initSharcResolvePipeline();
#endif
#ifdef MCVR_ENABLE_SHARC_QUERY_PASS
        module->initSharcQueryPipeline();
#endif
        // Refresh ALL contexts' stale SBT pointers — were nullptr at construction time
        for (size_t ci = 0; ci < module->contexts_.size(); ci++) {
            auto rtCtx = std::static_pointer_cast<RayTracingModuleContext>(module->contexts_[ci]);
            if (ci < module->sharcUpdateSbts_.size()) {
                rtCtx->sharcUpdateSbt = module->sharcUpdateSbts_[ci];
            }
        }
    }
#endif
#endif

#if defined(MCVR_ENABLE_SHARC) && defined(MCVR_ENABLE_SHARC_BUFFERS)
    if (Renderer::options.sharcEnabled && !accumulating) {
        auto chunkScheduler = Renderer::instance().world()->chunks()->chunkBuildScheduler();
        if (chunkScheduler) {
            auto stats = chunkScheduler->stats();
            sharcQueuesIdle = stats.inputQueue == 0 && stats.completedQueue == 0 && stats.inFlight == 0;
        }
        const bool hasWorldGeometry = worldPrepareContext->prevBlasSnapshot_.chunkInstanceCount > 0;
        if (sharcQueuesIdle && hasWorldGeometry) {
            module->sharcStreamStableFrames_++;
        } else {
            module->sharcStreamStableFrames_ = 0;
        }
        module->sharcDispatchReady_ =
            module->sharcStreamStableFrames_ >= kSharcDispatchStableFrameThreshold;
    } else {
        module->sharcStreamStableFrames_ = 0;
        module->sharcDispatchReady_ = false;
    }
#endif

    uint32_t sharcMainTraceWarmupFrames = 0;
    bool sharcMainTraceQueryActive = false;
#if defined(MCVR_ENABLE_SHARC) && defined(MCVR_ENABLE_SHARC_MAIN_TRACE_QUERY)
    sharcMainTraceWarmupFrames = effectiveSharcMainTraceWarmupFrames();
    sharcMainTraceQueryActive =
        Renderer::options.sharcEnabled && !accumulating && module->sharcHashEntries_
        && module->sharcBuffersInitialized_ && module->sharcDispatchReady_
        && module->sharcFrameIndex_ >= sharcMainTraceWarmupFrames;
#endif

    int sharcQueryMode = 0;
    bool sharcQueryPassActive = false;
#if defined(MCVR_ENABLE_SHARC) && defined(MCVR_ENABLE_SHARC_QUERY_PASS)
    sharcMainTraceWarmupFrames = effectiveSharcMainTraceWarmupFrames();
    sharcQueryMode = effectiveSharcQueryMode();
    sharcQueryPassActive =
        Renderer::options.sharcEnabled && !accumulating && sharcQueryMode > 0
        && module->sharcHashEntries_ && module->sharcResolved_
        && module->sharcBuffersInitialized_ && module->sharcDispatchReady_
        && module->sharcFrameIndex_ >= sharcMainTraceWarmupFrames
        && module->sharcQueryPipeline_ != VK_NULL_HANDLE
        && context->frameIndex < module->sharcQueryDescSets_.size()
        && module->sharcCandidatePosHitTImages_[context->frameIndex]
        && module->sharcCandidatePrefixRadianceFlagsImages_[context->frameIndex]
        && module->sharcQueryCounterBuffers_[context->frameIndex];
#endif
    bool sharcCacheConsumerRequested = false;
#if defined(MCVR_ENABLE_SHARC) && defined(MCVR_ENABLE_SHARC_MAIN_TRACE_QUERY)
    sharcCacheConsumerRequested = Renderer::options.sharcEnabled && !accumulating
        && module->sharcHashEntries_ && module->sharcResolved_;
#endif
#if defined(MCVR_ENABLE_SHARC) && defined(MCVR_ENABLE_SHARC_QUERY_PASS)
    sharcCacheConsumerRequested = sharcCacheConsumerRequested
        || (Renderer::options.sharcEnabled && !accumulating && sharcQueryMode > 0
            && module->sharcQueryPipeline_ != VK_NULL_HANDLE);
#endif
    {
        static bool lastLoggedSharcMainTraceQueryActive = false;
        static bool lastLoggedSharcQueryPassActive = false;
        static int lastLoggedSharcQueryMode = -1;
        static bool loggedSharcMainTraceQueryState = false;
        if (!loggedSharcMainTraceQueryState
            || lastLoggedSharcMainTraceQueryActive != sharcMainTraceQueryActive
            || lastLoggedSharcQueryPassActive != sharcQueryPassActive
            || lastLoggedSharcQueryMode != sharcQueryMode) {
            loggedSharcMainTraceQueryState = true;
            lastLoggedSharcMainTraceQueryActive = sharcMainTraceQueryActive;
            lastLoggedSharcQueryPassActive = sharcQueryPassActive;
            lastLoggedSharcQueryMode = sharcQueryMode;
            renderDiag("SHARC query state mainActive=%d passActive=%d passMode=%d legacyMode=%d frame=%u warmup=%u ready=%d initialized=%d",
                       sharcMainTraceQueryActive ? 1 : 0,
                       sharcQueryPassActive ? 1 : 0,
                       sharcQueryMode,
                       MCVR_SHARC_MAIN_TRACE_QUERY_MODE,
                       module->sharcFrameIndex_,
                       sharcMainTraceWarmupFrames,
                       module->sharcDispatchReady_ ? 1 : 0,
                       module->sharcBuffersInitialized_ ? 1 : 0);
        }
    }

    RayTracingPushConstant pushConstant{};
    pushConstant.numRayBounces = accumulating
        ? static_cast<int>(Renderer::options.offlineBounces)
        : static_cast<int>(Renderer::options.rayBounces);
    pushConstant.flags = (Renderer::options.simplifiedIndirect ? 1 : 0)
                       | (Renderer::options.areaLightsEnabled ? 2 : 0)
                       | (Renderer::options.restirEnabled ? 4 : 0)
                       | (Renderer::options.restirSimplifiedBRDF ? 8 : 0)
                       | (Renderer::options.restirBounceEnabled ? 16 : 0)
#if defined(MCVR_ENABLE_SHARC) && defined(MCVR_ENABLE_SHARC_MAIN_TRACE_QUERY)
                       | (sharcMainTraceQueryActive ? 32 : 0)
#elif defined(MCVR_ENABLE_SHARC) && defined(MCVR_ENABLE_SHARC_QUERY_PASS)
                       | (sharcQueryPassActive ? 32 : 0)
#endif
                       | (Renderer::options.noiseLOD ? 64 : 0)
                       | (Renderer::options.multiScatterGGX ? 128 : 0)
                       | (Renderer::options.eonDiffuse ? 256 : 0)
                       | (Renderer::options.beerLawShadows ? 512 : 0)
                       | (Renderer::options.noEmissionClamp ? 1024 : 0)
                       | (Renderer::options.physicalSunDisk ? 2048 : 0)
                       | (Renderer::options.noHandAmbient ? 4096 : 0)
                       | ((framework->device()->hasSER() && Renderer::options.serEnabled) ? 8192 : 0)
                       | ((framework->device()->hasSER() && Renderer::options.serEnabled && Renderer::options.serHintsEnabled) ? 16384 : 0)
                       | (Renderer::options.entityNormalsEnabled ? 32768 : 0);
    pushConstant.areaLightCount = worldPrepareContext->areaLightCount;
    pushConstant.shadowSoftness = Renderer::options.shadowSoftness;
    pushConstant.risCandidates = Renderer::options.restirCandidates;
    pushConstant.temporalMClamp = Renderer::options.restirTemporalMClamp;
    pushConstant.wClamp = Renderer::options.restirWClamp;
    // Fixed pre-exposure compresses HDR radiance into fp16-friendly range for DLSS-RR.
    // Must be CONSTANT across frames — varying pre-exposure contaminates DLSS-RR's temporal
    // history (accumulated at different scales) and causes visible brightness oscillation.
    // 0.1 maps Minecraft's typical luminance range into fp16's sweet spot:
    //   Sun/lava (1000) → 100, torch (500) → 50, dark cave (0.01) → 0.001
    // DLSS-RR undoes it via InExposureScale = 1/0.1 = 10.
    // Only apply when DLSS-RR is active (denoiserMode == 1).
    pushConstant.preExposure = (Renderer::options.denoiserMode == 1) ? 0.1f : 1.0f;

    // Shader displacement is temporarily forced off at runtime to isolate NVIDIA DMA page faults
    // from the full SHARC refactor path without changing the user's saved option.
    const bool displacementActive = Renderer::options.pomEnabled && shaderDisplacementRuntimeAllowed();
    pushConstant.pomHeightScale  = displacementActive ? Renderer::options.pomHeightScale : 0.0f;
    pushConstant.pomSteps        = Renderer::options.pomSteps;
    pushConstant.pomRefinement   = Renderer::options.pomRefinement;
    pushConstant.pomFadeDistance = Renderer::options.pomFadeDistance;

    // Color expansion
    pushConstant.colorExpansion = Renderer::options.colorExpansion;
    pushConstant.blueNoiseFrame = context->frameIndex;
    pushConstant.rtDebugFlags = Renderer::options.rtDebugFlags
        | (directLightExternalActive ? kRtDebugDisableDirectLighting : 0u);
    pushConstant.handInstanceCount = worldPrepareContext->handInstanceCount;
    pushConstant.sharcQueryMode = sharcQueryPassActive ? sharcQueryMode : 0;
    pushConstant.sharcQueryReserved = 0;

    // Structured logging: push constants (every ~1 second)
    if (RadianceLogger::isEnabled()) {
        static uint64_t lastPCLog = 0;
        uint64_t curFrame = g_crashRing.frameCount();
        if (curFrame - lastPCLog >= 60) {
            lastPCLog = curFrame;
            RadianceLogger::log("RayTracing", "INFO",
                "pushConst: bounces=%d flags=0x%x rtDebug=0x%x handInst=%u lights=%d shadowSoft=%.2f sharcQuery=%d sharcQueryPass=%d sharcQueryMode=%d sharcFrame=%u sharcWarmup=%u sharcMode=%d displacementEnabled=%d displacementRequested=%d displacementQuality=%u displacementDepth=%.4f displacementSteps=%d displacementRefinement=%d displacementFade=%.0f",
                pushConstant.numRayBounces, pushConstant.flags, pushConstant.rtDebugFlags,
                pushConstant.handInstanceCount, pushConstant.areaLightCount,
                pushConstant.shadowSoftness, sharcMainTraceQueryActive ? 1 : 0,
                sharcQueryPassActive ? 1 : 0, pushConstant.sharcQueryMode,
                module->sharcFrameIndex_, sharcMainTraceWarmupFrames,
                MCVR_SHARC_MAIN_TRACE_QUERY_MODE, displacementActive ? 1 : 0,
                Renderer::options.pomEnabled ? 1 : 0,
                Renderer::options.displacementQuality, pushConstant.pomHeightScale,
                pushConstant.pomSteps, pushConstant.pomRefinement, pushConstant.pomFadeDistance);
        }
    }

#ifdef MCVR_ENABLE_SHARC
    // SHARC radiance cache
#ifdef MCVR_ENABLE_SHARC_BUFFERS
    if (Renderer::options.sharcEnabled && module->sharcHashEntries_) {
        auto worldUBO = static_cast<vk::Data::WorldUBO *>(buffers->worldUniformBuffer()->mappedPtr());
        pushConstant.sharcHashEntries = module->sharcHashEntries_->bufferAddress();
        pushConstant.sharcAccumulation = module->sharcAccumulation_->bufferAddress();
        pushConstant.sharcResolved = module->sharcResolved_->bufferAddress();
        pushConstant.sharcCameraX = static_cast<float>(worldUBO->cameraPos.x);
        pushConstant.sharcCameraY = static_cast<float>(worldUBO->cameraPos.y);
        pushConstant.sharcCameraZ = static_cast<float>(worldUBO->cameraPos.z);
        pushConstant.sharcSceneScale = Renderer::options.sharcSceneScale;
        pushConstant.sharcCapacity = module->sharcCapacity_;
        pushConstant.sharcRadianceScale = 1e3f;
        pushConstant.sharcFrameIndex = module->sharcFrameIndex_;
        pushConstant.sharcRoughnessThreshold = Renderer::options.sharcRoughnessThreshold;
        pushConstant.sharcUpdateBlockSize = effectiveSharcUpdateBlockSize();
        pushConstant.sharcUpdateBounces = effectiveSharcUpdateBounces();
        pushConstant.sharcQueryMode = sharcQueryPassActive ? sharcQueryMode : 0;
        pushConstant.sharcQueryReserved = 0;
    }
#endif
#endif

    // Offline accumulation
    // offlineDenoised: 0=Raw Fast (RR on), 1=Raw Accurate (RR off), 2=Denoised (epoch-based DLSS-RR)
    // Bit 1 (disableRR): auto-set by preset — Raw Accurate forces RR off
    bool disableRR = Renderer::options.offlineDisableRR
                     || Renderer::options.offlineDenoised == 1;
    pushConstant.offlineFlags = (accumulating ? 1 : 0)
                              | (disableRR ? 2 : 0)
                              | (Renderer::options.offlineDisableClamp ? 4 : 0)
                              | (Renderer::options.offlineNativeResActive ? 8 : 0);
    pushConstant.accumFrameCount = static_cast<int>(Renderer::accumFrameCount);
    // DOF: active in both FREE (preview) and ACCUMULATING modes
    bool dofEnabled = (accumulating || Renderer::options.offlineState == 1)
                      && Renderer::options.offlineAperture > 0.0f;
    float effectiveAperture = Renderer::options.offlineAperture * Renderer::options.dofStrength;
    pushConstant.aperture = dofEnabled ? effectiveAperture : 0.0f;
    pushConstant.focalDistance = Renderer::options.offlineFocalDistance;

    // Pre-exposure locked to 1.0 during accumulation (all presets).
    // Gives full fp16 precision for the Welford accumulator input (DLSS output at
    // scene-referred scale). DLSS-RR exposure params are neutral (§3.7: not supported),
    // so output stays in the same pre-exposed space as input — consistent with histogram.
    // All modes disable temporal reuse (each frame is independent).
    if (accumulating) {
        pushConstant.preExposure = 1.0f;
        pushConstant.temporalMClamp = 0;
    }

    // Material SSBO BDA: pass buffer address via push constant (avoids descriptor lookup overhead)
    pushConstant.materialClassAddr = mcBuffer ? mcBuffer->bufferAddress() : 0;

    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), rayTracingDescriptorTable->vkPipelineLayout(),
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                           VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                           VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                       0, sizeof(RayTracingPushConstant), &pushConstant);

    auto chooseSrc = [](VkImageLayout oldLayout, VkPipelineStageFlags2 fallbackStage, VkAccessFlags2 fallbackAccess,
                        VkPipelineStageFlags2 &outStage, VkAccessFlags2 &outAccess) {
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
            outStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            outAccess = 0;
        } else {
            outStage = fallbackStage;
            outAccess = fallbackAccess;
        }
    };

    std::vector<vk::CommandBuffer::ImageMemoryBarrier> barriers;
    auto addBarrier = [&](const std::shared_ptr<vk::DeviceLocalImage> &img, VkImageLayout newLayout) {
        if (!img) return;
        VkPipelineStageFlags2 srcStage = 0;
        VkAccessFlags2 srcAccess = 0;
        chooseSrc(img->imageLayout(), VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, srcStage, srcAccess);
        barriers.push_back({
            .srcStageMask = srcStage,
            .srcAccessMask = srcAccess,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .oldLayout = img->imageLayout(),
            .newLayout = newLayout,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = img,
            .subresourceRange = vk::wholeColorSubresourceRange,
        });
        img->imageLayout() = newLayout;
    };

    addBarrier(hdrNoisyOutputImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(diffuseAlbedoImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(specularAlbedoImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(normalRoughnessImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(motionVectorImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(linearDepthImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(specularHitDepthImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitDepthImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitDiffuseDirectLightImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitDiffuseIndirectLightImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitSpecularImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitClearImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitBaseEmissionImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(directLightDepthImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(diffuseRayDirHitDistImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(specularRayDirHitDistImage, VK_IMAGE_LAYOUT_GENERAL);
#ifdef MCVR_ENABLE_SHARC_QUERY_PASS
    if (context->frameIndex < module->sharcCandidatePosHitTImages_.size()) {
        addBarrier(module->sharcCandidatePosHitTImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
        addBarrier(module->sharcCandidateNormalRoughnessImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
        addBarrier(module->sharcCandidateThroughputImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
        addBarrier(module->sharcCandidatePrefixRadianceFlagsImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    }
#endif
#ifdef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
    if (directLightPipelineActive && context->frameIndex < module->directLightPrimarySurfaceImages_.size()) {
        addBarrier(module->directLightPrimarySurfaceImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
        addBarrier(module->directLightReservoirPingImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
        addBarrier(module->directLightReservoirPongImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
        addBarrier(module->directLightOutputImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    }
#endif
    addBarrier(module->reservoirImages_[0], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->reservoirImages_[1], VK_IMAGE_LAYOUT_GENERAL);
    for (int b = 0; b < 3; b++) {
        addBarrier(module->bounceReservoirImages_[b], VK_IMAGE_LAYOUT_GENERAL);
    }
    addBarrier(atmosphereContext->atmCubeMapImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    if (!barriers.empty()) {
        renderDiag("RT imageBarriers begin count=%u", static_cast<unsigned>(barriers.size()));
        g_crashRing.record("RT:imageBarriers");
        ScopedGpuProfile profile(profileCmd, "RT.ImageBarriers");
        worldCommandBuffer->barriersBufferImage({}, barriers);
        renderDiag("RT imageBarriers end");
    }

#ifdef MCVR_ENABLE_SHARC_QUERY_PASS
    if (context->frameIndex < module->sharcQueryCounterBuffers_.size()
        && module->sharcQueryCounterBuffers_[context->frameIndex]) {
        auto counterBuffer = module->sharcQueryCounterBuffers_[context->frameIndex];
        counterBuffer->downloadFromBuffer();
        if (counterBuffer->mappedPtr()) {
            std::memcpy(module->sharcQueryLastCounters_, counterBuffer->mappedPtr(),
                        kSharcQueryCounterCount * sizeof(uint32_t));
        }
        vkCmdFillBuffer(worldCommandBuffer->vkCommandBuffer(), counterBuffer->vkBuffer(), 0,
                        kSharcQueryCounterCount * sizeof(uint32_t), 0);
        VkMemoryBarrier counterClearBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        counterClearBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        counterClearBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(worldCommandBuffer->vkCommandBuffer(),
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &counterClearBarrier, 0, nullptr, 0, nullptr);
    }
#endif

    // Light clustering compute pass — DISABLED: contribution-sorted global list replaces tile clustering.
    // Tile buffer stays allocated (descriptor layout unchanged); CHS reads tileCount=0 → global fallback.
#ifdef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
    if (directLightPipelineActive && context->frameIndex < module->directLightCounterBuffers_.size()
        && module->directLightCounterBuffers_[context->frameIndex]) {
        auto counterBuffer = module->directLightCounterBuffers_[context->frameIndex];
        counterBuffer->downloadFromBuffer();
        if (counterBuffer->mappedPtr()) {
            std::memcpy(module->directLightLastCounters_, counterBuffer->mappedPtr(),
                        kDirectLightCounterCount * sizeof(uint32_t));
        }
        vkCmdFillBuffer(worldCommandBuffer->vkCommandBuffer(), counterBuffer->vkBuffer(), 0,
                        kDirectLightCounterCount * sizeof(uint32_t), 0);
        VkMemoryBarrier counterClearBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        counterClearBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        counterClearBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(worldCommandBuffer->vkCommandBuffer(),
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &counterClearBarrier, 0, nullptr, 0, nullptr);
    } else if (!directLightPipelineActive) {
        std::fill_n(module->directLightLastCounters_, kDirectLightCounterCount, 0);
    }
#endif

    if (false && Renderer::options.restirEnabled && Renderer::options.areaLightsEnabled
        && module->clusterPipeline_ != VK_NULL_HANDLE && module->tileLightBuffer_
        && worldPrepareContext->areaLightCount > 0) {
        VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();
        uint32_t frameIdx = context->frameIndex;

        int w = hdrNoisyOutputImage->width();
        int h = hdrNoisyOutputImage->height();
        int tilesX = (w + RayTracingModule::TILE_SIZE - 1) / RayTracingModule::TILE_SIZE;
        int tilesY = (h + RayTracingModule::TILE_SIZE - 1) / RayTracingModule::TILE_SIZE;
        int totalTiles = tilesX * tilesY;

        // Resize tile buffer if needed
        size_t requiredSize = totalTiles * (1 + RayTracingModule::MAX_LIGHTS_PER_TILE) * sizeof(uint32_t);
        if (module->tileLightBuffer_->size() < requiredSize) {
            module->tileLightBuffer_ = vk::DeviceLocalBuffer::create(
                framework->vma(), framework->device(), requiredSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            rayTracingDescriptorTable->bindBuffer(module->tileLightBuffer_, 1, 9);
        }

        // Update cluster descriptor set bindings
        VkDescriptorBufferInfo lightBufInfo = {worldPrepareContext->areaLightBuffer->vkBuffer(), 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo tileBufInfo = {module->tileLightBuffer_->vkBuffer(), 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet writes[2] = {};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[0].dstSet = module->clusterDescSets_[frameIdx];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &lightBufInfo;
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[1].dstSet = module->clusterDescSets_[frameIdx];
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &tileBufInfo;
        vkUpdateDescriptorSets(framework->device()->vkDevice(), 2, writes, 0, nullptr);

        // Compute view-proj matrix for camera-relative positions (rotation only, no translation)
        auto worldUBO = static_cast<vk::Data::WorldUBO *>(buffers->worldUniformBuffer()->mappedPtr());
        glm::mat4 viewRot = glm::mat4(glm::mat3(worldUBO->cameraViewMat));
        glm::mat4 vpCameraRel = worldUBO->cameraProjMat * viewRot;

        struct {
            int32_t width, height, lightCount, maxPerTile;
            glm::mat4 vpCameraRel;
        } clusterPC = {w, h, worldPrepareContext->areaLightCount, RayTracingModule::MAX_LIGHTS_PER_TILE, vpCameraRel};

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->clusterPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->clusterPipelineLayout_,
            0, 1, &module->clusterDescSets_[frameIdx], 0, nullptr);
        vkCmdPushConstants(cmd, module->clusterPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(clusterPC), &clusterPC);
        vkCmdDispatch(cmd, totalTiles, 1, 1);

        // Barrier: compute writes → RT reads
        VkMemoryBarrier clusterBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        clusterBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        clusterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 1, &clusterBarrier, 0, nullptr, 0, nullptr);
    }

#ifdef MCVR_ENABLE_SHARC
#ifdef MCVR_ENABLE_SHARC_BUFFERS
    // Reset SHARC buffers when disabled so re-enable starts fresh (prevents stale cache artifacts)
    if (!Renderer::options.sharcEnabled && module->sharcBuffersInitialized_) {
        module->sharcBuffersInitialized_ = false;
        module->sharcFrameIndex_ = 0;
    }

    // Deferred SHARC resize: process pending capacity change at frame boundary.
    // Previous frame's SHARC dispatch used old buffers via BDA - drain GPU before freeing them.
    if (module->sharcResizePending_ && module->sharcHashEntries_) {
        uint32_t desiredCapacity = effectiveSharcCapacity();
        auto fw = module->framework_.lock();
        if (fw) {
            RadianceLogger::log("RayTracing", "INFO", "SHARC resize start capacity=%u", desiredCapacity);
            g_crashRing.record("SHARC:resize:start");
            vkDeviceWaitIdle(fw->device()->vkDevice());

            VkBufferUsageFlags sharcUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            module->sharcCapacity_ = desiredCapacity;
            module->sharcHashEntries_ = vk::DeviceLocalBuffer::create(
                fw->vma(), fw->device(), desiredCapacity * 8, sharcUsage);
            module->sharcAccumulation_ = vk::DeviceLocalBuffer::create(
                fw->vma(), fw->device(), desiredCapacity * 16, sharcUsage);
            module->sharcResolved_ = vk::DeviceLocalBuffer::create(
                fw->vma(), fw->device(), desiredCapacity * 16, sharcUsage);
            module->sharcBuffersInitialized_ = false;
            module->sharcFrameIndex_ = 0;
            module->sharcStreamStableFrames_ = 0;
            module->sharcDispatchReady_ = false;
            pushConstant.sharcHashEntries = module->sharcHashEntries_->bufferAddress();
            pushConstant.sharcAccumulation = module->sharcAccumulation_->bufferAddress();
            pushConstant.sharcResolved = module->sharcResolved_->bufferAddress();
            pushConstant.sharcCapacity = desiredCapacity;
            RadianceLogger::log("RayTracing", "INFO", "SHARC resize done ok=%d", module->sharcHashEntries_ ? 1 : 0);
            g_crashRing.record("SHARC:resize:done");
        }
        module->sharcResizePending_ = false;
    }

    // Check if SHARC capacity exponent changed - mark for deferred resize at next frame boundary.
    {
        uint32_t desiredCapacity = effectiveSharcCapacity();
        if (desiredCapacity != module->sharcCapacity_ && module->sharcHashEntries_) {
            module->sharcResizePending_ = true;
        }
    }

    const bool sharcBuffersReady = Renderer::options.sharcEnabled && !accumulating
        && sharcCacheConsumerRequested
        && module->sharcHashEntries_ && module->sharcAccumulation_ && module->sharcResolved_;
    if (sharcBuffersReady && !module->sharcDispatchReady_) {
        static uint64_t lastSharcGateLog = 0;
        uint64_t curFrame = g_crashRing.frameCount();
        if (curFrame - lastSharcGateLog >= 120) {
            lastSharcGateLog = curFrame;
            renderDiag("SHARC gated waiting for settled BLAS stream stableFrames=%u/%u queuesIdle=%d chunkInstances=%u requestedExp=%d effectiveExp=%d requestedBounces=%d effectiveBounces=%d",
                       module->sharcStreamStableFrames_,
                       kSharcDispatchStableFrameThreshold,
                       sharcQueuesIdle ? 1 : 0,
                       worldPrepareContext->prevBlasSnapshot_.chunkInstanceCount,
                       Renderer::options.sharcCapacityExponent,
                       effectiveSharcCapacityExponent(),
                       Renderer::options.sharcUpdateBounces,
                       effectiveSharcUpdateBounces());
        }
    }

    if (sharcBuffersReady && module->sharcDispatchReady_ && !module->sharcBuffersInitialized_) {
        worldCommandBuffer->beginLabel("RT:SHARC Init", 0.8f, 0.5f, 0.1f);
        ScopedGpuProfile sharcInitProfile(profileCmd, "RT.SHARCInit");
        VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();
        renderDiag("SHARC init fill begin capacity=%u", module->sharcCapacity_);
        g_crashRing.record("SHARC:initFill:start");
        vkCmdFillBuffer(cmd, module->sharcHashEntries_->vkBuffer(), 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, module->sharcAccumulation_->vkBuffer(), 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, module->sharcResolved_->vkBuffer(), 0, VK_WHOLE_SIZE, 0);

        VkMemoryBarrier fillBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        fillBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &fillBarrier, 0, nullptr, 0, nullptr);
        module->sharcBuffersInitialized_ = true;
        renderDiag("SHARC init fill end");
        g_crashRing.record("SHARC:initFill:done");
        sharcInitProfile.close();
        worldCommandBuffer->endLabel();
    }

#ifdef MCVR_ENABLE_SHARC_UPDATE_PASS
    if (sharcBuffersReady && module->sharcDispatchReady_ && module->sharcUpdatePipeline_ && sharcUpdateSbt) {
        worldCommandBuffer->beginLabel("RT:SHARC Update", 0.9f, 0.6f, 0.1f);
        ScopedGpuProfile sharcUpdateProfile(profileCmd, "RT.SHARCUpdate");
        VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();
        GpuDiag::checkpoint(cmd, GpuDiag::SHARC_UPDATE_RT);
        renderDiag("SHARC update begin downscale=%d block=%d bounces=%d requestedDownscale=%d requestedBlock=%d requestedBounces=%d",
                   effectiveSharcDownscale(), effectiveSharcUpdateBlockSize(), effectiveSharcUpdateBounces(),
                   Renderer::options.sharcDownscale, Renderer::options.sharcUpdateBlockSize,
                   Renderer::options.sharcUpdateBounces);
        g_crashRing.record("SHARC:updateDispatch:start");

        RayTracingPushConstant updatePC = pushConstant;
        updatePC.flags &= ~(4 | 16); // Clear ReSTIR and ReSTIR bounce; keep area lights.
        vkCmdPushConstants(cmd, rayTracingDescriptorTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                               VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                               VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                           0, sizeof(RayTracingPushConstant), &updatePC);

        int ds = effectiveSharcDownscale();
        worldCommandBuffer->bindDescriptorTable(rayTracingDescriptorTable, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
            ->bindRTPipeline(module->sharcUpdatePipeline_)
            ->raytracing(sharcUpdateSbt, hdrNoisyOutputImage->width() / ds, hdrNoisyOutputImage->height() / ds, 1);

        VkMemoryBarrier updateBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        updateBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        updateBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
#ifdef MCVR_ENABLE_SHARC_RESOLVE_PASS
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
#else
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
#endif
            0, 1, &updateBarrier, 0, nullptr, 0, nullptr);

        renderDiag("SHARC update end");
        g_crashRing.record("SHARC:updateDispatch:done");
        sharcUpdateProfile.close();
        worldCommandBuffer->endLabel();
    }
#endif

#ifdef MCVR_ENABLE_SHARC_RESOLVE_PASS
    if (sharcBuffersReady && module->sharcDispatchReady_ && module->sharcBuffersInitialized_
        && module->sharcResolvePipeline_ != VK_NULL_HANDLE) {
        worldCommandBuffer->beginLabel("RT:SHARC Resolve", 0.9f, 0.7f, 0.2f);
        ScopedGpuProfile sharcResolveProfile(profileCmd, "RT.SHARCResolve");
        VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();
        GpuDiag::checkpoint(cmd, GpuDiag::SHARC_RESOLVE);
        renderDiag("SHARC resolve begin capacity=%u frame=%u", module->sharcCapacity_, module->sharcFrameIndex_);
        g_crashRing.record("SHARC:resolveDispatch:start");

        struct SharcResolvePushConstant {
            float cameraPositionPrevX, cameraPositionPrevY, cameraPositionPrevZ;
            uint32_t accumulationFrameNum;
            uint32_t staleFrameNumMax;
            uint32_t frameIndex;
            uint64_t hashEntriesBDA;
            uint64_t accumulationBDA;
            uint64_t resolvedBDA;
            float cameraX, cameraY, cameraZ;
            float sceneScale;
            uint32_t capacity;
            float radianceScale;
            uint32_t enableAntiFirefly;
        } resolvePC = {};

        resolvePC.cameraPositionPrevX = module->sharcPrevCameraX_;
        resolvePC.cameraPositionPrevY = module->sharcPrevCameraY_;
        resolvePC.cameraPositionPrevZ = module->sharcPrevCameraZ_;
        resolvePC.accumulationFrameNum = static_cast<uint32_t>(Renderer::options.sharcAccumulationFrames);
        resolvePC.staleFrameNumMax = static_cast<uint32_t>(Renderer::options.sharcStaleFrames);
        resolvePC.frameIndex = module->sharcFrameIndex_;
        resolvePC.hashEntriesBDA = module->sharcHashEntries_->bufferAddress();
        resolvePC.accumulationBDA = module->sharcAccumulation_->bufferAddress();
        resolvePC.resolvedBDA = module->sharcResolved_->bufferAddress();
        resolvePC.cameraX = pushConstant.sharcCameraX;
        resolvePC.cameraY = pushConstant.sharcCameraY;
        resolvePC.cameraZ = pushConstant.sharcCameraZ;
        resolvePC.sceneScale = Renderer::options.sharcSceneScale;
        resolvePC.capacity = module->sharcCapacity_;
        resolvePC.radianceScale = 1e3f;
        resolvePC.enableAntiFirefly = 1;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->sharcResolvePipeline_);
        vkCmdPushConstants(cmd, module->sharcResolvePipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
            0, 76, &resolvePC);
        vkCmdDispatch(cmd, (module->sharcCapacity_ + 63) / 64, 1, 1);

        VkMemoryBarrier resolveBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        resolveBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        resolveBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 1, &resolveBarrier, 0, nullptr, 0, nullptr);

        module->sharcPrevCameraX_ = pushConstant.sharcCameraX;
        module->sharcPrevCameraY_ = pushConstant.sharcCameraY;
        module->sharcPrevCameraZ_ = pushConstant.sharcCameraZ;
        module->sharcFrameIndex_++;
        renderDiag("SHARC resolve end");
        g_crashRing.record("SHARC:resolveDispatch:done");
        sharcResolveProfile.close();
        worldCommandBuffer->endLabel();
    }
#endif
#endif
#endif // MCVR_ENABLE_SHARC

#ifdef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
    if (upstreamDirectLightActive) {
        renderDiag("RT upstream executor begin");
        g_crashRing.record("RT:upstreamExecutor");
        if (module->renderUpstreamDirectLight(*this)) {
            renderDiag("RT upstream executor end");
            g_crashRing.record("RT:upstreamExecutor:done");
            return;
        }
        renderDiag("RT upstream executor unavailable fallback=legacy");
        g_crashRing.record("RT:upstreamExecutor:fallbackLegacy");
    }
#endif

    // Re-push RT push constants (may have been invalidated by compute pipeline bind above)
    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), rayTracingDescriptorTable->vkPipelineLayout(),
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                           VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                           VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                       0, sizeof(RayTracingPushConstant), &pushConstant);

    // Pass 3: Main Render (existing RT dispatch — now queries SHARC cache on bounces >= 1)
    worldCommandBuffer->beginLabel("RT:MainTrace", 1.0f, 0.2f, 0.2f);
    ScopedGpuProfile mainTraceProfile(profileCmd, "RT.MainTrace");
    GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::RT_DISPATCH_MAIN);
    renderDiag("RT mainTrace begin w=%u h=%u", hdrNoisyOutputImage->width(), hdrNoisyOutputImage->height());
    g_crashRing.record("RT:mainTrace");
    worldCommandBuffer->bindDescriptorTable(rayTracingDescriptorTable, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
        ->bindRTPipeline(module->rayTracingPipeline_)
        ->raytracing(sbt, hdrNoisyOutputImage->width(), hdrNoisyOutputImage->height(), 1);
    renderDiag("RT mainTrace end");
    mainTraceProfile.close();
    worldCommandBuffer->endLabel(); // end MainTrace

#ifdef MCVR_ENABLE_DIRECT_LIGHT_PIPELINE
    bool directLightInitialRan = false;
    if (directLightPipelineActive
        && module->directLightPrimaryPipeline_ != VK_NULL_HANDLE
        && context->frameIndex < module->directLightPrimaryDescSets_.size()
        && context->frameIndex < module->directLightPrimarySurfaceImages_.size()
        && context->frameIndex < module->directLightCounterBuffers_.size()
        && module->directLightPrimarySurfaceImages_[context->frameIndex]
        && module->directLightCounterBuffers_[context->frameIndex]) {
        worldCommandBuffer->beginLabel("RT:Primary Surface Capture", 0.25f, 0.55f, 1.0f);
        ScopedGpuProfile primaryProfile(profileCmd, "RT.Primary");
        VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();
        uint32_t frameIdx = context->frameIndex;
        g_crashRing.record("DirectLight:primaryDispatch:start");

        VkMemoryBarrier prePrimaryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        prePrimaryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        prePrimaryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &prePrimaryBarrier, 0, nullptr, 0, nullptr);

        VkDescriptorSet primarySet = module->directLightPrimaryDescSets_[frameIdx];
        auto addPrimaryImg = [&](uint32_t binding, const std::shared_ptr<vk::DeviceLocalImage>& img,
                                 std::vector<VkWriteDescriptorSet>& writes,
                                 std::vector<std::unique_ptr<VkDescriptorImageInfo>>& infos) {
            auto info = std::make_unique<VkDescriptorImageInfo>();
            info->imageView = img->vkImageView(0);
            info->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            info->sampler = VK_NULL_HANDLE;
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, primarySet, binding, 0, 1,
                             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, info.get(), nullptr, nullptr});
            infos.push_back(std::move(info));
        };

        std::vector<VkWriteDescriptorSet> primaryWrites;
        std::vector<std::unique_ptr<VkDescriptorImageInfo>> primaryInfos;
        addPrimaryImg(0, module->positionViewSpaceImages_[frameIdx], primaryWrites, primaryInfos);
        addPrimaryImg(1, normalRoughnessImage, primaryWrites, primaryInfos);
        addPrimaryImg(2, linearDepthImage, primaryWrites, primaryInfos);
        addPrimaryImg(3, module->directLightPrimarySurfaceImages_[frameIdx], primaryWrites, primaryInfos);
        VkDescriptorBufferInfo counterInfo{
            module->directLightCounterBuffers_[frameIdx]->vkBuffer(), 0,
            module->directLightCounterBuffers_[frameIdx]->size()};
        primaryWrites.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, primarySet, 4, 0, 1,
                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &counterInfo, nullptr});
        vkUpdateDescriptorSets(framework->device()->vkDevice(),
            static_cast<uint32_t>(primaryWrites.size()), primaryWrites.data(), 0, nullptr);

        struct DirectLightPrimaryPushConstant {
            int32_t width;
            int32_t height;
        } primaryPC = {
            static_cast<int32_t>(hdrNoisyOutputImage->width()),
            static_cast<int32_t>(hdrNoisyOutputImage->height()),
        };

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->directLightPrimaryPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->directLightPrimaryPipelineLayout_,
            0, 1, &primarySet, 0, nullptr);
        vkCmdPushConstants(cmd, module->directLightPrimaryPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(primaryPC), &primaryPC);
        vkCmdDispatch(cmd,
            (hdrNoisyOutputImage->width() + 7) / 8,
            (hdrNoisyOutputImage->height() + 7) / 8,
            1);

        VkMemoryBarrier postPrimaryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        postPrimaryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        postPrimaryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 1, &postPrimaryBarrier, 0, nullptr, 0, nullptr);

        g_crashRing.record("DirectLight:primaryDispatch:done");
        primaryProfile.close();
        worldCommandBuffer->endLabel();
    } else if (directLightPipelineActive) {
        auto runDirectLightBoundaryPass = [&](const char* label, const char* profileName,
                                              float r, float g, float b) {
            worldCommandBuffer->beginLabel(label, r, g, b);
            ScopedGpuProfile profile(profileCmd, profileName);
            profile.close();
            worldCommandBuffer->endLabel();
        };
        runDirectLightBoundaryPass("RT:Primary Surface Unavailable", "RT.Primary", 0.25f, 0.55f, 1.0f);
    }

    if (directLightExternalActive) {
        if (module->directLightInitialPipeline_ != VK_NULL_HANDLE
            && context->frameIndex < module->directLightInitialDescSets_.size()
            && context->frameIndex < module->directLightPrimarySurfaceImages_.size()
            && context->frameIndex < module->directLightReservoirPingImages_.size()
            && context->frameIndex < module->directLightOutputImages_.size()
            && context->frameIndex < module->directLightCounterBuffers_.size()
            && module->directLightPrimarySurfaceImages_[context->frameIndex]
            && module->directLightReservoirPingImages_[context->frameIndex]
            && module->directLightOutputImages_[context->frameIndex]
            && module->directLightCounterBuffers_[context->frameIndex]) {
            worldCommandBuffer->beginLabel("RT:DirectLight Initial", 0.1f, 0.7f, 0.4f);
            ScopedGpuProfile initialProfile(profileCmd, "RT.DirectLight.Initial");
            VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();
            uint32_t frameIdx = context->frameIndex;
            g_crashRing.record("DirectLight:initialDispatch:start");

            VkMemoryBarrier preInitialBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            preInitialBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            preInitialBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &preInitialBarrier, 0, nullptr, 0, nullptr);

            VkDescriptorSet initialSet = module->directLightInitialDescSets_[frameIdx];
            auto addInitialImg = [&](uint32_t binding, const std::shared_ptr<vk::DeviceLocalImage>& img,
                                     std::vector<VkWriteDescriptorSet>& writes,
                                     std::vector<std::unique_ptr<VkDescriptorImageInfo>>& infos) {
                auto info = std::make_unique<VkDescriptorImageInfo>();
                info->imageView = img->vkImageView(0);
                info->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                info->sampler = VK_NULL_HANDLE;
                writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, initialSet, binding, 0, 1,
                                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, info.get(), nullptr, nullptr});
                infos.push_back(std::move(info));
            };

            std::vector<VkWriteDescriptorSet> initialWrites;
            std::vector<std::unique_ptr<VkDescriptorImageInfo>> initialInfos;
            addInitialImg(0, module->directLightPrimarySurfaceImages_[frameIdx], initialWrites, initialInfos);
            addInitialImg(1, firstHitDiffuseDirectLightImage, initialWrites, initialInfos);
            addInitialImg(2, directLightDepthImage, initialWrites, initialInfos);
            addInitialImg(3, module->directLightReservoirPingImages_[frameIdx], initialWrites, initialInfos);
            addInitialImg(4, module->directLightOutputImages_[frameIdx], initialWrites, initialInfos);
            addInitialImg(6, module->positionViewSpaceImages_[frameIdx], initialWrites, initialInfos);
            addInitialImg(7, diffuseAlbedoImage, initialWrites, initialInfos);
            VkDescriptorBufferInfo initialCounterInfo{
                module->directLightCounterBuffers_[frameIdx]->vkBuffer(), 0,
                module->directLightCounterBuffers_[frameIdx]->size()};
            initialWrites.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, initialSet, 5, 0, 1,
                                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &initialCounterInfo, nullptr});
            VkWriteDescriptorSetAccelerationStructureKHR initialAsInfo{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
            initialAsInfo.accelerationStructureCount = 1;
            initialAsInfo.pAccelerationStructures = &worldPrepareContext->tlas->tlas();
            initialWrites.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &initialAsInfo, initialSet, 8, 0, 1,
                                     VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr});
            VkDescriptorBufferInfo initialAreaLightInfo{
                worldPrepareContext->areaLightBuffer->vkBuffer(), 0, VK_WHOLE_SIZE};
            initialWrites.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, initialSet, 9, 0, 1,
                                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &initialAreaLightInfo, nullptr});
            VkDescriptorBufferInfo initialTileLightInfo{
                module->tileLightBuffer_->vkBuffer(), 0, VK_WHOLE_SIZE};
            initialWrites.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, initialSet, 10, 0, 1,
                                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &initialTileLightInfo, nullptr});
            VkDescriptorBufferInfo initialWorldInfo{worldBuffer->vkBuffer(), 0, VK_WHOLE_SIZE};
            initialWrites.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, initialSet, 11, 0, 1,
                                     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &initialWorldInfo, nullptr});
            VkDescriptorBufferInfo initialSkyInfo{buffers->skyUniformBuffer()->vkBuffer(), 0, VK_WHOLE_SIZE};
            initialWrites.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, initialSet, 12, 0, 1,
                                     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &initialSkyInfo, nullptr});
            vkUpdateDescriptorSets(framework->device()->vkDevice(),
                static_cast<uint32_t>(initialWrites.size()), initialWrites.data(), 0, nullptr);

            struct DirectLightInitialPushConstant {
                int32_t width;
                int32_t height;
                int32_t areaLightCount;
                int32_t maxLightsPerTile;
                float shadowSoftness;
                int32_t maxAreaLightSamples;
            } initialPC = {
                static_cast<int32_t>(hdrNoisyOutputImage->width()),
                static_cast<int32_t>(hdrNoisyOutputImage->height()),
                worldPrepareContext->areaLightCount,
                RayTracingModule::MAX_LIGHTS_PER_TILE,
                Renderer::options.shadowSoftness,
                std::min(Renderer::options.restirCandidates, 8),
            };

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->directLightInitialPipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->directLightInitialPipelineLayout_,
                0, 1, &initialSet, 0, nullptr);
            vkCmdPushConstants(cmd, module->directLightInitialPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(initialPC), &initialPC);
            vkCmdDispatch(cmd,
                (hdrNoisyOutputImage->width() + 7) / 8,
                (hdrNoisyOutputImage->height() + 7) / 8,
                1);

            VkMemoryBarrier postInitialBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            postInitialBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            postInitialBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                0, 1, &postInitialBarrier, 0, nullptr, 0, nullptr);

            directLightInitialRan = true;
            g_crashRing.record("DirectLight:initialDispatch:done");
            initialProfile.close();
            worldCommandBuffer->endLabel();
        }

        auto runDirectLightUtilityPass = [&](uint32_t utilityPass, int mode, const char* label, const char* profileName,
                                             const std::shared_ptr<vk::DeviceLocalImage>& input,
                                             const std::shared_ptr<vk::DeviceLocalImage>& output,
                                             float r, float g, float b) -> bool {
            constexpr uint32_t kDirectLightUtilityPasses = 4;
            if (module->directLightUtilityPipeline_ == VK_NULL_HANDLE
                || context->frameIndex >= module->directLightCounterBuffers_.size()
                || context->frameIndex >= module->directLightUtilityDescSets_.size() / kDirectLightUtilityPasses
                || context->frameIndex >= module->directLightOutputImages_.size()
                || !input || !output || !module->directLightOutputImages_[context->frameIndex]
                || !module->directLightCounterBuffers_[context->frameIndex]) {
                return false;
            }

            worldCommandBuffer->beginLabel(label, r, g, b);
            ScopedGpuProfile utilityProfile(profileCmd, profileName);
            VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();
            uint32_t frameIdx = context->frameIndex;
            const std::string crashBase = std::string("DirectLight:") + profileName;
            const std::string crashStart = crashBase + ":start";
            g_crashRing.record(crashStart.c_str());

            VkMemoryBarrier preUtilityBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            preUtilityBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            preUtilityBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &preUtilityBarrier, 0, nullptr, 0, nullptr);

            VkDescriptorSet utilitySet =
                module->directLightUtilityDescSets_[frameIdx * kDirectLightUtilityPasses + utilityPass];
            auto addUtilityImg = [&](uint32_t binding, const std::shared_ptr<vk::DeviceLocalImage>& img,
                                     std::vector<VkWriteDescriptorSet>& writes,
                                     std::vector<std::unique_ptr<VkDescriptorImageInfo>>& infos) {
                auto info = std::make_unique<VkDescriptorImageInfo>();
                info->imageView = img->vkImageView(0);
                info->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                info->sampler = VK_NULL_HANDLE;
                writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, utilitySet, binding, 0, 1,
                                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, info.get(), nullptr, nullptr});
                infos.push_back(std::move(info));
            };

            std::vector<VkWriteDescriptorSet> utilityWrites;
            std::vector<std::unique_ptr<VkDescriptorImageInfo>> utilityInfos;
            addUtilityImg(0, input, utilityWrites, utilityInfos);
            addUtilityImg(1, output, utilityWrites, utilityInfos);
            addUtilityImg(2, module->directLightOutputImages_[frameIdx], utilityWrites, utilityInfos);
            addUtilityImg(4, firstHitDiffuseDirectLightImage, utilityWrites, utilityInfos);
            VkDescriptorBufferInfo utilityCounterInfo{
                module->directLightCounterBuffers_[frameIdx]->vkBuffer(), 0,
                module->directLightCounterBuffers_[frameIdx]->size()};
            utilityWrites.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, utilitySet, 3, 0, 1,
                                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &utilityCounterInfo, nullptr});
            vkUpdateDescriptorSets(framework->device()->vkDevice(),
                static_cast<uint32_t>(utilityWrites.size()), utilityWrites.data(), 0, nullptr);

            struct DirectLightUtilityPushConstant {
                int32_t width;
                int32_t height;
                int32_t mode;
                int32_t pad0;
            } utilityPC = {
                static_cast<int32_t>(hdrNoisyOutputImage->width()),
                static_cast<int32_t>(hdrNoisyOutputImage->height()),
                mode,
                0,
            };

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->directLightUtilityPipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->directLightUtilityPipelineLayout_,
                0, 1, &utilitySet, 0, nullptr);
            vkCmdPushConstants(cmd, module->directLightUtilityPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(utilityPC), &utilityPC);
            vkCmdDispatch(cmd,
                (hdrNoisyOutputImage->width() + 7) / 8,
                (hdrNoisyOutputImage->height() + 7) / 8,
                1);

            VkMemoryBarrier postUtilityBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            postUtilityBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            postUtilityBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                0, 1, &postUtilityBarrier, 0, nullptr, 0, nullptr);

            const std::string crashDone = crashBase + ":done";
            g_crashRing.record(crashDone.c_str());
            utilityProfile.close();
            worldCommandBuffer->endLabel();
            return true;
        };

        bool directLightUtilityRan = false;
        if (directLightInitialRan
            && context->frameIndex < module->directLightReservoirPingImages_.size()
            && context->frameIndex < module->directLightReservoirPongImages_.size()
            && module->directLightReservoirPingImages_[context->frameIndex]
            && module->directLightReservoirPongImages_[context->frameIndex]) {
            uint32_t frameIdx = context->frameIndex;
            bool temporalRan = runDirectLightUtilityPass(0, 1, "RT:DirectLight Temporal", "RT.DirectLight.Temporal",
                module->directLightReservoirPingImages_[frameIdx],
                module->directLightReservoirPongImages_[frameIdx], 0.1f, 0.75f, 0.55f);
            bool spatialRan = runDirectLightUtilityPass(1, 2, "RT:DirectLight Spatial", "RT.DirectLight.Spatial",
                module->directLightReservoirPongImages_[frameIdx],
                module->directLightReservoirPingImages_[frameIdx], 0.1f, 0.8f, 0.7f);
            bool visibilityRan = runDirectLightUtilityPass(2, 3, "RT:DirectLight Visibility", "RT.DirectLight.Visibility",
                module->directLightReservoirPingImages_[frameIdx],
                module->directLightReservoirPongImages_[frameIdx], 0.9f, 0.75f, 0.25f);
            bool shadeRan = runDirectLightUtilityPass(3, 4, "RT:DirectLight Shade", "RT.DirectLight.Shade",
                module->directLightReservoirPongImages_[frameIdx],
                module->directLightReservoirPingImages_[frameIdx], 0.95f, 0.55f, 0.2f);
            directLightUtilityRan = temporalRan && spatialRan && visibilityRan && shadeRan;
        }

        auto runDirectLightBoundaryPass = [&](const char* label, const char* profileName,
                                              float r, float g, float b) {
            worldCommandBuffer->beginLabel(label, r, g, b);
            ScopedGpuProfile profile(profileCmd, profileName);
            profile.close();
            worldCommandBuffer->endLabel();
        };
        if (!directLightInitialRan) {
            runDirectLightBoundaryPass("RT:DirectLight Initial Skeleton", "RT.DirectLight.Initial", 0.1f, 0.7f, 0.4f);
        }
        if (!directLightUtilityRan) {
            runDirectLightBoundaryPass("RT:DirectLight Temporal Skeleton", "RT.DirectLight.Temporal", 0.1f, 0.75f, 0.55f);
            runDirectLightBoundaryPass("RT:DirectLight Spatial Skeleton", "RT.DirectLight.Spatial", 0.1f, 0.8f, 0.7f);
            runDirectLightBoundaryPass("RT:DirectLight Visibility Skeleton", "RT.DirectLight.Visibility", 0.9f, 0.75f, 0.25f);
            runDirectLightBoundaryPass("RT:DirectLight Shade Skeleton", "RT.DirectLight.Shade", 0.95f, 0.55f, 0.2f);
        }
    }
#endif

#ifdef MCVR_ENABLE_SHARC_QUERY_PASS
    if (sharcQueryPassActive) {
        worldCommandBuffer->beginLabel("RT:SHARC Query", 0.9f, 0.8f, 0.2f);
        ScopedGpuProfile sharcQueryProfile(profileCmd, "RT.SHARCQuery");
        VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();
        uint32_t frameIdx = context->frameIndex;
        g_crashRing.record("SHARC:queryDispatch:start");

        VkMemoryBarrier preQueryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        preQueryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        preQueryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &preQueryBarrier, 0, nullptr, 0, nullptr);

        VkDescriptorSet querySet = module->sharcQueryDescSets_[frameIdx];
        auto addQueryImg = [&](uint32_t binding, const std::shared_ptr<vk::DeviceLocalImage>& img,
                               std::vector<VkWriteDescriptorSet>& writes,
                               std::vector<std::unique_ptr<VkDescriptorImageInfo>>& infos) {
            auto info = std::make_unique<VkDescriptorImageInfo>();
            info->imageView = img->vkImageView(0);
            info->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            info->sampler = VK_NULL_HANDLE;
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, querySet, binding, 0, 1,
                             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, info.get(), nullptr, nullptr});
            infos.push_back(std::move(info));
        };

        std::vector<VkWriteDescriptorSet> queryWrites;
        std::vector<std::unique_ptr<VkDescriptorImageInfo>> queryInfos;
        addQueryImg(0, module->sharcCandidatePosHitTImages_[frameIdx], queryWrites, queryInfos);
        addQueryImg(1, module->sharcCandidateNormalRoughnessImages_[frameIdx], queryWrites, queryInfos);
        addQueryImg(2, module->sharcCandidateThroughputImages_[frameIdx], queryWrites, queryInfos);
        addQueryImg(3, module->sharcCandidatePrefixRadianceFlagsImages_[frameIdx], queryWrites, queryInfos);
        addQueryImg(4, firstHitDiffuseDirectLightImage, queryWrites, queryInfos);
        addQueryImg(5, firstHitBaseEmissionImage, queryWrites, queryInfos);
        addQueryImg(6, firstHitDiffuseIndirectLightImage, queryWrites, queryInfos);

        VkDescriptorBufferInfo counterInfo{
            module->sharcQueryCounterBuffers_[frameIdx]->vkBuffer(), 0,
            module->sharcQueryCounterBuffers_[frameIdx]->size()};
        queryWrites.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, querySet, 7, 0, 1,
                               VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &counterInfo, nullptr});
        vkUpdateDescriptorSets(framework->device()->vkDevice(),
            static_cast<uint32_t>(queryWrites.size()), queryWrites.data(), 0, nullptr);

        struct SharcQueryPushConstant {
            int32_t width, height, mode, reserved0;
            uint64_t hashEntriesBDA, resolvedBDA;
            float cameraX, cameraY, cameraZ, sceneScale;
            uint32_t capacity;
            float roughnessThreshold;
            uint32_t frameIndex;
            uint32_t reserved1;
        } queryPC = {};
        queryPC.width = static_cast<int32_t>(hdrNoisyOutputImage->width());
        queryPC.height = static_cast<int32_t>(hdrNoisyOutputImage->height());
        queryPC.mode = sharcQueryMode;
        queryPC.hashEntriesBDA = module->sharcHashEntries_->bufferAddress();
        queryPC.resolvedBDA = module->sharcResolved_->bufferAddress();
        queryPC.cameraX = pushConstant.sharcCameraX;
        queryPC.cameraY = pushConstant.sharcCameraY;
        queryPC.cameraZ = pushConstant.sharcCameraZ;
        queryPC.sceneScale = Renderer::options.sharcSceneScale;
        queryPC.capacity = module->sharcCapacity_;
        queryPC.roughnessThreshold = Renderer::options.sharcRoughnessThreshold;
        queryPC.frameIndex = module->sharcFrameIndex_;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->sharcQueryPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->sharcQueryPipelineLayout_,
            0, 1, &querySet, 0, nullptr);
        vkCmdPushConstants(cmd, module->sharcQueryPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(queryPC), &queryPC);
        vkCmdDispatch(cmd,
            (hdrNoisyOutputImage->width() + 7) / 8,
            (hdrNoisyOutputImage->height() + 7) / 8,
            1);

        VkMemoryBarrier postQueryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        postQueryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        postQueryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 1, &postQueryBarrier, 0, nullptr, 0, nullptr);

        renderDiag("SHARC query end mode=%d", sharcQueryMode);
        g_crashRing.record("SHARC:queryDispatch:done");
        sharcQueryProfile.close();
        worldCommandBuffer->endLabel();
    }
#endif

    // Spatial reuse compute pass (when ReSTIR and spatial reuse are both enabled)
    if (Renderer::options.restirEnabled && Renderer::options.restirSpatialEnabled && module->spatialPipeline_ != VK_NULL_HANDLE
        && module->reservoirImages_[0] && module->reservoirImages_[1]) {
        worldCommandBuffer->beginLabel("RT:ReSTIR Spatial", 0.2f, 0.8f, 0.8f);
        ScopedGpuProfile restirSpatialProfile(profileCmd, "RT.ReSTIRSpatial");
        VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();
        GpuDiag::checkpoint(cmd, GpuDiag::RESTIR_SPATIAL);

        // Barrier: RT shader writes → compute shader reads
        VkMemoryBarrier spatialBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        spatialBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        spatialBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &spatialBarrier, 0, nullptr, 0, nullptr);

        // Update spatial descriptor set with current frame's images
        uint32_t frameIdx = context->frameIndex;
        VkDescriptorSet spatialSet = module->spatialDescSets_[frameIdx];

        auto addSpatialImg = [&](uint32_t binding, const std::shared_ptr<vk::DeviceLocalImage>& img,
                                 std::vector<VkWriteDescriptorSet>& writes,
                                 std::vector<std::unique_ptr<VkDescriptorImageInfo>>& infos) {
            auto info = std::make_unique<VkDescriptorImageInfo>();
            info->imageView = img->vkImageView(0);
            info->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            info->sampler = VK_NULL_HANDLE;
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, spatialSet, binding, 0, 1,
                             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, info.get(), nullptr, nullptr});
            infos.push_back(std::move(info));
        };

        std::vector<VkWriteDescriptorSet> spatialWrites;
        std::vector<std::unique_ptr<VkDescriptorImageInfo>> spatialInfos;
        // Spatial reads CHS temporal output [0], writes spatial output [1]
        addSpatialImg(0, module->reservoirImages_[0], spatialWrites, spatialInfos);
        addSpatialImg(1, module->reservoirImages_[1], spatialWrites, spatialInfos);
        addSpatialImg(2, normalRoughnessImage, spatialWrites, spatialInfos);
        addSpatialImg(3, linearDepthImage, spatialWrites, spatialInfos);

        vkUpdateDescriptorSets(framework->device()->vkDevice(),
            (uint32_t)spatialWrites.size(), spatialWrites.data(), 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->spatialPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->spatialPipelineLayout_,
            0, 1, &spatialSet, 0, nullptr);

        struct { int32_t width, height, spatialTaps, spatialRadius, temporalMClamp, wClamp; } spatialPC = {
            (int32_t)hdrNoisyOutputImage->width(),
            (int32_t)hdrNoisyOutputImage->height(),
            Renderer::options.restirSpatialTaps,
            Renderer::options.restirSpatialRadius,
            Renderer::options.restirTemporalMClamp,
            Renderer::options.restirWClamp
        };
        vkCmdPushConstants(cmd, module->spatialPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(spatialPC), &spatialPC);

        vkCmdDispatch(cmd,
            (hdrNoisyOutputImage->width() + 15) / 16,
            (hdrNoisyOutputImage->height() + 15) / 16,
            1);

        // Barrier: compute writes → next stage reads
        VkMemoryBarrier postSpatialBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        postSpatialBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        postSpatialBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 1, &postSpatialBarrier, 0, nullptr, 0, nullptr);
        restirSpatialProfile.close();
        worldCommandBuffer->endLabel(); // end ReSTIR Spatial
    }

    // Offline accumulation: Welford running average into RGBA32F buffer
    // Raw Fast (0) and Raw Slow (1) accumulate here; DLSS-D (2) accumulates in DLSS module
    if (accumulating && Renderer::accumPipelineReady && Renderer::options.offlineDenoised != 2) {
        VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();
        uint32_t frameIdx = context->frameIndex;
        ScopedGpuProfile offlineAccumProfile(profileCmd, "RT.OfflineAccum");

        // Barrier: RT output → compute read
        VkMemoryBarrier accumBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        accumBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        accumBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &accumBarrier, 0, nullptr, 0, nullptr);

        // Update descriptor set (only when images change — typically once at init)
        VkDescriptorSet accumSet = Renderer::accumDescSets[frameIdx];
        VkDescriptorImageInfo accumImgInfo{VK_NULL_HANDLE, Renderer::accumBufferImage->vkImageView(0), VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo noisyImgInfo{VK_NULL_HANDLE, hdrNoisyOutputImage->vkImageView(0), VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet accumWrites[] = {
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, accumSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImgInfo, nullptr, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, accumSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &noisyImgInfo, nullptr, nullptr},
        };
        vkUpdateDescriptorSets(framework->device()->vkDevice(), 2, accumWrites, 0, nullptr);

        // Dispatch accumulation (writes averaged result to accumBuffer only)
        int32_t fc = static_cast<int32_t>(Renderer::accumFrameCount);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, Renderer::accumPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, Renderer::accumPipelineLayout,
            0, 1, &accumSet, 0, nullptr);
        vkCmdPushConstants(cmd, Renderer::accumPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &fc);
        vkCmdDispatch(cmd,
            (hdrNoisyOutputImage->width() + 7) / 8,
            (hdrNoisyOutputImage->height() + 7) / 8,
            1);

        // Barrier: compute write → blit read
        VkMemoryBarrier postAccumBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        postAccumBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        postAccumBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &postAccumBarrier, 0, nullptr, 0, nullptr);

        // Copy accumulated average from RGBA32F accumBuffer to RGBA16F hdrNoisyOutput
        VkImageBlit accumBlit{};
        accumBlit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        accumBlit.srcOffsets[1] = {(int)Renderer::accumBufferImage->width(), (int)Renderer::accumBufferImage->height(), 1};
        accumBlit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        accumBlit.dstOffsets[1] = {(int)hdrNoisyOutputImage->width(), (int)hdrNoisyOutputImage->height(), 1};
        vkCmdBlitImage(cmd,
            Renderer::accumBufferImage->vkImage(), VK_IMAGE_LAYOUT_GENERAL,
            hdrNoisyOutputImage->vkImage(), VK_IMAGE_LAYOUT_GENERAL,
            1, &accumBlit, VK_FILTER_NEAREST);

        // Barrier: blit write → next stage read
        VkMemoryBarrier postBlitBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        postBlitBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        postBlitBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &postBlitBarrier, 0, nullptr, 0, nullptr);

        Renderer::accumFrameCount++;
        Renderer::accumOutputImage = hdrNoisyOutputImage;  // expose for denoiser/upscaler bypass
        offlineAccumProfile.close();
    }

    renderDiag("RT end");
    g_crashRing.record("RT:end");

    // (P4 removed — DLSS temporal mode replaced by DLSS-D Converge with per-frame reset)
}
