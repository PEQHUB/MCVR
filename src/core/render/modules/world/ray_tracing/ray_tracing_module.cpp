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
#include <array>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>

namespace {
constexpr int kSharcMinCapacityExponent = 18;
constexpr int kSharcMaxCapacityExponent = 24;
constexpr uint32_t kSharcDispatchStableFrameThreshold = 60;
constexpr uint32_t kSharcMainTraceMaxWarmupFrames = 8;
constexpr uint32_t kSharcQueryCounterCount = 8;
constexpr uint32_t kBrdfFlagEonDiffuse = 1u << 8;
constexpr uint32_t kBrdfFlagVmfDiffuse = 1u << 16;
static_assert(kBrdfFlagEonDiffuse == 256u, "EON diffuse BRDF flag changed");
static_assert(kBrdfFlagVmfDiffuse == 65536u, "VMF diffuse BRDF flag changed");

constexpr uint32_t diffuseModelFlagBits(uint32_t model) {
    switch (model) {
        case 0u: return kBrdfFlagEonDiffuse;
        case 1u: return kBrdfFlagVmfDiffuse;
        default: return 0u;
    }
}

bool shaderDisplacementForceDisabled() {
    const char* value = std::getenv("RADSER_FORCE_DISABLE_SHADER_DISPLACEMENT");
    if (!value || value[0] == '\0') return false;
    return value[0] != '0' && value[0] != 'f' && value[0] != 'F' &&
           value[0] != 'n' && value[0] != 'N';
}

bool shaderDisplacementRuntimeAllowed() {
    return !shaderDisplacementForceDisabled();
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

std::string diagnosticAttributeValue(const std::vector<std::string> &attributeKVs,
                                     const std::string &key,
                                     const std::string &fallback) {
    for (size_t i = 0; i + 1 < attributeKVs.size(); i += 2) {
        if (attributeKVs[i] == key) return attributeKVs[i + 1];
    }
    return fallback;
}

std::string diagnosticEnumToken(const std::string &value) {
    auto pos = value.rfind('.');
    return pos == std::string::npos ? value : value.substr(pos + 1);
}

std::string diagnosticBackendFromMode(const std::string &mode) {
    if (mode == "off") return "off";
    if (mode == "default" || mode.empty()) return "unknown";
    return "shader_pack";
}

std::string shaderPackDiagnosticToken(std::string value) {
    for (char &c : value) {
        if (c == ',' || c == ':' || c == ' ' || c == '\t' || c == '\n' || c == '\r') { c = '_'; }
    }
    return value.empty() ? "none" : value;
}

std::string shaderPackVariableTypeFromValue(const std::string &value) {
    if (value == "true" || value == "false" || value == "1" || value == "0" ||
        value == "render_pipeline.true" || value == "render_pipeline.false") {
        return "bool";
    }

    size_t parsed = 0;
    try {
        (void)std::stoi(value, &parsed);
        if (parsed == value.size()) { return "int"; }
    } catch (...) {}

    parsed = 0;
    try {
        (void)std::stof(value, &parsed);
        if (parsed == value.size()) { return "float"; }
    } catch (...) {}

    return "string";
}

bool shaderPackBoolValue(const std::string &value, bool fallback) {
    if (value == "1" || value == "true" || value == "True" || value == "TRUE" ||
        value == "render_pipeline.true") {
        return true;
    }
    if (value == "0" || value == "false" || value == "False" || value == "FALSE" ||
        value == "render_pipeline.false") {
        return false;
    }
    return fallback;
}

bool shaderPackEnumEndsWith(const std::string &value, const char *token) {
    const std::string suffix = std::string(".") + token;
    return value == token ||
           (value.size() >= suffix.size() &&
            value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0);
}

int shaderPackCloudModeValue(const std::string &value) {
    if (shaderPackEnumEndsWith(value, "off")) { return 0; }
    if (shaderPackEnumEndsWith(value, "vanilla")) { return 1; }
    if (shaderPackEnumEndsWith(value, "volumetric")) { return 2; }
    return 2;
}

int shaderPackWaterSurfaceModeValue(const std::string &value) {
    if (shaderPackEnumEndsWith(value, "vanilla")) { return 0; }
    if (shaderPackEnumEndsWith(value, "realistic")) { return 1; }
    return 1;
}

int shaderPackVolumetricLightModeValue(const std::string &value) {
    if (shaderPackEnumEndsWith(value, "vanilla")) { return 0; }
    if (shaderPackEnumEndsWith(value, "volumetric")) { return 1; }
    return 1;
}

int shaderPackClearAmountValue(const std::string &value) {
    if (shaderPackEnumEndsWith(value, "few")) { return 0; }
    if (shaderPackEnumEndsWith(value, "medium")) { return 1; }
    if (shaderPackEnumEndsWith(value, "many")) { return 2; }
    return 1;
}

int shaderPackIntValue(const std::string &value, int fallback, int minValue, int maxValue) {
    try {
        return std::clamp(std::stoi(value), minValue, maxValue);
    } catch (...) {
        return fallback;
    }
}

float shaderPackFloatValue(const std::string &value, float fallback, float minValue, float maxValue) {
    try {
        return std::clamp(std::stof(value), minValue, maxValue);
    } catch (...) {
        return fallback;
    }
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
    const std::string requestedPath = shaderPackRequestedPath_.empty() ? shaderPackPath_ : shaderPackRequestedPath_;
    const bool fallbackActive = shaderPackRuntimeStatus_ == "fallback";
    const bool selectedVanillaPt = shaderPackPath_.find("vanilla-pt") != std::string::npos;
    const bool shaderPackHasRuntimeResources = shaderPack_ && shaderPack_->hasRuntimeResources();
    const bool shaderPackVisualAssetsReady = shaderPackRuntimeResourcesReady_ && shaderPackHasRuntimeResources;
    const std::string cloudMode = diagnosticEnumToken(diagnosticAttributeValue(
        shaderPackAttributeKVs_,
        "render_pipeline.module.ray_tracing.attribute.cloud_mode",
        "render_pipeline.module.ray_tracing.attribute.cloud_mode.volumetric"));
    const std::string waterSurfaceMode = diagnosticEnumToken(diagnosticAttributeValue(
        shaderPackAttributeKVs_,
        "render_pipeline.module.ray_tracing.attribute.water_surface_mode",
        "render_pipeline.module.ray_tracing.attribute.water_surface_mode.realistic"));
    const std::string volumetricLightMode = diagnosticEnumToken(diagnosticAttributeValue(
        shaderPackAttributeKVs_,
        "render_pipeline.module.ray_tracing.attribute.volumetric_light_mode",
        "render_pipeline.module.ray_tracing.attribute.volumetric_light_mode.volumetric"));
    const std::string cloudTemporal = diagnosticEnumToken(diagnosticAttributeValue(
        shaderPackAttributeKVs_,
        "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_temporal_accumulation",
        "render_pipeline.true"));
    const std::string cloudIndirect = diagnosticEnumToken(diagnosticAttributeValue(
        shaderPackAttributeKVs_,
        "render_pipeline.module.ray_tracing.attribute.capture_volumetric_cloud_indirect",
        "render_pipeline.true"));

    uint32_t outputFamiliesReady = 0;
    auto familyReady = [](const auto &images) {
        return !images.empty() && std::all_of(images.begin(), images.end(),
                                             [](const auto &image) { return image != nullptr; });
    };
    auto countReady = [&](const auto &images) {
        if (familyReady(images)) outputFamiliesReady++;
    };
    countReady(hdrNoisyOutputImages_);
    countReady(diffuseAlbedoImages_);
    countReady(specularAlbedoImages_);
    countReady(normalRoughnessImages_);
    countReady(motionVectorImages_);
    countReady(linearDepthImages_);
    countReady(specularHitDepthImages_);
    countReady(firstHitDepthImages_);
    countReady(firstHitDiffuseDirectLightImages_);
    countReady(firstHitDiffuseIndirectLightImages_);
    countReady(firstHitSpecularImages_);
    countReady(firstHitClearImages_);
    countReady(firstHitBaseEmissionImages_);
    countReady(directLightDepthImages_);
    countReady(diffuseRayDirHitDistImages_);
    countReady(specularRayDirHitDistImages_);
    countReady(reflectionMvImages_);
    countReady(animatedTexMaskImages_);
    countReady(particleMaskImages_);
    countReady(biasMaskImages_);
    countReady(rtHitDistImages_);
    countReady(motionVectors3DImages_);
    countReady(gbufferMetallicImages_);
    countReady(gbufferShadingModelIdImages_);
    countReady(gbufferMaterialIdImages_);
    countReady(positionViewSpaceImages_);
    countReady(transparencyLayerImages_);
    countReady(transparencyLayerOpacityImages_);
    countReady(transparencyLayerMvecsImages_);

    uint32_t dlssGuideFamiliesReady = 0;
    auto countGuideReady = [&](const auto &images) {
        if (familyReady(images)) dlssGuideFamiliesReady++;
    };
    countGuideReady(diffuseRayDirHitDistImages_);
    countGuideReady(specularRayDirHitDistImages_);
    countGuideReady(reflectionMvImages_);
    countGuideReady(animatedTexMaskImages_);
    countGuideReady(particleMaskImages_);
    countGuideReady(biasMaskImages_);
    countGuideReady(rtHitDistImages_);
    countGuideReady(motionVectors3DImages_);
    countGuideReady(gbufferMetallicImages_);
    countGuideReady(gbufferShadingModelIdImages_);
    countGuideReady(gbufferMaterialIdImages_);
    countGuideReady(positionViewSpaceImages_);
    countGuideReady(transparencyLayerImages_);
    countGuideReady(transparencyLayerOpacityImages_);
    countGuideReady(transparencyLayerMvecsImages_);

    out << "offlineAccumulating:" << (accumulating ? 1 : 0)
        << ",sharcOption:" << (Renderer::options.sharcEnabled ? 1 : 0);

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

    const bool shaderPackPartialAdapter = shaderPackBackendStatus_.rfind("adapter_", 0) == 0;
    const bool shaderPackExecutorFull = shaderPackBackendStatus_ == "executor_full";
    auto passDispatchCount = [&](const char *name) {
        auto iter = shaderPackPassDispatchCounts_.find(name);
        return iter == shaderPackPassDispatchCounts_.end() ? 0ull :
            static_cast<unsigned long long>(iter->second);
    };
    const auto shaderPackWorldDispatches = passDispatchCount("world");
    const bool vanillaPtWorldOutputProducerReady =
        selectedVanillaPt && shaderPackExecutorFull && shaderPackWorldDispatches > 0 &&
        outputFamiliesReady == 29 && dlssGuideFamiliesReady == 15;
    const uint32_t vanillaPtMissingForkOutImages =
        (selectedVanillaPt && shaderPackExecutorReady_) ? (29u - std::min<uint32_t>(outputFamiliesReady, 29u)) : 16u;
    const uint32_t vanillaPtMissingDlssGuideImages =
        (selectedVanillaPt && shaderPackExecutorReady_) ? (15u - std::min<uint32_t>(dlssGuideFamiliesReady, 15u)) : 15u;
    const char *shaderPackAdapterMode =
        shaderPackPartialAdapter ? "partial_adapter" :
        (shaderPackExecutorFull ? "pack_executor" :
         (shaderPackExecutorReady_ ? "pack_executor_pending_full_frame" : "fixed_fallback"));

    auto runtimeTextureStatus = [&](std::string_view name) {
        struct Status {
            bool declared = false;
            bool imageReady = false;
            bool cube = false;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t layers = 0;
            int sampledBinding = -1;
            int storageBinding = -1;
            uint32_t sampledViewIndex = 0;
        };
        Status status;
        if (!shaderPack_) { return status; }
        auto textureRef = shaderPack_->findRuntimeTexture(name);
        if (!textureRef.has_value()) { return status; }
        const auto &texture = textureRef->get();
        status.declared = true;
        status.cube = texture.config.dimension == ShaderPackLoader::TextureDimension::Cube;
        status.sampledBinding = texture.config.sampledBinding.has_value()
            ? static_cast<int>(*texture.config.sampledBinding)
            : -1;
        status.storageBinding = texture.config.storageBinding.has_value()
            ? static_cast<int>(*texture.config.storageBinding)
            : -1;
        status.sampledViewIndex = texture.sampledViewIndex;
        if (texture.config.imported) {
            if (!texture.importedImage) { return status; }
        } else if (texture.frameImages.empty()) {
            return status;
        }
        auto image = shaderPack_->findRuntimeVKTexture(texture, 0);
        if (image) {
            status.imageReady = true;
            status.width = image->width();
            status.height = image->height();
            status.layers = image->layer();
        }
        return status;
    };
    const auto skyCubeStatus = runtimeTextureStatus("sky_cube");
    const auto transLutStatus = runtimeTextureStatus("trans_lut");

    bool shaderPackWorldQuerySharcDeclared = false;
    if (shaderPack_) {
        for (const auto &packPass : shaderPack_->shaderPack().passes) {
            if (packPass.stage == ShaderPackLoader::Stage::RayTracing &&
                packPass.type == ShaderPackLoader::PassConfig::Type::RayTracing &&
                packPass.rayTracing.name == "world") {
                shaderPackWorldQuerySharcDeclared = packPass.rayTracing.querySharc;
                break;
            }
        }
    }

    uint32_t sbtGeometrySolid = 0;
    uint32_t sbtGeometryTransparent = 0;
    uint32_t sbtGeometryShadow = 0;
    uint32_t sbtGeometryOther = 0;
    uint32_t sbtGeometryTotal = 0;
    for (const auto &contextBase : contexts_) {
        auto rtContext = std::dynamic_pointer_cast<RayTracingModuleContext>(contextBase);
        if (!rtContext || !rtContext->worldPrepareContext ||
            rtContext->worldPrepareContext->lastGeometryTypes_.empty()) {
            continue;
        }
        for (uint32_t geometryTypeValue : rtContext->worldPrepareContext->lastGeometryTypes_) {
            sbtGeometryTotal++;
            switch (static_cast<World::GeometryTypes>(geometryTypeValue)) {
                case World::SHADOW:
                    sbtGeometryShadow++;
                    break;
                case World::WORLD_SOLID:
                    sbtGeometrySolid++;
                    break;
                case World::WORLD_TRANSPARENT:
                    sbtGeometryTransparent++;
                    break;
                default:
                    sbtGeometryOther++;
                    break;
            }
        }
        break;
    }

    out << ",sharcCompiled:" << (sharcCompiled ? 1 : 0)
        << ",rayTracingBackend:native"
        << ",shaderPackRequestedPath:" << requestedPath
        << ",shaderPackSelectedPath:" << shaderPackPath_
        << ",shaderPackPath:" << shaderPackPath_
        << ",shaderPackRuntimeValid:" << (shaderPackRuntimeValid_ ? 1 : 0)
        << ",shaderPackRuntimeStatus:" << shaderPackRuntimeStatus_
        << ",shaderPackBackendEnabled:" << (shaderPackBackendEnabled_ ? 1 : 0)
        << ",shaderPackBackendStatus:" << shaderPackBackendStatus_
        << ",shaderPackAdapterMode:" << shaderPackAdapterMode
        << ",shaderPackRuntimeResourceSetIndex:" << shaderPackRuntimeResourceSetIndex_
        << ",shaderPackExecutionSetIndex:" << (shaderPackExecutorReady_ ? std::to_string(shaderPackExecutionSetIndex_) : "none")
        << ",shaderPackIntermediatesGenerated:" << shaderPackIntermediatesGenerated_
        << ",shaderPackExecutorReady:" << (shaderPackExecutorReady_ ? 1 : 0)
        << ",shaderPackFullScreenPassDispatches:" << shaderPackFullScreenPassDispatches_
        << ",shaderPackComputePassDispatches:" << shaderPackComputePassDispatches_
        << ",shaderPackRayTracingPassDispatches:" << shaderPackRayTracingPassDispatches_
        << ",shaderPackPassDispatch_trans_lut:" << passDispatchCount("trans_lut")
        << ",shaderPackPassDispatch_sky_cube:" << passDispatchCount("sky_cube")
        << ",shaderPackPassDispatch_world:" << passDispatchCount("world")
        << ",shaderPackPassDispatch_volumetric_cloud_temporal:" << passDispatchCount("volumetric_cloud_temporal")
        << ",shaderPackPassDispatch_volumetric_cloud_apply:" << passDispatchCount("volumetric_cloud_apply")
        << ",shaderPackRuntimeResourcesDeclared:" << (shaderPackHasRuntimeResources ? 1 : 0)
        << ",shaderPackRuntimeResourcesReady:" << (shaderPackRuntimeResourcesReady_ ? 1 : 0)
        << ",shaderPackVisualAssetsReady:" << (shaderPackVisualAssetsReady ? 1 : 0)
        << ",shaderPackWorldQuerySharcDeclared:" << (shaderPackWorldQuerySharcDeclared ? 1 : 0)
        << ",shaderPackSet4ReservedForForkVisualSettings:1"
        << ",shaderPackSkyCubeDeclared:" << (skyCubeStatus.declared ? 1 : 0)
        << ",shaderPackSkyCubeReady:" << (skyCubeStatus.imageReady ? 1 : 0)
        << ",shaderPackSkyCubeIsCube:" << (skyCubeStatus.cube ? 1 : 0)
        << ",shaderPackSkyCubeWidth:" << skyCubeStatus.width
        << ",shaderPackSkyCubeHeight:" << skyCubeStatus.height
        << ",shaderPackSkyCubeLayers:" << skyCubeStatus.layers
        << ",shaderPackSkyCubeSampledBinding:" << skyCubeStatus.sampledBinding
        << ",shaderPackSkyCubeSampledViewIndex:" << skyCubeStatus.sampledViewIndex
        << ",shaderPackTransLutDeclared:" << (transLutStatus.declared ? 1 : 0)
        << ",shaderPackTransLutReady:" << (transLutStatus.imageReady ? 1 : 0)
        << ",shaderPackTransLutWidth:" << transLutStatus.width
        << ",shaderPackTransLutHeight:" << transLutStatus.height
        << ",shaderPackTransLutSampledBinding:" << transLutStatus.sampledBinding
        << ",shaderPackTransLutStorageBinding:" << transLutStatus.storageBinding
        << ",shaderPackFirstFrameAttempted:" << (shaderPackFirstFrameAttempted_ ? 1 : 0)
        << ",shaderPackFallbackReason:" << shaderPackDiagnosticToken(shaderPackFallbackReason_)
        << ",shaderPackGraphPasses:" << shaderPackGraphPassesExecuted_
        << ",shaderPackGraphUnsupportedPasses:" << shaderPackGraphUnsupportedPasses_
        << ",shaderPackGraphWorldPasses:" << shaderPackGraphWorldPasses_
        << ",shaderPackFallbackActive:" << (fallbackActive ? 1 : 0)
        << ",shaderPackSelectedVanillaPt:" << (selectedVanillaPt ? 1 : 0)
        << ",sbtGeometryTotal:" << sbtGeometryTotal
        << ",sbtGeometryShadow:" << sbtGeometryShadow
        << ",sbtGeometrySolid:" << sbtGeometrySolid
        << ",sbtGeometryTransparent:" << sbtGeometryTransparent
        << ",sbtGeometryOther:" << sbtGeometryOther
        << ",alphaModeExpectedOpaqueGeometry:" << sbtGeometrySolid
        << ",alphaModeExpectedTransparentGeometry:" << sbtGeometryTransparent
        << ",shaderPackCloudMode:" << cloudMode
        << ",shaderPackCloudModeEffective:" << shaderPackCloudMode_
        << ",shaderPackCloudSelectivePortActive:" << (shaderPackCloudMode_ == 2 ? 1 : 0)
        << ",shaderPackCloudBackend:" << diagnosticBackendFromMode(cloudMode)
        << ",shaderPackCloudShadowsEnabled:" << (shaderPackCloudShadowsEnabled_ ? 1 : 0)
        << ",shaderPackWaterSurfaceMode:" << waterSurfaceMode
        << ",shaderPackWaterSurfaceModeEffective:" << shaderPackWaterSurfaceMode_
        << ",shaderPackFftWaterSurfaceActive:"
        << ((shaderPackWaterSurfaceMode_ == 1 && shaderPackVisualAssetsReady) ? 1 : 0)
        << ",shaderPackWaterBackend:" << diagnosticBackendFromMode(waterSurfaceMode)
        << ",shaderPackWaterCausticsEnabled:" << (shaderPackWaterCausticsEnabled_ ? 1 : 0)
        << ",shaderPackWaterCausticsActive:"
        << ((shaderPackWaterCausticsEnabled_ && shaderPackVisualAssetsReady) ? 1 : 0)
        << ",shaderPackVolumetricLightMode:" << volumetricLightMode
        << ",shaderPackVolumetricLightModeEffective:" << shaderPackVolumetricLightMode_
        << ",shaderPackVolumetricLightSelectivePortActive:" << (shaderPackVolumetricLightMode_ == 1 ? 1 : 0)
        << ",shaderPackVolumetricLightBackend:" << diagnosticBackendFromMode(volumetricLightMode)
        << ",shaderPackCloudTemporalAccumulation:" << cloudTemporal
        << ",shaderPackCloudIndirectCapture:" << cloudIndirect
        << ",shaderPackVisualSettingsSetIndex:4"
        << ",shaderPackCloudBottomHeight:" << shaderPackVisualSettings_.volumetricCloudBottomHeight
        << ",shaderPackCloudTopHeight:" << shaderPackVisualSettings_.volumetricCloudTopHeight
        << ",shaderPackCloudBaseScale:" << shaderPackVisualSettings_.volumetricCloudBaseScale
        << ",shaderPackCloudDetailScale:" << shaderPackVisualSettings_.volumetricCloudDetailScale
        << ",shaderPackCloudCoverage:" << shaderPackVisualSettings_.volumetricCloudCoverage
        << ",shaderPackCloudDensity:" << shaderPackVisualSettings_.volumetricCloudDensity
        << ",shaderPackCloudViewSteps:" << shaderPackVisualSettings_.volumetricCloudViewSteps
        << ",shaderPackCloudLightSteps:" << shaderPackVisualSettings_.volumetricCloudLightSteps
        << ",shaderPackCloudAmbientSteps:" << shaderPackVisualSettings_.volumetricCloudAmbientSteps
        << ",shaderPackCloudShadowSoftness:" << shaderPackVisualSettings_.volumetricCloudShadowSoftness
        << ",shaderPackCloudAmbientStrength:" << shaderPackVisualSettings_.volumetricCloudAmbientStrength
        << ",shaderPackCloudPowderStrength:" << shaderPackVisualSettings_.volumetricCloudPowderStrength
        << ",shaderPackCloudWeatherScale:" << shaderPackVisualSettings_.volumetricCloudWeatherScale
        << ",shaderPackVolumetricLightSamples:" << shaderPackVisualSettings_.volumetricLightSamples
        << ",shaderPackVolumetricLightScatteringStrength:" << shaderPackVisualSettings_.volumetricLightScatteringStrength
        << ",shaderPackVolumetricLightMaxDistance:" << shaderPackVisualSettings_.volumetricLightMaxDistance
        << ",shaderPackVolumetricLightNearStepSize:" << shaderPackVisualSettings_.volumetricLightNearStepSize
        << ",shaderPackVolumetricLightFarStepSize:" << shaderPackVisualSettings_.volumetricLightFarStepSize
        << ",shaderPackVolumetricLightLuminanceLimit:" << shaderPackVisualSettings_.volumetricLightLuminanceLimit
        << ",forkNativeCloudModuleRegistered:0"
        << ",rtOutputContractExpected:29"
        << ",rtOutputResourceFamiliesReady:" << outputFamiliesReady
        << ",rtOutputContractComplete:" << (outputFamiliesReady == 29 ? 1 : 0)
        << ",dlssGuideInputContractExpected:24"
        << ",dlssGuideOptionalFamiliesExpected:15"
        << ",dlssGuideOptionalFamiliesReady:" << dlssGuideFamiliesReady
        << ",dlssGuideOptionalContractComplete:" << (dlssGuideFamiliesReady == 15 ? 1 : 0)
        << ",vanillaPtDeclaredOutImages:15"
        << ",vanillaPtExtraOutImages:2"
        << ",vanillaPtForkOutputProducer:" << (vanillaPtWorldOutputProducerReady ? "world_shader_patched" : "not_ready")
        << ",vanillaPtDlssGuideProducer:" << (vanillaPtWorldOutputProducerReady ? "world_shader_patched" : "not_ready")
        << ",vanillaPtMissingForkOutImages:" << vanillaPtMissingForkOutImages
        << ",vanillaPtMissingDlssGuideImages:" << vanillaPtMissingDlssGuideImages
        << ",vanillaPtRequiresSynthesizedDefaults:" << ((selectedVanillaPt && !shaderPackExecutorReady_) ? 1 : 0)
        << ",vanillaPtMissingDlssGuideList:" << (vanillaPtWorldOutputProducerReady ? "none" : "diffuse_ray_dir_hit_dist|specular_ray_dir_hit_dist|reflection_mv|animated_tex_mask|particle_mask|bias_mask|rt_hit_dist|motion_vectors_3d|gbuffer_metallic|gbuffer_shading_model_id|gbuffer_material_id|position_view_space|transparency_layer|transparency_layer_opacity|transparency_layer_mvecs")
        << ",vanillaPtExtraOutImageList:fog_image|first_hit_refraction"
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

    auto ensurePackImage =
        [&](std::vector<std::shared_ptr<vk::DeviceLocalImage>> &target, VkFormat format) {
            auto &img = target[frameIndex];
            if (!img || img->width() != width || img->height() != height || img->vkFormat() != format) {
                img = vk::DeviceLocalImage::create(
                    framework->device(), framework->vma(), false, width, height, 1, format,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            }
        };
    ensurePackImage(upstreamFogImages_, VK_FORMAT_R16G16B16A16_SFLOAT);
    ensurePackImage(upstreamFirstHitRefractionImages_, VK_FORMAT_R16G16B16A16_SFLOAT);


    // Publish depth and motion vectors for frame generation resource tagging
    if (Renderer::frameGenDepthImages.size() <= frameIndex) {
        Renderer::frameGenDepthImages.resize(frameIndex + 1);
        Renderer::frameGenMotionVectorImages.resize(frameIndex + 1);
    }
    Renderer::frameGenDepthImages[frameIndex] = linearDepthImages_[frameIndex];
    Renderer::frameGenMotionVectorImages[frameIndex] = motionVectorImages_[frameIndex];

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
    shaderPackAttributeKVs_ = attributeKVs;
    shaderPackVisualSettings_ = ShaderPackVisualSettings{};

    for (int i = 0; i < attributeCount; i++) {
        const std::string &key = attributeKVs[2 * i];
        const std::string &value = attributeKVs[2 * i + 1];

        if (key == "render_pipeline.module.ray_tracing.attribute.num_ray_bounces" ||
            key == "render_pipeline.module.dlss.attribute.num_ray_bounces") {
            numRayBounces_ = std::stoi(value);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.use_jitter") {
            useJitter_ = shaderPackBoolValue(value, useJitter_);
            Renderer::instance().buffers()->setUseJitter(useJitter_);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.shader_pack_path") {
            shaderPackRequestedPath_ = value.empty() ? "shaders/world/ray_tracing/vanilla-pt.zip" : value;
            shaderPackPath_ = shaderPackRequestedPath_;
        } else if (key == "render_pipeline.module.ray_tracing.attribute.cloud_mode") {
            shaderPackVisualSettings_.cloudMode = shaderPackCloudModeValue(value);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.capture_volumetric_cloud_indirect") {
            shaderPackVisualSettings_.captureVolumetricCloudIndirect =
                shaderPackBoolValue(value, shaderPackVisualSettings_.captureVolumetricCloudIndirect != 0) ? 1 : 0;
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_temporal_accumulation") {
            shaderPackVisualSettings_.volumetricCloudTemporalAccumulation =
                shaderPackBoolValue(value, shaderPackVisualSettings_.volumetricCloudTemporalAccumulation != 0) ? 1 : 0;
        } else if (key == "render_pipeline.module.ray_tracing.attribute.indirect_volumetric_cloud_view_steps") {
            shaderPackVisualSettings_.indirectVolumetricCloudViewSteps =
                shaderPackIntValue(value, shaderPackVisualSettings_.indirectVolumetricCloudViewSteps, 1, 256);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.indirect_volumetric_cloud_light_steps") {
            shaderPackVisualSettings_.indirectVolumetricCloudLightSteps =
                shaderPackIntValue(value, shaderPackVisualSettings_.indirectVolumetricCloudLightSteps, 1, 64);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.indirect_volumetric_cloud_ambient_steps") {
            shaderPackVisualSettings_.indirectVolumetricCloudAmbientSteps =
                shaderPackIntValue(value, shaderPackVisualSettings_.indirectVolumetricCloudAmbientSteps, 1, 64);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.indirect_volumetric_cloud_reflection_max_roughness") {
            shaderPackVisualSettings_.indirectVolumetricCloudReflectionMaxRoughness =
                shaderPackFloatValue(value, shaderPackVisualSettings_.indirectVolumetricCloudReflectionMaxRoughness, 0.0f, 1.0f);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_bottom_height") {
            shaderPackVisualSettings_.volumetricCloudBottomHeight =
                shaderPackFloatValue(value, shaderPackVisualSettings_.volumetricCloudBottomHeight, -1024.0f, 20000.0f);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_top_height") {
            shaderPackVisualSettings_.volumetricCloudTopHeight =
                shaderPackFloatValue(value, shaderPackVisualSettings_.volumetricCloudTopHeight, -1024.0f, 20000.0f);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_base_scale") {
            shaderPackVisualSettings_.volumetricCloudBaseScale =
                shaderPackFloatValue(value, shaderPackVisualSettings_.volumetricCloudBaseScale, 0.001f, 8.0f);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_detail_scale") {
            shaderPackVisualSettings_.volumetricCloudDetailScale =
                shaderPackFloatValue(value, shaderPackVisualSettings_.volumetricCloudDetailScale, 0.001f, 8.0f);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_coverage") {
            shaderPackVisualSettings_.volumetricCloudCoverage =
                shaderPackFloatValue(value, shaderPackVisualSettings_.volumetricCloudCoverage, 0.0f, 1.0f);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_clear_amount") {
            shaderPackVisualSettings_.volumetricCloudClearAmount = shaderPackClearAmountValue(value);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_density") {
            shaderPackVisualSettings_.volumetricCloudDensity =
                shaderPackFloatValue(value, shaderPackVisualSettings_.volumetricCloudDensity, 0.0f, 8.0f);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_view_steps") {
            shaderPackVisualSettings_.volumetricCloudViewSteps =
                shaderPackIntValue(value, shaderPackVisualSettings_.volumetricCloudViewSteps, 1, 256);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_light_steps") {
            shaderPackVisualSettings_.volumetricCloudLightSteps =
                shaderPackIntValue(value, shaderPackVisualSettings_.volumetricCloudLightSteps, 1, 64);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_cast_shadow") {
            shaderPackVisualSettings_.volumetricCloudCastShadow =
                shaderPackBoolValue(value, shaderPackVisualSettings_.volumetricCloudCastShadow != 0) ? 1 : 0;
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_shadow_softness") {
            shaderPackVisualSettings_.volumetricCloudShadowSoftness =
                shaderPackFloatValue(value, shaderPackVisualSettings_.volumetricCloudShadowSoftness, 0.0f, 1.0f);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_ambient_steps") {
            shaderPackVisualSettings_.volumetricCloudAmbientSteps =
                shaderPackIntValue(value, shaderPackVisualSettings_.volumetricCloudAmbientSteps, 1, 64);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_ambient_strength") {
            shaderPackVisualSettings_.volumetricCloudAmbientStrength =
                shaderPackFloatValue(value, shaderPackVisualSettings_.volumetricCloudAmbientStrength, 0.0f, 8.0f);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_powder_strength") {
            shaderPackVisualSettings_.volumetricCloudPowderStrength =
                shaderPackFloatValue(value, shaderPackVisualSettings_.volumetricCloudPowderStrength, 0.0f, 2.0f);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_cloud_weather_scale") {
            shaderPackVisualSettings_.volumetricCloudWeatherScale =
                shaderPackFloatValue(value, shaderPackVisualSettings_.volumetricCloudWeatherScale, 0.0001f, 1.0f);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.water_surface_mode") {
            shaderPackVisualSettings_.waterSurfaceMode = shaderPackWaterSurfaceModeValue(value);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.water_caustics_enabled") {
            shaderPackVisualSettings_.waterCausticsEnabled =
                shaderPackBoolValue(value, shaderPackVisualSettings_.waterCausticsEnabled != 0) ? 1 : 0;
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_light_mode") {
            shaderPackVisualSettings_.volumetricLightMode = shaderPackVolumetricLightModeValue(value);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_light_samples") {
            shaderPackVisualSettings_.volumetricLightSamples =
                shaderPackIntValue(value, shaderPackVisualSettings_.volumetricLightSamples, 1, 256);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_light_scattering_strength") {
            shaderPackVisualSettings_.volumetricLightScatteringStrength =
                shaderPackFloatValue(value, shaderPackVisualSettings_.volumetricLightScatteringStrength, 0.0f, 32.0f);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_light_max_distance") {
            shaderPackVisualSettings_.volumetricLightMaxDistance =
                shaderPackFloatValue(value, shaderPackVisualSettings_.volumetricLightMaxDistance, 0.0f, 2048.0f);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_light_near_step_size") {
            shaderPackVisualSettings_.volumetricLightNearStepSize =
                shaderPackFloatValue(value, shaderPackVisualSettings_.volumetricLightNearStepSize, 0.05f, 64.0f);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_light_far_step_size") {
            shaderPackVisualSettings_.volumetricLightFarStepSize =
                shaderPackFloatValue(value, shaderPackVisualSettings_.volumetricLightFarStepSize, 0.05f, 128.0f);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.volumetric_light_luminance_limit") {
            shaderPackVisualSettings_.volumetricLightLuminanceLimit =
                shaderPackFloatValue(value, shaderPackVisualSettings_.volumetricLightLuminanceLimit, 0.0f, 128.0f);
        }
    }

    if (shaderPackVisualSettings_.cloudMode != 2) {
        shaderPackVisualSettings_.volumetricCloudCastShadow = 0;
    }

    if (shaderPackVisualSettings_.volumetricCloudTopHeight <= shaderPackVisualSettings_.volumetricCloudBottomHeight) {
        shaderPackVisualSettings_.volumetricCloudTopHeight =
            shaderPackVisualSettings_.volumetricCloudBottomHeight + 32.0f;
    }

    shaderPackCloudMode_ = shaderPackVisualSettings_.cloudMode;
    shaderPackWaterSurfaceMode_ = shaderPackVisualSettings_.waterSurfaceMode;
    shaderPackVolumetricLightMode_ = shaderPackVisualSettings_.volumetricLightMode;
    shaderPackCloudShadowsEnabled_ = shaderPackVisualSettings_.volumetricCloudCastShadow != 0;
    shaderPackWaterCausticsEnabled_ = shaderPackVisualSettings_.waterCausticsEnabled != 0;
}

void RayTracingModule::validateShaderPackRuntime() {
    shaderPack_.reset();
    shaderPackRayTracingVariables_.clear();
    shaderPackUnsupportedPassesLogged_.clear();
    shaderPackRuntimeResourcesReady_ = false;
    shaderPackBackendEnabled_ = false;
    shaderPackFirstFrameAttempted_ = false;
    shaderPackFallbackReason_.clear();
    shaderPackBackendStatus_ = "validating";
    shaderPackGraphPassesExecuted_ = 0;
    shaderPackGraphUnsupportedPasses_ = 0;
    shaderPackGraphWorldPasses_ = 0;
    shaderPackExecutorReady_ = false;
    shaderPackIntermediatesGenerated_ = 0;
    shaderPackFullScreenPassDispatches_ = 0;
    shaderPackComputePassDispatches_ = 0;
    shaderPackRayTracingPassDispatches_ = 0;
    shaderPackPassDispatchCounts_.clear();
    shaderPackFullScreenPasses_.clear();
    shaderPackComputePasses_.clear();
    shaderPackRayTracingPasses_.clear();
    shaderPackFullScreenVertexShader_.reset();
    shaderPackRuntimeValid_ = false;
    shaderPackRuntimeStatus_ = "missing_framework";

    auto framework = framework_.lock();
    if (!framework) {
        shaderPackBackendStatus_ = "missing_framework";
        return;
    }

    ShaderPack::BuildConfig config = ShaderPack::buildConfigFromRayTracingAttributes(shaderPackAttributeKVs_);
    if (shaderPackRequestedPath_.empty()) {
        shaderPackRequestedPath_ = shaderPackPath_;
    }
    if (config.shaderPackPath.empty()) {
        config.shaderPackPath = shaderPackRequestedPath_;
    }
    config.staticAttributes["render_pipeline.module.ray_tracing.attribute.shader_pack_path"] =
        config.shaderPackPath;

    auto acceptShaderPack =
        [&](std::shared_ptr<ShaderPack> selectedPack,
            const std::string &runtimeStatus,
            const std::string &selectedPath) {
            shaderPack_ = std::move(selectedPack);
            shaderPackRuntimeValid_ = true;
            shaderPackRuntimeStatus_ = runtimeStatus;
            shaderPackPath_ = selectedPath.empty()
                ? "shaders/world/ray_tracing/vanilla-pt.zip"
                : selectedPath;
            shaderPackBackendEnabled_ = true;
            shaderPackBackendStatus_ = "adapter_validated";
            shaderPackFallbackReason_.clear();

            std::unordered_map<std::string, ShaderPackLoader::VariableConfig> variableConfigs;
            std::unordered_map<std::string, std::string> globalVariables;
            shaderPack_->copyStageExecutionState(ShaderPackLoader::Stage::RayTracing, variableConfigs, globalVariables);

            for (const auto &[name, config] : variableConfigs) {
                shaderPackRayTracingVariables_[name] = {
                    .name = name,
                    .value = config.defaultValue,
                    .type = config.type,
                };
            }
            for (const auto &[name, value] : globalVariables) {
                if (shaderPackRayTracingVariables_.find(name) != shaderPackRayTracingVariables_.end()) { continue; }
                shaderPackRayTracingVariables_[name] = {
                    .name = name,
                    .value = value,
                    .type = shaderPackVariableTypeFromValue(value),
                };
            }

            bool hasWorldRayTracingPass = false;
            size_t rayTracingPassCount = 0;
            size_t computePassCount = 0;
            size_t fullScreenPassCount = 0;
            size_t renderPassCount = 0;
            for (const auto &pass : shaderPack_->shaderPack().passes) {
                if (pass.stage != ShaderPackLoader::Stage::RayTracing) { continue; }
                switch (pass.type) {
                    case ShaderPackLoader::PassConfig::Type::RayTracing:
                        rayTracingPassCount++;
                        if (pass.rayTracing.name == "world") { hasWorldRayTracingPass = true; }
                        break;
                    case ShaderPackLoader::PassConfig::Type::Compute:
                        computePassCount++;
                        break;
                    case ShaderPackLoader::PassConfig::Type::FullScreen:
                        fullScreenPassCount++;
                        break;
                    case ShaderPackLoader::PassConfig::Type::Render:
                        renderPassCount++;
                        break;
                }
            }

            if (!hasWorldRayTracingPass) {
                disableShaderPackBackend("fallback_missing_world_pass",
                                         "selected shader pack has no ray tracing world pass");
            }

            renderDiag("ShaderPack adapter validated status=%s requested=%s root=%s rtPasses=%zu computePasses=%zu fullScreenPasses=%zu renderPasses=%zu variables=%zu",
                       shaderPackRuntimeStatus_.c_str(),
                       shaderPackPath_.c_str(),
                       shaderPack_->shaderPack().rootPath.string().c_str(),
                       rayTracingPassCount,
                       computePassCount,
                       fullScreenPassCount,
                       renderPassCount,
                       shaderPackRayTracingVariables_.size());
            RadianceLogger::log("RayTracing", "INFO",
                                "Shader pack adapter validated status=%s requested=%s root=%s rtPasses=%zu computePasses=%zu fullScreenPasses=%zu renderPasses=%zu variables=%zu",
                                shaderPackRuntimeStatus_.c_str(),
                                shaderPackPath_.c_str(),
                                shaderPack_->shaderPack().rootPath.string().c_str(),
                                rayTracingPassCount,
                                computePassCount,
                                fullScreenPassCount,
                                renderPassCount,
                                shaderPackRayTracingVariables_.size());
        };

    auto shaderPack = std::make_shared<ShaderPack>(framework);
    std::string error;
    if (shaderPack->initialize(config, error)) {
        acceptShaderPack(shaderPack, "validated", config.shaderPackPath);
        return;
    }

    RadianceLogger::log("RayTracing", "WARN",
                        "Shader pack validation failed requested=%s error=%s; trying built-in fallback",
                        config.shaderPackPath.c_str(),
                        error.c_str());

    config.shaderPackPath.clear();
    error.clear();
    shaderPack = std::make_shared<ShaderPack>(framework);
    if (shaderPack->initialize(config, error)) {
        acceptShaderPack(shaderPack, "fallback", "shaders/world/ray_tracing/vanilla-pt.zip");
        return;
    }

    shaderPackRuntimeStatus_ = "invalid";
    shaderPackBackendStatus_ = "fallback_validation_failed";
    shaderPackFallbackReason_ = error;
    renderDiag("ShaderPack fallback failed error=%s", error.c_str());
    RadianceLogger::log("RayTracing", "ERROR",
                        "Shader pack fallback failed error=%s",
                        error.c_str());
}

void RayTracingModule::disableShaderPackBackend(const std::string &status, const std::string &reason) {
    shaderPackBackendEnabled_ = false;
    shaderPackRuntimeResourcesReady_ = false;
    shaderPackBackendStatus_ = status.empty() ? "fallback" : status;
    shaderPackFallbackReason_ = reason.empty() ? shaderPackBackendStatus_ : reason;

    renderDiag("ShaderPack backend disabled status=%s reason=%s",
               shaderPackBackendStatus_.c_str(),
               shaderPackFallbackReason_.c_str());
    RadianceLogger::log("RayTracing", "WARN",
                        "Shader pack backend disabled status=%s reason=%s",
                        shaderPackBackendStatus_.c_str(),
                        shaderPackFallbackReason_.c_str());
}

void RayTracingModule::initShaderPackRuntimeResources() {
    if (!shaderPackBackendEnabled_ || !shaderPack_) { return; }
    if (hdrNoisyOutputImages_.empty() || !hdrNoisyOutputImages_[0]) {
        disableShaderPackBackend("fallback_no_reference_output",
                                 "ray tracing outputs are unavailable for shader pack runtime resources");
        return;
    }

    const uint32_t referenceWidth = hdrNoisyOutputImages_[0]->width();
    const uint32_t referenceHeight = hdrNoisyOutputImages_[0]->height();
    try {
        shaderPack_->setRuntimeResourceExpressionVariables({
            {.name = "RENDER_WIDTH", .value = static_cast<double>(referenceWidth)},
            {.name = "RENDER_HEIGHT", .value = static_cast<double>(referenceHeight)},
        });
        shaderPack_->ensureRuntimeResources(referenceWidth, referenceHeight);
        auto resetExecutionVariable = [&](const std::string &name, const std::string &value) {
            auto iter = shaderPackRayTracingVariables_.find(name);
            if (iter != shaderPackRayTracingVariables_.end()) { iter->second.value = value; }
        };
        resetExecutionVariable("VPT_TRANS_LUT_READY", "false");
        resetExecutionVariable("VPT_VOLUMETRIC_CLOUD_HISTORY_READY", "false");
        resetExecutionVariable("VPT_VOLUMETRIC_CLOUD_HISTORY_PING_INPUT", "true");
        shaderPackRuntimeResourcesReady_ = shaderPack_->runtimeResourcesReady() || !shaderPack_->hasRuntimeResources();
        if (shaderPackRuntimeResourcesReady_ && shaderPack_->hasRuntimeResources()) {
            for (uint32_t frameIndex = 0; frameIndex < rayTracingDescriptorTables_.size(); ++frameIndex) {
                shaderPack_->bindRuntimeResources(rayTracingDescriptorTables_[frameIndex], 5, frameIndex);
            }
        }
        if (shaderPackRuntimeResourcesReady_ && shaderPackBackendStatus_ == "adapter_validated") {
            shaderPackBackendStatus_ = "adapter_ready";
        }

        const bool hasRuntimeResources = shaderPack_->hasRuntimeResources();
        const bool fftWaterResourcesBound = shaderPackRuntimeResourcesReady_ && hasRuntimeResources;
        renderDiag("ShaderPack runtime resources status=%s ready=%d hasResources=%d reference=%ux%u waterMode=%d waterCaustics=%d fftWaterBound=%d",
                   shaderPackBackendStatus_.c_str(),
                   shaderPackRuntimeResourcesReady_ ? 1 : 0,
                   hasRuntimeResources ? 1 : 0,
                   referenceWidth,
                   referenceHeight,
                   shaderPackWaterSurfaceMode_,
                   shaderPackWaterCausticsEnabled_ ? 1 : 0,
                   fftWaterResourcesBound ? 1 : 0);
        RadianceLogger::log("RayTracing", "INFO",
                            "Shader pack runtime resources status=%s ready=%d hasResources=%d reference=%ux%u waterMode=%d waterCaustics=%d fftWaterBound=%d",
                            shaderPackBackendStatus_.c_str(),
                            shaderPackRuntimeResourcesReady_ ? 1 : 0,
                            hasRuntimeResources ? 1 : 0,
                            referenceWidth,
                            referenceHeight,
                            shaderPackWaterSurfaceMode_,
                            shaderPackWaterCausticsEnabled_ ? 1 : 0,
                            fftWaterResourcesBound ? 1 : 0);
    } catch (const std::exception &e) {
        disableShaderPackBackend("fallback_runtime_resource_failure", e.what());
    } catch (...) {
        disableShaderPackBackend("fallback_runtime_resource_failure", "unknown exception");
    }
}

void RayTracingModule::initShaderPackRayTracingExecutor() {
    shaderPackExecutorReady_ = false;
    shaderPackFullScreenPasses_.clear();
    shaderPackComputePasses_.clear();
    shaderPackRayTracingPasses_.clear();
    shaderPackFullScreenVertexShader_.reset();

    if (!shaderPackBackendEnabled_ || !shaderPack_ || !shaderPackRuntimeResourcesReady_) { return; }

    auto framework = framework_.lock();
    if (!framework) {
        disableShaderPackBackend("fallback_missing_framework", "framework unavailable while building shader-pack executor");
        return;
    }

    const uint32_t frameCount = framework->swapchain()->imageCount();
    const auto device = framework->device();
    const auto vma = framework->vma();
    const size_t executionBufferSize =
        shaderPack_->execution(ShaderPackLoader::Stage::RayTracing).variables.size() * sizeof(float);
    shaderPackRuntimeResourceSetIndex_ = 5;
    shaderPackExecutionSetIndex_ = shaderPack_->executionSet(shaderPackRuntimeResourceSetIndex_);
    if (shaderPackExecutionSetIndex_ != 6) {
        disableShaderPackBackend("fallback_execution_set_mismatch",
                                 "ray tracing descriptor table expects shader-pack execution set 6");
        return;
    }

    try {
        const std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
        shaderPackFullScreenVertexShader_ =
            vk::Shader::create(device, (shaderPath / "full_screen_vert.spv").string());

        auto findTargetTexture = [&](const std::string &target) -> ShaderPack::RuntimeTexture & {
            auto texture = shaderPack_->findRuntimeTexture(target);
            if (!texture.has_value()) {
                throw std::runtime_error("shader-pack pass target texture is missing: " + target);
            }
            return texture->get();
        };

        auto buildColorRenderPass =
            [&](const std::shared_ptr<vk::DeviceLocalImage> &targetImage) {
                return vk::RenderPassBuilder{}
                    .beginAttachmentDescription()
                    .defineAttachmentDescription({
                        .format = targetImage->vkFormat(),
                        .samples = VK_SAMPLE_COUNT_1_BIT,
                        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                        .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    })
                    .endAttachmentDescription()
                    .beginAttachmentReference()
                    .defineAttachmentReference({
                        .attachment = 0,
                        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    })
                    .endAttachmentReference()
                    .beginSubpassDescription()
                    .defineSubpassDescription({
                        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                        .colorAttachmentIndices = {0},
                    })
                    .endSubpassDescription()
                    .build(device);
            };

        for (const auto &passConfig : shaderPack_->shaderPack().passes) {
            if (passConfig.stage != ShaderPackLoader::Stage::RayTracing) { continue; }

            if (passConfig.type == ShaderPackLoader::PassConfig::Type::FullScreen) {
                FullScreenPass pass;
                pass.config = passConfig.fullScreen;
                pass.executionBuffer = ShaderPack::createPassExecutionBuffer(device, vma, executionBufferSize);

                auto &targetTexture = findTargetTexture(pass.config.target);
                auto targetImage = shaderPack_->findRuntimeVKTexture(targetTexture, 0);
                pass.renderPass = buildColorRenderPass(targetImage);

                uint32_t layerCount = 1;
                uint32_t firstLayer = 0;
                if (pass.config.face.has_value()) {
                    firstLayer = *pass.config.face;
                } else if (pass.config.layer.has_value()) {
                    firstLayer = *pass.config.layer;
                } else if (targetTexture.config.dimension == ShaderPackLoader::TextureDimension::Cube) {
                    layerCount = 6;
                }

                pass.fragmentShaders.resize(layerCount);
                pass.pipelines.resize(layerCount);
                pass.framebuffers.resize(frameCount);

                for (uint32_t layerOffset = 0; layerOffset < layerCount; ++layerOffset) {
                    const uint32_t layer = firstLayer + layerOffset;
                    auto definitions = pass.config.definitions;
                    if (targetTexture.config.dimension == ShaderPackLoader::TextureDimension::Cube ||
                        pass.config.face.has_value()) {
                        definitions["FACE"] = std::to_string(layer);
                    }
                    if (pass.config.layer.has_value()) {
                        definitions["LAYER"] = std::to_string(layer);
                    }

                    pass.fragmentShaders[layerOffset] = shaderPack_->createShader(
                        device,
                        pass.config.fragmentShaderPath,
                        VK_SHADER_STAGE_FRAGMENT_BIT,
                        definitions,
                        ShaderPackLoader::Stage::RayTracing,
                        shaderPackExecutionSetIndex_);

                    pass.pipelines[layerOffset] =
                        vk::GraphicsPipelineBuilder{}
                            .defineRenderPass(pass.renderPass, 0)
                            .beginShaderStage()
                            .defineShaderStage(shaderPackFullScreenVertexShader_, VK_SHADER_STAGE_VERTEX_BIT)
                            .defineShaderStage(pass.fragmentShaders[layerOffset], VK_SHADER_STAGE_FRAGMENT_BIT)
                            .endShaderStage()
                            .defineVertexInputState<void>()
                            .defineViewportScissorState({
                                .viewport =
                                    {
                                        .x = 0,
                                        .y = 0,
                                        .width = static_cast<float>(targetImage->width()),
                                        .height = static_cast<float>(targetImage->height()),
                                        .minDepth = 0.0f,
                                        .maxDepth = 1.0f,
                                    },
                                .scissor =
                                    {
                                        .offset = {.x = 0, .y = 0},
                                        .extent = {.width = targetImage->width(), .height = targetImage->height()},
                                    },
                            })
                            .beginColorBlendAttachmentState()
                            .defineDefaultColorBlendAttachmentState()
                            .endColorBlendAttachmentState()
                            .definePipelineLayout(rayTracingDescriptorTables_[0])
                            .build(device);
                }

                for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
                    auto frameImage = shaderPack_->findRuntimeVKTexture(targetTexture, frameIndex);
                    pass.framebuffers[frameIndex].resize(layerCount);
                    for (uint32_t layerOffset = 0; layerOffset < layerCount; ++layerOffset) {
                        const uint32_t layer = firstLayer + layerOffset;
                        uint32_t viewIndex = 0;
                        if (targetTexture.config.dimension == ShaderPackLoader::TextureDimension::Cube ||
                            targetTexture.config.dimension == ShaderPackLoader::TextureDimension::Texture2DArray ||
                            pass.config.face.has_value() || pass.config.layer.has_value()) {
                            viewIndex = shaderPack_->runtimeTextureSingleLayerViewIndex(targetTexture, layer);
                        }
                        pass.framebuffers[frameIndex][layerOffset] =
                            vk::FramebufferBuilder{}
                                .beginAttachment()
                                .defineAttachment(frameImage, static_cast<int>(viewIndex))
                                .endAttachment()
                                .build(device, pass.renderPass);
                    }
                }

                shaderPackFullScreenPasses_[pass.config.name] = std::move(pass);
                continue;
            }

            if (passConfig.type == ShaderPackLoader::PassConfig::Type::Compute) {
                ComputePass pass;
                pass.config = passConfig.compute;
                pass.executionBuffer = ShaderPack::createPassExecutionBuffer(device, vma, executionBufferSize);
                pass.computeShader = shaderPack_->createShader(
                    device,
                    pass.config.computeShaderPath,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    pass.config.definitions,
                    ShaderPackLoader::Stage::RayTracing,
                    shaderPackExecutionSetIndex_);
                pass.pipeline =
                    vk::ComputePipelineBuilder{}
                        .defineShader(pass.computeShader)
                        .definePipelineLayout(rayTracingDescriptorTables_[0])
                        .build(device);
                shaderPackComputePasses_[pass.config.name] = std::move(pass);
                continue;
            }

            if (passConfig.type == ShaderPackLoader::PassConfig::Type::RayTracing) {
                RayTracingPass pass;
                pass.config = passConfig.rayTracing;
                pass.executionBuffer = ShaderPack::createPassExecutionBuffer(device, vma, executionBufferSize);

                vk::RayTracingPipelineBuilder pipelineBuilder;
                auto &stageBuilder = pipelineBuilder.beginShaderStage();
                auto &groupBuilder = pipelineBuilder.beginShaderGroup();

                uint32_t stageIndex = 0;
                pass.rayGenUpdateShader = shaderPack_->createShader(
                    device,
                    pass.config.rayGenShaderPath,
                    VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                    pass.config.definitions,
                    ShaderPackLoader::Stage::RayTracing,
                    shaderPackExecutionSetIndex_);
                stageBuilder.defineShaderStage(pass.rayGenUpdateShader, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
                groupBuilder.defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                                               stageIndex++,
                                               VK_SHADER_UNUSED_KHR,
                                               VK_SHADER_UNUSED_KHR,
                                               VK_SHADER_UNUSED_KHR);

                for (const auto &missConfig : pass.config.missShaders) {
                    MissShader miss;
                    miss.name = missConfig.name;
                    miss.index = static_cast<uint32_t>(pass.missShaders.size());
                    miss.shader = shaderPack_->createShader(
                        device,
                        missConfig.shaderPath,
                        VK_SHADER_STAGE_MISS_BIT_KHR,
                        pass.config.definitions,
                        ShaderPackLoader::Stage::RayTracing,
                        shaderPackExecutionSetIndex_);
                    stageBuilder.defineShaderStage(miss.shader, VK_SHADER_STAGE_MISS_BIT_KHR);
                    groupBuilder.defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                                                   stageIndex++,
                                                   VK_SHADER_UNUSED_KHR,
                                                   VK_SHADER_UNUSED_KHR,
                                                   VK_SHADER_UNUSED_KHR);
                    pass.missShaders.push_back(std::move(miss));
                }

                for (const auto &hitConfig : pass.config.hitGroups) {
                    HitShaderGroup hitGroup;
                    hitGroup.name = hitConfig.name;
                    hitGroup.type = hitConfig.type;
                    uint32_t closestStage = VK_SHADER_UNUSED_KHR;
                    uint32_t anyStage = VK_SHADER_UNUSED_KHR;
                    uint32_t intersectionStage = VK_SHADER_UNUSED_KHR;

                    if (hitConfig.closestHit.has_value()) {
                        hitGroup.closestHitShader = shaderPack_->createShader(
                            device,
                            *hitConfig.closestHit,
                            VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                            pass.config.definitions,
                            ShaderPackLoader::Stage::RayTracing,
                            shaderPackExecutionSetIndex_);
                        stageBuilder.defineShaderStage(hitGroup.closestHitShader, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
                        closestStage = stageIndex++;
                    }
                    if (hitConfig.anyHit.has_value()) {
                        hitGroup.anyHitShader = shaderPack_->createShader(
                            device,
                            *hitConfig.anyHit,
                            VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                            pass.config.definitions,
                            ShaderPackLoader::Stage::RayTracing,
                            shaderPackExecutionSetIndex_);
                        stageBuilder.defineShaderStage(hitGroup.anyHitShader, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
                        anyStage = stageIndex++;
                    }
                    if (hitConfig.intersection.has_value()) {
                        hitGroup.intersectionShader = shaderPack_->createShader(
                            device,
                            *hitConfig.intersection,
                            VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                            pass.config.definitions,
                            ShaderPackLoader::Stage::RayTracing,
                            shaderPackExecutionSetIndex_);
                        stageBuilder.defineShaderStage(hitGroup.intersectionShader,
                                                       VK_SHADER_STAGE_INTERSECTION_BIT_KHR);
                        intersectionStage = stageIndex++;
                    }

                    pass.hitGroupNameToIndex[hitGroup.name] =
                        static_cast<uint32_t>(pass.hitShaderGroups.size());
                    groupBuilder.defineShaderGroup(hitGroup.type,
                                                   VK_SHADER_UNUSED_KHR,
                                                   closestStage,
                                                   anyStage,
                                                   intersectionStage);
                    pass.hitShaderGroups.push_back(std::move(hitGroup));
                }

                stageBuilder.endShaderStage();
                groupBuilder.endShaderGroup();
                pass.missGroupCount = static_cast<uint32_t>(pass.missShaders.size());
                pass.hitGroupCount = static_cast<uint32_t>(pass.hitShaderGroups.size());

                auto fallbackIter = pass.hitGroupNameToIndex.find(pass.config.defaultHitGroupName);
                if (fallbackIter == pass.hitGroupNameToIndex.end()) {
                    fallbackIter = pass.hitGroupNameToIndex.find("default");
                }
                if (fallbackIter == pass.hitGroupNameToIndex.end()) {
                    throw std::runtime_error("shader-pack RT pass has no default hit group: " + pass.config.name);
                }
                pass.fallbackHitGroupIndex = fallbackIter->second;
                auto shadowIter = pass.hitGroupNameToIndex.find("shadow");
                pass.shadowHitGroupIndex = shadowIter == pass.hitGroupNameToIndex.end()
                    ? pass.fallbackHitGroupIndex
                    : shadowIter->second;

                pass.updatePipeline =
                    pipelineBuilder
                        .definePipelineLayout(rayTracingDescriptorTables_[0])
                        .build(device);
                pass.updateSbts.resize(frameCount);
                for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
                    pass.updateSbts[frameIndex] =
                        vk::SBT::create(framework->physicalDevice(), device, vma,
                                        pass.updatePipeline, pass.missGroupCount, pass.hitGroupCount);
                }

                shaderPackRayTracingPasses_[pass.config.name] = std::move(pass);
            }
        }

        const bool hasExpectedPath =
            shaderPackFullScreenPasses_.find("trans_lut") != shaderPackFullScreenPasses_.end() &&
            shaderPackFullScreenPasses_.find("sky_cube") != shaderPackFullScreenPasses_.end() &&
            shaderPackRayTracingPasses_.find("world") != shaderPackRayTracingPasses_.end() &&
            shaderPackComputePasses_.find("volumetric_cloud_temporal") != shaderPackComputePasses_.end() &&
            shaderPackComputePasses_.find("volumetric_cloud_apply") != shaderPackComputePasses_.end();
        if (!hasExpectedPath) {
            disableShaderPackBackend("fallback_executor_missing_expected_pass",
                                     "vanilla-pt executor did not build every required ray-tracing-stage pass");
            return;
        }

        shaderPackExecutorReady_ = true;
        shaderPackBackendStatus_ = "executor_ready";
        shaderPackFallbackReason_.clear();
        renderDiag("ShaderPack executor built fullScreen=%zu compute=%zu rayTracing=%zu executionSet=%u",
                   shaderPackFullScreenPasses_.size(),
                   shaderPackComputePasses_.size(),
                   shaderPackRayTracingPasses_.size(),
                   shaderPackExecutionSetIndex_);
        RadianceLogger::log("RayTracing", "INFO",
                            "Shader pack executor built fullScreen=%zu compute=%zu rayTracing=%zu executionSet=%u",
                            shaderPackFullScreenPasses_.size(),
                            shaderPackComputePasses_.size(),
                            shaderPackRayTracingPasses_.size(),
                            shaderPackExecutionSetIndex_);
    } catch (const std::exception &e) {
        disableShaderPackBackend("fallback_executor_build_failure", e.what());
    } catch (...) {
        disableShaderPackBackend("fallback_executor_build_failure", "unknown exception");
    }
}

bool RayTracingModule::recordShaderPackRayTracingGraph(uint32_t width, uint32_t height, bool &worldPassRequested) {
    worldPassRequested = false;
    if (!shaderPackBackendEnabled_ || !shaderPack_) { return false; }

    const bool firstFrame = !shaderPackFirstFrameAttempted_;
    shaderPackFirstFrameAttempted_ = true;

    uint32_t localPasses = 0;
    uint32_t localUnsupportedPasses = 0;
    uint32_t localWorldPasses = 0;

    auto passKind = [&](const std::string &passName) {
        for (const auto &pass : shaderPack_->shaderPack().passes) {
            if (pass.stage != ShaderPackLoader::Stage::RayTracing) { continue; }
            switch (pass.type) {
                case ShaderPackLoader::PassConfig::Type::RayTracing:
                    if (pass.rayTracing.name == passName) { return std::string("ray_tracing"); }
                    break;
                case ShaderPackLoader::PassConfig::Type::Compute:
                    if (pass.compute.name == passName) { return std::string("compute"); }
                    break;
                case ShaderPackLoader::PassConfig::Type::FullScreen:
                    if (pass.fullScreen.name == passName) { return std::string("full_screen"); }
                    break;
                case ShaderPackLoader::PassConfig::Type::Render:
                    if (pass.render.name == passName) { return std::string("render"); }
                    break;
            }
        }
        return std::string("unknown");
    };

    try {
        shaderPack_->executeCommands(
            ShaderPackLoader::Stage::RayTracing,
            shaderPack_->execution(ShaderPackLoader::Stage::RayTracing).commands,
            shaderPackRayTracingVariables_,
            {
                {.name = "RENDER_WIDTH", .value = static_cast<double>(width)},
                {.name = "RENDER_HEIGHT", .value = static_cast<double>(height)},
            },
            true,
            64,
            [&](const std::string &passName, ShaderPack::ExecutionVariables &) {
                localPasses++;
                const std::string kind = passKind(passName);
                if (passName == "world" && kind == "ray_tracing") {
                    worldPassRequested = true;
                    localWorldPasses++;
                    return;
                }

                localUnsupportedPasses++;
                if (shaderPackUnsupportedPassesLogged_.insert(passName).second) {
                    renderDiag("ShaderPack graph adapter skipped unsupported pass name=%s kind=%s",
                               passName.c_str(),
                               kind.c_str());
                    RadianceLogger::log("RayTracing", "WARN",
                                        "Shader pack graph adapter skipped unsupported pass name=%s kind=%s",
                                        passName.c_str(),
                                        kind.c_str());
                }
            });
    } catch (const std::exception &e) {
        disableShaderPackBackend(firstFrame ? "fallback_first_frame_graph_failure" : "fallback_graph_failure",
                                 e.what());
        return false;
    } catch (...) {
        disableShaderPackBackend(firstFrame ? "fallback_first_frame_graph_failure" : "fallback_graph_failure",
                                 "unknown exception");
        return false;
    }

    shaderPackGraphPassesExecuted_ += localPasses;
    shaderPackGraphUnsupportedPasses_ += localUnsupportedPasses;
    shaderPackGraphWorldPasses_ += localWorldPasses;

    if (!worldPassRequested) {
        shaderPackBackendStatus_ = "fallback_graph_no_world";
        shaderPackFallbackReason_ = "ray tracing execution graph did not request world pass";
        return false;
    }

    if (localUnsupportedPasses > 0) {
        shaderPackBackendStatus_ = "adapter_partial_fixed_world";
        shaderPackFallbackReason_ = "dynamic non-world shader-pack passes are not recorded yet";
    } else {
        shaderPackBackendStatus_ = "adapter_world_fixed_dispatch";
    }

    if (firstFrame) {
        renderDiag("ShaderPack graph adapter first frame status=%s passes=%u worldPasses=%u unsupported=%u",
                   shaderPackBackendStatus_.c_str(),
                   localPasses,
                   localWorldPasses,
                   localUnsupportedPasses);
        RadianceLogger::log("RayTracing", "INFO",
                            "Shader pack graph adapter first frame status=%s passes=%u worldPasses=%u unsupported=%u",
                            shaderPackBackendStatus_.c_str(),
                            localPasses,
                            localWorldPasses,
                            localUnsupportedPasses);
    }

    return true;
}

bool RayTracingModule::recordShaderPackRayTracingExecutor(
    const std::shared_ptr<vk::CommandBuffer> &commandBuffer,
    const std::shared_ptr<vk::DescriptorTable> &descriptorTable,
    const std::shared_ptr<WorldPrepareContext> &worldPrepareContext,
    const RayTracingPushConstant &pushConstant,
    uint32_t frameIndex,
    uint32_t width,
    uint32_t height) {
    if (!shaderPackBackendEnabled_ || !shaderPack_ || !shaderPackExecutorReady_) { return false; }

    const bool firstFrame = !shaderPackFirstFrameAttempted_;
    shaderPackFirstFrameAttempted_ = true;

    auto framework = framework_.lock();
    if (!framework || !commandBuffer || !descriptorTable) { return false; }
    const uint32_t queueIndex = framework->physicalDevice()->mainQueueIndex();

    auto restoreFixedBinding13 = [&]() {
        if (frameIndex < directLightDepthImages_.size() && directLightDepthImages_[frameIndex]) {
            descriptorTable->bindImage(directLightDepthImages_[frameIndex], VK_IMAGE_LAYOUT_GENERAL, 3, 13);
        }
    };

    auto bindPackFogOutputs = [&]() {
        if (frameIndex < upstreamFogImages_.size() && upstreamFogImages_[frameIndex]) {
            descriptorTable->bindImage(upstreamFogImages_[frameIndex], VK_IMAGE_LAYOUT_GENERAL, 3, 15);
        }
        if (frameIndex < upstreamFirstHitRefractionImages_.size() &&
            upstreamFirstHitRefractionImages_[frameIndex]) {
            descriptorTable->bindImage(upstreamFirstHitRefractionImages_[frameIndex],
                                       VK_IMAGE_LAYOUT_GENERAL,
                                       3,
                                       14);
        }
    };

    auto passBarrier = [&](VkPipelineStageFlags2 srcStage) {
        commandBuffer->barriersMemory({{
            .srcStageMask = srcStage,
            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        }});
    };

    auto transitionImage = [&](const std::shared_ptr<vk::DeviceLocalImage> &image,
                               VkImageLayout newLayout,
                               VkPipelineStageFlags2 dstStage,
                               VkAccessFlags2 dstAccess) {
        if (!image) { return; }
        VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkAccessFlags2 srcAccess = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        if (image->imageLayout() == VK_IMAGE_LAYOUT_UNDEFINED) {
            srcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            srcAccess = 0;
        }
        commandBuffer->barriersBufferImage(
            {},
            {{
                .srcStageMask = srcStage,
                .srcAccessMask = srcAccess,
                .dstStageMask = dstStage,
                .dstAccessMask = dstAccess,
                .oldLayout = image->imageLayout(),
                .newLayout = newLayout,
                .srcQueueFamilyIndex = queueIndex,
                .dstQueueFamilyIndex = queueIndex,
                .image = image,
                .subresourceRange = vk::wholeColorSubresourceRange,
            }});
        image->imageLayout() = newLayout;
    };

    auto evaluateExtent = [&](const std::string &expression,
                              ShaderPack::ExecutionVariables &variables,
                              const std::vector<ExpressionEvaluator::Variable> &additionalVariables) {
        const double value = shaderPack_->evaluateNumericExpression(
            ShaderPackLoader::Stage::RayTracing,
            expression,
            variables,
            additionalVariables,
            true);
        if (!std::isfinite(value) || value <= 0.0) { return 1u; }
        return std::max<uint32_t>(1, static_cast<uint32_t>(std::ceil(value)));
    };

    auto uploadExecution = [&](const std::shared_ptr<vk::DeviceLocalBuffer> &buffer,
                               ShaderPack::ExecutionVariables &variables,
                               VkPipelineStageFlags2 dstStage) {
        shaderPack_->uploadExecutionBuffer(ShaderPackLoader::Stage::RayTracing,
                                           buffer,
                                           variables,
                                           commandBuffer,
                                           descriptorTable,
                                           shaderPackExecutionSetIndex_,
                                           queueIndex,
                                           dstStage);
    };

    auto hitGroupIndex = [](RayTracingPass &pass, const std::string &name) {
        auto iter = pass.hitGroupNameToIndex.find(name);
        return iter == pass.hitGroupNameToIndex.end() ? pass.fallbackHitGroupIndex : iter->second;
    };

    uint32_t sbtRouteTotal = 0;
    uint32_t sbtRouteEntity = 0;
    uint32_t sbtRouteChunk = 0;
    uint32_t sbtRouteEntityWorldMasked = 0;
    uint32_t sbtRouteEntityNoHeight = 0;
    uint32_t sbtRouteChunkFallback = 0;
    uint32_t sbtRouteMissingSource = 0;

    auto mapGeometryTypes = [&](RayTracingPass &pass) {
        std::vector<uint32_t> mapped;
        if (worldPrepareContext && !worldPrepareContext->lastGeometryTypes_.empty()) {
            mapped.reserve(worldPrepareContext->lastGeometryTypes_.size());
            const auto &geometryMasks = worldPrepareContext->lastGeometryMasks_;
            const auto &geometrySources = worldPrepareContext->lastGeometrySources_;
            sbtRouteTotal = 0;
            sbtRouteEntity = 0;
            sbtRouteChunk = 0;
            sbtRouteEntityWorldMasked = 0;
            sbtRouteEntityNoHeight = 0;
            sbtRouteChunkFallback = 0;
            sbtRouteMissingSource = 0;
            for (size_t i = 0; i < worldPrepareContext->lastGeometryTypes_.size(); ++i) {
                uint32_t geometryType = worldPrepareContext->lastGeometryTypes_[i];
                uint32_t geometryMask = i < geometryMasks.size() ? geometryMasks[i] : 0x01u;
                bool hasSource = i < geometrySources.size();
                bool entityLikeGeometry = hasSource
                    ? geometrySources[i] == WorldPrepareContext::SBT_SOURCE_ENTITY
                    : (geometryMask & 0x01u) == 0u;
                sbtRouteTotal++;
                if (!hasSource) {
                    sbtRouteMissingSource++;
                } else if (entityLikeGeometry) {
                    sbtRouteEntity++;
                    if ((geometryMask & 0x01u) != 0u) { sbtRouteEntityWorldMasked++; }
                } else {
                    sbtRouteChunk++;
                }
                switch (static_cast<World::GeometryTypes>(geometryType)) {
                    case World::SHADOW:
                        mapped.push_back(hitGroupIndex(pass, "shadow"));
                        break;
                    case World::WORLD_NO_REFLECT:
                        mapped.push_back(hitGroupIndex(pass, "world_no_reflect"));
                        break;
                    case World::WORLD_CLOUD:
                        mapped.push_back(hitGroupIndex(pass, "clouds"));
                        break;
                    case World::BOAT_WATER_MASK:
                        mapped.push_back(hitGroupIndex(pass, "water_mask"));
                        break;
                    case World::END_PORTAL:
                        mapped.push_back(hitGroupIndex(pass, "end_portal"));
                        break;
                    case World::END_GATE_WAY:
                        mapped.push_back(hitGroupIndex(pass, "end_gateway"));
                        break;
                    case World::WORLD_SOLID:
                    case World::WORLD_TRANSPARENT:
                        if (entityLikeGeometry) {
                            mapped.push_back(hitGroupIndex(pass, "entity_solid_z_offset_forward"));
                            sbtRouteEntityNoHeight++;
                            break;
                        }
                        sbtRouteChunkFallback++;
                        [[fallthrough]];
                    default:
                        mapped.push_back(pass.fallbackHitGroupIndex);
                        break;
                }
            }
        }
        if (mapped.empty()) { mapped.push_back(pass.fallbackHitGroupIndex); }
        return mapped;
    };

    uint32_t localPasses = 0;
    uint32_t localUnsupportedPasses = 0;
    uint32_t localWorldPasses = 0;

    std::vector<ExpressionEvaluator::Variable> additionalVariables = {
        {.name = "RENDER_WIDTH", .value = static_cast<double>(width)},
        {.name = "RENDER_HEIGHT", .value = static_cast<double>(height)},
    };

    bindPackFogOutputs();
    shaderPack_->transitionRuntimeImagesForUse(commandBuffer, frameIndex, queueIndex);

    try {
        shaderPack_->executeCommands(
            ShaderPackLoader::Stage::RayTracing,
            shaderPack_->execution(ShaderPackLoader::Stage::RayTracing).commands,
            shaderPackRayTracingVariables_,
            additionalVariables,
            true,
            64,
            [&](const std::string &passName, ShaderPack::ExecutionVariables &variables) {
                localPasses++;
                shaderPackPassDispatchCounts_[passName]++;

                if (auto iter = shaderPackFullScreenPasses_.find(passName);
                    iter != shaderPackFullScreenPasses_.end()) {
                    FullScreenPass &pass = iter->second;
                    auto textureRef = shaderPack_->findRuntimeTexture(pass.config.target);
                    if (!textureRef.has_value()) {
                        throw std::runtime_error("fullscreen target missing: " + pass.config.target);
                    }
                    auto &targetTexture = textureRef->get();
                    auto image = shaderPack_->findRuntimeVKTexture(targetTexture, frameIndex);
                    uploadExecution(pass.executionBuffer, variables, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

                    const bool storageBacked = targetTexture.config.storageBinding.has_value();
                    transitionImage(image,
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

                    for (size_t layer = 0; layer < pass.pipelines.size(); ++layer) {
                        commandBuffer->beginRenderPass({
                            .renderPass = pass.renderPass,
                            .framebuffer = pass.framebuffers[frameIndex][layer],
                            .renderAreaExtent = {image->width(), image->height()},
                            .clearValues = {},
                        });
                        commandBuffer->bindGraphicsPipeline(pass.pipelines[layer])
                            ->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS)
                            ->draw(3, 1)
                            ->endRenderPass();
                    }

                    transitionImage(image,
                                    storageBacked ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                    VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
                    shaderPackFullScreenPassDispatches_++;
                    shaderPackIntermediatesGenerated_ = 1;
                    return;
                }

                if (auto iter = shaderPackComputePasses_.find(passName); iter != shaderPackComputePasses_.end()) {
                    ComputePass &pass = iter->second;
                    uploadExecution(pass.executionBuffer, variables, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
                    const uint32_t gx = evaluateExtent(pass.config.gx, variables, additionalVariables);
                    const uint32_t gy = evaluateExtent(pass.config.gy, variables, additionalVariables);
                    const uint32_t gz = evaluateExtent(pass.config.gz, variables, additionalVariables);
                    commandBuffer->bindComputePipeline(pass.pipeline)
                        ->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE);
                    vkCmdDispatch(commandBuffer->vkCommandBuffer(), gx, gy, gz);
                    passBarrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
                    shaderPackComputePassDispatches_++;
                    shaderPackIntermediatesGenerated_ = 1;
                    return;
                }

                if (auto iter = shaderPackRayTracingPasses_.find(passName);
                    iter != shaderPackRayTracingPasses_.end()) {
                    RayTracingPass &pass = iter->second;
                    uploadExecution(pass.executionBuffer, variables, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);
                    auto mappedGeometryTypes = mapGeometryTypes(pass);
                    pass.updateSbts[frameIndex]->setupHitSBT(mappedGeometryTypes);

                    const uint32_t dispatchWidth = evaluateExtent(pass.config.width, variables, additionalVariables);
                    const uint32_t dispatchHeight = evaluateExtent(pass.config.height, variables, additionalVariables);
                    const uint32_t dispatchDepth = evaluateExtent(pass.config.depth, variables, additionalVariables);
                    vkCmdPushConstants(commandBuffer->vkCommandBuffer(),
                                       descriptorTable->vkPipelineLayout(),
                                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                           VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                           VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                                       0,
                                       sizeof(RayTracingPushConstant),
                                       &pushConstant);
                    commandBuffer->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
                        ->bindRTPipeline(pass.updatePipeline)
                        ->raytracing(pass.updateSbts[frameIndex], dispatchWidth, dispatchHeight, dispatchDepth);
                    passBarrier(VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);
                    shaderPackRayTracingPassDispatches_++;
                    if (passName == "world") { localWorldPasses++; }
                    return;
                }

                localUnsupportedPasses++;
                throw std::runtime_error("unsupported shader-pack ray-tracing-stage pass: " + passName);
            });
    } catch (const std::exception &e) {
        restoreFixedBinding13();
        disableShaderPackBackend(firstFrame ? "fallback_first_frame_executor_failure" : "fallback_executor_failure",
                                 e.what());
        return false;
    } catch (...) {
        restoreFixedBinding13();
        disableShaderPackBackend(firstFrame ? "fallback_first_frame_executor_failure" : "fallback_executor_failure",
                                 "unknown exception");
        return false;
    }

    restoreFixedBinding13();

    shaderPackGraphPassesExecuted_ += localPasses;
    shaderPackGraphUnsupportedPasses_ += localUnsupportedPasses;
    shaderPackGraphWorldPasses_ += localWorldPasses;
    shaderPackBackendStatus_ = "executor_full";
    shaderPackFallbackReason_.clear();

    if (firstFrame) {
        renderDiag("ShaderPack executor first frame passes=%u worldPasses=%u unsupported=%u fs=%llu compute=%llu rt=%llu",
                   localPasses,
                   localWorldPasses,
                   localUnsupportedPasses,
                   static_cast<unsigned long long>(shaderPackFullScreenPassDispatches_),
                   static_cast<unsigned long long>(shaderPackComputePassDispatches_),
                   static_cast<unsigned long long>(shaderPackRayTracingPassDispatches_));
        RadianceLogger::log("RayTracing", "INFO",
                            "Shader pack executor first frame passes=%u worldPasses=%u unsupported=%u fs=%llu compute=%llu rt=%llu",
                            localPasses,
                            localWorldPasses,
                            localUnsupportedPasses,
                            static_cast<unsigned long long>(shaderPackFullScreenPassDispatches_),
                            static_cast<unsigned long long>(shaderPackComputePassDispatches_),
                            static_cast<unsigned long long>(shaderPackRayTracingPassDispatches_));
        renderDiag("ShaderPack SBT routing total=%u entity=%u chunk=%u entityWorldMask=%u entityNoHeight=%u chunkFallback=%u missingSource=%u",
                   sbtRouteTotal,
                   sbtRouteEntity,
                   sbtRouteChunk,
                   sbtRouteEntityWorldMasked,
                   sbtRouteEntityNoHeight,
                   sbtRouteChunkFallback,
                   sbtRouteMissingSource);
        RadianceLogger::log("RayTracing", "INFO",
                            "Shader pack SBT routing total=%u entity=%u chunk=%u entityWorldMask=%u entityNoHeight=%u chunkFallback=%u missingSource=%u",
                            sbtRouteTotal,
                            sbtRouteEntity,
                            sbtRouteChunk,
                            sbtRouteEntityWorldMasked,
                            sbtRouteEntityNoHeight,
                            sbtRouteChunkFallback,
                            sbtRouteMissingSource);
    }

    return localWorldPasses > 0 && localUnsupportedPasses == 0;
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

    validateShaderPackRuntime();
    initDescriptorTables();
    initEnergyLUT();
    initImages();
    initShaderPackRuntimeResources();
    initShaderPackRayTracingExecutor();
    initPipeline();
    initSBT();
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
    // Upload immediately via important-upload path (synchronous stagingâ†’device before first RT dispatch)
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
    if (index < 0) return;

    {
        std::lock_guard<std::mutex> lock(legacyTextureBindingsMutex_);
        legacyTextureBindings_[index] = LegacyTextureBinding{std::move(sampler), std::move(image)};
    }
    legacyTextureBindingRevision_.fetch_add(1, std::memory_order_release);
}

void RayTracingModule::bindLegacyTexturesForSlot(const std::shared_ptr<vk::DescriptorTable>& descriptorTable,
                                                 uint32_t frameIndex) {
    if (!descriptorTable) return;
    if (frameIndex >= legacyTextureBindingRevisions_.size()) {
        legacyTextureBindingRevisions_.resize(frameIndex + 1, 0);
    }

    uint64_t targetRevision = legacyTextureBindingRevision_.load(std::memory_order_acquire);
    if (legacyTextureBindingRevisions_[frameIndex] == targetRevision) return;

    std::vector<std::pair<int, LegacyTextureBinding>> bindings;
    {
        std::lock_guard<std::mutex> lock(legacyTextureBindingsMutex_);
        targetRevision = legacyTextureBindingRevision_.load(std::memory_order_acquire);
        bindings.reserve(legacyTextureBindings_.size());
        for (const auto& binding : legacyTextureBindings_) {
            bindings.emplace_back(binding.first, binding.second);
        }
    }

    for (const auto& [index, binding] : bindings) {
        if (!binding.sampler || !binding.image) continue;
        descriptorTable->bindSamplerImage(binding.sampler, binding.image,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                          0, 0, index);
    }
    legacyTextureBindingRevisions_[frameIndex] = targetRevision;
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
        std::cerr << "[Offline] Failed to load accumulate_comp.spv â€” accumulation disabled" << std::endl;
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
        std::cerr << "[Offline] Failed to load emission_compose_comp.spv â€” emission preservation disabled" << std::endl;
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
    if (shaderPack_) {
        shaderPack_->preClose();
        shaderPack_.reset();
    }

    auto framework = framework_.lock();
    if (framework) {
        VkDevice dev = framework->device()->vkDevice();
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
        Renderer::accumPipeline = VK_NULL_HANDLE;
        Renderer::accumPipelineLayout = VK_NULL_HANDLE;
        Renderer::accumDescSetLayout = VK_NULL_HANDLE;
        Renderer::accumDescPool = VK_NULL_HANDLE;
        Renderer::accumPipelineReady = false;
        if (Renderer::emissionComposePipeline != VK_NULL_HANDLE) vkDestroyPipeline(dev, Renderer::emissionComposePipeline, nullptr);
        if (Renderer::emissionComposePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, Renderer::emissionComposePipelineLayout, nullptr);
        if (Renderer::emissionComposeDescSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, Renderer::emissionComposeDescSetLayout, nullptr);
        if (Renderer::emissionComposeDescPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, Renderer::emissionComposeDescPool, nullptr);
        Renderer::emissionComposePipeline = VK_NULL_HANDLE;
        Renderer::emissionComposePipelineLayout = VK_NULL_HANDLE;
        Renderer::emissionComposeDescSetLayout = VK_NULL_HANDLE;
        Renderer::emissionComposeDescPool = VK_NULL_HANDLE;
        Renderer::emissionComposePipelineReady = false;
    }
    Renderer::frameGenDepthImages.clear();
    Renderer::frameGenMotionVectorImages.clear();
    Renderer::accumOutputImage = nullptr;
    Renderer::accumBufferImage = nullptr;
    Renderer::accumDescSets.clear();
    Renderer::emissionComposeDescSets.clear();
    Renderer::emissionImages.clear();
    Renderer::renderResHdrImages.clear();
    textureDescriptorSlotStates_.clear();
    legacyTextureBindingRevisions_.clear();
    {
        std::lock_guard<std::mutex> lock(legacyTextureBindingsMutex_);
        legacyTextureBindings_.clear();
    }
    legacyTextureBindingRevision_.store(1, std::memory_order_release);
}

void RayTracingModule::initDescriptorTables() {
    auto framework = framework_.lock();
    if (!framework) return;

    uint32_t size = framework->swapchain()->imageCount();
    rayTracingDescriptorTables_.resize(size);
    shaderPackVisualSettingBuffers_.resize(size);
    textureDescriptorSlotStates_.assign(size, TextureDescriptorSlotState{});
    legacyTextureBindingRevisions_.assign(size, 0);

    for (int i = 0; i < size; i++) {
        shaderPackVisualSettingBuffers_[i] = vk::HostVisibleBuffer::create(
            framework->vma(), framework->device(), sizeof(ShaderPackVisualSettings),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

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
                .defineDescriptorLayoutSetBinding({
                    .binding = 6, // block sprite LabPBR flag sampler2DArray
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
                })                .defineDescriptorLayoutSetBinding({
                    .binding = 10, // binding 10: per-section biome color buffer
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })                .defineDescriptorLayoutSetBinding({
                    .binding = 11, // binding 11: texture-primary AutoPBR scalar rules
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })                .defineDescriptorLayoutSetBinding({
                    .binding = 13, // binding 13: SpriteRegistry SSBO (texture array metadata)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                })                .defineDescriptorLayoutSetBinding({
                    .binding = 14, // binding 14: MaterialRegistry SSBO (global material id indirection)
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
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_COMPUTE_BIT,
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
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_COMPUTE_BIT,
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
                                  VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
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
                                  VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 14, // binding 14: shader-pack firstHitRefractionImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 15, // binding 15: shader-pack private fog image
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
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
                .beginDescriptorLayoutSet() // set 4: fixed adapter shader-pack visual settings
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .beginDescriptorLayoutSet() // set 5: isolated vanilla-pt runtime visual assets
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0, // trans_lut sampled
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1, // trans_lut storage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 2, // sky_cube sampled
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 3, // material_state storage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 4, // post_motion sampled
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 5, // post_dof_focus_ping sampled
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 6, // post_dof_focus_pong sampled
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 22, // volumetric_cloud_basic_noise sampled
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 23, // volumetric_cloud_detail_noise sampled
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 24, // volumetric_cloud_weather sampled
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 25, // volumetric_cloud_curl sampled
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 26, // volumetric_cloud_coverage_noise sampled
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 27, // fft_water_flow sampled
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 28, // fft_water_normal_a sampled
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 29, // fft_water_normal_b sampled
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 31, // post_star_cloud_transmittance sampled
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 32, // post_star_cloud_transmittance storage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 33, // post_dof_blur sampled
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 34, // volumetric_cloud_temporal storage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 35, // volumetric_cloud_history storage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .beginDescriptorLayoutSet() // set 6: shader-pack execution variables
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = ShaderPackLoader::EXECUTION_BINDING,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
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
            {upstreamFirstHitRefractionImages_[i],     VK_IMAGE_LAYOUT_GENERAL, 3, 14},
            {upstreamFogImages_[i],                    VK_IMAGE_LAYOUT_GENERAL, 3, 15},
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
        rayTracingDescriptorTables_[i]->bindImages(imageBindings);

        // Energy compensation LUT (set 2, binding 4) â€” sampler image, stays individual
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
    const bool displacementRequested = Renderer::options.displacementEnabled && worldSolidTransparentClosestHitShader_;
    const bool useShaderDisplacement = displacementRequested && shaderDisplacementRuntimeAllowed();
    auto activeWorldSolidTransparentClosestHitShader =
        useShaderDisplacement
            ? worldSolidTransparentClosestHitShader_
            : worldSolidTransparentNoDisplacementClosestHitShader_;
    if (!activeWorldSolidTransparentClosestHitShader) {
        activeWorldSolidTransparentClosestHitShader = worldSolidTransparentClosestHitShader_;
    }
    RadianceLogger::log("RayTracing", "INFO",
                        "World closest-hit variant: %s (displacementRequested=%d displacementRuntimeAllowed=%d displacementForceDisabled=%d noDisplacementShader=%d)",
                        useShaderDisplacement ? "displacement" : "no_displacement",
                        displacementRequested ? 1 : 0,
                        shaderDisplacementRuntimeAllowed() ? 1 : 0,
                        shaderDisplacementForceDisabled() ? 1 : 0,
                        worldSolidTransparentNoDisplacementClosestHitShader_ ? 1 : 0);
    worldNoReflectClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_no_reflect_rchit.spv").string());
    worldCloudClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_cloud_rchit.spv").string());
    shadowAnyHitShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/shadow_rahit.spv").string());
    worldSolidAnyHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_solid_rahit.spv").string());
    const uint32_t activeWorldSolidAnyHitStage =
        (useShaderDisplacement && worldSolidAnyHitShader_) ? 19u : VK_SHADER_UNUSED_KHR;
    worldTransparentAnyHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_transparent_rahit.spv").string());
    auto worldSolidAnyHitStageShader =
        worldSolidAnyHitShader_ ? worldSolidAnyHitShader_ : worldTransparentAnyHitShader_;
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
            .defineShaderStage(worldSolidAnyHitStageShader, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                // 19
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
                               activeWorldSolidAnyHitStage,
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
    const bool displacementRequested = Renderer::options.displacementEnabled && worldSolidTransparentClosestHitShader_;
    const bool useShaderDisplacement = displacementRequested && shaderDisplacementRuntimeAllowed();
    auto activeWorldSolidTransparentClosestHitShader =
        useShaderDisplacement
            ? worldSolidTransparentClosestHitShader_
            : worldSolidTransparentNoDisplacementClosestHitShader_;
    if (!activeWorldSolidTransparentClosestHitShader) {
        activeWorldSolidTransparentClosestHitShader = worldSolidTransparentClosestHitShader_;
    }
    const uint32_t activeWorldSolidAnyHitStage =
        (useShaderDisplacement && worldSolidAnyHitShader_) ? 19u : VK_SHADER_UNUSED_KHR;
    auto worldSolidAnyHitStageShader =
        worldSolidAnyHitShader_ ? worldSolidAnyHitShader_ : worldTransparentAnyHitShader_;
    RadianceLogger::log("RayTracing", "INFO",
                        "SHARC closest-hit variant: %s (displacementRequested=%d displacementRuntimeAllowed=%d displacementForceDisabled=%d noDisplacementShader=%d)",
                        useShaderDisplacement ? "displacement" : "no_displacement",
                        displacementRequested ? 1 : 0,
                        shaderDisplacementRuntimeAllowed() ? 1 : 0,
                        shaderDisplacementForceDisabled() ? 1 : 0,
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
            .defineShaderStage(worldSolidAnyHitStageShader, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                 // 19
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
                               activeWorldSolidAnyHitStage,
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

    // Pipeline layout with push constant only (no descriptor sets â€” uses BDA)
    VkPushConstantRange pushRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 76}; // 19 fields Ã— 4 bytes (with uint64_t alignment)
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
        renderDiag("RT worldPrepare end tlas=%d instances=%u",
                   (int)(worldPrepareContext->tlas != nullptr),
                   worldPrepareContext->prevTlasInstanceCount_);
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
    const bool shaderPackVisualAssetsReady =
        module->shaderPackRuntimeResourcesReady_ && module->shaderPack_ && module->shaderPack_->hasRuntimeResources();

    ShaderPackVisualSettings shaderPackVisualSettings = module->shaderPackVisualSettings_;
    if (!shaderPackVisualAssetsReady) {
        shaderPackVisualSettings.waterSurfaceMode = 0;
        shaderPackVisualSettings.waterCausticsEnabled = 0;
    }
    if (shaderPackVisualSettings.cloudMode != 2) {
        shaderPackVisualSettings.volumetricCloudCastShadow = 0;
    }
    if (context->frameIndex < module->shaderPackVisualSettingBuffers_.size()
        && module->shaderPackVisualSettingBuffers_[context->frameIndex]) {
        module->shaderPackVisualSettingBuffers_[context->frameIndex]->uploadToBuffer(
            &shaderPackVisualSettings, sizeof(ShaderPackVisualSettings), 0);
    }

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
        {worldBuffer,                                2,  0},
        {buffers->lastWorldUniformBuffer(),          2,  1},
        {buffers->skyUniformBuffer(),                2,  2},
    };
    if (context->frameIndex < module->shaderPackVisualSettingBuffers_.size()
        && module->shaderPackVisualSettingBuffers_[context->frameIndex]) {
        bufferBindings.push_back({module->shaderPackVisualSettingBuffers_[context->frameIndex], 4, 0});
    }
    // Per-section biome colors for shader-side tinting
    if (worldPrepareContext->biomeColorBuffer) {
        bufferBindings.push_back({worldPrepareContext->biomeColorBuffer, 1, 10});
    }
    // SpriteRegistry SSBO for texture array metadata
    auto spriteRegBuffer = Renderer::textureSystem.registry().getBuffer();
    const VkBuffer spriteRegistryBuffer = spriteRegBuffer ? spriteRegBuffer->vkBuffer() : VK_NULL_HANDLE;
    if (spriteRegBuffer) {
        bufferBindings.push_back({spriteRegBuffer, 1, 13});
    }
    auto materialRegBuffer = Renderer::textureSystem.materials().getBuffer();
    const VkBuffer materialRegistryBuffer = materialRegBuffer ? materialRegBuffer->vkBuffer() : VK_NULL_HANDLE;
    if (materialRegBuffer) {
        bufferBindings.push_back({materialRegBuffer, 1, 14});
    }
    auto textureRuleBuffer = Renderer::textureSystem.textureRules().getBuffer();
    const VkBuffer textureRuleVkBuffer = textureRuleBuffer ? textureRuleBuffer->vkBuffer() : VK_NULL_HANDLE;
    if (textureRuleBuffer) {
        bufferBindings.push_back({textureRuleBuffer, 1, 11});
    }
    // Blue noise buffers (already uploaded at init time via queueImportantWorldUpload)
    if (module->blueNoise_) {
        bufferBindings.push_back({module->blueNoise_->sobolBuffer(),      2, 9});
        bufferBindings.push_back({module->blueNoise_->scramblingBuffer(), 2, 10});
    }
    rayTracingDescriptorTable->bindBufferBatch(bufferBindings);

    // Batch per-frame optional image rebindings.
    using IB = vk::DescriptorTable::ImageBinding;
    std::vector<IB> frameImageBindings;
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
    TextureArrayManager::ArrayInfo albedoInfo{};
    TextureArrayManager::ArrayInfo specInfo{};
    TextureArrayManager::ArrayInfo normInfo{};
    TextureArrayManager::ArrayInfo flagInfo{};
    const bool hasAlbedo = texArrayMgr.getArraySnapshot(texSystem.blockAlbedoArrayId(), albedoInfo) &&
                           albedoInfo.image && albedoInfo.sampler;
    const bool hasSpec = texArrayMgr.getArraySnapshot(texSystem.blockSpecularArrayId(), specInfo) &&
                         specInfo.image && specInfo.sampler;
    const bool hasNorm = texArrayMgr.getArraySnapshot(texSystem.blockNormalArrayId(), normInfo) &&
                         normInfo.image && normInfo.sampler;
    const bool hasFlag = texArrayMgr.getArraySnapshot(texSystem.blockFlagArrayId(), flagInfo) &&
                         flagInfo.image && flagInfo.sampler;

    const VkImageView albedoView = hasAlbedo ? albedoInfo.image->vkImageView() : VK_NULL_HANDLE;
    const VkImageView specView = hasSpec ? specInfo.image->vkImageView() : VK_NULL_HANDLE;
    const VkImageView normView = hasNorm ? normInfo.image->vkImageView() : VK_NULL_HANDLE;
    const VkImageView flagView = hasFlag ? flagInfo.image->vkImageView() : VK_NULL_HANDLE;

    if (!texSystem.isFinalized() || !hasAlbedo || !hasSpec || !hasNorm || !hasFlag ||
        !spriteRegBuffer || !materialRegBuffer || !textureRuleBuffer) {
        renderDiag("RT descriptors missing texture resources finalized=%d albedo=%d spec=%d norm=%d flag=%d spriteReg=%d materialReg=%d rules=%d; skipping RT",
                   texSystem.isFinalized() ? 1 : 0, hasAlbedo ? 1 : 0, hasSpec ? 1 : 0,
                   hasNorm ? 1 : 0, hasFlag ? 1 : 0, spriteRegBuffer ? 1 : 0,
                   materialRegBuffer ? 1 : 0,
                   textureRuleBuffer ? 1 : 0);
        g_crashRing.record("RT:descriptors_missing_textures");
        textureProfile.close();
        worldCommandBuffer->endLabel();
        return;
    }

    auto bindBlockTextureArrays = [&](const std::shared_ptr<vk::DescriptorTable>& table) {
        if (!table) return;
        if (hasAlbedo) {
            table->bindSamplerImage(albedoInfo.sampler, albedoInfo.image,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 3, 0);
        }
        if (hasSpec) {
            table->bindSamplerImage(specInfo.sampler, specInfo.image,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 4, 0);
        }
        if (hasNorm) {
            table->bindSamplerImage(normInfo.sampler, normInfo.image,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 5, 0);
        }
        if (hasFlag) {
            table->bindSamplerImage(flagInfo.sampler, flagInfo.image,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 6, 0);
        }
    };
    auto bindSpriteRegistry = [&](const std::shared_ptr<vk::DescriptorTable>& table) {
        if (!table || !spriteRegBuffer) return;
        table->bindBuffer(spriteRegBuffer, 1, 13);
    };
    auto bindMaterialRegistry = [&](const std::shared_ptr<vk::DescriptorTable>& table) {
        if (!table || !materialRegBuffer) return;
        table->bindBuffer(materialRegBuffer, 1, 14);
    };
    auto bindTextureRules = [&](const std::shared_ptr<vk::DescriptorTable>& table) {
        if (!table || !textureRuleBuffer) return;
        table->bindBuffer(textureRuleBuffer, 1, 11);
    };

    module->bindLegacyTexturesForSlot(rayTracingDescriptorTable, context->frameIndex);
    if (context->frameIndex >= module->textureDescriptorSlotStates_.size()) {
        module->textureDescriptorSlotStates_.resize(context->frameIndex + 1);
    }
    auto& descriptorSlotState = module->textureDescriptorSlotStates_[context->frameIndex];
    const bool textureDescriptorGenerationChanged =
        textureGeneration != descriptorSlotState.generation ||
        albedoView != descriptorSlotState.albedoTextureView ||
        specView != descriptorSlotState.specularTextureView ||
        normView != descriptorSlotState.normalTextureView ||
        flagView != descriptorSlotState.flagTextureView ||
        spriteRegistryBuffer != descriptorSlotState.spriteRegistryBuffer ||
        materialRegistryBuffer != descriptorSlotState.materialRegistryBuffer ||
        textureRuleVkBuffer != descriptorSlotState.textureRuleBuffer;

    if (textureDescriptorGenerationChanged) {
        bindBlockTextureArrays(rayTracingDescriptorTable);
        bindSpriteRegistry(rayTracingDescriptorTable);
        bindMaterialRegistry(rayTracingDescriptorTable);
        bindTextureRules(rayTracingDescriptorTable);
        RadianceLogger::log(
            "TextureDescriptors", "INFO",
            "refresh-slot gen=%llu frame=%u albedoId=%u image=0x%llx view=0x%llx sampler=0x%llx "
            "specId=%u image=0x%llx view=0x%llx sampler=0x%llx normId=%u image=0x%llx view=0x%llx sampler=0x%llx "
            "flagId=%u image=0x%llx view=0x%llx sampler=0x%llx "
            "spriteRegistry=0x%llx materialRegistry=0x%llx",
            static_cast<unsigned long long>(textureGeneration),
            context->frameIndex,
            texSystem.blockAlbedoArrayId(),
            hasAlbedo ? static_cast<unsigned long long>(vk::DebugUtils::objectHandle(albedoInfo.image->vkImage())) : 0ull,
            static_cast<unsigned long long>(vk::DebugUtils::objectHandle(albedoView)),
            hasAlbedo ? static_cast<unsigned long long>(vk::DebugUtils::objectHandle(albedoInfo.sampler->vkSamper())) : 0ull,
            texSystem.blockSpecularArrayId(),
            hasSpec ? static_cast<unsigned long long>(vk::DebugUtils::objectHandle(specInfo.image->vkImage())) : 0ull,
            static_cast<unsigned long long>(vk::DebugUtils::objectHandle(specView)),
            hasSpec ? static_cast<unsigned long long>(vk::DebugUtils::objectHandle(specInfo.sampler->vkSamper())) : 0ull,
            texSystem.blockNormalArrayId(),
            hasNorm ? static_cast<unsigned long long>(vk::DebugUtils::objectHandle(normInfo.image->vkImage())) : 0ull,
            static_cast<unsigned long long>(vk::DebugUtils::objectHandle(normView)),
            hasNorm ? static_cast<unsigned long long>(vk::DebugUtils::objectHandle(normInfo.sampler->vkSamper())) : 0ull,
            texSystem.blockFlagArrayId(),
            hasFlag ? static_cast<unsigned long long>(vk::DebugUtils::objectHandle(flagInfo.image->vkImage())) : 0ull,
            static_cast<unsigned long long>(vk::DebugUtils::objectHandle(flagView)),
            hasFlag ? static_cast<unsigned long long>(vk::DebugUtils::objectHandle(flagInfo.sampler->vkSamper())) : 0ull,
            static_cast<unsigned long long>(vk::DebugUtils::objectHandle(spriteRegistryBuffer)),
            static_cast<unsigned long long>(vk::DebugUtils::objectHandle(materialRegistryBuffer)));
        descriptorSlotState.generation = textureGeneration;
        descriptorSlotState.albedoTextureView = albedoView;
        descriptorSlotState.specularTextureView = specView;
        descriptorSlotState.normalTextureView = normView;
        descriptorSlotState.flagTextureView = flagView;
        descriptorSlotState.spriteRegistryBuffer = spriteRegistryBuffer;
        descriptorSlotState.materialRegistryBuffer = materialRegistryBuffer;
        descriptorSlotState.textureRuleBuffer = textureRuleVkBuffer;
    } else {
        bindBlockTextureArrays(rayTracingDescriptorTable);
        bindMaterialRegistry(rayTracingDescriptorTable);
        bindTextureRules(rayTracingDescriptorTable);
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
        // Refresh ALL contexts' stale SBT pointers â€” were nullptr at construction time
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
#if defined(MCVR_ENABLE_SHARC) && defined(MCVR_ENABLE_SHARC_MAIN_TRACE_QUERY)
                       | (sharcMainTraceQueryActive ? 32 : 0)
#elif defined(MCVR_ENABLE_SHARC) && defined(MCVR_ENABLE_SHARC_QUERY_PASS)
                       | (sharcQueryPassActive ? 32 : 0)
#endif
                       | (Renderer::options.multiScatterGGX ? 128 : 0)
                       | diffuseModelFlagBits(Renderer::options.diffuseModel)
                       | (Renderer::options.beerLawShadows ? 512 : 0)
                       | (Renderer::options.noEmissionClamp ? 1024 : 0)
                       | (Renderer::options.physicalSunDisk ? 2048 : 0)
                       | (Renderer::options.noHandAmbient ? 4096 : 0)
                       | ((framework->device()->hasSER() && Renderer::options.serEnabled) ? 8192 : 0)
                       | ((framework->device()->hasSER() && Renderer::options.serEnabled && Renderer::options.serHintsEnabled) ? 16384 : 0)
                       | (Renderer::options.entityNormalsEnabled ? 32768 : 0);
    // Shader-pack visual switches carried into the fixed fallback/adapter path:
    // 0=cloud mode, 1=cloud shadows, 2=water surface mode, 3=volumetric light mode, 4=water caustics.
    pushConstant.reservedLighting0 = module->shaderPackCloudMode_;
    pushConstant.reservedLighting1 = module->shaderPackCloudShadowsEnabled_ ? 1.0f : 0.0f;
    pushConstant.reservedLighting2 = shaderPackVisualAssetsReady ? module->shaderPackWaterSurfaceMode_ : 0;
    pushConstant.reservedLighting3 = module->shaderPackVolumetricLightMode_;
    pushConstant.reservedLighting4 =
        (shaderPackVisualAssetsReady && module->shaderPackWaterCausticsEnabled_) ? 1 : 0;
    // Keep the DLSS-RR path scene-referred. Earlier builds pre-exposed DLSS-D input
    // by 0.1 while leaving NGX exposure scale at 1.0, so RR history/output could drift
    // away from the histogram/tone-mapper scale and bloom into white while standing still.
    pushConstant.preExposure = 1.0f;
    // Shader displacement follows the user's saved option. Keep the runtime gate in place so
    // crash isolation can force the no-displacement variant during targeted repro work.
    const bool displacementActive = Renderer::options.displacementEnabled && shaderDisplacementRuntimeAllowed();
    pushConstant.displacementDepthScale = displacementActive ? Renderer::options.displacementDepthScale : 0.0f;
    pushConstant.displacementPrimarySteps = Renderer::options.displacementPrimarySteps;
    pushConstant.displacementRefinementSteps = Renderer::options.displacementRefinementSteps;
    pushConstant.displacementFadeDistanceBlocks = Renderer::options.displacementFadeDistanceBlocks;
    pushConstant.blueNoiseFrame = context->frameIndex;
    pushConstant.reservedPc0 = 0;
    pushConstant.reservedAddr = 0;
    pushConstant.rtDebugFlags = Renderer::options.rtDebugFlags;
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
                "pushConst: bounces=%d flags=0x%x rtDebug=0x%x handInst=%u cloudMode=%d cloudShadows=%d waterMode=%d volumetricLightMode=%d caustics=%d sharcQuery=%d sharcQueryPass=%d sharcQueryMode=%d sharcFrame=%u sharcWarmup=%u sharcMode=%d displacementActive=%d displacementOption=%d displacementRuntimeAllowed=%d displacementForceDisabled=%d displacementQuality=%u displacementDepth=%.4f displacementSteps=%d displacementRefinement=%d displacementFade=%.0f",
                pushConstant.numRayBounces, pushConstant.flags, pushConstant.rtDebugFlags,
                pushConstant.handInstanceCount,
                pushConstant.reservedLighting0,
                pushConstant.reservedLighting1 > 0.5f ? 1 : 0,
                pushConstant.reservedLighting2,
                pushConstant.reservedLighting3,
                pushConstant.reservedLighting4,
                sharcMainTraceQueryActive ? 1 : 0,
                sharcQueryPassActive ? 1 : 0, pushConstant.sharcQueryMode,
                module->sharcFrameIndex_, sharcMainTraceWarmupFrames,
                MCVR_SHARC_MAIN_TRACE_QUERY_MODE, displacementActive ? 1 : 0,
                Renderer::options.displacementEnabled ? 1 : 0,
                shaderDisplacementRuntimeAllowed() ? 1 : 0,
                shaderDisplacementForceDisabled() ? 1 : 0,
                Renderer::options.displacementQuality, pushConstant.displacementDepthScale,
                pushConstant.displacementPrimarySteps, pushConstant.displacementRefinementSteps,
                pushConstant.displacementFadeDistanceBlocks);
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
    // Bit 1 (disableRR): auto-set by preset â€” Raw Accurate forces RR off
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

    // Pre-exposure is already neutral; keep accumulation explicitly scene-referred.
    // All modes disable temporal reuse (each frame is independent).
    if (accumulating) {
        pushConstant.preExposure = 1.0f;
    }

    // Material SSBO BDA: pass buffer address via push constant (avoids descriptor lookup overhead)
    pushConstant.reservedAddr = 0;

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
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
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
    addBarrier(module->reflectionMvImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->animatedTexMaskImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->particleMaskImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->biasMaskImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->rtHitDistImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->motionVectors3DImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->gbufferMetallicImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->gbufferShadingModelIdImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->gbufferMaterialIdImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->positionViewSpaceImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->transparencyLayerImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->transparencyLayerOpacityImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->transparencyLayerMvecsImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->upstreamFogImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->upstreamFirstHitRefractionImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
#ifdef MCVR_ENABLE_SHARC_QUERY_PASS
    if (context->frameIndex < module->sharcCandidatePosHitTImages_.size()) {
        addBarrier(module->sharcCandidatePosHitTImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
        addBarrier(module->sharcCandidateNormalRoughnessImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
        addBarrier(module->sharcCandidateThroughputImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
        addBarrier(module->sharcCandidatePrefixRadianceFlagsImages_[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL);
    }
#endif

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

    // Light clustering compute pass â€” DISABLED: contribution-sorted global list replaces tile clustering.
    // Tile buffer stays allocated (descriptor layout unchanged); CHS reads tileCount=0 â†’ global fallback.
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
        updatePC.flags &= ~(4 | 16);
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


    // Re-push RT push constants (may have been invalidated by compute pipeline bind above)
    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), rayTracingDescriptorTable->vkPipelineLayout(),
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                           VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                           VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                       0, sizeof(RayTracingPushConstant), &pushConstant);

    // Pass 3: Main Render. Prefer the real shader-pack executor; keep fixed RT as emergency fallback.
    bool shaderPackExecutorActive = false;
    if (module->shaderPackExecutorReady_) {
        worldCommandBuffer->beginLabel("RT:ShaderPackExecutor", 0.4f, 0.7f, 1.0f);
        ScopedGpuProfile executorProfile(profileCmd, "RT.ShaderPackExecutor");
        GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::RT_DISPATCH_MAIN);
        g_crashRing.record("RT:shaderPackExecutor:start");
        shaderPackExecutorActive = module->recordShaderPackRayTracingExecutor(
            worldCommandBuffer,
            rayTracingDescriptorTable,
            worldPrepareContext,
            pushConstant,
            context->frameIndex,
            hdrNoisyOutputImage->width(),
            hdrNoisyOutputImage->height());
        renderDiag("RT shaderPackExecutor end active=%d backendStatus=%s",
                   shaderPackExecutorActive ? 1 : 0,
                   module->shaderPackBackendStatus_.c_str());
        g_crashRing.record(shaderPackExecutorActive ? "RT:shaderPackExecutor:done" :
                                                  "RT:shaderPackExecutor:fallback");
        executorProfile.close();
        worldCommandBuffer->endLabel();
    }

    if (!shaderPackExecutorActive) {
        worldCommandBuffer->beginLabel("RT:MainTrace", 1.0f, 0.2f, 0.2f);
        ScopedGpuProfile mainTraceProfile(profileCmd, "RT.MainTrace");
        GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::RT_DISPATCH_MAIN);
        renderDiag("RT mainTrace fallback begin w=%u h=%u backendStatus=%s",
                   hdrNoisyOutputImage->width(),
                   hdrNoisyOutputImage->height(),
                   module->shaderPackBackendStatus_.c_str());
        g_crashRing.record("RT:mainTrace");
        worldCommandBuffer->bindDescriptorTable(rayTracingDescriptorTable, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
            ->bindRTPipeline(module->rayTracingPipeline_)
            ->raytracing(sbt, hdrNoisyOutputImage->width(), hdrNoisyOutputImage->height(), 1);
        renderDiag("RT mainTrace end");
        mainTraceProfile.close();
        worldCommandBuffer->endLabel(); // end MainTrace
    }


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
    // Offline accumulation: Welford running average into RGBA32F buffer
    // Raw Fast (0) and Raw Slow (1) accumulate here; DLSS-D (2) accumulates in DLSS module
    if (accumulating && Renderer::accumPipelineReady && Renderer::options.offlineDenoised != 2) {
        VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();
        uint32_t frameIdx = context->frameIndex;
        ScopedGpuProfile offlineAccumProfile(profileCmd, "RT.OfflineAccum");

        // Barrier: RT output â†’ compute read
        VkMemoryBarrier accumBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        accumBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        accumBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &accumBarrier, 0, nullptr, 0, nullptr);

        // Update descriptor set (only when images change â€” typically once at init)
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

        // Barrier: compute write â†’ blit read
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

        // Barrier: blit write â†’ next stage read
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

    // (P4 removed â€” DLSS temporal mode replaced by DLSS-D Converge with per-frame reset)
}
