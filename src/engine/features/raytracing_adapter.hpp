#pragma once

// RT adapter: traces rays against the world TLAS when geometry is available.
// Falls back to sky gradient (miss shader only) when TLAS is empty.
//
// Produces all 34 V1 output images bound at descriptor set 3:
//   29 RT output storage images + 5 reservoir images.
// Formats match V1 exactly (see rt_descriptor_layout.hpp for binding constants).
//
// Uses V1-compatible 4-set descriptor layout via RtDescriptorLayout.

#include "feature_adapter.hpp"
#include "rt_descriptor_layout.hpp"
#include "platform/vulkan/vk2_image.hpp"
#include "platform/vulkan/vk2_shader.hpp"
#include "platform/vulkan/vk2_pipeline.hpp"
#include "platform/vulkan/vk2_rt_pipeline.hpp"
#include "platform/vulkan/vk2_descriptor.hpp"

#include <array>
#include <vector>

namespace engine {

class AtmosphereAdapter;
class GraphBuilder;

class RayTracingAdapter : public FeatureAdapter {
public:
    // Number of RT output images (set 3, bindings 0–33).
    static constexpr uint32_t OUTPUT_COUNT = rt_layout::OUT_BINDING_COUNT;  // 34

    void init(EngineServices& services) override;
    void configure(const ConfigSnapshot& config) override;
    void execute(const FrameContext& ctx, const PassResources& resources) override;
    void shutdown() override;
    const char* name() const override { return "RayTracing"; }

    // Register RT pass in graph. Returns the HDR radiance handle (primary output).
    // All 34 output handles are exposed via outputHandle() and named accessors.
    ResourceHandle registerPass(GraphBuilder& builder);

    // Generic accessor — binding index maps 1:1 to outputHandles_ array.
    ResourceHandle outputHandle(uint32_t binding) const { return outputHandles_[binding]; }

    // The 22-stage v2_full suite is still experimental. Downstream graph wiring
    // uses this to choose the safe minimal path unless the full contract is active.
    bool isFullShaderSuiteLoaded() const { return fullShadersLoaded_; }

    // Set the atmosphere adapter for LUT/cubemap descriptor binding.
    void setAtmosphere(AtmosphereAdapter* atmo) { atmosphere_ = atmo; }

    // Pre-exposure value from tonemapping adapter (auto-exposure feedback loop).
    // Called each frame before execute(). Default 1.0.
    void setPreExposure(float e) { preExposure_ = e; }

    // Offline accumulation state, set by EngineApp::processScene() each frame.
    // accumFrameCount: Welford N (0 = accumulation off or just reset).
    // offlineEnabled:  true when config.offlineAccumEnabled is set.
    // aperture/focalDist: depth-of-field parameters for DoF in RT shader.
    // When offlineBounces > 0, overrides the normal rayBounces push constant.
    void setAccumState(uint32_t accumFrameCount, bool offlineEnabled,
                       float aperture, float focalDist, uint32_t offlineBounces) {
        accumFrameCount_  = accumFrameCount;
        offlineEnabled_   = offlineEnabled;
        aperture_         = aperture;
        focalDistance_    = focalDist;
        offlineBounces_   = offlineBounces;
    }

    // Named accessors for key G-buffer outputs that downstream passes need.
    ResourceHandle radianceHandle()        const { return outputHandles_[rt_layout::OUT_BINDING_HDR_NOISY]; }
    ResourceHandle diffuseAlbedoHandle()   const { return outputHandles_[rt_layout::OUT_BINDING_DIFFUSE_ALBEDO]; }
    ResourceHandle specularAlbedoHandle()  const { return outputHandles_[rt_layout::OUT_BINDING_SPECULAR_ALBEDO]; }
    ResourceHandle normalHandle()          const { return outputHandles_[rt_layout::OUT_BINDING_NORMAL_ROUGHNESS]; }
    ResourceHandle motionHandle()          const { return outputHandles_[rt_layout::OUT_BINDING_MOTION_VECTOR]; }
    ResourceHandle linearDepthHandle()     const { return outputHandles_[rt_layout::OUT_BINDING_LINEAR_DEPTH]; }
    ResourceHandle specularHitDepthHandle() const { return outputHandles_[rt_layout::OUT_BINDING_SPECULAR_HIT_DEPTH]; }
    ResourceHandle firstHitDepthHandle()   const { return outputHandles_[rt_layout::OUT_BINDING_FIRST_HIT_DEPTH]; }
    ResourceHandle fhDiffuseDirectHandle() const { return outputHandles_[rt_layout::OUT_BINDING_FH_DIFFUSE_DIRECT]; }
    ResourceHandle fhDiffuseIndirectHandle() const { return outputHandles_[rt_layout::OUT_BINDING_FH_DIFFUSE_INDIRECT]; }
    ResourceHandle fhSpecularHandle()      const { return outputHandles_[rt_layout::OUT_BINDING_FH_SPECULAR]; }
    ResourceHandle fhClearHandle()         const { return outputHandles_[rt_layout::OUT_BINDING_FH_CLEAR]; }
    ResourceHandle fhBaseEmissionHandle()  const { return outputHandles_[rt_layout::OUT_BINDING_FH_BASE_EMISSION]; }
    ResourceHandle reflectionMvHandle()    const { return outputHandles_[rt_layout::OUT_BINDING_REFLECTION_MV]; }
    ResourceHandle motionVec3dHandle()     const { return outputHandles_[rt_layout::OUT_BINDING_MOTION_VEC_3D]; }
    ResourceHandle gbufferMetallicHandle() const { return outputHandles_[rt_layout::OUT_BINDING_GBUFFER_METALLIC]; }
    ResourceHandle positionViewSpaceHandle() const { return outputHandles_[rt_layout::OUT_BINDING_POSITION_VIEW_SPACE]; }

private:
    void buildPushConstants(RtPushConstants& pc, const FrameContext& ctx) const;

    // Dispatch SHARC resolve and ReSTIR spatial compute passes after the main RT trace.
    void dispatchComputePasses(VkCommandBuffer cmd, const PassResources& resources,
                               uint32_t w, uint32_t h, const FrameContext& ctx);

    EngineServices* services_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;

    // V2-full shader modules (22 stages when loaded from v2_full/).
    // By default the engine stays on the 3-stage v2_min path because the full
    // suite is still experimental and must be explicitly opted into.
    std::vector<vk2::ShaderModule> shaderModules_;
    bool fullShadersLoaded_ = false;   // true if all 22 v2_full shaders loaded

    vk2::RtPipeline rtPipeline_;
    vk2::ShaderBindingTable sbt_;

    RtDescriptorLayout layout_;          // V1-compatible 4-set layout
    vk2::DescriptorAllocator descAllocator_;

    // All 34 output image handles, indexed by set 3 binding number.
    std::array<ResourceHandle, OUTPUT_COUNT> outputHandles_{};

    AtmosphereAdapter* atmosphere_ = nullptr;
    bool initialized_ = false;
    float preExposure_ = 1.0f;  // auto-exposure feedback, updated per frame

    // Offline accumulation state (set by EngineApp via setAccumState each frame).
    uint32_t accumFrameCount_ = 0;
    bool offlineEnabled_ = false;
    float aperture_ = 0.0f;
    float focalDistance_ = 10.0f;
    uint32_t offlineBounces_ = 0;  // 0 = use config.rayBounces

    // Fallback 1x1 white texture + sampler for bindless array (set 0 binding 0).
    // Binding 0 is rewritten into each freshly allocated descriptor set; this flag
    // only suppresses repeated init logs after the first successful write.
    vk2::Image fallbackWhiteImage_;
    VkSampler fallbackSampler_ = VK_NULL_HANDLE;
    bool bindlessFallbackWritten_ = false;

    // Cached config snapshot, updated each frame via configure().
    ConfigSnapshot cachedConfig_{};

    // --- SHARC resolve compute pipeline ---
    // Push-constant-only pipeline (no descriptor sets). Dispatched after RT trace
    // when SHARC is enabled. One thread per hash entry (capacity / 64 workgroups).
    vk2::ShaderModule sharcResolveShader_;
    vk2::PipelineLayout sharcResolveLayout_;
    vk2::ComputePipeline sharcResolvePipeline_;
    bool sharcResolveReady_ = false;

    // --- ReSTIR spatial reuse compute pipeline ---
    // Uses its own descriptor set layout with 4 storage image bindings.
    // Dispatched after RT trace when ReSTIR is enabled.
    vk2::ShaderModule restirSpatialShader_;
    vk2::PipelineLayout restirSpatialLayout_;
    vk2::ComputePipeline restirSpatialPipeline_;
    vk2::DescriptorSetLayout restirSpatialDescLayout_;
    vk2::DescriptorAllocator restirSpatialDescAlloc_;
    bool restirSpatialReady_ = false;

    // Push constants for the ReSTIR spatial compute shader.
    struct RestirSpatialPC {
        int32_t width;
        int32_t height;
        int32_t spatialTaps;
        int32_t spatialRadius;
        int32_t temporalMClamp;
        int32_t wClamp;
    };

    // Push constants for the SHARC resolve compute shader.
    struct SharcResolvePC {
        float    cameraPositionPrevX;
        float    cameraPositionPrevY;
        float    cameraPositionPrevZ;
        uint32_t accumulationFrameNum;
        uint32_t staleFrameNumMax;
        uint32_t frameIndex;
        uint64_t hashEntriesBDA;
        uint64_t accumulationBDA;
        uint64_t resolvedBDA;
        float    cameraX;
        float    cameraY;
        float    cameraZ;
        float    sceneScale;
        uint32_t capacity;
        float    radianceScale;
        uint32_t enableAntiFirefly;
    };

    // Track previous camera pos for SHARC resolve stale entry eviction.
    float prevCameraX_ = 0.0f;
    float prevCameraY_ = 0.0f;
    float prevCameraZ_ = 0.0f;
    uint32_t sharcFrameCounter_ = 0;

    // Per-output metadata for table-driven registration and binding.
    struct OutputSpec {
        const char* name;
        VkFormat    format;
        uint32_t    binding;  // set 3 binding index
    };
    static const std::array<OutputSpec, OUTPUT_COUNT> kOutputSpecs;
};

} // namespace engine
