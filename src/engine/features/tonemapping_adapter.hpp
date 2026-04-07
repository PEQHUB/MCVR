#pragma once

// Ownership: Render graph (via PassDescriptor).
// Thread: execute() on Compute queue.
// Dependencies: ConfigService, DeviceService.
//
// Stages 1-2: histogram + exposure compute (PR29).
// Stage 3: fullscreen triangle tone curve (PR30).

#include "feature_adapter.hpp"
#include "platform/vulkan/vk2_shader.hpp"
#include "platform/vulkan/vk2_pipeline.hpp"
#include "platform/vulkan/vk2_buffer.hpp"
#include "platform/vulkan/vk2_image.hpp"
#include "platform/vulkan/vk2_descriptor.hpp"

#include <cstdint>

namespace engine {

class GraphBuilder;

// Push constants matching hist.comp / exposure.comp layout (47 floats = 188 bytes)
struct ToneMappingPushConstants {
    // Histogram parameters
    float log2Min = -12.0f;
    float log2Max = 18.0f;
    float epsilon = 1e-6f;
    float lowPercent = 0.10f;
    float highPercent = 0.90f;
    float middleGrey = 0.18f;
    float dt = 0.016f;
    float brightAdaptSpeed = 0.6f;
    float darkAdaptSpeed = 0.8f;
    float minExposure = 0.001f;
    float maxExposure = 100.0f;
    float sceneChangeThreshold = 3.0f;
    float centerWeightStrength = 0.0f;
    float tonemapMode = 0.0f;
    float Lwhite = 4.0f;
    // Exposure compensation
    float exposureCompensation = 0.0f;
    // HDR fields
    float hdrPipelineEnabled = 0.0f;
    float hdr10OutputEnabled = 0.0f;
    float peakNits = 1000.0f;
    float paperWhiteNits = 203.0f;
    float saturation = 1.3f;
    float sdrTransferFunction = 1.0f;
    float psychoPeakSDR = 2.0f;
    float manualExposureEnabled = 0.0f;
    float manualExposure = 1.0f;
    // PsychoV parameters
    float psychoEnabled = 1.0f;
    float psychoHighlights = 1.0f;
    float psychoShadows = 1.0f;
    float psychoContrast = 1.0f;
    float psychoPurity = 1.0f;
    float psychoBleaching = 0.0f;
    float psychoClipPoint = 1.0f;
    float psychoHueRestore = 1.0f;
    float psychoAdaptContrast = 1.0f;
    float psychoWhiteCurve = 0.0f;
    float psychoConeExponent = 0.43f;
    float saturationAdaptive = 0.0f;
    // Custom tonemap params
    float tonemapParam0 = 0.0f;
    float tonemapParam1 = 0.0f;
    float tonemapParam2 = 0.0f;
    float tonemapParam3 = 0.0f;
    float tonemapParam4 = 0.0f;
    float tonemapParam5 = 0.0f;
    float tonemapParam6 = 0.0f;
    float tonemapParam7 = 0.0f;
    // RT metering
    float preExposure = 1.0f;
    float highlightWeight = 0.0f;
};

class ToneMappingAdapter : public FeatureAdapter {
public:
    void init(EngineServices& services) override;
    void configure(const ConfigSnapshot& config) override;
    void execute(const FrameContext& ctx, const PassResources& resources) override;
    void shutdown() override;
    const char* name() const override { return "ToneMapping"; }

    // Register histogram + exposure compute pass with graph.
    // inputHandle: the HDR image to meter (e.g. test_compute_output).
    void registerPass(GraphBuilder& builder, ResourceHandle inputHandle);

    // Output handle for downstream passes (e.g. CAS post-process).
    ResourceHandle outputHandle() const { return outputHandle_; }

private:
    EngineServices* services_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;

    // Shaders & pipelines
    vk2::ShaderModule histShader_;
    vk2::ShaderModule exposureShader_;
    vk2::ShaderModule vertShader_;
    vk2::ShaderModule fragShader_;
    vk2::PipelineLayout pipelineLayout_;
    vk2::ComputePipeline histPipeline_;
    vk2::ComputePipeline exposurePipeline_;
    vk2::GraphicsPipeline graphicsPipeline_;

    // Buffers
    vk2::Buffer histogramBuffer_;    // 256 * uint32
    vk2::Buffer exposureBuffer_;     // 58 floats

    // Descriptor set layout & per-frame allocator
    VkDescriptorSetLayout descSetLayout_ = VK_NULL_HANDLE;
    vk2::DescriptorAllocator descAllocator_;

    // Sampler for input image
    VkSampler linearSampler_ = VK_NULL_HANDLE;

    // Dummy 1x1 image for unused sampler bindings
    vk2::Image dummyImage_;

    // Push constants (updated each frame from config)
    ToneMappingPushConstants pc_;

    ResourceHandle inputHandle_;
    ResourceHandle outputHandle_;
    bool initialized_ = false;
    bool exposureInitialized_ = false;
};

} // namespace engine
