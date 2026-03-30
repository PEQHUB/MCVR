#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"
#include <chrono>

#include "core/render/modules/world/world_module.hpp"

class Framework;
class FrameworkContext;
class WorldPipeline;
struct WorldModuleContext;

struct ToneMappingModuleContext;

struct ToneMappingModuleExposureData {
    float exposure;
    float avgLogLum;
    float tonemapMode;           // 0.0 = PBR Neutral, 1.0 = Reinhard Extended, etc.
    float Lwhite;                // White point for Reinhard Extended (default 4.0)
    float exposureCompensation;  // EV offset applied after auto-exposure
    // HDR fields (appended at end)
    float hdrPipelineEnabled;    // 0.0 = SDR pipeline, 1.0 = HDR pipeline behavior
    float hdr10OutputEnabled;    // 0.0 = SDR output, 1.0 = HDR10 output encoding
    float peakNits;              // Display peak brightness (e.g. 1000.0)
    float paperWhiteNits;        // ITU-R BT.2408 reference white (e.g. 203.0)
    float saturation;            // Saturation boost (1.0 = neutral)
    float sdrTransferFunction;   // 0.0 = Gamma 2.2, 1.0 = sRGB
    float _pad0;                 // padding (was capExposureSmoothed)
    float manualExposureEnabled; // 0.0 = auto exposure, 1.0 = manual exposure
    float manualExposure;        // direct exposure multiplier when manual is enabled
    // PsychoV tonemapper parameters
    float psychoEnabled;         // HDR tonemapper: 0.0 = PsychoVisual, 1.0 = BT.2390 EETF
    float psychoHighlights;
    float psychoShadows;
    float psychoContrast;
    float psychoPurity;
    float psychoBleaching;
    float psychoClipPoint;
    float psychoHueRestore;
    float psychoAdaptContrast;
    float psychoWhiteCurve;      // 0.0 = Neutwo, 1.0 = Naka-Rushton
    float psychoConeExponent;
    float saturationAdaptive;    // 0.0 = linear chroma multiply, 1.0 = adaptive (Special K style)
    float tonemapParam0;
    float tonemapParam1;
    float tonemapParam2;
    float tonemapParam3;
    float tonemapParam4;
    float tonemapParam5;
    float tonemapParam6;
    float tonemapParam7;
    float _pad1;                 // padding (was bootTimer)
};

struct ToneMappingModulePushConstant {
    float log2Min;              // e.g. -12
    float log2Max;              // e.g. +18
    float epsilon;              // e.g. 1e-6
    float lowPercent;           // trimmed percentile low (0.05)
    float highPercent;          // trimmed percentile high (0.95)
    float middleGrey;           // e.g. 0.18
    float dt;                   // frame delta time (seconds)
    float brightAdaptSpeed;     // exponential decay tau for bright adaptation (seconds)
    float darkAdaptSpeed;       // exponential decay tau for dark adaptation (seconds)
    float minExposure;          // safety clamp lower bound
    float maxExposure;          // safety clamp upper bound
    float sceneChangeThreshold; // EV diff triggering instant snap
    float centerWeightStrength; // center-weighted metering strength (0-1)
    float tonemapMode;          // 0.0 = PBR Neutral, 1.0 = Reinhard Extended, etc.
    float Lwhite;               // White point for Reinhard Extended (default 4.0)
    float exposureCompensation; // EV offset applied after auto-exposure
    // HDR fields (appended at end)
    float hdrPipelineEnabled;   // 0.0 = SDR pipeline, 1.0 = HDR pipeline behavior
    float hdr10OutputEnabled;   // 0.0 = SDR output, 1.0 = HDR10 output encoding
    float peakNits;             // Display peak brightness
    float paperWhiteNits;       // ITU-R BT.2408 reference white
    float saturation;           // Saturation boost
    float sdrTransferFunction;  // 0.0 = Gamma 2.2, 1.0 = sRGB
    float manualExposureEnabled; // 0.0 = auto exposure, 1.0 = manual exposure
    float manualExposure;        // direct exposure multiplier when manual is enabled
    // PsychoV tonemapper parameters
    float psychoEnabled;
    float psychoHighlights;
    float psychoShadows;
    float psychoContrast;
    float psychoPurity;
    float psychoBleaching;
    float psychoClipPoint;
    float psychoHueRestore;
    float psychoAdaptContrast;
    float psychoWhiteCurve;
    float psychoConeExponent;
    float saturationAdaptive;    // 0.0 = linear chroma multiply, 1.0 = adaptive
    float tonemapParam0;
    float tonemapParam1;
    float tonemapParam2;
    float tonemapParam3;
    float tonemapParam4;
    float tonemapParam5;
    float tonemapParam6;
    float tonemapParam7;
    float preExposure;  // RT pre-exposure value — histogram must undo this for correct metering
};

class ToneMappingModule : public WorldModule, public SharedObject<ToneMappingModule> {
    friend ToneMappingModuleContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.tone_mapping.name";
    constexpr static uint32_t inputImageNum = 1;
    constexpr static uint32_t outputImageNum = 1;

    ToneMappingModule();

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline);

    bool setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                std::vector<VkFormat> &formats,
                                uint32_t frameIndex) override;
    bool setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                 std::vector<VkFormat> &formats,
                                 uint32_t frameIndex) override;

    void setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) override;

    void build() override;

    std::vector<std::shared_ptr<WorldModuleContext>> &contexts() override;

    void
    bindTexture(std::shared_ptr<vk::Sampler> sampler, std::shared_ptr<vk::DeviceLocalImage> image, int index) override;

    void preClose() override;

  private:
    static constexpr uint32_t histSize = 256;

    void initDescriptorTables();
    void initImages();
    void initBuffers();
    void initRenderPass();
    void initFrameBuffers();
    void initPipeline();

  private:
    // input
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> hdrImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> emissionImages_;
    std::shared_ptr<vk::Sampler> emissionSampler_;
    std::shared_ptr<vk::Sampler> renderResHdrSampler_;  // NEAREST sampler for render-res histogram metering

    // tone mapping
    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables_;

    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> histBuffers_;
    std::shared_ptr<vk::DeviceLocalBuffer> exposureData_;
    std::shared_ptr<vk::HostVisibleBuffer> exposureReadback_;  // 4-byte staging for GPU→CPU readback
    float computedExposure_ = 0.001f;                            // CPU-side mirror, 1-frame delayed (neutral midpoint)
    bool pendingExposureReset_ = false;  // deferred GPU buffer zero on world load

    std::shared_ptr<vk::Shader> histShader_;
    std::shared_ptr<vk::ComputePipeline> histPipeline_;

    std::shared_ptr<vk::Shader> exposureShader_;
    std::shared_ptr<vk::ComputePipeline> exposurePipeline_;

    std::shared_ptr<vk::Shader> vertShader_;
    std::shared_ptr<vk::Shader> fragShader_;
    std::shared_ptr<vk::RenderPass> renderPass_;
    std::vector<std::shared_ptr<vk::Framebuffer>> framebuffers_;
    std::shared_ptr<vk::GraphicsPipeline> pipeline_;
    std::vector<std::shared_ptr<vk::Sampler>> samplers_;

    float middleGrey_ = 0.18f;
    float tonemapMode_ = 1.0f;  // default: Reinhard Extended
    float Lwhite_ = 4.0f;

    // output
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> ldrImages_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;

    std::chrono::time_point<std::chrono::high_resolution_clock> lastTimePoint_;

    uint32_t width_, height_;
};

struct ToneMappingModuleContext : public WorldModuleContext, SharedObject<ToneMappingModuleContext> {
    std::weak_ptr<ToneMappingModule> toneMappingModule;

    // input
    std::shared_ptr<vk::DeviceLocalImage> hdrImage;

    // tone mapping
    std::shared_ptr<vk::DescriptorTable> descriptorTable;
    std::shared_ptr<vk::Framebuffer> framebuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> histBuffer;

    // output
    std::shared_ptr<vk::DeviceLocalImage> ldrImage;

    ToneMappingModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                             std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                             std::shared_ptr<ToneMappingModule> toneMappingModule);

    void render() override;
};
