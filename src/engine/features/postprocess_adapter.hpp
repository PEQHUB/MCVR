#pragma once

// PostProcess adapter: bloom + CAS (Contrast-Adaptive Sharpening).
//
// Pipeline: Input (tone-mapped LDR)
//   -> bloom_downsample  -> bloomHalfRes_        (half-res bright extract)
//   -> bloom_composite   -> bloomIntermediate_   (full-res bloom + original)
//   -> CAS               -> Output               (sharpened LDR)
//
// When bloom is disabled, CAS reads directly from the graph input.
// When CAS is disabled (sharpenerMode==0), bloom composite writes to output.

#include "feature_adapter.hpp"
#include "platform/vulkan/vk2_shader.hpp"
#include "platform/vulkan/vk2_pipeline.hpp"
#include "platform/vulkan/vk2_descriptor.hpp"
#include "platform/vulkan/vk2_image.hpp"

#include <array>
#include <cstdint>

namespace engine {

class GraphBuilder;

struct CasPushConstants {
    std::array<uint32_t, 4> const0{};
    std::array<uint32_t, 4> const1{};
};

struct BloomPushConstants {
    float threshold;
    float intensity;
    float texelSizeX;
    float texelSizeY;
};

class PostProcessAdapter : public FeatureAdapter {
public:
    void init(EngineServices& services) override;
    void configure(const ConfigSnapshot& config) override;
    void execute(const FrameContext& ctx, const PassResources& resources) override;
    void shutdown() override;
    const char* name() const override { return "PostProcess"; }

    void registerPass(GraphBuilder& builder, ResourceHandle inputHandle);

    ResourceHandle outputHandle() const { return outputHandle_; }

private:
    EngineServices* services_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;

    // CAS
    vk2::ShaderModule casShader_;
    vk2::PipelineLayout casPipelineLayout_;
    vk2::ComputePipeline casPipeline_;
    VkDescriptorSetLayout casDescSetLayout_ = VK_NULL_HANDLE;

    // Bloom downsample
    vk2::ShaderModule bloomDownShader_;
    vk2::PipelineLayout bloomDownPipelineLayout_;
    vk2::ComputePipeline bloomDownPipeline_;
    VkDescriptorSetLayout bloomDownDescSetLayout_ = VK_NULL_HANDLE;

    // Bloom composite
    vk2::ShaderModule bloomCompShader_;
    vk2::PipelineLayout bloomCompPipelineLayout_;
    vk2::ComputePipeline bloomCompPipeline_;
    VkDescriptorSetLayout bloomCompDescSetLayout_ = VK_NULL_HANDLE;

    // Shared descriptor allocator (covers all passes)
    vk2::DescriptorAllocator descAllocator_;

    VkSampler linearSampler_ = VK_NULL_HANDLE;

    // Adapter-owned images (not graph resources)
    vk2::Image bloomHalfRes_;        // half-res bloom buffer (rgba16f)
    vk2::Image bloomIntermediate_;   // full-res intermediate for bloom+CAS chain (rgba16)
    uint32_t lastRenderWidth_ = 0;
    uint32_t lastRenderHeight_ = 0;

    float sharpness_ = 0.5f;
    uint32_t sharpenerMode_ = 0;
    bool bloomEnabled_ = true;
    float bloomIntensity_ = 0.15f;
    float bloomThreshold_ = 0.8f;

    ResourceHandle inputHandle_;
    ResourceHandle outputHandle_;
    bool initialized_ = false;
    bool bloomInitialized_ = false;

    void ensureBloomImages(uint32_t fullWidth, uint32_t fullHeight);
    void executeBloom(VkCommandBuffer cmd, const PassResources& resources,
                      uint32_t inIdx, uint32_t outIdx, uint32_t frameIndex);
};

} // namespace engine
