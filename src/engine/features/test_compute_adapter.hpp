#pragma once

// Test adapter: dispatches clear_image.comp to write cycling RGB to a storage image.
// Proves: shader loading → compute pipeline → descriptor set → graph dispatch → composite.
// Delete this adapter once real adapters are wired.

#include "feature_adapter.hpp"
#include "platform/vulkan/vk2_shader.hpp"
#include "platform/vulkan/vk2_pipeline.hpp"
#include "platform/vulkan/vk2_descriptor.hpp"

namespace engine {

class TestComputeAdapter : public FeatureAdapter {
public:
    void init(EngineServices& services) override;
    void configure(const ConfigSnapshot& config) override;
    void execute(const FrameContext& ctx, const PassResources& resources) override;
    void shutdown() override;
    const char* name() const override { return "TestCompute"; }

    // Register this adapter's pass in a graph builder.
    // Returns the output resource handle (the image to composite).
    ResourceHandle registerPass(class GraphBuilder& builder);

private:
    EngineServices* services_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;

    vk2::ShaderModule shader_;
    vk2::PipelineLayout pipelineLayout_;
    vk2::ComputePipeline pipeline_;
    VkDescriptorSetLayout descSetLayout_ = VK_NULL_HANDLE;
    vk2::DescriptorAllocator descAllocator_;

    ResourceHandle outputHandle_;
    bool initialized_ = false;
};

} // namespace engine
