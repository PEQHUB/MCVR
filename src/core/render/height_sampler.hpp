#pragma once

#include "common/shared.hpp"
#include "core/render/textures.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

struct HeightLayerView {
    uint32_t width = 0;
    uint32_t height = 0;
    std::shared_ptr<const std::vector<uint8_t>> values;

    bool valid() const {
        return values && width > 0 && height > 0 &&
               values->size() >= static_cast<size_t>(width) * height;
    }
};

// CPU-side height sampling that mirrors the GPU pipeline (pom.glsl).
// Used during chunk BLAS build to displace tessellated vertices.
class HeightSampler {
  public:
    // Filter modes (matches pomPacked0 bits 0-2)
    static constexpr int FILTER_NEAREST = 0;
    static constexpr int FILTER_BILINEAR = 1;
    static constexpr int FILTER_BICUBIC = 2;

    // Nearest-neighbor sampling: texelFetch at floor(uv * texSize).
    // Each texel is a flat plateau — sharp per-pixel extrusions.
    static float sampleNearest(const uint8_t *rgba, uint32_t width, uint32_t height,
                               float u, float v, int channel);

    // Bilinear interpolation of a single channel from RGBA data.
    // UV in [0,1] texture space, channel: 0=R, 1=G, 2=B, 3=A.
    static float sampleBilinear(const uint8_t *rgba, uint32_t width, uint32_t height,
                                float u, float v, int channel);

    // Filter-mode dispatch: calls sampleNearest or sampleBilinear.
    static float sampleFiltered(const uint8_t *rgba, uint32_t width, uint32_t height,
                                float u, float v, int channel, int filterMode);
    static float sampleFilteredPlane(const uint8_t *values, uint32_t width, uint32_t height,
                                     float u, float v, int filterMode);

    // LabPBR height: sample normal texture alpha channel.
    static float sampleLabPBR(const Textures::TextureRGBAData *normalRGBA,
                              float u, float v, float uvMinX, float uvMinY,
                              float uvMaxX, float uvMaxY, int filterMode = FILTER_BILINEAR);
    static float sampleLabPBR(const HeightLayerView& heightLayer,
                              float u, float v, float uvMinX, float uvMinY,
                              float uvMaxX, float uvMaxY, int filterMode = FILTER_BILINEAR);

    // AutoPBR height: derives height from albedo luminance pipeline.
    static float sampleAutoPBR(const Textures::TextureRGBAData *albedoRGBA,
                               float u, float v, float uvMinX, float uvMinY,
                               float uvMaxX, float uvMaxY,
                               int heightSourceMode, int filterMode,
                               float lumMin, float lumSpan, bool invertH,
                               float remapMin, float remapMax, float contrast, float offset);
    static float sampleAutoPBR(const HeightLayerView& heightLayer,
                               float u, float v, float uvMinX, float uvMinY,
                               float uvMaxX, float uvMaxY,
                               int filterMode, float lumMin, float lumSpan,
                               bool invertH, float remapMin, float remapMax,
                               float contrast, float offset);

    // Unified: sample height using material class params.
    // Returns height in [0,1] where 1.0 = surface, 0.0 = max depth.
    // Reads heightFilter and heightSource from pomPacked0 (new bit layout).
    static float sample(const Textures::TextureRGBAData *normalRGBA,
                        const Textures::TextureRGBAData *albedoRGBA,
                        const vk::Data::MaterialClassEntry *material,
                        bool hasLabPBRHeight, bool isAutoPBR,
                        float u, float v, float uvMinX, float uvMinY,
                        float uvMaxX, float uvMaxY);
    static float sample(const HeightLayerView& labPbrHeight,
                        const HeightLayerView& autoPbrHeight,
                        const Textures::TextureRGBAData *normalRGBA,
                        const Textures::TextureRGBAData *albedoRGBA,
                        const vk::Data::MaterialClassEntry *material,
                        bool hasLabPBRHeight, bool isAutoPBR,
                        float u, float v, float uvMinX, float uvMinY,
                        float uvMaxX, float uvMaxY);
};
