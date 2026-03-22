#pragma once

#include "common/shared.hpp"
#include "core/render/textures.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

// CPU-side height sampling that mirrors the GPU pipeline (pom.glsl).
// Used during chunk BLAS build to displace tessellated vertices.
class HeightSampler {
  public:
    // Bilinear interpolation of a single channel from RGBA data.
    // UV in [0,1] texture space, channel: 0=R, 1=G, 2=B, 3=A.
    static float sampleBilinear(const uint8_t *rgba, uint32_t width, uint32_t height,
                                float u, float v, int channel);

    // LabPBR height: bilinear sample of normal texture alpha channel.
    // normalRGBA = the normal texture's RGBA data.
    // uv in atlas-space [0,1], uvMin/uvMax = tile bounds.
    static float sampleLabPBR(const Textures::TextureRGBAData *normalRGBA,
                              float u, float v, float uvMinX, float uvMinY,
                              float uvMaxX, float uvMaxY);

    // AutoPBR height: derives height from albedo luminance pipeline.
    // Mirrors pomSampleHeightAutoPBR from pom.glsl exactly.
    static float sampleAutoPBR(const Textures::TextureRGBAData *albedoRGBA,
                               float u, float v, float uvMinX, float uvMinY,
                               float uvMaxX, float uvMaxY,
                               int heightSourceMode,
                               float lumMin, float lumSpan, bool invertH,
                               float remapMin, float remapMax, float contrast, float offset);

    // Unified: sample height using material class params.
    // Returns height in [0,1] where 1.0 = surface, 0.0 = max depth.
    // For LabPBR: samples normal alpha. For AutoPBR: luminance pipeline.
    static float sample(const Textures::TextureRGBAData *normalRGBA,
                        const Textures::TextureRGBAData *albedoRGBA,
                        const vk::Data::MaterialClassEntry *material,
                        bool hasLabPBRHeight, bool isAutoPBR,
                        float u, float v, float uvMinX, float uvMinY,
                        float uvMaxX, float uvMaxY);
};
