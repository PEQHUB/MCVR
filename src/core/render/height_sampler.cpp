#include "height_sampler.hpp"

#include <glm/glm.hpp>

// BT.2020 luminance weights (matches GPU shader)
static constexpr float LUM_R = 0.2627f;
static constexpr float LUM_G = 0.6780f;
static constexpr float LUM_B = 0.0593f;

float HeightSampler::sampleBilinear(const uint8_t *rgba, uint32_t width, uint32_t height,
                                    float u, float v, int channel) {
    // Mirror GPU pomSampleHeightBilinear: uv * texSize - 0.5, then bilinear
    float texelX = u * static_cast<float>(width) - 0.5f;
    float texelY = v * static_cast<float>(height) - 0.5f;

    float fx = texelX - std::floor(texelX);
    float fy = texelY - std::floor(texelY);

    int ix = static_cast<int>(std::floor(texelX));
    int iy = static_cast<int>(std::floor(texelY));

    int w = static_cast<int>(width);
    int h = static_cast<int>(height);

    auto clampFetch = [&](int cx, int cy) -> float {
        cx = std::clamp(cx, 0, w - 1);
        cy = std::clamp(cy, 0, h - 1);
        return static_cast<float>(rgba[(cy * w + cx) * 4 + channel]) / 255.0f;
    };

    float h00 = clampFetch(ix, iy);
    float h10 = clampFetch(ix + 1, iy);
    float h01 = clampFetch(ix, iy + 1);
    float h11 = clampFetch(ix + 1, iy + 1);

    float top = h00 + (h10 - h00) * fx;
    float bot = h01 + (h11 - h01) * fx;
    return top + (bot - top) * fy;
}

float HeightSampler::sampleLabPBR(const Textures::TextureRGBAData *normalRGBA,
                                  float u, float v, float uvMinX, float uvMinY,
                                  float uvMaxX, float uvMaxY) {
    if (!normalRGBA || normalRGBA->rgba.empty()) return 1.0f;

    // Clamp to tile bounds
    u = std::clamp(u, uvMinX, uvMaxX);
    v = std::clamp(v, uvMinY, uvMaxY);

    // Sample alpha channel (channel 3) of normal texture
    return sampleBilinear(normalRGBA->rgba.data(), normalRGBA->width, normalRGBA->height, u, v, 3);
}

float HeightSampler::sampleAutoPBR(const Textures::TextureRGBAData *albedoRGBA,
                                   float u, float v, float uvMinX, float uvMinY,
                                   float uvMaxX, float uvMaxY,
                                   int heightSourceMode,
                                   float lumMin, float lumSpan, bool invertH,
                                   float remapMin, float remapMax, float contrast, float offset) {
    if (!albedoRGBA || albedoRGBA->rgba.empty()) return 1.0f;

    u = std::clamp(u, uvMinX, uvMaxX);
    v = std::clamp(v, uvMinY, uvMaxY);

    // Sample raw channel value based on heightSourceMode
    float raw;
    switch (heightSourceMode) {
    case 1: // Red
        raw = sampleBilinear(albedoRGBA->rgba.data(), albedoRGBA->width, albedoRGBA->height, u, v, 0);
        break;
    case 2: // Green
        raw = sampleBilinear(albedoRGBA->rgba.data(), albedoRGBA->width, albedoRGBA->height, u, v, 1);
        break;
    case 3: // Blue
        raw = sampleBilinear(albedoRGBA->rgba.data(), albedoRGBA->width, albedoRGBA->height, u, v, 2);
        break;
    case 4: // Alpha
        raw = sampleBilinear(albedoRGBA->rgba.data(), albedoRGBA->width, albedoRGBA->height, u, v, 3);
        break;
    case 5: { // MaxRGB
        float r = sampleBilinear(albedoRGBA->rgba.data(), albedoRGBA->width, albedoRGBA->height, u, v, 0);
        float g = sampleBilinear(albedoRGBA->rgba.data(), albedoRGBA->width, albedoRGBA->height, u, v, 1);
        float b = sampleBilinear(albedoRGBA->rgba.data(), albedoRGBA->width, albedoRGBA->height, u, v, 2);
        raw = std::max({r, g, b});
        break;
    }
    case 6: { // MinRGB
        float r = sampleBilinear(albedoRGBA->rgba.data(), albedoRGBA->width, albedoRGBA->height, u, v, 0);
        float g = sampleBilinear(albedoRGBA->rgba.data(), albedoRGBA->width, albedoRGBA->height, u, v, 1);
        float b = sampleBilinear(albedoRGBA->rgba.data(), albedoRGBA->width, albedoRGBA->height, u, v, 2);
        raw = std::min({r, g, b});
        break;
    }
    default: { // 0 or 7: Luminance (BT.2020)
        float r = sampleBilinear(albedoRGBA->rgba.data(), albedoRGBA->width, albedoRGBA->height, u, v, 0);
        float g = sampleBilinear(albedoRGBA->rgba.data(), albedoRGBA->width, albedoRGBA->height, u, v, 1);
        float b = sampleBilinear(albedoRGBA->rgba.data(), albedoRGBA->width, albedoRGBA->height, u, v, 2);
        raw = r * LUM_R + g * LUM_G + b * LUM_B;
        break;
    }
    }

    // Height pipeline: normalize → invert → remap → contrast → offset
    float h = (lumSpan > 1e-6f) ? std::clamp((raw - lumMin) / lumSpan, 0.0f, 1.0f) : 0.5f;
    if (invertH) h = 1.0f - h;
    h = remapMin + (remapMax - remapMin) * h;
    if (contrast != 1.0f) h = std::pow(std::clamp(h, 0.001f, 1.0f), contrast);
    return std::clamp(h + offset, 0.0f, 1.0f);
}

float HeightSampler::sample(const Textures::TextureRGBAData *normalRGBA,
                            const Textures::TextureRGBAData *albedoRGBA,
                            const vk::Data::MaterialClassEntry *material,
                            bool hasLabPBRHeight, bool isAutoPBR,
                            float u, float v, float uvMinX, float uvMinY,
                            float uvMaxX, float uvMaxY) {
    // LabPBR has explicit height in normal alpha — use it directly
    if (hasLabPBRHeight && normalRGBA) {
        return sampleLabPBR(normalRGBA, u, v, uvMinX, uvMinY, uvMaxX, uvMaxY);
    }

    // AutoPBR: derive height from albedo using material pipeline
    if (isAutoPBR && material && albedoRGBA) {
        // Unpack height source from pomPacked0 bits 5-7
        int heightSource = (material->pomPacked0 >> 5) & 0x7;

        // Unpack height pipeline params from pomPacked1/2
        float heightContrast = static_cast<float>((material->pomPacked1 >> 24) & 0xFF) / 10.0f; // 0-25.5
        float remapMin = static_cast<float>((material->pomPacked2 >> 0) & 0xFF) / 100.0f;
        float remapMax = static_cast<float>((material->pomPacked2 >> 8) & 0xFF) / 100.0f;
        float heightOffset = (static_cast<float>((material->pomPacked2 >> 16) & 0xFF) - 100.0f) / 100.0f; // [-1, 1]
        bool invertH = (material->flags >> 6) & 0x1; // bit 6 = invertHeight

        float lumMin = material->lumMin;
        float lumMax = material->lumMax;
        float lumSpan = lumMax - lumMin;

        return sampleAutoPBR(albedoRGBA, u, v, uvMinX, uvMinY, uvMaxX, uvMaxY,
                             heightSource, lumMin, lumSpan, invertH,
                             remapMin, remapMax, heightContrast, heightOffset);
    }

    return 1.0f; // No height data — surface level
}
