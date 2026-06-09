#include "height_sampler.hpp"

#include <glm/glm.hpp>

// BT.2020 luminance weights (matches GPU shader)
static constexpr float LUM_R = 0.2627f;
static constexpr float LUM_G = 0.6780f;
static constexpr float LUM_B = 0.0593f;

float HeightSampler::sampleNearest(const uint8_t *rgba, uint32_t width, uint32_t height,
                                   float u, float v, int channel) {
    int ix = static_cast<int>(std::floor(u * static_cast<float>(width)));
    int iy = static_cast<int>(std::floor(v * static_cast<float>(height)));
    ix = std::clamp(ix, 0, static_cast<int>(width) - 1);
    iy = std::clamp(iy, 0, static_cast<int>(height) - 1);
    return static_cast<float>(rgba[(iy * width + ix) * 4 + channel]) / 255.0f;
}

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

float HeightSampler::sampleFiltered(const uint8_t *rgba, uint32_t width, uint32_t height,
                                    float u, float v, int channel, int filterMode) {
    if (filterMode == FILTER_NEAREST) {
        return sampleNearest(rgba, width, height, u, v, channel);
    }
    return sampleBilinear(rgba, width, height, u, v, channel);
}

float HeightSampler::sample(const HeightLayerView& labPbrHeight,
                            const HeightLayerView& autoPbrHeight,
                            const Textures::TextureRGBAData *normalRGBA,
                            const Textures::TextureRGBAData *albedoRGBA,
                            const vk::Data::MaterialClassEntry *material,
                            bool hasLabPBRHeight, bool isAutoPBR,
                            float u, float v, float uvMinX, float uvMinY,
                            float uvMaxX, float uvMaxY) {
    int filterMode = material ? static_cast<int>(material->pomPacked0 & 0x7) : FILTER_BILINEAR;

    if (hasLabPBRHeight) {
        if (labPbrHeight.valid()) {
            return sampleLabPBR(labPbrHeight, u, v, uvMinX, uvMinY, uvMaxX, uvMaxY, filterMode);
        }
        if (normalRGBA) {
            return sampleLabPBR(normalRGBA, u, v, uvMinX, uvMinY, uvMaxX, uvMaxY, filterMode);
        }
    }

    if (isAutoPBR && material) {
        int heightSource = (material->pomPacked0 >> 6) & 0x7;
        float heightContrast = static_cast<float>((material->pomPacked1 >> 24) & 0xFF) / 10.0f;
        float remapMin = static_cast<float>((material->pomPacked2 >> 0) & 0xFF) / 100.0f;
        float remapMax = static_cast<float>((material->pomPacked2 >> 8) & 0xFF) / 100.0f;
        float heightOffset = (static_cast<float>((material->pomPacked2 >> 16) & 0xFF) - 100.0f) / 100.0f;
        bool invertH = (material->flags >> 6) & 0x1;
        float lumMin = material->lumMin;
        float lumMax = material->lumMax;
        float lumSpan = lumMax - lumMin;

        if (autoPbrHeight.valid() && (heightSource == 0 || heightSource == 7)) {
            return sampleAutoPBR(autoPbrHeight, u, v, uvMinX, uvMinY, uvMaxX, uvMaxY,
                                 filterMode, lumMin, lumSpan, invertH, remapMin, remapMax,
                                 heightContrast, heightOffset);
        }
        if (albedoRGBA) {
            return sampleAutoPBR(albedoRGBA, u, v, uvMinX, uvMinY, uvMaxX, uvMaxY,
                                 heightSource, filterMode, lumMin, lumSpan, invertH,
                                 remapMin, remapMax, heightContrast, heightOffset);
        }
    }

    return 1.0f;
}

float HeightSampler::sampleFilteredPlane(const uint8_t *values, uint32_t width, uint32_t height,
                                         float u, float v, int filterMode) {
    if (!values || width == 0 || height == 0) return 1.0f;

    if (filterMode == FILTER_NEAREST) {
        int ix = static_cast<int>(std::floor(u * static_cast<float>(width)));
        int iy = static_cast<int>(std::floor(v * static_cast<float>(height)));
        ix = std::clamp(ix, 0, static_cast<int>(width) - 1);
        iy = std::clamp(iy, 0, static_cast<int>(height) - 1);
        return static_cast<float>(values[iy * width + ix]) / 255.0f;
    }

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
        return static_cast<float>(values[cy * w + cx]) / 255.0f;
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
                                  float uvMaxX, float uvMaxY, int filterMode) {
    if (!normalRGBA || normalRGBA->rgba.empty()) return 1.0f;

    u = std::clamp(u, uvMinX, uvMaxX);
    v = std::clamp(v, uvMinY, uvMaxY);

    return sampleFiltered(normalRGBA->rgba.data(), normalRGBA->width, normalRGBA->height, u, v, 3, filterMode);
}

float HeightSampler::sampleLabPBR(const HeightLayerView& heightLayer,
                                  float u, float v, float uvMinX, float uvMinY,
                                  float uvMaxX, float uvMaxY, int filterMode) {
    if (!heightLayer.valid()) return 1.0f;

    u = std::clamp(u, uvMinX, uvMaxX);
    v = std::clamp(v, uvMinY, uvMaxY);

    return sampleFilteredPlane(heightLayer.values->data(), heightLayer.width, heightLayer.height,
                               u, v, filterMode);
}

float HeightSampler::sampleAutoPBR(const Textures::TextureRGBAData *albedoRGBA,
                                   float u, float v, float uvMinX, float uvMinY,
                                   float uvMaxX, float uvMaxY,
                                   int heightSourceMode, int filterMode,
                                   float lumMin, float lumSpan, bool invertH,
                                   float remapMin, float remapMax, float contrast, float offset) {
    if (!albedoRGBA || albedoRGBA->rgba.empty()) return 1.0f;

    u = std::clamp(u, uvMinX, uvMaxX);
    v = std::clamp(v, uvMinY, uvMaxY);

    const uint8_t *data = albedoRGBA->rgba.data();
    uint32_t tw = albedoRGBA->width, th = albedoRGBA->height;

    // Sample raw channel value based on heightSourceMode, using selected filter
    float raw;
    switch (heightSourceMode) {
    case 1: raw = sampleFiltered(data, tw, th, u, v, 0, filterMode); break; // Red
    case 2: raw = sampleFiltered(data, tw, th, u, v, 1, filterMode); break; // Green
    case 3: raw = sampleFiltered(data, tw, th, u, v, 2, filterMode); break; // Blue
    case 4: raw = sampleFiltered(data, tw, th, u, v, 3, filterMode); break; // Alpha
    case 5: { // MaxRGB
        float r = sampleFiltered(data, tw, th, u, v, 0, filterMode);
        float g = sampleFiltered(data, tw, th, u, v, 1, filterMode);
        float b = sampleFiltered(data, tw, th, u, v, 2, filterMode);
        raw = std::max({r, g, b});
        break;
    }
    case 6: { // MinRGB
        float r = sampleFiltered(data, tw, th, u, v, 0, filterMode);
        float g = sampleFiltered(data, tw, th, u, v, 1, filterMode);
        float b = sampleFiltered(data, tw, th, u, v, 2, filterMode);
        raw = std::min({r, g, b});
        break;
    }
    default: { // 0 or 7: Luminance (BT.2020)
        float r = sampleFiltered(data, tw, th, u, v, 0, filterMode);
        float g = sampleFiltered(data, tw, th, u, v, 1, filterMode);
        float b = sampleFiltered(data, tw, th, u, v, 2, filterMode);
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

float HeightSampler::sampleAutoPBR(const HeightLayerView& heightLayer,
                                   float u, float v, float uvMinX, float uvMinY,
                                   float uvMaxX, float uvMaxY,
                                   int filterMode, float lumMin, float lumSpan,
                                   bool invertH, float remapMin, float remapMax,
                                   float contrast, float offset) {
    if (!heightLayer.valid()) return 1.0f;

    u = std::clamp(u, uvMinX, uvMaxX);
    v = std::clamp(v, uvMinY, uvMaxY);

    float raw = sampleFilteredPlane(heightLayer.values->data(), heightLayer.width, heightLayer.height,
                                    u, v, filterMode);
    float h = (lumSpan > 1e-6f) ? std::clamp((raw - lumMin) / lumSpan, 0.0f, 1.0f) : raw;
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
    // Extract filter mode from pomPacked0 bits 0-2 (new layout)
    int filterMode = material ? static_cast<int>(material->pomPacked0 & 0x7) : FILTER_BILINEAR;

    // LabPBR has explicit height in normal alpha — use it directly
    if (hasLabPBRHeight && normalRGBA) {
        return sampleLabPBR(normalRGBA, u, v, uvMinX, uvMinY, uvMaxX, uvMaxY, filterMode);
    }

    // AutoPBR: derive height from albedo using material pipeline
    if (isAutoPBR && material && albedoRGBA) {
        // When lumMin/lumMax are uninitialized defaults (0/1), use raw luminance directly
        // instead of bailing. This allows displacement to work even before AutoPBR
        // histogram is computed — the absolute luminance becomes the height.

        // Unpack height source from pomPacked0 bits 6-8 (new layout)
        int heightSource = (material->pomPacked0 >> 6) & 0x7;

        // Unpack height pipeline params from pomPacked1/2 (unchanged layout)
        float heightContrast = static_cast<float>((material->pomPacked1 >> 24) & 0xFF) / 10.0f;
        float remapMin = static_cast<float>((material->pomPacked2 >> 0) & 0xFF) / 100.0f;
        float remapMax = static_cast<float>((material->pomPacked2 >> 8) & 0xFF) / 100.0f;
        float heightOffset = (static_cast<float>((material->pomPacked2 >> 16) & 0xFF) - 100.0f) / 100.0f;
        bool invertH = (material->flags >> 6) & 0x1; // bit 6 = invertHeight

        float lumMin = material->lumMin;
        float lumMax = material->lumMax;
        float lumSpan = lumMax - lumMin;

        return sampleAutoPBR(albedoRGBA, u, v, uvMinX, uvMinY, uvMaxX, uvMaxY,
                             heightSource, filterMode, lumMin, lumSpan, invertH,
                             remapMin, remapMax, heightContrast, heightOffset);
    }

    return 1.0f; // No height data — surface level
}
