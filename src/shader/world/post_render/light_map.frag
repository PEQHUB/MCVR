#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 1, binding = 2) uniform LightMapUniform {
    LightMapUBO lightMapUbo;
};

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 fragColor;

float get_brightness(float level) {
    float curved_level = level / (4.0 - 3.0 * level);
    return mix(curved_level, 1.0, lightMapUbo.ambientLightFactor);
}

vec3 notGamma(vec3 x) {
    vec3 nx = 1.0 - x;
    return 1.0 - nx * nx * nx * nx;
}

vec3 custom_lightmap_color(int block_level, int sky_level) {
    vec3 block_color = lightMapUbo.customBlockLight[block_level].rgb;
    vec3 sky_color = lightMapUbo.customSkyLight[sky_level].rgb;
    return clamp(block_color + sky_color, 0.0, 1.0);
}

void main() {
    int block_level = clamp(int(floor(texCoord.x * 16)), 0, 15);
    int sky_level = clamp(int(floor(texCoord.y * 16)), 0, 15);

    vec3 color;
    if (lightMapUbo.customLightmapEnabled != 0) {
        color = custom_lightmap_color(block_level, sky_level);
    } else {
        float block_brightness = get_brightness(float(block_level) / 15) * lightMapUbo.blockFactor;
        float sky_brightness = get_brightness(float(sky_level) / 15) * lightMapUbo.skyFactor;

        // cubic nonsense, dips to yellowish in the middle, white when fully saturated
        color = vec3(
            block_brightness,
            block_brightness * ((block_brightness * 0.6 + 0.4) * 0.6 + 0.4),
            block_brightness * (block_brightness * block_brightness * 0.6 + 0.4)
        );

        if (lightMapUbo.useBrightLightmap != 0) {
            color = mix(color, vec3(0.99, 1.12, 1.0), 0.25);
            color = clamp(color, 0.0, 1.0);
        } else {
            color += lightMapUbo.skyLightColor * sky_brightness;
            color = mix(color, vec3(0.75), 0.04);

            vec3 darkened_color = color * vec3(0.7, 0.6, 0.6);
            color = mix(color, darkened_color, lightMapUbo.darkenWorldFactor);
        }
    }

    if (lightMapUbo.nightVisionFactor > 0.0
        && (lightMapUbo.customLightmapEnabled == 0 || lightMapUbo.customLightmapIncludesNightVision == 0)) {
        // scale up uniformly until 1.0 is hit by one of the colors
        float max_component = max(color.r, max(color.g, color.b));
        if (max_component < 1.0) {
            vec3 bright_color = color / max_component;
            color = mix(color, bright_color, lightMapUbo.nightVisionFactor);
        }
    }

    if (lightMapUbo.useBrightLightmap == 0) {
        color = clamp(color - vec3(lightMapUbo.darknessScale), 0.0, 1.0);
    }

    vec3 notGamma = notGamma(color);
    color = mix(color, notGamma, lightMapUbo.brightnessFactor);
    color = mix(color, vec3(0.75), 0.04);
    color = clamp(color, 0.0, 1.0);

    fragColor = vec4(color, 1.0);
}
