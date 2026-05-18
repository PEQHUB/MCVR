#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(set = 1, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 1, binding = 1) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(set = 2, binding = 0) readonly buffer TextureMappingBuffer {
    TextureMapping mapping;
};

layout(location = 0) in vec3 pos;
layout(location = 1) flat in uint flags;
layout(location = 2) in vec3 norm;
layout(location = 3) in vec4 colorLayer;
layout(location = 4) in vec2 textureUV;
layout(location = 5) flat in uint textureID;
layout(location = 6) in vec2 glintUV;
layout(location = 7) flat in uint glintTexture;
layout(location = 8) in vec4 lightMapColor;
layout(location = 9) in vec4 overlayColor;

layout(location = 0) out vec4 fragColor;

void main() {
    // Unpack flags
    bool useTexture    = (flags & PBR_FLAG_USE_TEXTURE) != 0u;
    bool useColorLayer = (flags & PBR_FLAG_USE_COLOR_LAYER) != 0u;
    bool useOverlay    = (flags & PBR_FLAG_USE_OVERLAY) != 0u;
    bool useLight      = (flags & PBR_FLAG_USE_LIGHT) != 0u;

    vec4 color = vec4(0.0);
    float emission = 0.0;
    int specularTextureID = mapping.entries[textureID].specular;
    if (useTexture) {
        color = texture(textures[nonuniformEXT(textureID)], textureUV);
        if (specularTextureID >= 0) {
            emission = texture(textures[nonuniformEXT(specularTextureID)], textureUV).a;
            int intEmission = int(round(emission * 255.0));
            if (intEmission == 255) {
                emission = 0.0;
            } else {
                emission = intEmission / 254.0;
            }
        }
    }
    if (color.a < 0.1) { discard; }
    if (useColorLayer) { color *= colorLayer; }
    if (useOverlay) { color.rgb = mix(overlayColor.rgb, color.rgb, overlayColor.a); }

    if (emission == 0.0) {
        if (!useLight)
            fragColor = color;
        else
            fragColor = color * lightMapColor;
    } else {
        fragColor = color * emission;
    }

    float linearDepth = -(mat4(mat3(worldUBO.cameraEffectedViewMat)) * vec4(pos, 1.0)).z;
    gl_FragDepth = clamp(linearDepth / 1000.0, 0.0, 1.0);

    fragColor.a = 1.0;
}
