#version 460

// V2 post-entity fragment shader.
// Samples the texture atlas, applies vertex color modulation and alpha test,
// and outputs pre-multiplied alpha color for blending onto the tone-mapped image.
//
// Depth is written as linearDepth / 1000.0 to match the C2D depth convention.
// The pipeline uses LESS_OR_EQUAL depth test with depth write DISABLED,
// so entities are tested against the RT world depth but don't occlude each other
// in a z-fighting way (painter's order from Java submission).

// --- Descriptor set 0: Textures ---
layout(set = 0, binding = 0) uniform sampler2DArray texAtlas;

// --- Inputs from vertex shader ---
layout(location = 0) in vec4 inColorLayer;
layout(location = 1) in vec2 inTextureUV;
layout(location = 2) in flat uint inTextureID;
layout(location = 3) in flat uint inFlags;
layout(location = 4) in flat float inAlbedoEmission;

// --- Output ---
layout(location = 0) out vec4 fragColor;

// PBR flag constants (mirroring common/shared.hpp)
#define PBR_FLAG_USE_COLOR_LAYER  (1u << 1)
#define PBR_FLAG_USE_TEXTURE      (1u << 2)

void main() {
    vec4 texColor = vec4(1.0);

    // Sample texture atlas if the USE_TEXTURE flag is set
    if ((inFlags & PBR_FLAG_USE_TEXTURE) != 0u) {
        texColor = texture(texAtlas, vec3(inTextureUV, float(inTextureID)));
    }

    // Apply vertex color modulation if USE_COLOR_LAYER flag is set
    vec4 color = texColor;
    if ((inFlags & PBR_FLAG_USE_COLOR_LAYER) != 0u) {
        color.rgb *= inColorLayer.rgb;
        color.a *= inColorLayer.a;
    }

    // Alpha test: discard nearly transparent fragments
    if (color.a < 0.004) {
        discard;
    }

    // Apply emissive brightness boost if the vertex has emission
    if (inAlbedoEmission > 0.0) {
        color.rgb *= (1.0 + inAlbedoEmission * 2.0);
    }

    fragColor = color;
}
