#version 460

// V2 post-entity vertex shader.
// Rasterizes particles, hand items, weather, and other post-RT entities
// on top of the tone-mapped output with depth testing against the RT depth buffer.
//
// Input: PBRTriangle vertex format (96 bytes, 12 attributes).
// Uses push constants for view/projection matrices since we only need
// a subset of WorldUBO and push constants avoid descriptor complexity.

// --- Push constants ---
layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    mat4 view;
    vec4 cameraPos;   // .xyz = camera position, .w = unused
} pc;

// --- PBRTriangle vertex inputs (96-byte stride, matches V1 layout) ---
layout(location = 0) in vec3 inPos;
layout(location = 1) in uint inFlags;
layout(location = 2) in vec3 inNorm;
layout(location = 3) in float inAlbedoEmission;
layout(location = 4) in vec4 inColorLayer;
layout(location = 5) in vec4 inPostBase;          // .xyz = entity origin, .w = emissiveBlockType (as float bits)
layout(location = 6) in vec2 inTextureUV;
layout(location = 7) in vec2 inGlintUV;
layout(location = 8) in uint inTextureID;
layout(location = 9) in uint inGlintTexture;
layout(location = 10) in uint inOverlayPacked;
layout(location = 11) in uint inLightPacked;

// --- Outputs to fragment shader ---
layout(location = 0) out vec4 outColorLayer;
layout(location = 1) out vec2 outTextureUV;
layout(location = 2) out flat uint outTextureID;
layout(location = 3) out flat uint outFlags;
layout(location = 4) out flat float outAlbedoEmission;

void main() {
    // Coordinate system from flags bits 8-10
    uint coordMode = (inFlags >> 8u) & 0x7u;

    vec3 worldPos;
    if (coordMode == 1u) {
        // CAMERA space: vertex position is in view/camera space.
        // Transform back to world space via inverse view (stored as viewInv).
        // postBase.xyz is the camera-relative entity origin.
        vec3 cameraSpacePos = inPos + inPostBase.xyz;
        worldPos = (inverse(pc.view) * vec4(cameraSpacePos, 1.0)).xyz;
    } else if (coordMode == 2u) {
        // CAMERA_SHIFT: Like CAMERA, but the entity origin is shifted by camera position.
        // The vertex is in camera space, origin is world-space.
        vec3 shifted = inPos + (inPostBase.xyz - pc.cameraPos.xyz);
        worldPos = (inverse(pc.view) * vec4(shifted, 1.0)).xyz;
    } else {
        // WORLD (coordMode == 0): direct world position
        worldPos = inPos + inPostBase.xyz;
    }

    // Transform to clip space
    gl_Position = pc.viewProj * vec4(worldPos, 1.0);

    // Compute linear depth for hardware depth test.
    // The C2D adapter wrote: gl_FragDepth = clamp(linearDepth / 1000.0, 0.0, 1.0)
    // We must produce compatible depth values.
    float viewZ = -(pc.view * vec4(worldPos, 1.0)).z;

    // Pass through attributes
    outColorLayer = inColorLayer;
    outTextureUV = inTextureUV;
    outTextureID = inTextureID;
    outFlags = inFlags;
    outAlbedoEmission = inAlbedoEmission;
}
