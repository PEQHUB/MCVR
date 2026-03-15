// Parallax Occlusion Mapping with secant refinement and self-shadowing.
// Operates in atlas UV space; tile boundaries prevent cross-tile bleeding.

// Manual bilinear interpolation for height sampling.
// Minecraft textures use NEAREST filtering — hard pixel edges cause POM to produce
// discontinuous UV offsets (visible as "sliding" textures). Bilinear gives smooth
// height data even from 16x16 textures, making POM temporally stable.
float pomSampleHeightBilinear(vec2 uv, int texID, int channel) {
    vec2 texSize = vec2(textureSize(textures[nonuniformEXT(texID)], 0));
    vec2 texelCoord = uv * texSize - 0.5;
    vec2 f = fract(texelCoord);
    ivec2 i = ivec2(floor(texelCoord));
    // Clamp to atlas bounds (no wrapping — tile boundaries handled by POM clamp)
    ivec2 size = ivec2(texSize);
    ivec2 i00 = clamp(i, ivec2(0), size - 1);
    ivec2 i10 = clamp(i + ivec2(1, 0), ivec2(0), size - 1);
    ivec2 i01 = clamp(i + ivec2(0, 1), ivec2(0), size - 1);
    ivec2 i11 = clamp(i + ivec2(1, 1), ivec2(0), size - 1);
    float h00, h10, h01, h11;
    if (channel == 0) { // R channel (dedicated height tex)
        h00 = texelFetch(textures[nonuniformEXT(texID)], i00, 0).r;
        h10 = texelFetch(textures[nonuniformEXT(texID)], i10, 0).r;
        h01 = texelFetch(textures[nonuniformEXT(texID)], i01, 0).r;
        h11 = texelFetch(textures[nonuniformEXT(texID)], i11, 0).r;
    } else { // W/A channel (normal alpha)
        h00 = texelFetch(textures[nonuniformEXT(texID)], i00, 0).w;
        h10 = texelFetch(textures[nonuniformEXT(texID)], i10, 0).w;
        h01 = texelFetch(textures[nonuniformEXT(texID)], i01, 0).w;
        h11 = texelFetch(textures[nonuniformEXT(texID)], i11, 0).w;
    }
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// Sample height from either a dedicated R-channel height texture or the normal alpha channel.
float pomSampleHeight(vec2 uv, int heightTexID, int normalTexID) {
    if (heightTexID >= 0) {
        return pomSampleHeightBilinear(uv, heightTexID, 0);
    }
    return pomSampleHeightBilinear(uv, normalTexID, 3);
}

vec2 parallaxOcclusionMapping(
    vec2 uv,             // Atlas UV at hit point
    vec3 viewDirTS,      // View direction in tangent space (z > 0 = facing surface)
    int normalTexID,     // Normal texture index (height in alpha channel)
    float heightScale,   // Depth scale in tile-local [0,1] units
    int steps,           // Linear search steps (8-512)
    int refinement,      // Binary refinement iterations (unused, kept for API compat)
    vec2 uvMin,          // Tile boundary min (atlas UV)
    vec2 uvMax,          // Tile boundary max (atlas UV)
    int heightTexID,     // Blender PBR dedicated height texture (-1 = use normal alpha)
    out float displacedHeight // Output: height at displaced point (0=bottom, 1=surface)
) {
    if (viewDirTS.z <= 0.001 || heightScale <= 0.0) {
        displacedHeight = 1.0;
        return uv;
    }

    vec2 tileSize = uvMax - uvMin;
    // Total UV displacement from full depth, scaled by tile size and view angle
    vec2 maxOffset = -viewDirTS.xy / viewDirTS.z * heightScale * tileSize;

    float stepSize = 1.0 / float(steps);
    vec2 stepUV = maxOffset * stepSize;

    vec2 currentUV = uv;
    vec2 prevUV = uv;
    float layerHeight = 1.0;
    float prevLayerHeight = 1.0;
    float texHeight = pomSampleHeight(currentUV, heightTexID, normalTexID);
    float prevTexHeight = texHeight;

    // Linear search: step through layers until ray enters height field
    for (int i = 0; i < steps && layerHeight > texHeight; i++) {
        prevUV = currentUV;
        prevLayerHeight = layerHeight;
        prevTexHeight = texHeight;
        currentUV += stepUV;
        layerHeight -= stepSize;
        texHeight = pomSampleHeight(currentUV, heightTexID, normalTexID);
    }

    // Secant interpolation: single-step sub-texel refinement between the two bracketing samples
    // prevLayerHeight > prevTexHeight (ray above surface) and layerHeight <= texHeight (ray below)
    float d1 = prevLayerHeight - prevTexHeight;  // positive (ray above)
    float d2 = layerHeight - texHeight;          // negative (ray below)
    float denom = d1 - d2;
    if (abs(denom) > 1e-6) {
        float t = d1 / denom;
        currentUV = mix(prevUV, currentUV, t);
        layerHeight = mix(prevLayerHeight, layerHeight, t);
    }

    displacedHeight = layerHeight;

    // Clamp to tile boundaries to prevent atlas bleeding
    return clamp(currentUV, uvMin, uvMax);
}

// Legacy overload without displacedHeight output
vec2 parallaxOcclusionMapping(
    vec2 uv, vec3 viewDirTS, int normalTexID,
    float heightScale, int steps, int refinement,
    vec2 uvMin, vec2 uvMax, int heightTexID
) {
    float h;
    return parallaxOcclusionMapping(uv, viewDirTS, normalTexID, heightScale, steps, refinement, uvMin, uvMax, heightTexID, h);
}

// Legacy overload without heightTexID
vec2 parallaxOcclusionMapping(
    vec2 uv, vec3 viewDirTS, int normalTexID,
    float heightScale, int steps, int refinement,
    vec2 uvMin, vec2 uvMax
) {
    float h;
    return parallaxOcclusionMapping(uv, viewDirTS, normalTexID, heightScale, steps, refinement, uvMin, uvMax, -1, h);
}

/// Self-shadowing: trace height field from displaced point toward light.
/// Returns 1.0 = fully lit, 0.0 = fully in POM shadow.
/// Uses soft shadow approximation (penumbra from partial occlusion).
float pomSelfShadow(
    vec2 uv,             // Displaced UV from POM
    vec3 lightDirTS,     // Light direction in tangent space
    int heightTexID,     // Dedicated height texture (-1 = use normal alpha)
    int normalTexID,     // Normal texture (height in alpha)
    float heightScale,   // Same depth scale used for POM
    float currentHeight, // Height at displaced point (from POM out param)
    int steps,           // Shadow trace steps (8-32)
    vec2 uvMin,          // Tile boundary min
    vec2 uvMax           // Tile boundary max
) {
    // Light below surface → fully shadowed
    if (lightDirTS.z <= 0.001) return 0.0;

    vec2 tileSize = uvMax - uvMin;
    vec2 lightStep = lightDirTS.xy / lightDirTS.z * heightScale * tileSize / float(steps);
    float heightStep = (1.0 - currentHeight) / float(steps);

    vec2 sampleUV = uv;
    float rayHeight = currentHeight;
    float shadow = 1.0;

    for (int i = 0; i < steps; i++) {
        sampleUV += lightStep;
        rayHeight += heightStep;

        // Escaped to surface level
        if (rayHeight >= 1.0) break;

        // Left tile boundary
        if (any(lessThan(sampleUV, uvMin)) || any(greaterThan(sampleUV, uvMax))) break;

        float texHeight = pomSampleHeight(sampleUV, heightTexID, normalTexID);
        float occlusion = texHeight - rayHeight;
        if (occlusion > 0.0) {
            // Soft shadow: deeper occlusion → harder shadow
            shadow = min(shadow, 1.0 - clamp(occlusion * 8.0 / heightScale, 0.0, 1.0));
            if (shadow <= 0.01) return 0.0;
        }
    }

    return shadow;
}
