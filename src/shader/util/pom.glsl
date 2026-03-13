// Parallax Occlusion Mapping with binary refinement.
// Operates in atlas UV space; tile boundaries prevent cross-tile bleeding.

// Sample height from either a dedicated R-channel height texture or the normal alpha channel.
float pomSampleHeight(vec2 uv, int heightTexID, int normalTexID) {
    if (heightTexID >= 0) {
        return textureLod(textures[nonuniformEXT(heightTexID)], uv, 0).r;
    }
    return textureLod(textures[nonuniformEXT(normalTexID)], uv, 0).w;
}

vec2 parallaxOcclusionMapping(
    vec2 uv,             // Atlas UV at hit point
    vec3 viewDirTS,      // View direction in tangent space (z > 0 = facing surface)
    int normalTexID,     // Normal texture index (height in alpha channel)
    float heightScale,   // Depth scale in tile-local [0,1] units
    int steps,           // Linear search steps (8-512)
    int refinement,      // Binary refinement iterations (0-8)
    vec2 uvMin,          // Tile boundary min (atlas UV)
    vec2 uvMax,          // Tile boundary max (atlas UV)
    int heightTexID      // Blender PBR dedicated height texture (-1 = use normal alpha)
) {
    if (viewDirTS.z <= 0.001 || heightScale <= 0.0) return uv;

    vec2 tileSize = uvMax - uvMin;
    // Total UV displacement from full depth, scaled by tile size and view angle
    vec2 maxOffset = -viewDirTS.xy / viewDirTS.z * heightScale * tileSize;

    float stepSize = 1.0 / float(steps);
    vec2 stepUV = maxOffset * stepSize;

    vec2 currentUV = uv;
    float layerHeight = 1.0;
    float texHeight = pomSampleHeight(currentUV, heightTexID, normalTexID);

    // Linear search: step through layers until ray enters height field
    for (int i = 0; i < steps && layerHeight > texHeight; i++) {
        currentUV += stepUV;
        layerHeight -= stepSize;
        texHeight = pomSampleHeight(currentUV, heightTexID, normalTexID);
    }

    // Binary refinement: narrow down intersection point
    vec2 halfStep = stepUV * 0.5;
    float halfLayer = stepSize * 0.5;
    for (int i = 0; i < refinement; i++) {
        vec2 midUV = currentUV - halfStep;
        float midLayer = layerHeight + halfLayer;
        float midTex = pomSampleHeight(midUV, heightTexID, normalTexID);
        if (midTex > midLayer) {
            // Intersection is in upper half
            currentUV = midUV;
            layerHeight = midLayer;
        }
        halfStep *= 0.5;
        halfLayer *= 0.5;
    }

    // Clamp to tile boundaries to prevent atlas bleeding
    return clamp(currentUV, uvMin, uvMax);
}

// Legacy overload for callers that don't have a dedicated height texture
vec2 parallaxOcclusionMapping(
    vec2 uv, vec3 viewDirTS, int normalTexID,
    float heightScale, int steps, int refinement,
    vec2 uvMin, vec2 uvMax
) {
    return parallaxOcclusionMapping(uv, viewDirTS, normalTexID, heightScale, steps, refinement, uvMin, uvMax, -1);
}
