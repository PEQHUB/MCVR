#version 460
#extension GL_EXT_ray_tracing : require

// V2 PR39-prep MVP miss shader.
// Reads SkyUBO for sun direction and base/horizon colors.
// Produces a simple gradient + sun disc.

layout(set = 2, binding = 2, std140) uniform SkyUBO {
    vec3 baseColor;
    uint skyType;
    vec4 horizonColor;
    vec3 sunDirection;
    uint isSunRisingOrSetting;
    vec3 moonDirection;
    float moonDirPad;
} sky;

struct HitPayload {
    vec4 colorAndHit;     // rgb + hit flag
    vec4 posAndT;         // world pos + distance
    vec4 normalAndRough;  // world normal + roughness
};
layout(location = 0) rayPayloadInEXT HitPayload hitPayload;

void main() {
    vec3 dir = normalize(gl_WorldRayDirectionEXT);

    // Vertical gradient between horizon and overhead colors
    float t = clamp(dir.y, 0.0, 1.0);
    vec3 col = mix(sky.horizonColor.rgb, sky.baseColor, t);

    // Cheap sun disc
    float sunDot = dot(dir, normalize(sky.sunDirection));
    if (sunDot > 0.998) {
        col += vec3(1.5, 1.4, 1.1);
    } else if (sunDot > 0.99) {
        // Soft halo
        float halo = (sunDot - 0.99) / 0.008;
        col += vec3(0.6, 0.55, 0.4) * halo * halo;
    }

    // hit=0 signals miss → denoiser/motion code treats this pixel as sky.
    hitPayload.colorAndHit    = vec4(col, 0.0);
    hitPayload.posAndT        = vec4(0.0);
    hitPayload.normalAndRough = vec4(-dir, 1.0); // use inverse ray dir as "normal"
}
