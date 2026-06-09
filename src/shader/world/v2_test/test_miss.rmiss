#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 hitColor;

void main() {
    // Gradient sky
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    float t = 0.5 * (dir.y + 1.0);
    hitColor = mix(vec3(0.8, 0.85, 0.9), vec3(0.3, 0.5, 0.8), t);
}
