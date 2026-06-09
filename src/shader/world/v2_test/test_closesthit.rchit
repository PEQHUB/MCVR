#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 hitColor;

hitAttributeEXT vec2 attribs;

void main() {
    // Simple normal visualization from barycentric coordinates
    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    // Depth-based shading: closer = brighter
    float depth = gl_HitTEXT;
    float brightness = exp(-depth * 0.02);

    hitColor = bary * brightness;
}
