#include "common/constants.glsl"
#ifndef VPT_END_PORTAL_GLSL
#define VPT_END_PORTAL_GLSL

vec4 projectPosition(vec4 position) {
    vec4 projection = position * 0.5;
    projection.xy = vec2(projection.x + projection.w, projection.y + projection.w);
    projection.zw = position.zw;
    return projection;
}


vec3 computeEndPortalColor(vec4 texProj0, int iterations, uint endSkyTextureID, uint endPortalTextureID, float gameTime) {
    vec3 color = vec3(0.0);
    if (endSkyTextureID != 0xFFFFFFFFu)
        color += textureProj(textures[nonuniformEXT(endSkyTextureID)], texProj0).rgb * VPT_COLORS[0];
    for (int i = 0; i < iterations; i++) {
        if (endPortalTextureID != 0xFFFFFFFFu) {
            float layer = float(i + 1);
            mat4 translate = mat4(
                1.0, 0.0, 0.0, 17.0 / layer,
                0.0, 1.0, 0.0, (2.0 + layer / 1.5) * (gameTime * 1.5),
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            );
            float rotationAngle = radians((layer * layer * 4321.0 + layer * 9.0) * 2.0);
            mat2 rotate = mat2(cos(rotationAngle), -sin(rotationAngle), sin(rotationAngle), cos(rotationAngle));
            mat2 scale = mat2((4.5 - layer / 4.0) * 2.0);
            mat4 portalLayer = mat4(scale * rotate) * translate * VPT_SCALE_TRANSLATE;
            color += textureProj(textures[nonuniformEXT(endPortalTextureID)], texProj0 * portalLayer).rgb *
                     VPT_COLORS[i];
        }
    }
    return color;
}
#endif
