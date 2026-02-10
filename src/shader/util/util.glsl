#ifndef UTIL_GLSL
#define UTIL_GLSL

vec2 computeCameraMotionVector(mat4 prevMVP, vec2 pixelCenter, vec4 motionOrigin) {
    highp vec4 oldPos = prevMVP * motionOrigin;
    oldPos.xy /= oldPos.w;
    oldPos.xy = (oldPos.xy * 0.5 + 0.5) * vec2(gl_LaunchSizeEXT.xy);
    vec2 motion = oldPos.xy - pixelCenter;
    return motion;
}

mat3 buildMirrorMatrix(vec3 normal) {
    return mat3(-2.0 * (vec3(normal.x) * normal) + vec3(1.0, 0.0, 0.0), //
                -2.0 * (vec3(normal.y) * normal) + vec3(0.0, 1.0, 0.0), //
                -2.0 * (vec3(normal.z) * normal) + vec3(0.0, 0.0, 1.0));
}

mat4 buildMirrorMatrix4(vec3 normal, vec3 posOnMirror) {
    mat3 R = buildMirrorMatrix(normal);
    vec3 t = posOnMirror - R * posOnMirror;

    return mat4(vec4(R[0], 0.0), //
                vec4(R[1], 0.0), //
                vec4(R[2], 0.0), //
                vec4(t, 1.0));
}

vec3 reinhardMax(vec3 color) {
    float luminance = max(1e-7, max(max(color.x, color.y), color.z));
    float reinhard = luminance / (luminance + 1);
    return color * (reinhard / luminance);
}

#endif