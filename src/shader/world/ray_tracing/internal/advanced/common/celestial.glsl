#ifndef ADV_CELESTIAL_GLSL
#define ADV_CELESTIAL_GLSL

vec3 celestialNormalize(vec3 direction, vec3 fallback) {
    float len2 = dot(direction, direction);
    if (len2 <= 1e-8 || any(isnan(direction)) || any(isinf(direction))) { return fallback; }
    return direction * inversesqrt(len2);
}

vec3 celestialApplySouthOffset(vec3 direction) {
    const float southOffsetSin = 0.17364817766693033;
    const float southOffsetCos = 0.984807753012208;
    vec3 rotatedDirection = vec3(direction.x,
                                 direction.y * southOffsetCos - direction.z * southOffsetSin,
                                 direction.y * southOffsetSin + direction.z * southOffsetCos);
    return celestialNormalize(rotatedDirection, vec3(0.0, 1.0, 0.0));
}

vec3 celestialSunDirection() {
    return celestialApplySouthOffset(celestialNormalize(skyUBO.sunDirection, vec3(0.0, 1.0, 0.0)));
}

vec3 celestialMoonDirection() {
    return -celestialSunDirection();
}

#endif
