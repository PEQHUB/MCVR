#ifndef VPT_PHYSICAL_LIGHTING_GLSL
#define VPT_PHYSICAL_LIGHTING_GLSL

vec3 vptPhysicalSunRadiance() {
    vec3 radiance = skyUBO.sunRadiance * max(skyUBO.envCelestial.z, 0.0);
    if (dot(radiance, radiance) <= 0.0 || any(isnan(radiance)) || any(isinf(radiance))) {
        radiance = vec3(100000.0);
    }
    return max(radiance, vec3(0.0));
}

vec3 vptPhysicalMoonRadiance() {
    vec3 radiance = skyUBO.moonRadiance * max(skyUBO.envCelestial.w, 0.0);
    if (dot(radiance, radiance) <= 0.0 || any(isnan(radiance)) || any(isinf(radiance))) {
        radiance = vec3(0.05, 0.06, 0.12);
    }
    return max(radiance, vec3(0.0));
}

vec3 vptPhysicalNightSkyFloor() {
    return max(vptPhysicalMoonRadiance() * 0.03, vec3(0.0005));
}

#endif
