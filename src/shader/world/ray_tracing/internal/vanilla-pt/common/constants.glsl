#ifndef VPT_CONSTANTS_GLSL
#define VPT_CONSTANTS_GLSL

#ifndef VPT_INDIRECT_VOLUMETRIC_CLOUD_REFLECTION_MAX_ROUGHNESS
#    define VPT_INDIRECT_VOLUMETRIC_CLOUD_REFLECTION_MAX_ROUGHNESS 0.12
#endif

const int VPT_PRIMARY_TRACE_STEP_LIMIT = 2;
const uint VPT_RAY_FLAGS = gl_RayFlagsNoneEXT;
const uint VPT_TRANSPARENT_SPLIT_MODE_DETERMINISTIC = 0u;
const float VPT_FFT_WATER_ORIGIN_BIAS = 0.0006;
const float VPT_PARALLAX_MIN_VIEW_DOT = 0.001;
const float VPT_PARALLAX_CLOSE_DISTANCE = 48.0;
const int VPT_PARALLAX_PRIMARY_MAX_STEPS = 16;
const int VPT_PARALLAX_SECONDARY_MAX_STEPS = 4;
const vec3[] VPT_COLORS = vec3[](
    vec3(0.022087, 0.098399, 0.110818),
    vec3(0.011892, 0.095924, 0.089485),
    vec3(0.027636, 0.101689, 0.100326),
    vec3(0.046564, 0.109883, 0.114838),
    vec3(0.064901, 0.117696, 0.097189),
    vec3(0.063761, 0.086895, 0.123646),
    vec3(0.084817, 0.111994, 0.166380),
    vec3(0.097489, 0.154120, 0.091064),
    vec3(0.106152, 0.131144, 0.195191),
    vec3(0.097721, 0.110188, 0.187229),
    vec3(0.133516, 0.138278, 0.148582),
    vec3(0.070006, 0.243332, 0.235792),
    vec3(0.196766, 0.142899, 0.214696),
    vec3(0.047281, 0.315338, 0.321970),
    vec3(0.204675, 0.390010, 0.302066),
    vec3(0.080955, 0.314821, 0.661491)
);
const mat4 VPT_SCALE_TRANSLATE = mat4(
    0.5, 0.0, 0.0, 0.25,
    0.0, 0.5, 0.0, 0.25,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0
);

bool vptShouldCaptureIndirectVolumetricCloud(uint lobeType, float roughness) {
    if (lobeType == 2u) { return true; }
    if (lobeType != 1u) { return false; }
    return roughness <= clamp(VPT_INDIRECT_VOLUMETRIC_CLOUD_REFLECTION_MAX_ROUGHNESS, 0.0, 1.0);
}

#endif
