#ifndef VPT_RAYTRACE_TRANSFORMS_GLSL
#define VPT_RAYTRACE_TRANSFORMS_GLSL

vec3 vptObjectToWorldVector(vec3 v) {
    return v.x * gl_ObjectToWorldEXT[0] +
           v.y * gl_ObjectToWorldEXT[1] +
           v.z * gl_ObjectToWorldEXT[2];
}

vec3 vptObjectToWorldPoint(vec3 p) {
    return vptObjectToWorldVector(p) + gl_ObjectToWorldEXT[3];
}

vec3 vptWorldToObjectVector(vec3 v) {
    return v.x * gl_WorldToObjectEXT[0] +
           v.y * gl_WorldToObjectEXT[1] +
           v.z * gl_WorldToObjectEXT[2];
}

#endif
