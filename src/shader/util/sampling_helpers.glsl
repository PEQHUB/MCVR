#ifndef SAMPLING_HELPERS_GLSL
#define SAMPLING_HELPERS_GLSL

#include "disney.glsl"

vec3 sampleCosineHemisphere(inout uint seed, vec3 N) {
    vec3 T, B;
    Onb(N, T, B);
    vec3 localL = CosineSampleHemisphere(rand(seed), rand(seed));
    return ToWorld(T, B, N, localL);
}

// Overload for Xi (blue noise)
vec3 sampleCosineHemisphereXi(vec2 xi, vec3 N) {
    vec3 T, B;
    Onb(N, T, B);
    vec3 localL = CosineSampleHemisphere(xi.x, xi.y);
    return ToWorld(T, B, N, localL);
}

vec3 sampleGGX(inout uint seed, vec3 N, float roughness) {
    vec3 T, B;
    Onb(N, T, B);
    
    float r1 = rand(seed);
    float r2 = rand(seed);
    float a = roughness * roughness;
    float phi = 2.0 * PI * r1;
    float cosTheta2 = (1.0 - r2) / (r2 * (a * a - 1.0) + 1.0);
    float cosTheta = sqrt(clamp(cosTheta2, 0.0, 1.0));
    float sinTheta = sqrt(clamp(1.0 - cosTheta2, 0.0, 1.0));
    
    vec3 localH = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    
    vec3 H = ToWorld(T, B, N, localH);
    return H;
}

vec3 sampleGGXXi(vec2 xi, vec3 N, float roughness) {
    vec3 T, B;
    Onb(N, T, B);
    
    float r1 = xi.x;
    float r2 = xi.y;
    float a = roughness * roughness;
    float phi = 2.0 * PI * r1;
    float cosTheta2 = (1.0 - r2) / (r2 * (a * a - 1.0) + 1.0);
    float cosTheta = sqrt(clamp(cosTheta2, 0.0, 1.0));
    float sinTheta = sqrt(clamp(1.0 - cosTheta2, 0.0, 1.0));
    
    vec3 localH = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    
    vec3 H = ToWorld(T, B, N, localH);
    return H;
}

vec3 sampleVMF(inout uint seed, vec3 mu, float kappa) {
    return SampleVMF(seed, mu, kappa);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * SchlickWeight(cosTheta);
}

#endif
