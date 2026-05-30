#version 460
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 outColor;

bool intersectSphere(vec3 rayOrigin, vec3 rayDir, float radius, out float tNear, out float tFar) {
    float b = dot(rayOrigin, rayDir);
    float c = dot(rayOrigin, rayOrigin) - radius * radius;
    float height = b * b - c;
    if (height < 0.0) return false;
    height = sqrt(height);
    tNear = -b - height;
    tFar = -b + height;
    return true;
}

float densityExp(float height, float scaleHeight) {
    return exp(-max(height, 0.0) / scaleHeight);
}

void main() {
    float mu = texCoord.x * 2.0 - 1.0;
    float r = mix(ADV_ATMOSPHERE_RG, ADV_ATMOSPHERE_RT, texCoord.y);

    vec3 rayOrigin = vec3(0.0, r, 0.0);
    float sinTheta = sqrt(max(1.0 - mu * mu, 0.0));
    vec3 rayDir = normalize(vec3(sinTheta, mu, 0.0));

    float t0, t1;
    if (!intersectSphere(rayOrigin, rayDir, ADV_ATMOSPHERE_RT, t0, t1)) {
        outColor = vec4(1.0);
        return;
    }
    t0 = max(t0, 0.0);

    const int stepCount = 128;
    float dt = (t1 - t0) / float(stepCount);
    vec3 opticalDepth = vec3(0.0);

    for (int i = 0; i < stepCount; i++) {
        float t = t0 + (float(i) + 0.5) * dt;
        vec3 x = rayOrigin + rayDir * t;
        float rr = length(x);
        float height = rr - ADV_ATMOSPHERE_RG;

        float dR = densityExp(height, ADV_ATMOSPHERE_HR);
        float dM = densityExp(height, ADV_ATMOSPHERE_HM);

        vec3 sigmaT = ADV_ATMOSPHERE_BETA_R * dR + ADV_ATMOSPHERE_BETA_M * dM;
        opticalDepth += sigmaT * dt;
    }

    outColor = vec4(exp(-opticalDepth), 1.0);
}
