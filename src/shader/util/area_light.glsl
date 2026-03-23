#ifndef AREA_LIGHT_GLSL
#define AREA_LIGHT_GLSL

struct CubeSample {
    vec3 worldPos;   // sampled point in world space
    int faceAxis;    // 0=X, 1=Y, 2=Z (for geometric term)
};

// Sample a point on the closest face of a cube area light.
// Point-like sources get a small virtual disk jitter for soft shadows.
// shadowSoft: 0.0 = no jitter (hard shadows), 1.0 = default, 2.0 = extra soft
CubeSample sampleCubeLight(AreaLight light, vec3 shadingPos, float shadowSoft, inout uint seed) {
    CubeSample s;

    if (light.halfExtent < 0.01) {
        // Point-like source — jitter in plane perpendicular to light direction
        // for soft penumbras (virtual disk radius ~0.08 blocks)
        vec3 toCenter = light.position - shadingPos;
        vec3 dir = normalize(toCenter);
        // Build tangent frame
        vec3 up = abs(dir.y) < 0.99 ? vec3(0, 1, 0) : vec3(1, 0, 0);
        vec3 tangent = normalize(cross(dir, up));
        vec3 bitangent = cross(dir, tangent);
        float virtualRadius = 0.08 * shadowSoft;
        float ju = (rand(seed) - 0.5) * virtualRadius;
        float jv = (rand(seed) - 0.5) * virtualRadius;
        s.worldPos = light.position + tangent * ju + bitangent * jv;
        s.faceAxis = 0;
        return s;
    }

    vec3 toCenter = light.position - shadingPos;
    vec3 absDir = abs(toCenter);
    float u = rand(seed) - 0.5;  // [-0.5, 0.5]
    float v = rand(seed) - 0.5;

    if (absDir.x >= absDir.y && absDir.x >= absDir.z) {
        vec3 fn = vec3(sign(toCenter.x), 0.0, 0.0);
        s.worldPos = light.position - fn * light.halfExtent
                   + vec3(0.0, u, v) * light.halfExtent * 2.0;
        s.faceAxis = 0;
    } else if (absDir.y >= absDir.z) {
        vec3 fn = vec3(0.0, sign(toCenter.y), 0.0);
        s.worldPos = light.position - fn * light.halfExtent
                   + vec3(u, 0.0, v) * light.halfExtent * 2.0;
        s.faceAxis = 1;
    } else {
        vec3 fn = vec3(0.0, 0.0, sign(toCenter.z));
        s.worldPos = light.position - fn * light.halfExtent
                   + vec3(u, v, 0.0) * light.halfExtent * 2.0;
        s.faceAxis = 2;
    }
    return s;
}

// Face area for geometric term
float cubeFaceArea(float halfExtent) {
    float side = halfExtent * 2.0;
    return side * side;
}

#endif
