#ifndef RAY_PAYLOAD_GLSL
#define RAY_PAYLOAD_GLSL

#include "common/mapping.hpp"

struct ShadowRayPayload {
    T_VEC3 radiance;
    T_VEC3 throughput;
    T_UINT seed;
};

struct RayPayload {
    T_VEC3 origin;
    T_VEC3 direction;
    T_VEC3 radiance;
    T_VEC3 throughput;
    T_UINT bounces;
    T_UINT seed;
    T_FLOAT t;
    T_BOOL storeDLSSAux;
};

struct MainRay {
    T_VEC3 origin;
    T_VEC3 direction;
    T_VEC3 radiance;
    T_VEC3 throughput;
    T_FLOAT t;
    T_UINT seed;
    T_UINT bounces;
    T_VEC3 intermediateRadiance;
    T_UINT isHand;
    T_UINT bouncedOnSolid;
    T_UINT rayIndex;
};

struct PrimaryRay {
    // in
    T_UINT index;

    // in out, maintained during all ray tracing
    T_VEC3 origin;
    T_VEC3 direction;
    T_VEC3 radiance;
    T_VEC3 intermediateRadiance;
    T_VEC3 throughput;
    T_FLOAT hitT;
    T_UINT seed;
    T_FLOAT coneWidth;
    T_FLOAT coneSpread;
    T_UINT insideBoat;
    T_UINT isHand;

    // out, updated by the latest ray tracing
    T_UINT instanceIndex;
    T_UINT geometryIndex;
    T_UINT primitiveIndex;
    T_VEC3 baryCoords;
    T_VEC3 worldPos;
    T_VEC3 normal;
    T_VEC4 albedoValue;
    T_VEC4 specularValue;
    T_VEC4 normalValue;
    T_IVEC4 flagValue;
    T_UINT stop;
    T_UINT cont;
    T_UINT noisy;
    T_UINT lobeType;
    T_VEC3 directLightRadiance;
    T_FLOAT directLightHitT;
};

struct ShadowRay {
    T_VEC3 radiance;
    T_VEC3 throughput;
    T_UINT seed;
    T_FLOAT hitT;
    T_UINT insideBoat;
};

#endif
