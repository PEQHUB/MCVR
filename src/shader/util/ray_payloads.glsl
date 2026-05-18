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

    // in/out, maintained during all ray tracing
    T_VEC3 origin;
    T_VEC3 direction;
    T_VEC3 radiance;
    T_VEC3 throughput;
    T_FLOAT hitT;
    T_UINT seed;
    T_FLOAT coneWidth;
    T_FLOAT coneSpread;

    // Packed bitfield: boolean/small-int flags (replaces 7 separate uint fields)
    //   bit  0      : stop        (1 = stop bouncing)
    //   bit  1      : cont        (1 = PSR continuation, skip this hit)
    //   bit  2      : noisy       (1 = stochastic BSDF sample, needs denoising)
    //   bit  3      : isHand      (1 = hand geometry pass)
    //   bit  4      : insideBoat  (1 = inside boat water mask)
    //   bits 5-6    : lobeType    (0=diffuse, 1=specular, 2=transmission)
    //   bits 8-15   : emBlockTypeOut (0-255, emissive block ordinal for bloom)
    T_UINT flags;

    // out
    T_UINT instanceIndex;
    T_UINT geometryIndex;
    T_UINT primitiveIndex;
    T_VEC3 baryCoords;
    T_VEC3 worldPos;
    T_VEC3 normal;
    T_VEC4 albedoValue;
    T_VEC3 directLightRadiance;
    T_FLOAT directLightHitT;
    T_FLOAT roughness;
    T_FLOAT albedoEmission;
    T_UINT pixelPacked;
    // Post-override material for DLSS-RR guide buffers (CHS writes after WorldUBO override)
    T_FLOAT metallic;
    T_VEC3 f0;
    T_FLOAT emission;
    T_FLOAT subSurface;
};

// ── PrimaryRay.flags bitfield accessors (GLSL only) ──
// These constants and macros provide clean read/write access to the packed flags field.
#ifndef __cplusplus
#define PR_STOP_BIT        0x01u   // bit 0
#define PR_CONT_BIT        0x02u   // bit 1
#define PR_NOISY_BIT       0x04u   // bit 2
#define PR_ISHAND_BIT      0x08u   // bit 3
#define PR_INSIDEBOAT_BIT  0x10u   // bit 4
#define PR_LOBETYPE_SHIFT  5u
#define PR_LOBETYPE_MASK   0x03u   // bits 5-6 (2 bits)
#define PR_EMBLOCK_SHIFT   8u
#define PR_EMBLOCK_MASK    0xFFu   // bits 8-15 (8 bits)

// Boolean flag read/write
#define prGetStop(ray)        (((ray).flags & PR_STOP_BIT) != 0u)
#define prGetCont(ray)        (((ray).flags & PR_CONT_BIT) != 0u)
#define prGetNoisy(ray)       (((ray).flags & PR_NOISY_BIT) != 0u)
#define prGetIsHand(ray)      (((ray).flags & PR_ISHAND_BIT) != 0u)
#define prGetInsideBoat(ray)  (((ray).flags & PR_INSIDEBOAT_BIT) != 0u)

#define prSetStop(ray, v)        { if (v) (ray).flags |= PR_STOP_BIT;       else (ray).flags &= ~PR_STOP_BIT; }
#define prSetCont(ray, v)        { if (v) (ray).flags |= PR_CONT_BIT;       else (ray).flags &= ~PR_CONT_BIT; }
#define prSetNoisy(ray, v)       { if (v) (ray).flags |= PR_NOISY_BIT;      else (ray).flags &= ~PR_NOISY_BIT; }
#define prSetIsHand(ray, v)      { if (v) (ray).flags |= PR_ISHAND_BIT;     else (ray).flags &= ~PR_ISHAND_BIT; }
#define prSetInsideBoat(ray, v)  { if (v) (ray).flags |= PR_INSIDEBOAT_BIT; else (ray).flags &= ~PR_INSIDEBOAT_BIT; }

// Multi-bit field read/write
#define prGetLobeType(ray)       (((ray).flags >> PR_LOBETYPE_SHIFT) & PR_LOBETYPE_MASK)
#define prSetLobeType(ray, v)    { (ray).flags = ((ray).flags & ~(PR_LOBETYPE_MASK << PR_LOBETYPE_SHIFT)) | (((v) & PR_LOBETYPE_MASK) << PR_LOBETYPE_SHIFT); }
#define prGetEmBlockType(ray)    (((ray).flags >> PR_EMBLOCK_SHIFT) & PR_EMBLOCK_MASK)
#define prSetEmBlockType(ray, v) { (ray).flags = ((ray).flags & ~(PR_EMBLOCK_MASK << PR_EMBLOCK_SHIFT)) | (((v) & PR_EMBLOCK_MASK) << PR_EMBLOCK_SHIFT); }
#endif

struct ShadowRay {
    T_VEC3 radiance;
    T_VEC3 throughput;
    T_UINT seed;
    T_FLOAT hitT;
    T_UINT insideBoat;
    T_UINT bounceIndex;
    // Beer's Law medium tracking (active when BEER_LAW_SHADOWS flag is set)
    T_VEC3 mediumAbsorption;  // sigma_a of current medium (vec3(0) = in air)
    T_FLOAT mediumEntryT;     // gl_HitTEXT when entering transmissive medium
};

#endif
