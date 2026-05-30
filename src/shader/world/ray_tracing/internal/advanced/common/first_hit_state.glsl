#include "common/constants.glsl"
#ifndef ADV_FIRST_HIT_STATE_GLSL
#define ADV_FIRST_HIT_STATE_GLSL

float encodeFirstHitFlags(bool hitSky,
                          bool noisy,
                          bool isHand,
                          bool insideBoat,
                          bool skipFog,
                          bool indirectVolumetricCloud) {
    uint flags = 0u;
    if (hitSky) { flags |= ADV_FIRST_HIT_FLAG_HIT_SKY; }
    if (noisy) { flags |= ADV_FIRST_HIT_FLAG_NOISY; }
    if (isHand) { flags |= ADV_FIRST_HIT_FLAG_HAND; }
    if (insideBoat) { flags |= ADV_FIRST_HIT_FLAG_INSIDE_BOAT; }
    if (skipFog) { flags |= ADV_FIRST_HIT_FLAG_SKIP_FOG; }
    if (indirectVolumetricCloud) { flags |= ADV_FIRST_HIT_FLAG_INDIRECT_VOLUMETRIC_CLOUD; }
    return float(flags);
}

uint decodeFirstHitFlags(float encoded) {
    return uint(max(encoded + 0.5, 0.0));
}

bool hasFirstHitFlag(uint flags, uint flagBit) {
    return (flags & flagBit) != 0u;
}

vec2 encodeFirstHitUint(uint value) {
    return vec2(float(value % ADV_FIRST_HIT_UINT_SPLIT_BASE), float(value / ADV_FIRST_HIT_UINT_SPLIT_BASE));
}

uint decodeFirstHitUint(vec2 encoded) {
    uint low = uint(max(encoded.x + 0.5, 0.0));
    uint high = uint(max(encoded.y + 0.5, 0.0));
    return low + high * ADV_FIRST_HIT_UINT_SPLIT_BASE;
}

#endif
