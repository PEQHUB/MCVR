#include "common/constants.glsl"
#ifndef ADV_PARALLAX_CONDITION_GLSL
#define ADV_PARALLAX_CONDITION_GLSL

bool shouldTraceRestirParallax(float lod, vec3 hitWorldPos) {
    vec3 cameraOrigin = vec3(worldUBO.cameraEffectedViewMatInv * vec4(0.0, 0.0, 0.0, 1.0));
    return lod == 0.0 || distance(hitWorldPos, cameraOrigin) <= ADV_PARALLAX_CLOSE_DISTANCE;
}

#endif
