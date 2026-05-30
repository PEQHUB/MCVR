#include "common/constants.glsl"
#ifndef ADV_PRIMARY_TRACE_GLSL
#define ADV_PRIMARY_TRACE_GLSL

struct FirstHitPayload {
    bool hitSky;
    float surfaceDepth;
    float viewDepth;
    float firstHitLinearDepth;
    float tracedDistance;
    int startBounce;
    vec3 incomingDirection;
    vec3 outgoingDirection;
    vec3 normal;
    vec3 worldPos;
};

const uint ADV_WORLD_MISS_INDEX = 1u;
const uint ADV_HAND_MISS_INDEX = 2u;
const uint ADV_WORLD_NO_VOLUMETRIC_MISS_INDEX = 3u;

void initMainRay(vec3 origin, vec3 direction, uint seed, float coneSpread, out MainRay ray) {
    ray.origin = origin;
    ray.hitT = 0.0;
    ray.direction = direction;
    ray.seed = xxhash32(uvec3(gl_LaunchIDEXT.x, gl_LaunchIDEXT.y, seed));
    ray.radiance = vec3(0.0);
    ray.coneWidth = 0.0;
    ray.throughput = vec3(1.0);
    ray.coneSpread = coneSpread;
    ray.stateBits = 0u;
    raySetBounce(ray, 0u);
    raySetInsideBoat(ray, false);
    raySetStop(ray, false);
    raySetContinue(ray, false);
    raySetNoisy(ray, false);
    raySetSkipFog(ray, false);
    raySetSurfaceCacheWritten(ray, false);
    raySetSurfaceCacheTargetSecondary(ray, false);
    ray.prevScenePos = vec3(0.0);
    ray.hasPrevScenePos = 0u;
    ray.normal = vec3(0.0);
    raySetLobeType(ray, 0u);
    ray.directLightRadiance = vec3(0.0);
    ray.pad0 = 0u;
}

void resetMainRay(inout MainRay ray, int bounce) {
    raySetBounce(ray, uint(bounce));
    ray.hitT = 0.0;
    raySetStop(ray, false);
    raySetContinue(ray, false);
    raySetNoisy(ray, false);
    raySetSkipFog(ray, false);
    raySetSurfaceCacheWritten(ray, false);
    ray.prevScenePos = vec3(0.0);
    ray.hasPrevScenePos = 0u;
    ray.normal = vec3(0.0);
    raySetLobeType(ray, 0u);
    ray.directLightRadiance = vec3(0.0);
    ray.pad0 = 0u;
}

void preparePrimary(int step, int isHand, out float len, out uint mask, out uint missId) {
    if (isHand == 0) {
        missId = step == 0 ? ADV_WORLD_MISS_INDEX : ADV_WORLD_NO_VOLUMETRIC_MISS_INDEX;
        if (step == 0) {
            len = 1000.0;
            if (worldUBO.isFirstPerson > 0) {
                mask = WORLD_MASK | FISHING_BOBBER_MASK | WEATHER_MASK | PARTICLE_MASK | CLOUD_MASK | BOAT_WATER_MASK;
            } else {
                mask = WORLD_MASK | PLAYER_MASK | FISHING_BOBBER_MASK | WEATHER_MASK | PARTICLE_MASK | CLOUD_MASK |
                       BOAT_WATER_MASK;
            }
        } else if (step <= 1) {
            len = 1000.0;
            mask = WORLD_MASK | PLAYER_MASK | FISHING_BOBBER_MASK | WEATHER_MASK | PARTICLE_MASK | CLOUD_MASK |
                   BOAT_WATER_MASK;
        } else {
            len = 384.0;
            mask = WORLD_MASK | PLAYER_MASK | FISHING_BOBBER_MASK | BOAT_WATER_MASK;
        }
    } else {
        if (step == 0) {
            len = 10.0;
            mask = HAND_MASK;
            missId = ADV_HAND_MISS_INDEX;
        } else if (step <= 1) {
            len = 384.0;
            mask = WORLD_MASK | FISHING_BOBBER_MASK | WEATHER_MASK | PARTICLE_MASK | CLOUD_MASK | BOAT_WATER_MASK;
            missId = step == 1 ? ADV_WORLD_MISS_INDEX : ADV_WORLD_NO_VOLUMETRIC_MISS_INDEX;
        } else {
            len = 384.0;
            mask = WORLD_MASK | FISHING_BOBBER_MASK | BOAT_WATER_MASK;
            missId = ADV_WORLD_NO_VOLUMETRIC_MISS_INDEX;
        }
    }
}

void tracePrimary(vec3 eyePos,
                  vec3 orgDirection,
                  uint rayFlags,
                  int isHand,
                  inout MainRay rayState,
                  out FirstHitPayload firstHit) {
    firstHit.hitSky = false;
    firstHit.surfaceDepth = INF_DISTANCE;
    firstHit.viewDepth = INF_DISTANCE;
    firstHit.firstHitLinearDepth = INF_DISTANCE;
    firstHit.tracedDistance = 0.0;
    firstHit.startBounce = 0;
    firstHit.incomingDirection = vec3(0.0);
    firstHit.outgoingDirection = vec3(0.0);
    firstHit.normal = vec3(0.0);
    firstHit.worldPos = vec3(0.0);

    bool surfaceDepthSampled = false;
    for (int step = 0; step < ADV_PRIMARY_TRACE_STEP_LIMIT; ++step) {
        resetMainRay(rayState, step);

        float len;
        uint mask;
        uint missId;
        preparePrimary(step, isHand, len, mask, missId);

        vec3 tracedOrigin = rayState.origin;
        vec3 tracedDirection = rayState.direction;
        mainRay = rayState;
        traceRayEXT(topLevelAS, rayFlags, mask, 1, 1, missId, mainRay.origin, 0.0001, mainRay.direction, len, 0);
        rayState = mainRay;

        if (!surfaceDepthSampled) {
            firstHit.surfaceDepth = rayState.hitT;
            surfaceDepthSampled = true;
            if (rayState.hitT != INF_DISTANCE) {
                vec3 hitPos = eyePos + orgDirection * rayState.hitT;
                hitPos -= worldUBO.cameraViewMatInv[3].xyz;
                firstHit.firstHitLinearDepth = -(mat3(worldUBO.cameraEffectedViewMat) * hitPos).z;
            }
        }

        if (rayState.hitT == INF_DISTANCE) {
            firstHit.hitSky = true;
            firstHit.startBounce = step + 1;
            return;
        }

        firstHit.tracedDistance += rayState.hitT;
        firstHit.startBounce = step + 1;

        if (rayShouldContinue(rayState)) { continue; }

        firstHit.incomingDirection = tracedDirection;
        firstHit.outgoingDirection = rayState.direction;
        firstHit.normal = rayState.normal;
        firstHit.worldPos = tracedOrigin + tracedDirection * rayState.hitT;
        break;
    }

    vec3 virtualOrigin = eyePos + orgDirection * firstHit.tracedDistance;
    firstHit.viewDepth = -(worldUBO.cameraEffectedViewMat * vec4(virtualOrigin, 1.0)).z;
}

#endif
