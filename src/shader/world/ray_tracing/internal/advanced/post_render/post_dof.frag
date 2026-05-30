#version 460

#ifndef ADV_POST_ENABLE_DOF
#define ADV_POST_ENABLE_DOF 0
#endif

#ifndef ADV_POST_DOF_AUTO_FOCUS
#define ADV_POST_DOF_AUTO_FOCUS 1
#endif

#ifndef ADV_POST_DOF_AUTO_FOCUS_RADIUS
#define ADV_POST_DOF_AUTO_FOCUS_RADIUS 0.012
#endif

#ifndef ADV_POST_DOF_FOCUS_DISTANCE
#define ADV_POST_DOF_FOCUS_DISTANCE 12.0
#endif

#ifndef ADV_POST_DOF_FOCUS_RANGE
#define ADV_POST_DOF_FOCUS_RANGE 4.0
#endif

#ifndef ADV_POST_DOF_MAX_RADIUS
#define ADV_POST_DOF_MAX_RADIUS 6.0
#endif

#ifndef ADV_POST_DOF_SAMPLE_COUNT
#define ADV_POST_DOF_SAMPLE_COUNT 24
#endif

#ifndef ADV_POST_DOF_FOCUS_BINDING
#define ADV_POST_DOF_FOCUS_BINDING 27
#endif

#ifndef ADV_POST_DOF_SOURCE_BINDING
#define ADV_POST_DOF_SOURCE_BINDING 21
#endif

#ifndef ADV_POST_DOF_DIRECTION_X
#define ADV_POST_DOF_DIRECTION_X 1.0
#endif

#ifndef ADV_POST_DOF_DIRECTION_Y
#define ADV_POST_DOF_DIRECTION_Y 0.0
#endif

layout(set = 3, binding = 1) uniform sampler2D postFirstHitDepth;
layout(set = 4, binding = ADV_POST_DOF_SOURCE_BINDING) uniform sampler2D postDofInput;
layout(set = 4, binding = ADV_POST_DOF_FOCUS_BINDING) uniform sampler2D postDofFocus;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

const vec2 ADV_AUTO_FOCUS_OFFSETS[9] = vec2[](
    vec2(0.0, 0.0),
    vec2(0.0, -1.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(-1.0, 0.0),
    vec2(0.7071, -0.7071),
    vec2(0.7071, 0.7071),
    vec2(-0.7071, 0.7071),
    vec2(-0.7071, -0.7071)
);

vec2 clampUv(vec2 uv, vec2 texelSize) {
    return clamp(uv, texelSize * 0.5, vec2(1.0) - texelSize * 0.5);
}

float sampleRawDepth(vec2 uv) {
    return texture(postFirstHitDepth, uv).r;
}

void sortDepths(inout float depths[9], int count) {
    for (int i = 1; i < count; ++i) {
        float value = depths[i];
        int j = i - 1;
        while (j >= 0 && depths[j] > value) {
            depths[j + 1] = depths[j];
            --j;
        }
        depths[j + 1] = value;
    }
}

void addValidDepth(inout float depths[9], inout int count, float depthValue) {
    if (depthValue <= 0.0) {
        return;
    }

    depths[count] = depthValue;
    ++count;
}

float sampleStableDepth(vec2 uv, vec2 texelSize) {
    uv = clampUv(uv, texelSize);

    float depths[9];
    int count = 0;
    addValidDepth(depths, count, sampleRawDepth(uv));
    addValidDepth(depths, count, sampleRawDepth(clampUv(uv + vec2(texelSize.x, 0.0), texelSize)));
    addValidDepth(depths, count, sampleRawDepth(clampUv(uv - vec2(texelSize.x, 0.0), texelSize)));
    addValidDepth(depths, count, sampleRawDepth(clampUv(uv + vec2(0.0, texelSize.y), texelSize)));
    addValidDepth(depths, count, sampleRawDepth(clampUv(uv - vec2(0.0, texelSize.y), texelSize)));

    if (count == 0) {
        return sampleRawDepth(uv);
    }

    sortDepths(depths, count);

    float nearestDepth = depths[0];
    float medianDepth = depths[count / 2];
    float farthestDepth = depths[count - 1];
    float nearAverageDepth = count > 1 ? 0.5 * (depths[0] + depths[1]) : nearestDepth;
    float relativeSpread = (farthestDepth - nearestDepth) / max(medianDepth, 1.0);
    float foregroundBlend = 0.35 * smoothstep(0.02, 0.16, relativeSpread);
    return mix(medianDepth, nearAverageDepth, foregroundBlend);
}

float computeFocusRange(float focusDistance) {
    float baseRange = max(ADV_POST_DOF_FOCUS_RANGE, 1e-3);
    float distance = max(focusDistance, 0.0);
    float distanceBlend = smoothstep(6.0, 48.0, distance);
    float proportionalRange = distance * mix(0.25, 1.2, distanceBlend);
    float baseBoost = baseRange * mix(1.0, 2.2, smoothstep(4.0, 20.0, distance));
    return max(baseRange, baseBoost + proportionalRange);
}

float computeFocusDeadZone(float focusDistance, float focusRange) {
    float distance = max(focusDistance, 0.0);
    float baseDeadZone = max(1.2, ADV_POST_DOF_FOCUS_RANGE * 0.45);
    float distanceDeadZone = distance * 0.10;
    float rangeDeadZone = focusRange * 0.45;
    return min(max(baseDeadZone, max(distanceDeadZone, rangeDeadZone)), focusRange * 0.85);
}

float computeSignedCoC(float depthValue, float focusDistance) {
    float focusRange = computeFocusRange(focusDistance);
    float deadZone = computeFocusDeadZone(focusDistance, focusRange);
    float focusDelta = depthValue - focusDistance;
    float cocMagnitude = max(abs(focusDelta) - deadZone, 0.0) / max(focusRange - deadZone, 1e-3);
    return sign(focusDelta) * clamp(cocMagnitude, 0.0, 1.0);
}

float computeBlurRadius(float depthValue, float focusDistance) {
    return abs(computeSignedCoC(depthValue, focusDistance)) * ADV_POST_DOF_MAX_RADIUS;
}

float gaussianWeight(float distancePixels, float sigma) {
    float sigma2 = max(sigma * sigma, 1e-3);
    return exp(-0.5 * distancePixels * distancePixels / sigma2);
}

void main() {
    vec2 resolution = vec2(textureSize(postDofInput, 0));
    vec2 texelSize = 1.0 / resolution;
    vec3 baseColor = texture(postDofInput, fragTexCoord).rgb;

#if ADV_POST_ENABLE_DOF == 0
    outColor = vec4(baseColor, 1.0);
    return;
#else
    float focusDistance = ADV_POST_DOF_FOCUS_DISTANCE;
#if ADV_POST_DOF_AUTO_FOCUS != 0
    float temporalFocus = texture(postDofFocus, vec2(0.5)).r;
    if (temporalFocus > 0.0 && temporalFocus < 100000.0) {
        focusDistance = temporalFocus;
    } else {
        float autoFocusSampleRadius = max(ADV_POST_DOF_AUTO_FOCUS_RADIUS, max(texelSize.x, texelSize.y));
        float autoFocusCenterDepth = sampleRawDepth(vec2(0.5));
        float autoFocusDepths[9];
        int validAutoFocusDepthCount = 0;

        for (int i = 0; i < 9; ++i) {
            vec2 uv = clampUv(vec2(0.5) + ADV_AUTO_FOCUS_OFFSETS[i] * autoFocusSampleRadius, texelSize);
            float depthValue = sampleRawDepth(uv);
            if (depthValue <= 0.0) {
                continue;
            }

            autoFocusDepths[validAutoFocusDepthCount] = depthValue;
            ++validAutoFocusDepthCount;
        }

        if (validAutoFocusDepthCount > 0) {
            sortDepths(autoFocusDepths, validAutoFocusDepthCount);

            int mid0 = (validAutoFocusDepthCount - 1) / 2;
            int mid1 = validAutoFocusDepthCount / 2;
            float medianDepth = 0.5 * (autoFocusDepths[mid0] + autoFocusDepths[mid1]);

            float bandDepth = medianDepth;
            if (validAutoFocusDepthCount >= 5) {
                int lower = max(mid0 - 1, 0);
                int upper = min(mid1 + 1, validAutoFocusDepthCount - 1);
                float sum = 0.0;
                int count = 0;
                for (int i = lower; i <= upper; ++i) {
                    sum += autoFocusDepths[i];
                    ++count;
                }
                bandDepth = sum / max(count, 1);
            }

            if (autoFocusCenterDepth <= 0.0) {
                autoFocusCenterDepth = medianDepth;
            }

            float farBlend = smoothstep(16.0, 96.0, max(autoFocusCenterDepth, bandDepth));
            focusDistance = mix(autoFocusCenterDepth, bandDepth, farBlend);
        }
    }
#endif

    vec2 blurAxis = vec2(ADV_POST_DOF_DIRECTION_X, ADV_POST_DOF_DIRECTION_Y);
    float axisLength = length(blurAxis);
    if (axisLength <= 1e-6) {
        outColor = vec4(baseColor, 1.0);
        return;
    }
    blurAxis /= axisLength;

    float centerDepth = sampleStableDepth(fragTexCoord, texelSize);
    float blurRadius = computeBlurRadius(centerDepth, focusDistance);
    if (blurRadius < 0.25) {
        outColor = vec4(baseColor, 1.0);
        return;
    }

    float centerCoC = computeSignedCoC(centerDepth, focusDistance);
    float normalizedBlur = clamp(blurRadius / max(ADV_POST_DOF_MAX_RADIUS, 1e-3), 0.0, 1.0);
    float focusRange = computeFocusRange(focusDistance);
    int maxTapRadius = max(ADV_POST_DOF_SAMPLE_COUNT / 2, 4);
    int tapRadius = int(clamp(ceil(blurRadius * 1.75), 1.0, float(maxTapRadius)));
    float sigma = max(blurRadius * 0.55, 0.85);
    float centerWeight = gaussianWeight(0.0, sigma) * mix(1.15, 0.75, normalizedBlur);
    vec3 accum = baseColor * centerWeight;
    float weightSum = centerWeight;

    for (int i = 1; i <= tapRadius; ++i) {
        float tapDistance = float(i);
        float baseWeight = gaussianWeight(tapDistance, sigma);
        vec2 axisStep = blurAxis * (tapDistance / resolution);

        for (int sign = -1; sign <= 1; sign += 2) {
            vec2 sampleUv = clampUv(fragTexCoord + axisStep * float(sign), texelSize);
            vec3 sampleColor = texture(postDofInput, sampleUv).rgb;
            float sampleDepth = sampleStableDepth(sampleUv, texelSize);
            float sampleRadius = computeBlurRadius(sampleDepth, focusDistance);
            float sampleCoC = computeSignedCoC(sampleDepth, focusDistance);

            float depthTolerance = max(0.45, min(centerDepth, focusDistance) * 0.028 + focusRange * 0.18);
            float depthDelta = sampleDepth - centerDepth;
            float farDepthDelta = max(depthDelta, 0.0);
            float farOcclusionBase =
                1.0 - smoothstep(depthTolerance * 0.9, depthTolerance * 6.0, farDepthDelta);
            float deepBleed =
                smoothstep(depthTolerance * 0.35, depthTolerance * 4.5, farDepthDelta) *
                smoothstep(0.12, 0.75, normalizedBlur);
            float farOcclusionWeight = mix(farOcclusionBase, 1.0, 0.32 * deepBleed);
            float nearCoverage = smoothstep(tapDistance - 1.4, tapDistance + 1.4, sampleRadius + 0.45);
            float nearOcclusionWeight = mix(0.28, 1.0, nearCoverage);
            float depthWeight = sampleDepth < centerDepth ? nearOcclusionWeight : farOcclusionWeight;
            float edgeContinuity =
                1.0 - smoothstep(depthTolerance * 1.8, depthTolerance * 7.5, abs(depthDelta));
            float cocTolerance = mix(0.75, 2.5, normalizedBlur);
            float cocWeight = 1.0 - smoothstep(cocTolerance, cocTolerance * 2.8, abs(sampleCoC - centerCoC));
            float sideWeight = sampleCoC * centerCoC < -0.001 ? 0.3 : 1.0;
            float coverageWeight =
                smoothstep(tapDistance - 1.35, tapDistance + 1.7, max(sampleRadius, blurRadius) + 0.55);
            float weight = baseWeight * depthWeight * mix(0.7, 1.0, edgeContinuity) *
                           mix(0.4, 1.0, cocWeight) * sideWeight * mix(0.45, 1.0, coverageWeight);

            accum += sampleColor * weight;
            weightSum += weight;
        }
    }

    outColor = vec4(accum / max(weightSum, 1e-4), 1.0);
#endif
}
