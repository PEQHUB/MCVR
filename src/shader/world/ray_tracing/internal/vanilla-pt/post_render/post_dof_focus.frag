#version 460

#ifndef VPT_POST_DOF_AUTO_FOCUS
#define VPT_POST_DOF_AUTO_FOCUS 1
#endif

#ifndef VPT_POST_DOF_AUTO_FOCUS_RADIUS
#define VPT_POST_DOF_AUTO_FOCUS_RADIUS 0.012
#endif

#ifndef VPT_POST_DOF_FOCUS_DISTANCE
#define VPT_POST_DOF_FOCUS_DISTANCE 12.0
#endif

#ifndef VPT_POST_DOF_FOCUS_SPEED
#define VPT_POST_DOF_FOCUS_SPEED 0.2
#endif

#ifndef VPT_POST_DOF_FOCUS_HISTORY_READY
#define VPT_POST_DOF_FOCUS_HISTORY_READY false
#endif

#ifndef VPT_POST_DOF_PREVIOUS_FOCUS_BINDING
#define VPT_POST_DOF_PREVIOUS_FOCUS_BINDING 5
#endif

layout(set = 3, binding = 1) uniform sampler2D postFirstHitDepth;
layout(set = 4, binding = VPT_POST_DOF_PREVIOUS_FOCUS_BINDING) uniform sampler2D postPreviousDofFocus;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out float outFocusDistance;

const vec2 VPT_AUTO_FOCUS_OFFSETS[9] = vec2[](
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

void main() {
    vec2 depthResolution = vec2(textureSize(postFirstHitDepth, 0));
    vec2 texelSize = 1.0 / depthResolution;
    float targetFocus = VPT_POST_DOF_FOCUS_DISTANCE;
#if VPT_POST_DOF_AUTO_FOCUS != 0
    float sampleRadius = max(VPT_POST_DOF_AUTO_FOCUS_RADIUS, max(texelSize.x, texelSize.y));
    float centerDepth = sampleRawDepth(vec2(0.5));
    float depths[9];
    int validCount = 0;

    for (int i = 0; i < 9; ++i) {
        vec2 uv = clampUv(vec2(0.5) + VPT_AUTO_FOCUS_OFFSETS[i] * sampleRadius, texelSize);
        float depthValue = sampleRawDepth(uv);
        if (depthValue <= 0.0) {
            continue;
        }

        depths[validCount] = depthValue;
        ++validCount;
    }

    if (validCount > 0) {
        sortDepths(depths, validCount);

        int mid0 = (validCount - 1) / 2;
        int mid1 = validCount / 2;
        float medianDepth = 0.5 * (depths[mid0] + depths[mid1]);

        float bandDepth = medianDepth;
        if (validCount >= 5) {
            int lower = max(mid0 - 1, 0);
            int upper = min(mid1 + 1, validCount - 1);
            float sum = 0.0;
            int count = 0;
            for (int i = lower; i <= upper; ++i) {
                sum += depths[i];
                ++count;
            }
            bandDepth = sum / max(count, 1);
        }

        if (centerDepth <= 0.0) {
            centerDepth = medianDepth;
        }

        float farBlend = smoothstep(16.0, 96.0, max(centerDepth, bandDepth));
        targetFocus = mix(centerDepth, bandDepth, farBlend);
    }
#endif

    float previousFocus = texture(postPreviousDofFocus, vec2(0.5)).r;
    if (!VPT_POST_DOF_FOCUS_HISTORY_READY || previousFocus <= 0.0 || previousFocus >= 100000.0) {
        outFocusDistance = targetFocus;
        return;
    }

    float focusDelta = abs(targetFocus - previousFocus);
    float nearThreshold = max(0.5, targetFocus * 0.04);
    float farThreshold = max(4.0, targetFocus * 0.35);
    float distanceBoost = smoothstep(nearThreshold, farThreshold, focusDelta);
    float focusBlend = clamp(VPT_POST_DOF_FOCUS_SPEED * mix(0.45, 2.5, distanceBoost), 0.0, 1.0);
    outFocusDistance = mix(previousFocus, targetFocus, focusBlend);
}
