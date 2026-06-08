#version 460

#ifndef VPT_POST_ENABLE_MOTION_BLUR
#define VPT_POST_ENABLE_MOTION_BLUR 0
#endif

#ifndef VPT_POST_MOTION_BLUR_STRENGTH
#define VPT_POST_MOTION_BLUR_STRENGTH 1.35
#endif

#ifndef VPT_POST_MOTION_BLUR_SAMPLE_COUNT
#define VPT_POST_MOTION_BLUR_SAMPLE_COUNT 10
#endif

layout(set = 3, binding = 0) uniform sampler2D postInputLdr;
layout(set = 3, binding = 1) uniform sampler2D postFirstHitDepth;
layout(set = 3, binding = 3) uniform sampler2D postMotionVector;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

vec2 clampUv(vec2 uv, vec2 texelSize) {
    return clamp(uv, texelSize * 0.5, vec2(1.0) - texelSize * 0.5);
}

float sampleConservativeDepth(vec2 uv, vec2 texelSize) {
    uv = clampUv(uv, texelSize);

    float depthValue = texture(postFirstHitDepth, uv).r;
    depthValue = min(depthValue, texture(postFirstHitDepth, clampUv(uv + vec2(texelSize.x, 0.0), texelSize)).r);
    depthValue = min(depthValue, texture(postFirstHitDepth, clampUv(uv - vec2(texelSize.x, 0.0), texelSize)).r);
    depthValue = min(depthValue, texture(postFirstHitDepth, clampUv(uv + vec2(0.0, texelSize.y), texelSize)).r);
    depthValue = min(depthValue, texture(postFirstHitDepth, clampUv(uv - vec2(0.0, texelSize.y), texelSize)).r);
    return depthValue;
}

void main() {
    vec2 resolution = vec2(textureSize(postInputLdr, 0));
    vec2 texelSize = 1.0 / resolution;
    vec3 baseColor = texture(postInputLdr, fragTexCoord).rgb;

#if VPT_POST_ENABLE_MOTION_BLUR == 0
    outColor = vec4(baseColor, 1.0);
    return;
#else
    vec2 centerMotion = texture(postMotionVector, fragTexCoord).xy;
    float centerMotionLength = length(centerMotion);
    float blurRadiusPixels = min(centerMotionLength * VPT_POST_MOTION_BLUR_STRENGTH * 1.15, 20.0);
    if (blurRadiusPixels < 0.25) {
        outColor = vec4(baseColor, 1.0);
        return;
    }

    vec2 blurDirectionUv = centerMotion / max(centerMotionLength, 1e-4) * (blurRadiusPixels / resolution);
    float centerDepth = sampleConservativeDepth(fragTexCoord, texelSize);

    float centerPreserve = mix(2.3, 1.2, clamp(blurRadiusPixels / 20.0, 0.0, 1.0));
    vec3 accum = baseColor * centerPreserve;
    float weightSum = centerPreserve;

    for (int i = 0; i < VPT_POST_MOTION_BLUR_SAMPLE_COUNT; ++i) {
        float t = float(i + 1) / float(VPT_POST_MOTION_BLUR_SAMPLE_COUNT + 1);
        float shutter = mix(t, t * t, 0.35);
        float radialWeight = exp2(-2.3 * shutter);

        for (int sign = -1; sign <= 1; sign += 2) {
            vec2 sampleUv = clampUv(fragTexCoord + blurDirectionUv * shutter * float(sign), texelSize);
            vec3 sampleColor = texture(postInputLdr, sampleUv).rgb;
            float sampleDepth = sampleConservativeDepth(sampleUv, texelSize);
            vec2 sampleMotion = texture(postMotionVector, sampleUv).xy;

            float relativeDepthDelta = abs(sampleDepth - centerDepth) / max(1.0, min(centerDepth, sampleDepth));
            float depthWeight = 1.0 - smoothstep(0.02, 0.1, relativeDepthDelta);
            float motionWeight =
                1.0 - smoothstep(1.5, blurRadiusPixels * 1.8 + 1.5, length(sampleMotion - centerMotion));
            float colorWeight = 1.0 - smoothstep(0.2, 0.9, length(sampleColor - baseColor));
            float weight = radialWeight * depthWeight * mix(0.65, 1.0, motionWeight) * mix(0.6, 1.0, colorWeight);

            accum += sampleColor * weight;
            weightSum += weight;
        }
    }

    outColor = vec4(accum / max(weightSum, 1e-4), 1.0);
#endif
}
