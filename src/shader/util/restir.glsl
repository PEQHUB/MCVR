#ifndef RESTIR_GLSL
#define RESTIR_GLSL

struct Reservoir {
    uint lightStableId;  // stableId of selected light
    float W;             // importance weight
    float M;             // sample count
    float targetPdf;     // p-hat of selected light
    float wSum;          // running weight sum (internal)
    int lightIdx;        // index into current frame's light buffer (-1 if none)
    vec3 unshadowed;     // cached unshadowed contribution of selected light
    vec3 lightDir;       // direction to selected light
    float lightDist;     // distance to selected light
};

Reservoir createReservoir() {
    Reservoir r;
    r.lightStableId = 0u;
    r.W = 0.0;
    r.M = 0.0;
    r.targetPdf = 0.0;
    r.wSum = 0.0;
    r.lightIdx = -1;
    r.unshadowed = vec3(0.0);
    r.lightDir = vec3(0.0);
    r.lightDist = 0.0;
    return r;
}

// Update reservoir with a new candidate sample (streaming WRS)
bool updateReservoir(inout Reservoir r, uint stableId, int idx, float weight,
                     float targetPdf, vec3 unshadowed, vec3 dir, float dist,
                     inout uint seed) {
    r.wSum += weight;
    r.M += 1.0;
    if (rand(seed) * r.wSum < weight) {
        r.lightStableId = stableId;
        r.lightIdx = idx;
        r.targetPdf = targetPdf;
        r.unshadowed = unshadowed;
        r.lightDir = dir;
        r.lightDist = dist;
        return true;
    }
    return false;
}

// Finalize reservoir: compute importance weight W
void finalizeReservoir(inout Reservoir r) {
    if (r.targetPdf > 0.0 && r.M > 0.0) {
        r.W = r.wSum / (r.M * r.targetPdf);
    } else {
        r.W = 0.0;
    }
}

// Merge another reservoir into this one
void mergeReservoir(inout Reservoir dst, Reservoir src, float srcTargetPdfAtDst, inout uint seed) {
    float weight = srcTargetPdfAtDst * src.W * src.M;
    dst.wSum += weight;
    dst.M += src.M;
    if (rand(seed) * dst.wSum < weight) {
        dst.lightStableId = src.lightStableId;
        dst.lightIdx = src.lightIdx;
        dst.targetPdf = srcTargetPdfAtDst;
        dst.unshadowed = src.unshadowed;
        dst.lightDir = src.lightDir;
        dst.lightDist = src.lightDist;
    }
}

// Pack reservoir into vec4 for image storage
vec4 packReservoir(Reservoir r) {
    return vec4(
        uintBitsToFloat(r.lightStableId),
        r.W,
        r.M,
        r.targetPdf
    );
}

// Unpack reservoir from vec4 image load
Reservoir unpackReservoir(vec4 packed) {
    Reservoir r = createReservoir();
    r.lightStableId = floatBitsToUint(packed.x);
    r.W = packed.y;
    r.M = abs(packed.z);  // abs: sign encodes previous visibility
    r.targetPdf = packed.w;
    return r;
}

// Pack reservoir with visibility bit encoded in sign of M
vec4 packReservoirVis(Reservoir r, bool visible) {
    return vec4(
        uintBitsToFloat(r.lightStableId),
        r.W,
        visible ? r.M : -r.M,
        r.targetPdf
    );
}

// Compute target PDF (luminance of unshadowed contribution)
float computeTargetPdf(vec3 unshadowedContrib) {
    return dot(unshadowedContrib, vec3(0.2126, 0.7152, 0.0722));
}

// BRDF-aware target PDF: includes NdotL for better importance sampling
float computeTargetPdfBRDF(vec3 unshadowedContrib, float NdotL) {
    return dot(unshadowedContrib * max(NdotL, 0.0), vec3(0.2126, 0.7152, 0.0722));
}

#endif
