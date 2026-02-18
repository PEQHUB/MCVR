#ifndef CLOUDS_GLSL
#define CLOUDS_GLSL

// Lightweight procedural cloud slab.
// Designed to be used from raygen (segment integration) and closest-hit (sun transmittance).

struct CloudSegmentResult {
    vec3 L; // in-scattered radiance (already accounts for self-shadow)
    vec3 T; // transmittance along the segment
};

bool cloudTileEnabled(SkyUBO skyUBO) {
    // cloudTile.x is textures[] index for the 65x65 R8 mask
    return skyUBO.cloudTile.x >= 0;
}

// Vanilla cloud mapping constants.
const float CLOUD_CELL_SIZE = 12.0;
const float CLOUD_LAYER_THICKNESS = 4.0;
const float CLOUD_SCROLL_SPEED_X = 0.030000001;
const float CLOUD_Z_BIAS = 3.96;

uint cloudHash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float cloudHash31(ivec3 p) {
    uint x = uint(p.x) * 0x8da6b343u;
    uint y = uint(p.y) * 0xd8163841u;
    uint z = uint(p.z) * 0xcb1ab31fu;
    return float(cloudHash_u32(x ^ y ^ z)) * (1.0 / 4294967296.0);
}

vec4 cloudFetchMaskRGBA(SkyUBO skyUBO, ivec2 tileCoord) {
    // textures[] must be declared by the including shader (set=0,binding=0)
    return texelFetch(textures[nonuniformEXT(skyUBO.cloudTile.x)], tileCoord, 0);
}

float cloudOcc01(SkyUBO skyUBO, ivec2 tc) {
    if (tc.x < 0 || tc.x > 64 || tc.y < 0 || tc.y > 64) return 0.0;
    return cloudFetchMaskRGBA(skyUBO, tc).r > 0.5 ? 1.0 : 0.0;
}

// Borders from mask texture: G=N, B=E, A=S. West is taken from left neighbor's east.
void cloudBorders(SkyUBO skyUBO, ivec2 tc, out float bN, out float bE, out float bS, out float bW) {
    if (tc.x < 0 || tc.x > 64 || tc.y < 0 || tc.y > 64) {
        bN = bE = bS = bW = 0.0;
        return;
    }
    vec4 m = cloudFetchMaskRGBA(skyUBO, tc);
    bN = m.g > 0.5 ? 1.0 : 0.0;
    bE = m.b > 0.5 ? 1.0 : 0.0;
    bS = m.a > 0.5 ? 1.0 : 0.0;
    bW = (tc.x > 0) ? (cloudFetchMaskRGBA(skyUBO, tc + ivec2(-1, 0)).b > 0.5 ? 1.0 : 0.0) : 0.0;
}

float cloudEdgeDistance01(vec2 inCell, float bN, float bE, float bS, float bW) {
    float dW = inCell.x;
    float dE = 1.0 - inCell.x;
    float dN = inCell.y;
    float dS = 1.0 - inCell.y;

    float d = 1e6;
    // Axis-aligned edge distances for single exposed borders.
    if (bW > 0.5) d = min(d, dW);
    if (bE > 0.5) d = min(d, dE);
    if (bN > 0.5) d = min(d, dN);
    if (bS > 0.5) d = min(d, dS);

    // Corner rounding: where two perpendicular borders meet, use Euclidean
    // distance to the corner point instead of the axis-aligned minimum.
    // This rounds the puffiness fade at corners instead of leaving a sharp 90° edge.
    if (bN > 0.5 && bW > 0.5) d = min(d, length(vec2(dW, dN)));
    if (bN > 0.5 && bE > 0.5) d = min(d, length(vec2(dE, dN)));
    if (bS > 0.5 && bW > 0.5) d = min(d, length(vec2(dW, dS)));
    if (bS > 0.5 && bE > 0.5) d = min(d, length(vec2(dE, dS)));

    if (d > 1e5) d = 1.0;
    return d;
}

// Map camera-relative position into the 65x65 Fancy tile mask.
// skyUBO.cloudWrap.xy stores vanilla intra-cell offsets (l,m) in blocks ([0,12)).
bool cloudSampleTileMask(vec3 pCam, SkyUBO skyUBO, out vec2 inCell, out ivec2 tileCoord) {
    if (!cloudTileEnabled(skyUBO)) return false;

    float xLocal = pCam.x + skyUBO.cloudWrap.x;
    float zLocal = pCam.z + skyUBO.cloudWrap.y;

    float fx = xLocal / CLOUD_CELL_SIZE;
    float fz = zLocal / CLOUD_CELL_SIZE;

    int cellX = int(floor(fx));
    int cellZ = int(floor(fz));

    // Match vanilla draw distance (+/- 32 cells around center, inclusive).
    if (abs(cellX) > 32 || abs(cellZ) > 32) return false;

    tileCoord = ivec2(cellX + 32, cellZ + 32);
    inCell = fract(vec2(fx, fz));
    return true;
}

float cloudValueNoise3(vec3 p) {
    ivec3 i = ivec3(floor(p));
    vec3 f = fract(p);
    // Smoothstep for interpolation
    f = f * f * (3.0 - 2.0 * f);

    float n000 = cloudHash31(i + ivec3(0, 0, 0));
    float n100 = cloudHash31(i + ivec3(1, 0, 0));
    float n010 = cloudHash31(i + ivec3(0, 1, 0));
    float n110 = cloudHash31(i + ivec3(1, 1, 0));
    float n001 = cloudHash31(i + ivec3(0, 0, 1));
    float n101 = cloudHash31(i + ivec3(1, 0, 1));
    float n011 = cloudHash31(i + ivec3(0, 1, 1));
    float n111 = cloudHash31(i + ivec3(1, 1, 1));

    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);

    float nxy0 = mix(nx00, nx10, f.y);
    float nxy1 = mix(nx01, nx11, f.y);

    return mix(nxy0, nxy1, f.z);
}

float cloudFbm3(vec3 p) {
    float a = 0.5;
    float s = 0.0;
    for (int i = 0; i < 3; i++) {
        s += a * cloudValueNoise3(p);
        p *= 2.02;
        a *= 0.5;
    }
    return s;
}

float cloudHash21(ivec2 p) {
    return cloudHash31(ivec3(p, 0));
}

vec2 cloudHash22(ivec2 p) {
    return vec2(cloudHash31(ivec3(p, 0)), cloudHash31(ivec3(p, 1)));
}

// 2D Worley F1 distance (0 = at feature point).
float cloudWorley2(vec2 p) {
    ivec2 i = ivec2(floor(p));
    vec2 f = fract(p);

    float minD2 = 1e9;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            ivec2 g = ivec2(x, y);
            vec2 o = cloudHash22(i + g);
            vec2 d = vec2(g) + o - f;
            minD2 = min(minD2, dot(d, d));
        }
    }
    return sqrt(minD2);
}

float cloudWorleyFbm2(vec2 p) {
    // Convert distance to "cellular" signal in [0,1].
    float d0 = cloudWorley2(p);
    float w0 = 1.0 - clamp(d0 / 1.41421356, 0.0, 1.0);

    float d1 = cloudWorley2(p * 2.03);
    float w1 = 1.0 - clamp(d1 / 1.41421356, 0.0, 1.0);

    return 0.65 * w0 + 0.35 * w1;
}

// Double-precision coordinate helpers to keep noise stable at large world coordinates.
float cloudWorley2_d(dvec2 p) {
    ivec2 i = ivec2(floor(p));
    vec2 f = vec2(fract(p));

    float minD2 = 1e9;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            ivec2 g = ivec2(x, y);
            vec2 o = cloudHash22(i + g);
            vec2 d = vec2(g) + o - f;
            minD2 = min(minD2, dot(d, d));
        }
    }
    return sqrt(minD2);
}

float cloudWorleyFbm2_d(dvec2 p) {
    float d0 = cloudWorley2_d(p);
    float w0 = 1.0 - clamp(d0 / 1.41421356, 0.0, 1.0);

    float d1 = cloudWorley2_d(p * 2.03);
    float w1 = 1.0 - clamp(d1 / 1.41421356, 0.0, 1.0);

    return 0.65 * w0 + 0.35 * w1;
}

float phaseHG(float cosTheta, float g) {
    float g2 = g * g;
    float denom = pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5);
    return (1.0 - g2) / (4.0 * PI * denom);
}

bool cloudIntersectSlabY(vec3 ro, vec3 rd, float y0, float y1, out float t0, out float t1) {
    float dy = rd.y;
    if (abs(dy) < 1e-6) {
        if (ro.y < min(y0, y1) || ro.y > max(y0, y1)) return false;
        t0 = 0.0;
        t1 = INF_DISTANCE;
        return true;
    }

    float ta = (y0 - ro.y) / dy;
    float tb = (y1 - ro.y) / dy;
    t0 = min(ta, tb);
    t1 = max(ta, tb);
    return t1 > 0.0;
}

// Density field: vanilla Fancy tile mask (hard silhouette) + continuous puffy interior modulation.
// pCam is camera-relative world position.
float cloudDensityAt(vec3 pCam, WorldUBO worldUBO, SkyUBO skyUBO) {
    // Fast clouds: uniform extruded slab driven by vanilla Fancy tile occupancy.
    float thickness = skyUBO.envCloud.y;
    float sigmaT = skyUBO.envCloud.z;
    if (isnan(skyUBO.envCloud.x) || thickness <= 0.0 || sigmaT <= 0.0) return 0.0;
    if (!cloudTileEnabled(skyUBO)) return 0.0;

    float baseYRel = skyUBO.envCloud.x - float(worldUBO.cameraPos.y);
    float topYRel = baseYRel + thickness;
    if (pCam.y < baseYRel || pCam.y > topYRel) return 0.0;

    vec2 inCell;
    ivec2 tc;
    if (!cloudSampleTileMask(pCam, skyUBO, inCell, tc)) return 0.0;

    float puffiness = clamp(skyUBO.cloudShape.x, 0.0, 2.0);
    float edgeSoft  = mix(0.0, 0.18, clamp(puffiness, 0.0, 1.0));

    bool occupied = cloudOcc01(skyUBO, tc) > 0.5;

    if (!occupied) {
        // Concave corner fill: when this empty cell has two adjacent axis-aligned
        // occupied neighbors that share a corner (forming a 90-degree internal notch),
        // treat the void as part of the cloud mass by returning density proportional
        // to how far the sample sits inside the filled corner radius.
        if (puffiness > 1e-4) {
            float oN = cloudOcc01(skyUBO, tc + ivec2( 0, -1));
            float oS = cloudOcc01(skyUBO, tc + ivec2( 0,  1));
            float oE = cloudOcc01(skyUBO, tc + ivec2( 1,  0));
            float oW = cloudOcc01(skyUBO, tc + ivec2(-1,  0));

            // Check all four concave corner configurations.
            // For each, the corner point is the shared corner between the two occupied
            // neighbors. The distance is measured from the corner towards the cell interior.
            // NW corner: N and W neighbors occupied → corner at (0,0) of this cell.
            float cornerDist = 1e6;
            if (oN > 0.5 && oW > 0.5)
                cornerDist = min(cornerDist, length(vec2(inCell.x, inCell.y)));
            // NE corner: N and E neighbors occupied → corner at (1,0).
            if (oN > 0.5 && oE > 0.5)
                cornerDist = min(cornerDist, length(vec2(1.0 - inCell.x, inCell.y)));
            // SW corner: S and W neighbors occupied → corner at (0,1).
            if (oS > 0.5 && oW > 0.5)
                cornerDist = min(cornerDist, length(vec2(inCell.x, 1.0 - inCell.y)));
            // SE corner: S and E neighbors occupied → corner at (1,1).
            if (oS > 0.5 && oE > 0.5)
                cornerDist = min(cornerDist, length(vec2(1.0 - inCell.x, 1.0 - inCell.y)));

            if (cornerDist < 1e5) {
                // The fill radius matches the edge-softening radius so the rounded
                // corner blends continuously with the adjacent cells' puffiness fade.
                float fillRadius = edgeSoft * 1.41421356; // diagonal extent of edgeSoft
                float fade = 1.0 - smoothstep(0.0, fillRadius, cornerDist);
                if (fade > 1e-3) return fade;
            }
        }
        return 0.0;
    }

    // Cell is occupied — apply optional edge softening near exposed borders.
    if (puffiness > 1e-4) {
        float bN, bE, bS, bW;
        cloudBorders(skyUBO, tc, bN, bE, bS, bW);
        float edgeDist = cloudEdgeDistance01(inCell, bN, bE, bS, bW);

        float fade = smoothstep(0.0, edgeSoft, edgeDist);
        if (fade <= 1e-3) return 0.0;
        return fade;
    }

    return 1.0;
}

float cloudDensityAtVisual(vec3 pCam, WorldUBO worldUBO, SkyUBO skyUBO) {
    float base = cloudDensityAt(pCam, worldUBO, skyUBO);
    if (base <= 0.0) return 0.0;

    // detailStrength = 0 (Fast mode): flat uniform slab, no FBM modulation.
    float detailStrength = clamp(skyUBO.cloudShape.z, 0.0, 3.0);
    if (detailStrength <= 1e-5) return base;

    float detailScale = max(skyUBO.cloudShape.y, 1e-3);

    // World-anchored 3D value FBM. cloudFbm3 (3 octaves) returns ~[0, 0.875].
    dvec3 pWorld = worldUBO.cameraPos.xyz + dvec3(pCam);
    vec3 pw = vec3(mod(pWorld, 4096.0));

    // At detailScale=1: base wavelength ~24 blocks. Higher detailScale = finer blobs.
    vec3 noiseCoord = pw * (detailScale / 24.0);
    float n = cloudFbm3(noiseCoord);

    // Symmetric deviation around 0.5: denser centers, lighter pockets — no holes.
    // detailStrength=1.0 → modD in [0.6, 1.4]. detailStrength=2.0 → [0.2, 1.8].
    float deviation = (n - 0.5) * 2.0;
    float modD = clamp(1.0 + deviation * 0.4 * detailStrength, 0.2, 1.8);

    // Altitude fade: suppress noise at top/bottom 15% of slab for smooth silhouette.
    float thickness = skyUBO.envCloud.y;
    float baseYRel = skyUBO.envCloud.x - float(worldUBO.cameraPos.y);
    float h = clamp((pCam.y - baseYRel) / max(thickness, 1e-3), 0.0, 1.0);
    float altFade = smoothstep(0.0, 0.15, h) * (1.0 - smoothstep(0.85, 1.0, h));
    modD = mix(1.0, modD, altFade);

    return base * modD;
}


vec3 cloudMainLightRadiance(SkyUBO skyUBO, out vec3 toLight) {
    vec3 sunDir = normalize(skyUBO.sunDirection);
    vec3 radiance;

    // Use moon when sun is below horizon.
    if (sunDir.y > 0.0) {
        toLight = sunDir;
        radiance = (skyUBO.sunRadiance * skyUBO.envCelestial.z);
    } else {
        toLight = normalize(skyUBO.moonDirection);
        radiance = (skyUBO.moonRadiance * skyUBO.envCelestial.w) * 0.05;
    }

    float threshold = 0.3;
    float factor = 1.0;
    if (abs(toLight.y) < threshold) {
        factor = sin(PI / (2.0 * threshold) * abs(toLight.y));
    }
    return radiance * factor;
}

float cloudTransmittance(vec3 ro, vec3 rd, float tMin, float tMax, WorldUBO worldUBO, SkyUBO skyUBO, int steps) {
    float thickness = skyUBO.envCloud.y;
    float sigmaT = skyUBO.envCloud.z;
    if (isnan(skyUBO.envCloud.x) || thickness <= 0.0 || sigmaT <= 0.0) return 1.0;
    if (!cloudTileEnabled(skyUBO)) return 1.0;

    float baseYRel = skyUBO.envCloud.x - float(worldUBO.cameraPos.y);
    float topYRel = baseYRel + thickness;

    float slabT0, slabT1;
    if (!cloudIntersectSlabY(ro, rd, baseYRel, topYRel, slabT0, slabT1)) return 1.0;

    float a = max(tMin, slabT0);
    float b = min(tMax, slabT1);
    if (b <= a) return 1.0;

    // DDA traversal through the 12x12 cell grid in local cloud space.
    vec3 p0 = ro + rd * a;
    float x = p0.x + skyUBO.cloudWrap.x;
    float z = p0.z + skyUBO.cloudWrap.y;

    int cellX = int(floor(x / CLOUD_CELL_SIZE));
    int cellZ = int(floor(z / CLOUD_CELL_SIZE));
    if (abs(cellX) > 32 || abs(cellZ) > 32) return 1.0;

    int stepX = (rd.x > 0.0) ? 1 : -1;
    int stepZ = (rd.z > 0.0) ? 1 : -1;

    float invDx = (abs(rd.x) > 1e-6) ? (1.0 / rd.x) : 0.0;
    float invDz = (abs(rd.z) > 1e-6) ? (1.0 / rd.z) : 0.0;

    float nextBoundaryX = (float(cellX + (rd.x > 0.0 ? 1 : 0)) * CLOUD_CELL_SIZE);
    float nextBoundaryZ = (float(cellZ + (rd.z > 0.0 ? 1 : 0)) * CLOUD_CELL_SIZE);

    float tMaxX = (abs(rd.x) > 1e-6) ? (a + (nextBoundaryX - x) * invDx) : INF_DISTANCE;
    float tMaxZ = (abs(rd.z) > 1e-6) ? (a + (nextBoundaryZ - z) * invDz) : INF_DISTANCE;
    float tDeltaX = (abs(rd.x) > 1e-6) ? (CLOUD_CELL_SIZE / abs(rd.x)) : INF_DISTANCE;
    float tDeltaZ = (abs(rd.z) > 1e-6) ? (CLOUD_CELL_SIZE / abs(rd.z)) : INF_DISTANCE;

    float tauSum = 0.0;
    float t = a;
    int maxIter = 256;
    for (int iter = 0; iter < maxIter && t < b && tauSum < 20.0; iter++) {
        float tNext = min(b, min(tMaxX, tMaxZ));
        float ds = tNext - t;
        if (ds <= 0.0) break;

        // Evaluate a stable density at the segment midpoint.
        float tm = t + 0.5 * ds;
        vec3 pm = ro + rd * tm;
        bool noiseAffects = skyUBO.cloudLighting.w > 0.5;
        float d = noiseAffects ? cloudDensityAtVisual(pm, worldUBO, skyUBO) : cloudDensityAt(pm, worldUBO, skyUBO);
        if (d > 1e-4) {
            tauSum += (sigmaT * d) * ds;
        }

        t = tNext;

        bool stepBoth = abs(tMaxX - tMaxZ) < 1e-6;
        if (stepBoth) {
            tMaxX += tDeltaX;
            tMaxZ += tDeltaZ;
            cellX += stepX;
            cellZ += stepZ;
        } else if (tMaxX < tMaxZ) {
            tMaxX += tDeltaX;
            cellX += stepX;
        } else {
            tMaxZ += tDeltaZ;
            cellZ += stepZ;
        }

        if (abs(cellX) > 32 || abs(cellZ) > 32) break;
    }

    return exp(-tauSum);
}

void cloudIntegrateSubSegment(vec3 ro, vec3 rd,
                             float runStart, float runLen,
                             float tA, float tB,
                             WorldUBO worldUBO, SkyUBO skyUBO,
                             float sigmaT, float sigmaS,
                             vec3 lightRadiance, vec3 skyAmbient, float ambientStrength,
                             float brightness,
                             float phaseSS, float phaseISO,
                             float tau0, float tau1,
                             inout vec3 Lacc, inout float Tr) {
    float ds = tB - tA;
    if (ds <= 1e-6 || Tr <= 1e-4) return;

    float tm = tA + 0.5 * ds;
    vec3 pm = ro + rd * tm;
    float u = clamp((tm - runStart) / max(runLen, 1e-6), 0.0, 1.0);
    float tauL = mix(tau0, tau1, u);

    float density = cloudDensityAtVisual(pm, worldUBO, skyUBO);
    if (density <= 1e-4) return;

    float ext = sigmaT * density;
    float stepT = exp(-ext * ds);

    float ms = 1.0 - pow(max(Tr, 1e-6), 0.25);
    float phase = mix(phaseSS, phaseISO, ms);

    float tauLEff = tauL * (1.0 - ms * 0.75);
    float TLightEff = exp(-tauLEff);

    vec3 Li = (lightRadiance * TLightEff + skyAmbient * ambientStrength) * brightness;
    Li *= (1.0 + ms * 6.0);

    float scatterIntegral = (1.0 - stepT) / max(ext, 1e-6);
    vec3 dL = (sigmaS * density * scatterIntegral) * phase * Li;
    Lacc += Tr * dL;
    Tr *= stepT;
}

void cloudIntegrateRun(vec3 ro, vec3 rd,
                       float s, float e,
                       WorldUBO worldUBO, SkyUBO skyUBO, inout uint seed,
                       vec3 toLight,
                       vec3 lightRadiance, vec3 skyAmbient,
                       float ambientStrength,
                       float sigmaT, float sigmaS,
                       float brightness,
                       float phaseSS, float phaseISO,
                       int stepsView, int stepsLight,
                       bool analyticMode,
                       inout vec3 Lacc, inout float Tr) {
    float len = e - s;
    if (len <= 1e-6 || Tr <= 1e-4) return;

    // Self-shadowing + sun blocking along the sun ray.
    // Sample tau at 1-2 points along the run and interpolate.
    int lightSamples = (stepsLight >= 6) ? 2 : 1;
    float tau0;
    float tau1;
    if (lightSamples == 1) {
        vec3 pMid = ro + rd * (0.5 * (s + e));
        float TL = cloudTransmittance(pMid + toLight * 0.01, toLight, 0.0, 1e6, worldUBO, skyUBO, 0);
        tau0 = -log(max(TL, 1e-6));
        tau1 = tau0;
    } else {
        vec3 pA = ro + rd * mix(s, e, 0.25);
        vec3 pB = ro + rd * mix(s, e, 0.75);
        float TL0 = cloudTransmittance(pA + toLight * 0.01, toLight, 0.0, 1e6, worldUBO, skyUBO, 0);
        float TL1 = cloudTransmittance(pB + toLight * 0.01, toLight, 0.0, 1e6, worldUBO, skyUBO, 0);
        tau0 = -log(max(TL0, 1e-6));
        tau1 = -log(max(TL1, 1e-6));
    }

    float sunOcc = max(skyUBO.cloudLighting.z, 0.0);
    // pow(T, sunOcc) == exp(-tau * sunOcc)
    tau0 *= sunOcc;
    tau1 *= sunOcc;

    if (analyticMode) {
        // Analytic integration for a constant-density slab.
        float ext = sigmaT;
        float Tseg = exp(-ext * len);

        float TrMid = Tr * exp(-ext * len * 0.5);
        float ms = 1.0 - pow(max(TrMid, 1e-6), 0.25);
        float phase = mix(phaseSS, phaseISO, ms);

        float tauMid = 0.5 * (tau0 + tau1);
        float tauLEff = tauMid * (1.0 - ms * 0.75);
        float TLightEff = exp(-tauLEff);

        vec3 Li = (lightRadiance * TLightEff + skyAmbient * ambientStrength) * brightness;
        Li *= (1.0 + ms * 6.0);

        float scatterIntegral = (1.0 - Tseg) / max(ext, 1e-6);
        vec3 dL = (sigmaS * scatterIntegral) * phase * Li;
        Lacc += Tr * dL;
        Tr *= Tseg;
        return;
    }

    // Fixed step-count integration with a per-run random phase.
    // Step count does not depend on ray direction, so it avoids the view-dependent ring bands.
    int N = clamp(stepsView, 6, 24);
    float dt = len / float(N);
    float phaseJitter = rand(seed);

    float tCur = s;
    float tFirst = min(e, tCur + phaseJitter * dt);

    // Prefix segment [s, tFirst]
    if (tFirst > tCur) {
        cloudIntegrateSubSegment(ro, rd, s, len, tCur, tFirst,
                                 worldUBO, skyUBO,
                                 sigmaT, sigmaS,
                                 lightRadiance, skyAmbient, ambientStrength,
                                 brightness,
                                 phaseSS, phaseISO,
                                 tau0, tau1,
                                 Lacc, Tr);
    }

    tCur = tFirst;
    for (int i = 0; i < N && tCur < e && Tr > 1e-4; i++) {
        float tNext = min(e, tCur + dt);
        cloudIntegrateSubSegment(ro, rd, s, len, tCur, tNext,
                                 worldUBO, skyUBO,
                                 sigmaT, sigmaS,
                                 lightRadiance, skyAmbient, ambientStrength,
                                 brightness,
                                 phaseSS, phaseISO,
                                 tau0, tau1,
                                 Lacc, Tr);
        tCur = tNext;
    }
}

CloudSegmentResult integrateCloudSegment(vec3 ro, vec3 rd, float tMin, float tMax, WorldUBO worldUBO, SkyUBO skyUBO,
                                         samplerCube skyFull, inout uint seed,
                                         int stepsView, int stepsLight) {
    CloudSegmentResult o;
    o.L = vec3(0.0);
    o.T = vec3(1.0);

    // Guard.
    float thickness = skyUBO.envCloud.y;
    float sigmaT = skyUBO.envCloud.z;
    if (isnan(skyUBO.envCloud.x) || thickness <= 0.0 || sigmaT <= 0.0) return o;
    if (!cloudTileEnabled(skyUBO)) return o;
    if (skyUBO.skyType != 1) return o; // NORMAL only
    if (skyUBO.hasBlindnessOrDarkness > 0) return o;
    if (skyUBO.cameraSubmersionType != 3) return o; // NONE only

    float baseYRel = skyUBO.envCloud.x - float(worldUBO.cameraPos.y);
    float topYRel = baseYRel + thickness;

    float slabT0, slabT1;
    if (!cloudIntersectSlabY(ro, rd, baseYRel, topYRel, slabT0, slabT1)) return o;
    float a = max(tMin, slabT0);
    float b = min(tMax, slabT1);
    if (b <= a) return o;


    vec3 toLight;
    vec3 lightRadiance = cloudMainLightRadiance(skyUBO, toLight);

    // Ambient fill from sky cubemap.
    vec3 skyAmbient = texture(skyFull, vec3(0.0, 1.0, 0.0)).rgb * skyUBO.envSky.x;
    skyAmbient = max(skyAmbient, vec3(0.0));

    float ambientStrength = max(skyUBO.cloudLighting.y, 0.0);

    float g = clamp(skyUBO.cloudShape.w, 0.0, 0.95);
    float cosTheta = clamp(dot(toLight, -rd), -1.0, 1.0);
    float phaseSS = phaseHG(cosTheta, g);
    float phaseISO = 1.0 / (4.0 * PI);

    // Cloud brightness: user multiplier (1.0 = default).
    float brightness = clamp(skyUBO.envCloud.w, 0.0, 3.0);

    // Map brightness to a higher effective scattering albedo so 100% reads "white".
    // (Brightness is a creative control, not physically strict.)
    float albedo = clamp(0.70 + 0.30 * brightness, 0.0, 0.995);
    float sigmaS = sigmaT * albedo;

    // Decide whether we can use an analytic constant-density slab integration.
    // If any shape/detail modulation is enabled, fall back to stepped integration.
    float puffiness = clamp(skyUBO.cloudShape.x, 0.0, 2.0);
    float detailStrength = clamp(skyUBO.cloudShape.z, 0.0, 3.0);
    // Analytic (flat slab) only when detailStrength == 0 (Fast mode).
    // Any non-zero detailStrength (Fancy mode) uses stepped 3D FBM integration.
    bool analyticMode = (puffiness <= 1e-4) && (detailStrength <= 1e-5);

    // DDA traversal through the 12x12 cell grid in local cloud space.
    vec3 p0 = ro + rd * a;
    float x = p0.x + skyUBO.cloudWrap.x;
    float z = p0.z + skyUBO.cloudWrap.y;

    int cellX = int(floor(x / CLOUD_CELL_SIZE));
    int cellZ = int(floor(z / CLOUD_CELL_SIZE));
    if (abs(cellX) > 32 || abs(cellZ) > 32) return o;

    int stepX = (rd.x > 0.0) ? 1 : -1;
    int stepZ = (rd.z > 0.0) ? 1 : -1;

    float invDx = (abs(rd.x) > 1e-6) ? (1.0 / rd.x) : 0.0;
    float invDz = (abs(rd.z) > 1e-6) ? (1.0 / rd.z) : 0.0;

    float nextBoundaryX = (float(cellX + (rd.x > 0.0 ? 1 : 0)) * CLOUD_CELL_SIZE);
    float nextBoundaryZ = (float(cellZ + (rd.z > 0.0 ? 1 : 0)) * CLOUD_CELL_SIZE);

    float tMaxX = (abs(rd.x) > 1e-6) ? (a + (nextBoundaryX - x) * invDx) : INF_DISTANCE;
    float tMaxZ = (abs(rd.z) > 1e-6) ? (a + (nextBoundaryZ - z) * invDz) : INF_DISTANCE;
    float tDeltaX = (abs(rd.x) > 1e-6) ? (CLOUD_CELL_SIZE / abs(rd.x)) : INF_DISTANCE;
    float tDeltaZ = (abs(rd.z) > 1e-6) ? (CLOUD_CELL_SIZE / abs(rd.z)) : INF_DISTANCE;

    float Tr = 1.0;
    float t = a;

    // Group adjacent occupied cells into contiguous ray intervals.
    // This prevents visible "cell blocks" when multiple clouds are directly adjacent.
    bool inRun = false;
    float runStart = a;
    float runEnd = a;

    // Integrate a contiguous occupied run.

    int maxIter = 256;
    for (int iter = 0; iter < maxIter && t < b && Tr > 1e-4; iter++) {
        float tNext = min(b, min(tMaxX, tMaxZ));
        float ds = tNext - t;
        if (ds <= 0.0) break;

        ivec2 tc = ivec2(cellX + 32, cellZ + 32);
        float occ = cloudOcc01(skyUBO, tc);

        if (occ > 0.5) {
            if (!inRun) {
                inRun = true;
                runStart = t;
                runEnd = tNext;
            } else {
                runEnd = tNext;
            }
        } else {
            if (inRun) {
                cloudIntegrateRun(ro, rd,
                                  runStart, runEnd,
                                  worldUBO, skyUBO, seed,
                                  toLight,
                                  lightRadiance, skyAmbient,
                                  ambientStrength,
                                  sigmaT, sigmaS,
                                  brightness,
                                  phaseSS, phaseISO,
                                  stepsView, stepsLight,
                                  analyticMode,
                                  o.L, Tr);
                inRun = false;
            }
        }

        t = tNext;

        bool stepBoth = abs(tMaxX - tMaxZ) < 1e-6;
        if (stepBoth) {
            tMaxX += tDeltaX;
            tMaxZ += tDeltaZ;
            cellX += stepX;
            cellZ += stepZ;
        } else if (tMaxX < tMaxZ) {
            tMaxX += tDeltaX;
            cellX += stepX;
        } else {
            tMaxZ += tDeltaZ;
            cellZ += stepZ;
        }

        if (abs(cellX) > 32 || abs(cellZ) > 32) break;
    }

    if (inRun && Tr > 1e-4) {
        cloudIntegrateRun(ro, rd,
                          runStart, runEnd,
                          worldUBO, skyUBO, seed,
                          toLight,
                          lightRadiance, skyAmbient,
                          ambientStrength,
                          sigmaT, sigmaS,
                          brightness,
                          phaseSS, phaseISO,
                          stepsView, stepsLight,
                          analyticMode,
                          o.L, Tr);
    }

    o.T = vec3(Tr);
    return o;
}

#endif // CLOUDS_GLSL
