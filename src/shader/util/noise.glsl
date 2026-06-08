#ifndef NOISE_GLSL
#define NOISE_GLSL

// Noise type IDs (packed in material UBO pack3.w)
#define NOISE_SIMPLEX     0
#define NOISE_WORLEY_F1   1
#define NOISE_WORLEY_F2F1 2
#define NOISE_VORONOI     3
#define NOISE_RIDGED      4
#define NOISE_TURBULENCE  5
#define NOISE_MARBLE      6
#define NOISE_WOOD         7
#define NOISE_CHECKER      8
#define NOISE_BRICK        9
#define NOISE_HEX         10
#define NOISE_SCRATCHES   11
#define NOISE_DOTS        12
#define NOISE_GRADIENT    13
#define NOISE_RINGS       14
#define NOISE_CRACKLE     15
#define NOISE_WAVES       16
#define NOISE_CELLULAR    17
#define NOISE_EROSION     18
#define NOISE_FABRIC      19
#define NOISE_VALUE       20  // cheap smooth noise (~5 ALU)
#define NOISE_HASH_GRID   21  // per-cell random (~3 ALU)
#define NOISE_SINE_LINES  22  // parallel lines (~4 ALU)
#define NOISE_DIAMOND     23  // diamond plate grid (~8 ALU)

// ── Hash functions for Worley/Voronoi ──

// Hash vec3 → vec3 in [0,1]^3, using xxhash-style mixing
vec3 hash33(vec3 p) {
    uvec3 u = uvec3(floatBitsToUint(p.x), floatBitsToUint(p.y), floatBitsToUint(p.z));
    const uint PRIME1 = 2246822519u;
    const uint PRIME2 = 3266489917u;
    const uint PRIME3 = 668265263u;
    u.x += u.y * PRIME2;  u.x *= PRIME3;  u.x ^= (u.x >> 16);
    u.y += u.z * PRIME2;  u.y *= PRIME3;  u.y ^= (u.y >> 16);
    u.z += u.x * PRIME2;  u.z *= PRIME3;  u.z ^= (u.z >> 16);
    u += u.yzx * PRIME1;
    u ^= u >> 16;
    return vec3(u) * (1.0 / float(0xFFFFFFFFu));
}

// Hash vec3 → float in [0,1], for Voronoi cell ID
float hash31(vec3 p) {
    uvec3 u = uvec3(floatBitsToUint(p.x), floatBitsToUint(p.y), floatBitsToUint(p.z));
    uint h = u.x * 2246822519u + u.y * 3266489917u + u.z * 668265263u;
    h ^= h >> 16;  h *= 2246822519u;  h ^= h >> 16;
    return float(h) * (1.0 / float(0xFFFFFFFFu));
}

// ── 3D Simplex noise — Ashima Arts / Ian McEwan (MIT License) ──

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 10.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);

    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;

    i = mod289(i);
    vec4 p = permute(permute(permute(
        i.z + vec4(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4(0.0, i1.x, i2.x, 1.0));

    float n_ = 0.142857142857;
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);

    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);

    vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    vec4 m = max(0.5 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    return 105.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

// ── Worley (cellular) noise — 27-cell neighborhood search ──

// Returns (F1, F2) — distances to nearest and second-nearest feature points
vec2 worley(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    float f1 = 1.0;
    float f2 = 1.0;
    for (int z = -1; z <= 1; z++)
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
        vec3 neighbor = vec3(x, y, z);
        vec3 point = hash33(i + neighbor);
        float d = length(neighbor + point - f);
        if (d < f1) { f2 = f1; f1 = d; }
        else if (d < f2) { f2 = d; }
    }
    return vec2(f1, f2);
}

float worleyF1(vec3 p) { return worley(p).x; }
float worleyF2F1(vec3 p) { vec2 w = worley(p); return w.y - w.x; }

// Voronoi cell ID — flat value per cell
float voronoiId(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    float minDist = 1.0;
    vec3 nearestCell = vec3(0.0);
    for (int z = -1; z <= 1; z++)
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
        vec3 neighbor = vec3(x, y, z);
        vec3 point = hash33(i + neighbor);
        float d = length(neighbor + point - f);
        if (d < minDist) { minDist = d; nearestCell = i + neighbor; }
    }
    return hash31(nearestCell);
}

// ── Single-sample typed noise evaluation ──
// Returns value in approximately [-1, 1] for simplex types,
// [0, 1] for cellular types. Caller normalizes as needed.

// ── Geometric / structured patterns (IDs 8-19) ──
// All use full 3D coordinates. Output [0, 1].

// Checkerboard: alternating 3D cubes. ~3 ALU. [Light]
float checkerboard(vec3 p) {
    return mod(floor(p.x) + floor(p.y) + floor(p.z), 2.0);
}

// Brick: running bond mortar pattern. ~15 ALU. [Light]
// Uses XY of input — wrapping mode in the shader handles face projection.
float brick(vec3 p) {
    vec2 uv = p.xy;
    uv.x *= 0.5;
    float row = floor(uv.y);
    uv.x += mod(row, 2.0) * 0.5;
    vec2 f = fract(uv);
    vec2 mortar = smoothstep(0.0, 0.06, f) * smoothstep(0.0, 0.06, 1.0 - f);
    return mortar.x * mortar.y;
}

// Hexagonal: hex tiles with edges. ~25 ALU. [Light]
float hexagonal(vec3 p) {
    vec2 uv = p.xy;
    const vec2 s = vec2(1.0, 1.7320508);
    vec4 hC = floor(vec4(uv, uv - vec2(0.5, 1.0)) / s.xyxy) + 0.5;
    vec4 hF = vec4(uv - hC.xy * s, uv - (hC.zw + 0.5) * s);
    vec2 d = (dot(hF.xy, hF.xy) < dot(hF.zw, hF.zw)) ? hF.xy : hF.zw;
    float hexDist = max(abs(d.x), abs(d.y * 0.57735 + abs(d.x) * 0.5));
    return smoothstep(0.42, 0.48, 1.0 - hexDist * 2.0);
}

// Scratches: anisotropic 3D noise (brushed metal). ~30 ALU. [Medium]
// Now uses simplex directly — FBM octaves add detail layers.
float scratches(vec3 p) {
    vec3 stretched = vec3(p.x * 0.125, p.y, p.z * 0.125); // 8:1 stretch in XZ
    float raw = snoise(stretched);
    float absRaw = abs(raw);
    return clamp(absRaw * (3.0 - 2.0 * absRaw), 0.0, 1.0);
}

// Dots: circular holes. ~10 ALU. [Light]
// Uses XY of input — wrapping mode handles projection.
float dots(vec3 p) {
    vec2 f = fract(p.xy) - 0.5;
    float d = length(f);
    return smoothstep(0.28, 0.32, d);
}

// Gradient: linear ramp along Y. ~4 ALU. [Light]
float gradientNoise(vec3 p) {
    return fract(p.y);
}

// Rings: concentric circles. ~6 ALU. [Light]
// Uses XY of input.
float rings(vec3 p) {
    float d = length(p.xy);
    return sin(d * 6.2831853) * 0.5 + 0.5;
}

// Crackle: Voronoi cell edges (F2-F1). ~120 ALU. [Heavy]
float crackle(vec3 p) {
    float f2f1 = worleyF2F1(p);
    return smoothstep(0.0, 0.15, f2f1);
}

// ── New patterns (IDs 16-19) ──

// Waves: smooth 3D sine interference. ~8 ALU. [Light]
float waves(vec3 p) {
    float w = sin(p.x * 3.0) * sin(p.y * 3.7) * sin(p.z * 4.3);
    return w * 0.5 + 0.5;
}

// Cellular: smooth Worley F1 (rounded cells). ~80 ALU. [Heavy]
float cellular(vec3 p) {
    return 1.0 - worleyF1(p); // invert: 1 at cell centers, 0 at edges
}

// Erosion: turbulent ridged (layered cracks). ~40 ALU. [Medium]
// Single-sample version; FBM octaves add depth.
float erosion(vec3 p) {
    float n = abs(snoise(p));
    return 1.0 - n * n; // squared abs simplex, inverted = eroded surface
}

// Fabric: woven cross-hatch pattern. ~12 ALU. [Light]
float fabric(vec3 p) {
    vec2 uv = p.xy;
    float warp = sin(uv.x * 6.2831853) * 0.5 + 0.5;
    float weft = sin(uv.y * 6.2831853) * 0.5 + 0.5;
    return mix(warp, weft, step(0.5, fract(uv.x + uv.y)));
}

// ── Cheap noise types (IDs 20-23) — performance-focused ──

// Value noise: trilinear-interpolated hash. ~5 ALU per eval. [Very Light]
float valueNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f); // smoothstep
    float n00 = fract(sin(dot(i, vec3(127.1, 311.7, 74.7))) * 43758.5453);
    float n10 = fract(sin(dot(i + vec3(1,0,0), vec3(127.1, 311.7, 74.7))) * 43758.5453);
    float n01 = fract(sin(dot(i + vec3(0,1,0), vec3(127.1, 311.7, 74.7))) * 43758.5453);
    float n11 = fract(sin(dot(i + vec3(1,1,0), vec3(127.1, 311.7, 74.7))) * 43758.5453);
    float n001 = fract(sin(dot(i + vec3(0,0,1), vec3(127.1, 311.7, 74.7))) * 43758.5453);
    float n101 = fract(sin(dot(i + vec3(1,0,1), vec3(127.1, 311.7, 74.7))) * 43758.5453);
    float n011 = fract(sin(dot(i + vec3(0,1,1), vec3(127.1, 311.7, 74.7))) * 43758.5453);
    float n111 = fract(sin(dot(i + vec3(1,1,1), vec3(127.1, 311.7, 74.7))) * 43758.5453);
    return mix(mix(mix(n00, n10, f.x), mix(n01, n11, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

// Hash grid: flat random per cell. ~3 ALU. [Very Light]
float hashGrid(vec3 p) {
    vec3 i = floor(p);
    return fract(sin(dot(i, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}

// Sine lines: parallel lines. ~4 ALU. [Very Light]
float sineLines(vec3 p) {
    return sin(p.x * 6.2831853) * 0.5 + 0.5;
}

// Diamond plate: raised diamond grid. ~8 ALU. [Light]
float diamondPlate(vec3 p) {
    vec2 uv = p.xy;
    vec2 d = abs(fract(uv) - 0.5);
    float diamond = (d.x + d.y); // Manhattan distance = diamond shape
    return smoothstep(0.3, 0.5, diamond);
}

float noiseSample(vec3 p, int type) {
    switch (type) {
        case NOISE_SIMPLEX:     return snoise(p);
        case NOISE_WORLEY_F1:   return worleyF1(p) * 2.0 - 1.0;
        case NOISE_WORLEY_F2F1: return worleyF2F1(p) * 2.0 - 1.0;
        case NOISE_VORONOI:     return voronoiId(p) * 2.0 - 1.0;
        case NOISE_RIDGED:      return 1.0 - abs(snoise(p)) * 2.0;
        case NOISE_TURBULENCE:  return abs(snoise(p)) * 2.0 - 1.0;
        // Geometric patterns: remap [0,1] to [-1,1] for FBM compatibility
        case NOISE_CHECKER:     return checkerboard(p) * 2.0 - 1.0;
        case NOISE_BRICK:       return brick(p) * 2.0 - 1.0;
        case NOISE_HEX:         return hexagonal(p) * 2.0 - 1.0;
        case NOISE_SCRATCHES:   return scratches(p) * 2.0 - 1.0;
        case NOISE_DOTS:        return dots(p) * 2.0 - 1.0;
        case NOISE_GRADIENT:    return gradientNoise(p) * 2.0 - 1.0;
        case NOISE_RINGS:       return rings(p) * 2.0 - 1.0;
        case NOISE_CRACKLE:     return crackle(p) * 2.0 - 1.0;
        case NOISE_WAVES:       return waves(p) * 2.0 - 1.0;
        case NOISE_CELLULAR:    return cellular(p) * 2.0 - 1.0;
        case NOISE_EROSION:     return erosion(p) * 2.0 - 1.0;
        case NOISE_FABRIC:      return fabric(p) * 2.0 - 1.0;
        case NOISE_VALUE:       return valueNoise(p) * 2.0 - 1.0;
        case NOISE_HASH_GRID:   return hashGrid(p) * 2.0 - 1.0;
        case NOISE_SINE_LINES:  return sineLines(p) * 2.0 - 1.0;
        case NOISE_DIAMOND:     return diamondPlate(p) * 2.0 - 1.0;
        default:                return snoise(p);
    }
}

// ── Fractional Brownian Motion (original simplex-only, kept for compatibility) ──

float fbm(vec3 pos, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < octaves; i++) {
        value += amplitude * snoise(pos * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

// ── Typed FBM — dispatches to selected noise type per octave ──

float fbmTyped(vec3 pos, int octaves, int type, float lacunarity) {
    // Binary/geometric patterns: single evaluation — FBM would destroy their structure.
    // Smooth noise-based patterns (scratches, crackle, cellular, erosion) benefit from octaves.
    if (type == NOISE_CHECKER || type == NOISE_BRICK || type == NOISE_HEX ||
        type == NOISE_DOTS || type == NOISE_GRADIENT || type == NOISE_RINGS ||
        type == NOISE_WAVES || type == NOISE_FABRIC ||
        type == NOISE_HASH_GRID || type == NOISE_SINE_LINES || type == NOISE_DIAMOND) {
        return noiseSample(pos, type);
    }

    // Marble and Wood are composite types that use simplex FBM internally
    if (type == NOISE_MARBLE) {
        float f = fbm(pos, octaves);
        return sin(pos.x * 4.0 + f * 6.0); // domain-warped sine bands
    }
    if (type == NOISE_WOOD) {
        float f = fbm(pos, octaves);
        return fract(length(pos.xz) * 2.0 + f * 0.5) * 2.0 - 1.0; // concentric rings
    }

    // Ridged noise: special accumulation with gain feedback
    if (type == NOISE_RIDGED) {
        float value = 0.0;
        float amplitude = 0.5;
        float frequency = 1.0;
        float weight = 1.0;
        for (int i = 0; i < octaves; i++) {
            float n = 1.0 - abs(snoise(pos * frequency));
            n *= n;
            n *= weight;
            weight = clamp(n * 2.0, 0.0, 1.0);
            value += amplitude * n;
            amplitude *= 0.5;
            frequency *= lacunarity;
        }
        return value * 2.0 - 1.0;
    }

    // Turbulence: absolute value accumulation
    if (type == NOISE_TURBULENCE) {
        float value = 0.0;
        float amplitude = 0.5;
        float frequency = 1.0;
        for (int i = 0; i < octaves; i++) {
            value += amplitude * abs(snoise(pos * frequency));
            amplitude *= 0.5;
            frequency *= lacunarity;
        }
        return value * 2.0 - 1.0;
    }

    // Standard FBM for all other types
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float totalAmp = 0.0;
    for (int i = 0; i < octaves; i++) {
        value += amplitude * noiseSample(pos * frequency, type);
        totalAmp += amplitude;
        amplitude *= 0.5;
        frequency *= lacunarity;
    }
    return totalAmp > 0.0 ? value / totalAmp : 0.0; // normalize to [-1, 1], guard octaves=0
}

// ── Typed gradient via central differences ──

vec3 noiseGradientTyped(vec3 v, float eps, int octaves, int type, float lacunarity) {
    return vec3(
        fbmTyped(v + vec3(eps, 0, 0), octaves, type, lacunarity) - fbmTyped(v - vec3(eps, 0, 0), octaves, type, lacunarity),
        fbmTyped(v + vec3(0, eps, 0), octaves, type, lacunarity) - fbmTyped(v - vec3(0, eps, 0), octaves, type, lacunarity),
        fbmTyped(v + vec3(0, 0, eps), octaves, type, lacunarity) - fbmTyped(v - vec3(0, 0, eps), octaves, type, lacunarity)
    ) / (2.0 * eps);
}

// Noise gradient via central differences (for normal perturbation) — original simplex-only
vec3 snoiseGradient(vec3 v, float eps) {
    return vec3(
        snoise(v + vec3(eps, 0, 0)) - snoise(v - vec3(eps, 0, 0)),
        snoise(v + vec3(0, eps, 0)) - snoise(v - vec3(0, eps, 0)),
        snoise(v + vec3(0, 0, eps)) - snoise(v - vec3(0, 0, eps))
    ) / (2.0 * eps);
}

#endif
