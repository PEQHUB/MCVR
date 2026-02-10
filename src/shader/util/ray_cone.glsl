#ifndef RAY_CONE_GLSL
#define RAY_CONE_GLSL

float fovXFromProj(mat4 P) {
    float pos00 = P[0][0];
    return 2.0 * atan(1.0 / abs(pos00));
}

float fovYFromProj(mat4 P) {
    float pos11 = P[1][1];
    return 2.0 * atan(1.0 / abs(pos11));
}

float coneSpreadFromFov(float fovY, float fovX, vec2 screenSize) {
    float H = screenSize.y;
    float W = screenSize.x;

    float thetaY = atan(tan(fovY * 0.5) / H);
    float thetaX = atan(tan(fovX * 0.5) / W);

    float theta = max(thetaX, thetaY);
    return tan(theta);
}

void computedposduDv(vec3 pos0, vec3 pos1, vec3 pos2, vec2 uv0, vec2 uv1, vec2 uv2, out vec3 dposdu, out vec3 dposdv) {
    vec3 dpos1 = pos1 - pos0;
    vec3 dpos2 = pos2 - pos0;
    vec2 duv1 = uv1 - uv0;
    vec2 duv2 = uv2 - uv0;

    float det = duv1.x * duv2.y - duv1.y * duv2.x;
    if (abs(det) < 1e-12) {
        dposdu = normalize(dpos1);
        dposdv = normalize(dpos2);
        return;
    }

    float invDet = 1.0 / det;
    dposdu = (dpos1 * duv2.y - dpos2 * duv1.y) * invDet;
    dposdv = (-dpos1 * duv2.x + dpos2 * duv1.x) * invDet;
}

float lodWithCone(sampler2D tex, vec2 uv, float coneRadiusWorld, vec3 dposdu, vec3 dposdv) {
    float su = max(length(dposdu), 1e-6);
    float sv = max(length(dposdv), 1e-6);

    float du = coneRadiusWorld / su;
    float dv = coneRadiusWorld / sv;

    ivec2 texDim = textureSize(tex, 0);
    float footprintTexels = max(du * float(texDim.x), dv * float(texDim.y));

    float lod = max(log2(max(footprintTexels, 1e-6)), 0.0);

    return lod;
}

float roughnessToExtraSpread(float roughness) {
    float extraTheta = roughness * 0.35;
    return tan(extraTheta);
}

#endif