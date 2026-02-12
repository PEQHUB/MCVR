#ifndef LABPBR_GLSL
#define LABPBR_GLSL

#define EPS 1e-6

struct LabPBRMat {
    vec3 albedo;
    vec3 f0;
    float roughness;
    float metallic;
    float subSurface;
    float transmission;
    float ior;
    float emission;
    vec3 normal;
    float ao;
    float height;
};

vec3 CalculateF0(vec3 n, vec3 k) {
    vec3 k2 = max(k * k, vec3(1e-6));
    vec3 r = ((n - 1.0) * (n - 1.0) + k2) / ((n + 1.0) * (n + 1.0) + k2);
    return r;
}

LabPBRMat convertLabPBRMaterial(vec4 texAlbedo, vec4 texSpecular, vec4 texNormal) {
    LabPBRMat mat;

    mat.roughness = pow(1.0 - texSpecular.r, 2.0);
    // LabPBR: A value of 255 (100%) results in a very smooth material (e.g. polished granite)
    mat.roughness = mix(0.01, 1.0, mat.roughness);

    float sssOffset = 65.0 / 255.0;
    mat.subSurface = texSpecular.b < sssOffset ? 0.0 : (texSpecular.b - sssOffset) / (1.0 - sssOffset);

    int metalIdx = int(round(texSpecular.g * 255.0));

    mat.metallic = 0.0;
    mat.transmission = 0.0;
    mat.ior = 1.5;
    mat.f0 = vec3(0.04);

    int intEmission = int(round(texSpecular.a * 255.0));
    if (intEmission == 255) {
        mat.emission = 0;
    } else {
        mat.emission = intEmission / 254.0;
    }

    if (metalIdx < 230) {
        mat.metallic = 0.0;
        mat.albedo = texAlbedo.rgb;

        float specularValue = texSpecular.g;
        float F0 = max(specularValue, 0.02); // LabPBR clamp
        mat.f0 = vec3(F0);

        float sqrtF0 = sqrt(F0);
        mat.ior = (1.0 + sqrtF0) / max(1.0 - sqrtF0, EPS);

        if (texAlbedo.a < 1.0 - EPS) { mat.transmission = 1.0; }
    } else if (metalIdx <= 237) {
        vec3 n = vec3(1.0);
        vec3 k = vec3(0.0);

        if (metalIdx == 230) { // Iron
            n = vec3(2.9114, 2.9497, 2.5845);
            k = vec3(3.0893, 2.9318, 2.7670);
        } else if (metalIdx == 231) { // Gold
            n = vec3(0.18299, 0.42108, 1.3734);
            k = vec3(3.4242, 2.3459, 1.7704);
        } else if (metalIdx == 232) { // Aluminium
            n = vec3(1.3456, 0.96521, 0.61722);
            k = vec3(7.4746, 6.3995, 5.3031);
        } else if (metalIdx == 233) { // Chrome
            n = vec3(3.1071, 3.1812, 2.3230);
            k = vec3(3.3314, 3.3291, 3.1350);
        } else if (metalIdx == 234) { // Copper
            n = vec3(0.27105, 0.67693, 1.3164);
            k = vec3(3.6092, 2.6248, 2.2921);
        } else if (metalIdx == 235) { // Lead
            n = vec3(1.9100, 1.8300, 1.4400);
            k = vec3(3.5100, 3.4000, 3.1800);
        } else if (metalIdx == 236) { // Platinum
            n = vec3(2.3757, 2.0847, 1.8453);
            k = vec3(4.2655, 3.7153, 3.1365);
        } else if (metalIdx == 237) { // Silver
            n = vec3(0.15943, 0.14512, 0.13547);
            k = vec3(3.9291, 3.1900, 2.3808);
        }

        mat.metallic = 1.0;
        mat.f0 = CalculateF0(n, k);
        mat.albedo = mat.f0;
    } else {
        mat.metallic = 1.0;
        mat.albedo = texAlbedo.rgb;
        mat.f0 = texAlbedo.rgb;
    }

    mat.normal.xy = texNormal.xy * 2.0 - 1.0;
    mat.normal.z = sqrt(1.0 - dot(mat.normal.xy, mat.normal.xy));

    mat.ao = texNormal.z;
    mat.height = texNormal.w;

    return mat;
}

#endif
