#version 450

layout(binding = 0) uniform sampler2D srcImage;
layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

void main() {
    vec4 color = texture(srcImage, fragTexCoord);
    outColor = vec4(color.r, color.g, color.b, color.a);
}