#version 460

layout(set = 3, binding = 0) uniform sampler2D postCopyInput;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(postCopyInput, fragTexCoord);
}
