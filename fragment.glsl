#version 330

uniform float time;
layout(std140) uniform reds_block
{
    float reds[256];
};
in float varying_blue;
in vec2 texCoord;
out vec4 fragColor;

void main()
{
    int red_idx = int(texCoord.y * 256);
    float red = reds[red_idx];
    float blue = pow(sin(varying_blue * 8 + time * 4), 2);
    fragColor = vec4(red, 0.0, blue, 1.0);
}
