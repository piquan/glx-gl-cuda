#version 430

#define NREDS 65536

uniform float time;
layout(std140) buffer reds_block
{
    struct {
        float red;
        float pad1, pad2, pad3;
    } reds[NREDS];
};
in float varying_blue;
in vec2 texCoord;
out vec4 fragColor;

void main()
{
    int red_idx = int(texCoord.y * NREDS);
    float red = reds[red_idx].red;
    float blue = pow(sin(texCoord.x * 8 + time * 4), 2) * varying_blue;
    fragColor = vec4(red, 0.0, blue, 1.0);
}
