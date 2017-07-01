#version 430

in vec3 position;
in float blue;

out float varying_blue;
out vec2 texCoord;

void main()
{
    gl_Position = vec4(position, 1.0);
    texCoord = position.xy / 2.0 + 0.5;
    varying_blue = blue;
}
