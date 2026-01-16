#version 330 core
layout(location = 0) in vec3 in_position;
uniform mat4 uModel;
uniform float uPointSize;
uniform float uTime;
out float vIntensity;
void main() {
    vec4 world = uModel * vec4(in_position, 1.0);
    gl_Position = world;
    float pulse = 0.6 + 0.4 * sin(uTime * 1.5 + world.x * 8.0);
    vIntensity = pulse;
    gl_PointSize = uPointSize * pulse;
}
