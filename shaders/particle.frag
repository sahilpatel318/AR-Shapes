#version 330 core
in float vIntensity;
out vec4 fragColor;
void main() {
    vec2 uv = gl_PointCoord - vec2(0.5);
    float dist = length(uv);
    float alpha = smoothstep(0.5, 0.0, dist);
    float glow = pow(alpha, 1.4);
    vec3 baseColor = vec3(0.55, 0.85, 1.0);
    fragColor = vec4(baseColor * (0.8 + 0.2 * vIntensity), glow);
}
