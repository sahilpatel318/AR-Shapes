from __future__ import annotations

import ctypes
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import glfw
import numpy as np
from OpenGL import GL

from rendering.particles import ParticleSystem
from utils.config import RenderConfig, SharedState, project_path
from utils.fps import FPSCounter
from utils.smoothing import ExponentialSmoother


class ARRenderer:
    """OpenGL renderer responsible for fusing particles with the camera feed."""

    def __init__(self, config: RenderConfig, shared_state: SharedState) -> None:
        self._config = config
        self._state = shared_state
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._stop_event = threading.Event()
        self._window: Optional[glfw._GLFWwindow] = None
        self._particle_program = None
        self._quad_program = None
        self._bg_texture: Optional[int] = None
        self._particle_vbo: Optional[int] = None
        self._particle_vao: Optional[int] = None
        self._quad_vao: Optional[int] = None
        self._particle_system: Optional[ParticleSystem] = None
        self._last_shape: str = "sphere"
        self._anchor_smoother = ExponentialSmoother(self._config.anchor_smooth)
        self._identity = np.identity(4, dtype=np.float32)
        self._bg_size: Optional[Tuple[int, int]] = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()

    def _render_loop(self) -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW. Ensure a valid OpenGL context is available.")
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self._window = glfw.create_window(
            self._config.window_width,
            self._config.window_height,
            "AR Shapes",
            None,
            None,
        )
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Unable to create GLFW window.")
        glfw.make_context_current(self._window)
        glfw.set_key_callback(self._window, self._on_key)
        glfw.swap_interval(1)

        try:
            self._compile_shaders()
            self._setup_quad()
            self._setup_particles()
            fps = FPSCounter()
            prev_time = time.time()

            while not glfw.window_should_close(self._window) and not self._stop_event.is_set():
                frame, anchor, shape, changed_at = self._state.consume()
                if frame is None:
                    time.sleep(0.002)
                    glfw.poll_events()
                    continue
                anchor_ndc = self._smooth_anchor(anchor)
                if shape != self._last_shape:
                    self._particle_system.set_shape(shape, anchor_ndc)
                    self._last_shape = shape
                else:
                    self._particle_system.move_anchor(anchor_ndc)
                self._upload_frame(frame)
                now = time.time()
                delta = max(1e-4, now - prev_time)
                prev_time = now
                self._particle_system.step(delta)
                self._draw_scene(now, anchor_ndc)
                glfw.swap_buffers(self._window)
                glfw.poll_events()
                fps.tick()
        finally:
            glfw.terminate()
            self._state.request_shutdown()

    def _smooth_anchor(self, anchor: Tuple[float, float, float]) -> np.ndarray:
        smoothed = self._anchor_smoother.update(anchor)
        return smoothed

    def _compile_shaders(self) -> None:
        self._particle_program = self._build_program(
            project_path("shaders", "particle.vert"),
            project_path("shaders", "particle.frag"),
        )
        self._quad_program = self._build_program_from_source(_QUAD_VERT, _QUAD_FRAG)

    def _setup_quad(self) -> None:
        quad_vertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        self._quad_vao = GL.glGenVertexArrays(1)
        quad_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._quad_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, quad_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL.GL_STATIC_DRAW)
        stride = 4 * quad_vertices.itemsize
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(8))
        self._bg_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._bg_texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glUseProgram(self._quad_program)
        tex_uniform = GL.glGetUniformLocation(self._quad_program, "uFrame")
        GL.glUniform1i(tex_uniform, 0)

    def _setup_particles(self) -> None:
        self._particle_system = ParticleSystem(self._config)
        self._particle_vao = GL.glGenVertexArrays(1)
        self._particle_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._particle_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._particle_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            self._particle_system.positions.nbytes,
            self._particle_system.positions,
            GL.GL_DYNAMIC_DRAW,
        )
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)

    def _upload_frame(self, frame_rgb: np.ndarray) -> None:
        h, w, _ = frame_rgb.shape
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._bg_texture)
        if self._bg_size != (w, h):
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGB,
                w,
                h,
                0,
                GL.GL_RGB,
                GL.GL_UNSIGNED_BYTE,
                frame_rgb,
            )
            self._bg_size = (w, h)
        else:
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D,
                0,
                0,
                0,
                w,
                h,
                GL.GL_RGB,
                GL.GL_UNSIGNED_BYTE,
                frame_rgb,
            )

    def _draw_scene(self, now: float, anchor: np.ndarray) -> None:
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        # Background quad
        GL.glUseProgram(self._quad_program)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._bg_texture)
        GL.glBindVertexArray(self._quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        # Particles
        GL.glUseProgram(self._particle_program)
        GL.glBindVertexArray(self._particle_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._particle_vbo)
        GL.glBufferSubData(
            GL.GL_ARRAY_BUFFER,
            0,
            self._particle_system.positions.nbytes,
            self._particle_system.positions,
        )
        time_uniform = GL.glGetUniformLocation(self._particle_program, "uTime")
        point_uniform = GL.glGetUniformLocation(self._particle_program, "uPointSize")
        color_uniform = GL.glGetUniformLocation(self._particle_program, "uBaseColor")
        mvp_uniform = GL.glGetUniformLocation(self._particle_program, "uModel")
        GL.glUniform1f(time_uniform, now)
        GL.glUniform1f(point_uniform, self._config.particle_size)
        GL.glUniform3f(color_uniform, 0.75, 0.9, 1.0)
        GL.glUniformMatrix4fv(mvp_uniform, 1, GL.GL_FALSE, self._identity)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        GL.glDrawArrays(GL.GL_POINTS, 0, self._particle_system.count)
        GL.glDisable(GL.GL_BLEND)

    def _build_program(self, vertex_path: Path, fragment_path: Path) -> int:
        with vertex_path.open("r", encoding="utf-8") as vf:
            vert_src = vf.read()
        with fragment_path.open("r", encoding="utf-8") as ff:
            frag_src = ff.read()
        return self._build_program_from_source(vert_src, frag_src)

    def _build_program_from_source(self, vert_src: str, frag_src: str) -> int:
        vertex_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vertex_shader, vert_src)
        GL.glCompileShader(vertex_shader)
        self._assert_shader(vertex_shader)
        fragment_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fragment_shader, frag_src)
        GL.glCompileShader(fragment_shader)
        self._assert_shader(fragment_shader)
        program = GL.glCreateProgram()
        GL.glAttachShader(program, vertex_shader)
        GL.glAttachShader(program, fragment_shader)
        GL.glLinkProgram(program)
        self._assert_program(program)
        GL.glDeleteShader(vertex_shader)
        GL.glDeleteShader(fragment_shader)
        return program

    def _assert_shader(self, shader: int) -> None:
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        if status != GL.GL_TRUE:
            log = GL.glGetShaderInfoLog(shader).decode()
            raise RuntimeError(f"Shader compilation failed: {log}")

    def _assert_program(self, program: int) -> None:
        status = GL.glGetProgramiv(program, GL.GL_LINK_STATUS)
        if status != GL.GL_TRUE:
            log = GL.glGetProgramInfoLog(program).decode()
            raise RuntimeError(f"Program link failed: {log}")

    def _on_key(self, window, key, scancode, action, mods) -> None:  # pragma: no cover - GLFW callback
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self._stop_event.set()
            self._state.request_shutdown()
            glfw.set_window_should_close(window, True)


_QUAD_VERT = """
#version 330 core
layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = vec2(in_uv.x, 1.0 - in_uv.y);
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

_QUAD_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D uFrame;
void main() {
    vec3 color = texture(uFrame, v_uv).rgb;
    fragColor = vec4(color, 1.0);
}
"""
