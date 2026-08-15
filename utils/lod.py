"""
lod.py
======
LoD (Level of Detail) mesh loading, rendering, and analysis utilities.

Supports loading LoD building meshes from .obj and .ply files.

``LoD`` is the primary class: load **.obj / .ply** mesh files, perform
footprint cropping, plane/segment extraction, and depth/mask/segmentation
rendering on CPU or GPU (via an EGL-accelerated rasterizer)."""

from __future__ import annotations

import torch
import subprocess
_lod_polygon_cache = {}
def mesh_to_polygons(*args, **kwargs):
    # fallback
    pass

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import ctypes
import numpy as np
import cv2
import trimesh
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Literal, Union, Any
from collections import defaultdict


class _CameraPose:
    """Minimal internal camera pose helper (camera-to-world convention)."""
    def __init__(self, position, R_c2w):
        self.position = np.asarray(position, dtype=np.float64)
        self.R_c2w = np.asarray(R_c2w, dtype=np.float64)




# ── Standalone LoD mesh render (vendored from LoD-Loc-v2/lodloc2/rendering.py) ──

Device = Literal["cpu", "gpu"]
ImageSize = Union[Tuple[int, int], int]


class _LodFastMeshRenderer:
    """GPU mesh renderer (EGL + OpenGL). Vendored subset: metric depth only."""

    def __init__(self, vertices, faces, resolutions):
        from OpenGL.EGL import (
            eglGetDisplay,
            eglInitialize,
            eglChooseConfig,
            eglCreateContext,
            eglCreatePbufferSurface,
            eglMakeCurrent,
            EGL_DEFAULT_DISPLAY,
            EGL_SURFACE_TYPE,
            EGL_PBUFFER_BIT,
            EGL_RED_SIZE,
            EGL_GREEN_SIZE,
            EGL_BLUE_SIZE,
            EGL_DEPTH_SIZE,
            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_BIT,
            EGL_NONE,
            EGL_WIDTH,
            EGL_HEIGHT,
            EGL_NO_CONTEXT,
            EGL_OPENGL_API,
        )
        from OpenGL.EGL import eglBindAPI
        from OpenGL import GL

        self._GL = GL
        self.n_faces = len(faces)
        self._max_h = max(r[0] for r in resolutions)
        self._max_w = max(r[1] for r in resolutions)

        display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
        major, minor = ctypes.c_int(), ctypes.c_int()
        eglInitialize(display, ctypes.pointer(major), ctypes.pointer(minor))

        config_attribs = [
            EGL_SURFACE_TYPE,
            EGL_PBUFFER_BIT,
            EGL_RED_SIZE,
            8,
            EGL_GREEN_SIZE,
            8,
            EGL_BLUE_SIZE,
            8,
            EGL_DEPTH_SIZE,
            24,
            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_BIT,
            EGL_NONE,
        ]
        config_attribs = (ctypes.c_int * len(config_attribs))(*config_attribs)
        config = (ctypes.c_void_p * 1)()
        num_configs = ctypes.c_int()
        eglChooseConfig(display, config_attribs, config, 1, ctypes.pointer(num_configs))

        eglBindAPI(EGL_OPENGL_API)
        context = eglCreateContext(display, config[0], EGL_NO_CONTEXT, None)

        surface_attribs = (ctypes.c_int * 5)(
            EGL_WIDTH, self._max_w, EGL_HEIGHT, self._max_h, EGL_NONE
        )
        surface = eglCreatePbufferSurface(display, config[0], surface_attribs)
        eglMakeCurrent(display, surface, surface, context)

        # Store EGL handles for cleanup
        self._egl_display = display
        self._egl_context = context
        self._egl_surface = surface

        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

        color_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, color_rb)
        GL.glRenderbufferStorage(
            GL.GL_RENDERBUFFER, GL.GL_R8, self._max_w, self._max_h
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_RENDERBUFFER,
            color_rb,
        )

        depth_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_rb)
        GL.glRenderbufferStorage(
            GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, self._max_w, self._max_h
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_RENDERBUFFER,
            depth_rb,
        )

        assert (
            GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            == GL.GL_FRAMEBUFFER_COMPLETE
        )
        self._fbo = fbo

        vert_src = """
        #version 330 core
        layout(location = 0) in vec3 position;
        uniform mat4 MVP;
        void main() {
            gl_Position = MVP * vec4(position, 1.0);
        }
        """
        frag_src = """
        #version 330 core
        out float fragColor;
        void main() {
            fragColor = 1.0;
        }
        """
        vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vs, vert_src)
        GL.glCompileShader(vs)
        assert GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS)

        fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fs, frag_src)
        GL.glCompileShader(fs)
        assert GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS)

        self._program = GL.glCreateProgram()
        GL.glAttachShader(self._program, vs)
        GL.glAttachShader(self._program, fs)
        GL.glLinkProgram(self._program)
        assert GL.glGetProgramiv(self._program, GL.GL_LINK_STATUS)
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)

        self._mvp_loc = GL.glGetUniformLocation(self._program, "MVP")

        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)

        self._vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        vert_data = vertices.astype(np.float32)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, vert_data.nbytes, vert_data, GL.GL_STATIC_DRAW
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        self._ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        idx_data = faces.astype(np.uint32)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER, idx_data.nbytes, idx_data, GL.GL_STATIC_DRAW
        )

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self._program)
        self._cur_size = None

        depth_vert_src = """
        #version 330 core
        layout(location = 0) in vec3 position;
        uniform mat4 MVP;
        uniform mat4 view_mat;
        out float vDepth;
        void main() {
            gl_Position = MVP * vec4(position, 1.0);
            vec4 cam = view_mat * vec4(position, 1.0);
            vDepth = -cam.z;
        }
        """
        depth_frag_src = """
        #version 330 core
        in float vDepth;
        out float fragDepth;
        void main() {
            fragDepth = vDepth;
        }
        """
        dvs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(dvs, depth_vert_src)
        GL.glCompileShader(dvs)
        assert GL.glGetShaderiv(dvs, GL.GL_COMPILE_STATUS)

        dfs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(dfs, depth_frag_src)
        GL.glCompileShader(dfs)
        assert GL.glGetShaderiv(dfs, GL.GL_COMPILE_STATUS)

        self._depth_program = GL.glCreateProgram()
        GL.glAttachShader(self._depth_program, dvs)
        GL.glAttachShader(self._depth_program, dfs)
        GL.glLinkProgram(self._depth_program)
        assert GL.glGetProgramiv(self._depth_program, GL.GL_LINK_STATUS)
        GL.glDeleteShader(dvs)
        GL.glDeleteShader(dfs)

        self._depth_mvp_loc = GL.glGetUniformLocation(self._depth_program, "MVP")
        self._depth_view_loc = GL.glGetUniformLocation(self._depth_program, "view_mat")

        self._depth_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._depth_fbo)

        depth_color_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_color_rb)
        GL.glRenderbufferStorage(
            GL.GL_RENDERBUFFER, GL.GL_R32F, self._max_w, self._max_h
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_RENDERBUFFER,
            depth_color_rb,
        )

        depth_depth_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_depth_rb)
        GL.glRenderbufferStorage(
            GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, self._max_w, self._max_h
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_RENDERBUFFER,
            depth_depth_rb,
        )

        assert (
            GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            == GL.GL_FRAMEBUFFER_COMPLETE
        )

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glUseProgram(self._program)

        # ── normals renderer: per-triangle face normals in camera space ─────
        # Matches CPU :func:`_lod_raster_normals_cpu_zbuf` (cross of edges in
        # camera space). Uses a geometry shader so each fragment gets the same
        # flat normal without duplicating vertices.
        normals_vert_src = """
        #version 330 core
        layout(location = 0) in vec3 position;
        uniform mat4 MVP;
        out vec3 vPosWorld;
        void main() {
            vPosWorld = position;
            gl_Position = MVP * vec4(position, 1.0);
        }
        """
        normals_geom_src = """
        #version 330 core
        layout(triangles) in;
        layout(triangle_strip, max_vertices = 3) out;

        uniform mat4 w2c_raw;

        in vec3 vPosWorld[];

        flat out vec3 fragNormalCam;

        void main() {
            vec3 p0 = (w2c_raw * vec4(vPosWorld[0], 1.0)).xyz;
            vec3 p1 = (w2c_raw * vec4(vPosWorld[1], 1.0)).xyz;
            vec3 p2 = (w2c_raw * vec4(vPosWorld[2], 1.0)).xyz;
            vec3 e1 = p1 - p0;
            vec3 e2 = p2 - p0;
            vec3 n = cross(e1, e2);
            float ln = length(n);
            if (ln < 1e-8) {
                n = vec3(0.0, 0.0, 0.0);
            } else {
                n = n / ln;
            }

            fragNormalCam = n;
            gl_Position = gl_in[0].gl_Position;
            EmitVertex();
            fragNormalCam = n;
            gl_Position = gl_in[1].gl_Position;
            EmitVertex();
            fragNormalCam = n;
            gl_Position = gl_in[2].gl_Position;
            EmitVertex();
            EndPrimitive();
        }
        """
        normals_frag_src = """
        #version 330 core
        flat in vec3 fragNormalCam;
        out vec3 fragNormal;
        void main() {
            fragNormal = fragNormalCam;
        }
        """

        # Normal program
        vns = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vns, normals_vert_src)
        GL.glCompileShader(vns)
        assert GL.glGetShaderiv(vns, GL.GL_COMPILE_STATUS)

        gns = GL.glCreateShader(GL.GL_GEOMETRY_SHADER)
        GL.glShaderSource(gns, normals_geom_src)
        GL.glCompileShader(gns)
        assert GL.glGetShaderiv(gns, GL.GL_COMPILE_STATUS)

        fns = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fns, normals_frag_src)
        GL.glCompileShader(fns)
        assert GL.glGetShaderiv(fns, GL.GL_COMPILE_STATUS)

        self._normals_program = GL.glCreateProgram()
        GL.glAttachShader(self._normals_program, vns)
        GL.glAttachShader(self._normals_program, gns)
        GL.glAttachShader(self._normals_program, fns)
        GL.glLinkProgram(self._normals_program)
        assert GL.glGetProgramiv(self._normals_program, GL.GL_LINK_STATUS)
        GL.glDeleteShader(vns)
        GL.glDeleteShader(gns)
        GL.glDeleteShader(fns)

        self._normals_mvp_loc = GL.glGetUniformLocation(
            self._normals_program, "MVP"
        )
        self._normals_w2c_loc = GL.glGetUniformLocation(
            self._normals_program, "w2c_raw"
        )

        # Normal FBO with float RGB color
        self._normals_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._normals_fbo)

        normals_color_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, normals_color_rb)
        GL.glRenderbufferStorage(
            GL.GL_RENDERBUFFER, GL.GL_RGB32F, self._max_w, self._max_h
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_RENDERBUFFER,
            normals_color_rb,
        )

        normals_depth_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, normals_depth_rb)
        GL.glRenderbufferStorage(
            GL.GL_RENDERBUFFER,
            GL.GL_DEPTH_COMPONENT24,
            self._max_w,
            self._max_h,
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_RENDERBUFFER,
            normals_depth_rb,
        )

        assert (
            GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            == GL.GL_FRAMEBUFFER_COMPLETE
        )

        # Restore default FBO
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glUseProgram(self._program)

    def destroy(self):
        """Release EGL/OpenGL GPU resources to prevent VRAM leaks."""
        try:
            from OpenGL.EGL import (
                eglDestroySurface, eglDestroyContext, eglTerminate,
                eglMakeCurrent, EGL_NO_SURFACE, EGL_NO_CONTEXT,
            )
            d = self._egl_display
            eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)
            eglDestroySurface(d, self._egl_surface)
            eglDestroyContext(d, self._egl_context)
            eglTerminate(d)
        except Exception:
            pass
        self._egl_display = None
        self._egl_context = None
        self._egl_surface = None

    def __del__(self):
        self.destroy()


    # ── Face-ID renderer (lazy init) ────────────────────────────────────

    def _ensure_face_id_program(self):
        """Lazily create face-ID shader program and integer FBO."""
        if hasattr(self, '_face_id_program'):
            return
        GL = self._GL

        vert_src = """
        #version 330 core
        layout(location = 0) in vec3 position;
        uniform mat4 MVP;
        void main() {
            gl_Position = MVP * vec4(position, 1.0);
        }
        """
        frag_src = """
        #version 330 core
        out int fragFaceId;
        void main() {
            fragFaceId = gl_PrimitiveID;
        }
        """

        vs = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vs, vert_src)
        GL.glCompileShader(vs)
        assert GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS)

        fs = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fs, frag_src)
        GL.glCompileShader(fs)
        assert GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS)

        self._face_id_program = GL.glCreateProgram()
        GL.glAttachShader(self._face_id_program, vs)
        GL.glAttachShader(self._face_id_program, fs)
        GL.glLinkProgram(self._face_id_program)
        assert GL.glGetProgramiv(self._face_id_program, GL.GL_LINK_STATUS)
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)

        self._face_id_mvp_loc = GL.glGetUniformLocation(
            self._face_id_program, "MVP"
        )

        # Integer FBO (GL_R32I color + depth)
        self._face_id_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._face_id_fbo)

        fid_color_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, fid_color_rb)
        GL.glRenderbufferStorage(
            GL.GL_RENDERBUFFER, GL.GL_R32I, self._max_w, self._max_h
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_RENDERBUFFER,
            fid_color_rb,
        )

        fid_depth_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, fid_depth_rb)
        GL.glRenderbufferStorage(
            GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24,
            self._max_w, self._max_h,
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_RENDERBUFFER,
            fid_depth_rb,
        )

        assert (
            GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            == GL.GL_FRAMEBUFFER_COMPLETE
        )

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glUseProgram(self._program)

    def render_face_id_map(self, w2c, K, image_size):
        """Render per-pixel face IDs.  Returns (H, W) int32, -1 for background."""
        self._ensure_face_id_program()
        GL = self._GL
        H, W = image_size

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._face_id_fbo)
        GL.glUseProgram(self._face_id_program)
        GL.glBindVertexArray(self._vao)

        GL.glViewport(0, 0, W, H)
        GL.glScissor(0, 0, W, H)
        GL.glEnable(GL.GL_SCISSOR_TEST)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        near, far = 0.5, 10000.0

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 2.0 * fx / W
        proj[1, 1] = 2.0 * fy / H
        proj[0, 2] = 1.0 - 2.0 * cx / W
        proj[1, 2] = 2.0 * cy / H - 1.0
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2.0 * far * near / (far - near)
        proj[3, 2] = -1.0

        view = w2c.copy().astype(np.float32)
        view[1, :] = -view[1, :]
        view[2, :] = -view[2, :]

        mvp = (proj @ view).T

        GL.glClearBufferiv(GL.GL_COLOR, 0, np.array([-1], dtype=np.int32))
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glUniformMatrix4fv(self._face_id_mvp_loc, 1, GL.GL_FALSE, mvp)
        GL.glDrawElements(
            GL.GL_TRIANGLES, self.n_faces * 3, GL.GL_UNSIGNED_INT, None
        )

        data = GL.glReadPixels(0, 0, W, H, GL.GL_RED_INTEGER, GL.GL_INT)
        face_ids = np.frombuffer(data, dtype=np.int32).reshape((H, W))
        face_ids = np.flipud(face_ids).copy()

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glUseProgram(self._program)
        self._cur_size = None

        return face_ids

    def render_depth_map(self, w2c, K, image_size):
        GL = self._GL
        H, W = image_size

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._depth_fbo)
        GL.glUseProgram(self._depth_program)
        GL.glBindVertexArray(self._vao)

        GL.glViewport(0, 0, W, H)
        GL.glScissor(0, 0, W, H)
        GL.glEnable(GL.GL_SCISSOR_TEST)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        near, far = 0.5, 10000.0

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 2.0 * fx / W
        proj[1, 1] = 2.0 * fy / H
        proj[0, 2] = 1.0 - 2.0 * cx / W
        proj[1, 2] = 2.0 * cy / H - 1.0
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2.0 * far * near / (far - near)
        proj[3, 2] = -1.0

        view = w2c.copy().astype(np.float32)
        view[1, :] = -view[1, :]
        view[2, :] = -view[2, :]

        mvp = (proj @ view).T

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glUniformMatrix4fv(self._depth_mvp_loc, 1, GL.GL_FALSE, mvp)
        GL.glUniformMatrix4fv(self._depth_view_loc, 1, GL.GL_FALSE, view.T.copy())
        GL.glDrawElements(
            GL.GL_TRIANGLES, self.n_faces * 3, GL.GL_UNSIGNED_INT, None
        )

        data = GL.glReadPixels(0, 0, W, H, GL.GL_RED, GL.GL_FLOAT)
        buf = np.frombuffer(data, dtype=np.float32).reshape((H, W))
        depth = np.flipud(buf).copy()

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glUseProgram(self._program)
        self._cur_size = None

        return depth

    def render_normals_map(self, w2c, K, image_size):
        GL = self._GL
        H, W = image_size

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._normals_fbo)
        GL.glUseProgram(self._normals_program)
        GL.glBindVertexArray(self._vao)

        GL.glViewport(0, 0, W, H)
        GL.glScissor(0, 0, W, H)
        GL.glEnable(GL.GL_SCISSOR_TEST)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        near, far = 0.5, 10000.0

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 2.0 * fx / W
        proj[1, 1] = 2.0 * fy / H
        proj[0, 2] = 1.0 - 2.0 * cx / W
        proj[1, 2] = 2.0 * cy / H - 1.0
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2.0 * far * near / (far - near)
        proj[3, 2] = -1.0

        view = w2c.copy().astype(np.float32)
        view[1, :] = -view[1, :]
        view[2, :] = -view[2, :]

        mvp = (proj @ view).T

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glUniformMatrix4fv(
            self._normals_mvp_loc, 1, GL.GL_FALSE, mvp
        )
        # Raw w2c (no Y/Z row flip): same camera space as CPU normals rasterizer.
        w2c_f = w2c.astype(np.float32)
        GL.glUniformMatrix4fv(
            self._normals_w2c_loc, 1, GL.GL_FALSE, w2c_f.T.copy()
        )

        GL.glDrawElements(
            GL.GL_TRIANGLES, self.n_faces * 3, GL.GL_UNSIGNED_INT, None
        )

        data = GL.glReadPixels(0, 0, W, H, GL.GL_RGB, GL.GL_FLOAT)
        buf = np.frombuffer(data, dtype=np.float32).reshape((H, W, 3))
        normals = np.flipud(buf).copy()

        # Restore default FBO
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glUseProgram(self._program)
        self._cur_size = None

        return normals


class LoD:
    """Triangle mesh: .obj/.ply, MovingDrone .npz, cropping, planes, depth/mask render."""

    @staticmethod
    def _lod_compute_building_ids(faces: np.ndarray, num_vertices: int) -> np.ndarray:
        """Per-face building IDs: faces sharing an edge belong to the same building.

        Vectorised face-adjacency connected components using scipy sparse CC."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components as sparse_cc

        faces = np.asarray(faces, dtype=np.int64)
        F = len(faces)
        if F == 0:
            return np.zeros(0, dtype=np.int32)

        # All 3 edges per face (sorted vertex pairs), vectorised
        # edges_raw: (F*3, 2)
        edges_raw = np.stack([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ], axis=1).reshape(-1, 2)
        edges_sorted = np.sort(edges_raw, axis=1)

        # Unique edge key for grouping
        edge_keys = edges_sorted[:, 0] * (num_vertices + 1) + edges_sorted[:, 1]

        # Face index for each edge entry
        face_of_edge = np.repeat(np.arange(F, dtype=np.int64), 3)

        # Sort by edge key to find shared edges
        order = np.argsort(edge_keys)
        sorted_keys = edge_keys[order]
        sorted_faces = face_of_edge[order]

        # Consecutive entries with same key → faces share that edge
        same = sorted_keys[:-1] == sorted_keys[1:]
        fa = sorted_faces[:-1][same]
        fb = sorted_faces[1:][same]

        if len(fa) == 0:
            return np.arange(F, dtype=np.int32)

        # Build sparse face adjacency and run CC
        data = np.ones(len(fa) * 2, dtype=np.int8)
        row = np.concatenate([fa, fb])
        col = np.concatenate([fb, fa])
        adj = csr_matrix((data, (row, col)), shape=(F, F))
        _, labels = sparse_cc(adj, directed=False)
        return labels.astype(np.int32)

    @staticmethod
    def _lod_merge_building_parts(
        faces: np.ndarray,
        vertices: np.ndarray,
        face_component_ids: np.ndarray,
        labels: np.ndarray,
        dist_thresh: float = 1.0,
        small_absorb_faces: int = 20,
        small_absorb_dist: float = 5.0,
    ) -> np.ndarray:
        """Merge connected components whose vertices are within *dist_thresh*.

        CityGML meshes have separate geometry groups for walls, roofs, etc.
        so edge-connectivity alone splits each building into parts.  This
        post-processing step merges components that share co-located vertices
        (wall tops ↔ roof edges) into a single building ID.

        Two passes:
          1. **Vertex proximity** — merge components with any vertex pair < *dist_thresh*
             (uses scipy cKDTree + sparse CC on the component graph).
          2. **Small-component absorption** — merge tiny components (< *small_absorb_faces*)
             into the nearest larger building within *small_absorb_dist* (centroid distance).

        Fully vectorised — no Python loops over individual components.
        Only components labelled wall/roof/ground/closure/building are merged;
        "other" components are left untouched."""
        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components as sparse_cc

        _BUILDING_LABELS = {"wall", "roof", "ground", "closure", "building"}
        n_comps = int(face_component_ids.max()) + 1
        F = len(face_component_ids)
        if n_comps < 2:
            return face_component_ids.copy()

        # --- Vectorised: identify building components ---
        face_is_building = np.array(
            [str(l) in _BUILDING_LABELS for l in labels], dtype=bool
        )
        bldg_count = np.bincount(face_component_ids, weights=face_is_building.astype(np.float64), minlength=n_comps)
        is_building_comp = bldg_count > 0
        building_cids = np.where(is_building_comp)[0]
        if len(building_cids) < 2:
            return face_component_ids.copy()

        # --- Vectorised: collect (vertex_index, component_id) pairs ---
        face_idx_rep = np.repeat(np.arange(F, dtype=np.int64), 3)
        vert_idx_all = faces.ravel()
        comp_idx_all = face_component_ids[face_idx_rep]

        bldg_mask = is_building_comp[comp_idx_all]
        bldg_vert_idx = vert_idx_all[bldg_mask]
        bldg_comp_idx = comp_idx_all[bldg_mask]

        # Unique (vert, comp) pairs
        vc_pairs = np.stack([bldg_vert_idx, bldg_comp_idx], axis=1)
        vc_unique = np.unique(vc_pairs, axis=0)
        all_vis = vc_unique[:, 0]
        all_cids = vc_unique[:, 1].astype(np.int32)
        all_coords = vertices[all_vis]

        # --- Pass 1: vertex proximity merge via KDTree + scipy CC ---
        tree = cKDTree(all_coords)
        kd_pairs = tree.query_pairs(dist_thresh, output_type='ndarray')

        # Build component-level adjacency graph from vertex-proximity pairs
        comp_edge_a = []
        comp_edge_b = []
        if len(kd_pairs) > 0:
            cid_a = all_cids[kd_pairs[:, 0]]
            cid_b = all_cids[kd_pairs[:, 1]]
            diff_mask = cid_a != cid_b
            comp_edge_a.append(cid_a[diff_mask])
            comp_edge_b.append(cid_b[diff_mask])

        # Sparse CC on component graph (covers Pass 1)
        if comp_edge_a:
            ea = np.concatenate(comp_edge_a)
            eb = np.concatenate(comp_edge_b)
            data = np.ones(len(ea) * 2, dtype=np.int8)
            row = np.concatenate([ea, eb]).astype(np.int32)
            col = np.concatenate([eb, ea]).astype(np.int32)
            comp_adj = csr_matrix((data, (row, col)), shape=(n_comps, n_comps))
            _, pass1_labels = sparse_cc(comp_adj, directed=False)
        else:
            pass1_labels = np.arange(n_comps, dtype=np.int32)

        # --- Pass 2: absorb small fragments into nearest large building ---
        # Remap to dense group IDs after pass 1
        unique_groups, group_inv = np.unique(pass1_labels, return_inverse=True)
        n_groups = len(unique_groups)

        # Face count per group
        face_count_per_comp = np.bincount(face_component_ids, minlength=n_comps)
        face_count_per_group = np.bincount(group_inv, weights=face_count_per_comp.astype(np.float64), minlength=n_groups)

        # Group centroids (weighted by vertex count per component)
        entry_group_1 = group_inv[all_cids]  # group index for each vert-comp entry
        centroid_sum = np.zeros((n_groups, 3), dtype=np.float64)
        centroid_cnt = np.zeros(n_groups, dtype=np.int64)
        np.add.at(centroid_sum, entry_group_1, all_coords.astype(np.float64))
        np.add.at(centroid_cnt, entry_group_1, 1)
        group_centroids = centroid_sum / np.maximum(centroid_cnt[:, None], 1)

        # Identify building groups (some groups may mix building + non-building)
        is_building_group = np.zeros(n_groups, dtype=bool)
        for gi in range(n_groups):
            comp_mask = group_inv == gi
            if is_building_comp[comp_mask].any():
                is_building_group[gi] = True

        # Small/large classification among building groups
        bldg_groups = np.where(is_building_group)[0]
        small_mask = face_count_per_group[bldg_groups] < small_absorb_faces
        large_mask = ~small_mask

        # Use union-find only for the small absorption pass (few groups)
        parent = np.arange(n_groups, dtype=np.int32)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        if small_mask.any() and large_mask.any():
            large_gi = bldg_groups[large_mask]
            small_gi = bldg_groups[small_mask]
            large_centroids = group_centroids[large_gi]
            small_centroids = group_centroids[small_gi]
            large_tree = cKDTree(large_centroids)
            dists, idxs = large_tree.query(small_centroids)
            for i in range(len(small_gi)):
                if dists[i] <= small_absorb_dist:
                    union(int(small_gi[i]), int(large_gi[idxs[i]]))

        # --- Vectorised relabelling ---
        final_roots = np.array([find(g) for g in range(n_groups)], dtype=np.int32)
        unique_final, inv_final = np.unique(final_roots, return_inverse=True)
        # face → comp → pass1 group → pass2 root → final ID
        result = inv_final[group_inv[face_component_ids]].astype(np.int32)
        return result



    @staticmethod
    def _lod_as_hw(image_size: ImageSize) -> Tuple[int, int]:
        if isinstance(image_size, int):
            return image_size, image_size
        h, w = image_size
        return int(h), int(w)

    @staticmethod
    def _lod_clip_triangle_near_plane(v0, v1, v2, near_plane: float):
        verts = [v0, v1, v2]
        inside = [v[2] >= near_plane for v in verts]
        n_in = sum(inside)

        if n_in == 3:
            return [(v0, v1, v2)]
        if n_in == 0:
            return []

        def _lerp_z(a, b):
            dz = b[2] - a[2]
            if abs(dz) < 1e-10:
                return a.copy()
            t = (near_plane - a[2]) / dz
            return a + t * (b - a)

        if n_in == 1:
            idx = next(i for i, v in enumerate(inside) if v)
            a = verts[idx]
            b = verts[(idx + 1) % 3]
            c = verts[(idx + 2) % 3]
            return [(a, _lerp_z(a, b), _lerp_z(a, c))]
        idx_out = next(i for i, v in enumerate(inside) if not v)
        a = verts[idx_out]
        b = verts[(idx_out + 1) % 3]
        c = verts[(idx_out + 2) % 3]
        ab = _lerp_z(a, b)
        ac = _lerp_z(a, c)
        return [(b, c, ac), (b, ac, ab)]


    @staticmethod
    def _lod_load_mesh_vertices_faces(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        path = Path(path)
        if path.suffix.lower() not in {".obj", ".ply"}:
            raise ValueError(f"Unsupported mesh format: {path.suffix} (use .obj or .ply)")
        loaded = trimesh.load(str(path), force="mesh")
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in loaded.geometry.values()
                ]
            )
        else:
            mesh = loaded
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        return verts, faces

    @staticmethod
    def _lod_collect_screen_triangles_with_z(
        vertices: np.ndarray,
        faces: np.ndarray,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: Tuple[int, int],
        near_plane: float,
    ):
        H, W = image_size
        SCALE = 16

        ones = np.ones((len(vertices), 1), dtype=np.float64)
        pts_h = np.hstack([vertices, ones])
        pts_cam = (w2c @ pts_h.T).T[:, :3]

        v0_cam = pts_cam[faces[:, 0]]
        v1_cam = pts_cam[faces[:, 1]]
        v2_cam = pts_cam[faces[:, 2]]

        z0, z1, z2 = v0_cam[:, 2], v1_cam[:, 2], v2_cam[:, 2]
        n_in = (
            (z0 >= near_plane).astype(np.int8)
            + (z1 >= near_plane).astype(np.int8)
            + (z2 >= near_plane).astype(np.int8)
        )
        fully_in = n_in == 3
        partial = (n_in == 1) | (n_in == 2)

        def project_cam(pts):
            proj = (K @ pts.T).T
            z = np.where(np.abs(proj[:, 2]) > 1e-8, proj[:, 2], 1e-8)
            u = proj[:, 0] / z * SCALE
            v = proj[:, 1] / z * SCALE
            return np.stack([u, v], axis=1)

        out = []
        fi_idx = np.where(fully_in)[0]
        if len(fi_idx) > 0:
            p0 = project_cam(v0_cam[fi_idx])
            p1 = project_cam(v1_cam[fi_idx])
            p2 = project_cam(v2_cam[fi_idx])
            tris = np.stack([p0, p1, p2], axis=1).astype(np.int32)
            ztri = np.stack([z0[fi_idx], z1[fi_idx], z2[fi_idx]], axis=1)
            for i in range(len(tris)):
                out.append((tris[i], ztri[i]))

        pa_idx = np.where(partial)[0]
        if len(pa_idx) > 0:
            for i in pa_idx:
                for tri in LoD._lod_clip_triangle_near_plane(
                    v0_cam[i], v1_cam[i], v2_cam[i], near_plane
                ):
                    a, b, c = tri
                    p = project_cam(np.stack([a, b, c]))
                    ztri = np.array([a[2], b[2], c[2]], dtype=np.float64)
                    out.append((p.astype(np.int32), ztri))

        if not out:
            return []

        tris_arr = np.stack([t[0] for t in out], axis=0)
        W_s, H_s = W * SCALE, H * SCALE
        in_bounds = ~(
            (tris_arr[:, :, 0].max(axis=1) < 0)
            | (tris_arr[:, :, 0].min(axis=1) >= W_s)
            | (tris_arr[:, :, 1].max(axis=1) < 0)
            | (tris_arr[:, :, 1].min(axis=1) >= H_s)
        )
        return [out[i] for i in np.where(in_bounds)[0]]

    @staticmethod
    def _lod_raster_depth_cpu_zbuf(
        vertices: np.ndarray,
        faces: np.ndarray,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: Tuple[int, int],
        near_plane: float = 0.5,
    ) -> np.ndarray:
        H, W = image_size
        SCALE = 16
        tris = LoD._lod_collect_screen_triangles_with_z(
            vertices, faces, w2c, K, image_size, near_plane
        )
        zbuf = np.full((H, W), np.inf, dtype=np.float32)

        for tri_sp, z_cam in tris:
            ua, va = tri_sp[0, 0] / SCALE, tri_sp[0, 1] / SCALE
            ub, vb = tri_sp[1, 0] / SCALE, tri_sp[1, 1] / SCALE
            uc, vc = tri_sp[2, 0] / SCALE, tri_sp[2, 1] / SCALE
            za, zb, zc = float(z_cam[0]), float(z_cam[1]), float(z_cam[2])

            umin = int(np.clip(np.floor(min(ua, ub, uc)), 0, W - 1))
            umax = int(np.clip(np.ceil(max(ua, ub, uc)), 0, W - 1))
            vmin = int(np.clip(np.floor(min(va, vb, vc)), 0, H - 1))
            vmax = int(np.clip(np.ceil(max(va, vb, vc)), 0, H - 1))
            if umin > umax or vmin > vmax:
                continue

            uu = np.arange(umin, umax + 1, dtype=np.float64)
            vv = np.arange(vmin, vmax + 1, dtype=np.float64)
            gu, gv = np.meshgrid(uu, vv, indexing="xy")
            p = np.stack([gu.ravel(), gv.ravel()], axis=1)

            v0 = np.array([ub - ua, vb - va])
            v1 = np.array([uc - ua, vc - va])
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            v2 = p - np.array([ua, va])
            d20 = (v2 * v0).sum(axis=1)
            d21 = (v2 * v1).sum(axis=1)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-20:
                continue
            v_b = (d11 * d20 - d01 * d21) / denom
            w_b = (d00 * d21 - d01 * d20) / denom
            u_b = 1.0 - v_b - w_b
            inside = (u_b >= -1e-8) & (v_b >= -1e-8) & (w_b >= -1e-8)
            if not np.any(inside):
                continue
            z_pix = u_b * za + v_b * zb + w_b * zc
            ri = np.floor(gv.ravel()[inside] + 0.5).astype(np.int64)
            ci = np.floor(gu.ravel()[inside] + 0.5).astype(np.int64)
            zi = z_pix[inside].astype(np.float32)
            for r, c, z in zip(ri, ci, zi):
                if z < near_plane:
                    continue
                if z < zbuf[r, c]:
                    zbuf[r, c] = z

        out = np.zeros((H, W), dtype=np.float32)
        valid = np.isfinite(zbuf) & (zbuf < np.inf)
        out[valid] = zbuf[valid]
        return out

    @staticmethod
    def _lod_collect_screen_triangles_with_z_and_normals(
        vertices: np.ndarray,
        faces: np.ndarray,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: Tuple[int, int],
        near_plane: float,
    ):
        """Like :func:`_lod_collect_screen_triangles_with_z`, but also returns a
        camera-space unit normal for each (possibly clipped) triangle piece.

        Returns:
          list of (tri_sp_int32, ztri_cam, normal_cam_unit)
            tri_sp_int32: (3,2) integer screen coords (subpixel-quantized by SCALE)
            ztri_cam: (3,) depths in camera coordinates for each vertex
            normal_cam_unit: (3,) unit vector in camera coordinates"""
        H, W = image_size
        SCALE = 16

        ones = np.ones((len(vertices), 1), dtype=np.float64)
        pts_h = np.hstack([vertices, ones])
        pts_cam = (w2c @ pts_h.T).T[:, :3]

        v0_cam = pts_cam[faces[:, 0]]
        v1_cam = pts_cam[faces[:, 1]]
        v2_cam = pts_cam[faces[:, 2]]

        z0, z1, z2 = v0_cam[:, 2], v1_cam[:, 2], v2_cam[:, 2]
        n_in = (
            (z0 >= near_plane).astype(np.int8)
            + (z1 >= near_plane).astype(np.int8)
            + (z2 >= near_plane).astype(np.int8)
        )
        fully_in = n_in == 3
        partial = (n_in == 1) | (n_in == 2)

        def project_cam(pts):
            proj = (K @ pts.T).T
            z = np.where(np.abs(proj[:, 2]) > 1e-8, proj[:, 2], 1e-8)
            u = proj[:, 0] / z * SCALE
            v = proj[:, 1] / z * SCALE
            return np.stack([u, v], axis=1)

        out = []
        fi_idx = np.where(fully_in)[0]
        if len(fi_idx) > 0:
            p0 = project_cam(v0_cam[fi_idx])
            p1 = project_cam(v1_cam[fi_idx])
            p2 = project_cam(v2_cam[fi_idx])
            tris = np.stack([p0, p1, p2], axis=1).astype(np.int32)
            ztri = np.stack([z0[fi_idx], z1[fi_idx], z2[fi_idx]], axis=1)

            n_raw = np.cross(
                (v1_cam[fi_idx] - v0_cam[fi_idx]),
                (v2_cam[fi_idx] - v0_cam[fi_idx]),
            )
            n_norm = np.linalg.norm(n_raw, axis=1, keepdims=True)
            n_unit = n_raw / np.where(n_norm > 1e-12, n_norm, 1.0)

            for i in range(len(tris)):
                out.append((tris[i], ztri[i], n_unit[i].astype(np.float64)))

        pa_idx = np.where(partial)[0]
        if len(pa_idx) > 0:
            for i in pa_idx:
                for tri in LoD._lod_clip_triangle_near_plane(
                    v0_cam[i], v1_cam[i], v2_cam[i], near_plane
                ):
                    a, b, c = tri
                    p = project_cam(np.stack([a, b, c]))
                    ztri = np.array([a[2], b[2], c[2]], dtype=np.float64)
                    n_raw = np.cross(b - a, c - a)
                    n_norm = float(np.linalg.norm(n_raw))
                    n_unit = n_raw / n_norm if n_norm > 1e-12 else np.zeros(3)
                    out.append((p.astype(np.int32), ztri, n_unit.astype(np.float64)))

        if not out:
            return []

        tris_arr = np.stack([t[0] for t in out], axis=0)
        W_s, H_s = W * SCALE, H * SCALE
        in_bounds = ~(
            (tris_arr[:, :, 0].max(axis=1) < 0)
            | (tris_arr[:, :, 0].min(axis=1) >= W_s)
            | (tris_arr[:, :, 1].max(axis=1) < 0)
            | (tris_arr[:, :, 1].min(axis=1) >= H_s)
        )
        return [out[i] for i in np.where(in_bounds)[0]]

    @staticmethod
    def _lod_raster_normals_cpu_zbuf(
        vertices: np.ndarray,
        faces: np.ndarray,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: Tuple[int, int],
        near_plane: float = 0.5,
    ) -> np.ndarray:
        """CPU z-buffer rasterization of per-pixel camera normals.

        For the closest triangle covering each pixel, we store the triangle normal
        in camera coordinates."""
        H, W = image_size
        zbuf = np.full((H, W), np.inf, dtype=np.float32)
        normals = np.zeros((H, W, 3), dtype=np.float32)

        tris = LoD._lod_collect_screen_triangles_with_z_and_normals(
            vertices, faces, w2c, K, image_size, near_plane
        )

        for tri_sp, z_cam, n_cam in tris:
            ua, va = tri_sp[0, 0] / 16, tri_sp[0, 1] / 16
            ub, vb = tri_sp[1, 0] / 16, tri_sp[1, 1] / 16
            uc, vc = tri_sp[2, 0] / 16, tri_sp[2, 1] / 16
            za, zb, zc = float(z_cam[0]), float(z_cam[1]), float(z_cam[2])

            umin = int(np.clip(np.floor(min(ua, ub, uc)), 0, W - 1))
            umax = int(np.clip(np.ceil(max(ua, ub, uc)), 0, W - 1))
            vmin = int(np.clip(np.floor(min(va, vb, vc)), 0, H - 1))
            vmax = int(np.clip(np.ceil(max(va, vb, vc)), 0, H - 1))
            if umin > umax or vmin > vmax:
                continue

            uu = np.arange(umin, umax + 1, dtype=np.float64)
            vv = np.arange(vmin, vmax + 1, dtype=np.float64)
            gu, gv = np.meshgrid(uu, vv, indexing="xy")
            p = np.stack([gu.ravel(), gv.ravel()], axis=1)

            v0 = np.array([ub - ua, vb - va])
            v1 = np.array([uc - ua, vc - va])
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            v2 = p - np.array([ua, va])
            d20 = (v2 * v0).sum(axis=1)
            d21 = (v2 * v1).sum(axis=1)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-20:
                continue
            v_b = (d11 * d20 - d01 * d21) / denom
            w_b = (d00 * d21 - d01 * d20) / denom
            u_b = 1.0 - v_b - w_b
            inside = (u_b >= -1e-8) & (v_b >= -1e-8) & (w_b >= -1e-8)
            if not np.any(inside):
                continue

            z_pix = u_b * za + v_b * zb + w_b * zc
            ri = np.floor(gv.ravel()[inside] + 0.5).astype(np.int64)
            ci = np.floor(gu.ravel()[inside] + 0.5).astype(np.int64)
            zi = z_pix[inside].astype(np.float32)
            for r, c, z in zip(ri, ci, zi):
                if z < near_plane:
                    continue
                if z < zbuf[r, c]:
                    zbuf[r, c] = z
                    normals[r, c] = np.asarray(n_cam, dtype=np.float32)

        valid = np.isfinite(zbuf) & (zbuf < np.inf)
        normals[~valid] = 0.0
        return normals

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path).resolve()
        self.vertices, self.faces = LoD._lod_load_mesh_vertices_faces(self.path)
        # Centre vertices around their mean so GPU float32 stays precise.
        # Store the mean as utm_offset so vertices_abs still returns world coords.
        centroid = self.vertices.mean(axis=0)
        self.vertices = self.vertices - centroid
        self.utm_offset = centroid.copy()
        self.labels = np.full((len(self.faces),), "building", dtype=object)
        self.polygon_ids: Optional[np.ndarray] = None
        self._building_ids: Optional[np.ndarray] = None
        self._centroids_abs: Optional[np.ndarray] = None
        self._glr: Optional[_LodFastMeshRenderer] = None
        self._glr_hw: Optional[Tuple[int, int]] = None


    @property
    def vertices_abs(self) -> np.ndarray:
        """World coordinates (local storage + UTM offset)."""
        return self.vertices + self.utm_offset

    def face_normals(self) -> np.ndarray:
        """Per-face unit normals in world coordinates. Shape (F, 3)."""
        v = self.vertices_abs
        v0, v1, v2 = v[self.faces[:, 0]], v[self.faces[:, 1]], v[self.faces[:, 2]]
        raw = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        return raw / norms

    def face_centroids(self) -> np.ndarray:
        """Per-face centroids in world coordinates. Shape (F, 3)."""
        if self._centroids_abs is None:
            v = self.vertices_abs
            self._centroids_abs = (
                v[self.faces[:, 0]] + v[self.faces[:, 1]] + v[self.faces[:, 2]]
            ) / 3.0
        return self._centroids_abs

    def get_building_ids(self) -> np.ndarray:
        """Per-face building IDs (edge-CC + vertex-proximity merge). Cached."""
        if self._building_ids is None:
            raw = LoD._lod_compute_building_ids(self.faces, self.num_vertices)
            self._building_ids = LoD._lod_merge_building_parts(
                self.faces, self.vertices_abs, raw, self.labels,
            )
        return self._building_ids

    @classmethod
    def from_npz(cls, npz_path: Union[str, Path]) -> "LoD":
        """Load MovingDrone / MovingDrone ``lod*.npz`` (vertices, faces, labels, utm_offset)."""
        return cls.from_file(npz_path)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "LoD":
        """Load mesh from .npz, .obj, or .ply."""
        path = Path(path)
        ext = path.suffix.lower()
        if ext == ".npz":
            data = np.load(path, allow_pickle=True)
            inst = object.__new__(cls)
            inst.path = path.resolve()
            inst.vertices = data["vertices"].astype(np.float64)
            inst.faces = data["faces"].astype(np.int64)
            raw_lab = data.get("labels")
            if raw_lab is None or len(raw_lab) == 0:
                inst.labels = np.full((len(inst.faces),), "building", dtype=object)
            else:
                inst.labels = np.asarray(raw_lab)
            inst.utm_offset = (
                data["utm_offset"].astype(np.float64)
                if "utm_offset" in data
                else np.zeros(3, dtype=np.float64)
            )
            inst.polygon_ids = None
            inst._building_ids = None
            inst._centroids_abs = None
            inst._glr = None
            inst._glr_hw = None
            return inst
        elif ext in [".obj", ".ply"]:
            mesh = trimesh.load(str(path), process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            labels = np.full((len(faces),), "building", dtype=object)
            utm_offset = np.zeros(3, dtype=np.float64)
            return cls.from_arrays(
                vertices=vertices,
                faces=faces,
                labels=labels,
                utm_offset=utm_offset,
                source_path=path
            )
        else:
            raise ValueError(f"Unsupported LoD file extension: {ext}")


    @classmethod
    def from_arrays(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        labels: Optional[np.ndarray] = None,
        utm_offset: Optional[np.ndarray] = None,
        polygon_ids: Optional[np.ndarray] = None,
        source_path: Optional[Union[str, Path]] = None,
    ) -> "LoD":
        """Build LoD from arrays (vertices, faces, labels, utm_offset)."""
        inst = object.__new__(cls)
        inst.path = Path(source_path).resolve() if source_path else Path()
        inst.vertices = np.asarray(vertices, dtype=np.float64)
        inst.faces = np.asarray(faces, dtype=np.int64)
        inst.utm_offset = (
            np.asarray(utm_offset, dtype=np.float64)
            if utm_offset is not None
            else np.zeros(3, dtype=np.float64)
        )
        if labels is not None and len(labels) > 0:
            inst.labels = np.asarray(labels)
        elif len(inst.faces) > 0:
            inst.labels = np.full((len(inst.faces),), "building", dtype=object)
        else:
            inst.labels = np.array([], dtype=object)
        inst.polygon_ids = (
            np.asarray(polygon_ids) if polygon_ids is not None else None
        )
        inst._building_ids = None
        inst._centroids_abs = None
        inst._glr = None
        inst._glr_hw = None

        return inst

    def save(self, path: Union[str, Path]) -> None:
        """Save mesh as ``.obj`` or ``.ply`` (from suffix)."""
        path = Path(path)
        suf = path.suffix.lower()
        if suf not in {".obj", ".ply"}:
            raise ValueError(f"Unsupported extension {suf}; use .obj or .ply")
        if len(self.faces) == 0:
            raise ValueError("Cannot save empty mesh")
        mesh = trimesh.Trimesh(
            vertices=np.asarray(self.vertices, dtype=np.float64),
            faces=np.asarray(self.faces, dtype=np.int64),
        )
        mesh.export(str(path))

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

    def _gpu_renderer(self, H: int, W: int) -> _LodFastMeshRenderer:
        if self._glr is not None and self._glr_hw == (H, W):
            return self._glr
        # Use LOCAL vertices (without UTM offset) to preserve float32
        # precision on the GPU.  Callers must use _w2c_local() to shift
        # the view matrix accordingly.
        verts = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.uint32)
        self._glr = _LodFastMeshRenderer(verts, faces, [(H, W)])
        self._glr_hw = (H, W)
        return self._glr

    def _w2c_local(self, w2c: np.ndarray) -> np.ndarray:
        """Adjust world-to-camera matrix for local (non-offset) GPU vertices.

        The GPU vertex buffer stores ``self.vertices`` (local coords) to avoid
        float32 precision loss at large UTM offsets.  This shifts the w2c
        translation so that  R @ v_local + t_local  ==  R @ v_world + t_world."""
        w2c_l = np.array(w2c, dtype=np.float64)
        w2c_l[:3, 3] = w2c[:3, :3] @ self.utm_offset + w2c[:3, 3]
        return w2c_l


    def render_normals(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: ImageSize,
        *,
        device: Device = "gpu",
        near_plane: float = 0.5,
        cpu_max_side: Optional[int] = 1536,
    ) -> np.ndarray:
        """Render per-pixel unit normals in camera coordinates.

        Returns:
          normals: (H, W, 3) float32. Uncovered pixels are zero."""
        H, W = LoD._lod_as_hw(image_size)
        w2c = np.asarray(w2c, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)

        v_world = self.vertices_abs

        if device == "gpu":
            glr = self._gpu_renderer(H, W)
            n = glr.render_normals_map(self._w2c_local(w2c), K, (H, W)).astype(np.float32)
            return n

        scale = 1.0
        if cpu_max_side is not None:
            m = max(H, W)
            if m > cpu_max_side:
                scale = cpu_max_side / m
        if scale < 1.0:
            Hs, Ws = int(round(H * scale)), int(round(W * scale))
            Ks = K.copy()
            Ks[0, :] *= Ws / W
            Ks[1, :] *= Hs / H
            n_small = LoD._lod_raster_normals_cpu_zbuf(
                v_world, self.faces, w2c, Ks, (Hs, Ws), near_plane
            )
            return cv2.resize(n_small, (W, H), interpolation=cv2.INTER_LINEAR)

        return LoD._lod_raster_normals_cpu_zbuf(
            v_world, self.faces, w2c, K, (H, W), near_plane
        )

    def render_depth(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: ImageSize,
        *,
        device: Device = "gpu",
        near_plane: float = 0.5,
        cpu_max_side: Optional[int] = 1536,
    ) -> np.ndarray:
        H, W = LoD._lod_as_hw(image_size)
        w2c = np.asarray(w2c, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        v_world = self.vertices_abs

        if device == "gpu":
            glr = self._gpu_renderer(H, W)
            return glr.render_depth_map(self._w2c_local(w2c), K, (H, W)).astype(np.float32)

        scale = 1.0
        if cpu_max_side is not None:
            m = max(H, W)
            if m > cpu_max_side:
                scale = cpu_max_side / m
        if scale < 1.0:
            Hs, Ws = int(round(H * scale)), int(round(W * scale))
            Ks = K.copy()
            Ks[0, :] *= Ws / W
            Ks[1, :] *= Hs / H
            d_small = LoD._lod_raster_depth_cpu_zbuf(
                v_world, self.faces, w2c, Ks, (Hs, Ws), near_plane
            )
            return cv2.resize(d_small, (W, H), interpolation=cv2.INTER_LINEAR).astype(
                np.float32
            )

        return LoD._lod_raster_depth_cpu_zbuf(
            v_world, self.faces, w2c, K, (H, W), near_plane
        )


    def render_shaded(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: ImageSize,
        *,
        device: Device = "gpu",
        near_plane: float = 0.5,
        light_dir_cam: Optional[np.ndarray] = None,
        ambient: float = 0.20,
        base_color: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render Phong-shaded RGB image of the mesh.

        Uses per-pixel normals from the existing normals renderer plus
        a simple directional light model (ambient + diffuse + specular).

        Args:
            light_dir_cam: (3,) unit light direction in camera coords, pointing
                *towards* the surface.  Default: slightly above-camera sun.
            ambient: ambient light intensity [0, 1].
            base_color: (3,) RGB base color [0-255].  Default warm concrete gray.

        Returns:
            shaded: (H, W, 3) uint8 RGB image.  Background is black."""
        normals = self.render_normals(
            w2c, K, image_size, device=device, near_plane=near_plane,
        )  # (H, W, 3) float32, camera-space normals; zero where no mesh

        if light_dir_cam is None:
            # Sun-like: from upper-left-front
            l = np.array([0.3, -0.6, -0.7], dtype=np.float32)
            light_dir_cam = l / np.linalg.norm(l)

        if base_color is None:
            base_color = np.array([200, 195, 185], dtype=np.float32)  # warm concrete
        else:
            base_color = np.asarray(base_color, dtype=np.float32)

        # Identify mesh pixels (normals are zero on background)
        mask = np.linalg.norm(normals, axis=2) > 0.3
        n = normals.copy()
        n_len = np.linalg.norm(n, axis=2, keepdims=True)
        n_len = np.where(n_len > 1e-6, n_len, 1.0)
        n = n / n_len

        # Diffuse: max(-n · l, 0)  (light_dir points towards surface)
        ndotl = np.clip(-np.sum(n * light_dir_cam, axis=2), 0.0, 1.0)
        diffuse = 0.65 * ndotl

        # Specular (Blinn-Phong): view dir is (0,0,-1) in camera space
        view = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        half_vec = -light_dir_cam + (-view)
        half_vec = half_vec / (np.linalg.norm(half_vec) + 1e-8)
        ndoth = np.clip(np.sum(n * half_vec, axis=2), 0.0, 1.0)
        specular = 0.15 * (ndoth ** 16)

        shade = ambient + diffuse + specular  # (H, W)
        shade = np.clip(shade, 0.0, 1.0)

        img = np.zeros((*normals.shape[:2], 3), dtype=np.float32)
        for c in range(3):
            img[:, :, c] = shade * base_color[c]
        img[~mask] = 0
        return np.clip(img, 0, 255).astype(np.uint8)


    # ────────── Semantic & instance segmentation ──────────

    # Unified label IDs: 0=background, 1=roof, 2=wall, 3=other
    SEMANTIC_BG = 0
    SEMANTIC_ROOF = 1
    SEMANTIC_WALL = 2
    SEMANTIC_OTHER = 3

    _SEMANTIC_LABEL_COLORS = {
        SEMANTIC_BG:    np.array([0, 0, 0], dtype=np.uint8),
        SEMANTIC_ROOF:  np.array([220, 20, 60], dtype=np.uint8),     # crimson
        SEMANTIC_WALL:  np.array([70, 130, 180], dtype=np.uint8),    # steel blue
        SEMANTIC_OTHER: np.array([128, 128, 128], dtype=np.uint8),   # gray
    }

    # Mapping from LoD mesh string labels → unified IDs
    _MESH_LABEL_TO_SEMANTIC = {
        'wall': SEMANTIC_WALL,
        'roof': SEMANTIC_ROOF,
        'ground': SEMANTIC_OTHER,
        'closure': SEMANTIC_OTHER,
        'other': SEMANTIC_OTHER,
        'building': SEMANTIC_OTHER,  # generic catch-all
    }

    def render_segmentation(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: "ImageSize",
        *,
        mode: str = "semantic",
        face_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render GT building segmentation from LoD mesh.

        For ``mode='semantic'`` delegates to :meth:`render_semantic` with
        ``label_source='mesh'``.

        Args:
            w2c: 4x4 world-to-camera matrix
            K: 3x3 intrinsics
            image_size: (H, W) or int for square
            mode: ``'semantic'``  — per-pixel class (roof/wall/other via mesh labels)
                  ``'instance'``  — per-pixel building ID (connected component)
            face_ids: optional pre-rendered face ID map; if None, will render

        Returns:
            color_map: (H, W, 3) uint8 — colored segmentation
            id_map:    (H, W) int32  — class IDs (semantic) or building IDs (instance),
                       -1 for sky/background"""
        H, W = LoD._lod_as_hw(image_size)
        w2c = np.asarray(w2c, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)

        # GPU-render per-pixel face index (or reuse pre-computed)
        if face_ids is None:
            glr = self._gpu_renderer(H, W)
            face_ids = glr.render_face_id_map(self._w2c_local(w2c), K, (H, W))

        if mode == "semantic":
            label_map, color_map = self.render_semantic(
                w2c, K, (H, W), label_source="mesh", face_ids=face_ids,
            )
            id_map = label_map.astype(np.int32)
            id_map[face_ids < 0] = -1
            return color_map, id_map

        elif mode == "instance":
            color_map = np.zeros((H, W, 3), dtype=np.uint8)
            id_map = np.full((H, W), -1, dtype=np.int32)
            building_ids = self.get_building_ids()
            labels = self.labels

            # Exclude non-building faces (e.g. "other" = terrain/ground mesh)
            _BUILDING_LABELS = {'wall', 'roof', 'ground', 'closure', 'building'}
            is_building_face = np.array(
                [str(l) in _BUILDING_LABELS for l in labels], dtype=bool
            ) if len(labels) == len(self.faces) else np.ones(len(self.faces), dtype=bool)

            bg = face_ids < 0
            valid = ~bg
            fids_valid = face_ids[valid]
            fids_clamped = np.clip(fids_valid, 0, len(building_ids) - 1)
            bids_valid = building_ids[fids_clamped]

            # Mask out non-building faces
            is_bldg_px = is_building_face[fids_clamped]
            bids_valid[~is_bldg_px] = -1
            id_map[valid] = bids_valid

            # Generate maximally distinct colors per building ID
            # Uses evenly spaced hues with high saturation/value to avoid
            # visually similar neighbours (e.g. green / light-green).
            unique_bids = np.unique(bids_valid[bids_valid >= 0])
            n_bids = len(unique_bids)
            bid_to_color = {}
            if n_bids > 0:
                import colorsys
                golden_ratio_conjugate = 0.618033988749895
                h = 0.0
                for idx, bid in enumerate(unique_bids):
                    # Alternate saturation/value to push adjacent buildings apart
                    s = 0.7 + 0.3 * (idx % 2)
                    v = 0.95 - 0.15 * (idx % 3)
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    bid_to_color[bid] = np.array(
                        [int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8
                    )
                    h = (h + golden_ratio_conjugate) % 1.0

            for bid, color in bid_to_color.items():
                mask = (id_map == bid)
                color_map[mask] = color
        else:
            raise ValueError(f"mode must be 'semantic' or 'instance', got '{mode}'")

        return color_map, id_map


    def get_plane_segmentation(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: "ImageSize",
        *,
        device: "Device" = "gpu",
        normal_cos_thresh: float = 0.995,
        depth_rel_thresh: float = 0.02,
        min_pixels: int = 50,
        depth: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Render depth + normals, then segment into planes via region-growing.

        Two pixels are grouped if:
          1. Normal cosine similarity >= normal_cos_thresh
          2. Relative depth difference < depth_rel_thresh

        Uses vectorized edge computation + scipy sparse connected components
        for speed (~100× faster than pure-Python BFS).

        Args:
            w2c: 4x4 world-to-camera matrix
            K: 3x3 intrinsics
            image_size: (H, W) or single int for square
            device: "gpu" or "cpu"
            normal_cos_thresh: cosine threshold (0.995 ~ 5.7 deg)
            depth_rel_thresh: max relative depth step between neighbors (0.02 = 2%)
            min_pixels: minimum connected-component size to keep
            depth: optional pre-rendered depth (H, W); if None, will render
            normals: optional pre-rendered normals (H, W, 3); if None, will render

        Returns:
            depth: (H, W) float32 rendered depth
            normals: (H, W, 3) float32 camera-space normals
            label_map: (H, W) int32 plane IDs (-1 = background/discarded)
            segments: list of dicts per plane with keys:
                'plane_id': int
                'normal_cam': (3,) mean unit normal in camera space
                'pixel_count': int
                'pixel_indices': (N, 2) int (row, col)"""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components as sparse_cc

        H, W = LoD._lod_as_hw(image_size)
        w2c = np.asarray(w2c, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)

        if depth is None:
            depth = self.render_depth(w2c, K, (H, W), device=device)
        if normals is None:
            normals = self.render_normals(w2c, K, (H, W), device=device)

        valid = (depth > 0)
        nlen_sq = np.einsum('ijk,ijk->ij', normals, normals)
        valid &= (nlen_sq > 0.25)
        safe_nlen_sq = np.maximum(nlen_sq, 1e-16)
        normals_n = normals * (1.0 / np.sqrt(safe_nlen_sq))[..., None]

        N = H * W
        flat_valid = valid.ravel()

        # Map valid pixels to a compact index space [0, n_valid)
        # so the sparse matrix is n_valid × n_valid instead of N × N.
        valid_indices = np.where(flat_valid)[0]  # flat indices of valid pixels
        n_valid = len(valid_indices)
        # full→compact mapping (-1 for invalid)
        compact_id = np.full(N, -1, dtype=np.int32)
        compact_id[valid_indices] = np.arange(n_valid, dtype=np.int32)

        # ----- Vectorized edge computation (flat indexing, no repeat/tile) ----
        # Flat pixel indices: horizontal pairs (i, i+1) skipping column W-1
        all_px = np.arange(N, dtype=np.int32)
        col_idx = all_px % W
        # Horizontal: every pixel except last column
        h_mask = col_idx < (W - 1)
        h_a = all_px[h_mask]
        h_b = h_a + 1
        # Vertical: every pixel except last row
        v_mask = all_px < (N - W)
        v_a = all_px[v_mask]
        v_b = v_a + W

        ea = np.concatenate([h_a, v_a])
        eb = np.concatenate([h_b, v_b])

        # Both ends must be valid
        both_valid = flat_valid[ea] & flat_valid[eb]
        ea = ea[both_valid]
        eb = eb[both_valid]

        # Normal cosine similarity + depth check (fused)
        normals_flat = normals_n.reshape(N, 3)
        dots = np.einsum('ij,ij->i', normals_flat[ea], normals_flat[eb])
        depth_flat = depth.ravel()
        da = depth_flat[ea]
        db = depth_flat[eb]
        keep = (dots >= normal_cos_thresh) & (np.abs(da - db) < depth_rel_thresh * np.maximum(da, db))
        ea = ea[keep]
        eb = eb[keep]

        # Remap to compact indices and build sparse adjacency
        ea_c = compact_id[ea]
        eb_c = compact_id[eb]
        data = np.ones(len(ea_c) * 2, dtype=np.int8)
        row = np.concatenate([ea_c, eb_c])
        col = np.concatenate([eb_c, ea_c])
        adj = csr_matrix((data, (row, col)), shape=(n_valid, n_valid))
        n_cc, cc_labels_compact = sparse_cc(adj, directed=False)

        # Map compact CC labels back to full image
        cc_labels = np.full(N, -1, dtype=cc_labels_compact.dtype)
        cc_labels[valid_indices] = cc_labels_compact

        # ----- Fast segment collection via bincount + argsort -----
        cc_flat = cc_labels.copy()
        cc_flat[~flat_valid] = -1

        # Count pixels per label and filter by min_pixels
        max_label = int(cc_flat.max()) + 1 if cc_flat.max() >= 0 else 0
        counts = np.bincount(cc_flat[cc_flat >= 0], minlength=max_label)
        big_labels = np.where(counts >= min_pixels)[0]

        if len(big_labels) == 0:
            label_map = np.full((H, W), -1, dtype=np.int32)
            return depth, normals, label_map, []

        # Sort all valid pixels by their CC label for O(N) grouping
        valid_mask = cc_flat >= 0
        valid_idx = np.where(valid_mask)[0]
        valid_labels = cc_flat[valid_idx]
        order = np.argsort(valid_labels, kind='mergesort')
        sorted_labels = valid_labels[order]
        sorted_idx = valid_idx[order]

        # Find boundaries between label groups
        breaks = np.concatenate([[0], np.where(np.diff(sorted_labels) != 0)[0] + 1, [len(sorted_labels)]])
        label_at_break = sorted_labels[breaks[:-1]]

        # Build a set of big labels for fast lookup
        big_set = np.zeros(max_label, dtype=bool)
        big_set[big_labels] = True

        label_map = np.full((H, W), -1, dtype=np.int32)
        normals_flat3 = normals_n.reshape(N, 3)
        segments = []
        plane_id = 0

        for i in range(len(breaks) - 1):
            cc_id = label_at_break[i]
            if not big_set[cc_id]:
                continue
            pix = sorted_idx[breaks[i]:breaks[i + 1]]
            rows = (pix // W).astype(np.int32)
            cols = (pix % W).astype(np.int32)
            avg_n = normals_flat3[pix].mean(axis=0)
            avg_n /= np.linalg.norm(avg_n) + 1e-12
            label_map[rows, cols] = plane_id
            segments.append({
                'plane_id': plane_id,
                'normal_cam': avg_n.astype(np.float32),
                'pixel_count': len(pix),
                'pixel_indices': np.stack([rows, cols], axis=1),
            })
            plane_id += 1

        return depth, normals, label_map, segments

    def create_3d_planes(
        self,
        segments: List[Dict],
        depth: np.ndarray,
        K: np.ndarray,
        w2c: np.ndarray,
        *,
        min_3d_area: float = 1.0,
        min_extent: float = 0.3,
        max_aspect_ratio: float = 25.0,
    ) -> List[Dict]:
        """Convert 2D plane segments into 3D planes with robust filtering.

        Back-projects each segment's pixels to 3D, fits a plane, and filters
        out tiny, thin, or degenerate planes.

        Filtering criteria:
          - min_3d_area: minimum convex-hull-like area in 3D (m^2)
          - min_extent: minimum extent along the two principal axes (m)
          - max_aspect_ratio: max ratio of major to minor extent;
            rejects extreme slivers (e.g., 25 = 25:1 elongation)

        Args:
            segments: list of segment dicts from get_plane_segmentation
            depth: (H, W) float32 depth map
            K: 3x3 intrinsics
            w2c: 4x4 world-to-camera

        Returns:
            planes: list of dicts with keys:
                'plane_id': int (from segment)
                'normal_cam': (3,) unit normal in camera space
                'normal_world': (3,) unit normal in world space
                'centroid_cam': (3,) centroid in camera space
                'centroid_world': (3,) centroid in world space
                'd_world': float, plane offset (n . x = d)
                'area_3d': float, approximate 3D area (m^2)
                'extent_major': float, extent along 1st principal axis (m)
                'extent_minor': float, extent along 2nd principal axis (m)
                'pixel_count': int
                'pixel_indices': (N, 2) int (row, col)"""
        w2c = np.asarray(w2c, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]
        R_c2w = R_w2c.T

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        planes = []
        n_rejected = {'tiny': 0, 'thin_extent': 0, 'sliver': 0}

        for seg in segments:
            rows = seg['pixel_indices'][:, 0]
            cols = seg['pixel_indices'][:, 1]
            z = depth[rows, cols].astype(np.float64)

            # Back-project to camera space
            x = (cols - cx) * z / fx
            y = (rows - cy) * z / fy
            pts_cam = np.stack([x, y, z], axis=1)  # (N, 3)

            # Transform to world space
            pts_world = (R_c2w @ (pts_cam - t_w2c).T).T  # (N, 3)

            # Centroid
            centroid_cam = pts_cam.mean(axis=0)
            centroid_world = pts_world.mean(axis=0)

            # Plane normal in world (area-weighted from segment)
            n_cam = seg['normal_cam'].astype(np.float64)
            n_world = R_c2w @ n_cam
            n_world /= np.linalg.norm(n_world) + 1e-12

            # Plane offset
            d_world = float(np.dot(n_world, centroid_world))

            # --- Robust filtering via SVD of centered points ---
            centered = pts_world - centroid_world
            if len(centered) < 3:
                n_rejected['tiny'] += 1
                continue

            # SVD to get principal extents
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
            # S values are proportional to spread along each axis
            # Scale to approximate half-extents: S[i] / sqrt(N)
            n_pts = len(centered)
            extents = S / np.sqrt(n_pts)  # approximate half-extents

            extent_major = 2.0 * extents[0]  # full extent along 1st axis
            extent_minor = 2.0 * extents[1]  # full extent along 2nd axis
            thickness = 2.0 * extents[2] if len(extents) > 2 else 0.0

            # Approximate 3D area as extent_major * extent_minor (bounding rectangle)
            area_3d = extent_major * extent_minor

            # Filter: minimum area
            if area_3d < min_3d_area:
                n_rejected['tiny'] += 1
                continue

            # Filter: minimum extent (reject very narrow planes)
            if extent_minor < min_extent:
                n_rejected['thin_extent'] += 1
                continue

            # Filter: reject extreme slivers (major/minor aspect ratio)
            if extent_minor > 1e-6 and extent_major / extent_minor > max_aspect_ratio:
                n_rejected['sliver'] += 1
                continue

            planes.append({
                'plane_id': seg['plane_id'],
                'normal_cam': n_cam.astype(np.float32),
                'normal_world': n_world.astype(np.float32),
                'centroid_cam': centroid_cam.astype(np.float32),
                'centroid_world': centroid_world.astype(np.float32),
                'd_world': d_world,
                'area_3d': float(area_3d),
                'extent_major': float(extent_major),
                'extent_minor': float(extent_minor),
                'pixel_count': seg['pixel_count'],
                'pixel_indices': seg['pixel_indices'],
            })

        print(f"  create_3d_planes: {len(planes)} kept, "
              f"rejected {n_rejected['tiny']} tiny, "
              f"{n_rejected['thin_extent']} thin-extent, "
              f"{n_rejected['sliver']} slivers")
        return planes

    # ────────── Semantic rendering (unified) ──────────

    def render_semantic(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: "ImageSize",
        *,
        label_source: str = "normals",
        device: "Device" = "gpu",
        face_ids: Optional[np.ndarray] = None,
        roof_nz_thresh: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render per-pixel semantic label map (roof / wall / other).

        Supports two labelling strategies via *label_source*:

        ``'normals'`` (default)
            Classify from face-normal z-component:
            - roof:  |n_z| > *roof_nz_thresh*
            - wall:  |n_z| <= *roof_nz_thresh*

        ``'mesh'``
            Use per-face LoD string labels (``self.labels``) mapped through
            ``_MESH_LABEL_TO_SEMANTIC`` (wall→WALL, roof→ROOF, rest→OTHER).

        Args:
            w2c: 4×4 world-to-camera matrix.
            K: 3×3 intrinsics.
            image_size: (H, W) or single int for square.
            label_source: ``'normals'`` or ``'mesh'``.
            device: ``'gpu'`` or ``'cpu'``.
            face_ids: Pre-computed face-id map (H, W) int32.  Rendered if None.
            roof_nz_thresh: |n_z| threshold (only used when *label_source='normals'*).

        Returns:
            label_map: (H, W) uint8 — 0=bg, 1=roof, 2=wall, 3=other.
            color_map: (H, W, 3) uint8 — per-pixel colour from ``_SEMANTIC_LABEL_COLORS``."""
        H, W = LoD._lod_as_hw(image_size)
        w2c = np.asarray(w2c, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)

        if face_ids is None:
            glr = self._gpu_renderer(H, W)
            face_ids = glr.render_face_id_map(self._w2c_local(w2c), K, (H, W))

        valid = face_ids >= 0
        n_faces = len(self.faces)
        fid_safe = np.clip(face_ids, 0, n_faces - 1)

        label_map = np.zeros((H, W), dtype=np.uint8)

        if label_source == "normals":
            fn = self.face_normals()  # (F, 3) world-space
            nz = np.abs(fn[fid_safe, 2])
            label_map[valid & (nz > roof_nz_thresh)] = self.SEMANTIC_ROOF
            label_map[valid & (nz <= roof_nz_thresh)] = self.SEMANTIC_WALL

        elif label_source == "mesh":
            # Build per-face semantic ID array from string labels
            face_sem = np.full(n_faces, self.SEMANTIC_OTHER, dtype=np.uint8)
            for lbl, sid in self._MESH_LABEL_TO_SEMANTIC.items():
                face_sem[self.labels == lbl] = sid
            label_map[valid] = face_sem[fid_safe[valid]]

        else:
            raise ValueError(f"label_source must be 'normals' or 'mesh', got '{label_source}'")

        # Colorize
        color_map = np.zeros((H, W, 3), dtype=np.uint8)
        for sid, color in self._SEMANTIC_LABEL_COLORS.items():
            mask = label_map == sid
            if mask.any():
                color_map[mask] = color

        return label_map, color_map

    # ────────── Edge map & wireframe rendering ──────────

    def render_edges(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: "ImageSize",
        *,
        device: "Device" = "gpu",
        face_ids: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        normal_cos_thresh: float = 0.999,
        depth_rel_thresh: float = 0.05,
    ) -> np.ndarray:
        """Render structural edge map from dense LoD renders.

        Detects structural edges (adjacent faces with different normals),
        silhouette edges (building boundary vs sky), and depth discontinuities.
        Filters internal triangulation edges (coplanar adjacent faces).

        Args:
            face_ids: Pre-computed face-id map (H, W) int32.  Computed if None.
            depth: Pre-computed depth map (H, W) float32.  Computed if None.
            normal_cos_thresh: Cosine threshold above which two adjacent faces
                are treated as coplanar (=> internal triangulation edge, skipped).
            depth_rel_thresh: Relative depth difference above which a depth
                discontinuity edge is emitted.

        Returns:
            edge_map: (H, W) float32 in {0, 1}.  1 = edge pixel."""
        H, W = LoD._lod_as_hw(image_size)
        w2c = np.asarray(w2c, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)

        if face_ids is None:
            glr = self._gpu_renderer(H, W)
            face_ids = glr.render_face_id_map(self._w2c_local(w2c), K, (H, W))
        if depth is None:
            depth = self.render_depth(w2c, K, (H, W), device=device)

        fn = self.face_normals()
        n_faces = len(fn)

        fid = face_ids.astype(np.int32)

        # Compute shifted face-id arrays once
        fid_l = fid[:, :-1]   # left  of horizontal pair
        fid_r = fid[:, 1:]    # right of horizontal pair
        fid_t = fid[:-1, :]   # top   of vertical pair
        fid_b = fid[1:, :]    # bottom of vertical pair

        edge = np.zeros((H, W), dtype=np.uint8)

        # ── Silhouette: face vs background ──
        sil_h = (fid_l >= 0) != (fid_r >= 0)
        sil_v = (fid_t >= 0) != (fid_b >= 0)

        # ── Structural: different non-coplanar faces ──
        diff_h = (fid_l != fid_r) & (fid_l >= 0) & (fid_r >= 0)
        diff_v = (fid_t != fid_b) & (fid_t >= 0) & (fid_b >= 0)

        # Check non-coplanarity only where faces differ
        if diff_h.any():
            fa = fid_l[diff_h].clip(0, n_faces - 1)
            fb = fid_r[diff_h].clip(0, n_faces - 1)
            cos = np.abs(np.einsum('ij,ij->i', fn[fa], fn[fb]))
            struct_h = np.zeros_like(diff_h)
            struct_h[diff_h] = cos <= normal_cos_thresh
        else:
            struct_h = diff_h

        if diff_v.any():
            fa = fid_t[diff_v].clip(0, n_faces - 1)
            fb = fid_b[diff_v].clip(0, n_faces - 1)
            cos = np.abs(np.einsum('ij,ij->i', fn[fa], fn[fb]))
            struct_v = np.zeros_like(diff_v)
            struct_v[diff_v] = cos <= normal_cos_thresh
        else:
            struct_v = diff_v

        # Combine silhouette + structural into edge map
        edge[:, :-1] |= (sil_h | struct_h).view(np.uint8)
        edge[:-1, :] |= (sil_v | struct_v).view(np.uint8)

        # ── Depth discontinuity ──
        inv_d = np.float32(1.0) / np.maximum(depth, np.float32(1.0))
        edge[:, :-1] |= (np.abs(depth[:, :-1] - depth[:, 1:]) * inv_d[:, :-1] > depth_rel_thresh).view(np.uint8)
        edge[:-1, :] |= (np.abs(depth[:-1, :] - depth[1:, :]) * inv_d[:-1, :] > depth_rel_thresh).view(np.uint8)

        return edge.astype(np.float32)

    def render_wireframe(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: "ImageSize",
        *,
        interval: float = 2.0,
        thickness: int = 1,
        color: Tuple[int, int, int] = (0, 255, 0),
        depth: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample points along unique mesh edges and project as a wireframe.

        Like LoD-Loc v1 but faster: deduplicates edges by vertex-index pairs
        (cheap int sort) instead of by sampled 3D positions (expensive float
        sort), then samples only unique edges.

        Frustum-culls edges before sampling to avoid wasting time on
        the ~95% of edges that are not visible.

        Args:
            interval: Sampling distance in metres along each edge.
            thickness: Pixel radius for drawn wireframe points.
            color: RGB colour of wireframe dots on the returned overlay.
            depth: optional pre-rendered depth (H, W); if None, will render.

        Returns:
            wireframe_img: (H, W, 3) uint8 image — coloured dots on black bg.
            wire_info: dict with ``'pts_3d'`` (N,3), ``'normals'`` (N,3),
                ``'uv'`` (N,2), ``'depth'`` (N,), ``'visible_mask'`` (N,)."""
        H, W = LoD._lod_as_hw(image_size)
        w2c = np.asarray(w2c, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        verts = self.vertices_abs       # (V, 3)
        faces = self.faces              # (F, 3)
        fn = self.face_normals()        # (F, 3)

        # 1. Extract ALL face edges and deduplicate by sorted vertex pairs.
        e0 = faces[:, [0, 1]]          # (F, 2)
        e1 = faces[:, [1, 2]]
        e2 = faces[:, [2, 0]]
        all_edges = np.concatenate([e0, e1, e2], axis=0)  # (3F, 2)
        face_idx_all = np.repeat(np.arange(len(faces)), 3)

        # Sort each edge so (a,b) with a<=b, then unique
        sorted_edges = np.sort(all_edges, axis=1).astype(np.int64)
        max_v = int(sorted_edges.max()) + 1
        edge_keys = sorted_edges[:, 0] * max_v + sorted_edges[:, 1]
        _, uniq_idx = np.unique(edge_keys, return_index=True)
        edges = all_edges[uniq_idx]     # (E, 2)  unique undirected edges
        face_idx = face_idx_all[uniq_idx]

        # 2. Frustum-cull: project ALL unique vertices to camera, then keep
        #    only edges where at least one endpoint is roughly in-frame.
        ones_v = np.ones((len(verts), 1), dtype=verts.dtype)
        v_cam = (w2c[:3] @ np.hstack([verts, ones_v]).T).T.astype(np.float32)  # (V, 3)
        v_z = v_cam[:, 2]
        inv_vz = np.where(v_z > 1e-6, 1.0 / v_z, 0.0).astype(np.float32)
        v_u = K[0, 0] * v_cam[:, 0] * inv_vz + K[0, 2]
        v_v = K[1, 1] * v_cam[:, 1] * inv_vz + K[1, 2]
        # Large margin so edges crossing the frustum boundary are kept
        margin = max(H, W) * 0.3
        v_in = (v_z > 0) & (v_u > -margin) & (v_u < W + margin) & (v_v > -margin) & (v_v < H + margin)

        # Keep edge if at least one endpoint is visible
        e_visible = v_in[edges[:, 0]] | v_in[edges[:, 1]]
        edges = edges[e_visible]
        face_idx = face_idx[e_visible]

        # 3. Edge endpoints and normals (only for survived edges)
        p0 = verts[edges[:, 0]]         # (E, 3) float64→float32 below
        p1 = verts[edges[:, 1]]
        n0 = fn[face_idx]
        n1 = fn[face_idx]

        # Filter zero-length edges
        diff = (p1 - p0).astype(np.float32)
        lengths = np.linalg.norm(diff, axis=1)
        valid = lengths > 0
        p0, p1, n0, n1, lengths = (
            p0[valid].astype(np.float32), p1[valid].astype(np.float32),
            n0[valid], n1[valid], lengths[valid],
        )

        # 4. Vectorised sampling along all edges
        n_samples = np.maximum(np.ceil(lengths / interval).astype(np.int32) + 1, 2)
        total_pts = int(n_samples.sum())
        offsets = np.zeros(len(n_samples) + 1, dtype=np.int64)
        np.cumsum(n_samples, out=offsets[1:])

        edge_idx = np.repeat(np.arange(len(n_samples), dtype=np.int32), n_samples)
        local_idx = np.arange(total_pts, dtype=np.float32) - offsets[edge_idx].astype(np.float32)
        denom = (n_samples[edge_idx] - 1).clip(min=1).astype(np.float32)
        t = (local_idx / denom)[:, None]  # (N, 1)

        pts_3d = p0[edge_idx] + t * (p1[edge_idx] - p0[edge_idx])
        normals_3d = n0[edge_idx] + t * (n1[edge_idx] - n0[edge_idx])

        # 5. Project 3D → 2D  (w2c is float64 for accuracy, downcast result)
        ones = np.ones((len(pts_3d), 1), dtype=np.float32)
        cam = (w2c[:3] @ np.hstack([pts_3d, ones]).T).astype(np.float32)  # (3, N)
        z = cam[2]
        uv = np.empty((len(pts_3d), 2), dtype=np.float32)
        inv_z = np.float32(1.0) / np.maximum(z, np.float32(1e-12))
        uv[:, 0] = K[0, 0] * cam[0] * inv_z + K[0, 2]
        uv[:, 1] = K[1, 1] * cam[1] * inv_z + K[1, 2]
        in_frame = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

        # 6. Depth-based occlusion culling
        if depth is None:
            depth_map = self.render_depth(w2c, K, (H, W), device="gpu")
        else:
            depth_map = depth
        uv_safe = np.clip(np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0), -1, max(W, H))
        px = np.clip(uv_safe[:, 0].astype(np.int32), 0, W - 1)
        py = np.clip(uv_safe[:, 1].astype(np.int32), 0, H - 1)
        rendered_z = depth_map[py, px]
        # Tolerance accounts for depth-map interpolation artefacts at building edges.
        depth_tol = np.float32(2.0)
        vis = in_frame & (rendered_z > 0) & (z <= rendered_z + depth_tol)

        # 7. Draw wireframe
        wireframe_img = np.zeros((H, W, 3), dtype=np.uint8)
        uv_vis = uv[vis].astype(np.int32)
        if len(uv_vis) > 0:
            if thickness <= 1:
                wireframe_img[uv_vis[:, 1], uv_vis[:, 0]] = color
            else:
                for i in range(len(uv_vis)):
                    cv2.circle(wireframe_img, (uv_vis[i, 0], uv_vis[i, 1]),
                               thickness, color, -1)

        return wireframe_img, {
            'pts_3d': pts_3d,
            'normals': normals_3d,
            'uv': uv,
            'depth': z,
            'visible_mask': vis,
        }



if __name__ == "__main__":
    """Run the LoD visualisation demo on a single .obj mesh + query image.

    Usage:
        python lod.py                                   # use bundled sample data
        python lod.py --lod path/to/mesh.obj \
                      --image_path path/to/image.jpg \
                      --pose "qw qx qy qz tx ty tz" \
                      --intrinsics "fx fy cx cy W H"

    Resolution options (mutually exclusive: --scale OR --width/--height):
        python lod.py --scale 0.5          # render at 50% of native resolution
        python lod.py --width 1280         # fix width; height auto from aspect ratio
        python lod.py --height 720         # fix height; width auto from aspect ratio
        python lod.py --width 1280 --height 720   # explicit both dimensions

    Coordinate conventions:
        --pose: W2C (world-to-camera) transform encoded as
                    qw qx qy qz  (scalar-first unit quaternion for the rotation)
                    tx ty tz     (translation)
        --intrinsics: pinhole camera model in pixels
                    fx fy cx cy W H

    Output is saved to ./outputs/lod_demo/overlay_<name>.png.
    """
    import argparse
    import sys

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.spatial.transform import Rotation

    HERE = Path(__file__).resolve().parent
    SAMPLE_DIR = HERE / "sample"
    OUT = HERE / "outputs"
    OUT.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="LoD visualisation: 3x3 overlay grid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lod", type=str, default=str(SAMPLE_DIR / "scene.obj"),
                        help="Path to LoD mesh file (.obj or .ply). Default: sample/scene.obj")
    parser.add_argument("--image_path", type=str, default=str(SAMPLE_DIR / "query.jpg"),
                        help="Path to query image. Default: sample/query.jpg")
    parser.add_argument("--pose", type=str,
                        default="0.5483200139832795 0.8317844604598373 -0.07307994901175187 0.04625034762413314 -207.7587975240633 -19.13229526124387 274.68994520214846",
                        help='W2C pose as "qw qx qy qz tx ty tz"')
    parser.add_argument("--intrinsics", type=str,
                        default="1373.972 1370.394 968.591 698.301 1920 1439",
                        help='Camera intrinsics as "fx fy cx cy W H"')
    # Resolution
    parser.add_argument("--scale", type=float, default=None,
                        help="Render scale relative to native resolution (default 0.5)")
    parser.add_argument("--width", type=int, default=None,
                        help="Render width in pixels (height auto if omitted)")
    parser.add_argument("--height", type=int, default=None,
                        help="Render height in pixels (width auto if omitted)")
    args = parser.parse_args()

    # ── Load LoD ──
    lod_path = Path(args.lod)
    if not lod_path.exists():
        print(f"LoD file not found: {lod_path}"); sys.exit(1)
    if lod_path.suffix.lower() == ".npz":
        lod = LoD.from_npz(str(lod_path))
    else:
        lod = LoD(str(lod_path))

    # ── Parse pose: "qw qx qy qz tx ty tz" ──
    pvals = list(map(float, args.pose.split()))
    if len(pvals) != 7:
        print("--pose must be 7 values: qw qx qy qz tx ty tz"); sys.exit(1)
    qw, qx, qy, qz, tx, ty, tz = pvals
    R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = [tx, ty, tz]

    # ── Parse intrinsics: "fx fy cx cy W H" ──
    ivals = list(map(float, args.intrinsics.split()))
    if len(ivals) != 6:
        print("--intrinsics must be 6 values: fx fy cx cy W H"); sys.exit(1)
    fx, fy, cx, cy = ivals[:4]
    W_full, H_full = int(ivals[4]), int(ivals[5])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # ── Load query image ──
    img_path = Path(args.image_path)
    img_full = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB) if img_path.exists() else None
    title_str = lod_path.stem

    # ── Resolve render resolution ──
    import time as _time

    def _resolve_render_size(H_full, W_full, scale, width, height):
        aspect = W_full / H_full
        if width is not None or height is not None:
            if width is not None and height is not None:
                return int(height), int(width), None
            elif height is not None:
                return int(height), max(1, round(int(height) * aspect)), None
            else:
                return max(1, round(int(width) / aspect)), int(width), None
        else:
            s = scale if scale is not None else 0.5
            return int(H_full * s), int(W_full * s), s

    H_r, W_r, scale_used = _resolve_render_size(H_full, W_full, args.scale, args.width, args.height)
    K_r = K.copy()
    K_r[0, :] *= W_r / W_full
    K_r[1, :] *= H_r / H_full
    scale_label = f"scale={scale_used:.3g}" if scale_used is not None else f"{W_r}x{H_r}px"

    if img_full is not None:
        img = cv2.resize(img_full, (W_r, H_r))
    else:
        img = np.zeros((H_r, W_r, 3), dtype=np.uint8)

    print(f"\nRendering at {W_r}x{H_r} ({scale_label}) ...")
    timings = {}

    # ── Render passes ──
    t0 = _time.perf_counter()
    depth = lod.render_depth(w2c, K_r, (H_r, W_r), device="gpu")
    timings["depth"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    normals = lod.render_normals(w2c, K_r, (H_r, W_r), device="gpu")
    timings["normals"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    glr = lod._gpu_renderer(H_r, W_r)
    face_ids = glr.render_face_id_map(lod._w2c_local(w2c), K_r, (H_r, W_r))
    timings["face_ids"] = _time.perf_counter() - t0

    valid_d = depth > 0
    print(f"  Depth: {valid_d.sum()} valid px, range "
          f"[{depth[valid_d].min():.1f}, {depth[valid_d].max():.1f}] m"
          if valid_d.any() else "  Depth: 0 valid px")

    t0 = _time.perf_counter()
    sem_id, sem_color = lod.render_semantic(w2c, K_r, (H_r, W_r), label_source="normals", face_ids=face_ids)
    timings["semantic"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    inst_color, inst_id = lod.render_segmentation(w2c, K_r, (H_r, W_r), mode="instance", face_ids=face_ids)
    timings["instance"] = _time.perf_counter() - t0
    n_buildings = len(np.unique(inst_id[inst_id >= 0]))
    print(f"  Instance: {n_buildings} buildings")

    t0 = _time.perf_counter()
    depth_p, normals_p, label_map, segments = lod.get_plane_segmentation(
        w2c, K_r, (H_r, W_r),
        normal_cos_thresh=0.995, depth_rel_thresh=0.02, min_pixels=50,
        depth=depth, normals=normals,
    )
    timings["plane_seg"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    planes = lod.create_3d_planes(segments, depth_p, K_r, w2c,
                                   min_3d_area=2.0, min_extent=0.3, max_aspect_ratio=25.0)
    timings["plane_3d"] = _time.perf_counter() - t0
    planes_sorted = sorted(planes, key=lambda p: p["area_3d"], reverse=True)
    print(f"  Planes: {len(planes)} filtered 3D planes")

    t0 = _time.perf_counter()
    shaded = lod.render_shaded(w2c, K_r, (H_r, W_r), device="gpu")
    timings["shaded"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    edge_map = lod.render_edges(w2c, K_r, (H_r, W_r), face_ids=face_ids, depth=depth)
    timings["edges"] = _time.perf_counter() - t0
    n_edge_px = int((edge_map > 0.5).sum())
    print(f"  Edge pixels: {n_edge_px}")

    t0 = _time.perf_counter()
    wireframe_img, wire_info = lod.render_wireframe(w2c, K_r, (H_r, W_r), interval=0.5, depth=depth)
    timings["wireframe"] = _time.perf_counter() - t0
    n_wire_vis = int(wire_info["visible_mask"].sum())
    print(f"  Wireframe: {len(wire_info['pts_3d'])} sampled pts, {n_wire_vis} visible")

    total_render = sum(timings.values())

    # Standalone timings (include precomputed dependencies)
    standalone = {
        "depth":     timings["depth"],
        "normals":   timings["normals"],
        "face_ids":  timings["face_ids"],
        "semantic":  timings["face_ids"] + timings["semantic"],
        "instance":  timings["face_ids"] + timings["instance"],
        "plane_seg": timings["depth"] + timings["normals"] + timings["plane_seg"],
        "plane_3d":  timings["depth"] + timings["normals"] + timings["plane_seg"] + timings["plane_3d"],
        "shaded":    timings["shaded"],
        "edges":     timings["face_ids"] + timings["depth"] + timings["edges"],
        "wireframe": timings["depth"] + timings["wireframe"],
    }

    print(f"\n  Timings (total {total_render:.3f}s):")
    print(f"  {'name':12s}  {'step':>7s}  {'standalone':>10s}")
    for name, t in timings.items():
        print(f"    {name:12s}: {t:7.3f}s  {standalone[name]:10.3f}s")

    # ── Build overlays ──
    alpha = 0.45
    img_f = img.astype(np.float32)

    depth_overlay = img_f.copy()
    if valid_d.any():
        d_norm = np.zeros((H_r, W_r), dtype=np.float32)
        d_norm[valid_d] = (depth[valid_d] - depth[valid_d].min()) / (depth[valid_d].max() - depth[valid_d].min() + 1e-8)
        cmap = plt.cm.turbo(d_norm)[..., :3] * 255.0
        depth_overlay[valid_d] = img_f[valid_d] * (1 - alpha) + cmap[valid_d] * alpha

    normals_overlay = img_f.copy()
    nvis = ((normals + 1.0) / 2.0 * 255.0).clip(0, 255)
    normals_overlay[depth > 0] = img_f[depth > 0] * (1 - alpha) + nvis[depth > 0] * alpha

    sem_overlay = img_f.copy()
    sem_valid = sem_id > 0
    sem_overlay[sem_valid] = img_f[sem_valid] * (1 - alpha) + sem_color[sem_valid].astype(np.float32) * alpha

    inst_overlay = img_f.copy()
    inst_valid = inst_id >= 0
    inst_overlay[inst_valid] = img_f[inst_valid] * (1 - alpha) + inst_color[inst_valid].astype(np.float32) * alpha

    shaded_f = shaded.astype(np.float32)
    mesh_mask = np.linalg.norm(shaded_f, axis=2) > 10
    bg_white = np.full_like(img_f, 245.0)
    shaded_bg_overlay = img_f * 0.15 + bg_white * 0.85
    shaded_bg_overlay[mesh_mask] = shaded_f[mesh_mask] * 0.7 + img_f[mesh_mask] * 0.3

    edge_overlay = img_f.copy()
    edge_mask = edge_map > 0.5
    kernel = np.ones((2, 2), dtype=np.uint8)
    edge_dilated = cv2.dilate(edge_mask.astype(np.uint8), kernel).astype(bool)
    edge_overlay[edge_dilated] = [0, 255, 0]

    wireframe_overlay = img_f.copy()
    wire_kernel = np.ones((3, 3), dtype=np.uint8)
    wire_dilated = cv2.dilate((wireframe_img[:, :, 1] > 0).astype(np.uint8), wire_kernel).astype(bool)
    wireframe_overlay[wire_dilated] = [0, 255, 0]

    top_colors = [
        (255,0,0),(0,200,0),(0,0,255),(255,200,0),(255,0,255),
        (0,220,220),(255,128,0),(128,0,255),(0,128,255),(128,255,0),
        (200,100,100),(100,200,100),(100,100,200),(200,200,100),(200,100,200),
    ]
    planes_overlay = img_f.copy()
    for i, p in enumerate(planes_sorted):
        rc = p["pixel_indices"]
        c = np.array(top_colors[i % len(top_colors)], dtype=np.float32)
        planes_overlay[rc[:, 0], rc[:, 1]] = img_f[rc[:, 0], rc[:, 1]] * 0.45 + c * 0.55

    # ── Plot grid ──
    def _t(key):
        s, t = standalone[key], timings[key]
        return f"{s:.2f}s" if abs(s - t) < 0.005 else f"{t:.2f}s / {s:.2f}s"

    to_u8 = lambda x: np.clip(x, 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(3, 3, figsize=(27, 21))
    panels = [
        (axes[0, 0], img,                      "Query Image"),
        (axes[0, 1], to_u8(depth_overlay),      f"Depth ({_t('depth')})"),
        (axes[0, 2], to_u8(normals_overlay),    f"Normals ({_t('normals')})"),
        (axes[1, 0], to_u8(shaded_bg_overlay),  f"Shaded LoD ({_t('shaded')})"),
        (axes[1, 1], to_u8(edge_overlay),       f"Edges ({n_edge_px} px, {_t('edges')})"),
        (axes[1, 2], to_u8(wireframe_overlay),  f"Wireframe @0.5m ({n_wire_vis} pts, {_t('wireframe')})"),
        (axes[2, 0], to_u8(sem_overlay),        f"Semantic ({_t('semantic')})"),
        (axes[2, 1], to_u8(inst_overlay),       f"Instance: {n_buildings} bldgs ({_t('instance')})"),
        (axes[2, 2], to_u8(planes_overlay),     f"Planes ({len(planes_sorted)}) ({_t('plane_3d')})"),
    ]
    for ax, im, t in panels:
        ax.imshow(im); ax.set_title(t, fontsize=13); ax.axis("off")

    from matplotlib.patches import Patch
    axes[2, 0].legend(handles=[
        Patch(facecolor=np.array([220, 20, 60]) / 255.0, label="Roof"),
        Patch(facecolor=np.array([70, 130, 180]) / 255.0, label="Wall"),
        Patch(facecolor=np.array([128, 128, 128]) / 255.0, label="Other"),
    ], loc="lower right", fontsize=10, framealpha=0.8, edgecolor="white")

    plt.suptitle(f"{title_str}  —  total {total_render:.2f}s", fontsize=16, fontweight="bold")
    plt.tight_layout()

    out_path = OUT / f"overlay_{title_str}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out_path}")


# ------------------------------------------------------------------ #
#  LOD mesh loading helpers (moved from pipeline.py)                   #
# ------------------------------------------------------------------ #

def load_lod_from_gml(gml_files, output_dir):
    """Convert CityGML .gml files to a merged OBJ, then load as LoD.

    Parameters
    ----------
    gml_files : list of Path
        Paths to CityGML .gml files.
    output_dir : Path
        Directory to write (or read from cache) the merged OBJ.

    Returns
    -------
    LoD"""
    import tempfile
    import sys
    from pathlib import Path as _Path
    from lod_citygml_to_obj import main as citygml_main

    output_dir = _Path(output_dir)
    merged_obj = output_dir / "lod_merged.obj"
    if merged_obj.exists():
        print(f"  LOD: using cached merged OBJ {merged_obj}")
        lod = LoD(str(merged_obj))
        print(f"  LOD mesh loaded: {len(lod.vertices)} verts, {len(lod.faces)} faces")
        return lod

    all_vertices = []
    all_faces = []
    vert_offset = 0
    for gml_path in gml_files:
        print(f"  LOD: converting {gml_path.name} ...")
        tmp_obj = tempfile.NamedTemporaryFile(suffix='.obj', delete=False)
        tmp_obj.close()
        old_argv = sys.argv
        sys.argv = ['citygml_to_obj', '-i', str(gml_path), '-o', tmp_obj.name]
        try:
            citygml_main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv
        with open(tmp_obj.name, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.split()
                    all_vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif line.startswith('f '):
                    parts = line.split()
                    face = [int(p.split('/')[0]) + vert_offset for p in parts[1:]]
                    all_faces.append(face)
        vert_offset = len(all_vertices)
        _Path(tmp_obj.name).unlink(missing_ok=True)

    merged_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_obj, 'w') as f:
        f.write(f"# Merged from {len(gml_files)} CityGML files\n")
        for v in all_vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in all_faces:
            f.write("f " + " ".join(str(i) for i in face) + "\n")
    print(f"  LOD: merged {len(gml_files)} GML files -> {len(all_vertices)} verts, "
          f"{len(all_faces)} faces -> {merged_obj}")
    lod = LoD(str(merged_obj))
    return lod


def merge_obj_files(obj_files, output_dir):
    """Merge multiple OBJ files into a single LoD mesh.

    Parameters
    ----------
    obj_files : list of Path
        Paths to OBJ files.
    output_dir : Path
        Directory to write the merged OBJ.

    Returns
    -------
    LoD"""
    from pathlib import Path as _Path

    output_dir = _Path(output_dir)
    all_vertices = []
    all_faces = []
    vert_offset = 0
    for obj_path in obj_files:
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.split()
                    all_vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif line.startswith('f '):
                    parts = line.split()
                    face = [int(p.split('/')[0]) + vert_offset for p in parts[1:]]
                    all_faces.append(face)
        vert_offset = len(all_vertices)

    merged_obj = output_dir / "lod_merged.obj"
    merged_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_obj, 'w') as f:
        f.write(f"# Merged from {len(obj_files)} OBJ files\n")
        for v in all_vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in all_faces:
            f.write("f " + " ".join(str(i) for i in face) + "\n")
    print(f"  LOD: merged {len(obj_files)} OBJ files -> {len(all_vertices)} verts, "
          f"{len(all_faces)} faces")
    lod = LoD(str(merged_obj))
    return lod
def get_lod_polygons_cached(lod_vertices, lod_faces, lod_labels, cache_key,
                           normal_threshold=0.9998):
    """Return reconstructed polygons for the given mesh, using an in-memory cache."""
    if cache_key not in _lod_polygon_cache:
        _lod_polygon_cache[cache_key] = mesh_to_polygons(
            lod_vertices, lod_faces, lod_labels,
            normal_threshold=normal_threshold
        )
    return _lod_polygon_cache[cache_key]

def draw_lod_polygons(img, vertices_3d, vertices_proj, polygons,
                      is_elevation=False, alpha=0.4,
                      use_semantics=True, view_dir=None):
    """Render reconstructed LoD polygons (no interior diagonal lines).

    Each polygon in `polygons` is {'poly_idx': np.ndarray(N,) int32, 'label': str}
    where poly_idx indexes into vertices_3d / vertices_proj.

    Strategy for speed: vectorised pre-filtering over all polygon *representatives*
    (first-vertex 3D positions + per-polygon centroid stats) eliminates the bulk of
    polygons before any per-polygon Python work starts.  Only the surviving polygons
    (typically <5% of total for an oblique view) enter the cv2 draw loop.

    Depth awareness: for perspective views (view_dir is not None, is_elevation=False),
    a self-consistent Z-buffer is built by rasterising the LoD polygons themselves
    (far→near, 32-band uint8 encoding)."""
    if not polygons or len(vertices_proj) == 0:
        return

    if torch.is_tensor(vertices_proj):
        vertices_proj = vertices_proj.cpu().numpy()
    if torch.is_tensor(vertices_3d):
        vertices_3d = vertices_3d.cpu().numpy()

    h, w = img.shape[:2]

    SEMANTIC_COLORS = {
        'roof':     (255,   0,   0),
        'wall':     (  0,   0, 255),
        'ground':   (  0, 255,   0),
        'other':    (210, 180, 140),
        'building': (255, 165,   0),
        'closure':  (255, 225,   0),
    }
    DEFAULT_COLOR = (200, 200, 200)
    OUTLINE_COLOR = (0, 255, 0)   # green
    SUBPIXEL_SCALE = 16
    SUBPIXEL_SHIFT = 4
    ZBUF_SCALE = 4

    xs = vertices_proj[:, 0]
    ys = vertices_proj[:, 1]
    zs = vertices_proj[:, 2]

    # ── Vectorised pre-filtering ──────────────────────────────────────────────
    # For each polygon we compute a few scalar stats using only the FIRST THREE
    # vertices (faster than full loop).  Polygons that fail any early-reject test
    # are masked out before entering the draw loop.
    n = len(polygons)
    # Stack first/second/third vertex per polygon for normal computation
    i0 = np.array([pg['poly_idx'][0] for pg in polygons], dtype=np.int64)
    i1 = np.array([pg['poly_idx'][1] for pg in polygons], dtype=np.int64)
    i2 = np.array([pg['poly_idx'][2] for pg in polygons], dtype=np.int64)

    # 2-D centroid (mean of first-3 projected coords)
    cx = (xs[i0] + xs[i1] + xs[i2]) / 3.0
    cy = (ys[i0] + ys[i1] + ys[i2]) / 3.0
    cz = (zs[i0] + zs[i1] + zs[i2]) / 3.0  # approximate mean Z
    # Span using first-3 verts (good enough for rejection)
    xmin3 = np.minimum(np.minimum(xs[i0], xs[i1]), xs[i2])
    xmax3 = np.maximum(np.maximum(xs[i0], xs[i1]), xs[i2])
    ymin3 = np.minimum(np.minimum(ys[i0], ys[i1]), ys[i2])
    ymax3 = np.maximum(np.maximum(ys[i0], ys[i1]), ys[i2])

    keep = np.ones(n, dtype=bool)

    # Reject behind-camera
    keep &= cz >= 0.1

    # Reject extreme projections
    keep &= (xmax3 - xmin3) < w * 3
    keep &= (ymax3 - ymin3) < h * 3

    # Reject fully out-of-image polygons (generous margin)
    clip = max(w, h) * 1.5
    keep &= (cx > -clip) & (cx < w + clip) & (cy > -clip) & (cy < h + clip)

    # Back-face culling (perspective view only) — vectorised
    if view_dir is not None and not is_elevation:
        v0 = vertices_3d[i0]
        v1 = vertices_3d[i1]
        v2 = vertices_3d[i2]
        raw_n = np.cross(v1 - v0, v2 - v0)           # (N, 3)
        dot = raw_n @ view_dir                         # (N,) — unnormalised, sign is what matters
        keep &= dot <= 0                               # keep front-facing only

    # ── Self-consistent Z-buffer (built from LoD polygons, perspective only) ──
    # Uses the same 32-band encoding trick as draw_lod_mesh: encode camera-Z into
    # a uint8 raster (far→near painter's order), decode back for the Z-test.
    # This is fully consistent because both the zbuf fill and the Z-test operate
    # on the same cz values from the LoD projection — no coordinate space mismatch.
    zbuf = None
    Z_TOLERANCE = None
    if not is_elevation and view_dir is not None:
        pre_survivors = np.where(keep)[0]  # polygons that passed all pre-zbuf filters
        if len(pre_survivors) > 0:
            z_valid = cz[pre_survivors]
            z_min = float(z_valid.min())
            z_max = float(z_valid.max())
            z_range = max(z_max - z_min, 0.1)
            Z_TOLERANCE = z_range * 0.05 + 0.5   # 5% of depth range + 0.5 m slack

            # Build low-res Z-buffer: encode Z as uint8 (255=nearest, 1=farthest, 0=empty)
            zb_h = max(1, h // ZBUF_SCALE)
            zb_w = max(1, w // ZBUF_SCALE)
            zbuf_enc = np.zeros((zb_h, zb_w), dtype=np.uint8)

            # Sort far→near so nearer polygons overwrite farther ones in the raster
            order_zbuf = pre_survivors[np.argsort(cz[pre_survivors])[::-1]]
            for pi in order_zbuf:
                pg = polygons[pi]
                idx = pg['poly_idx']
                px = xs[idx]
                py = ys[idx]
                # Scale to zbuf coordinates (ZBUF_SCALE downsample, then subpixel for accuracy)
                pts_zb = (np.stack([px, py], axis=1) * (SUBPIXEL_SCALE / ZBUF_SCALE)
                          ).astype(np.int32).reshape(-1, 1, 2)
                enc = int(1 + round(254 * (1.0 - (cz[pi] - z_min) / z_range)))
                enc = max(1, min(255, enc))
                cv2.fillPoly(zbuf_enc, [pts_zb], enc, shift=SUBPIXEL_SHIFT)

            # Decode back to camera-Z float (0 = no surface rasterised → treat as very far)
            zbuf = np.where(zbuf_enc > 0,
                            z_min + z_range * (1.0 - (zbuf_enc.astype(np.float32) - 1) / 254.0),
                            z_max * 10.0)

            # Apply Z-buffer test on centroid
            zb_xi = np.clip((cx / ZBUF_SCALE).astype(int), 0, zbuf.shape[1] - 1)
            zb_yi = np.clip((cy / ZBUF_SCALE).astype(int), 0, zbuf.shape[0] - 1)
            zbuf_vals = zbuf[zb_yi, zb_xi]
            keep &= cz <= zbuf_vals + Z_TOLERANCE

    # Sort surviving polygons for painter's algorithm
    surviving = np.where(keep)[0]
    if len(surviving) == 0:
        return
    if is_elevation:
        # Top-down view: Z = world altitude. Draw ground (low Z) first,
        # rooftops (high Z) last so buildings are visible on top.
        order = surviving[np.argsort(cz[surviving])]       # ascending Z
    else:
        # Perspective view: Z = camera depth. Draw far (high Z) first,
        # near (low Z) last so closer objects occlude farther ones.
        order = surviving[np.argsort(cz[surviving])[::-1]] # descending Z (far → near)

    # ── Fill pass ─────────────────────────────────────────────────────────────
    overlay = img.copy()
    for pi in order:
        pg = polygons[pi]
        idx = pg['poly_idx']
        px = xs[idx]
        py = ys[idx]
        pts_int = (np.stack([px, py], axis=1) * SUBPIXEL_SCALE).astype(np.int32).reshape(-1, 1, 2)
        color = SEMANTIC_COLORS.get(pg['label'], DEFAULT_COLOR) if use_semantics else DEFAULT_COLOR
        cv2.fillPoly(overlay, [pts_int], color=color, shift=SUBPIXEL_SHIFT)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # ── Outline pass ─────────────────────────────────────────────────────────
    for pi in order:
        pg = polygons[pi]
        idx = pg['poly_idx']
        px = xs[idx]
        py = ys[idx]
        pz = zs[idx]
        # Only draw outline vertices within image bounds
        in_img = (px > -clip) & (px < w + clip) & (py > -clip) & (py < h + clip) & (pz >= 0.1)
        if in_img.sum() < 3:
            continue
        pts_int = (np.stack([px, py], axis=1) * SUBPIXEL_SCALE).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts_int], isClosed=True, color=OUTLINE_COLOR,
                      thickness=1, lineType=cv2.LINE_AA, shift=SUBPIXEL_SHIFT)

def convert_lod_to_obj(gml_path: Path, output_dir: Path, citygml2obj_script: Path, lod_level: int = 2) -> bool:
    """
    Standardized conversion of a single LoD GML tile to OBJ using CityGML2OBJv2.
    
    Args:
        gml_path: Path to GML file
        output_dir: Root LoDv{N}_obj directory
        citygml2obj_script: Path to CityGML2OBJs.py
        lod_level: Level of detail (1 or 2)
        
    Returns:
        True if conversion succeeded"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already converted (subdirectory with .obj files OR flat .obj file)
    expected_output = output_dir / gml_path.stem
    flat_obj = output_dir / f"{gml_path.stem}.obj"
    if flat_obj.exists() or (expected_output.exists() and expected_output.is_dir() and any(expected_output.glob('*.obj'))):
        print(f"   LoD{lod_level} already converted: {gml_path.stem}")
        return True
    
    print(f"   Converting LoD{lod_level}: {gml_path.name} -> OBJ...")
    
    try:
        cmd = [
            sys.executable,
            str(citygml2obj_script),
            '-i', str(gml_path.absolute()),
            '-o', str(output_dir.absolute()),
            '-t', '0',  # Keep original UTM coordinates (no translation)
            '-s', '1'   # Preserve semantics
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            cwd=citygml2obj_script.parent
        )
        
        if result.returncode != 0:
            print(f"   Warning: LoD2 conversion failed: {result.stderr[:200]}")
            return False
        
        return True
        
    except Exception as e:
        print(f"   Warning: LoD2 conversion error: {e}")
        return False

def convert_lod_tiles_parallel(
    gml_paths: List[Path],
    output_dir: Path,
    citygml2obj_script: Path,
    lod_level: int = 2,
    max_workers: int = 4
) -> List[str]:
    """
    Convert multiple LoD GML tiles to OBJ in parallel using ThreadPoolExecutor.
    
    Returns list of successfully converted OBJ subdirectory paths (relative strings)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if not gml_paths:
        return []
    
    results = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(gml_paths))) as executor:
        future_to_path = {
            executor.submit(convert_lod_to_obj, p, output_dir, citygml2obj_script, lod_level): p
            for p in gml_paths
        }
        for future in as_completed(future_to_path):
            gml_path = future_to_path[future]
            try:
                if future.result():
                    results.append(gml_path.stem)
            except Exception as e:
                print(f"   Warning: LoD{lod_level} conversion exception for {gml_path.name}: {e}")
    
    return results

