"""Visualisation for CommunityHealthEnv.

* ``OpenGLRenderer``: 3D scene (PyOpenGL + pygame window) with lit, shaded
  houses, clinic, river, bridges and the health worker, plus a HUD panel.
* ``Pygame2DRenderer``: top-down fallback used when no OpenGL context is
  available (e.g. headless CI). Both return RGB frames in ``rgb_array`` mode.
"""

from __future__ import annotations

import math
import os

import numpy as np

from environment.custom_env import (
    ACTION_NAMES, BRIDGES, CLINIC, GRID, HOUSEHOLDS, MAX_KITS, MAX_STEPS, RIVER_X, is_river,
)

SCENE_W, PANEL_W, H = 760, 340, 680
W = SCENE_W + PANEL_W

C_GRASS = (0.42, 0.66, 0.33)
C_GRASS2 = (0.38, 0.61, 0.30)
C_RIVER = (0.20, 0.45, 0.85)
C_BRIDGE = (0.55, 0.38, 0.22)
C_UNKNOWN = (0.85, 0.80, 0.68)
C_HIGH = (0.86, 0.22, 0.20)
C_REFERRED = (0.40, 0.35, 0.85)
C_LOW = (0.30, 0.75, 0.40)
C_EDU = (0.95, 0.85, 0.25)
C_ROOF = (0.50, 0.25, 0.18)
C_AGENT = (0.05, 0.60, 0.65)


def household_state(env, i):
    if not env.screened[i]:
        return "unscreened", C_UNKNOWN
    if env.high_risk[i]:
        return ("referred", C_REFERRED) if env.referred[i] else ("HIGH RISK", C_HIGH)
    return ("educated", C_EDU) if env.educated[i] else ("low risk", C_LOW)


def _hud_surface(env, pygame, font, small):
    surf = pygame.Surface((PANEL_W, H))
    surf.fill((24, 28, 36))
    y = 16

    def line(text, f=small, col=(225, 228, 235), dy=24):
        nonlocal y
        surf.blit(f.render(text, True, col), (16, y))
        y += dy

    line("Community Health Outreach", font, (120, 220, 210), 34)
    info = env._info()
    line(f"Step      {env.t:>3} / {MAX_STEPS}")
    line(f"Kits      {env.kits} / {MAX_KITS}")
    line(f"Screened  {info['screened']} / {len(HOUSEHOLDS)}")
    line(f"High risk found {info['high_risk_found']}  referred {info['referred']}")
    line(f"Educated  {info['educated']}")
    line(f"Return    {env.episode_return:7.2f}")
    y += 6
    act = "-" if env.last_action is None else ACTION_NAMES[env.last_action]
    line(f"Action: {act}", col=(255, 210, 120))
    line(f"Reward: {env.last_reward:+.2f}", col=(255, 210, 120))
    line(env.last_event[:38], col=(255, 255, 255), dy=34)
    line("Households", font, (120, 220, 210), 30)
    for i in range(len(HOUSEHOLDS)):
        label, col = household_state(env, i)
        c = tuple(int(255 * v) for v in col)
        pygame.draw.rect(surf, c, (16, y + 3, 14, 14))
        line(f"   H{i} {HOUSEHOLDS[i]}  {label}", dy=22)
    if info["success"]:
        y += 8
        line("MISSION COMPLETE", font, (120, 255, 140))
    return surf


class OpenGLRenderer:
    def __init__(self, render_mode: str):
        import pygame
        from OpenGL import GL, GLU

        self.pg, self.gl, self.glu = pygame, GL, GLU
        self.mode = render_mode
        pygame.init()
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if render_mode == "rgb_array":
            flags |= pygame.HIDDEN
        pygame.display.set_caption("Community Health Outreach - RL agent (OpenGL)")
        self.screen = pygame.display.set_mode((W, H), flags)
        self.font = pygame.font.SysFont("menlo,dejavusansmono,courier", 20, bold=True)
        self.small = pygame.font.SysFont("menlo,dejavusansmono,courier", 15)
        self.clock = pygame.time.Clock()
        self.quadric = GLU.gluNewQuadric()
        self.angle = 0.0
        gl = GL
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (0.35, 0.35, 0.38, 1))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (0.85, 0.85, 0.8, 1))
        gl.glShadeModel(gl.GL_SMOOTH)

    # -------------------------------------------------------------- shapes
    def box(self, x, y, z, sx, sy, sz, col):
        gl = self.gl
        gl.glColor3f(*col)
        gl.glPushMatrix()
        gl.glTranslatef(x, y, z)
        gl.glScalef(sx, sy, sz)
        gl.glBegin(gl.GL_QUADS)
        faces = [
            ((0, 1, 0), [(-.5, .5, -.5), (-.5, .5, .5), (.5, .5, .5), (.5, .5, -.5)]),
            ((0, -1, 0), [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, -.5, .5), (-.5, -.5, .5)]),
            ((0, 0, 1), [(-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)]),
            ((0, 0, -1), [(-.5, -.5, -.5), (-.5, .5, -.5), (.5, .5, -.5), (.5, -.5, -.5)]),
            ((1, 0, 0), [(.5, -.5, -.5), (.5, .5, -.5), (.5, .5, .5), (.5, -.5, .5)]),
            ((-1, 0, 0), [(-.5, -.5, -.5), (-.5, -.5, .5), (-.5, .5, .5), (-.5, .5, -.5)]),
        ]
        for n, vs in faces:
            gl.glNormal3f(*n)
            for v in vs:
                gl.glVertex3f(*v)
        gl.glEnd()
        gl.glPopMatrix()

    def roof(self, x, y, z, s, h, col):
        gl = self.gl
        gl.glColor3f(*col)
        apex = (x, y + h, z)
        corners = [(x - s, y, z - s), (x + s, y, z - s), (x + s, y, z + s), (x - s, y, z + s)]
        gl.glBegin(gl.GL_TRIANGLES)
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            u = np.subtract(b, a)
            v = np.subtract(apex, a)
            gl.glNormal3f(*np.cross(v, u))
            gl.glVertex3f(*a)
            gl.glVertex3f(*apex)
            gl.glVertex3f(*b)
        gl.glEnd()

    def cylinder(self, x, y, z, r, h, col):
        gl, glu = self.gl, self.glu
        gl.glColor3f(*col)
        gl.glPushMatrix()
        gl.glTranslatef(x, y, z)
        gl.glRotatef(-90, 1, 0, 0)
        glu.gluCylinder(self.quadric, r, r, h, 18, 1)
        glu.gluDisk(self.quadric, 0, r, 18, 1)
        gl.glTranslatef(0, 0, h)
        glu.gluDisk(self.quadric, 0, r, 18, 1)
        gl.glPopMatrix()

    def sphere(self, x, y, z, r, col):
        gl = self.gl
        gl.glColor3f(*col)
        gl.glPushMatrix()
        gl.glTranslatef(x, y, z)
        self.glu.gluSphere(self.quadric, r, 16, 12)
        gl.glPopMatrix()

    # --------------------------------------------------------------- frame
    def draw(self, env):
        pg, gl, glu = self.pg, self.gl, self.glu
        for e in pg.event.get():
            if e.type == pg.QUIT:
                self.close()
                raise SystemExit
        gl.glViewport(0, 0, SCENE_W, H)
        gl.glClearColor(0.62, 0.80, 0.95, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(42, SCENE_W / H, 0.1, 100)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        self.angle += 0.25
        cx = 4.5 + 2.0 * math.sin(math.radians(self.angle))
        glu.gluLookAt(cx, 13.5, 17.0, 4.5, 0, 4.0, 0, 1, 0)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (3, 12, 6, 1))

        for gx in range(GRID):
            for gz in range(GRID):
                if is_river(gx, gz):
                    self.box(gx, -0.25, gz, 1, 0.2, 1, C_RIVER)
                elif gx == RIVER_X and gz in BRIDGES:
                    self.box(gx, -0.1, gz, 1, 0.2, 1, C_RIVER)
                    self.box(gx, 0.05, gz, 1.05, 0.12, 0.7, C_BRIDGE)
                else:
                    self.box(gx, -0.1, gz, 0.98, 0.2, 0.98, C_GRASS if (gx + gz) % 2 else C_GRASS2)

        # clinic: white building with red cross
        x, z = CLINIC
        self.box(x, 0.45, z, 0.85, 0.9, 0.85, (0.95, 0.95, 0.95))
        self.box(x, 0.93, z, 0.55, 0.06, 0.16, (0.9, 0.1, 0.1))
        self.box(x, 0.93, z, 0.16, 0.06, 0.55, (0.9, 0.1, 0.1))

        t = pg.time.get_ticks() / 1000.0
        for i, (hx, hz) in enumerate(HOUSEHOLDS):
            _, col = household_state(env, i)
            self.box(hx, 0.3, hz, 0.6, 0.6, 0.6, col)
            self.roof(hx, 0.6, hz, 0.38, 0.35, C_ROOF)
            if env.screened[i] and env.high_risk[i] and not env.referred[i]:
                # pulsing alert beacon over unreferred high-risk households
                self.sphere(hx, 1.35 + 0.08 * math.sin(4 * t), hz, 0.12, (1, 0.15, 0.1))

        ax, az = env.pos
        self.cylinder(ax, 0.0, az + 0.15, 0.17, 0.55, C_AGENT)
        self.sphere(ax, 0.72, az + 0.15, 0.14, (0.95, 0.80, 0.65))
        for k in range(env.kits):  # test kits carried, shown as a stack
            self.box(ax + 0.28, 0.08 + 0.1 * k, az + 0.15, 0.16, 0.08, 0.16, (1, 1, 1))

        # HUD panel drawn as a pygame surface blitted with glDrawPixels
        gl.glViewport(0, 0, W, H)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_DEPTH_TEST)
        hud = _hud_surface(env, pg, self.font, self.small)
        data = pg.image.tostring(hud, "RGBA", True)
        gl.glWindowPos2i(SCENE_W, 0)
        gl.glDrawPixels(PANEL_W, H, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_DEPTH_TEST)

        frame = None
        if self.mode == "rgb_array":
            gl.glReadBuffer(gl.GL_BACK)
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
            raw = gl.glReadPixels(0, 0, W, H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            frame = np.flipud(np.frombuffer(raw, np.uint8).reshape(H, W, 3)).copy()
        pg.display.flip()
        if self.mode == "human":
            self.clock.tick(int(os.environ.get("RENDER_FPS", 6)))
        return frame

    def close(self):
        self.pg.display.quit()


class Pygame2DRenderer:
    """Top-down fallback renderer (no OpenGL needed)."""

    def __init__(self, render_mode: str):
        import pygame

        if render_mode == "rgb_array":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        self.pg = pygame
        self.mode = render_mode
        pygame.init()
        self.cell = SCENE_W // GRID if SCENE_W // GRID * GRID <= H else H // GRID
        self.cell = min(H // GRID, SCENE_W // GRID)
        if render_mode == "human":
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("Community Health Outreach (2D)")
        else:
            self.screen = pygame.Surface((W, H))
        self.font = pygame.font.SysFont("menlo,dejavusansmono,courier", 20, bold=True)
        self.small = pygame.font.SysFont("menlo,dejavusansmono,courier", 15)
        self.clock = pygame.time.Clock()

    def draw(self, env):
        pg, s, c = self.pg, self.screen, self.cell
        if self.mode == "human":
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    raise SystemExit
        s.fill((160, 205, 240))
        rgb = lambda col: tuple(int(255 * v) for v in col)
        for gx in range(GRID):
            for gy in range(GRID):
                col = C_RIVER if is_river(gx, gy) else (C_BRIDGE if gx == RIVER_X else
                                                        (C_GRASS if (gx + gy) % 2 else C_GRASS2))
                pg.draw.rect(s, rgb(col), (gx * c + 20, gy * c, c - 1, c - 1))
        x, y = CLINIC
        pg.draw.rect(s, (245, 245, 245), (x * c + 26, y * c + 6, c - 13, c - 13))
        pg.draw.rect(s, (220, 20, 20), (x * c + 20 + c // 2 - 4, y * c + 12, 8, c - 25))
        pg.draw.rect(s, (220, 20, 20), (x * c + 32, y * c + c // 2 - 4, c - 25, 8))
        for i, (hx, hy) in enumerate(HOUSEHOLDS):
            _, col = household_state(env, i)
            pg.draw.rect(s, rgb(col), (hx * c + 30, hy * c + 14, c - 21, c - 24))
            pg.draw.polygon(s, rgb(C_ROOF), [(hx * c + 26, hy * c + 16), (hx * c + 20 + c // 2, hy * c + 3),
                                              (hx * c + 14 + c, hy * c + 16)])
        ax, ay = env.pos
        pg.draw.circle(s, rgb(C_AGENT), (ax * c + 20 + c // 2, ay * c + c // 2 + 6), c // 4)
        s.blit(_hud_surface(env, pg, self.font, self.small), (SCENE_W, 0))
        if self.mode == "human":
            pg.display.flip()
            self.clock.tick(int(os.environ.get("RENDER_FPS", 6)))
            return None
        return np.transpose(pg.surfarray.array3d(s), (1, 0, 2)).copy()

    def close(self):
        self.pg.display.quit()


def make_renderer(kind: str, render_mode: str):
    if kind == "opengl":
        try:
            return OpenGLRenderer(render_mode)
        except Exception as exc:  # no GL context available
            print(f"[rendering] OpenGL unavailable ({exc}); falling back to 2D pygame renderer")
    return Pygame2DRenderer(render_mode)
