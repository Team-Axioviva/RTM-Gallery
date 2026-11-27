"""Navier-Stokes finite-time blowup -- 30 s RTM scene.

Timeline (30.0 s, 60 fps):
  0.0-1.2   NS equation fades in at screen centre
  1.2-2.0   hold
  2.0-3.8   equation slides to top-left and shrinks
  4.2-5.2   3D axes fade in; camera orbit starts (28 deg/s)
  5.4-7.4   water stripe reveals along z, far (+z) to near (-z)
  7.4-25.0  stripe twists and tightens (vortex stretching); blowup at t = 25
  25.0      flash + shock ring + particle burst; equation pops and fades
  25.0-26.8 burst settles, then FREEZES; camera keeps rotating
  29.4-30.0 global fade to black
"""
import ctypes
import os
import sys

import numpy as np

import real_time_manim  # noqa: F401  (must be first; installs bundled manim)
from real_time_manim import (DEGREES, Animation, Circle, MathTex, MeshBatch,
                             MLWindow, Rectangle, Text, ThreeDScene)
from real_time_manim.frames import FrameServer
from real_time_manim.rate_functions import linear
from real_time_manim.record import fast_record_scene

# ----------------------------------------------------------------- knobs ----
T_TOTAL = 30.0
T_EQ_IN = 1.2      # equation fade-in
T_EQ_HOLD = 2.0    # hold at centre
T_EQ_MOVE = 3.8    # done sliding to corner
T_AXES0, T_AXES1 = 4.2, 5.2
T_REV0, T_REV1 = 5.4, 7.4
T_BLOW = 25.0
T_FADE0 = 29.4

PHI = 70.0         # camera elevation (90 - eps)
THETA0 = -80.0     # start azimuth
ORBIT = 28.0       # deg/s after T_AXES0

Z_HALF = 3.6       # stripe half-length along z
N_S = 260          # stripe segments
W0 = 0.62          # stripe initial width
AXIS_R = 0.022     # axis tube radius
AXIS_LEN = 3.4     # axis half-length
AXIS_SEG = 60

N_PART = 1500      # explosion particles
SEED = 20261126

EQ_CORNER_W = 4.7  # equation width once parked top-left

# ------------------------------------------------------------- style table --
BG = "#070b14"
GOLD = np.array([0.949, 0.776, 0.424])
CYAN = np.array([0.435, 0.839, 0.800])
INK = np.array([0.937, 0.914, 0.863])
WHITE = np.array([1.000, 0.964, 0.910])
DIM = np.array([0.482, 0.576, 0.651])
AXIS_COL = {0: np.array([0.82, 0.42, 0.42]),   # x
            1: np.array([0.42, 0.72, 0.42]),   # y
            2: np.array([0.42, 0.55, 0.85])}   # z
DEEP = np.array([0.060, 0.240, 0.420])
SHALLOW = np.array([0.170, 0.640, 0.820])
CREAM = np.array([1.0, 0.94, 0.82])
RIMC = np.array([0.32, 0.80, 0.78])

LIGHT = np.array([-0.72, -0.28, 0.63]); LIGHT = LIGHT / np.linalg.norm(LIGHT)
AMBIENT, DIFFUSE, SPECULAR, SHININESS, RIM = 0.20, 0.95, 0.30, 38, 0.30
FOCAL = 20.0

GOLD_HEX = "#f2c66d"
CYAN_HEX = "#6fd6cd"
WHITE_HEX = "#fff6e8"
INK_HEX = "#efe9dc"
DIM_HEX = "#7b93a6"


def ss(t, a, b):
    u = np.clip((np.asarray(t, dtype=float) - a) / (b - a), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def lerp3(a, b, u):
    return np.asarray(a) * (1.0 - u) + np.asarray(b) * u


def _norm(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 1e-9, n, 1.0)


class _Driver(Animation):
    """One 30 s play; every piece of state is a pure function of t."""

    def __init__(self, scene, carrier, run_time=T_TOTAL):
        super().__init__(carrier, run_time=run_time, rate_func=linear)
        self._scene = scene

    def interpolate_mobject(self, alpha):
        self._scene.write_frame_state(float(alpha) * self.run_time)


class NSBlowup(ThreeDScene):
    def construct(self):
        win = MLWindow(1920, 1080)
        win.scene = self
        self._patch_background(win)
        self.camera.background_color = BG
        self.set_camera_orientation(phi=PHI * DEGREES, theta=THETA0 * DEGREES)

        self._build_hud()
        self._build_mesh()
        win.set_mesh(self.batch)

        self.flash = Rectangle(width=14.5, height=8.3, fill_color=WHITE_HEX,
                               fill_opacity=1.0, stroke_width=0)
        self.ring = Circle(radius=0.2, stroke_color=CYAN_HEX, stroke_width=3,
                           fill_opacity=0)
        self.fade = Rectangle(width=14.5, height=8.3, fill_color="#000000",
                              fill_opacity=1.0, stroke_width=0)
        self.add_fixed_in_frame_mobjects(self.eq, self.flash, self.ring,
                                         self.fade, *self.labels.values(),
                                         *self.timer)
        self._tk = -1

        win.play(_Driver(self, carrier=self.flash), run_time=T_TOTAL)
        win.close()

    # ----------------------------------------------------------------- build
    @staticmethod
    def _patch_background(win):
        fn = getattr(win.dll, "Render_SetBackgroundColor", None)
        if fn is not None:
            fn.restype = None
            fn.argtypes = [ctypes.c_float] * 3
            win._set_background = fn

    def _build_hud(self):
        # --- the equation ---------------------------------------------------
        self.eq = MathTex(
            r"\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot "
            r"\nabla)\, \mathbf{u} \;=\; -\nabla p \;+\;",
            r"\nu\, \nabla^{2} \mathbf{u}",
            r",\qquad \nabla \cdot \mathbf{u} = 0",
            font_size=46,
        )
        self.eq_base_colors = [GOLD, CYAN, GOLD]
        for i, sm in enumerate(self.eq):
            sm.set_color(hex_from(self.eq_base_colors[i]))
        self.eq_natw = float(self.eq.width)
        self.eq_scale_corner = EQ_CORNER_W / self.eq_natw
        self.eq.scale_to_fit_width(EQ_CORNER_W)
        self.eq.to_corner([-1, 1, 0], buff=0.62)
        self.corner_center = self.eq.get_center()
        self.eq.scale_to_fit_width(self.eq_natw)
        self.eq.move_to([0, 0, 0])

        # --- axis tip labels ------------------------------------------------
        self.tips = {"x": np.array([3.55, 0.0, 0.0]),
                     "y": np.array([0.0, 3.55, 0.0]),
                     "z": np.array([0.0, 0.0, 3.95])}
        self.labels = {}
        for name, tip in self.tips.items():
            lab = Text(name, font_size=26, color=INK_HEX)
            lab.set_opacity(0.0)
            self.labels[name] = lab

        # --- timer -----------------------------------------------------------
        self.timer = []
        for k in range(int(T_TOTAL) + 1):
            txt = Text(f"t = {k} s", font_size=28, color=DIM_HEX)
            txt.to_corner([1, -1, 0], buff=0.62)
            txt.set_opacity(0.0)
            self.timer.append(txt)
        self.timer[0].set_opacity(1.0)

    def _build_mesh(self):
        q_ax = AXIS_SEG * 4 * 3
        q_rib = N_S - 1
        self.q_ax, self.q_rib, self.q_par = q_ax, q_rib, N_PART
        self.batch = MeshBatch()
        self.batch.append(
            np.zeros((q_ax + q_rib + N_PART, 4, 3), dtype=np.float64),
            np.zeros((q_ax + q_rib + N_PART, 4, 4), dtype=np.float32),
        )
        self._build_axes()
        self._build_particles()
        # ribbon scratch
        self.s = np.linspace(0.0, 1.0, N_S)
        self.s_mid = 0.5 * (self.s[:-1] + self.s[1:])

    def _build_axes(self):
        quads, cols, nrm = [], [], []
        for axis, u in enumerate([np.array([1., 0, 0]),
                                  np.array([0., 1, 0]),
                                  np.array([0., 0, 1])]):
            if axis == 0:
                e1, e2 = np.array([0., 1, 0]), np.array([0., 0, 1])
            elif axis == 1:
                e1, e2 = np.array([0., 0, 1]), np.array([1., 0, 0])
            else:
                e1, e2 = np.array([1., 0, 0]), np.array([0., 1, 0])
            ring = np.array([e1, e2, -e1, -e2]) * AXIS_R
            for i in range(AXIS_SEG):
                p = u * (-AXIS_LEN + 2 * AXIS_LEN * i / AXIS_SEG)
                q = u * (-AXIS_LEN + 2 * AXIS_LEN * (i + 1) / AXIS_SEG)
                for j in range(4):
                    a = p + ring[j]
                    b = p + ring[(j + 1) % 4]
                    c = q + ring[(j + 1) % 4]
                    d = q + ring[j]
                    quad = np.array([a, b, c, d])
                    quads.append(quad)
                    cols.append(AXIS_COL[axis])
                    nrm.append(_norm(np.cross(quad[3] - quad[0],
                                              quad[1] - quad[0])))
        self.axis_quads = np.array(quads)
        self.axis_base = np.array(cols)
        self.axis_n = np.array(nrm).reshape(-1, 3)
        self.batch.quads[:self.q_ax] = self.axis_quads

    def _build_particles(self):
        rng = np.random.default_rng(SEED)
        z0 = Z_HALF - 2 * Z_HALF * rng.uniform(0, 1, N_PART)
        self.par_p0 = np.stack([rng.normal(0, 0.06, N_PART),
                                rng.normal(0, 0.06, N_PART), z0], axis=1)
        psi = rng.uniform(0, 2 * np.pi, N_PART)
        az = rng.normal(0, 0.6, N_PART)
        v = np.stack([np.cos(psi), np.sin(psi), az], axis=1)
        self.par_v = _norm(v)
        self.par_L = 0.6 + 2.4 * rng.uniform(0, 1, N_PART) ** 0.5
        self.par_tau0 = 0.7
        self.par_sz = 0.035 + 0.055 * rng.uniform(0, 1, N_PART)
        self.par_u = rng.uniform(0, 1, N_PART)

    # ------------------------------------------------------- per-frame state
    def write_frame_state(self, t):
        # --- camera ---------------------------------------------------------
        if t >= T_AXES0:
            theta = THETA0 + ORBIT * (t - T_AXES0) * ss(t, T_AXES0,
                                                        T_AXES0 + 1.2)
        else:
            theta = THETA0
        self.camera.set_theta(np.radians(theta))
        if t >= T_BLOW:
            amp = 0.05 * np.exp(-(t - T_BLOW) / 0.12)
            fc = np.array([amp * np.sin(23 * t + 1.3),
                           amp * np.cos(17 * t), 0.0])
        else:
            fc = np.zeros(3)
        self.camera.frame_center = fc

        rot = self.camera.generate_rotation_matrix()
        if os.environ.get("NS_DEBUG_PROJ"):
            import sys as _s
            pts = {"+x": (3.4, 0, 0), "-x": (-3.4, 0, 0), "+y": (0, 3.4, 0),
                   "-y": (0, -3.4, 0), "+z": (0, 0, 3.4), "-z": (0, 0, -3.4)}
            for k, p in pts.items():
                q = rot @ (np.array(p) - fc)
                f = FOCAL / max(FOCAL - q[2], 1e-6)
                print(f"[proj t={t:.2f}] {k}: screen=({q[0]*f:+.2f},"
                      f"{q[1]*f:+.2f}) px=({960+q[0]*f*135:.0f},"
                      f"{540-q[1]*f*135:.0f}) qz={q[2]:+.2f}",
                      file=_s.stderr)
        vw = rot.T @ np.array([0.0, 0.0, 1.0])
        vw = vw / np.linalg.norm(vw)
        hw = LIGHT + vw
        hw = hw / np.linalg.norm(hw)
        right = rot.T @ np.array([1.0, 0.0, 0.0])
        up = rot.T @ np.array([0.0, 1.0, 0.0])

        # --- axes (static geometry, per-frame lighting + fade-in) -----------
        a_ax = ss(t, T_AXES0, T_AXES1)
        self._paint(self.axis_base, self.axis_n, vw, hw, self.q_ax, 0,
                    alpha=0.9 * a_ax)

        # --- stripe -----------------------------------------------------------
        self._write_ribbon(t, vw, hw)

        # --- particles --------------------------------------------------------
        self._write_particles(t, right, up)

        # --- HUD --------------------------------------------------------------
        self._write_hud(t, rot, fc)

    def _paint(self, base, normals, vw, hw, n, off, alpha=1.0):
        # axes: chrome-like even lighting, visible from every camera angle
        ndl = np.abs(normals @ LIGHT) ** 0.85
        ndv = np.abs(normals @ vw)
        col = (base * (0.55 + 0.70 * ndl)[:, None]
               + (RIM * 0.8 * (1.0 - ndv) ** 4)[:, None] * RIMC)
        col = np.clip(col, 0.0, 1.0).astype(np.float32)
        c = self.batch.colors[off:off + n]
        c[..., :3] = col[:, None, :]
        c[..., 3] = np.float32(alpha)

    def _write_ribbon(self, t, vw, hw):
        tc = min(t, T_BLOW)
        tt = tc / T_BLOW
        inv = max(1.0 - tt, 1e-6)
        K = min(5.2, 0.15 + 1.7 * tt * tt * inv ** (-0.35))
        r_amp = 0.30 * min(1.0, tt / 0.12) * max(1.0 - tt, 0.0) ** 0.15
        phi = 2 * np.pi * K * self.s + 0.10 * np.sin(2 * np.pi *
                                                     (3 * self.s - 1.2 * tt))
        r = r_amp * (1.0 + 0.15 * np.sin(2 * np.pi * (1.8 * self.s - 2.0 * tt)))
        C = np.stack([r * np.cos(phi), r * np.sin(phi),
                      Z_HALF - 2 * Z_HALF * self.s], axis=1)
        T = _norm(np.diff(C, axis=0))
        phim = 0.5 * (phi[:-1] + phi[1:])
        Rhat = np.stack([np.cos(phim), np.sin(phim),
                         np.zeros_like(phim)], axis=1)
        Wv = _norm(np.cross(T, Rhat))
        w = W0 * max(0.06, (1.0 - tt) ** 0.5)
        p = C[:-1]
        q = C[1:]
        quads = np.stack([p + 0.5 * w * Wv, p - 0.5 * w * Wv,
                          q - 0.5 * w * Wv, q + 0.5 * w * Wv], axis=1)
        n = _norm(np.cross(quads[:, 3] - quads[:, 0],
                           quads[:, 1] - quads[:, 0]))

        # reveal sweep far -> near; unrevealed quads collapse to a point
        s_f = ss(t, T_REV0, T_REV1) if t < T_BLOW else 1.0
        active = self.s_mid < s_f
        if not np.all(active):
            front = self._centerline_at(s_f, K, r_amp)
            quads[~active] = front
        edge = (np.ones_like(self.s_mid) if s_f >= 0.999
                else np.clip(ss(s_f, self.s_mid, self.s_mid + 0.06), 0.0, 1.0))

        # water colour: deep -> shallow gradient, travelling sheen,
        # whitening as energy concentrates near blowup
        u = self.s_mid ** 0.75
        base = lerp3(DEEP, SHALLOW, u[:, None])
        base += 0.05 * np.sin(2 * np.pi * (3.1 * self.s_mid
                                           - 2.4 * tc / 4.0))[:, None]
        E = ss(tt, 0.74, 0.97)
        base = base * (1.0 - 0.72 * E) + 0.72 * E * np.array([0.82, 0.90, 0.98])

        # translucent water: two-sided lighting (light scatters through it),
        # high ambient + emissive floor so no orientation drops to black
        base = np.clip(base + 0.10, 0.0, 1.0)
        ndl = np.abs(n @ LIGHT) ** 0.85
        ndv = np.abs(n @ vw)
        ndh = np.abs(n @ hw) ** SHININESS
        col = (base * (0.48 + 0.80 * ndl)[:, None]
               + (SPECULAR * ndh)[:, None] * CREAM
               + (RIM * (1.0 - ndv) ** 4)[:, None] * RIMC)
        if os.environ.get("NS_DEBUG"):
            import sys as _s
            ndl_raw = n @ LIGHT
            ndv_raw = n @ vw
            print(f"[dbg t={t:.2f}] n@L: mean={ndl_raw.mean():+.3f} "
                  f"| frac pos={(ndl_raw > 0).mean():.2f} | "
                  f"n@V: mean={ndv_raw.mean():+.3f} frac pos="
                  f"{(ndv_raw > 0).mean():.2f} | col mean="
                  f"{col.mean(0).round(3)}", file=_s.stderr)
        col = np.clip(col, 0.0, 1.0).astype(np.float32)
        alpha = 0.92 * (1.0 if t < T_BLOW else 0.0)
        if os.environ.get("NS_NO_RIBBON"):
            alpha = 0.0
            quads[:] = self._centerline_at(0.5, K, r_amp)
        if t >= T_BLOW:                      # thread consumed by the burst
            quads[:] = self._centerline_at(1.0, K, r_amp)
        qs = self.batch.quads[self.q_ax:self.q_ax + self.q_rib]
        qs[:] = quads
        cs = self.batch.colors[self.q_ax:self.q_ax + self.q_rib]
        cs[..., :3] = col[:, None, :]
        cs[..., 3] = (alpha * edge)[:, None].astype(np.float32)

    def _centerline_at(self, s, K, r_amp):
        phi = 2 * np.pi * K * s + 0.10 * np.sin(2 * np.pi * (3 * s - 1.2))
        r = r_amp * (1.0 + 0.15 * np.sin(2 * np.pi * (1.8 * s - 2.0)))
        return np.array([r * np.cos(phi), r * np.sin(phi),
                         Z_HALF - 2 * Z_HALF * s])

    def _write_particles(self, t, right, up):
        qs = self.batch.quads[self.q_ax + self.q_rib:]
        cs = self.batch.colors[self.q_ax + self.q_rib:]
        if t < T_BLOW:
            qs[:] = self.par_p0[:, None, :]      # degenerate (zero area)
            cs[..., 3] = 0.0
            return
        tau = np.minimum(np.full(self.q_par, t - T_BLOW), 1.8)
        disp = self.par_L * (1.0 - np.exp(-tau / self.par_tau0))
        P = self.par_p0 + self.par_v * disp[:, None]
        sz = self.par_sz[:, None]
        qs[:] = np.stack([P - right * sz - up * sz, P + right * sz - up * sz,
                          P + right * sz + up * sz, P - right * sz + up * sz],
                         axis=1)
        u = np.minimum(tau / 0.5, 1.0)
        col = lerp3(np.array([1.0, 0.98, 0.92]),
                    np.array([0.45, 0.85, 0.85]), u[:, None])
        col = np.clip(col, 0.0, 1.0).astype(np.float32)
        a = np.minimum(tau / 0.06, 1.0) * (1.0 - 0.22 * tau / 1.8)
        cs[..., :3] = col[:, None, :]
        cs[..., 3] = (0.95 * a)[:, None].astype(np.float32)

    def _write_hud(self, t, rot, fc):
        # equation: intro, slide to corner, blowup flash
        intro = ss(t, 0.0, T_EQ_IN)
        u = 0.0 if t < T_EQ_HOLD else ss(t, T_EQ_HOLD, T_EQ_MOVE)
        s_eq = 1.0 - (1.0 - self.eq_scale_corner) * u
        pos = lerp3([0, 0, 0], self.corner_center, u)
        alpha = intro
        b = 0.0
        if t >= T_BLOW:
            b = ss(t - T_BLOW, 0.0, 0.5)
            s_eq *= 1.0 + 0.9 * b
            alpha = intro * (1.0 - b)
        self.eq.scale_to_fit_width(self.eq_natw * s_eq)
        self.eq.move_to(pos)
        self.eq.set_opacity(alpha)
        for i, sm in enumerate(self.eq):
            sm.set_color(hex_from(lerp3(self.eq_base_colors[i], WHITE, b)))

        # flash / ring / fade
        if T_BLOW <= t < T_BLOW + 0.9:
            self.flash.set_opacity(0.55 * np.exp(-(t - T_BLOW) / 0.09))
        else:
            self.flash.set_opacity(0.0)
        tau_r = t - T_BLOW
        if 0.0 <= tau_r < 2.2:
            r = 0.18 + 3.2 * (1.0 - np.exp(-2.2 * tau_r))
            self.ring.scale_to_fit_width(2.0 * r)
            self.ring.set_opacity(0.85 * np.exp(-tau_r / 0.9))
        else:
            self.ring.set_opacity(0.0)
        self.fade.set_opacity(ss(t, T_FADE0, T_TOTAL))

        # axis tip labels
        a_lab = ss(t, T_AXES0, T_AXES1)
        for name, lab in self.labels.items():
            q = rot @ (self.tips[name] - fc)
            f = FOCAL / max(FOCAL - q[2], 1e-6)
            lab.move_to([q[0] * f, q[1] * f, 0.0])
            lab.set_opacity(a_lab)

        # timer
        k = min(int(T_TOTAL), int(t))
        if k != self._tk:
            if 0 <= self._tk <= len(self.timer) - 1:
                self.timer[self._tk].set_opacity(0.0)
            self.timer[k].set_opacity(1.0)
            self._tk = k


def hex_from(rgb):
    rgb = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in rgb)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "ns_blowup.mp4")
    if "--stills" in sys.argv:
        i = sys.argv.index("--stills")
        times = [float(x) for x in sys.argv[i + 1].split(",")]
        fs = FrameServer(NSBlowup)
        for t in times:
            p = os.path.join(here, f"still_{t:05.2f}.png")
            fs.save_frame(t, p)
            print("saved", p)
    else:
        r = fast_record_scene(NSBlowup, out, overwrite=True, jobs=None)
        print(r["files"])
