#!/usr/bin/env python3
"""Three-body chaos — 30 s RTM render with real SDO solar textures.

Phases:   0-10 s  solar motion only
         10-20 s  + net force on each star (thick arrow, the star's colour)
         20-30 s  + pairwise force by each star on each star (thin arrows,
                   coloured by the EXERTING star)

Geometry: one depth-tested MeshBatch — three texture-mapped spheres
(per-vertex colour sampled from real 4096 px SDO imagery) + 9 force arrows
(shaft + head quads at z = +0.005, so the depth buffer occludes them
correctly against the spheres).

Physics: Pythagorean three-body (masses 3, 4, 5 at (1,3), (-2,-1), (1,-1),
v = 0), DOP853 rtol 1e-12, played on a constant-kinetic-speed clock so the
close-encounter whips stay smooth at 60 fps. See sim.py for the solver and
data/frames.npz for the sampled trajectory.
"""
import ctypes
import os

import numpy as np

import real_time_manim  # noqa: F401  (must precede any `manim` import)
from real_time_manim import (
    DEGREES,
    UP,
    Animation,
    Dot,
    ManimColor,
    Text,
    ThreeDScene,
)
from real_time_manim import FrameServer, fast_record_scene
from real_time_manim.mesh import MeshBatch
from real_time_manim.vulkan_bind import MLWindow

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

FRAMES = np.load(os.path.join(DATA, "frames.npz"))
STATIC = np.load(os.path.join(DATA, "mesh_static.npz"))

POS = FRAMES["pos"]              # (1801, 3, 2)  true trajectory
POS_DISP = FRAMES["pos_disp"]    # (1801, 3, 2)  display (never touching)
NET_DIR = FRAMES["net_dir"]      # (1801, 3, 2)
NET_LEN = FRAMES["net_len"]      # (1801, 3)
PAIR_DIR = FRAMES["pair_dir"]    # (1801, 3, 3, 2)
PAIR_LEN = FRAMES["pair_len"]    # (1801, 3, 3)
PAIR_VIS = FRAMES["pair_vis"]    # (1801, 3, 3)
R_STAR = FRAMES["r_star"]        # (3,)
ZOOM = float(FRAMES["zoom"])

UNIT = STATIC["unit"]            # (N_vert, 3)
QUAD_IDX = STATIC["quads"]       # (Q_sph, 4)
TEX = STATIC["tex_colors"]       # (3, N_vert, 3)
ARROW_C = STATIC["arrow_colors"]  # (3, 3)

N_VERT = UNIT.shape[0]
N_SPH = QUAD_IDX.shape[0]
N_SPH_TOT = 3 * N_SPH
N_ARROW = 9
Q_TOT = N_SPH_TOT + 2 * N_ARROW

# --- fading trails (3b1b FourierCirclesScene.add_path_fader, ported to mesh:
# per-segment colour-lerp toward bg IS the opacity channel; width fades too)
T_TRAIL = 2.5        # s of history kept
N_TRAIL = 150        # segments (60 fps * 2.5 s)
TRAIL_Z = 0.002      # below arrows (0.005), above plane
W_TRAIL = 0.06       # max ribbon width (world units)
ALPHA_TRAIL = 0.5    # max brightness of the freshest segment
FADE_EXP = 0.55      # (1 - age/T)^FADE_EXP  -- 3b1b's power-law tail
TRAIL_LO = Q_TOT
Q_TOT += 3 * N_TRAIL

_ages = np.arange(N_TRAIL) / 60.0
FAC_TRAIL = np.maximum(1.0 - _ages / T_TRAIL, 0.0) ** FADE_EXP

BG = np.array([0.02745098, 0.04313725, 0.07843137])   # #070b14
ARROW_Z = 0.005

FPS = 60
DUR = 30.0
N_FR = int(DUR * FPS) + 1

W_NET, HL_NET, HW_NET = 0.13, 0.28, 0.28
W_PAIR, HL_PAIR, HW_PAIR = 0.055, 0.14, 0.14

# experiment overrides (render-width bisection)
W_NET = float(os.environ.get("T3B_NET_W", W_NET))
W_PAIR = float(os.environ.get("T3B_PAIR_W", W_PAIR))
NO_NET = os.environ.get("T3B_NO_NET") == "1"
NO_PAIR = os.environ.get("T3B_NO_PAIR") == "1"

INK = ManimColor("#efe9dc")


def sstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def patch_background(win):
    try:
        if getattr(win, "_set_background", None) is None:
            fn = win.dll.Render_SetBackgroundColor
            fn.restype = None
            fn.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
            win._set_background = fn
        print("[bg] background colour active")
    except Exception as exc:
        print("[bg] patch skipped:", exc)


class ChaosAnim(Animation):
    """Drive mesh + captions from precomputed per-frame data (pure f(t))."""

    def __init__(self, mesh, caps, **kwargs):
        super().__init__(Dot(point=np.zeros(3), radius=0.001), **kwargs)
        self.mesh = mesh
        self.caps = caps
        self.q = mesh.quads
        self.c = mesh.colors

    def _arrow(self, row, c2, u2, L, r_star, w, hl, hw, col, hidden):
        s = np.array([c2[0], c2[1], ARROW_Z])
        u = np.array([u2[0], u2[1], 0.0])
        perp = np.array([-u2[1], u2[0], 0.0])
        start = s + u * (r_star + 0.05)
        if hidden:
            self.q[row] = start
            self.q[row + 1] = start
            self.c[row, :, :3] = BG
            self.c[row + 1, :, :3] = BG
            return
        end = start + u * L
        self.q[row] = [start - perp * (w / 2), end - perp * (w / 2),
                       end + perp * (w / 2), start + perp * (w / 2)]
        tip = end + u * hl
        self.q[row + 1] = [end + perp * (hw / 2), end - perp * (hw / 2),
                           tip, tip]
        for r in (row, row + 1):
            self.c[r, :, :3] = col

    def _write_trail(self, st, k):
        """3b1b-style fading path: ribbon of the last 2.5 s of the star's
        displayed path; each segment's width and colour fade with age."""
        lo = TRAIL_LO + st * N_TRAIL
        jj = np.arange(1, N_TRAIL + 1)
        prev_idx = k - jj - 1
        cur_idx = k - jj
        valid = (prev_idx >= 0) & (FAC_TRAIL > 1e-6)
        prev = POS_DISP[np.maximum(prev_idx, 0), st]
        cur = POS_DISP[np.maximum(cur_idx, 0), st]
        d = cur - prev
        L = np.linalg.norm(d, axis=1)
        u = np.zeros_like(d)
        ok = L > 1e-9
        u[ok] = d[ok] / L[ok, None]
        perp = np.stack([-u[:, 1], u[:, 0]], axis=1)
        h = (W_TRAIL / 2.0) * FAC_TRAIL[:, None]
        p = np.zeros((N_TRAIL, 3))
        p[:, :2] = perp * h
        A = np.zeros((N_TRAIL, 3))
        A[:, :2] = prev
        A[:, 2] = TRAIL_Z
        B = np.zeros((N_TRAIL, 3))
        B[:, :2] = cur
        B[:, 2] = TRAIL_Z
        q = np.stack([A - p, B - p, B + p, A + p], axis=1)
        col = (BG[None, None, :]
               + (ARROW_C[st][None, None, :] - BG[None, None, :])
               * (ALPHA_TRAIL * FAC_TRAIL)[:, None, None])
        inv = ~valid
        if inv.any():
            q[inv] = A[inv][:, None, :]
            col[inv] = BG
        self.q[lo:lo + N_TRAIL] = q
        self.c[lo:lo + N_TRAIL, :, :3] = col

    def write_frame(self, k, t):
        q = self.q
        for st in range(3):
            v = UNIT[QUAD_IDX] * R_STAR[st]
            v[:, :, :2] += POS_DISP[k, st]
            q[st * N_SPH:(st + 1) * N_SPH] = v
        a_net = sstep((t - 10.0) / 0.8)
        a_pair = sstep((t - 20.0) / 0.8)
        row = N_SPH_TOT
        for j in range(3):
            col = BG + (ARROW_C[j] - BG) * a_net
            self._arrow(row, POS_DISP[k, j], NET_DIR[k, j], NET_LEN[k, j],
                        R_STAR[j], W_NET, HL_NET, HW_NET, col,
                        hidden=(NET_LEN[k, j] < 0.18) or (a_net <= 1e-6)
                        or NO_NET)
            row += 2
        for j in range(3):
            for i in range(3):
                if i == j:
                    continue
                col = BG + (ARROW_C[i] - BG) * a_pair
                self._arrow(row, POS_DISP[k, j], PAIR_DIR[k, j, i], PAIR_LEN[k, j, i],
                            R_STAR[j], W_PAIR, HL_PAIR, HW_PAIR, col,
                            hidden=(not PAIR_VIS[k, j, i]) or (a_pair <= 1e-6)
                            or NO_PAIR)
                row += 2
        for st in range(3):
            self._write_trail(st, k)
        self.caps[0].set_opacity(float(a_net))
        self.caps[1].set_opacity(float(a_pair))

    def interpolate_mobject(self, alpha):
        t = alpha * self.run_time
        k = int(round(t * FPS))
        if k > N_FR - 1:
            k = N_FR - 1
        self.write_frame(k, t)


class ThreeBodyChaos(ThreeDScene):
    def construct(self):
        win = MLWindow(960, 540)
        win.scene = self
        self.play = win.play

        patch_background(win)
        self.camera.background_color = "#070b14"
        self.camera.set_phi(15 * DEGREES)
        self.camera.set_theta(-90 * DEGREES)
        self.camera.set_zoom(ZOOM)

        cap_net = Text("net force on each star", font="Baskerville",
                       font_size=24, color=INK)
        cap_net.to_corner(np.array([-1, -1, 0]), buff=0.62)
        cap_pair = Text("force by each star on each star", font="Baskerville",
                        font_size=24, color=INK)
        cap_pair.to_corner(np.array([-1, -1, 0]), buff=0.62).shift(UP * 0.55)
        cap_net.set_opacity(0.0)
        cap_pair.set_opacity(0.0)
        self.add_fixed_in_frame_mobjects(cap_net, cap_pair)

        mesh = MeshBatch()
        mesh.quads = np.zeros((Q_TOT, 4, 3))
        mesh.colors = np.ones((Q_TOT, 4, 4), dtype=np.float32)
        for st in range(3):
            lo = st * N_SPH
            mesh.colors[lo:lo + N_SPH, :, :3] = TEX[st][QUAD_IDX]
        win.set_mesh(mesh)

        anim = ChaosAnim(mesh, (cap_net, cap_pair), run_time=DUR)
        anim.write_frame(0, 0.0)
        win.play(anim, run_time=DUR)
        win.close()


if __name__ == "__main__":
    out = os.path.join(HERE, "three_body_chaos.mp4")
    r = fast_record_scene(ThreeBodyChaos, out, jobs=None)
    print("files:", r["files"])
