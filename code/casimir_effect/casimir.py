#!/usr/bin/env python3
r"""Casimir effect — 30 s RTM film (1080p60, 1800 frames).

Physics kept honest:
- The modes between the plates are standing waves sin(n pi x / a) with
  NODES pinned on the conductor faces (E_parallel = 0 at a conductor).
  The full mode set n = 1..300 is drawn as a CARPET: one row per mode,
  stacked vertically inside the gap (n=1 centre, higher n toward the
  plate edges), each row brightness-encoded with its |sin(n pi zeta)|
  node pattern and breathing at its own frequency n*OMEGA0 (1:2:3:...).
- Rows appear chronologically, n = 1 -> 300 (~27 modes/s): the addition
  effect. A live counter shows the mode sum 1+2+...+N = N(N+1)/2 racing
  upward — the naive sum diverges; the corner formula zeta(-1) = -1/12
  is its regularization (verify mode gates it numerically).
- Resultant force: attractive — arrows grow FROM the plate faces pointing
  INTO the gap.

Camera: physically FIXED at phi = 60 deg, theta = 0 for the whole film.
All motion is the world transform  p(t) = R_z(-theta_film(t)) . s(t) . p0
(counter-rotation is image-identical to an orbiting camera; uniform scale
about the gap centre is image-identical to a dolly). Segment 2 freezes
theta_film, so the camera is fixed while the "zoom" is the dolly.

Timeline (t in s):
  0.0- 1.2   plates settle in (scale 0.90 -> 1.00)
  0.0- 7.0   camera orbit theta_film 0 -> 90 deg (the gap opens up)
  7.0-12.0   dolly into the gap (s 1 -> 2.6), camera frozen
 12.0-23.0   mode carpet grows n = 1 -> 300 (lower term first); counter
             ticks at 4 Hz: 1+2+...+N = N(N+1)/2
 23.0-24.5   carpet holds at 300 modes (1+2+...+300 = 45150)
 21.0-26.0   dolly out (s 2.6 -> 1.3)
 22.0-24.0   force arrows grow from the plate faces
 24.5-25.5   carpet + counter fade out
 26.0-27.5   zeta(-1) = 1+2+3+... = -1/12 fades in, bottom right
 26.0-30.0   slow drift theta_film 90 -> 96 deg
 29.0-30.0   global fade to background
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

import real_time_manim  # noqa: F401  (must be the first import)
from real_time_manim import Animation, DEGREES, Rectangle, ThreeDScene, VGroup, linear
from real_time_manim.mesh import MeshBatch
from real_time_manim.record import fast_record_scene
from real_time_manim.vulkan_bind import MLWindow

try:
    from real_time_manim import TypstMath
except ImportError:  # pragma: no cover — older bundle: bundled manim holds it
    from manim import TypstMath  # noqa: E402

# ------------------------------------------------------------------- knobs
FPS = 60
DURATION = 30.0

L = 4.0                     # plate square side
T = 0.55                    # plate thickness (gap direction)
GAP = 1.1                   # inner-face separation
WAVE_HALF = 0.52            # carpet span from gap centre (clears the faces)

N_MAX = 300                 # number of modes shown ("hundreds")
NZQ = 96                    # quads per carpet row
DZ = 1.7 / 149.0            # row spacing (rows fill z in +-1.7)
HZ = 0.004                  # carpet row half-height (z)
OMEGA0 = 1.3                # mode-1 angular frequency; mode n -> n*OMEGA0

S_MAX = 2.6                 # dolly-in scale (gap fills the frame)
S_END = 1.3                 # dolly-out scale (formula corner stays clear)

PHI = 60.0 * DEGREES        # camera elevation (constant for the whole film)
CE = math.sin(PHI)          # cos(eps) = sin(phi)
SE = math.cos(PHI)          # sin(eps) = cos(phi)
VIEW = np.array([CE, 0.0, SE])          # camera view direction (world)

ARROW_L = 0.34              # arrow shaft length
ARROW_H = 0.13              # arrow head length
ARROW_HW = 0.09             # arrow head half-width (z)
ARROW_SW = 0.035            # arrow shaft half-width (z)
ARROW_WY = 0.012            # arrow thickness (y — minimal, rasterizer needs area)
ARROW_Z = 1.85              # arrow row in z (above the carpet rows)

# timeline (start, length) / (start, length, amount)
SETTLE = (0.0, 1.2)         # plates settle 0.90 -> 1.00
ORBIT = (0.0, 7.0)          # theta_film 0 -> 90 deg
ZOOMIN = (7.0, 5.0)         # s 1 -> S_MAX
ZOOMOUT = (21.0, 5.0)       # s S_MAX -> S_END
CARPET = (12.0, 11.0)       # n 1 -> N_MAX
WAVE_OUT = (24.5, 1.0)      # carpet + counter fade
ARROWS = (22.0, 2.0)        # force arrows grow
DRIFT = (26.0, 4.0, 6.0)    # theta_film 90 -> 96 deg
FORMULA = (26.0, 1.5)       # zeta(-1) formula fade-in
FADE = (29.0, 1.0)          # global fade to background

# palette (rtm-style-params)
GOLD = "#f2c66d"
GOLD_RGB = np.array([0.949, 0.776, 0.427], np.float32)
WHITE = "#fff6e8"
WHITE_RGB = np.array([1.000, 0.964, 0.910], np.float32)
BG_RGB = np.array([0.0275, 0.0431, 0.0784])

# lighting (fixed world light; plates turn under it)
STEEL = np.array([0.55, 0.60, 0.68], np.float32)
AMBIENT, DIFFUSE = 0.20, 0.95
SPEC, SPEC_N = 0.35, 8
RIM, RIM_N = 0.25, 4
LIGHT = np.array([0.55, 0.30, 0.78], np.float64)
LIGHT /= np.linalg.norm(LIGHT)
HALFV = VIEW + LIGHT
HALFV /= np.linalg.norm(HALFV)
SPEC_COL = np.array([1.0, 0.94, 0.82], np.float32)
RIM_COL = np.array([0.32, 0.80, 0.78], np.float32)

# ----------------------------------------------------------------- helpers
def smoothstep(x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def theta_film(t):
    th = 90.0 * smoothstep(t / ORBIT[1])
    if t > DRIFT[0]:
        th += DRIFT[2] * smoothstep((t - DRIFT[0]) / DRIFT[1])
    return th


def scale_of(t):
    s = 0.90 + 0.10 * smoothstep(t / SETTLE[1])
    if t >= ZOOMIN[0]:
        s = 1.0 + (S_MAX - 1.0) * smoothstep((t - ZOOMIN[0]) / ZOOMIN[1])
    if t >= ZOOMOUT[0]:
        s = S_MAX + (S_END - S_MAX) * smoothstep((t - ZOOMOUT[0]) / ZOOMOUT[1])
    return s


def n_of(t):
    """Modes currently in the sum."""
    return 1.0 + (N_MAX - 1.0) * smoothstep((t - CARPET[0]) / CARPET[1])


def carpet_fade(t):
    return 1.0 - smoothstep((t - WAVE_OUT[0]) / WAVE_OUT[1])


def project(p, t):
    """Base-coord point p -> (px, py) screen pixels, 1080p, camera fixed."""
    th = math.radians(theta_film(t))
    s = scale_of(t)
    q = (s * p) @ rot_z(-th).T
    d = 20.0 - (CE * q[0] + SE * q[2])
    sx = -q[1] * 20.0 / d          # screen right = -y at theta=0
    sy = (-SE * q[0] + CE * q[2]) * 20.0 / d
    return 960.0 + 135.0 * sx, 540.0 - 135.0 * sy


# ------------------------------------------------------------- static data
def build_plates():
    """Two square metal plates: 12 quads (6 faces each) + base normals."""
    x0, x1, h = GAP / 2.0, GAP / 2.0 + T, L / 2.0
    faces, norms = [], []
    for sgn in (+1.0, -1.0):
        a, b = x0, x1                       # inner .. outer along +x
        if sgn < 0.0:
            a, b = -x1, -x0
        faces.append(((b, -h, -h), (b, h, -h), (b, h, h), (b, -h, h)))   # outer
        norms.append((sgn, 0.0, 0.0))
        faces.append(((a, -h, h), (a, h, h), (a, h, -h), (a, -h, -h)))   # inner
        norms.append((-sgn, 0.0, 0.0))
        faces.append(((a, h, -h), (b, h, -h), (b, h, h), (a, h, h)))     # +y
        norms.append((0.0, 1.0, 0.0))
        faces.append(((a, -h, h), (b, -h, h), (b, -h, -h), (a, -h, -h)))  # -y
        norms.append((0.0, -1.0, 0.0))
        faces.append(((a, -h, h), (b, -h, h), (b, h, h), (a, h, h)))     # +z
        norms.append((0.0, 0.0, 1.0))
        faces.append(((a, h, -h), (b, h, -h), (b, -h, -h), (a, -h, -h)))  # -z
        norms.append((0.0, 0.0, -1.0))
    return (np.array(faces, dtype=np.float64),
            np.array(norms, dtype=np.float64))


QUADS, NORMS = build_plates()


def shade_face(n):
    ndl = max(float(n @ LIGHT), 0.0)
    ndh = max(float(n @ HALFV), 0.0)
    rim = min(max(1.0 - float(n @ VIEW), 0.0), 1.0) ** RIM_N
    c = (STEEL * (AMBIENT + DIFFUSE * ndl ** 0.85)
         + SPEC_COL * (SPEC * ndh ** SPEC_N)
         + RIM_COL * (RIM * rim))
    return np.clip(c, 0.0, 1.0)


def build_arrows():
    """Arrow quads as affine-in-ell: quads(ell) = Q0 + ell * DQ.

    Left arrow: base at x=-GAP/2 pointing +x.  Right arrow: mirrored.
    Each arrow = shaft quad + head (two degenerate-quad triangles).
    """
    q0, dq = [], []
    for sgn in (+1.0, -1.0):
        bx = sgn * GAP / 2.0
        dx = -sgn                      # force arrows point INTO the gap (attraction)

        def add(cx, y, z, dxc):
            q0.append((bx, y, ARROW_Z + z))
            dq.append((dxc * dx, 0.0, 0.0))

        # shaft: x from base to base+ARROW_L
        add(0.0, -ARROW_WY, -ARROW_SW, 0.0)
        add(ARROW_L, -ARROW_WY, -ARROW_SW, ARROW_L)
        add(ARROW_L, ARROW_WY, ARROW_SW, ARROW_L)
        add(0.0, ARROW_WY, ARROW_SW, 0.0)
        # head: base line at base+ARROW_L spanning +-ARROW_HW, tip ahead
        add(ARROW_L, -ARROW_WY, -ARROW_HW, ARROW_L)
        add(ARROW_L, -ARROW_WY, ARROW_HW, ARROW_L)
        add(ARROW_L + ARROW_H, 0.0, 0.0, ARROW_L + ARROW_H)
        add(ARROW_L + ARROW_H, 0.0, 0.0, ARROW_L + ARROW_H)
        add(ARROW_L, ARROW_WY, -ARROW_HW, ARROW_L)
        add(ARROW_L, ARROW_WY, ARROW_HW, ARROW_L)
        add(ARROW_L + ARROW_H, 0.0, 0.0, ARROW_L + ARROW_H)
        add(ARROW_L + ARROW_H, 0.0, 0.0, ARROW_L + ARROW_H)
    return (np.array(q0, dtype=np.float64).reshape(-1, 4, 3),
            np.array(dq, dtype=np.float64).reshape(-1, 4, 3))


ARROW_Q0, ARROW_DQ = build_arrows()

GOLD_QUAD = np.tile(np.append(GOLD_RGB, 1.0), (6, 4, 1)).astype(np.float32)
BLACK_QUAD = np.zeros((6, 4, 4), dtype=np.float32)
BLACK_QUAD[..., 3] = 1.0
OFF_Q = np.zeros((6, 4, 3), dtype=np.float64)
OFF_Q[..., 0] = 100.0               # far off-frame when arrows are hidden


def build_carpet():
    """Row n = mode n: a strip of NZQ quads at z_n, brightness |sin(n pi zeta)|.

    z_n = DZ * (-1)^(n+1) * floor(n/2) — n=1 centre, alternating up/down,
    so the carpet grows outward chronologically (lower terms first).
    """
    car = np.empty((N_MAX, NZQ, 4, 3), dtype=np.float64)
    node = np.empty((N_MAX, NZQ), dtype=np.float64)
    ramp = np.empty((N_MAX, 3), dtype=np.float32)
    xs = np.linspace(-WAVE_HALF, WAVE_HALF, NZQ + 1)
    zc = (np.arange(NZQ) + 0.5) / NZQ          # quad-centre zeta
    for n in range(1, N_MAX + 1):
        z = DZ * ((-1) ** (n + 1)) * (n // 2)
        u = (n - 1) / (N_MAX - 1)
        ramp[n - 1] = GOLD_RGB * (1.0 - u) + WHITE_RGB * u
        node[n - 1] = np.abs(np.sin(n * np.pi * zc))
        for j in range(NZQ):
            car[n - 1, j] = ((xs[j], 0.0, z - HZ), (xs[j + 1], 0.0, z - HZ),
                             (xs[j + 1], 0.0, z + HZ), (xs[j], 0.0, z + HZ))
    return car, node, ramp


CARPET_Q, NODE, RAMP = build_carpet()
NC = N_MAX * NZQ
RN = np.repeat(np.arange(1, N_MAX + 1), NZQ).astype(np.float64)  # row index per quad

NQ = 12 + NC + 6                # plates + carpet + arrows
assert NQ * 6 <= MeshBatch.MAX_MESH_VERTS, "mesh over the native vertex cap"


def counter_strings():
    """Precomputed TypstMath sources for the running mode sum (4 Hz ticks)."""
    out = []
    k = 0
    while True:
        t = CARPET[0] + k * 0.25
        n = int(round(n_of(t)))
        s = n * (n + 1) // 2
        if n <= 1:
            txt = "1 = 1"
        elif n == 2:
            txt = "1 + 2 = 3"
        else:
            txt = "1 + 2 + dots.c + %d = %d" % (n, s)
        out.append(txt)
        if t >= CARPET[0] + CARPET[1]:
            break
        k += 1
    return out


# ------------------------------------------------------------ master anim
class CasimirMovie(Animation):
    """Per-frame closed-form state (pure function of t — seek-safe)."""

    def __init__(self, write_state, carrier, duration=DURATION):
        self.write_state = write_state
        super().__init__(carrier, run_time=duration, rate_func=linear)

    def interpolate_mobject(self, alpha):
        self.write_state(alpha * self.run_time)


class CasimirScene(ThreeDScene):
    def construct(self):
        win = MLWindow(960, 540)
        win.scene = self
        self.set_camera_orientation(phi=PHI, theta=0.0)   # camera fixed forever

        mesh = MeshBatch()
        mesh.quads = np.empty((NQ, 4, 3), dtype=np.float64)
        mesh.colors = np.empty((NQ, 4, 4), dtype=np.float32)
        win.set_mesh(mesh)

        labels = []
        for txt in counter_strings():
            lab = TypstMath(txt, font_size=38).set_color(GOLD)
            lab.set_opacity(0.0)
            lab.to_corner([-1, -1, 0], buff=0.62)
            labels.append(lab)

        formula = TypstMath(
            "zeta(-1) = 1 + 2 + 3 + dots.c = -frac(1, 12)",
            font_size=40,
        ).set_color(GOLD).set_opacity(0.0).to_corner([1, -1, 0], buff=0.62)

        fade = Rectangle(width=14.6, height=8.6, stroke_width=0.0)
        fade.set_fill("#070b14", 0.0)

        self.add_fixed_in_frame_mobjects(*labels, formula, fade)

        def write_state(t):
            th = math.radians(theta_film(t))
            s = scale_of(t)
            R = rot_z(-th)

            mesh.quads[:12] = (s * QUADS) @ R.T
            N = NORMS @ R.T
            cols = np.empty((12, 4, 4), dtype=np.float32)
            for i in range(12):
                c = shade_face(N[i])
                cols[i] = (c[0], c[1], c[2], 1.0)
            mesh.colors[:12] = cols

            # carpet: geometry static in base coords, colours carry the waves
            mesh.quads[12:12 + NC] = (s * CARPET_Q.reshape(NC, 4, 3)) @ R.T
            n_now = n_of(t)
            breath = 0.75 + 0.25 * np.cos(RN * OMEGA0 * (t - CARPET[0]))
            active = (RN <= n_now).astype(np.float32)
            fac = ((0.30 + 0.70 * NODE.reshape(NC)) * breath * active
                   * carpet_fade(t))
            ccol = np.empty((NC, 4, 4), dtype=np.float32)
            rgb = RAMP[(RN - 1).astype(int)] * fac[:, None]      # (NC, 3)
            ccol[..., :3] = rgb[:, None, :]                      # 4 corners share
            ccol[..., 3] = 1.0
            mesh.colors[12:12 + NC] = ccol

            ell = smoothstep((t - ARROWS[0]) / ARROWS[1])   # growth factor 0 -> 1
            if ell > 1e-4:
                mesh.quads[12 + NC:] = (s * (ARROW_Q0 + ell * ARROW_DQ)) @ R.T
                mesh.colors[12 + NC:] = GOLD_QUAD
            else:
                mesh.quads[12 + NC:] = OFF_Q
                mesh.colors[12 + NC:] = BLACK_QUAD

            fade_w = carpet_fade(t)
            if t < CARPET[0]:
                k = -1
            else:
                k = min(len(labels) - 1, int((t - CARPET[0]) * 4.0))
            for i, lab in enumerate(labels):
                lab.set_opacity(fade_w if i == k else 0.0)

            formula.set_opacity(smoothstep((t - FORMULA[0]) / FORMULA[1]))
            fade.set_fill(opacity=smoothstep((t - FADE[0]) / FADE[1]))

        write_state(0.0)
        win.play(CasimirMovie(write_state, VGroup(), DURATION))
        win.close()


# ------------------------------------------------------------------- verify
def verify():
    print("mode carpet: %d modes, %d quads/row" % (N_MAX, NZQ))
    print("  row spacing %.4f, row half-height %.3f, rows fill z in +-%.2f" % (
        DZ, HZ, DZ * (N_MAX // 2)))
    print("  freq range %.2f .. %.1f rad/s (mode 1 .. %d)" % (
        OMEGA0, N_MAX * OMEGA0, N_MAX))
    print("standing-wave nodes pinned: |sin(n pi zeta)| = 0 at both walls, by construction")
    print("running sum at 300 modes: 1+2+...+300 = %d" % (300 * 301 // 2))
    print("regularized mode sum  sum_n n e^{-eps n} = e^{-eps}/(1-e^{-eps})^2 = 1/eps^2 - 1/12 + O(eps^2):")
    for eps in (1e-2, 1e-3, 1e-4, 1e-5):
        e = math.exp(-eps)
        S = e / (1.0 - e) ** 2          # closed form, no truncation
        print("  eps=%.0e   S - 1/eps^2 = %+.6f   (target -1/12 = -0.083333)" % (eps, S - 1.0 / eps ** 2))
    print("mesh quads: %d  verts: %d  native cap: %d" % (NQ, NQ * 6, MeshBatch.MAX_MESH_VERTS))
    print("arrow tips stay inside the gap: reach %.2f < %.2f" % (
        ARROW_L + ARROW_H, GAP / 2.0))
    print("arrow row %.2f clears the top carpet row %.3f" % (
        ARROW_Z, DZ * (N_MAX // 2) + HZ))
    print("projected landmarks (px):")
    for t in (0.5, 2.0, 5.0, 7.0, 9.0, 11.0, 14.0, 17.0, 20.0, 22.0, 24.0, 26.0, 27.5, 29.5):
        lo = project(np.array([-1.1, 0.0, 0.0]), t)[0]
        li = project(np.array([-0.55, 0.0, 0.0]), t)[0]
        ri = project(np.array([0.55, 0.0, 0.0]), t)[0]
        ro = project(np.array([1.1, 0.0, 0.0]), t)[0]
        print("  t=%5.2f  plate [%6.1f %6.1f]  slit [%6.1f %6.1f]  s=%.3f  th=%.2f  N=%d" % (
            t, lo, ro, li, ri, scale_of(t), theta_film(t), int(n_of(t))))


# ------------------------------------------------------------------- render
def render(out_path="out/casimir.mp4"):
    out = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    t0 = time.time()
    res = fast_record_scene(CasimirScene, out, overwrite=True)
    print("[render] %.1f s -> %s" % (time.time() - t0, res.get("files")))
    return res


def stills(times, out_dir="out/stills"):
    from real_time_manim.frames import FrameServer
    os.makedirs(out_dir, exist_ok=True)
    srv = FrameServer(CasimirScene)
    for t in times:
        p = os.path.join(out_dir, "cas_t%06.2f.png" % t)
        srv.save_frame(t, p)
        print("[still] t=%.2f -> %s" % (t, p))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "render"
    if mode == "verify":
        verify()
    elif mode == "stills":
        stills([float(x) for x in sys.argv[2:]] or [0.0])
    else:
        render()
