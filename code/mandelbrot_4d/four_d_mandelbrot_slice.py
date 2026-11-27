#!/usr/bin/env python3
r"""4D Mandelbrot set sliced by a moving hyperplane — RTM scene, no HUD.

Geometry
--------
The quaternion Mandelbrot set  M_4 = {(a, q) : q in R^3, |q| <= R(a)}  is the
solid of revolution of the 2D set around the real axis (R(a) = half-height of
the 2D set at real coordinate a).  A hyperplane  n.p = h  cuts it; the cut is
always a TUBE inside the hyperplane: a solid of revolution about the line
    ell = H ∩ span(e_a, u),    u = n_im / |n_im|,
with profile  rho(a) = sqrt(R(a)^2 - t(a)^2),  t(a) = (h - a n_a)/|n_im|.
The boundary is the lathe surface over a in A = {a : R(a) >= |t(a)|} — exact
formula, no marching cubes.

Story (33 s, 60 fps)
--------------------
1. [0, 8]   hyperplane far away (h = 0.75), slides in along its normal until it
            passes through (0,0,0,0) at h = 0.  The slice first appears on the
            cardioid's shoulder (~h = 0.59), grows and settles centred on the
            origin.
2. [8, 26]  pivot the hyperplane about the origin through three 4D rotations,
            each a simple rotation with a fixed axis 2-plane P_j = W_j^perp.
            Every P_j is generic: not a coordinate plane, not inside the
            hyperplane, not perpendicular to it (P_j ∩ H is a line).  The axes
            are as mutually perpendicular as the geometry allows: W2 = W1^perp
            exactly (P2 = W1, so P1 ⊥ P2 at 90 deg), and W3 is chosen to be as
            orthogonal as possible to both.  The slice morphs: blob -> full
            vase with period-2 bulb + antenna spike (k -> 0) -> bulb separates
            at the pinch on the mirrored side (k ~ -0.21) and re-merges ->
            collapse back to the cardioid ball (k -> 1.8).  Camera orbits 150
            deg meanwhile, always pivoting about the 4D origin (0,0,0,0);
            auto-zoom frames the slice around that fixed pivot.
3. [26, 31] camera frozen; the hyperplane moves back out (h -> 0.75); the
            slice shrinks, slides off and vanishes (~29 s); hold to 33 s.

Usage
-----
    python four_d_mandelbrot_slice.py --probe          # trajectory table, no render
    python four_d_mandelbrot_slice.py --still 5,9,14,17,23,26,28,30  # PNGs
    python four_d_mandelbrot_slice.py --jobs 4         # -> out mp4
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

_RTM = os.path.expanduser("~/PHormation codes/real-time-manim")
if _RTM not in sys.path:
    sys.path.insert(0, _RTM)

import real_time_manim  # noqa: F401  (must precede any manim import)
from real_time_manim import DEGREES, Animation, ThreeDScene, VMobject, linear
from real_time_manim.mesh import MeshBatch
from real_time_manim.record import fast_record_scene
from real_time_manim.vulkan_bind import MLWindow

# ---------------------------------------------------------------------------
# look
# ---------------------------------------------------------------------------
BG_RGB = np.array([0.0275, 0.0431, 0.0784])

SURFACE_PALETTE = np.array(
    [
        [0.020, 0.055, 0.115],
        [0.040, 0.180, 0.280],
        [0.050, 0.420, 0.520],
        [0.220, 0.720, 0.720],
        [0.950, 0.760, 0.380],
        [1.000, 0.940, 0.820],
    ]
)
WARM = np.array([1.0, 0.94, 0.82]) * 0.30  # unused (matte shading)
COOL = np.array([0.32, 0.80, 0.78]) * 0.30

# light fixed in scene space (rides the slice, like a studio light)
LIGHT = np.array([-0.72, -0.28, 0.63])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

# ---------------------------------------------------------------------------
# timeline constants
# ---------------------------------------------------------------------------
T1, T2, T3, TEND = 8.0, 26.0, 31.0, 33.0     # phase boundaries (s)
H_START, H_END = 0.75, 0.75                   # hyperplane offset, far sides
SEG_LEN = 6.0                                # each rotation segment
K_TARGET_3 = 1.8                             # k target for S3 (collapse)

# initial hyperplane normal (tilted; k0 = n_a/|n_im| ~ 1.18)
N0 = np.array([0.75, 0.40, 0.35, 0.35])
N0 = N0 / np.linalg.norm(N0)

# frame anchors for the scene basis (Gram-Schmidt against n); the two used
# are picked at build time for best conditioning over the whole timeline
F_CANDIDATES = [
    np.array([0.35, -0.30, 0.78, -0.40]),
    np.array([-0.44, 0.62, 0.21, 0.60]),
    np.array([0.58, 0.41, -0.55, 0.44]),
    np.array([0.21, 0.83, -0.36, 0.37]),
    np.array([-0.63, -0.21, 0.55, 0.50]),
    np.array([1.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 1.0]),
]


def pick_anchors(sol):
    """Choose the two frame anchors best conditioned over the whole film."""
    ts = np.arange(0.0, TEND + 0.01, 0.05)
    ns = [timeline(float(t), sol)[1] for t in ts]
    best = None
    for i in range(len(F_CANDIDATES)):
        for j in range(i + 1, len(F_CANDIDATES)):
            a1, a2 = F_CANDIDATES[i], F_CANDIDATES[j]
            if abs(float(a1 @ a2)) > 0.9:
                continue
            score = min(
                min(abs(float(n @ a1)), abs(float(n @ a2))) for n in ns
            )
            if best is None or score > best[0]:
                best = (score, a1, a2)
    return best[1], best[2], best[0]

# camera
FRAME_H = 8.0          # world height visible at zoom 1 (measured: 8.00/z exactly)
FRAME_W = 14.22
Z_MIN, Z_MAX = 0.8, 4.5
ZOOM_TAU = 2.5         # s, auto-zoom smoothing (slow glide: no shimmer)
Z_INIT = 1.8

# mesh resolution
N_PROF = 520           # profile samples per component
N_AZ = 72              # azimuth samples (fine facets: no banding/moire)
A0, A1, N_A_TEST = -2.05, 0.39, 2000

# mandelbrot field window (same as the reference sample; MAXIT high so the
# slow-escape halo above the cardioid stays sub-pixel; 1600 px so the small
# bulbs along the antenna survive in the profile)
CX, CY = -0.765, 0.0
SPAN_X = 1.295
IM_HALF = 1.18
IMG_PX_W = 1600
IMG_PX_H = int(round(IMG_PX_W * 2.0 * IM_HALF / (2.0 * SPAN_X)))
MAXIT = 2500

# ---------------------------------------------------------------------------
# small maths helpers
# ---------------------------------------------------------------------------
EPS4 = np.zeros((4, 4, 4, 4))
for _i in range(4):
    for _j in range(4):
        for _k in range(4):
            for _l in range(4):
                _s = 0
                if len({_i, _j, _k, _l}) == 4:
                    _p = [_i, _j, _k, _l]
                    _s = 1
                    for _x in range(4):
                        for _y in range(_x + 1, 4):
                            if _p[_y] < _p[_x]:
                                _s = -_s
                                _p[_x], _p[_y] = _p[_y], _p[_x]
                EPS4[_i, _j, _k, _l] = _s


def cross4(a, b, c):
    return np.einsum("ijkl,j,k,l->i", EPS4, a, b, c)


def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def palette(t):
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0) * (len(SURFACE_PALETTE) - 1)
    i = np.clip(t.astype(int), 0, len(SURFACE_PALETTE) - 2)
    f = (t - i)[..., None]
    return SURFACE_PALETTE[i] * (1.0 - f) + SURFACE_PALETTE[i + 1] * f


def rot_plane(v, g1, g2, th):
    """Simple 4D rotation: rotate the plane W = span(g1, g2) by th; axis W^perp."""
    c, s = math.cos(th), math.sin(th)
    a1 = float(v @ g1)
    a2 = float(v @ g2)
    return v + (c - 1.0) * (a1 * g1 + a2 * g2) + s * (a2 * g1 - a1 * g2)


def k_of(n):
    n_m = float(np.linalg.norm(n[1:]))
    return float(n[0] / n_m) if n_m > 1e-9 else 1e9


def frame_of(n, f1, f2):
    """Orthonormal scene basis (e1, e2, e3) of n^perp, 4x3 columns."""
    e1 = f1 - float(f1 @ n) * n
    e1 = e1 / np.linalg.norm(e1)
    e2 = f2 - float(f2 @ n) * n - float(f2 @ e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    e3 = cross4(n, e1, e2)
    return np.stack([e1, e2, e3], axis=1)


# ---------------------------------------------------------------------------
# R(a): half-height profile of the 2D Mandelbrot set
# ---------------------------------------------------------------------------
def mandel_field():
    xs = np.linspace(CX - SPAN_X, CX + SPAN_X, IMG_PX_W)
    ys = np.linspace(CY - IM_HALF, CY + IM_HALF, IMG_PX_H)
    re, im = np.meshgrid(xs, ys)
    zr = np.zeros_like(re)
    zi = np.zeros_like(im)
    esc = np.full(re.shape, float(MAXIT))
    alive = np.ones(re.shape, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(1, MAXIT + 1):
            zr, zi = zr * zr - zi * zi + re, 2.0 * zr * zi + im
            m2 = zr * zr + zi * zi
            newly = alive & (m2 > 256.0)
            if newly.any():
                esc[newly] = n + 1.0 - np.log2(np.log2(m2[newly]) * 0.5)
                alive &= ~newly
            if not alive.any():
                break
    return xs, ys, esc


def radius_profile():
    xs, ys, esc = mandel_field()
    mask = esc >= MAXIT * 0.998
    raw = np.zeros(len(xs))
    for j in range(len(xs)):
        if mask[:, j].any():
            raw[j] = float(ys[mask[:, j]].max())
    # smooth the profile but never let smoothing wash out the pinch points
    # (the two cusps at a = -0.75 and a = -1.25 where R must vanish)
    R = raw.copy()
    for _ in range(3):
        k = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        k /= k.sum()
        q = R.copy()
        q[2:-2] = (
            k[0] * R[:-4] + k[1] * R[1:-3] + k[2] * R[2:-2]
            + k[3] * R[3:-1] + k[4] * R[4:]
        )
        R = q
    R = np.minimum(R, raw + 0.004)
    R = np.maximum(R, 0.0)
    return xs, R


# ---------------------------------------------------------------------------
# solve the three rotation angles (deterministic)
# ---------------------------------------------------------------------------
def _orthonormalize(ga, gb):
    gb = gb - float(gb @ ga) * ga
    gb = gb / np.linalg.norm(gb)
    ga = ga / np.linalg.norm(ga)
    return ga, gb


def solve_rotations():
    """Chain-solve theta_max for the three segments; return (ths, N1, N2, N3).

    Each segment j is a simple 4D rotation about the axis plane P_j = W_j^perp,
    W_j = span(g_a, g_b).  The axis must stay generic w.r.t. the hyperplane
    (P_j not in H, not perpendicular to H), so the starting normal keeps a
    component outside W_j (|P| = |n_s - proj_W n_s| >= 0.15) while the circle
    n(theta) must reach the segment's target k.  Candidates are tried and the
    best accepted one is printed.
    """
    e_a = np.array([1.0, 0.0, 0.0, 0.0])
    v_anc = np.array([0.0, 0.55, -0.62, 0.56])
    v_anc = v_anc / np.linalg.norm(v_anc)
    scan = np.radians(np.linspace(6.0, 165.0, 800))

    def u_of(n):
        u = np.array([0.0, n[1], n[2], n[3]])
        return u / np.linalg.norm(u)

    def candidates(n_s, vcoefs=(0.35,), cs=(0.15, 0.30, 0.50, 0.80),
                   gb_anchors=((0.24, -0.51, 0.68, 0.47),)):
        u_s = u_of(n_s)
        v_s = v_anc - float(v_anc @ u_s) * u_s - float(v_anc @ e_a) * e_a
        v_s = v_s / np.linalg.norm(v_s)
        for c in cs:
            for vc in vcoefs:
                for sgn in (+1.0, -1.0):
                    ga = sgn * e_a + c * u_s + vc * v_s
                    ga = ga / np.linalg.norm(ga)
                    for anchor in gb_anchors:
                        gb = np.array(anchor)
                        ga2, gb2 = _orthonormalize(ga, gb)
                        yield ga2, gb2

    def plane_angle_cos(A, B):
        """cos of the smallest principal angle between planes A, B (4x2 bases)."""
        s = np.linalg.svd(np.asarray(A).T @ np.asarray(B), compute_uv=False)
        return float(s[0])

    # -- segment 1: N0 -> min |n_a| (slice opens into the full vase) ---------
    # W1 tilted off span(e_a, u0) (c = 0.8): e_a split ~0.73/0.69 between W1
    # and W2 = W1^perp, balancing the two perpendicular axes' e_a-reach.
    n_s = N0
    best = None
    for ga, gb in candidates(n_s, cs=(1.0,)):
        P = n_s - (n_s @ ga) * ga - (n_s @ gb) * gb
        if np.linalg.norm(P) < 0.15:
            continue
        nas = np.abs(np.array([rot_plane(n_s, ga, gb, th)[0] for th in scan]))
        i = int(np.argmin(nas))
        if nas[i] < 0.06 and (best is None or nas[i] < best[0]):
            best = (float(nas[i]), ga, gb, float(scan[i]))
    assert best is not None, "S1: no candidate rotation reaches k ~ 0"
    (_, g1a, g1b, th1) = best
    N1 = rot_plane(n_s, g1a, g1b, th1)

    # -- segment 2: rotate in W2 = W1^perp EXACTLY ---------------------------
    # Axis P2 = W1: perfectly orthogonal to P1 = W1^perp.  The circle swings n
    # to the negative-real side (k ~ -0.21): same profile family as +k, so the
    # slice shows the bulb separating at the pinch while the sweep direction
    # is fully perpendicular to segment 1.
    P0 = N1 - (N1 @ g1a) * g1a - (N1 @ g1b) * g1b
    phat = P0 / np.linalg.norm(P0)
    r1 = e_a - (e_a @ g1a) * g1a - (e_a @ g1b) * g1b
    r1 = r1 / np.linalg.norm(r1)
    # orthonormalise the W2 basis (p_hat and r1 are NOT orthogonal in general)
    r1 = r1 - float(r1 @ phat) * phat
    r1 = r1 / np.linalg.norm(r1)
    g2a, g2b = phat, r1
    ks2 = np.array([abs(k_of(rot_plane(N1, g2a, g2b, th))) for th in scan])
    i2 = int(np.argmax(ks2))
    assert ks2[i2] > 0.12, \
        "S2: perpendicular sweep too shallow (max |k| = %.3f)" % ks2[i2]
    th2 = float(scan[i2])
    N2 = rot_plane(N1, g2a, g2b, th2)

    # -- segment 3: N1 -> k = K_TARGET_3 (S2's out-and-back returns to N1) ----
    # Among candidates that reach k = 3, pick the plane most orthogonal to
    # both W1 and W2 (minimize the max cosine of the smallest principal angle).
    n_s = N1
    B1 = np.stack([g1a, g1b], axis=1)
    B2 = np.stack([g2a, g2b], axis=1)
    best = None
    rng = np.random.default_rng(7)
    extra_anchors = rng.normal(size=(40, 4))
    extra_anchors /= np.linalg.norm(extra_anchors, axis=1, keepdims=True)
    gb_anchors = ((0.24, -0.51, 0.68, 0.47), (-0.33, 0.71, -0.44, 0.44),
                  (0.55, 0.38, -0.62, -0.42), (0.17, 0.28, 0.79, 0.52),
                  tuple(r1), tuple(phat)) + tuple(tuple(a) for a in extra_anchors)
    for ga, gb in candidates(
        n_s,
        vcoefs=(0.12, 0.35),
        cs=(0.15, 0.30, 0.50, 0.80, 1.20),
        gb_anchors=gb_anchors,
    ):
        P = n_s - (n_s @ ga) * ga - (n_s @ gb) * gb
        if np.linalg.norm(P) < 0.15:
            continue
        ks = np.array([k_of(rot_plane(n_s, ga, gb, th)) for th in scan])
        err = np.abs(ks - K_TARGET_3)
        i = int(np.argmin(err))
        if err[i] < 0.06:
            B3 = np.stack([ga, gb], axis=1)
            orth = max(plane_angle_cos(B3, B1), plane_angle_cos(B3, B2))
            if best is None or orth < best[0]:
                best = (orth, ga, gb, float(scan[i]))
    assert best is not None, "S3: no candidate rotation reaches k = %.1f" % K_TARGET_3
    (_, g3a, g3b, th3) = best
    N3 = rot_plane(n_s, g3a, g3b, th3)

    pairs = [(g1a, g1b), (g2a, g2b), (g3a, g3b)]
    return (th1, th2, th3), N1, N2, N3, pairs


# ---------------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------------
def timeline(t, sol):
    ths, N1, N2, N3, pairs = sol
    th1, th2, th3 = ths
    if t < T1:
        return H_START * (1.0 - smoothstep(t / T1)), N0, 0, "approach"
    if t < T2:
        u = (t - T1) / SEG_LEN
        if u < 1.0:
            th = th1 * smoothstep(u)
            n = rot_plane(N0, *pairs[0], th)
            seg = 1
        elif u < 2.0:
            v = u - 1.0
            th = th2 * (math.sin(math.pi * v)) ** 1.15
            n = rot_plane(N1, *pairs[1], th)
            seg = 2
        else:
            v = u - 2.0
            th = th3 * smoothstep(v)
            n = rot_plane(N1, *pairs[2], th)
            seg = 3
        return 0.0, n, seg, "rotate"
    if t < T3:
        return H_END * smoothstep((t - T2) / (T3 - T2)), N3, 0, "retreat"
    return H_END, N3, 0, "hold"


def camera_at(t):
    """(theta, phi) in degrees; zoom handled separately (auto)."""
    if t < T1:
        u = smoothstep(t / T1)
        return -90.0 + 12.0 * u, 52.0 + 12.0 * u
    if t < T2:
        u = smoothstep((t - T1) / (T2 - T1))
        return -78.0 + 150.0 * u, 64.0 - 12.0 * u
    return 72.0, 52.0


# ---------------------------------------------------------------------------
# the slice tube (one mesh, rebuilt per frame)
# ---------------------------------------------------------------------------
class SliceTube:
    def __init__(self, xs, R, anchors):
        self.xs = xs
        self.R = R
        self.f1, self.f2 = anchors
        self.a_test = np.linspace(A0, A1, N_A_TEST)
        self.da = self.a_test[1] - self.a_test[0]
        self.phi = 2.0 * np.pi * np.arange(N_AZ) / N_AZ
        self.cos_phi = np.cos(self.phi)
        self.sin_phi = np.sin(self.phi)
        # w1 anchor in bcd space
        self.g_bcd = np.array([0.0, 0.71, -0.48, 0.51])
        self.g_bcd /= np.linalg.norm(self.g_bcd)

    def build(self, n, h, cam_dir):
        """Return (quads, colors) or None when the slice is empty."""
        n_a = float(n[0])
        n_im = n[1:]
        n_m = float(np.linalg.norm(n_im))
        if n_m < 1e-9:
            return None
        u = np.zeros(4)
        u[1:] = n_im / n_m
        k = n_a / n_m

        E = frame_of(n, self.f1, self.f2)
        d = E.T @ np.array([1.0, 0.0, 0.0, 0.0]) - k * (E.T @ u)
        dn = float(np.linalg.norm(d))
        dhat = d / dn
        s0 = (h / n_m) * (E.T @ u)

        # circle plane basis (w1, w2 in 4D; W1, W2 in scene)
        w1 = self.g_bcd - float(self.g_bcd @ u) * u
        m1 = float(np.linalg.norm(w1))
        if m1 < 1e-4:
            g2 = np.array([0.0, 0.0, 1.0, 0.0])
            w1 = g2 - float(g2 @ u) * u
            w1 = w1 / np.linalg.norm(w1)
        else:
            w1 = w1 / m1
        w2 = np.zeros(4)
        w2[1:] = np.cross(u[1:], w1[1:])
        W1 = E.T @ w1
        W2 = E.T @ w2

        R_test = np.interp(self.a_test, self.xs, self.R, left=0.0, right=0.0)
        t_test = (h - self.a_test * n_a) / n_m
        rho2 = R_test * R_test - t_test * t_test
        ok = rho2 > 1e-8

        # contiguous runs
        runs = []
        i = 0
        while i < len(ok):
            if not ok[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if j - i + 1 >= 3:
                runs.append((i, j))
            i = j + 1

        # budget: profile rows proportional to each run's a-span, and a hard
        # quad cap that keeps the widest runs first (tiny antenna slivers
        # yield before the main body - RTM's native mesh holds 524288 verts)
        quads_budget = 125_000
        planned = []
        for (i0, i1) in runs:
            lo = i0 - 1 if i0 > 0 else i0
            hi = i1 + 1 if i1 < len(ok) - 1 else i1
            span = self.a_test[hi] - self.a_test[lo]
            n_prof = max(10, min(N_PROF, int(round(N_PROF * span / 0.5))))
            planned.append((lo, hi, span, n_prof))
        planned.sort(key=lambda x: -x[2])
        kept = []
        tot = 0
        for (lo, hi, span, n_prof) in planned:
            if tot + n_prof * N_AZ > quads_budget:
                continue
            tot += n_prof * N_AZ
            kept.append((lo, hi, n_prof))
        kept.sort(key=lambda x: x[0])

        quads_all = []
        cols_all = []
        verts_all = []
        for (lo, hi, n_prof) in kept:
            a_k = np.linspace(self.a_test[lo], self.a_test[hi], n_prof)
            Rk = np.interp(a_k, self.xs, self.R, left=0.0, right=0.0)
            t_k = (h - a_k * n_a) / n_m
            rho = np.sqrt(np.clip(Rk * Rk - t_k * t_k, 0.0, None))
            rhop = np.gradient(rho, a_k)

            V = (
                s0[None, None, :]
                + a_k[:, None, None] * d[None, None, :]
                + rho[:, None, None]
                * (self.cos_phi[None, :, None] * W1[None, None, :]
                   + self.sin_phi[None, :, None] * W2[None, None, :])
            )

            e_phi = (
                self.cos_phi[None, :, None] * W1[None, None, :]
                + self.sin_phi[None, :, None] * W2[None, None, :]
            )
            nrm = dn * e_phi - rhop[:, None, None] * dhat[None, None, :]
            nrm = nrm / (np.linalg.norm(nrm, axis=2, keepdims=True) + 1e-12)

            frac = np.clip((a_k + 2.0) / 2.25, 0.0, 1.0)
            base = palette(0.10 + 0.85 * frac ** 0.9)

            ndl = np.clip(np.einsum("ijk,k->ij", nrm, LIGHT), 0.0, 1.0)
            ndv = np.clip(np.einsum("ijk,k->ij", nrm, cam_dir), 0.0, 1.0)

            # matte shading: diffuse + a soft rim only (no specular - the
            # scene-fixed light made it slide/jump across the surface while
            # the camera orbited)
            col = base[:, None, :] * (0.20 + 0.95 * ndl ** 0.85)[..., None]
            col = col + ((1.0 - ndv) ** 4.0)[..., None] * COOL
            col = np.clip(col, 0.0, 1.0)
            col = np.concatenate([col, np.ones(col.shape[:2] + (1,))], axis=2)

            i_idx = np.arange(n_prof - 1)[:, None]
            j_idx = np.arange(N_AZ)[None, :]
            jn = (j_idx + 1) % N_AZ
            q = np.stack(
                [V[i_idx, j_idx], V[i_idx, jn], V[i_idx + 1, jn], V[i_idx + 1, j_idx]],
                axis=2,
            ).reshape(-1, 4, 3)
            c = np.stack(
                [col[i_idx, j_idx], col[i_idx, jn], col[i_idx + 1, jn], col[i_idx + 1, j_idx]],
                axis=2,
            ).reshape(-1, 4, 4)

            # open ends (interval runs into the profile's domain edge): cap fan
            jj = np.arange(N_AZ)
            jn1 = (jj + 1) % N_AZ
            for row, ring in ((0, V[0]), (n_prof - 1, V[-1])):
                if rho[row] > 0.004:
                    cen = ring[0]
                    cap_q = np.stack(
                        [np.broadcast_to(cen, (N_AZ, 3)), ring[jn1], ring[jj],
                         np.broadcast_to(cen, (N_AZ, 3))],
                        axis=1,
                    ).reshape(-1, 4, 3)
                    cc = np.broadcast_to(col[row, 0], (N_AZ, 4, 4))
                    q = np.vstack([q, cap_q])
                    c = np.vstack([c, cc])

            quads_all.append(q)
            cols_all.append(c)
            verts_all.append(V.reshape(-1, 3))

        if not quads_all:
            return None
        quads = np.vstack(quads_all)
        cols = np.vstack(cols_all)
        verts = np.vstack(verts_all)
        return quads, cols.astype(np.float32), verts


# ---------------------------------------------------------------------------
# animation
# ---------------------------------------------------------------------------
def build_zoom_trajectory(camera, tube, sol):
    """Zoom as a pure function of t.  The camera always rotates about the 4D
    origin (0,0,0,0), so the auto-framing measures the extent around the
    origin; the zoom freezes at T2.  Frame-exact in stills, seeks and the
    recorded video alike."""
    tg = np.arange(0.0, T2 + 0.025, 0.05)
    targets = np.full(len(tg), np.nan)
    for i, t in enumerate(tg):
        h, n, seg, phase = timeline(float(t), sol)
        theta, phi = camera_at(float(t))
        camera.set_theta(np.radians(theta))
        camera.set_phi(np.radians(phi))
        rot = np.asarray(camera.generate_rotation_matrix(), dtype=float)
        cam_dir = rot.T @ np.array([0.0, 0.0, 1.0])
        built = tube.build(n, h, cam_dir)
        if built is None:
            continue
        proj = (rot @ built[2].T).T
        e_y = float(np.abs(proj[:, 1]).max())
        e_x = float(np.abs(proj[:, 0]).max()) * (FRAME_H / FRAME_W)
        e = max(e_y, e_x)
        targets[i] = float(np.clip(FRAME_H / (2.3 * e + 0.7), Z_MIN, Z_MAX))
    # forward-fill empty stretches
    last = Z_INIT
    for i in range(len(targets)):
        if np.isnan(targets[i]):
            targets[i] = last
        else:
            last = targets[i]
    # 60 fps exponential smoothing recursion over the 20 Hz curve
    tf = np.arange(0.0, T2 + 1.0 / 120.0, 1.0 / 60.0)
    zs = np.empty(len(tf))
    zs[0] = Z_INIT
    f = 1.0 - math.exp(-1.0 / (60.0 * ZOOM_TAU))
    for k in range(1, len(tf)):
        zs[k] = zs[k - 1] + (float(np.interp(tf[k], tg, targets)) - zs[k - 1]) * f

    def zoom_at(t):
        if t >= T2:
            return float(zs[-1])
        return float(np.interp(t, tf, zs))

    return zoom_at


class Story(Animation):
    """The whole film: one pure function of absolute time."""

    def __init__(self, scene, tube, mesh, sol, zoom_at, t0, run_time):
        super().__init__(VMobject(), run_time=run_time, rate_func=linear)
        self.scene = scene
        self.camera = scene.camera
        self.tube = tube
        self.mesh = mesh
        self.sol = sol
        self.zoom_at = zoom_at
        self.t0 = float(t0)

    def state(self, t):
        h, n, seg, phase = timeline(t, self.sol)

        cam_dir = None
        rot = np.asarray(self.camera.generate_rotation_matrix(), dtype=float)
        cam_dir = rot.T @ np.array([0.0, 0.0, 1.0])

        built = self.tube.build(n, h, cam_dir)

        # camera
        theta, phi = camera_at(t)
        self.camera.set_theta(np.radians(theta))
        self.camera.set_phi(np.radians(phi))
        self.camera.set_zoom(self.zoom_at(t))

        if built is None:
            self.mesh.quads = np.empty((0, 4, 3))
            self.mesh.colors = np.empty((0, 4, 4), dtype=np.float32)
            return

        quads, cols, _ = built
        # no recentering: the camera rotates about the 4D origin (0,0,0,0)
        self.mesh.quads = quads
        self.mesh.colors = cols

    def interpolate_mobject(self, alpha):
        self.state(self.t0 + alpha * self.run_time)


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------
class FourDSlice(ThreeDScene):
    def construct(self):
        win = MLWindow(1920, 1080)
        win.scene = self
        self.camera.background_color = "#070b14"
        self.set_camera_orientation(phi=0.0, theta=-90.0 * DEGREES, zoom=1.0)

        xs, R = radius_profile()
        mesh = MeshBatch()
        sol = solve_rotations()
        f1, f2, score = pick_anchors(sol)
        print("[build] frame anchors: %s | %s  (min |n.f| = %.3f over film)"
              % (np.round(f1, 3), np.round(f2, 3), score))
        tube = SliceTube(xs, R, (f1, f2))
        win.set_mesh(mesh)

        zoom_at = build_zoom_trajectory(self.camera, tube, sol)
        story = Story(self, tube, mesh, sol, zoom_at, t0=0.0, run_time=TEND)
        win.play(story, run_time=TEND)
        win.close()


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------
def principal_angles(A, B):
    """Both principal angles (deg) between two planes given as 4x2 bases."""
    s = np.linalg.svd(np.asarray(A).T @ np.asarray(B), compute_uv=False)
    return tuple(np.degrees(np.arccos(np.clip(s, -1.0, 1.0))))


def probe(sol):
    ths, N1, N2, N3, pairs = sol
    print("theta_max (deg): %.2f  %.2f  %.2f" % tuple(np.degrees(ths)))
    print("k at boundaries: n0 %.3f -> N1 %.3f -> N2(peak) %.3f -> N3 %.3f"
          % (k_of(N0), k_of(N1), k_of(N2), k_of(N3)))
    for j, (ga, gb) in enumerate(pairs):
        print("W%d = span(g%d, g%d): g's = %s | %s"
              % (j + 1, 2 * j + 1, 2 * j + 2, np.round(ga, 3), np.round(gb, 3)))
    B = [np.stack([ga, gb], axis=1) for ga, gb in pairs]
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        a1, a2 = principal_angles(B[i], B[j])
        print("principal angles W%d-W%d: %.1f deg, %.1f deg"
              % (i + 1, j + 1, a1, a2))
    print("%-5s %-8s %-5s %-6s %-6s %-8s %-6s" %
          ("t", "phase", "seg", "h", "k", "n_a", "|n_im|"))
    for t in np.arange(0.0, TEND + 0.5, 0.5):
        h, n, seg, phase = timeline(float(t), sol)
        print("%-5.1f %-8s %-5d %-6.3f %-6.3f %-8.3f %-6.3f" %
              (t, phase, seg, h, k_of(n), float(n[0]), float(np.linalg.norm(n[1:]))))
    print("checks (axis P_j = W_j^perp: |P|>0 = not in H; |X|>0 = rotation moves n):")
    for j, (ga, gb) in enumerate(pairs):
        if j == 0:
            n_s = N0
        else:
            n_s = N1
        X = (n_s @ ga) * ga + (n_s @ gb) * gb
        P = n_s - X
        print("  segment %d: |P|=%.3f  |X|=%.3f   (axis generic: both > 0)"
              % (j + 1, np.linalg.norm(P), np.linalg.norm(X)))
    f1, f2, score = pick_anchors(sol)
    print("  frame anchors: %s | %s -> min |n.f| over film = %.3f"
          % (np.round(f1, 3), np.round(f2, 3), score))


def stills(times, out_dir="frames", sol=None):
    from real_time_manim import FrameServer

    os.makedirs(out_dir, exist_ok=True)
    fs = FrameServer(FourDSlice)
    print("[prepare] %.1f s, duration %.2f s" % (time.time() - _T0, fs.duration))
    try:
        for t in times:
            p = os.path.join(out_dir, "slice_t%05.2f.png" % t)
            fs.save_frame(t, p)
            print(p)
    finally:
        fs.close()


_T0 = time.time()


def render(out_path, jobs=None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    t0 = time.time()
    res = fast_record_scene(FourDSlice, out_path, overwrite=True, jobs=jobs, cleanup=False)
    print("[render] %.1f s -> %s" % (time.time() - t0, res.get("files")))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="4D Mandelbrot slice (RTM)")
    ap.add_argument("--out", default=os.path.expanduser("~/Desktop/four_d_mandelbrot_slice.mp4"))
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--still", default=None, help="comma separated seconds; PNGs instead")
    ap.add_argument("--still-dir", default="frames")
    ap.add_argument("--probe", action="store_true", help="print trajectory, no render")
    args = ap.parse_args(argv)
    sol = solve_rotations()
    if args.probe:
        probe(sol)
        return 0
    if args.still:
        times = [float(x) for x in args.still.split(",") if x.strip()]
        return stills(times, args.still_dir, sol)
    render(args.out, jobs=args.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
