#!/usr/bin/env python3
"""
Thurston-corrugation sphere eversion -- explicit regular homotopy of immersions.

X: [0,1]_s x S^1_theta x [0,1]_tau -> R^3, with X(.,.,0) = unit sphere
(outward orientation) and X(.,.,1) = unit sphere with reversed orientation,
immersed for every tau.  The two sides of the material are drawn as two
surfaces offset +-EPS along the continuous normal field n_hat(s,theta,tau):
red = outside at tau=0, blue = inside.  n_hat(1) is inward, so the visible
colour swaps exactly once per half cycle.

Construction:
    r(s)   = sin(pi s)
    z0(s,tau) piecewise-linear in tau between
        P0:  cos(pi s)          (sphere)
        P1:  cos(3 pi s)        (double sphere -- the loop stage)
        P2: -cos(pi s)          (inverted sphere)
The profile's only cusp candidate (r' = 0 and z0_s = 0) sits at (s,tau)=(1/2,1/8).
The corrugation keeps X immersed there:
    a(s,th,tau) = A(tau) b(s) cos(m th)     (radial wiggle)
    c(s,th,tau) = C(tau) b(s) sin(m th)     (vertical wiggle, phase-shifted)
with  A(tau) = A0 exp(-((tau-1/8)/W)^2),  b(s) = (1/2) sin u (1+cos u),
u = pi(s-1/2)/SIGMA, zero outside |s-1/2| <= SIGMA.  b'(1/2) = pi/SIGMA != 0
and b vanishes (with zero slope) at the band edge and the poles, so both
wiggles carry a nonzero s-slope at the pinch and leave the poles untouched.

Timeline:  t in [0,15]   tau = t/15        (inside -> outside)
           t in [15,30]  tau = 2 - t/15    (outside -> inside)
Camera: full 360 deg azimuth orbit at phi = 58 deg; light rides the camera.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import real_time_manim  # noqa: F401
from real_time_manim import Animation, DEGREES, TAU, ThreeDScene, VGroup, linear
from real_time_manim.mesh import MeshBatch
from real_time_manim.record import fast_record_scene
from real_time_manim.vulkan_bind import MLWindow

# ---------------------------------------------------------------- parameters
FPS = 60
DURATION = 30.0
HALF = DURATION / 2.0
TAU_PINCH = 1.0 / 8.0

NS = 96                      # quads in s
NT = 192                     # quads in theta
M_WAVES = 8                  # corrugation azimuthal frequency (8 nodes)
A0 = 0.32                    # radial corrugation amplitude
C0 = 0.10                    # vertical corrugation amplitude
W = 0.09                     # corrugation tau-width (centred on the pinch)
SIGMA = 0.16                 # corrugation band half-width in s
EPS = 0.022                  # side offset ("thickness" of the material)
SCALE = 2.4                  # world scale (sphere diameter ~650 px at 1080p)

PHI = 58 * DEGREES
LIGHT0 = np.array([0.42, -0.48, 0.85])
LIGHT0 = LIGHT0 / np.linalg.norm(LIGHT0)
AMBIENT, DIFFUSE = 0.30, 0.70

COL_OUT = np.array([0.92, 0.30, 0.24], dtype=np.float32)   # outside at tau=0
COL_IN = np.array([0.20, 0.45, 0.95], dtype=np.float32)    # inside  at tau=0

# -------------------------------------------------------------- static grids
S = np.linspace(0.0, 1.0, NS + 1)
T = np.linspace(0.0, TAU, NT + 1)
SS, TT = np.meshgrid(S, T, indexing="ij")          # (NS+1, NT+1)

R = np.sin(np.pi * SS)
RP = np.pi * np.cos(np.pi * SS)

U_ = np.pi * (SS - 0.5) / SIGMA
INBAND = np.abs(SS - 0.5) <= SIGMA
B = np.where(INBAND, 0.5 * np.sin(U_) * (1.0 + np.cos(U_)), 0.0)
BS = np.where(INBAND, (np.pi / SIGMA) * 0.5
              * (np.cos(U_) * (1.0 + np.cos(U_)) - np.sin(U_) ** 2), 0.0)

CM = np.cos(M_WAVES * TT)
SM = np.sin(M_WAVES * TT)
COST, SINT = np.cos(TT), np.sin(TT)

I0, J0 = np.meshgrid(np.arange(NS), np.arange(NT), indexing="ij")
CIDX = np.stack([np.stack([I0, J0], -1), np.stack([I0 + 1, J0], -1),
                 np.stack([I0 + 1, J0 + 1], -1), np.stack([I0, J0 + 1], -1)],
                axis=2).reshape(-1, 4, 2)         # (Q, 4, 2)
NQ = CIDX.shape[0]
II, JJ = CIDX[..., 0], CIDX[..., 1]


def tau_of(t):
    return t / HALF if t <= HALF else 2.0 - t / HALF


def profile(tau):
    """z0(s,tau) and its s-derivative, as (NS+1,) columns."""
    if tau <= 0.5:
        z = (1.0 - 2.0 * tau) * np.cos(np.pi * S) + 2.0 * tau * np.cos(3.0 * np.pi * S)
        zs = (-np.pi * (1.0 - 2.0 * tau) * np.sin(np.pi * S)
              - 6.0 * np.pi * tau * np.sin(3.0 * np.pi * S))
    else:
        z = ((2.0 - 2.0 * tau) * np.cos(3.0 * np.pi * S)
             - (2.0 * tau - 1.0) * np.cos(np.pi * S))
        zs = (-3.0 * np.pi * (2.0 - 2.0 * tau) * np.sin(3.0 * np.pi * S)
              + np.pi * (2.0 * tau - 1.0) * np.sin(np.pi * S))
    return z, zs


def geometry(tau):
    """Grid positions, s/theta derivatives and the unit normal field."""
    z0, zs0 = profile(tau)
    z0 = z0[:, None]
    zs0 = zs0[:, None]
    g = float(np.exp(-(((tau - TAU_PINCH) / W) ** 2)))
    # gate the rise so the start sphere is exactly clean (g(0)=0, g(0.05)=1)
    g *= min(1.0, (tau / 0.05) ** 2 * (3.0 - 2.0 * tau / 0.05)) if tau < 0.05 else 1.0
    A, C = A0 * g, C0 * g

    a = A * B * CM
    a_s = A * BS * CM
    a_t = -A * B * M_WAVES * SM
    c = C * B * SM
    c_s = C * BS * SM
    c_t = C * B * M_WAVES * CM

    rr = R * (1.0 + a)
    rr_s = RP * (1.0 + a) + R * a_s

    X = np.stack([rr * COST, rr * SINT, z0 + c], axis=-1)
    Xs = np.stack([rr_s * COST, rr_s * SINT, zs0 + c_s], axis=-1)
    Xt = np.stack([-rr * SINT - R * a_t * COST,
                   rr * COST - R * a_t * SINT, c_t], axis=-1)

    n = np.cross(Xs, Xt)
    nlen = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.where(nlen < 1e-12, 1.0, nlen)
    n[0] = [0.0, 0.0, 1.0]       # north-pole row: normal limit is +z for all tau
    n[-1] = [0.0, 0.0, -1.0]     # south-pole row: -z
    return X, Xs, Xt, n


def state(tau, t):
    """World-space geometry of both colour layers + per-vertex shading."""
    X, _Xs, _Xt, n = geometry(tau)
    th = TAU * t / DURATION            # camera azimuth; light rides it
    l = np.array([LIGHT0[0] * np.cos(th) - LIGHT0[1] * np.sin(th),
                  LIGHT0[0] * np.sin(th) + LIGHT0[1] * np.cos(th),
                  LIGHT0[2]])
    shade = AMBIENT + DIFFUSE * np.clip(n @ l, 0.0, None)      # (NS+1, NT+1)
    Xr = SCALE * (X + EPS * n)
    Xb = SCALE * (X - EPS * n)
    return Xr, Xb, shade


class EversionMovie(Animation):
    """Per-frame: rewrite both colour layers from the frame time in closed
    form (pure function of t, so seeking and recording agree), and orbit the
    camera azimuth."""

    def __init__(self, camera, write_state, carrier, duration=DURATION):
        self.camera = camera
        self.write_state = write_state
        super().__init__(carrier, run_time=duration, rate_func=linear)

    def interpolate_mobject(self, alpha):
        t = alpha * self.run_time
        self.write_state(tau_of(t), t)
        self.camera.set_theta(TAU * t / DURATION)


class EversionScene(ThreeDScene):
    def construct(self):
        win = MLWindow(960, 540)
        win.scene = self
        self.set_camera_orientation(phi=PHI, theta=0.0)

        mesh = MeshBatch()
        total = 2 * NQ
        assert total * 6 <= mesh.MAX_MESH_VERTS, (
            "mesh over the native vertex cap: %d quads" % total)
        mesh.quads = np.empty((total, 4, 3), dtype=np.float64)
        mesh.colors = np.empty((total, 4, 4), dtype=np.float32)
        win.set_mesh(mesh)

        def write_state(tau, t):
            Xr, Xb, shade = state(tau, t)
            mesh.quads[:NQ] = Xr[II, JJ]
            mesh.quads[NQ:] = Xb[II, JJ]
            cr = np.empty((NS + 1, NT + 1, 4), dtype=np.float32)
            cr[..., :3] = COL_OUT * shade[..., None]
            cr[..., 3] = 1.0
            cb = np.empty((NS + 1, NT + 1, 4), dtype=np.float32)
            cb[..., :3] = COL_IN * shade[..., None]
            cb[..., 3] = 1.0
            mesh.colors[:NQ] = cr[II, JJ]
            mesh.colors[NQ:] = cb[II, JJ]

        write_state(0.0, 0.0)                        # t = 0 state
        win.play(EversionMovie(self.camera, write_state, VGroup(), DURATION))
        win.close()


# ------------------------------------------------------------------- verify
def verify(ntau=201):
    """Math check BEFORE rendering: the family must be immersed everywhere."""
    taus = np.linspace(0.0, 1.0, ntau)
    worst, arg = np.inf, None
    for tau in taus:
        X, Xs, Xt, _n = geometry(tau)
        cr = np.linalg.norm(np.cross(Xs, Xt), axis=-1)
        m = cr[1:-1].min()                    # interior rows only
        if m < worst:
            worst = m
            i, j = np.unravel_index(np.argmin(cr[1:-1]), cr[1:-1].shape)
            arg = (tau, float(S[i + 1]), float(T[j] / TAU))
    print("min |dX/ds x dX/dtheta| over interior grid: %.4e" % worst)
    print("  at (tau, s, theta/2pi) = (%.3f, %.4f, %.4f)" % arg)
    for tau in (0.0, 0.5, 1.0):
        X, Xs, Xt, _n = geometry(tau)
        print("tau=%.1f  max||X|-1| = %.2e" % (
            tau, float(np.abs(np.linalg.norm(X, axis=-1) - 1.0).max())))
    # close-up of the pinch window
    print("pinch window (min cross per tau):")
    for tau in np.linspace(0.10, 0.16, 7):
        _X, Xs, Xt, _n = geometry(tau)
        cr = np.linalg.norm(np.cross(Xs, Xt), axis=-1)
        print("  tau=%.3f  min=%.4e" % (tau, cr[1:-1].min()))


# ------------------------------------------------------------------- render
def render(out_path="out/sphere_eversion_th.mp4"):
    out = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    t0 = time.time()
    res = fast_record_scene(EversionScene, out, overwrite=True)
    print("[render] %.1f s -> %s" % (time.time() - t0, res.get("files")))
    return res


def probe():
    res = fast_record_scene(EversionScene, "out/_probe.mp4", count_only=True)
    print("[probe] frames:", res)


def stills(times, out_dir="out/stills"):
    from real_time_manim.frames import FrameServer
    os.makedirs(out_dir, exist_ok=True)
    srv = FrameServer(EversionScene)
    for t in times:
        p = os.path.join(out_dir, "ev_t%06.2f.png" % t)
        srv.save_frame(t, p)
        print("[still] t=%.2f -> %s" % (t, p))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "render"
    if mode == "verify":
        verify()
    elif mode == "probe":
        probe()
    elif mode == "stills":
        stills([float(x) for x in sys.argv[2:]] or [0.0])
    else:
        render()
