r"""Kerr black hole -- 30 s RTM film (1920x1080 @ 60).

Brief:
  0-1 s   create a 2D coordinate plane (polar grid, camera top-down)
  2-5 s   black hole appears at (0,0) with mass 0, mass ramps to M=1;
          the coordinate grid responds (isotropic-coordinate bunching:
          ring k at r_iso moves to r_phys = r_iso (1 + M/2r_iso)^2),
          horizon disc grows to r_s = 2M
  5-10 s  go 3D: Flamm paraboloid extrudes (z = 2 sqrt(r_s (r - r_s))),
          rings relax back to Schwarzschild spacing, camera tilts
          (phi 0 -> 55 deg) and starts an azimuth sweep
 10-20 s  animate the 3D coordinate system in the Schwarzschild state:
          rain-frame (Gullstrand-Painleve) -- the rings fall inward along
          exact radial geodesics r = F(tau - t), v = sqrt(r_s/r),
          accelerating, fading at the horizon and respawning at the rim;
          camera orbits slowly
 20-30 s  spin-up a: 0 -> 0.9 M. Frame dragging: the spokes wind up
          differentially with the exact Lense-Thirring/ZAMO frequency
          omega(r) = 2 M a / (r^3 + a^2 r + 2 M a^2); the horizon shrinks
          to r_+ = M + sqrt(M^2 - a^2); the disc rotates at the exact
          horizon angular velocity Omega_H = a / (2 M r_+); rings keep
          falling

Physics notes:
- The equatorial plane of Kerr is NOT embeddable in E^3 for a > ~0.87 M
  (its Gaussian curvature turns positive near the throat -- checked
  numerically, max K = +0.0015 at a = 0.9), so the sheet keeps the
  Schwarzschild Flamm profile while the spin is carried by the dragging,
  the horizon radius and the disc rotation -- every one of those is the
  exact formula.
- Rain-frame fall: rings are surfaces of constant GP time; consecutive
  rings follow shifted solutions of dr/dt = -sqrt(r_s/r), i.e. parabolic
  infall from rest at infinity, F(s) = ((3/2) sqrt(r_s) s)^(2/3).
- Frame dragging dies as 1/r^3 away from the hole; the wind-up is local
  to the throat (by design -- that is the effect).

Geometry: one depth-tested MeshBatch (wireframe grid + horizon disc +
collar cone + disc rotation marker). Per-frame state is a pure function
of t (all integrals precomputed on the frame grid), so FrameServer stills
and the fast-record video come from the same function.
"""
import ctypes
import os

import numpy as np

import real_time_manim  # noqa: F401  (must precede any manim import)
from real_time_manim import (Animation, DEGREES, FrameServer, MeshBatch,
                             Mobject, ThreeDScene, fast_record_scene)
from real_time_manim.vulkan_bind import MLWindow

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser("~/gallary/video/kerr_blackhole.mp4")
STILLS_DIR = os.path.join(HERE, "stills")

# ----------------------------------------------------------------- knobs ---
FPS = 60
DUR = 30.0
N_FR = int(round(DUR * FPS))          # 1800

M_FINAL = 1.0
A_MAX = 0.9
RS = 2.0 * M_FINAL                    # Schwarzschild radius at full mass

R_FLOW = 34.0                         # outer grid edge / ring respawn radius
R_IN = 2.5                            # innermost static ring
R_GRID_MAX = 32.5                     # outermost static ring
R_HUB = 0.08                          # spoke inner hub (under the disc)
R_COLLAR_TOP = 2.08                   # sheet inner rim (3D phase)

N_RINGS = 64
N_RING_SEG = 120
N_SPOKES = 96
N_SPOKE_SEG = 64
N_DISC_SEG = 64
N_DISC_BAND = 2
N_COLLAR_SEG = 64
N_DISCSPOKE = 12

W_LINE = 0.05                         # grid line width (world)
HW = W_LINE / 2.0
W_DS = 0.10                           # disc rotation marker width
HW_DS = W_DS / 2.0

Z_DISC = 0.01
Z_DS = 0.02

# camera: 2D top-down -> tilted 3D
ZOOM_2D = 0.45
ZOOM_3D = 0.50
FOCAL = 40.0                         # camera distance from the frame centre
PHI_3D = 55.0                         # degrees from vertical
TH_TILT = 45.0                        # azimuth swept during the tilt
TH_END = 135.0                        # azimuth at t = 30
FC_Z = 1.0                            # frame-centre z shift in 3D (aim above disc)

# colours (style-params table)
BG = np.array([0.0275, 0.0431, 0.0784])
GRID_C = np.array([0.552, 0.651, 0.722])      # dim blue-grey, the grid itself
GOLD = np.array([0.949, 0.776, 0.427])
DISC_C = np.array([0.0, 0.0, 0.0])           # pure black: no chroma to bleed
                                              # under yuv420 subsampling (the
                                              # old dark-blue smeared to 0.1+
                                              # beside the bright grid)
RIM_C = np.array([0.160, 0.118, 0.068])       # faint warm photon-rim
COLLAR_C = np.array([0.012, 0.018, 0.035])
GRID_ALPHA = 0.88
FADE_COLLAPSE = 0.02                          # below this, quads collapse

# ----------------------------------------------------------------- ease ----
def sstep(x):
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def fade_g(t):
    """Grid fade-in over 0-1 s."""
    return sstep(t / 1.0)


def m_t(t):
    return M_FINAL * sstep((t - 2.0) / 3.0)


def a_t(t):
    return A_MAX * sstep((t - 20.0) / 10.0)


def r_hor_t(t):
    return M_FINAL + np.sqrt(M_FINAL * M_FINAL - a_t(t) ** 2)


def r_disc_t(t):
    return 2.0 * m_t(t) if t < 5.0 else r_hor_t(t)


def s_ext(t):
    return sstep((t - 5.0) / 3.0)


def flow_clock(t):
    """Rain-frame clock: integral of the ease-in; t >= 12 runs 1:1."""
    if t < 10.0:
        return 0.0
    if t < 12.0:
        v = (t - 10.0) / 2.0
        return 2.0 * (v ** 3 - v ** 4 / 2.0)
    return t - 11.0


# ------------------------------------------------------------- geometry ----
def z_fl(r):
    return np.where(r >= RS, 2.0 * np.sqrt(np.maximum(RS * (r - RS), 0.0)), 0.0)


def z_at(r, t):
    return s_ext(t) * z_fl(r)


def zp_fl(r):
    return np.where(r >= RS, np.sqrt(RS / np.maximum(r - RS, 1e-9)), 0.0)


# rings: base radii, isotropic shift, rain-frame labels
R_K0 = np.linspace(R_IN, R_GRID_MAX, N_RINGS)
D_K = M_FINAL + M_FINAL ** 2 / (4.0 * R_K0)     # isotropic shift at M = 1
SQRT_RS = np.sqrt(RS)
S_HOR = (2.0 / 3.0) * RS ** 1.5 / SQRT_RS       # rain clock at the horizon
S_RIM = (2.0 / 3.0) * R_FLOW ** 1.5 / SQRT_RS   # rain clock at the rim
P_FLOW = S_RIM - S_HOR                          # full transit period
TAU0 = (2.0 / 3.0) * R_K0 ** 1.5 / SQRT_RS      # slot fall labels at t = 10


def F_free(s):
    return ((1.5 * SQRT_RS) * s) ** (2.0 / 3.0)


def ring_radii(t):
    if t < 10.0:
        if t < 5.0:
            m = m_t(t)
            return R_K0 + m + m * m / (4.0 * R_K0)
        if t < 8.0:
            w = 1.0 - sstep((t - 5.0) / 3.0)
            return R_K0 + w * D_K
        return R_K0
    s = flow_clock(t)
    n = np.maximum(0.0, np.floor((s - (TAU0 - S_HOR)) / P_FLOW) + 1.0)
    tau = TAU0 + n * P_FLOW - s
    return F_free(tau)


def ring_alphas(r, t):
    a = np.full_like(r, GRID_ALPHA * fade_g(t))
    if t >= 10.0:
        a *= np.clip((r - (RS + 0.08)) / 0.40, 0.0, 1.0)
        a *= np.clip((R_FLOW - r) / 1.50, 0.0, 1.0)
    return a


# spokes: fixed segment radii, per-frame azimuth = base + wind table
R_SEG = np.linspace(R_HUB, R_FLOW, N_SPOKE_SEG + 1)      # (65,)
R_MID = 0.5 * (R_SEG[:-1] + R_SEG[1:])
Z_FL_SEG = z_fl(R_SEG)
ZP_SEG = zp_fl(R_SEG)
ZP_MID = 0.5 * (ZP_SEG[:-1] + ZP_SEG[1:])
TH_S0 = np.linspace(0.0, 2.0 * np.pi, N_SPOKES, endpoint=False)

# angular grids
TH_RING = np.linspace(0.0, 2.0 * np.pi, N_RING_SEG, endpoint=False)
E_J = np.stack([np.cos(TH_RING), np.sin(TH_RING)], axis=1)     # (120, 2)
TH_RING_N = TH_RING + 2.0 * np.pi / N_RING_SEG
E_N = np.stack([np.cos(TH_RING_N), np.sin(TH_RING_N)], axis=1)
TH_DISC = np.linspace(0.0, 2.0 * np.pi, N_DISC_SEG, endpoint=False)
ED_J = np.stack([np.cos(TH_DISC), np.sin(TH_DISC)], axis=1)    # (64, 2)
ED_N = np.stack([np.cos(TH_DISC + 2.0 * np.pi / N_DISC_SEG),
                 np.sin(TH_DISC + 2.0 * np.pi / N_DISC_SEG)], axis=1)

# ----------------------------------------------------------------- tables --
def omega_lt(r, a):
    return 2.0 * M_FINAL * a / (r ** 3 + a * a * r + 2.0 * M_FINAL * a * a)


def build_wind():
    W = np.zeros((N_SPOKE_SEG + 1, N_FR))
    prev = np.zeros(N_SPOKE_SEG + 1)
    for k in range(1, N_FR):
        t = k / FPS
        ak = a_t(t)
        if ak > 0.0:
            r = np.maximum(R_SEG, r_hor_t(t) + 0.02)
            prev = prev + omega_lt(r, ak) / FPS
        W[:, k] = prev
    return W


def build_disc_angle():
    ph = np.zeros(N_FR)
    prev = 0.0
    for k in range(1, N_FR):
        t = k / FPS
        ak = a_t(t)
        if ak > 0.0:
            rp = r_hor_t(t)
            prev = prev + (ak / (2.0 * M_FINAL * rp)) / FPS
        ph[k] = prev
    return ph


WIND = build_wind()
PHI_D = build_disc_angle()

# ----------------------------------------------------------------- layers ---
NQ_RINGS = N_RINGS * N_RING_SEG
NQ_SPOKES = N_SPOKES * N_SPOKE_SEG
NQ_DISC = N_DISC_SEG * N_DISC_BAND
NQ_COLLAR = N_COLLAR_SEG
NQ_DS = N_DISCSPOKE
NQ = NQ_RINGS + NQ_SPOKES + NQ_DISC + NQ_COLLAR + NQ_DS
S_RINGS = 0
S_SPOKES = S_RINGS + NQ_RINGS
S_DISC = S_SPOKES + NQ_SPOKES
S_COLLAR = S_DISC + NQ_DISC
S_DS = S_COLLAR + NQ_COLLAR

R_DS_FRAC = np.linspace(0.15, 0.92, N_DISCSPOKE + 1)


def compose_batch():
    b = MeshBatch()
    b.quads = np.zeros((NQ, 4, 3))
    b.colors = np.zeros((NQ, 4, 4), dtype=np.float32)
    return b


# --------------------------------------------------------------- camera ----
def phi_t(t):
    if t <= 5.0:
        return 0.0
    return PHI_3D * sstep((t - 5.0) / 4.5)


def theta_t(t):
    if t <= 5.0:
        return 0.0
    if t < 10.0:
        return TH_TILT * sstep((t - 5.0) / 4.5)
    return TH_TILT + (TH_END - TH_TILT) * (t - 10.0) / 20.0


def zoom_t(t):
    if t <= 5.0:
        return ZOOM_2D
    return ZOOM_2D + (ZOOM_3D - ZOOM_2D) * sstep((t - 5.0) / 4.5)


def fcz_t(t):
    if t <= 5.0:
        return 0.0
    return FC_Z * sstep((t - 5.0) / 4.5)


# ------------------------------------------------------------- animation ---
def patch_background(win):
    try:
        if getattr(win, "_set_background", None) is None:
            fn = win.dll.Render_SetBackgroundColor
            fn.restype = None
            fn.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
            win._set_background = fn
    except Exception as exc:
        print("[bg] patch skipped:", exc)


class KerrAnim(Animation):
    """Drive mesh + camera from precomputed per-frame state (pure f(t))."""

    def __init__(self, mesh, camera, **kw):
        super().__init__(Mobject(), rate_func=lambda a: a, **kw)
        self.mesh = mesh
        self.camera = camera
        self.q = mesh.quads
        self.c = mesh.colors

    def _write_rings(self, t):
        r = ring_radii(t)
        a = ring_alphas(r, t)
        se = s_ext(t)
        zl = se * z_fl(r - HW)
        zh = se * z_fl(r + HW)
        q = self.q[S_RINGS:S_RINGS + NQ_RINGS].reshape(
            N_RINGS, N_RING_SEG, 4, 3)
        q[..., 0, :2] = (r[:, None] - HW)[:, None, :] * E_J[None, :, :]
        q[..., 1, :2] = (r[:, None] + HW)[:, None, :] * E_J[None, :, :]
        q[..., 2, :2] = (r[:, None] + HW)[:, None, :] * E_N[None, :, :]
        q[..., 3, :2] = (r[:, None] - HW)[:, None, :] * E_N[None, :, :]
        q[..., 0, 2] = zl[:, None]
        q[..., 1, 2] = zh[:, None]
        q[..., 2, 2] = zh[:, None]
        q[..., 3, 2] = zl[:, None]
        col = self.c[S_RINGS:S_RINGS + NQ_RINGS].reshape(
            N_RINGS, N_RING_SEG, 4, 4)
        col[..., :3] = GRID_C
        col[..., 3] = a[:, None, None]
        bad = a < FADE_COLLAPSE
        q[bad] = 0.0
        col[bad] = 0.0

    def _write_spokes(self, t, k):
        th = TH_S0[:, None] + WIND[:, k][None, :]            # (96, 65)
        cth, sth = np.cos(th), np.sin(th)
        x = R_SEG[None, :] * cth
        y = R_SEG[None, :] * sth
        z = s_ext(t) * np.broadcast_to(Z_FL_SEG[None, :], cth.shape)
        P = np.stack([x, y, z], axis=-1)                     # (96, 65, 3)
        T = np.diff(P, axis=1)                               # (96, 64, 3)
        thm = 0.5 * (th[:, :-1] + th[:, 1:])
        N = np.stack([-ZP_MID[None, :] * np.cos(thm),
                      -ZP_MID[None, :] * np.sin(thm),
                      np.ones_like(thm)], axis=-1)          # (96, 64, 3)
        w = np.cross(T, N)
        nw = np.linalg.norm(w, axis=-1, keepdims=True)
        w = w / np.maximum(nw, 1e-9)
        Pi = P[:, :-1]
        Pj = P[:, 1:]
        q = self.q[S_SPOKES:S_SPOKES + NQ_SPOKES].reshape(
            N_SPOKES, N_SPOKE_SEG, 4, 3)
        q[..., 0, :] = Pi - w * HW
        q[..., 1, :] = Pi + w * HW
        q[..., 2, :] = Pj + w * HW
        q[..., 3, :] = Pj - w * HW
        warm = sstep((3.4 - R_MID) / 1.4)[None, :, None]     # (1, 64, 1)
        prog = a_t(t) / A_MAX
        cc = GRID_C[None, None, None, :] + (
            GOLD - GRID_C)[None, None, None, :] * (warm * prog)[..., None]
        fade = fade_g(t)
        col = self.c[S_SPOKES:S_SPOKES + NQ_SPOKES].reshape(
            N_SPOKES, N_SPOKE_SEG, 4, 4)
        col[..., :3] = cc
        col[..., 3] = fade
        if fade < FADE_COLLAPSE:
            q[:] = 0.0
            col[:] = 0.0

    def _write_disc(self, t):
        rd = r_disc_t(t)
        lo = np.array([0.0, 0.72]) * rd
        hi = np.array([0.72, 1.0]) * rd
        q = self.q[S_DISC:S_DISC + NQ_DISC].reshape(
            N_DISC_BAND, N_DISC_SEG, 4, 3)
        q[..., 0, :2] = lo[:, None, None] * ED_J[None, :, :]
        q[..., 1, :2] = hi[:, None, None] * ED_J[None, :, :]
        q[..., 2, :2] = hi[:, None, None] * ED_N[None, :, :]
        q[..., 3, :2] = lo[:, None, None] * ED_N[None, :, :]
        q[..., 2] = Z_DISC
        col = self.c[S_DISC:S_DISC + NQ_DISC].reshape(
            N_DISC_BAND, N_DISC_SEG, 4, 4)
        col[0, :, :, :3] = DISC_C
        col[1, :, :, :3] = RIM_C
        col[..., 3] = 1.0

    def _write_collar(self, t):
        rd = r_disc_t(t)
        rc_top = rd + 0.08 if t < 5.0 else R_COLLAR_TOP
        zt = s_ext(t) * z_fl(rc_top)
        q = self.q[S_COLLAR:S_COLLAR + NQ_COLLAR].reshape(
            N_COLLAR_SEG, 4, 3)
        q[..., 0, :2] = rd * ED_J
        q[..., 1, :2] = rc_top * ED_J
        q[..., 2, :2] = rc_top * ED_N
        q[..., 3, :2] = rd * ED_N
        q[..., 0, 2] = Z_DISC
        q[..., 1, 2] = zt
        q[..., 2, 2] = zt
        q[..., 3, 2] = Z_DISC
        col = self.c[S_COLLAR:S_COLLAR + NQ_COLLAR].reshape(
            N_COLLAR_SEG, 4, 4)
        col[..., :3] = COLLAR_C
        col[..., 3] = 1.0

    def _write_disc_spoke(self, t, k):
        rd = r_disc_t(t)
        ph = PHI_D[k]
        u = np.array([np.cos(ph), np.sin(ph)])
        w = np.array([-np.sin(ph), np.cos(ph)])
        ri = R_DS_FRAC[:-1] * rd
        rj = R_DS_FRAC[1:] * rd
        q = self.q[S_DS:S_DS + NQ_DS].reshape(N_DISCSPOKE, 4, 3)
        q[..., 0, :2] = ri[:, None] * u[None, :] - w * HW_DS
        q[..., 1, :2] = ri[:, None] * u[None, :] + w * HW_DS
        q[..., 2, :2] = rj[:, None] * u[None, :] + w * HW_DS
        q[..., 3, :2] = rj[:, None] * u[None, :] - w * HW_DS
        q[..., 2] = Z_DS
        col = self.c[S_DS:S_DS + NQ_DS].reshape(N_DISCSPOKE, 4, 4)
        col[..., :3] = GOLD
        al = 0.9 * sstep((t - 20.0) / 4.0)
        col[..., 3] = al
        if al < FADE_COLLAPSE:
            q[:] = 0.0
            col[:] = 0.0

    def write_frame(self, k, t):
        self._write_rings(t)
        self._write_spokes(t, k)
        self._write_disc(t)
        self._write_collar(t)
        self._write_disc_spoke(t, k)
        self._cull_near_plane()
        cam = self.camera
        cam.set_phi(phi_t(t) * DEGREES)
        cam.set_theta(theta_t(t) * DEGREES)
        cam.set_zoom(zoom_t(t))
        cam.frame_center[:] = (0.0, 0.0, fcz_t(t))

    def _cull_near_plane(self):
        """Collapse quads whose corners get within ~1.5 units of the camera
        plane: the mesh pass has no near clipping and would paint them
        mirrored across the focal plane (factor flips sign)."""
        try:
            R = np.asarray(self.camera.generate_rotation_matrix(),
                           dtype=np.float64)
            fc = np.asarray(self.camera.frame_center, dtype=np.float64)[:3]
        except Exception:
            return
        pts = self.q.reshape(-1, 3)
        qz = ((pts - fc) @ R.T)[:, 2].reshape(-1, 4)
        bad = (qz > FOCAL - 1.5).any(axis=1)
        if bad.any():
            self.q[bad] = 0.0
            self.c[bad] = 0.0

    def interpolate_mobject(self, alpha):
        t = alpha * self.run_time
        k = int(round(t * FPS))
        if k > N_FR - 1:
            k = N_FR - 1
        self.write_frame(k, t)


# ----------------------------------------------------------------- scene ----
class KerrBlackHole(ThreeDScene):
    def construct(self):
        win = MLWindow(1920, 1080)
        win.scene = self
        self.play = win.play
        patch_background(win)
        try:
            win._set_background(float(BG[0]), float(BG[1]), float(BG[2]))
        except Exception as exc:
            print("[bg] set skipped:", exc)
        self.camera.background_color = "#070b14"
        self.camera.set_phi(0.0)
        self.camera.set_theta(0.0)
        self.camera.set_zoom(ZOOM_2D)
        # RTM's camera sits at focal_distance from the frame centre (zoom only
        # scales the projection).  The funnel mouth (r=18, z~11) would pass
        # within ~5 units of a focal-20 camera during the tilt and explode in
        # the perspective divide; 40 keeps every quad > ~15 units away.
        self.camera.set_focal_distance(FOCAL)

        mesh = compose_batch()
        win.set_mesh(mesh)
        anim = KerrAnim(mesh, self.camera, run_time=DUR)
        anim.write_frame(0, 0.0)
        win.play(anim, run_time=DUR)
        win.close()


# ------------------------------------------------------------------ main ----
def report():
    print(f"[build] quads: rings {NQ_RINGS} + spokes {NQ_SPOKES} + disc "
          f"{NQ_DISC} + collar {NQ_COLLAR} + marker {NQ_DS} = {NQ} "
          f"({NQ * 6} verts, cap {524288})")
    print(f"[phys]  r_s = {RS}, r_+ (a=0.9) = {1 + np.sqrt(1 - 0.81):.4f}, "
          f"Omega_H (a=0.9) = {0.9 / (2 * 1.0 * (1 + np.sqrt(1 - 0.81))):.4f}")
    print(f"[phys]  omega_LT(r=2.5, a=0.9) = {omega_lt(2.5, 0.9):.4f}, "
          f"omega_LT(r=4, a=0.9) = {omega_lt(4.0, 0.9):.4f}")
    i25 = np.argmin(abs(R_SEG - 2.5))
    i6 = np.argmin(abs(R_SEG - 6.0))
    print(f"[phys]  wind at r=2.5, t=30: {WIND[i25, -1]:.3f} rad "
          f"({np.degrees(WIND[i25, -1]):.1f} deg); at r=6: "
          f"{WIND[i6, -1]:.3f} rad; at rim ~0: {WIND[-1, -1]:.2e}")
    print(f"[phys]  disc marker angle at t=30: "
          f"{np.degrees(PHI_D[-1]):.1f} deg")
    s_hit = TAU0[0] - S_HOR
    ts = np.linspace(10.0, 30.0, 20001)
    t_hit = ts[np.argmin(abs(np.array([flow_clock(x) for x in ts]) - s_hit))]
    print(f"[phys]  rain: innermost ring reaches horizon at clock s="
          f"{s_hit:.3f} -> t = {t_hit:.2f} s")
    print(f"[phys]  isotropic shift at M=1: r={R_IN} -> {R_IN + D_K[0]:.2f}, "
          f"r={R_GRID_MAX:.2f} -> {R_GRID_MAX + D_K[-1]:.2f}")
    print(f"[cam]   zoom {ZOOM_2D} -> {ZOOM_3D}, phi 0 -> {PHI_3D}, "
          f"theta {TH_TILT} -> {TH_END}, frame_center z -> {FC_Z}")


def stills():
    os.makedirs(STILLS_DIR, exist_ok=True)
    srv = FrameServer(KerrBlackHole)
    for t in (0.3, 1.5, 3.0, 4.5, 6.5, 9.0, 12.0, 16.0, 22.0, 26.0, 29.5):
        p = os.path.join(STILLS_DIR, f"t{t:05.2f}.png")
        srv.save_frame(t, p)
        print("still", p, flush=True)
    srv.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        report()
    elif len(sys.argv) > 1 and sys.argv[1] == "stills":
        stills()
    else:
        report()
        r = fast_record_scene(KerrBlackHole, OUT, overwrite=True)
        print("files:", r["files"])
