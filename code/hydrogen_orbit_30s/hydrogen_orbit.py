"""30-second hydrogen-orbital film — RTM (1920x1080 @ 60). v2 (2026-11-26).

Brief (8 steps):
1. Classic-showcase opening: title + subtitle fade in, the Schroedinger
   equation fades in large and centred, then shrinks to the top-left corner.
2. 2p "membrane": flat polar-plot dumbbell (depth mesh).
3. Morph to the real 3D 2p solution: the probability cloud |psi|^2, drawn as
   SOFT camera-facing sprites whose opacity IS the density (cloud opacity =
   transparency), not as square particles.
4. Dynamic model (breathing) + rotating camera.
5. Higher orbitals, same treatment: 3d_z^2 then 4f_xyz.
6. 90-90-90 corner cut; camera keeps rotating; corner always faces the lens.
7. Combine hydrogen-like shells into the 37-Rb cloud.

v2 changes (user 2026-11-26):
- Square billboard particles -> radial-falloff diamond sprites (centre quad +
  4 ring quads, per-sprite random rotation): the density function shows up as
  varying transparency of a continuous cloud, not as dots.
- Sprites are drawn back-to-front per frame (sorted by frame-relative world z,
  stable + per-sprite dither) so the mesh pipeline's LESS_OR_EQUAL depth test
  composites them like a painter instead of occluding them randomly.
- Opening shot is the classic-showcase treatment (style-params table: title
  top-left 46px Cambria, subtitle 26px, gold centred equation, fades only).
"""
import ctypes

import numpy as np

import real_time_manim  # noqa: F401  MUST precede any manim import
from real_time_manim import (Scene, ThreeDScene, MathTex, Text, Dot, Mobject,
                             MeshBatch, Animation, smooth, ORIGIN, UL, DL, DOWN,
                             LEFT, DEGREES, LaggedStart)
from real_time_manim.record import fast_record_scene
from real_time_manim.vulkan_bind import (MLWindow, Write, Succession, Wait,
                                         FadeIn, FadeOut, Transform)

# ----------------------------------------------------------------- knobs ---
SEED = 7
FPS = 60

BG_HEX = "#070b14"
INK_HEX = "#efe9dc"
DIM_HEX = "#7b93a6"
GOLD_HEX = "#f2c66d"
CYAN_HEX = "#6fd6cd"
CREAM_HEX = "#fff6e8"

GOLD_RGB = np.array([242, 198, 109], dtype=float) / 255.0
CYAN_RGB = np.array([111, 214, 205], dtype=float) / 255.0

PHI_DEG = 54.7356            # polar angle: the 90-90-90 corner faces the lens
THETA_START = -150.0         # degrees
THETA_END = 70.0

R_MEMB = 2.8                 # membrane radius (world units)
R_VIS = {"2p": 2.9, "3dz2": 3.0, "4fxyz": 3.1}
N_PTS = {"2p": 10000, "3dz2": 10000, "4fxyz": 10500}
R_CUT = {"2p": 15.0, "3dz2": 40.0, "4fxyz": 48.0}
R_RB_OUTER = 3.0             # the 5s shell's rim lands here

# ---- soft-sprite cloud knobs ----
SPRITE_R = 0.14              # sprite radius (world units, ~19 px)

P_ALPHA = 0.40               # alpha = (rho/rho_max)^P_ALPHA
ALPHA_RIM = 0.18             # opacity at the visual rim (defines the scale)
MAX_ALPHA = 0.85             # peak sprite alpha (soft sprites overlap)

BREATHE_AMP = 0.05
BREATHE_PERIOD = 1.8
RB_BREATHE_AMP = 0.03
RB_BREATHE_PERIOD = 2.6

RB_SHELLS = [("1s", 1, 0, 864), ("2s", 2, 0, 1152), ("2p", 2, 1, 1440),
             ("3s", 3, 0, 1440), ("3p", 3, 1, 1728), ("3d", 3, 2, 2016),
             ("4s", 4, 0, 2304), ("4p", 4, 1, 1728), ("4d", 4, 2, 2304),
             ("4f", 4, 3, 2592), ("5s", 5, 0, 3456)]

SURFACE_PALETTE = np.array([
    [0.020, 0.055, 0.115], [0.040, 0.180, 0.280], [0.050, 0.420, 0.520],
    [0.220, 0.720, 0.720], [0.950, 0.760, 0.380], [1.000, 0.940, 0.820]])

# ------------------------------------------------------------- physics -----
def radial(n, l, r):
    """Hydrogen-like radial part (Z=1), Laguerre factors hardcoded."""
    x = 2.0 * r / n
    L = {
        (1, 0): np.ones_like(r),
        (2, 0): 2.0 - x,
        (2, 1): np.ones_like(r),
        (3, 0): 3.0 - 3.0 * x + x ** 2 / 2.0,
        (3, 1): 4.0 - x,
        (3, 2): 6.0 - x,
        (4, 0): 4.0 - 6.0 * x + 2.0 * x ** 2 - x ** 3 / 6.0,
        (4, 1): 10.0 - 5.0 * x + x ** 2 / 2.0,
        (4, 2): 6.0 - x,
        (4, 3): np.ones_like(r),
        (5, 0): 5.0 - 10.0 * x + 5.0 * x ** 2 - 5.0 * x ** 3 / 6.0 + x ** 4 / 24.0,
    }[(n, l)]
    return r ** l * L * np.exp(-r / n)


def angular(kind, th, ph):
    if kind == "2p":
        return np.cos(th)
    if kind == "3dz2":
        return 3.0 * np.cos(th) ** 2 - 1.0
    if kind == "4fxyz":
        return np.sin(th) ** 2 * np.cos(th) * np.sin(ph) * np.cos(ph)
    raise ValueError(kind)


def _ang_max(kind):
    th = np.linspace(0, np.pi, 240)
    ph = np.linspace(0, 2 * np.pi, 480)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    return float(np.max(angular(kind, TH, PH) ** 2))


def _rad_peak_rim(n, l, r_cut, alpha_floor):
    """r at which (rho_r/rho_r_max)^P_ALPHA = alpha_floor (visual rim)."""
    r = np.linspace(1e-4, r_cut, 8000)
    rho = r ** 2 * radial(n, l, r) ** 2
    pk = float(rho.max())
    tgt = pk * (alpha_floor / MAX_ALPHA) ** (1.0 / P_ALPHA)
    idx = np.where(rho >= tgt)[0]
    return float(r[idx[-1]])


def build_cloud(kind, n, l, npts, r_cut, r_vis, rng):
    """Importance-sample the 3D density; opacity = |psi|^2."""
    ang_max = _ang_max(kind)
    r_grid = np.linspace(1e-4, r_cut, 8000)
    rho_r = r_grid ** 2 * radial(n, l, r_grid) ** 2
    rad_max = float(rho_r.max())
    rho_max = rad_max * ang_max
    parts = []
    while sum(len(p) for p in parts) < npts:
        ch = 200_000
        P = rng.uniform(-1.0, 1.0, (ch, 3))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        P *= r_cut * rng.random((ch, 1)) ** (1.0 / 3.0)
        r = np.linalg.norm(P, axis=1)
        th = np.arccos(np.clip(P[:, 2] / r, -1.0, 1.0))
        ph = np.arctan2(P[:, 1], P[:, 0])
        rho = r ** 2 * radial(n, l, r) ** 2 * angular(kind, th, ph) ** 2
        keep = rng.random(ch) < rho / rho_max
        parts.append(P[keep])
    P = np.concatenate(parts)[:npts]
    r = np.linalg.norm(P, axis=1)
    th = np.arccos(np.clip(P[:, 2] / r, -1.0, 1.0))
    ph = np.arctan2(P[:, 1], P[:, 0])
    Y = angular(kind, th, ph)
    rho = r ** 2 * radial(n, l, r) ** 2 * Y ** 2
    r_floor = _rad_peak_rim(n, l, r_cut, ALPHA_RIM)
    scale = r_vis / r_floor
    rest = P * scale
    alpha = np.clip((rho / rho_max) ** P_ALPHA, 0.0, 1.0) * MAX_ALPHA
    col = np.where((Y >= 0)[:, None], GOLD_RGB, CYAN_RGB)
    rgba = np.concatenate([col, alpha[:, None]], axis=1)
    print(f"[build] {kind}: {npts} pts, rim r={r_floor:.2f} "
          f"-> scale {scale:.3f}")
    return {"centers": rest, "rgba": rgba, "kind": kind}


def sample_shell(n, l, npts, rng):
    r_max = 3.5 * n * n
    r = np.linspace(1e-4, r_max, 6000)
    rho = r ** 2 * radial(n, l, r) ** 2
    cdf = np.cumsum(rho)
    cdf /= cdf[-1]
    rr = np.interp(rng.random(npts), cdf, r)
    z = rng.uniform(-1.0, 1.0, npts)
    t = rng.uniform(0.0, 2.0 * np.pi, npts)
    s = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    P = np.stack([s * np.cos(t), s * np.sin(t), z], axis=1) * rr[:, None]
    rho_pt = rr ** 2 * radial(n, l, rr) ** 2
    return P, rr, rho_pt


def _palette(t):
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0) * (len(SURFACE_PALETTE) - 1)
    i = np.clip(np.floor(t).astype(int), 0, len(SURFACE_PALETTE) - 2)
    f = (t - i)[:, None]
    return SURFACE_PALETTE[i] * (1.0 - f) + SURFACE_PALETTE[i + 1] * f


def build_rb(rng):
    """All occupied hydrogen-like shells (Z=1); one layer per shell."""
    shells = []
    five_s = None
    for idx, (name, n, l, npts) in enumerate(RB_SHELLS):
        P, rr, rho = sample_shell(n, l, npts, rng)
        if name == "5s":
            five_s = (P, rr, rho)
        lo, hi = np.percentile(rr, [2, 98])
        t = np.clip((rr - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        base = _palette([0.06 + 0.88 * (idx / (len(RB_SHELLS) - 1)) ** 0.95])
        col = np.clip(base * (0.78 + 0.22 * t)[:, None], 0.0, 1.0)
        alpha = np.clip((rho / rho.max()) ** P_ALPHA, 0.0, 1.0) * MAX_ALPHA
        shells.append({"name": name, "P": P, "rr": rr,
                       "rgba": np.concatenate([col, alpha[:, None]], axis=1)})
    P5, rr5, rho5 = five_s
    r_grid = np.linspace(1e-4, 3.5 * 25, 8000)
    rho_r = r_grid ** 2 * radial(5, 0, r_grid) ** 2
    tgt = rho_r.max() * (ALPHA_RIM / MAX_ALPHA) ** (1.0 / P_ALPHA)
    r_floor5 = float(r_grid[np.where(rho_r >= tgt)[0][-1]])
    scale = R_RB_OUTER / r_floor5
    for sh in shells:
        sh["centers"] = sh["P"] * scale
    print(f"[build] Rb: {sum(len(s['P']) for s in shells)} pts, "
          f"global scale {scale:.4f} (5s rim r={r_floor5:.1f} a0)")
    return shells


def make_membrane(r, a, na=128, nu=10):
    """Flat 2p polar plot r(ang) = r*cos^2(ang) in the xz-plane."""

    def pt(ang, u):
        c = np.cos(ang)
        return np.array([r * c * c * u * np.sin(ang), 0.0,
                         r * c * c * u * np.cos(ang)])

    def edge(u):
        return 0.28 + 0.72 * (1.0 - u) ** 1.3

    angs = np.linspace(0.0, 2.0 * np.pi, na, endpoint=False)
    us = np.linspace(0.0, 1.0, nu)
    quads, cols = [], []
    for i in range(na):
        a0, a1 = angs[i], angs[(i + 1) % na]
        for j in range(nu - 1):
            u0, u1 = us[j], us[j + 1]
            quads.append([pt(a0, u0), pt(a0, u1), pt(a1, u1), pt(a1, u0)])
            c0 = GOLD_RGB if np.cos(a0) >= 0 else CYAN_RGB
            c1 = GOLD_RGB if np.cos(a1) >= 0 else CYAN_RGB
            cols.append([np.append(c0, edge(u0) * a),
                         np.append(c0, edge(u1) * a),
                         np.append(c1, edge(u1) * a),
                         np.append(c1, edge(u0) * a)])
    quads = np.array(quads, dtype=np.float64)
    colors = np.array(cols, dtype=np.float32)
    print(f"[build] membrane: {len(quads)} quads")
    return {"quads": quads, "colors": colors}


# ------------------------------------------------------------ mesh layers --
def cloud_layer(d):
    """Soft-sprite layer: per sprite = 4 CONE triangles (centre -> rim).

    Sprites are FLAT DISCS IN THE WORLD XY-PLANE (constant z per sprite):
    the mesh pass derives depth from unrotated world z, so constant-z sprites
    get an exact per-sprite depth and the back-to-front sort composites them
    like a painter with no partial-occlusion notches.  The camera's fixed
    54.7 deg elevation foreshortens every disc by the same cos(phi), so each
    sprite reads as a soft ellipse.

    Each sprite is 4 degenerate quads -> 4 fan triangles:
        qk: corners (c, c, r_out*d_{k+1}, r_out*d_k), alpha (w, w, 0, 0)
    i.e. a linear radial falloff from w at the centre to 0 at the rim, with
    NO internal hard edges (the old centre-diamond caused visible speckle).
    d_k are the sprite's 4 unit disk directions, rotated by a random angle
    per sprite so the fans don't align into a lattice.
    """
    n = len(d["centers"])
    seed = sum(ord(c) for c in d.get("kind", d.get("name", "cloud"))) % 2 ** 32
    rng = np.random.default_rng(seed)
    rot = rng.uniform(0.0, 2.0 * np.pi, n)
    ang = rot[:, None] + np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    dirs3 = np.concatenate(
        [np.cos(ang)[..., None], np.sin(ang)[..., None],
         np.zeros((n, 4, 1))], axis=2)                            # (n,4,3)

    a = d["rgba"][:, 3]
    levels = np.array([1.0, 1.0, 0.0, 0.0] * 4)
    base_alpha = (a[:, None] * levels[None, :]).astype(np.float32)  # (n,16)
    rgb = d["rgba"][:, :3].astype(np.float32)
    dither = rng.uniform(-1.0, 1.0, n) * 1e-5
    return {"kind": "cloud", "centers": d["centers"], "dirs": dirs3,
            "rgb": rgb, "base_alpha": base_alpha, "dither": dither,
            "half": SPRITE_R}


def membrane_layer(d):
    return {"kind": "mem", "base_quads": d["quads"].copy(),
            "base_ca": d["colors"][:, :, 3].copy(),
            "quads": d["quads"], "colors": d["colors"]}


def compose_batch(*layers):
    nq = 0
    for L in layers:
        if L["kind"] == "cloud":
            L["nq"] = 4 * len(L["centers"])
        else:
            L["nq"] = len(L["quads"])
        nq += L["nq"]
    # pre-fill each layer's slice with its BASE geometry/colors: membrane
    # frames only overwrite the alpha column per frame, so the RGB channels
    # must come from the base colors (a zeros prefill renders black ghosts)
    quads = np.zeros((nq, 4, 3), dtype=np.float64)
    colors = np.zeros((nq, 4, 4), dtype=np.float32)
    n = 0
    for L in layers:
        if L["kind"] == "mem":
            quads[n:n + L["nq"]] = L["quads"]
            colors[n:n + L["nq"]] = L["colors"]
        L["q0"], L["q1"] = n, n + L["nq"]
        n += L["nq"]
    b = MeshBatch()
    b.quads = quads
    b.colors = colors
    print(f"[mesh] batch: {n} quads / {n * 6} verts (cap 524288)")
    return b


# ------------------------------------------------------------ animations ---
def sstep(x):
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def camera_rotation(th_deg, phi_deg=PHI_DEG):
    """CE convention: R = Rx(-phi) @ Rz(-theta - 90); rows = camera axes."""
    th = np.radians(th_deg)
    phi = np.radians(phi_deg)
    a = -th - np.pi / 2.0
    ca, sa = np.cos(a), np.sin(a)
    cp, sp = np.cos(phi), np.sin(phi)
    Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]])
    return Rx @ Rz


def octant_mask(P, th_deg):
    """1 = keep, 0 = inside the camera-facing 90-90-90 corner."""
    th = np.radians(th_deg)
    phi = np.radians(PHI_DEG)
    v = np.array([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th),
                  np.cos(phi)])
    u1 = np.array([-np.sin(th), np.cos(th), 0.0])
    u2 = np.cross(v, u1)
    inside = (P @ v > 0.0) & (P @ u1 > 0.0) & (P @ u2 > 0.0)
    return (~inside).astype(np.float64)


class OrbitCam(Animation):
    """Real camera orbit: set_theta per frame (skill-verified recipe)."""

    def __init__(self, camera, t0_deg, t1_deg, **kw):
        self.camera, self.t0, self.t1 = camera, t0_deg, t1_deg
        super().__init__(Mobject(), rate_func=lambda a: a, **kw)

    def interpolate_mobject(self, alpha):
        self.camera.set_theta((self.t0 + (self.t1 - self.t0) * alpha) * DEGREES)


class CloudFrame(Animation):
    """Per-frame soft-sprite quads for one or more cloud layers.

    Sprite discs are world-xy-flat (constant z), so their per-frame rebuild
    is just the breathe scale; they are sorted back-to-front by world z
    (stable, with a per-sprite dither so coplanar sprites never flicker) and
    written into the batch.  The sort makes the mesh pipeline's LESS_OR_EQUAL
    depth test composite the cloud like a painter instead of occluding
    sprites randomly.
    """

    _CEN_OFF3 = None  # cone sprites: no centre quad

    def __init__(self, batch, layers, t0_deg, t1_deg, alpha_fns,
                 cut_strength_fn=None, breathes=None, frame_center=(0, 0, 0),
                 **kw):
        self.batch = batch
        self.layers = layers            # list of cloud layers
        self.t0, self.t1 = t0_deg, t1_deg
        self.alpha_fns = alpha_fns      # list of alpha(t) fns
        self.cut_strength_fn = cut_strength_fn
        self.breathes = breathes or [None] * len(layers)
        self.fc_z = frame_center[2]
        # snapshot the batch slice NOW: compose_batch re-stamps layer q0/q1 on
        # later plays, and this anim must keep the slice it was built for
        self._q0 = layers[0]["q0"]
        self._q1 = layers[-1]["q1"]
        super().__init__(Mobject(), rate_func=lambda a: a, **kw)

    def interpolate_mobject(self, alpha):
        th = self.t0 + (self.t1 - self.t0) * alpha
        quads_all, cols_all, keys = [], [], []
        for L, af, br in zip(self.layers, self.alpha_fns, self.breathes):
            s = 1.0
            if br is not None:
                ph0, amp, T = br
                s = 1.0 + amp * np.sin(2.0 * np.pi * (ph0 + alpha * self.run_time) / T)
            c = L["centers"] * s                                   # (n,3)
            rout3 = L["dirs"] * SPRITE_R
            rout_n = np.roll(rout3, -1, axis=1)
            cc = np.repeat(c[:, None, None, :], 4, axis=1)         # (n,4,1,3)
            # 4 cone triangles per sprite: (n,4,4,3) -> (4n,4,3)
            q = np.stack([cc, cc, rout_n[:, :, None, :],
                          rout3[:, :, None, :]], axis=2).reshape(-1, 4, 3)
            mult = np.full(len(c), af(alpha))
            if self.cut_strength_fn is not None:
                m = octant_mask(L["centers"], th)
                mult = mult * (1.0 - (1.0 - m) * self.cut_strength_fn(alpha))
            cols = np.zeros((len(c), 4, 4, 4), dtype=np.float32)
            cols[..., :3] = L["rgb"][:, None, None, :]
            cols[..., 3] = L["base_alpha"].reshape(-1, 4, 4) * mult[:, None, None]
            quads_all.append(q)
            cols_all.append(cols.reshape(-1, 4, 4))
            keys.append(L["centers"][:, 2] - self.fc_z + L["dither"])
        quads = np.concatenate(quads_all, axis=0)
        colors = np.concatenate(cols_all, axis=0)
        key = np.concatenate(keys)
        n_sprites = len(key)
        order = np.argsort(key, kind="stable")                     # far first
        block = (np.repeat(order * 4, 4) +
                 np.tile(np.arange(4), n_sprites))
        q0, q1 = self._q0, self._q1
        self.batch.quads[q0:q1] = quads[block]
        self.batch.colors[q0:q1] = colors[block]


class MembraneFrame(Animation):
    """Per-frame membrane quads/alpha for its slice of the batch."""

    def __init__(self, batch, layer, alpha_fn, scale_fn=lambda a: 1.0, **kw):
        self.batch = batch
        self.layer = layer
        self.alpha_fn, self.scale_fn = alpha_fn, scale_fn
        self._q0, self._q1 = layer["q0"], layer["q1"]
        super().__init__(Mobject(), rate_func=lambda a: a, **kw)

    def interpolate_mobject(self, alpha):
        q0, q1 = self._q0, self._q1
        self.batch.quads[q0:q1] = self.layer["base_quads"] * self.scale_fn(alpha)
        self.batch.colors[q0:q1, :, 3] = (
            self.layer["base_ca"] * self.alpha_fn(alpha)).astype(np.float32)


# ---------------------------------------------------------------- scene ----
class HydrogenOrbit(ThreeDScene):
    def setup(self):
        super().setup()
        rng = np.random.default_rng(SEED)
        print("[build] clouds ...")
        self.d2p = build_cloud("2p", 2, 1, N_PTS["2p"], R_CUT["2p"],
                               R_VIS["2p"], rng)
        self.d3d = build_cloud("3dz2", 3, 2, N_PTS["3dz2"], R_CUT["3dz2"],
                               R_VIS["3dz2"], rng)
        self.d4f = build_cloud("4fxyz", 4, 3, N_PTS["4fxyz"], R_CUT["4fxyz"],
                               R_VIS["4fxyz"], rng)
        self.rb_shells = build_rb(rng)
        self.mem_d = make_membrane(R_MEMB, 0.9)

        self.l2p = cloud_layer(self.d2p)
        self.l3d = cloud_layer(self.d3d)
        self.l4f = cloud_layer(self.d4f)
        self.l_mem = membrane_layer(self.mem_d)
        self.l_rb = [cloud_layer(sh) for sh in self.rb_shells]

        # ---- showcase opening HUD ----
        self.title = Text("The Hydrogen Atom", font="Times New Roman",
                          font_size=46, color=INK_HEX).to_corner(UL, buff=0.62)
        self.subtitle = Text("the Schrödinger equation, solved as a cloud",
                             font="Times New Roman", font_size=26,
                             color=DIM_HEX).next_to(
                                 self.title, DOWN, buff=0.12,
                                 aligned_edge=LEFT)
        self.eq = MathTex(
            r"i\hbar\, \frac{\partial}{\partial t} \Psi(r,t) = "
            r"\left[-\frac{\hbar^{2}}{2m}\nabla^{2} + V(r)\right] \Psi(r,t)",
            font_size=44, color=GOLD_HEX)
        self.eq.scale(7.2 / self.eq.width)
        self.eq_corner = self.eq.copy().scale(4.3 / self.eq.width).to_corner(
            UL, buff=0.62)

        self.label2p = MathTex("2p", font_size=34, color=INK_HEX).to_corner(
            DL, buff=0.62)
        self.label3d = MathTex(r"3d_{z^2}", font_size=34,
                               color=INK_HEX).to_corner(DL, buff=0.62)
        self.label4f = MathTex(r"4f_{xyz}", font_size=34,
                               color=INK_HEX).to_corner(DL, buff=0.62)
        self.label_rb = Text("³⁷Rb", font="Times New Roman", font_size=30,
                             color=INK_HEX).to_corner(DL, buff=0.62)
        self.caption_rb = Text("1s² 2s² 2p⁶ 3s² 3p⁶ 4s² 3d¹⁰ 4p⁶ 5s¹",
                               font="Times New Roman", font_size=22,
                               color=DIM_HEX).next_to(self.label_rb, 0,
                                                      buff=0.35)
        self.nucleus = Dot(radius=0.035, color=CREAM_HEX)

    def construct(self):
        win = MLWindow(1920, 1080)
        win.scene = self
        self.play = win.play
        self._patch_background(win)
        self.camera.background_color = BG_HEX
        self.camera.set_phi(PHI_DEG * DEGREES)
        self.camera.set_theta(THETA_START * DEGREES)
        fc_z = self.camera.frame_center[2]

        eq = self.eq
        self.add_fixed_in_frame_mobjects(eq, self.title, self.subtitle)
        l2p, l3d, l4f, l_mem, l_rb = (self.l2p, self.l3d, self.l4f,
                                      self.l_mem, self.l_rb)

        # 1. showcase opening --------------------------------- 0.0-3.6
        win.play(LaggedStart(FadeIn(self.title, run_time=0.9, rate_func=smooth),
                             FadeIn(self.subtitle, run_time=0.9,
                                    rate_func=smooth),
                             lag_ratio=0.22), run_time=1.12)
        win.play(FadeIn(eq, run_time=1.1, rate_func=smooth), run_time=1.1)
        win.play(Wait(0.6))
        # shrink to corner; title/subtitle leave ----------------- 2.6-3.6
        win.play(Transform(eq, self.eq_corner, rate_func=smooth),
                 FadeOut(self.title, run_time=0.8, rate_func=smooth),
                 FadeOut(self.subtitle, run_time=0.8, rate_func=smooth),
                 run_time=1.0)

        # 2. 2p membrane ------------------------------------------ 3.6-5.6
        b = compose_batch(l_mem)
        win.set_mesh(b)
        win.play(MembraneFrame(b, l_mem,
                               lambda a: sstep(a * 1.0 / 1.0),
                               lambda a: 0.9 + 0.1 * sstep(a * 1.0 / 1.0)),
                 run_time=1.0)
        win.play(Wait(1.0))

        # 3. morph to the real 3D 2p cloud ------------------------ 5.6-7.6
        b = compose_batch(l_mem, l2p)
        win.set_mesh(b)
        win.play(MembraneFrame(b, l_mem,
                               lambda a: 1.0 - sstep(a * 1.8 / 1.8),
                               lambda a: 1.0 - 0.04 * sstep(a * 1.8 / 1.8)),
                 CloudFrame(b, [l2p], THETA_START, THETA_START,
                            [lambda a: sstep(a * 1.8 / 1.8)],
                            frame_center=(0, 0, fc_z)),
                 run_time=1.8)
        win.play(Wait(0.2))

        # 4. dynamic 2p + camera orbit ---------------------------- 7.6-10.6
        b = compose_batch(l2p)
        win.set_mesh(b)
        self.add_fixed_in_frame_mobjects(self.nucleus)
        self.add_fixed_in_frame_mobjects(self.label2p)
        win.play(OrbitCam(self.camera, -150.0, -118.0),
                 CloudFrame(b, [l2p], -150.0, -118.0, [lambda a: 1.0],
                            breathes=[(7.6, BREATHE_AMP, BREATHE_PERIOD)],
                            frame_center=(0, 0, fc_z)),
                 Succession(Wait(0.4),
                            FadeIn(self.nucleus, run_time=0.4,
                                   rate_func=smooth),
                            FadeIn(self.label2p, run_time=0.5,
                                   rate_func=smooth)),
                 run_time=3.0)

        # 5a. 3d_z^2 -------------------------------------------- 10.6-13.0
        b = compose_batch(l2p, l3d)
        win.set_mesh(b)
        self.add_fixed_in_frame_mobjects(self.label3d)
        win.play(CloudFrame(b, [l2p, l3d], -118.0, -93.0,
                            [lambda a: 1.0 - sstep(a * 2.4 / 1.6),
                             lambda a: sstep(a * 2.4 / 1.6)],
                            breathes=[None, (10.6, BREATHE_AMP,
                                             BREATHE_PERIOD)],
                            frame_center=(0, 0, fc_z)),
                 OrbitCam(self.camera, -118.0, -93.0),
                 Succession(Wait(0.2),
                            FadeOut(self.label2p, run_time=0.4,
                                    rate_func=smooth),
                            FadeIn(self.label3d, run_time=0.5,
                                   rate_func=smooth)),
                 run_time=2.4)
        self.remove(self.label2p)

        # 5b. 4f_xyz -------------------------------------------- 13.0-15.4
        b = compose_batch(l3d, l4f)
        win.set_mesh(b)
        self.add_fixed_in_frame_mobjects(self.label4f)
        win.play(CloudFrame(b, [l3d, l4f], -93.0, -68.0,
                            [lambda a: 1.0 - sstep(a * 2.4 / 1.6),
                             lambda a: sstep(a * 2.4 / 1.6)],
                            breathes=[None, (13.0, BREATHE_AMP,
                                             BREATHE_PERIOD)],
                            frame_center=(0, 0, fc_z)),
                 OrbitCam(self.camera, -93.0, -68.0),
                 Succession(Wait(0.2),
                            FadeOut(self.label3d, run_time=0.4,
                                    rate_func=smooth),
                            FadeIn(self.label4f, run_time=0.5,
                                   rate_func=smooth)),
                 run_time=2.4)
        self.remove(self.label3d)

        # 6. 90-90-90 corner cut, camera rotating ---------------- 15.4-19.4
        b = compose_batch(l4f)
        win.set_mesh(b)
        win.play(OrbitCam(self.camera, -68.0, -30.0),
                 CloudFrame(b, [l4f], -68.0, -30.0, [lambda a: 1.0],
                            cut_strength_fn=lambda a: sstep(a * 4.0 / 1.2),
                            breathes=[(15.4, BREATHE_AMP, BREATHE_PERIOD)],
                            frame_center=(0, 0, fc_z)),
                 run_time=4.0)

        # 7a. heal the cut, fade the 4f out ----------------------- 19.4-22.0
        b = compose_batch(l4f)
        win.set_mesh(b)
        self.add_fixed_in_frame_mobjects(self.label_rb, self.caption_rb)
        win.play(OrbitCam(self.camera, -30.0, -5.2),
                 CloudFrame(b, [l4f], -30.0, -5.2,
                            [lambda a: 1.0 - sstep(a * 2.6 / 2.6)],
                            cut_strength_fn=lambda a: 1.0 - sstep(a * 2.6 / 1.4),
                            frame_center=(0, 0, fc_z)),
                 Succession(Wait(0.6),
                            FadeOut(self.label4f, run_time=0.4,
                                    rate_func=smooth),
                            FadeIn(self.label_rb, run_time=0.5,
                                   rate_func=smooth),
                            FadeIn(self.caption_rb, run_time=0.5,
                                   rate_func=smooth)),
                 run_time=2.6)
        self.remove(self.label4f)

        # 7b. combine shells -> 37-Rb ----------------------------- 22.0-26.2
        b = compose_batch(*l_rb)
        win.set_mesh(b)
        t_stagger = [0.3 + i * 0.18 for i in range(len(l_rb))]
        win.play(OrbitCam(self.camera, -5.2, 35.0),
                 CloudFrame(b, l_rb, -5.2, 35.0,
                            [lambda a, ti=ti: sstep((a * 4.2 - ti) / 0.6)
                             for ti in t_stagger],
                            frame_center=(0, 0, fc_z)),
                 run_time=4.2)

        # settle, rotating ----------------------------------------- 26.2-28.2
        b = compose_batch(*l_rb)
        win.set_mesh(b)
        win.play(OrbitCam(self.camera, 35.0, 58.0),
                 CloudFrame(b, l_rb, 35.0, 58.0, [lambda a: 1.0] * len(l_rb),
                            breathes=[(26.2, RB_BREATHE_AMP,
                                       RB_BREATHE_PERIOD)] * len(l_rb),
                            frame_center=(0, 0, fc_z)),
                 run_time=2.0)

        # fade out ------------------------------------------------- 28.2-30.0
        b = compose_batch(*l_rb)
        win.set_mesh(b)
        fade = lambda a: 1.0 - sstep(a * 1.8 / 1.8)  # noqa: E731
        win.play(OrbitCam(self.camera, 58.0, 70.0),
                 CloudFrame(b, l_rb, 58.0, 70.0, [fade] * len(l_rb),
                            frame_center=(0, 0, fc_z)),
                 FadeOut(eq, run_time=1.8, rate_func=smooth),
                 FadeOut(self.label_rb, run_time=1.8, rate_func=smooth),
                 FadeOut(self.caption_rb, run_time=1.8, rate_func=smooth),
                 FadeOut(self.nucleus, run_time=1.8, rate_func=smooth),
                 run_time=1.8)
        win.close()

    @staticmethod
    def _patch_background(win):
        fn = win.dll.Render_SetBackgroundColor
        fn.restype = None
        fn.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
        win._set_background = fn


if __name__ == "__main__":
    r = fast_record_scene(HydrogenOrbit, "hydrogen_orbit_30s.mp4",
                          overwrite=True)
    print(r["files"])
