r"""Boy's surface — 30 s film (RTM).

Timeline (brief):
  0-5    create the Moebius strip: annulus {r_min <= r <= 1} grows from the
         folded boundary circle (r_min: 1 -> R0).  RP2 minus a cap disk.
  5-10   close it: the disk {r <= r_min} fills in from the pole (r_min: R0 -> 0)
         -> the whole Boy surface; the camera starts orbiting.
  10-15  nothing new (camera keeps rotating).
  15     an INVISIBLE plane appears at the top (clip active: keep n.P <= d).
  15-20  the plane moves to the middle = the horizontal plane through the
         TRIPLE POINT (origin).  The top becomes invisible; the surviving
         bottom half is the Moebius band (boundary = the 3-loop cut curve).
  20-25  the plane rotates about the x-axis (an axis in the plane -> "small
         angle with the plane"), pivoting around the triple point.
  25-30  the plane moves out along its normal; the whole surface returns.

Geometry: Bryant-Kusner parametrization (boy.py); analytic normals via the
complex chain rule (gate.py).  Shading: house style (rtm-style-params.md):
two-sided |n.L| (the surface is non-orientable), ambient, specular, rim,
base colour along the chart radius.
"""
import os
import sys

import numpy as np

import real_time_manim as rtm
from real_time_manim import Animation, MeshBatch, MLWindow, Mobject, ThreeDScene

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boy
from gate import ders

# ----------------------------------------------------------------------
# knobs
# ----------------------------------------------------------------------
NR = 160            # r cells
NPHI = 320          # phi cells
SCALE = 2.0         # world scale
PHI_DEG = 62.0      # camera elevation
FPS = 60
TOTAL = 30.0

QA_THETA = None     # degrees; overrides theta_of(t) when set (QA stills)

PHI_SWEEP_END = 2.8   # seconds: the ribbon finishes its sweep, then widens
N_EXTRA = 8000        # extra quad slots for the exact-clip pieces

R0 = boy.R0
Z_TOP = 0.4426      # global z max (computed)
TILT_DEG = 25.0     # plane tilt about the x-axis (pivoted at the origin)

LIGHT = np.array([-0.72, -0.28, 0.63])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
AMBIENT = 0.20
DIFFUSE = 0.95
SPEC_POW = 38.0
SPEC_GAIN = 0.30
SPEC_COL = np.array([1.0, 0.94, 0.82])
RIM_POW = 4.0
RIM_GAIN = 0.30
RIM_COL = np.array([0.32, 0.80, 0.78])

# 3D surface palette (house style), sampled along 0.06 + 0.88 r^0.95
_PAL = np.array([
    [0.020, 0.055, 0.115], [0.040, 0.180, 0.280], [0.050, 0.420, 0.520],
    [0.220, 0.720, 0.720], [0.950, 0.760, 0.380], [1.000, 0.940, 0.820]])


def _sample_palette(t):
    t = np.clip(t, 0.0, 1.0)
    x = t * (len(_PAL) - 1)
    i = np.minimum(x.astype(int), len(_PAL) - 2)
    f = (x - i)[..., None]
    return _PAL[i] * (1 - f) + _PAL[i + 1] * f


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def qstart(x):
    """Quick-start ease-out: rate'(0)=0.6, rate'(1)=0, monotone on [0,1].

    smoothstep starts with zero velocity, so every phase began with a
    sub-pixel crawl (measured: mean|d| 0.014 over the first 0.25 s) and the
    film read as laggy.  qstart gives each phase a real start velocity and
    a gentle settle at the end.
    """
    x = np.clip(x, 0.0, 1.0)
    return 0.6 * x + 1.8 * x * x - 1.4 * x * x * x


def qfast(x):
    """Very quick start (rate'(0)=1.0) with residual end velocity 0.35 —
    the sweep opens at full speed and hands off to the widening without
    stopping."""
    x = np.clip(x, 0.0, 1.0)
    return x + 0.65 * x * x - 0.65 * x * x * x


def qw(x):
    """rate'(0)=0.35 (picks up from the sweep's end), rate'(1)=0.3 (hands
    off to the cap) — the widen never stops."""
    x = np.clip(x, 0.0, 1.0)
    return 0.35 * x + 2.0 * x * x - 1.35 * x * x * x


def qc(x):
    """rate'(0)=0.3 (picks up from the widen), rate'(1)=0 — the cap closes
    the surface and comes to rest (the construction's natural completion)."""
    x = np.clip(x, 0.0, 1.0)
    return 0.3 * x + 2.4 * x * x - 1.7 * x * x * x


# ----------------------------------------------------------------------
# per-timeline state (pure functions of t)
# ----------------------------------------------------------------------
def r_min(t):
    R_MIN0 = 0.85          # ribbon width while the sweep draws the band
    if t < PHI_SWEEP_END:
        return R_MIN0
    if t <= 5.0:
        u = (t - PHI_SWEEP_END) / (5.0 - PHI_SWEEP_END)
        return R_MIN0 - (R_MIN0 - R0) * qw(u)
    if t <= 10.0:
        return R0 * (1.0 - qc((t - 5.0) / 5.0))
    return 0.0


def phi_max(t):
    """Sweep: the ribbon draws itself around the core loop, 0 -> 2pi."""
    if t <= 0.0:
        return 0.0
    if t < PHI_SWEEP_END:
        return 2.0 * np.pi * qfast(t / PHI_SWEEP_END)
    return 2.0 * np.pi


def theta_of(t):
    if t <= 5.0:
        return 0.0
    if t <= 15.0:
        return np.pi * qstart((t - 5.0) / 10.0)
    return np.pi


def plane_of(t):
    """(nhat, d) in SCALED units; None before t=15.  Keep n.P <= d."""
    if t < 15.0:
        return None
    if t <= 20.0:
        u = (t - 15.0) / 5.0                      # linear descent
        d = Z_TOP * SCALE - u * (Z_TOP * SCALE - 0.0)
        return np.array([0.0, 0.0, 1.0]), d
    if t <= 25.0:
        b = np.radians(TILT_DEG) * qstart((t - 20.0) / 5.0)
        return np.array([0.0, -np.sin(b), np.cos(b)]), 0.0
    b = np.radians(TILT_DEG)
    nhat = np.array([0.0, -np.sin(b), np.cos(b)])
    u = qstart((t - 25.0) / 5.0)
    return nhat, u * D_OUT          # D_OUT > 0: exit up along the normal


D_OUT = 1.9 * SCALE   # overwritten at build time from the actual grid


# ----------------------------------------------------------------------
# mesh builder
# ----------------------------------------------------------------------
class BoyMesh:
    def __init__(self):
        r = np.linspace(0.0, 1.0, NR + 1)
        phi = np.linspace(0.0, 2 * np.pi, NPHI + 1)
        R, PH = np.meshgrid(r, phi, indexing="ij")
        w = R * np.exp(1j * PH)
        P, Pw, Pwb = ders(w)

        # analytic frame: Pr = Pw e^{i phi} + Pwb e^{-i phi},
        #                  Pp = i r (Pw e^{i phi} - Pwb e^{-i phi})
        eip = np.exp(1j * PH)
        eim = np.exp(-1j * PH)
        Pr = Pw * eip[..., None] + Pwb * eim[..., None]
        Pp = 1j * R[..., None] * (Pw * eip[..., None] - Pwb * eim[..., None])
        N = np.cross(Pr, Pp).real
        nm = np.linalg.norm(N, axis=-1, keepdims=True)
        # pole row: the chart frame degenerates (r=0).  The real Jacobian
        # at w=0 is d/dx = (0,6,0), d/dy = (6,0,0) -> normal = -zhat.
        N[0, :] = np.array([0.0, 0.0, -1.0])
        nm[0] = 1.0
        N = N / np.maximum(nm, 1e-9)

        self.P = P * SCALE
        self.N = N
        self.r = r
        self.phi = phi

        global D_OUT
        b = np.radians(TILT_DEG)
        n_out = np.array([0.0, -np.sin(b), np.cos(b)])
        D_OUT = float((self.P @ n_out).max()) + 0.5
        print("D_OUT =", round(D_OUT, 3))

        qi = boy.quad_indices(NR, NPHI)          # (Q,4) flat vertex indices
        self.qi = qi
        self.base_quads = self.P.reshape(-1, 3)[qi]     # (Q,4,3) scaled
        self.centers = self.base_quads.mean(axis=1)     # quad centres
        # per-quad centre radius / phi in chart units (creation masks)
        rc = R.reshape(-1)[qi].mean(axis=1)
        self.rc = rc
        self.phi_c = PH.reshape(-1)[qi].mean(axis=1)
        # per-quad chart cell bounds (rows i, cols j)
        I = qi[:, 0] // (NPHI + 1)
        J = qi[:, 0] % (NPHI + 1)
        self.rlo = r[I]
        self.rhi = r[I + 1]
        self.plo = phi[J]
        self.phi_hi = phi[J + 1]
        self.base_col = _sample_palette(0.06 + 0.88 * R ** 0.95)  # (nr,np,3)
        self.collapsed = self.base_quads[0, 0].copy()  # any fixed point

        self.nq = qi.shape[0]
        self.batch = MeshBatch()
        self.batch.append(
            np.zeros((self.nq + N_EXTRA, 4, 3), dtype=np.float64),
            np.zeros((self.nq + N_EXTRA, 4, 4), dtype=np.float32),
        )

    def shade(self, eye):
        """Per-quad-corner colours under the current camera.  eye: world."""
        V = self.P - eye[None, None, :]
        V /= np.linalg.norm(V, axis=-1, keepdims=True)
        H = LIGHT + V
        H /= np.maximum(np.linalg.norm(H, axis=-1, keepdims=True), 1e-9)
        nl = np.abs((self.N * LIGHT).sum(-1))
        nh = np.abs((self.N * H).sum(-1))
        nv = np.abs((self.N * V).sum(-1))
        dif = nl ** 0.85
        col = (self.base_col * (AMBIENT + DIFFUSE * dif)[..., None])
        col = col + (nh ** SPEC_POW)[..., None] * SPEC_GAIN * SPEC_COL
        col = col + ((1.0 - nv) ** RIM_POW)[..., None] * RIM_GAIN * RIM_COL
        col = np.clip(col, 0.0, 1.0)
        return col.reshape(-1, 3)[self.qi]            # (Q,4,3)

    def fill(self, t, eye):
        """Write quads/colors for wall time t.  Collapsed = invisible.

        Construction phases (0-15 s) clip cells EXACTLY in chart space at
        the moving boundaries (the sweep front phi = phi_max(t) and the
        inner edge r = r_min(t)): the kept part of a straddling cell is a
        rectangle whose new vertices lie ON the boundary, so the ribbon
        front and the band edge advance sub-pixel per frame instead of
        popping a whole quad column at a time (the earlier 'lagging').
        Plane phase (15-30 s): exact Sutherland-Hodgman clip at the plane.
        No alpha anywhere: hidden pieces are collapsed.
        """
        qout = self.batch.quads
        q = self.base_quads
        col = self.shade(eye).astype(np.float32)        # (nq,4,3)
        full = np.zeros((self.nq + N_EXTRA, 4, 4), dtype=np.float32)
        full[:self.nq, :, :3] = col
        full[:, :, 3] = 1.0

        m = 0
        pl = plane_of(t)
        if pl is None:
            rmin, pmax = r_min(t), phi_max(t)
            full_in = (self.rlo >= rmin) & (self.phi_hi <= pmax)
            strad = (self.rhi > rmin) & (self.plo < pmax) & ~full_in
            qout[:self.nq] = np.where(full_in[:, None, None],
                                      q, q[:, :1, :])
            P, C = self._clip_chart(q, col, t, strad)
            m = min(len(P), N_EXTRA)
            qout[self.nq:self.nq + m] = P[:m]
            full[self.nq:self.nq + m, :, :3] = C[:m]
        else:
            nhat, d = pl
            sv = q @ nhat - d                           # (nq,4) signed dists
            hid = sv > 0
            nhid = hid.sum(axis=1)
            keep = (nhid == 0)
            qout[:self.nq] = np.where(keep[:, None, None], q, q[:, :1, :])
            P, C = self._clip(q, col, sv, hid, nhid)
            m = min(len(P), N_EXTRA)
            qout[self.nq:self.nq + m] = P[:m]
            full[self.nq:self.nq + m, :, :3] = C[:m]
        qout[self.nq + m:self.nq + N_EXTRA] = self.collapsed
        self.batch.colors[:] = full

    def _clip_chart(self, q, col, t, strad):
        """Exact clip of boundary cells in chart space.

        The visible region is {r >= r_min} x {phi <= phi_max}; a straddling
        cell keeps the rectangle [max(r0,rmin), r1] x [phi0, min(phi1,
        phi_max)], with interpolated positions/colors on the boundary
        (the interpolation lies exactly on the mesh's own edge polylines).
        """
        st = np.where(strad)[0]
        if len(st) == 0:
            return np.zeros((0, 4, 3)), np.zeros((0, 4, 3))
        rmin, pmax = r_min(t), phi_max(t)
        fr = np.clip((rmin - self.rlo[st]) / (self.rhi[st] - self.rlo[st]),
                     0.0, 1.0)[:, None]
        fp = np.clip((pmax - self.plo[st]) / (self.phi_hi[st] - self.plo[st]),
                     0.0, 1.0)[:, None]
        BL = q[st, 0]
        BR = q[st, 1]
        TR = q[st, 2]
        TL = q[st, 3]
        cBL = col[st, 0]
        cBR = col[st, 1]
        cTR = col[st, 2]
        cTL = col[st, 3]
        B = BL + fr * (BR - BL)                        # (r_m, phi_j)
        R = BR + fp * (TR - BR)                        # (r_1, phi_m)
        Cm = (BL * (1 - fr) * (1 - fp) + BR * fr * (1 - fp)
              + TR * fr * fp + TL * (1 - fr) * fp)     # (r_m, phi_m)
        cB = cBL + fr * (cBR - cBL)
        cR = cBR + fp * (cTR - cBR)
        cCm = (cBL * (1 - fr) * (1 - fp) + cBR * fr * (1 - fp)
               + cTR * fr * fp + cTL * (1 - fr) * fp)
        P = np.stack([B, BR, R, Cm], axis=1)
        C = np.stack([cB, cBR, cR, cCm], axis=1)
        return P, C

    def _clip(self, q, col, sv, hid, nhid):
        """Exact Sutherland-Hodgman clip of the straddling quads against
        the halfspace {s <= 0}.  Returns (M,4,3) positions and colours
        (triangles as degenerate quads, corner order preserved)."""
        P = []
        C = []

        def edge_x(a, b, idx):
            """intersection on edge a->b; a visible (s<=0), b hidden (s>0)."""
            t = sv[idx, a] / (sv[idx, a] - sv[idx, b])
            p = q[idx, a] + t[:, None] * (q[idx, b] - q[idx, a])
            c = col[idx, a] + t[:, None] * (col[idx, b] - col[idx, a])
            return p, c

        def emit_tri(a, b, c, idx):
            P.append(np.stack([q[idx, a], q[idx, b], q[idx, c],
                               q[idx, c]], axis=1))
            C.append(np.stack([col[idx, a], col[idx, b], col[idx, c],
                               col[idx, c]], axis=1))

        for k in range(4):
            k1, k2, k3 = (k + 1) % 4, (k + 2) % 4, (k + 3) % 4
            # 1 hidden corner k -> pentagon (3 triangles)
            idx = np.where(nhid == 1)[0]
            if len(idx):
                m1 = hid[idx, k]
                if m1.any():
                    i = idx[m1]
                    xa, ca = edge_x(k3, k, i)     # edge k3->k
                    xb, cb = edge_x(k1, k, i)     # edge k1->k
                    emit_tri(k1, k2, k3, i)
                    P.append(np.stack([q[i, k1], q[i, k3], xa, xa], axis=1))
                    C.append(np.stack([col[i, k1], col[i, k3], ca, ca],
                                      axis=1))
                    P.append(np.stack([q[i, k1], xa, xb, xb], axis=1))
                    C.append(np.stack([col[i, k1], ca, cb, cb], axis=1))
            # 2 adjacent hidden {k, k1} -> quad
            idx2 = np.where(nhid == 2)[0]
            if len(idx2):
                m2 = hid[idx2, k] & hid[idx2, k1]
                if m2.any():
                    i = idx2[m2]
                    xa, ca = edge_x(k3, k, i)     # edge k3->k
                    xb, cb = edge_x(k2, k1, i)    # edge k2->k1
                    P.append(np.stack([q[i, k2], q[i, k3], xa, xb], axis=1))
                    C.append(np.stack([col[i, k2], col[i, k3], ca, cb],
                                      axis=1))
                # 2 opposite hidden {k, k2} -> hexagon (4 triangles)
                m3 = hid[idx2, k] & hid[idx2, k2]
                if m3.any():
                    i = idx2[m3]
                    x1, c1 = edge_x(k1, k, i)     # edge k->k1
                    x2, c2 = edge_x(k1, k2, i)    # edge k1->k2
                    x3, c3 = edge_x(k3, k2, i)    # edge k2->k3
                    x4, c4 = edge_x(k3, k, i)     # edge k3->k
                    qk1 = q[i, k1]
                    ck1 = col[i, k1]
                    qk3 = q[i, k3]
                    ck3 = col[i, k3]
                    P.append(np.stack([qk1, x2, x3, x3], axis=1))
                    C.append(np.stack([ck1, c2, c3, c3], axis=1))
                    P.append(np.stack([qk1, x3, qk3, qk3], axis=1))
                    C.append(np.stack([ck1, c3, ck3, ck3], axis=1))
                    P.append(np.stack([qk1, qk3, x4, x4], axis=1))
                    C.append(np.stack([ck1, ck3, c4, c4], axis=1))
                    P.append(np.stack([qk1, x4, x1, x1], axis=1))
                    C.append(np.stack([ck1, c4, c1, c1], axis=1))
            # 3 hidden, k visible -> triangle
            idx3 = np.where(nhid == 3)[0]
            if len(idx3):
                m4 = ~hid[idx3, k]
                if m4.any():
                    i = idx3[m4]
                    x1, c1 = edge_x(k, k1, i)     # edge k->k1
                    x2, c2 = edge_x(k, k3, i)     # edge k->k3
                    P.append(np.stack([q[i, k], x1, x2, x2], axis=1))
                    C.append(np.stack([col[i, k], c1, c2, c2], axis=1))
        if not P:
            return (np.zeros((0, 4, 3)), np.zeros((0, 4, 3)))
        return np.concatenate(P, axis=0), np.concatenate(C, axis=0)


# ----------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------
class BoyAnimation(Animation):
    def __init__(self, carrier, mesh, camera, run_time=TOTAL, **kwargs):
        super().__init__(carrier, run_time=run_time, **kwargs)
        self.mesh = mesh
        self.camera = camera

    def interpolate_mobject(self, alpha):
        t = alpha * self.run_time
        th = theta_of(t) if QA_THETA is None else np.radians(QA_THETA)
        self.camera.theta_tracker.set_value(th)
        R = self.camera.get_rotation_matrix()
        eye = R.T @ np.array([0.0, 0.0, self.camera.get_focal_distance()])
        self.mesh.fill(t, eye)


class BoyScene(ThreeDScene):
    def construct(self):
        win = MLWindow(1920, 1080)
        win.scene = self
        camera = self.camera
        camera.set_phi(np.radians(PHI_DEG))
        camera.set_theta(0.0)
        camera.frame_center = np.array([0.0, 0.0, -0.78 * SCALE])

        mesh = BoyMesh()
        R = camera.get_rotation_matrix()
        eye = R.T @ np.array([0.0, 0.0, camera.get_focal_distance()])
        mesh.fill(0.0, eye)
        win.set_mesh(mesh.batch)

        carrier = Mobject()
        self.add(carrier)
        win.play(BoyAnimation(carrier, mesh, camera, run_time=TOTAL),
                 run_time=TOTAL)
        win.close()


if __name__ == "__main__":
    import argparse
    from real_time_manim.frames import FrameServer
    from real_time_manim.record import fast_record_scene

    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", default="",
                    help="comma list of times for FrameServer PNGs")
    ap.add_argument("--record", default="",
                    help="mp4 output path (FrameServer pipeline)")
    ap.add_argument("--fast", action="store_true",
                    help="use fast_record_scene instead")
    ap.add_argument("--theta", type=float, default=None,
                    help="QA: override camera theta in degrees")
    args = ap.parse_args()

    if args.theta is not None:
        QA_THETA = args.theta

    if args.stills:
        srv = FrameServer(BoyScene)
        outdir = os.path.expanduser("~/gallary/boy_surface/stills")
        os.makedirs(outdir, exist_ok=True)
        for ts in args.stills.split(","):
            t = float(ts)
            tag = "" if args.theta is None else "_th%g" % args.theta
            p = os.path.join(outdir, "still_%05.2f%s.png" % (t, tag))
            srv.save_frame(t, p)
            print("saved", p)
        srv.close()
    elif args.record:
        out = os.path.expanduser(args.record)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if args.fast:
            r = fast_record_scene(BoyScene, out, overwrite=True)
            print("files:", r["files"])
        else:
            import record
            record.render(BoyScene, out)
    else:
        print("nothing to do; use --stills or --record")
