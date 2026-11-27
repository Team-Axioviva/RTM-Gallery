#!/usr/bin/env python3
"""
Langton's ant -- 30 s, 1080p60, RTM.

Story: grid + ant + rule card fade in; five didactic steps showing the
colour -> arrow -> move loop (step 4 is the black-cell LEFT turn, so both
rule branches get a slow show); then a monotonically accelerating run until
the chaotic pattern emerges and the ant locks into its 104-step highway;
the final pull zooms out to the trail's full scope against an infinite
grid (the grid always extends past the frame - no visible board edge).

Everything time-varying is a pure function of t (offline sim + snapshots +
step schedule + precomputed smoothed view path), so FrameServer stills and
the recording agree exactly.  One MeshBatch holds every cell quad (per-frame
colours); the ant, arrows and rule card are painter mobjects on top.

Rule: on a white cell turn LEFT 90 deg, flip the cell to black, move one
cell; on a black cell turn RIGHT 90 deg, flip to white, move one cell.
(white/black roles exchanged relative to the classic rule - the mirrored
ant.) Visual: black background - unvisited cells render dark, the trail
renders light; the rule card states the rule as seen on screen.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import real_time_manim  # noqa: F401  (must precede any manim import)
from real_time_manim import (Animation, Arrow, BLACK, BOLD, DEGREES, DOWN,
                             LEFT, RIGHT, RoundedRectangle, Scene, Square,
                             Text, UR, VGroup, VMobject, WHITE, YELLOW,
                             linear)
from real_time_manim.mesh import MeshBatch
from real_time_manim.record import fast_record_scene
from real_time_manim.animations.base import set_anim_opacity
from real_time_manim.vulkan_bind import FadeIn, MLWindow

# ---------------------------------------------------------------- knobs
S_TOTAL = 17000              # steps simulated in total
SNAP = 32                    # state snapshot stride (steps)
FPS = 60
DUR = 30.0

T_INTRO = 1.6                # intro beat (grid fade-in, ant, rule card)
ND = 5                       # didactic steps (0..4; step 4 = left turn)
T_FLIP_DUR = 0.40            # colour-fade beat
T_ARROW_DUR = 0.55           # arrow + turn beat
T_MOVE_DUR = 0.55            # move beat
DID_T = T_FLIP_DUR + T_ARROW_DUR + T_MOVE_DUR
T_CS = T_INTRO + ND * DID_T  # accelerating phase starts (9.1)
T_CE = 20.0                  # accelerating phase ends
T_ZOOM = 26.0                # final pull to full scope starts
T_CARD_FADE = 2.0            # caption fade-out duration (starts at T_CS)

S_MAX = 0.30                 # opening cell size, world units (~45-cell window)
GAP = 0.07                   # inset per cell -> grid lines
TRACK_PAD = 6.0              # view window padding around the trail bbox
BIAS = 0.30                  # view centre pulled toward the ant while tracking
TRAIL_PAD = 8.0              # cells of trail margin kept in the final view
EDGE_MARGIN = 0.4            # world units of grid that must reach past the
                             # frame at the final zoom (grid reads infinite)
TAU_TRACK = 1.0              # view smoothing time constant, tracking phase
TAU_FINAL = 0.9              # view smoothing time constant, final pull

KN_U = [0.0, 0.30, 0.55, 0.75, 1.0]   # accelerating phase: rate ramp knots
KN_F = [2.0, 45.0, 160.0, 420.0, 1000.0]   # steps/s
F0_D = 1000.0                # highway phase: ramp start rate (steps/s)

ANT_MIN = 0.10               # smallest ant size, world units (~7.6 px)

# Visual mapping (user: black background): unvisited cells render DARK
# (the background), flipped cells render LIGHT (the trail).
COL_UNVIS = np.array([0.10, 0.115, 0.13], dtype=np.float32)
COL_FLIPPED = np.array([0.88, 0.90, 0.93], dtype=np.float32)
COL_ANT = "#F25238"
ARROW_COL = "#FFD129"

# ant triangle template (cell units, pointing +x = east)
TEMPL = np.array([[0.62, 0.0], [-0.45, 0.40], [-0.45, -0.40]],
                 dtype=np.float64)
ANG = np.array([90.0, 0.0, -90.0, 180.0])   # heading N,E,S,W -> degrees

# ---------------------------------------------------------------- sim
_SIM = None


def build_sim():
    # --- Langton's ant on the infinite grid (headings 0=N,1=E,2=S,3=W)
    MV = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)], dtype=np.int64)
    x = y = 0
    h = 0
    cells = {}
    pos = np.zeros((S_TOTAL + 1, 2), dtype=np.int64)   # position at step k
    hd = np.zeros(S_TOTAL + 1, dtype=np.int64)         # heading during step k
    flips = np.zeros((S_TOTAL, 2), dtype=np.int64)
    for k in range(S_TOTAL):
        c = cells.get((x, y), 0)
        if c == 0:          # white: turn LEFT, flip to black
            h = (h - 1) % 4
            cells[(x, y)] = 1
        else:               # black: turn RIGHT, flip to white
            h = (h + 1) % 4
            cells[(x, y)] = 0
        flips[k] = (x, y)
        x += MV[h][0]
        y += MV[h][1]
        pos[k + 1] = (x, y)
        hd[k + 1] = h

    # --- bounding boxes (pos covers every visited cell: pos[k]=flips[k-1])
    bx0 = np.minimum.accumulate(pos[:, 0])
    bx1 = np.maximum.accumulate(pos[:, 0])
    by0 = np.minimum.accumulate(pos[:, 1])
    by1 = np.maximum.accumulate(pos[:, 1])
    x_lo, x_hi = int(bx0[-1]), int(bx1[-1])
    y_lo, y_hi = int(by0[-1]), int(by1[-1])
    bb_w = x_hi - x_lo + 1
    bb_h = y_hi - y_lo + 1
    # Final view: the whole trail + TRAIL_PAD margin. The grid must extend
    # past the frame at that scale (infinite-grid look, no visible board
    # edge at ANY zoom), so size it to cover the full frame + EDGE_MARGIN
    # on every side at the final scale.
    S_FINAL = min(13.6 / (bb_w + 2.0 * TRAIL_PAD),
                  7.6 / (bb_h + 2.0 * TRAIL_PAD))
    need_w = (14.22 + 2.0 * EDGE_MARGIN) / S_FINAL
    need_h = (8.0 + 2.0 * EDGE_MARGIN) / S_FINAL
    PAD_X = int(np.ceil((need_w - bb_w) / 2.0))
    PAD_Y = int(np.ceil((need_h - bb_h) / 2.0))
    X_LO = x_lo - PAD_X
    Y_LO = y_lo - PAD_Y
    W = bb_w + 2 * PAD_X
    H = bb_h + 2 * PAD_Y
    Q = W * H
    assert Q * 6 <= MeshBatch.MAX_MESH_VERTS, "grid over mesh cap: %d quads" % Q

    FLATQ = ((flips[:, 1] - Y_LO) * W + (flips[:, 0] - X_LO)).astype(np.int64)

    # --- state snapshots every SNAP steps (frame access: base + <SNAP flips)
    nsnap = S_TOTAL // SNAP + 1
    snaps = np.zeros((nsnap, Q), dtype=np.uint8)
    state = np.zeros(Q, dtype=np.uint8)
    i = 1
    for k in range(S_TOTAL):
        state[FLATQ[k]] ^= 1
        if (k + 1) % SNAP == 0:
            snaps[i] = state
            i += 1

    # --- didactic pre/post colours of the first ND flips
    pre_col, post_col, post_state = [], [], []
    st = {}
    for j in range(ND):
        key = (int(flips[j, 0]), int(flips[j, 1]))
        c = st.get(key, 0)
        pre_col.append(COL_FLIPPED if c else COL_UNVIS)
        nc = 1 - c
        post_col.append(COL_FLIPPED if nc else COL_UNVIS)
        post_state.append(nc)
        st[key] = nc

    # --- step schedule: steps as a pure function of t (monotone acceleration)
    TG = np.linspace(T_CS, T_CE, 10001)
    U = (TG - T_CS) / (T_CE - T_CS)
    F = np.interp(U, KN_U, KN_F)
    NG = ND + np.concatenate(
        [[0.0], np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(TG))])
    K_C = float(np.interp(T_CE, TG, NG))
    F1_D = 2.0 * (S_TOTAL - K_C) / 10.0 - F0_D
    assert F1_D >= F0_D, "phase-D ramp negative: F1_D=%.1f" % F1_D

    def n_of_t(t):
        t = np.asarray(t, dtype=np.float64)
        out = np.full(t.shape, float(ND), dtype=np.float64)
        mid = (t >= T_CS) & (t < T_CE)
        if mid.any():
            out[mid] = np.interp(t[mid], TG, NG)
        hi = t >= T_CE
        if hi.any():
            d = t[hi] - T_CE
            out[hi] = K_C + F0_D * d + 0.5 * (F1_D - F0_D) * d * d / 10.0
        return out

    # --- cell corner positions in cell units (row-major: q = row*W + col)
    cols = np.arange(W, dtype=np.float64)
    rows = np.arange(H, dtype=np.float64)
    CX = (X_LO + cols + 0.5)[None, :, None]     # (1,W,1)
    CY = (Y_LO + rows + 0.5)[:, None, None]     # (H,1,1)
    half = 0.5 * (1.0 - GAP)
    offx = np.array([-half, half, half, -half],
                    dtype=np.float64)[None, None, :]
    offy = np.array([-half, -half, half, half],
                    dtype=np.float64)[None, None, :]
    CELLX = (CX + offx) * np.ones((H, 1, 1))    # (H,W,4) BL,BR,TR,TL
    CELLY = (CY + offy) * np.ones((1, W, 1))    # (H,W,4)
    CELLX = CELLX.reshape(-1, 4)
    CELLY = CELLY.reshape(-1, 4)

    # --- view path: fit the trail bbox (biased toward the ant), smoothed,
    #     opening pinned to the origin, finale pinned to the whole trail
    BCX = 0.5 * (x_lo + x_hi)
    BCY = 0.5 * (y_lo + y_hi)
    VT = np.arange(0.0, DUR + 0.5 / FPS, 1.0 / FPS)
    N_ = np.minimum(n_of_t(VT).astype(np.int64), S_TOTAL)
    bx0t, bx1t = bx0[N_], bx1[N_]
    by0t, by1t = by0[N_], by1[N_]
    w_t = bx1t - bx0t + 1 + 2 * TRACK_PAD
    h_t = by1t - by0t + 1 + 2 * TRACK_PAD
    s_need = np.minimum(13.6 / np.maximum(w_t, 1e-6),
                        7.6 / np.maximum(h_t, 1e-6))
    s_t = np.clip(s_need, S_FINAL, S_MAX)
    bxc = 0.5 * (bx0t + bx1t)
    byc = 0.5 * (by0t + by1t)
    cxt = (1.0 - BIAS) * bxc + BIAS * pos[N_, 0]
    cyt = (1.0 - BIAS) * byc + BIAS * pos[N_, 1]
    m0 = VT < T_CS
    s_t[m0] = S_MAX
    cxt[m0] = 0.0
    cyt[m0] = 0.0
    m1 = VT >= T_ZOOM
    s_t[m1] = S_FINAL
    cxt[m1] = BCX
    cyt[m1] = BCY
    tau = np.where(VT < T_ZOOM, TAU_TRACK, TAU_FINAL)
    al = 1.0 - np.exp(-(1.0 / FPS) / tau)
    VS = np.empty_like(s_t)
    VCX = np.empty_like(s_t)
    VCY = np.empty_like(s_t)
    vs, vcx, vcy = S_MAX, 0.0, 0.0
    for i in range(VT.size):
        vs += (s_t[i] - vs) * al[i]
        vcx += (cxt[i] - vcx) * al[i]
        vcy += (cyt[i] - vcy) * al[i]
        VS[i], VCX[i], VCY[i] = vs, vcx, vcy

    # --- highway onset: once the 104-cycle starts the shift is constant
    #     forever, so K0 = 1 + last index whose shift differs from the final
    shifts = pos[104:] - pos[:-104]
    final_shift = shifts[-1]
    differ = np.where((shifts != final_shift).any(axis=1))[0]
    hw_k = int(differ[-1]) + 1 if differ.size else 0

    return dict(
        pos=pos, hd=hd, MV=MV, FLATQ=FLATQ,
        snaps=snaps, snap0=np.zeros(Q, np.uint8),
        pre_col=np.asarray(pre_col), post_col=np.asarray(post_col),
        post_state=np.asarray(post_state, np.uint8),
        CELLX=CELLX, CELLY=CELLY, W=W, H=H, Q=Q,
        X_LO=X_LO, Y_LO=Y_LO, BCX=BCX, BCY=BCY, S_FINAL=S_FINAL,
        VT=VT, VS=VS, VCX=VCX, VCY=VCY,
        TG=TG, NG=NG, K_C=K_C, F1_D=F1_D, n_of_t=n_of_t,
        bx0=bx0, bx1=bx1, by0=by0, by1=by1,
        hw_k=hw_k, final_shift=final_shift, black_count=int(state.sum()),
        TEMPL=TEMPL, ANG=ANG,
    )


def sim():
    global _SIM
    if _SIM is None:
        _SIM = build_sim()
    return _SIM


# ------------------------------------------------------------- frame state
def write_frame(mesh, t):
    """Mesh quads + colours from absolute time t (pure function)."""
    S = sim()
    s = float(np.interp(t, S["VT"], S["VS"]))
    cx = float(np.interp(t, S["VT"], S["VCX"]))
    cy = float(np.interp(t, S["VT"], S["VCY"]))
    mesh.quads[..., 0] = (S["CELLX"] - cx) * s
    mesh.quads[..., 1] = (S["CELLY"] - cy) * s
    mesh.quads[..., 2] = 0.0

    fade_q, fade_pre, fade_post, fade_u = -1, None, None, 0.0
    if t < T_CS:                      # didactic: flips land at fixed times
        st = S["snap0"].copy()
        for j in range(ND):
            tf = T_INTRO + j * DID_T
            if t < tf:
                break
            u = (t - tf) / T_FLIP_DUR
            if u < 1.0:               # mid-fade: one cell interpolates
                fade_q = int(S["FLATQ"][j])
                fade_pre = S["pre_col"][j]
                fade_post = S["post_col"][j]
                fade_u = u * u * (3.0 - 2.0 * u)
                break
            st[int(S["FLATQ"][j])] = S["post_state"][j]
    else:                             # running phase: snapshots + tail flips
        k = min(int(S["n_of_t"](np.array([t]))[0]), S_TOTAL)
        base = k // SNAP
        st = S["snaps"][base].copy()
        for j in range(base * SNAP, k):
            st[int(S["FLATQ"][j])] ^= 1

    mesh.colors[..., :3] = np.where(st[:, None, None] > 0,
                                    COL_FLIPPED, COL_UNVIS)
    if fade_q >= 0:
        mesh.colors[fade_q, :, :3] = (fade_pre * (1.0 - fade_u)
                                      + fade_post * fade_u)
    if t < 0.8:                       # grid fade-in
        a = (t / 0.8) ** 2 * (3.0 - 2.0 * t / 0.8)
    else:
        a = 1.0
    mesh.colors[..., 3] = a
    return s, cx, cy


class MeshSeg(Animation):
    """A time window where only the mesh moves (pure function of t)."""

    def __init__(self, mesh, t0, t1):
        self.mesh = mesh
        self.t0, self.t1 = t0, t1
        super().__init__(VGroup(), run_time=t1 - t0, rate_func=linear)

    def interpolate_mobject(self, alpha):
        write_frame(self.mesh, self.t0 + alpha * (self.t1 - self.t0))


def arrow_opacity(t, t_in0, t_in1, t_out0, t_out1):
    """Arrow visibility as a pure function of t: smoothstep in over the
    arrow beat, hold, smoothstep out over the move beat."""
    if t_in0 <= t < t_in1:
        u = (t - t_in0) / (t_in1 - t_in0)
        return u * u * (3.0 - 2.0 * u)
    if t_in1 <= t < t_out0:
        return 1.0
    if t_out0 <= t < t_out1:
        u = (t - t_out0) / (t_out1 - t_out0)
        return 1.0 - u * u * (3.0 - 2.0 * u)
    return 0.0


class ArrowFade(Animation):
    """Drive one arrow's opacity from absolute time (RTM FadeIn fades to the
    authored opacity, so a pre-hidden arrow stays hidden -- do it ourselves)."""

    def __init__(self, arrow, t0, t1, t_in0, t_in1, t_out0, t_out1):
        self.arrow = arrow
        self.t0, self.t1 = t0, t1
        self.t_in0, self.t_in1 = t_in0, t_in1
        self.t_out0, self.t_out1 = t_out0, t_out1
        super().__init__(VGroup(), run_time=t1 - t0, rate_func=linear)

    def interpolate_mobject(self, alpha):
        t = self.t0 + alpha * (self.t1 - self.t0)
        self.arrow.set_opacity(arrow_opacity(
            t, self.t_in0, self.t_in1, self.t_out0, self.t_out1))


class CardFade(Animation):
    """Fade the rule card out as the ant speeds up. Parks the animated
    opacity as a pure function of t. Deliberately NOT RTM's FadeOut: this
    venv's timeline still runs finish() on ended fades, and FadeOut.finish
    releases the parked opacity -> the card ghosts back at full opacity."""

    def __init__(self, card, t0, t1):
        self.card = card
        self.t0, self.t1 = t0, t1
        super().__init__(VGroup(), run_time=t1 - t0, rate_func=linear)

    def interpolate_mobject(self, alpha):
        t = self.t0 + alpha * (self.t1 - self.t0)
        u = max(0.0, min(1.0, (t - self.t0) / (self.t1 - self.t0)))
        set_anim_opacity(self.card,
                         1.0 - u * u * (3.0 - 2.0 * u))   # smoothstep


class TurnSeg(Animation):
    """Didactic turn: mesh + ant rotating in place (arrival -> post-turn
    heading) + the arrow's opacity, all pure functions of t. Deliberately
    NOT manim's Rotate: its become() rebuilds a VMobject's points into an
    8-point bezier form and the painter then draws a distorted blob."""

    def __init__(self, mesh, ant, arrow, t0, t1, cell, a0_deg, a1_deg,
                 t_in0, t_in1, t_out0, t_out1):
        self.mesh = mesh
        self.ant = ant
        self.arrow = arrow
        self.t0, self.t1 = t0, t1
        self.cell = np.asarray(cell, dtype=np.float64) + 0.5   # centre
        self.a0, self.a1 = a0_deg, a1_deg
        self.t_in0, self.t_in1 = t_in0, t_in1
        self.t_out0, self.t_out1 = t_out0, t_out1
        super().__init__(VGroup(), run_time=t1 - t0, rate_func=linear)

    def interpolate_mobject(self, alpha):
        t = self.t0 + alpha * (self.t1 - self.t0)
        s, cx, cy = write_frame(self.mesh, t)
        u = alpha * alpha * (3.0 - 2.0 * alpha)          # smoothstep
        da = (self.a1 - self.a0 + 180.0) % 360.0 - 180.0
        ang = np.deg2rad(self.a0 + da * u)
        # CCW rotation of row vectors: v @ [[cos, sin], [-sin, cos]]
        Rm = np.array([[np.cos(ang), np.sin(ang)],
                       [-np.sin(ang), np.cos(ang)]])
        pts = TEMPL @ Rm * (1.05 * s)
        ox = (self.cell[0] - cx) * s
        oy = (self.cell[1] - cy) * s
        self.ant.set_points_as_corners(np.concatenate(
            [pts + np.array([ox, oy]), np.zeros((3, 1))], axis=1))
        self.arrow.set_opacity(arrow_opacity(
            t, self.t_in0, self.t_in1, self.t_out0, self.t_out1))


class MoveSeg(Animation):
    """Didactic move: lerp the ant from cell c0 to c1 at a fixed heading
    (the TurnSeg in the arrow beat set it). Pure function of t."""

    def __init__(self, mesh, ant, t0, t1, c0, c1, heading_deg):
        self.mesh = mesh
        self.ant = ant
        self.t0, self.t1 = t0, t1
        self.c0 = np.asarray(c0, dtype=np.float64)
        self.c1 = np.asarray(c1, dtype=np.float64)
        self.ang = heading_deg
        super().__init__(VGroup(), run_time=t1 - t0, rate_func=linear)

    def interpolate_mobject(self, alpha):
        s, cx, cy = write_frame(self.mesh,
                                self.t0 + alpha * (self.t1 - self.t0))
        u = alpha * alpha * (3.0 - 2.0 * alpha)     # smoothstep
        cell = self.c0 * (1.0 - u) + self.c1 * u + 0.5   # cell CENTER
        a = np.deg2rad(self.ang)
        # CCW rotation of row vectors: v @ [[cos, sin], [-sin, cos]]
        Rm = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        pts = TEMPL @ Rm * (1.05 * s)
        ox = (cell[0] - cx) * s
        oy = (cell[1] - cy) * s
        self.ant.set_points_as_corners(np.concatenate(
            [pts + np.array([ox, oy]), np.zeros((3, 1))], axis=1))


class RunSeg(Animation):
    """The accelerating run + finale: mesh AND the ant, from t."""

    def __init__(self, mesh, ant, t0, t1):
        self.mesh = mesh
        self.ant = ant
        self.t0, self.t1 = t0, t1
        super().__init__(VGroup(), run_time=t1 - t0, rate_func=linear)

    def interpolate_mobject(self, alpha):
        t = self.t0 + alpha * (self.t1 - self.t0)
        s, cx, cy = write_frame(self.mesh, t)
        S = sim()
        n = float(S["n_of_t"](np.array([t]))[0])
        k = min(int(n), S_TOTAL - 1)
        fr = n - k
        px = S["pos"][k, 0] * (1.0 - fr) + S["pos"][k + 1, 0] * fr + 0.5
        py = S["pos"][k, 1] * (1.0 - fr) + S["pos"][k + 1, 1] * fr + 0.5
        # arrival heading at pos[k] is hd[k]; the move to pos[k+1] uses hd[k+1]
        hprev = int(S["hd"][k])
        hcur = int(S["hd"][k + 1])
        a0, a1 = S["ANG"][hprev], S["ANG"][hcur]
        da = (a1 - a0 + 180.0) % 360.0 - 180.0
        ang = np.deg2rad(a0 + da * min(1.0, fr * 3.0))
        sc = max(1.05 * s, ANT_MIN)
        c, sn = np.cos(ang), np.sin(ang)
        # CCW rotation of row vectors: v @ [[cos, sin], [-sin, cos]]
        Rm = np.array([[c, sn], [-sn, c]])
        pts = S["TEMPL"] @ Rm * sc
        ox = (px - cx) * s
        oy = (py - cy) * s
        self.ant.set_points_as_corners(np.concatenate(
            [pts + np.array([ox, oy]), np.zeros((3, 1))], axis=1))


# ---------------------------------------------------------------- scene
class AntScene(Scene):
    def construct(self):
        S = sim()
        win = MLWindow(960, 540)
        win.scene = self

        mesh = MeshBatch()
        Q = S["Q"]
        assert Q * 6 <= mesh.MAX_MESH_VERTS, "grid over mesh cap: %d" % Q
        mesh.quads = np.zeros((Q, 4, 3), dtype=np.float64)
        mesh.colors = np.zeros((Q, 4, 4), dtype=np.float32)
        win.set_mesh(mesh)

        s0 = float(S["VS"][0])

        # ant (plain VMobject: Polygram points have a subpath structure
        # that set_points_as_corners destroys, which squishes the triangle)
        # ant at the CENTER of cell (0,0): world (0.5, 0.5) * s0
        ant = VMobject(color=COL_ANT, fill_color=COL_ANT, fill_opacity=1.0,
                       stroke_color=BLACK, stroke_width=2.0)
        ant.set_points_as_corners(np.concatenate(
            [S["TEMPL"] * (1.05 * s0) + np.array([0.5 * s0, 0.5 * s0]),
             np.zeros((3, 1))], axis=1))
        ant.rotate(float(S["ANG"][0]) * DEGREES)
        self.add(ant)

        # rule card (top right): title + 2-column rule alignment with REAL
        # blocks (the cell colours exactly as they render on screen)
        title = Text("Langton's Ant", font_size=28, weight=BOLD, color=YELLOW)
        white_block = Square(side_length=0.55, color=WHITE,
                             fill_color="#E0E5ED", fill_opacity=1.0,
                             stroke_width=1.5)
        black_block = Square(side_length=0.55, color=WHITE,
                             fill_color="#1A1D21", fill_opacity=1.0,
                             stroke_width=1.5)
        col_r = VGroup(white_block,
                       Text("turn right 90°", font_size=20,
                            color=WHITE)).arrange(DOWN, buff=0.12)
        col_l = VGroup(black_block,
                       Text("turn left 90°", font_size=20,
                            color=WHITE)).arrange(DOWN, buff=0.12)
        rule = VGroup(col_r, col_l).arrange(RIGHT, buff=0.5)
        rect = RoundedRectangle(width=3.6, height=2.30, corner_radius=0.18,
                                fill_color=BLACK, fill_opacity=0.85,
                                stroke_color=WHITE, stroke_width=1.5)
        title.next_to(rect.get_top(), DOWN, buff=0.16)
        rule.next_to(title, DOWN, buff=0.18)
        card = VGroup(rect, title, rule).to_corner(UR, buff=0.30)
        self.add(card)

        # didactic arrows (hidden until their beat); arrow dir = heading
        # DURING step j, i.e. hd[j+1] (hd[j+1] is the heading set at pos[j])
        arrows = []
        for j in range(ND):
            d = S["MV"][int(S["hd"][j + 1])]
            c0 = S["pos"][j] + 0.5                      # cell centre
            start = np.array([c0[0] * s0, c0[1] * s0, 0.0])
            end = np.array([(c0[0] + d[0]) * s0, (c0[1] + d[1]) * s0, 0.0])
            ar = Arrow(start, end, buff=0.06, color=ARROW_COL, stroke_width=6,
                       max_tip_length_to_length_ratio=0.30)
            ar.set_opacity(0.0)
            self.add(ar)
            arrows.append(ar)

        # intro
        win.play(MeshSeg(mesh, 0.0, T_INTRO),
                 FadeIn(ant, run_time=0.8),
                 FadeIn(card, run_time=1.0))

        # didactic loop: colour -> arrow (+turn) -> move
        for j in range(ND):
            t_flip = T_INTRO + j * DID_T
            t_arrow = t_flip + T_FLIP_DUR
            t_move = t_arrow + T_ARROW_DUR
            # arrival heading at pos[j] is hd[j]; after the turn it is hd[j+1]
            win.play(MeshSeg(mesh, t_flip, t_arrow))
            win.play(TurnSeg(mesh, ant, arrows[j], t_arrow, t_move,
                             S["pos"][j],
                             float(S["ANG"][int(S["hd"][j])]),
                             float(S["ANG"][int(S["hd"][j + 1])]),
                             t_arrow, t_arrow + 0.15,
                             t_move, t_move + T_MOVE_DUR))
            win.play(MoveSeg(mesh, ant, t_move, t_move + T_MOVE_DUR,
                             S["pos"][j], S["pos"][j + 1],
                             float(S["ANG"][int(S["hd"][j + 1])])),
                     ArrowFade(arrows[j], t_move, t_move + T_MOVE_DUR,
                               t_arrow, t_arrow + 0.15,
                               t_move, t_move + T_MOVE_DUR))

        # the run: acceleration -> chaos -> highway -> the whole scope
        # (the rule card fades out as the ant speeds up)
        win.play(RunSeg(mesh, ant, T_CS, DUR),
                 CardFade(card, T_CS, T_CS + T_CARD_FADE))
        win.close()


# ---------------------------------------------------------------- verify
def verify():
    S = sim()
    print("[sim] steps:", S_TOTAL)
    print("[sim] final trail bbox: x[%d,%d] y[%d,%d]  (%dx%d cells)" % (
        S["bx0"][-1], S["bx1"][-1], S["by0"][-1], S["by1"][-1],
        S["bx1"][-1] - S["bx0"][-1] + 1, S["by1"][-1] - S["by0"][-1] + 1))
    print("[sim] grid: %dx%d = %d quads (x6 = %d, cap %d) %s" % (
        S["W"], S["H"], S["Q"], S["Q"] * 6, MeshBatch.MAX_MESH_VERTS,
        "OK" if S["Q"] * 6 <= MeshBatch.MAX_MESH_VERTS else "OVER CAP"))
    print("[sim] origin cell at grid (%d,%d); trail bbox centre (%.1f,%.1f)" % (
        0 - S["X_LO"], 0 - S["Y_LO"], S["BCX"], S["BCY"]))
    print("[sim] final coverage: grid spans %.2fx%.2f units at s_final=%.4f"
          " vs frame 14.22x8 (+%.1f margin) %s" % (
              S["W"] * S["S_FINAL"], S["H"] * S["S_FINAL"], S["S_FINAL"],
              EDGE_MARGIN,
              "OK" if (S["W"] * S["S_FINAL"] >= 14.22 + 2 * EDGE_MARGIN
                       and S["H"] * S["S_FINAL"] >= 8.0 + 2 * EDGE_MARGIN)
              else "TOO SMALL"))
    print("[sim] black cells at end: %d / %d (%.1f%%)" % (
        S["black_count"], S["Q"], 100.0 * S["black_count"] / S["Q"]))
    for k in (5, int(S["K_C"]), max(S["hw_k"], 0), S_TOTAL):
        print("[sim] bbox at step %6d: x[%d,%d] y[%d,%d]" % (
            k, S["bx0"][k], S["bx1"][k], S["by0"][k], S["by1"][k]))
    if S["hw_k"] >= 0:
        print("[highway] onset step %d, 104-shift %s (classic: 9977, "
              "2 cells/cycle)" % (S["hw_k"], S["final_shift"]))
    else:
        print("[highway] onset not found within %d steps" % S_TOTAL)
    print("[schedule] K_C=%.1f steps, F1_D=%.1f steps/s, n(30)=%.1f" % (
        S["K_C"], S["F1_D"], float(S["n_of_t"](np.array([30.0]))[0])))
    for t in (0.0, 2.0, 9.1, 12.0, 15.0, 18.0, 20.0, 22.0, 24.0, 25.0, 26.0,
              28.0, 30.0):
        n = float(S["n_of_t"](np.array([t]))[0])
        s = float(np.interp(t, S["VT"], S["VS"]))
        cx = float(np.interp(t, S["VT"], S["VCX"]))
        cy = float(np.interp(t, S["VT"], S["VCY"]))
        print("[t] t=%5.1f  n=%9.1f  s=%.4f  c=(%7.2f,%7.2f)" % (t, n, s, cx, cy))
    try:
        import PIL  # noqa: F401
        print("[probe] PIL available")
    except ImportError:
        print("[probe] PIL MISSING")


# ---------------------------------------------------------------- render
def render(out_path="out/langtons_ant.mp4"):
    out = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    t0 = time.time()
    res = fast_record_scene(AntScene, out, overwrite=True)
    print("[render] %.1f s -> %s" % (time.time() - t0, res.get("files")))
    return res


def stills(times, out_dir="out/stills"):
    from real_time_manim.frames import FrameServer
    os.makedirs(out_dir, exist_ok=True)
    srv = FrameServer(AntScene)
    for t in times:
        p = os.path.join(out_dir, "ant_t%05.2f.png" % t)
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
