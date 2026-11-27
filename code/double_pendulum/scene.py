r"""
Double pendulum — RTM render (30 s @ 1080p60).

Timeline:
  0-1 s    title fade in, pendulum held at its IC
  1-6 s    swing (phase 1)
  6-13 s   force diagram: gravity + rod tensions on both bobs (phase 2)
  13-20 s  energy & Lagrangian: T/V/L formulas, stacked KE+PE bars
           (constant total = conservation), live numeric readouts (phase 3)
  20-30 s  chaos: two ghost pendulums revealed, same ICs except
           theta2(0) += 1e-3, integrated since release (phase 4)

One master Animation drives every mobject as a pure function of t
(per-frame point writes via set_points_as_corners / put_start_and_end_on;
timed fades via set_opacity ramps).
"""
import ctypes
import os

import numpy as np

import real_time_manim  # noqa: F401  (must be first)
from real_time_manim import (Scene, Text, Line, Circle, VMobject, VGroup,
                             Rectangle, Dot, Animation, TypstMath, RIGHT,
                             DOWN, UP, LEFT)
from real_time_manim.vulkan_bind import MLWindow
from real_time_manim import fast_record_scene

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = np.load(os.path.join(HERE, "data", "frames.npz"))
P1, P2 = DATA["P1"], DATA["P2"]           # (3, 1800, 2)
FMAIN = DATA["FMAIN"]                     # (1800, 5, 2)
KE, PE, E0 = DATA["KE"], DATA["PE"], float(DATA["E0"])
N_FR = P1.shape[1]

# ---- style (rtm-style-params table) ----
BG = "#070b14"
INK = "#efe9dc"
DIM = "#7b93a6"
GOLD = "#f2c66d"
CYAN = "#6fd6cd"
WHITE = "#fff6e8"

FPS = 60.0
DUR = 30.0
PIVOT = np.array([-2.6, 1.0])
PIV3 = np.array([-2.6, 1.0, 0.0])

PEND_COL = [GOLD, WHITE, DIM]             # main, ghost+, ghost-
PEND_W = [5.5, 3.0, 3.0]                  # rod stroke widths

F_SCALE = 0.03
F_CAP = 1.15
F_MIN = 0.05
F_PAD = 0.11
HEAD_L, HEAD_W = 0.15, 0.06

TRAIL_N = 120
TRAIL_CHUNKS = 8
CHUNK_LEN = TRAIL_N // TRAIL_CHUNKS
TRAIL_OP = 0.5            # newest-chunk opacity; older chunks fade to ~0.06
BAR_H = 1.9
BAR_BASE = -1.25
BAR_X = 3.9
BAR_W = 0.55
READ_X = 4.75


def sstep(x):
    x = float(np.clip(x, 0.0, 1.0))
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


def make_arrow(color, width=5):
    line = Line(PIV3, PIV3 + np.array([1.0, 0.0, 0.0]), color=color)
    line.set_stroke(color, width)
    head = VMobject()
    head.set_stroke(color, 2.5)
    return VGroup(line, head)


def set_arrow(grp, tail, vec, alpha):
    """Place arrow grp (line + triangle head) from tail along vec."""
    line, head = grp
    L = float(np.hypot(vec[0], vec[1]))
    if L < F_MIN or alpha <= 1e-4:
        grp.set_opacity(0.0)
        return
    u = vec / L
    length = min(L * F_SCALE, F_CAP)
    # clamp tip inside frame (half-extents 7.05 x 3.92)
    for _ in range(60):
        tip = tail + u * length
        if (-7.05 <= tip[0] <= 7.05) and (-3.92 <= tip[1] <= 3.92):
            break
        length *= 0.95
    t3 = np.array([tail[0], tail[1], 0.0])
    tip3 = np.array([tip[0], tip[1], 0.0])
    line.put_start_and_end_on(t3 + np.array([u[0], u[1], 0.0]) * F_PAD, tip3)
    perp = np.array([-u[1], u[0], 0.0])
    head.set_points_as_corners(np.array([
        tip3 - np.array([u[0], u[1], 0.0]) * HEAD_L + perp * HEAD_W,
        tip3,
        tip3 - np.array([u[0], u[1], 0.0]) * HEAD_L - perp * HEAD_W,
        tip3 - np.array([u[0], u[1], 0.0]) * HEAD_L + perp * HEAD_W,
    ]))
    grp.set_opacity(alpha)


class PendAnim(Animation):
    def __init__(self, mobs, **kwargs):
        super().__init__(Dot(point=np.zeros(3), radius=0.001), **kwargs)
        (self.rod, self.bob, self.trail, self.arrows, self.caps,
         self.title, self.legend_f, self.legend_ghost, self.panel,
         self.f_t, self.f_v, self.f_l, self.bar_pe, self.bar_ke,
         self.rds) = mobs

    def write_frame(self, k, t):
        # ---- pendulums (rods, bobs, trails) ----
        ghost_a = sstep((t - 20.0) / 1.0)
        for i in range(3):
            p1 = P1[i, k]
            p2 = P2[i, k]
            a = 1.0 if i == 0 else ghost_a
            self.rod[i][0].put_start_and_end_on(PIV3, np.array([p1[0], p1[1], 0.0]))
            self.rod[i][1].put_start_and_end_on(np.array([p1[0], p1[1], 0.0]),
                                                np.array([p2[0], p2[1], 0.0]))
            self.bob[i][0].move_to(np.array([p1[0], p1[1], 0.0]))
            self.bob[i][1].move_to(np.array([p2[0], p2[1], 0.0]))
            self.rod[i][0].set_opacity(a)
            self.rod[i][1].set_opacity(a)
            self.bob[i][0].set_opacity(a)
            self.bob[i][1].set_opacity(a)
            # trail: last TRAIL_N bob2 positions, pure function of frame k,
            # split into age chunks with per-chunk opacity (RTM ignores
            # stroke opacity; object opacity works and repaints)
            t0 = 1.05 if i == 0 else 20.05
            k0 = int(np.ceil(t0 * FPS))
            for c in range(TRAIL_CHUNKS):
                c_lo = max(k - (c + 1) * CHUNK_LEN + 1, k0)
                c_hi = k - c * CHUNK_LEN
                if c_hi >= c_lo:
                    pts = P2[i, c_lo:c_hi + 1]
                    self.trail[i][c].set_points_as_corners(
                        np.column_stack([pts, np.full(len(pts), 0.05)]))
                    self.trail[i][c].set_opacity(
                        a * TRAIL_OP * (1.0 - c / TRAIL_CHUNKS))
                else:
                    self.trail[i][c].set_opacity(0.0)

        # ---- force diagram (phase 2) ----
        f_a = sstep((t - 6.0) / 0.8) * (1.0 - sstep((t - 12.6) / 0.8))
        for j in range(5):
            f = FMAIN[k, j]
            tail = P1[0, k] if j < 3 else P2[0, k]
            set_arrow(self.arrows[j], tail, f, f_a)
        self.legend_f.set_opacity(f_a)

        # ---- energy panel (phase 3) ----
        e_a = sstep((t - 13.3) / 0.8)
        self.panel.set_opacity(e_a)
        h_pe = max(PE[k] / E0 * BAR_H, 0.002)
        h_ke = max(KE[k] / E0 * BAR_H, 0.002)
        self.bar_pe.stretch_to_fit_height(h_pe, about_edge=DOWN)
        self.bar_ke.stretch_to_fit_height(h_ke, about_edge=DOWN)
        pe_top = self.bar_pe.get_top()[1]
        self.bar_ke.shift(np.array([0.0, pe_top - self.bar_ke.get_bottom()[1], 0.0]))
        # live readouts: pre-rendered strings, opacity-swapped per frame
        # (set_text does not repaint in RTM; opacity writes do)
        q = int(t / 0.4)
        kq = min(int(round(q * 0.4 * FPS)), N_FR - 1)
        s_ke = f"KE {KE[kq]:.2f}"
        s_pe = f"PE {PE[kq]:.2f}"
        s_e = f"E {KE[kq] + PE[kq]:.2f} J"
        for d, s in zip(self.rds, (s_ke, s_pe, s_e)):
            for v, txt in d.items():
                txt.set_opacity(e_a if v == s else 0.0)

        # ---- phase captions (bottom-left, crossfading) ----
        T_BOUNDS = [1.0, 6.0, 13.0, 20.0, 30.0]
        for i, cap in enumerate(self.caps):
            a = sstep((t - T_BOUNDS[i]) / 0.5) * \
                (1.0 - sstep((t - (T_BOUNDS[i + 1] - 0.5)) / 0.5))
            cap.set_opacity(a)

        # ---- title / ghost legend ----
        self.title.set_opacity(sstep(t / 0.8))
        self.legend_ghost.set_opacity(ghost_a)

    def interpolate_mobject(self, alpha):
        t = alpha * self.run_time
        k = int(round(t * FPS))
        if k > N_FR - 1:
            k = N_FR - 1
        self.write_frame(k, t)


class DoublePendulum(Scene):
    def construct(self):
        win = MLWindow(960, 540)
        win.scene = self
        self.play = win.play
        patch_background(win)
        self.camera.background_color = BG

        FONT = "Baskerville"
        mobs = []

        # pendulum: 3 x (rod1, rod2), (bob1, bob2)
        rod, bob, trail = [], [], []
        for i in range(3):
            col = PEND_COL[i]
            w = PEND_W[i]
            r1 = Line(PIV3, PIV3 + np.array([0.0, -1.0, 0.0]), color=col)
            r2 = Line(PIV3 + np.array([0.0, -1.0, 0.0]),
                      PIV3 + np.array([0.0, -2.0, 0.0]), color=col)
            r1.set_stroke(col, w)
            r2.set_stroke(col, w)
            rod.append([r1, r2])
            b1 = Circle(radius=0.10, color=col).set_fill(col, 0.9)
            b2 = Circle(radius=0.10, color=col).set_fill(col, 0.9)
            b1.set_stroke(INK if i == 0 else BG, 2.0 if i == 0 else 1.0)
            b2.set_stroke(INK if i == 0 else BG, 2.0 if i == 0 else 1.0)
            bob.append([b1, b2])
            tr = [VMobject() for _ in range(TRAIL_CHUNKS)]
            for tch in tr:
                tch.set_stroke(col, 2.5)
            trail.append(tr)
            self.add(r1, r2, b1, b2)
            for tch in tr:
                self.add(tch)

        pivot = Circle(radius=0.075, color=INK).set_fill(INK, 1.0)
        self.add(pivot)

        # force arrows on the main pendulum: [g1, T1, T21, g2, T2]
        arrows = [make_arrow(c, 5) for c in (INK, GOLD, CYAN, INK, CYAN)]
        for grp in arrows:
            grp.set_opacity(0.0)
            self.add(grp)

        # force legend (right panel, phase 2)
        def swatch(col, dir_, lab):
            l = Line(np.zeros(3), np.array([dir_[0], dir_[1], 0.0]) * 0.85,
                     color=col)
            l.set_stroke(col, 5)
            h = VMobject()
            h.set_stroke(col, 2.5)
            u = np.array([dir_[0], dir_[1], 0.0])
            perp = np.array([-u[1], u[0], 0.0])
            tip = u * 0.85
            h.set_points_as_corners(np.array([
                tip - u * HEAD_L + perp * HEAD_W, tip,
                tip - u * HEAD_L - perp * HEAD_W, tip - u * HEAD_L + perp * HEAD_W]))
            txt = Text(lab, font=FONT, font_size=24, color=INK)
            grp = VGroup(l, h, txt)
            return grp
        legend_f = VGroup(
            swatch(INK, (0, -1), "gravity  m\u2081g, m\u2082g"),
            swatch(GOLD, (0.55, 0.83), "rod 1  T\u2081"),
            swatch(CYAN, (0.55, -0.83), "rod 2  T\u2082"),
        )
        for g in legend_f:
            g[2].next_to(g[0].get_end(), RIGHT, buff=0.45)
        legend_f.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        legend_f.move_to(np.array([3.9, 0.0, 0.0]))
        legend_f.set_opacity(0.0)
        self.add(legend_f)

        # energy panel (phase 3)
        p_title = Text("Energy & Lagrangian", font=FONT, font_size=22,
                       color=DIM)
        f_t = TypstMath(r"T = 1/2 m_1 v_1^2 + 1/2 m_2 v_2^2",
                        color=GOLD).scale_to_fit_width(3.1)
        f_v = TypstMath(r"V = m_1 g y_1 + m_2 g y_2",
                        color=CYAN).scale_to_fit_width(3.1)
        f_l = TypstMath(r"L = T - V", color=INK).scale_to_fit_width(3.1)
        bar_pe = Rectangle(width=BAR_W, height=BAR_H, color=CYAN)
        bar_pe.set_fill(CYAN, 0.85)
        bar_pe.set_stroke(width=0)
        bar_ke = Rectangle(width=BAR_W, height=0.01, color=GOLD)
        bar_ke.set_fill(GOLD, 0.85)
        bar_ke.set_stroke(width=0)

        # readouts: every possible string pre-rendered once, swapped by opacity
        def readouts(fmt, arr, color, y):
            d = {}
            qs = range(int(13.3 / 0.4), int(30.0 / 0.4) + 1)
            vals = {fmt.format(0.0)}
            for q in qs:
                kq = min(int(round(q * 0.4 * FPS)), N_FR - 1)
                vals.add(fmt.format(arr[kq]))
            for v in vals:
                t = Text(v, font=FONT, font_size=22, color=color)
                t.move_to(np.array([READ_X + t.width / 2.0, y, 0.0]))
                t.set_opacity(0.0)
                d[v] = t
                self.add(t)
            return d
        rds = [readouts("KE {:.2f}", KE, GOLD, 0.55),
               readouts("PE {:.2f}", PE, CYAN, 0.05),
               readouts("E {:.2f} J", KE + PE, INK, -0.45)]

        for m, y in ((p_title, 2.95), (f_t, 2.32), (f_v, 1.72), (f_l, 1.12)):
            m.move_to(np.array([3.45 + m.width / 2.0, y, 0.0]))
        bar_pe.move_to(np.array([BAR_X, BAR_BASE + BAR_H / 2.0, 0.0]))
        bar_ke.move_to(np.array([BAR_X, BAR_BASE, 0.0]))
        panel = VGroup(p_title, f_t, f_v, f_l, bar_pe, bar_ke)
        panel.set_opacity(0.0)
        self.add(panel)

        # phase captions
        caps = [
            Text("1 \u00b7 Swing", font=FONT, font_size=24, color=INK),
            Text("2 \u00b7 Force diagram", font=FONT, font_size=24, color=INK),
            Text("3 \u00b7 Energy & Lagrangian", font=FONT, font_size=24,
                 color=INK),
            Text("4 \u00b7 Chaos \u2014 \u03b4\u03b8\u2082(0) = \u00b10.001",
                 font=FONT, font_size=24, color=INK),
        ]
        for cap in caps:
            cap.to_corner(np.array([-1, -1, 0]), buff=0.62)
            cap.set_opacity(0.0)
            self.add(cap)

        # title
        title = Text("Double Pendulum \u00b7 RK4", font=FONT, font_size=26,
                     color=INK)
        title.to_corner(np.array([-1, 1, 0]), buff=0.35)
        title.set_opacity(0.0)
        self.add(title)

        # ghost legend (phase 4, under the pendulum)
        def gline(col, lab):
            d = Circle(radius=0.055, color=col).set_fill(col, 1.0)
            t = Text(lab, font=FONT, font_size=22, color=INK)
            grp = VGroup(d, t)
            t.next_to(d, RIGHT, buff=0.25)
            return grp
        legend_ghost = VGroup(
            gline(WHITE, "\u03b8\u2082(0) = +0.001"),
            gline(DIM, "\u03b8\u2082(0) = \u22120.001"),
        )
        legend_ghost.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        legend_ghost.move_to(np.array([-4.6, -2.55, 0.0]))
        legend_ghost.set_opacity(0.0)
        self.add(legend_ghost)

        mobs = (rod, bob, trail, arrows, caps, title, legend_f, legend_ghost,
                panel, f_t, f_v, f_l, bar_pe, bar_ke, rds)
        anim = PendAnim(mobs, run_time=DUR)
        anim.write_frame(0, 0.0)
        win.play(anim, run_time=DUR)
        win.close()


if __name__ == "__main__":
    # NOTE: fast_record_scene produced a frozen video for this mobject-only
    # scene (10 unique decoded frames of 1800; stills were fine). Record via
    # the FrameServer render_at loop instead — see record.py.
    from record import render
    render()
