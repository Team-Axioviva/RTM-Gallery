r"""Thurston's corrugation sphere eversion — RTM render.

Geometry: numpy port of Nathaniel Thurston's `evert` (eversion.py), verified
pointwise against the compiled original C++ (gate.py, cpp_oracle/).

Structure (the C++ Draw()): each strip is the meridian + figure-eight,
splayed across one 2*pi/N strip by RotZ(v/N); the strip's figure-eight
vanishes at v=0 and v=1, so adjacent strips meet at uncorrugated guide
meridians.  The surface = 8 northern strips at RotZ(-s*2pi/N) plus 8
southern strips at RotZ((s+1)*2pi/N) o RotY(pi) -- the film's sixteen
strips, with the south's +1 stagger forming the equator zipper.

Colour: two-sided via X +/- eps*n layers in one depth-tested MeshBatch
(gold outside, blue inside); the normal field flips through the eversion,
so the colour swaps automatically.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import real_time_manim as rtm
from real_time_manim import (Animation, MeshBatch, MLWindow, Text,
                             ThreeDScene)
from real_time_manim.record import fast_record_scene

import eversion as ev

# ----------------------------------------------------------------------
# knobs
# ----------------------------------------------------------------------
NSTRIPS = 8          # strips per hemisphere (the film's count: 16 total)
NT = 41              # tau grid points across a strip
NP = 49              # phi grid points, pole to equator
EPS = 0.02           # two-layer offset, in unit coordinates
SCALE = 2.4          # world scale: max|X| = 1.48 => 7.1 units < frame height
HOLD = 1.5           # seconds held at each endpoint
T_EV = 12.0          # eversion duration (matches eversion.SCHEDULE)
FPS = 60
TOTAL = 2 * HOLD + T_EV

GOLD = np.array([0.95, 0.72, 0.30])
BLUE = np.array([0.45, 0.50, 0.95])
LIGHT = np.array([0.45, 0.65, 0.62])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

PHASES = [(name, tini, tini + tdel) for name, tini, tdel in ev.SCHEDULE]

OUT = os.path.expanduser("~/gallary/video/thurston_eversion.mp4")

# waypoint list for the time map: (duration_s, t_start, t_end)
MODE = "forward"      # forward | inverse | cycle (CLI: --inverse / --cycle)
TIMELINES = {
    "forward": [(1.5, 0.0, 0.0), (12.0, 0.0, 12.0), (1.5, 12.0, 12.0)],
    "inverse": [(1.5, 12.0, 12.0), (12.0, 12.0, 0.0), (1.5, 0.0, 0.0)],
    "cycle":   [(2.0, 12.0, 12.0), (12.0, 12.0, 0.0), (2.0, 0.0, 0.0),
                (12.0, 0.0, 12.0), (2.0, 12.0, 12.0)],
}


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ----------------------------------------------------------------------
# mesh builder: one strip per frame, 16 placements
# ----------------------------------------------------------------------
class EversionMesh:
    def __init__(self):
        self.tau = np.linspace(0.0, 1.0, NT)
        self.phi = np.linspace(0.0, np.pi / 2, NP)

        i = np.arange(NT - 1)
        j = np.arange(NP - 1)
        ii = np.broadcast_to(i[:, None], (NT - 1, NP - 1))
        ii1 = ii + 1
        jj = np.broadcast_to(j[None, :], (NT - 1, NP - 1))
        self.quads_strip = np.empty((NT - 1, NP - 1, 4, 2), dtype=np.int64)
        self.quads_strip[..., 0, 0] = ii
        self.quads_strip[..., 0, 1] = jj
        self.quads_strip[..., 1, 0] = ii1
        self.quads_strip[..., 1, 1] = jj
        self.quads_strip[..., 2, 0] = ii1
        self.quads_strip[..., 2, 1] = jj + 1
        self.quads_strip[..., 3, 0] = ii
        self.quads_strip[..., 3, 1] = jj + 1

        # the sixteen placements: 8 north + 8 south (draw order of the C++)
        mirror = np.diag([-1.0, 1.0, -1.0])          # RotY(pi)
        self.copies = []
        for s in range(NSTRIPS):
            self.copies.append(_rotz(-2 * np.pi * s / NSTRIPS))
        for s in range(NSTRIPS):
            self.copies.append(_rotz(2 * np.pi * (s + 1) / NSTRIPS) @ mirror)

        self.nq = (NT - 1) * (NP - 1)
        nc = len(self.copies)
        self.batch = MeshBatch()
        self.batch.append(
            np.zeros((nc * self.nq, 4, 3), dtype=np.float64),
            np.zeros((nc * self.nq, 4, 4), dtype=np.float32),
        )
        self.eye = np.array([0.0, 0.0, 20.0])   # replaced by the camera's

    def fill(self, t):
        P, nhat, _ = ev.eval_strip(t, NT, NP, NSTRIPS)
        for c, A in enumerate(self.copies):
            Pc = P @ A.T
            ncr = nhat @ A.T
            q = Pc[self.quads_strip[..., 0], self.quads_strip[..., 1]] * SCALE
            # per-quad facing: the original's glFrontFace(GL_CW) two-sided
            # lighting, done on the CPU.  One layer, so fold valleys (where
            # the pleat curvature radius is smaller than any eps offset)
            # cannot punch one colour's layer through the other's.
            face = np.cross(q[..., 1, :] - q[..., 0, :],
                            q[..., 3, :] - q[..., 0, :])
            centroid = q.mean(axis=-2)
            front = ((face * (self.eye - centroid)).sum(-1) > 0.0).reshape(-1)
            shade = np.abs(ncr @ LIGHT) ** 0.85
            sh = shade[self.quads_strip[..., 0], self.quads_strip[..., 1]]
            base = 0.30 + 0.75 * sh[..., None].reshape(-1, 4, 1)
            col = np.where(front[:, None], GOLD[None, :], BLUE[None, :])
            ones = np.ones((self.nq, 4, 1), dtype=np.float32)
            k = c * self.nq
            self.batch.quads[k:k + self.nq] = q.reshape(-1, 4, 3)
            self.batch.colors[k:k + self.nq] = np.concatenate(
                [base * col[:, None, :], ones], axis=-1).astype(np.float32)
        return P, nhat


# ----------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------
def time_map(wall, timeline):
    """Piecewise-linear wall clock -> eversion time."""
    acc = 0.0
    for dur, t0, t1 in timeline:
        if wall < acc + dur or (dur, t0, t1) is timeline[-1]:
            u = np.clip((wall - acc) / max(dur, 1e-9), 0.0, 1.0)
            return float(t0 + (t1 - t0) * u)
        acc += dur
    return float(timeline[-1][2])


class EversionAnimation(Animation):
    def __init__(self, carrier, mesh, timeline=None, run_time=TOTAL,
                 rate_func=None):
        super().__init__(carrier, run_time=run_time, rate_func=rate_func)
        self.mesh = mesh
        self.timeline = timeline or TIMELINES["forward"]

    def interpolate_mobject(self, alpha):
        t = time_map(alpha * self.run_time, self.timeline)
        self.mesh.fill(t)


class EversionScene(ThreeDScene):
    def __init__(self, mode=None, **kwargs):
        super().__init__(**kwargs)
        self.mode = MODE if mode is None else mode

    def construct(self):
        win = MLWindow(1920, 1080)
        win.scene = self

        # camera: 35 deg elevation, slightly off-axis azimuth
        camera = self.camera
        camera.set_phi(np.radians(55.0))
        camera.set_theta(np.radians(-20.0))

        timeline = TIMELINES[self.mode]
        total = sum(seg[0] for seg in timeline)

        mesh = EversionMesh()
        R = camera.get_rotation_matrix()
        mesh.eye = R.T @ np.array([0.0, 0.0, camera.get_focal_distance()])
        mesh.fill(timeline[0][1])
        win.set_mesh(mesh.batch)

        carrier = rtm.Mobject()
        self.add(carrier)
        win.play(EversionAnimation(carrier, mesh, timeline=timeline,
                                   run_time=total), run_time=total)
        win.close()


if __name__ == "__main__":
    if "--cycle" in sys.argv:
        MODE = "cycle"
    elif "--inverse" in sys.argv:
        MODE = "inverse"
    suffix = "" if MODE == "forward" else "_" + MODE
    out = os.path.expanduser("~/gallary/video/thurston_eversion%s.mp4"
                             % suffix)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r = fast_record_scene(EversionScene, out, overwrite=True)
    print("files:", r["files"])
