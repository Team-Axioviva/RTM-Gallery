r"""Gate for the Thurston corrugation eversion — verified against the C++ source.

Checks, in order:
  1. ORACLE: the port reproduces the compiled original `evert`
     (cpp_oracle/evert_dump, built from osbm/sphereEversion-0.4-src)
     pointwise on the strip patch at six times spanning all five phases.
     Tolerance 1e-5 (the C++ stores float32; the true match is ~1e-7).
  2. Endpoints are exact unit spheres, X(12) = -X(0).
  3. The strip is an immersion over every phase (min |dtau X x dphi X|).
  4. The two hemispheres' equator edges coincide (closed surface).
  5. max|X| stays bounded.

Run before any render; a crease only whispers in a picture, this makes it
a number -- and the oracle makes "like the real thing" a number too.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import eversion as ev

HERE = os.path.dirname(os.path.abspath(__file__))
NS = 8
U, V = 24, 96
# the six C++ dump times and the corresponding times in our schedule
ORACLE = [("0.0", 0.0), ("0.05", 0.5), ("0.165", 2.0),
          ("0.415", 6.0), ("0.765", 10.0), ("0.965", 11.5)]


def cpp_surface(T):
    path = os.path.join(HERE, "cpp_oracle", f"dump_{T}.txt")
    d = np.loadtxt(path)
    C = np.zeros((U + 1, V + 1, 3))
    for row in d:
        C[int(row[0]), int(row[1])] = row[2:5]
    return C


def my_surface(t):
    u = np.linspace(0, 1, U + 1)
    v = np.linspace(0, 1, V + 1)
    TV, PV = np.meshgrid(v, u * np.pi / 2, indexing="ij")
    tauj = (TV, np.zeros_like(TV), np.ones_like(TV),
            np.zeros_like(TV), np.zeros_like(TV), np.zeros_like(TV))
    phj = (PV, np.ones_like(PV), np.zeros_like(PV),
           np.zeros_like(PV), np.zeros_like(PV), np.zeros_like(PV))
    X = ev.strip(t, tauj, phj, NS)
    P = np.stack([X[i][0] for i in range(3)], axis=-1)   # (V+1, U+1, 3)
    return np.transpose(P, (1, 0, 2))                     # (U+1, V+1, 3)


def main():
    # 1. oracle ------------------------------------------------------
    print("ORACLE (vs compiled original evert):")
    worst = 0.0
    for T, t in ORACLE:
        err = np.abs(my_surface(t) - cpp_surface(T)).max()
        worst = max(worst, err)
        print(f"  t={t:4.1f}  (C++ time {T:>5}):  max|dX| = {err:.3e}")
    print(f"  worst = {worst:.3e}  (float32 storage floor ~1e-7)")
    assert worst < 1e-5, "port diverges from the original C++"

    # 2. endpoints ---------------------------------------------------
    phg = np.linspace(0.0, np.pi / 2, 64)
    tao = np.linspace(0.0, 1.0, 64)
    TT, PP = np.meshgrid(tao, phg, indexing="ij")
    one, zero = np.ones_like(TT), np.zeros_like(TT)
    tauj = (TT, zero, one, zero, zero, zero)
    phj = (PP, one, zero, zero, zero, zero)

    def PX(t):
        X = ev.strip(t, tauj, phj, NS)
        return np.stack([X[i][0] for i in range(3)], axis=-1)

    P0, P12 = PX(0.0), PX(12.0)
    e0 = ev.unit_sphere_check(P0)
    e12 = ev.unit_sphere_check(P12)
    mirror = np.max(np.abs(P12 + P0))
    print(f"\nendpoints: max||X|-1| t=0 {e0:.2e}, t=12 {e12:.2e}, "
          f"|X(12)+X(0)| {mirror:.2e}")
    assert e0 < 1e-12 and e12 < 1e-12 and mirror < 1e-12

    # 3. strip immersion sweep ---------------------------------------
    print("\nphase         t-range    min |dtau x dphi| (phi > 0.2)")
    TA, PH = np.meshgrid(np.linspace(0, 1, 120), np.linspace(0, np.pi / 2, 96),
                         indexing="ij")
    lo = np.searchsorted(np.linspace(0, np.pi / 2, 96), 0.2)
    tauj_s = (TA, np.zeros_like(TA), np.ones_like(TA),
              np.zeros_like(TA), np.zeros_like(TA), np.zeros_like(TA))
    phj_s = (PH, np.ones_like(PH), np.zeros_like(PH),
             np.zeros_like(PH), np.zeros_like(PH), np.zeros_like(PH))
    overall = np.inf
    for name, tini, tdel in ev.SCHEDULE:
        worst = np.inf
        for t in np.linspace(tini, tini + tdel, 41, endpoint=True):
            X = ev.strip(t, tauj_s, phj_s, NS)
            dph = np.stack([X[i][1] for i in range(3)], axis=-1)
            dta = np.stack([X[i][2] for i in range(3)], axis=-1)
            area = np.linalg.norm(np.cross(dph, dta), axis=-1)
            worst = min(worst, area[:, lo:].min())
        print(f"{name:<13} [{tini:4.1f},{tini+tdel:4.1f}]  {worst:10.5f}")
        overall = min(overall, worst)
    print(f"\nOVERALL min strip |dtau x dphi| = {overall:.5f}")
    assert overall > 1e-6, "strip pinches"

    # 4. equator seam: the south edge web must lie on the north edge web
    # (each hemisphere is 8 rotated copies; the strips interleave -- the
    # zipper -- so the closure is web-to-web, not edge-to-edge).
    def rotz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    edge_t = np.linspace(0, 1, 400)
    phj_e = (np.full((1, 400), np.pi / 2), np.ones((1, 400)),
             np.zeros((1, 400)), np.zeros((1, 400)),
             np.zeros((1, 400)), np.zeros((1, 400)))
    tauj_e = (edge_t[None, :], np.zeros((1, 400)), np.ones((1, 400)),
              np.zeros((1, 400)), np.zeros((1, 400)), np.zeros((1, 400)))
    worst_seam = 0.0
    for t in np.linspace(0.0, 12.0, 13):
        X = ev.strip(t, tauj_e, phj_e, NS)
        E = np.stack([X[i][0][0] for i in range(3)], axis=-1)   # (400, 3)
        EM = E @ np.diag([-1.0, 1.0, -1.0])                      # RotY(pi) E
        Npts = np.concatenate([E @ rotz(-2 * np.pi * s / NS).T
                               for s in range(NS)], axis=0)
        Spts = np.concatenate([EM @ rotz(2 * np.pi * (s + 1) / NS).T
                               for s in range(NS)], axis=0)
        d2 = ((Spts[:, None, :] - Npts[None, :, :]) ** 2).sum(-1)
        worst_seam = max(worst_seam, np.sqrt(d2.min(axis=1)).max())
    print(f"equator seam: max dist south-web -> north-web = {worst_seam:.2e}")
    assert worst_seam < 1e-9

    # 5. bounded -----------------------------------------------------
    rmax = 0.0
    for t in np.linspace(0.0, 12.0, 49):
        rmax = max(rmax, np.linalg.norm(PX(t), axis=-1).max())
    print(f"max |X| = {rmax:.4f}")
    assert rmax < 5.0

    print("\nGATE PASSED: port == original C++ pointwise; strip immersed; "
          "surface closed.")


if __name__ == "__main__":
    main()
