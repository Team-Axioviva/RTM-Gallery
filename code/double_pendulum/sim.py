r"""
Double pendulum — RK4 simulation + per-frame render data.

Physics: m1 = m2 = L1 = L2 = 1 (SI), g = 9.81.
Angles theta measured from the downward vertical, positive CCW.
ICs: theta1 = 120 deg, theta2 = 0, omega = 0; ghosts perturb theta2 by +-1e-3.

Timeline (video t): 0-1 s static hold at IC; release at t = 1 s
(sim time s = t - 1); ghosts evolve from their own t = 1 release,
revealed at t = 20 s.

Per 60 fps frame k: positions, rod geometry, forces on each bob
(gravity, rod tensions along the rods), KE/PE of the main pendulum.

Diagnostics printed: RK4 energy drift, rod-force perpendicular residue
(EOM sanity), ghost divergence at reveal.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

G = 9.81
M1 = M2 = 1.0
L1 = L2 = 1.0

TH1_0 = np.deg2rad(120.0)
TH2_0 = 0.0
DELTA = 0.001            # ghost perturbation in theta2(0)

FPS = 60
DUR = 30.0               # video duration
N_FR = int(DUR * FPS)    # 1800 frames, t = k/60
RELEASE_T = 1.0          # video time of release
SIM_S = DUR - RELEASE_T  # 29 s of dynamics
DT = 1.0 / 2400.0        # 4 RK4 substeps per frame
N_STEP = int(round(SIM_S / DT))

PIVOT = np.array([-2.6, 1.0])   # world
DS = 1.0                        # display scale (world units per metre)

# force-arrow display params
F_SCALE = 0.03          # world units per newton
F_CAP = 1.15            # max arrow length (world)
F_MIN = 0.05            # hide shorter arrows


def deriv(y):
    """EOM rhs from the Lagrangian. y = [th1, w1, th2, w2]."""
    th1, w1, th2, w2 = y
    d = th1 - th2
    c = np.cos(d)
    s = np.sin(d)
    den = 2.0 - c * c
    a1 = (-w1 * w1 * s * c + G * np.sin(th2) * c
          - w2 * w2 * s - 2.0 * G * np.sin(th1)) / den
    a2 = -c * a1 + w1 * w1 * s - G * np.sin(th2)
    return np.array([w1, a1, w2, a2])


def rk4_step(y, dt):
    k1 = deriv(y)
    k2 = deriv(y + 0.5 * dt * k1)
    k3 = deriv(y + 0.5 * dt * k2)
    k4 = deriv(y + dt * k3)
    return y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------- integrate the three pendulums ----------------
ICS = []
for dth2 in (0.0, DELTA, -DELTA):
    ICS.append(np.array([TH1_0, 0.0, TH2_0 + dth2, 0.0]))

states = np.zeros((3, N_STEP + 1, 4))
for i in range(3):
    y = ICS[i].copy()
    states[i, 0] = y
    for n in range(N_STEP):
        y = rk4_step(y, DT)
        states[i, n + 1] = y
print("RK4 done:", N_STEP, "steps x 3 pendulums")

# ---------------- per-frame geometry / forces / energy ----------------
P1 = np.zeros((3, N_FR, 2))
P2 = np.zeros((3, N_FR, 2))
FMAIN = np.zeros((N_FR, 5, 2))   # [g1, T1on1, T2on1, g2, T2on2]
KE = np.zeros(N_FR)
PE = np.zeros(N_FR)

e_drift = 0.0
perp_max = 0.0
g_vec = np.array([0.0, -G])

for k in range(N_FR):
    t = k / FPS
    s = t - RELEASE_T
    if s <= 0.0:
        ys = [ic.copy() for ic in ICS]
        as_ = [np.zeros(2) for _ in range(3)]
    else:
        idx = int(round(s / DT))
        idx = min(idx, N_STEP)
        ys = [states[i, idx] for i in range(3)]
        as_ = [deriv(y)[1::2] for y in ys]   # (a1, a2) pairs

    for i in range(3):
        th1, w1, th2, w2 = ys[i]
        a1, a2 = as_[i]
        p1 = PIVOT + DS * np.array([L1 * np.sin(th1), -L1 * np.cos(th1)])
        p2 = p1 + DS * np.array([L2 * np.sin(th2), -L2 * np.cos(th2)])
        P1[i, k] = p1
        P2[i, k] = p2

        e1t = np.array([np.cos(th1), np.sin(th1)])
        e1r = np.array([np.sin(th1), -np.cos(th1)])
        e2t = np.array([np.cos(th2), np.sin(th2)])
        e2r = np.array([np.sin(th2), -np.cos(th2)])
        a1v = L1 * a1 * e1t - L1 * w1 * w1 * e1r
        a2v = a1v + L2 * a2 * e2t - L2 * w2 * w2 * e2r

        if i == 0:
            fg1 = M1 * g_vec
            fg2 = M2 * g_vec
            ft1 = M1 * a1v + M2 * a2v - (M1 + M2) * g_vec   # rod1 on m1
            ft2 = M2 * a2v - M2 * g_vec                      # rod2 on m2
            FMAIN[k] = [fg1, ft1, -ft2, fg2, ft2]
            # EOM sanity: rod forces must lie along the rods (only during
            # true dynamics; the t<1 hold is an external pin, not a rod)
            if s > 0.0:
                u1 = (p1 - PIVOT) / np.linalg.norm(p1 - PIVOT)
                u2 = (p2 - p1) / np.linalg.norm(p2 - p1)
                perp_max = max(perp_max,
                               abs(u1[0] * ft1[1] - u1[1] * ft1[0]),
                               abs(u2[0] * ft2[1] - u2[1] * ft2[0]))
            # energy (PE measured from the straight-down configuration)
            v1 = L1 * w1 * e1t
            v2 = v1 + L2 * w2 * e2t
            ke = 0.5 * np.dot(v1, v1) + 0.5 * np.dot(v2, v2)
            pe = (-(M1 + M2) * G * L1 * (np.cos(th1) - 1.0)
                  - M2 * G * L2 * (np.cos(th2) - 1.0))
            KE[k] = ke
            PE[k] = pe
            e_drift = max(e_drift, abs(ke + pe - KE[0] - PE[0]))

E0 = KE[0] + PE[0]
# divergence at reveal (video t = 20 -> s = 19)
k20 = int(20 * FPS)
d1 = abs((states[0, int(19 / DT), 2] - states[1, int(19 / DT), 2] + np.pi)
         % (2 * np.pi) - np.pi)
d2 = abs((states[0, int(19 / DT), 2] - states[2, int(19 / DT), 2] + np.pi)
         % (2 * np.pi) - np.pi)
print(f"E0 = {E0:.6f} J")
print(f"max |E - E0| = {e_drift:.3e} J   (rel {e_drift/E0:.3e})")
print(f"max rod-force perp residue = {perp_max:.3e} N  (must be ~0)")
print(f"theta2 divergence at reveal: {d1:.3f} rad, {d2:.3f} rad")

np.savez(os.path.join(DATA, "frames.npz"),
         P1=P1, P2=P2, FMAIN=FMAIN, KE=KE, PE=PE, E0=E0)
print("saved", os.path.join(DATA, "frames.npz"))
