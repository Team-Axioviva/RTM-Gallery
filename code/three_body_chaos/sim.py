"""
Pythagorean three-body problem — simulate, reparameterize, sample.

ICs (Szebehely & Peters 1967): masses 3,4,5 at (1,3), (-2,-1), (1,-1), v = 0, G = 1.
The canonical maximally-chaotic three-body configuration.

Pipeline:
  1. DOP853 (rtol 1e-12) integrate to T_MAX, dense output.
  2. Diagnostics: energy/momentum drift, chaos divergence, encounter log.
  3. Cut T_end before the ejection (r_max threshold) so everything stays in frame.
  4. Reparameterize to ~constant kinetic speed (with floor) so close-encounter
     whips stay smooth at 60 fps, then invert to sample the 30 s video grid.
  5. Compute per-frame positions, velocities, pairwise forces F_ij (on j by i),
     net forces, and arrow geometry (rim starts, power-compressed lengths).
  6. Save data/frames.npz + diagnostics.json for the RTM scene.
"""
import json
import os

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

G = 1.0
SCALE = 1.7                     # world scale: bigger inter-star distances
MASS = np.array([3.0, 4.0, 5.0])
R0 = np.array([[1.0, 3.0], [-2.0, -1.0], [1.0, -1.0]]) * SCALE
V0 = np.zeros((3, 2))
T_MAX = 60.0 * SCALE ** 1.5     # same physical stage under the r->l r, t->l^1.5 t symmetry
R_MAX_TRIGGER = 13.5 * SCALE    # cut before ejection carries a star off frame

DURATION = 30.0
FPS = 60
N_FRAMES = int(DURATION * FPS) + 1          # 1801
T_VIDEO = np.arange(N_FRAMES) / FPS

# --- star visuals (same-density radii)
R_STAR = 0.34 * (MASS / 4.0) ** (1.0 / 3.0)
# display clamp: stars never touch (centroid-conserving, smoothstep-softened)
KISS_DELTA = 0.12

# --- arrow compression params
P_EXP = 0.3                    # length ~ F^P_EXP (huge dynamic range -> readable)
L0 = 1.6
LMAX_NET = 4.0
LMIN = 0.18
HIDE_LEN = 0.15
RIM_PAD = 0.05                 # arrow starts just outside the star limb


def accel(r):
    a = np.zeros_like(r)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            d = r[j] - r[i]
            a[i] += G * MASS[j] * d / np.linalg.norm(d) ** 3
    return a


def rhs(t, y):
    r = y[:6].reshape(3, 2)
    v = y[6:].reshape(3, 2)
    return np.concatenate([v.ravel(), accel(r).ravel()])


y0 = np.concatenate([R0.ravel(), V0.ravel()])
sol = solve_ivp(rhs, (0.0, T_MAX), y0, method="DOP853",
                rtol=1e-12, atol=1e-13, dense_output=True, max_step=0.05).sol
print(f"integrator done, nfev n/a (dense)")

# ---------------- diagnostics ----------------
N_FINE = 40000
t_fine = np.linspace(0.0, T_MAX, N_FINE)
Y = sol(t_fine)                                # (12, N)
R = Y[:6].reshape(3, 2, -1)
V = Y[6:].reshape(3, 2, -1)

r_mag = np.linalg.norm(R, axis=1)              # (3, N)
r_max = r_mag.max(axis=0)

# energy: T = 1/2 m v^2, U = - sum G m_i m_j / r_ij
kin = 0.5 * np.einsum("i,ikn,ikn->n", MASS, V, V)
pot = np.zeros(N_FINE)
for i in range(3):
    for j in range(i + 1, 3):
        d = np.linalg.norm(R[i] - R[j], axis=0)
        pot += -G * MASS[i] * MASS[j] / d
E = kin + pot
dE = np.abs(E - E[0]) / np.abs(E[0])

# momentum
P = np.einsum("i,ikn->kn", MASS, V)
dP = np.linalg.norm(P, axis=0)

# close encounters
dists = np.zeros((3, 3, N_FINE))
for i in range(3):
    for j in range(i + 1, 3):
        dists[i, j] = np.linalg.norm(R[i] - R[j], axis=0)
dmin_pair = {}
for i in range(3):
    for j in range(i + 1, 3):
        k = np.argmin(dists[i, j])
        dmin_pair[f"{i}{j}"] = (float(t_fine[k]), float(dists[i, j][k]))
n_enc = int(np.sum(np.min(np.stack([dists[0, 1], dists[0, 2], dists[1, 2]]), axis=0) < 0.5))

# chaos: one-body position perturbed by 1e-9, track separation
y1 = y0.copy()
y1[0] += 1e-9
sol1 = solve_ivp(rhs, (0.0, T_MAX), y1, method="DOP853",
                 rtol=1e-12, atol=1e-13, dense_output=True, max_step=0.05).sol
sep = np.linalg.norm(sol(t_fine)[:6] - sol1(t_fine)[:6], axis=0)
i_sep = int(np.argmax(sep > 1.0)) if np.any(sep > 1.0) else -1
t_sep = float(t_fine[i_sep]) if i_sep >= 0 else None
# Lyapunov slope: d/dt ln(sep) over the linear-growth window
_m = (t_fine > 10) & (sep > 1e-12) & (sep < 0.5)
lya = float(np.polyfit(t_fine[_m], np.log(sep[_m]), 1)[0]) if _m.sum() > 100 else float("nan")
print(f"separation at t=60: {sep[-1]:.3e}  lyapunov slope ~ {lya:.3f}/unit")

# ---------------- window cut ----------------
for _tt in (10, 20, 30, 40, 45, 50, 55, 58, 60):
    _k = int(np.argmin(np.abs(t_fine - _tt)))
    print(f"t={_tt}: r_max={r_max[_k]:.3f}   " +
          "  ".join(f"r{i}={r_mag[i, _k]:.3f}" for i in range(3)))
trig = np.where(r_max > R_MAX_TRIGGER)[0]
T_END = float(t_fine[trig[0]]) if trig.size else T_MAX
t_win = t_fine[t_fine <= T_END]
print(f"T_END = {T_END:.3f} (r_max trigger {R_MAX_TRIGGER})")

# ---------------- reparameterize: constant kinetic speed ----------------
Yw = sol(t_win)
Rw = Yw[:6].reshape(3, 2, -1)
Vw = Yw[6:].reshape(3, 2, -1)
S = np.sqrt(np.einsum("i,ikn,ikn->n", MASS, Vw, Vw))     # sqrt(2T)
S_floor = 0.20 * np.median(S)
S_cap = 2.5 * np.median(S)          # deep encounters whip through, not linger
S_eff = np.minimum(np.sqrt(S ** 2 + S_floor ** 2), S_cap)
dt = np.diff(t_win)
tau = np.concatenate([[0.0], np.cumsum(0.5 * (S_eff[1:] + S_eff[:-1]) * dt)])
tau_max = float(tau[-1])

# invert tau -> t at the video grid
t_sim = np.interp(T_VIDEO / DURATION * tau_max, tau, t_win)

# ---------------- sample at video grid ----------------
Ys = sol(t_sim)                                  # (12, 1801)
Rs = Ys[:6].reshape(3, 2, -1)                    # (3,2,N)
Vs = Ys[6:].reshape(3, 2, -1)

f_pair = np.zeros((3, 3, 2, N_FRAMES))           # [j, i] = force ON j BY i
for j in range(3):
    for i in range(3):
        if i == j:
            continue
        d = Rs[i] - Rs[j]
        dist = np.linalg.norm(d, axis=0)
        f_pair[j, i] = G * MASS[i] * MASS[j] * d / dist ** 3
f_net = f_pair.sum(axis=1)                       # (3,2,N)

F_pair_mag = np.linalg.norm(f_pair, axis=2)
F_net_mag = np.linalg.norm(f_net, axis=1)
F_REF = float(np.median(F_pair_mag[F_pair_mag > 0]))


def arrow_len(F):
    L = L0 * (F / F_REF) ** P_EXP
    return np.clip(L, LMIN, LMAX_NET)


# ---------------- arrow geometry ----------------
print("shapes:", f_pair.shape, f_net.shape, F_pair_mag.shape, F_net_mag.shape)
# net arrows: [star, frame] -> dir (2), len
net_dir = np.zeros((3, 2, N_FRAMES))
net_len = np.zeros((3, N_FRAMES))
for j in range(3):
    Fm = F_net_mag[j]
    ok = Fm > 1e-9
    u = np.zeros((2, N_FRAMES))
    u[:, ok] = f_net[j][:, ok] / Fm[ok][None, :]
    net_dir[j] = u
    net_len[j, ok] = arrow_len(Fm[ok])

# pair arrows: [on j, by i, frame] -> dir, len, vis
pair_dir = np.zeros((3, 3, 2, N_FRAMES))
pair_len = np.zeros((3, 3, N_FRAMES))
pair_vis = np.zeros((3, 3, N_FRAMES), dtype=bool)
for j in range(3):
    for i in range(3):
        if i == j:
            continue
        d = Rs[i] - Rs[j]
        dist = np.linalg.norm(d, axis=0)
        u = d / dist[None, :]
        L = arrow_len(F_pair_mag[j, i])
        cap = dist - R_STAR[j] - R_STAR[i] - 0.12
        L = np.minimum(L, np.maximum(cap, 0.0))
        pair_dir[j, i] = u
        pair_len[j, i] = L
        pair_vis[j, i] = L >= HIDE_LEN

# ---------------- display clamp: stars never touch ----------------
# Centroid-conserving: when a pair would overlap (d < R_i+R_j), displace the
# two stars symmetrically along their axis so the disks just kiss. Smoothstep
# over KISS_DELTA hides the engagement kink. Physics (forces, directions,
# energy) stays on the TRUE trajectory; only the displayed positions move.
pos_disp = Rs.transpose(2, 0, 1).copy()          # (N, 3, 2)
n_clamped = 0
for fr in range(N_FRAMES):
    for i in range(3):
        for j in range(i + 1, 3):
            d = np.linalg.norm(pos_disp[fr, i] - pos_disp[fr, j])
            rsum = R_STAR[i] + R_STAR[j]
            if d < rsum + KISS_DELTA:
                x = np.clip((rsum + KISS_DELTA - d) / KISS_DELTA, 0.0, 1.0)
                w = x * x * (3.0 - 2.0 * x)
                dd = d + (rsum - d) * w
                u = (pos_disp[fr, j] - pos_disp[fr, i]) / max(d, 1e-12)
                shift = (dd - d) / 2.0
                pos_disp[fr, i] -= u * shift
                pos_disp[fr, j] += u * shift
                n_clamped += 1
sep_min = min(
    np.linalg.norm(pos_disp[:, i] - pos_disp[:, j], axis=1).min()
    - R_STAR[i] - R_STAR[j]
    for i in range(3) for j in range(i + 1, 3)
)
print(f"clamp: {n_clamped} frame-pairs adjusted, "
      f"min displayed separation = {sep_min:.4f} (>= 0 required)")

# zoom to keep every star in frame (height = 8/z world units)
r_frame = r_mag.max(axis=0)
r_within = np.max(np.linalg.norm(Rs, axis=1) + R_STAR[:, None])
ZOOM = 4.0 / (float(r_within) + 1.0)
print(f"zoom = {ZOOM:.4f}  (r_within = {r_within:.3f})")

np.savez_compressed(
    os.path.join(DATA, "frames.npz"),
    t_sim=t_sim, pos=Rs.transpose(2, 0, 1), pos_disp=pos_disp,
    vel=Vs.transpose(2, 0, 1),
    f_pair=f_pair.transpose(3, 0, 1, 2), f_net=f_net.transpose(2, 0, 1),
    net_dir=net_dir.transpose(2, 0, 1), net_len=net_len.T,
    pair_dir=pair_dir.transpose(3, 0, 1, 2), pair_len=pair_len.transpose(2, 0, 1),
    pair_vis=pair_vis.transpose(2, 0, 1),
    mass=MASS, r_star=R_STAR, f_ref=F_REF, zoom=ZOOM,
)

diag = {
    "T_end": T_END, "r_max_trigger": R_MAX_TRIGGER,
    "r_max_in_window": float(r_max[t_fine <= T_END].max()),
    "energy_drift_max": float(dE.max()),
    "momentum_max": float(dP.max()),
    "n_close_encounters_lt0.5": n_enc,
    "closest_approaches": {k: {"t": v[0], "d": v[1]} for k, v in dmin_pair.items()},
    "chaos_t_sep_gt_1_after_1e-9_perturb": t_sep,
    "median_speed": float(np.median(S)), "speed_floor": float(S_floor),
    "tau_max": tau_max, "F_REF": F_REF,
    "video_t_span_of_sim": [float(t_sim[0]), float(t_sim[-1])],
    "sim_t_at_video_10s": float(t_sim[600]),
    "sim_t_at_video_20s": float(t_sim[1200]),
}
with open(os.path.join(DATA, "diagnostics.json"), "w") as fh:
    json.dump(diag, fh, indent=2)
print(json.dumps(diag, indent=2))
print("saved", os.path.join(DATA, "frames.npz"))
