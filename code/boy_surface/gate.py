r"""Analytic immersion gate for the Bryant-Kusner Boy surface.

|P_r x P_phi| = 2 r |P_w x P_wb|  (exact, from w = r e^{i phi})
with P_w = dP/dw, P_wb = dP/d(w_bar), computed by the complex chain rule
(all functions rational with real coefficients).
"""
import numpy as np
import boy

def ders(w):
    """P, P_w, P_wb at complex w (any shape)."""
    w2 = w * w
    w3 = w2 * w
    w4 = w2 * w2
    w6 = w3 * w3
    D = w6 + boy.SQ5 * w3 - 1.0
    Dp = 6.0 * (w2 * w3) + 3.0 * boy.SQ5 * w2          # 6 w^5 + 3 sqrt5 w^2
    # f1 = w(1-w^4)/D
    f1 = w * (1.0 - w4) / D
    f1p = ((1.0 - 5.0 * w4) * D - w * (1.0 - w4) * Dp) / (D * D)
    # f2 = w(1+w^4)/D
    f2 = w * (1.0 + w4) / D
    f2p = ((1.0 + 5.0 * w4) * D - w * (1.0 + w4) * Dp) / (D * D)
    # f3 = (1+w^6)/D
    f3 = (1.0 + w6) / D
    f3p = (6.0 * (w2 * w3) * D - (1.0 + w6) * Dp) / (D * D)

    # g1 = -(3/2) Im f1 = (3i/4)(f1 - f1b),  g2 = -(3/2) Re f2 = -(3/4)(f2+f2b)
    # g3 = Im f3 - 1/2 = (f3 - f3b)/(2i) - 1/2
    g1 = -1.5 * f1.imag
    g2 = -1.5 * f2.real
    g3 = f3.imag - 0.5
    g1w = 0.75j * f1p
    g2w = -0.75 * f2p
    g3w = f3p / (2.0j)
    g1b = -0.75j * np.conj(f1p)     # d/dw_bar f1(w_bar) = f1'(w_bar) = conj(f1'(w))
    g2b = -0.75 * np.conj(f2p)
    g3b = -np.conj(f3p) / (2.0j)

    n = g1 * g1 + g2 * g2 + g3 * g3
    P = np.stack([g1 / n, g2 / n, g3 / n], axis=-1)
    Gw = np.stack([g1w, g2w, g3w], axis=-1)
    Gb = np.stack([g1b, g2b, g3b], axis=-1)
    G = np.stack([g1, g2, g3], axis=-1)
    nw = 2.0 * (G * Gw).sum(-1, keepdims=True)
    nb = 2.0 * (G * Gb).sum(-1, keepdims=True)
    Pw = (Gw * n[..., None] - G * nw) / (n * n)[..., None]
    Pwb = (Gb * n[..., None] - G * nb) / (n * n)[..., None]
    return P, Pw, Pwb


def gate(r, phi):
    R, PH = np.meshgrid(r, phi, indexing="ij")
    w = R * np.exp(1j * PH)
    P, Pw, Pwb = ders(w)
    cross = np.cross(Pw, Pwb)
    mag = 2.0 * R * np.linalg.norm(cross, axis=-1)
    return P, mag


if __name__ == "__main__":
    NR, NPHI = 600, 1200
    r = np.linspace(0.0, 1.0, NR + 1)
    phi = np.linspace(0.0, 2 * np.pi, NPHI + 1)
    P, mag = gate(r, phi)
    R, PH = np.meshgrid(r, phi, indexing="ij")
    mask = (R > 0.02) & (np.abs(R - boy.R0) > 0.02) & (R < 0.999)
    mm = np.where(mask, mag, np.inf)
    i = np.unravel_index(np.argmin(mm), mm.shape)
    print("analytic gate min |P_r x P_phi| = %.6e at r=%.4f phi=%.4f (deg %.2f)"
          % (mm[i], r[i[0]], phi[i[1]], np.degrees(phi[i[1]])))
    # sample around it
    r0_, p0_ = r[i[0]], phi[i[1]]
    rr = np.linspace(max(r0_ - 0.03, 0.01), min(r0_ + 0.03, 0.999), 400)
    pp = np.linspace(p0_ - 0.05, p0_ + 0.05, 400)
    _, mm2 = gate(rr, pp)
    j = np.unravel_index(np.argmin(mm2), mm2.shape)
    print("refined min %.6e at r=%.6f phi=%.6f" % (mm2[j], rr[j[0]], pp[j[1]]))
    print("P there =", np.round(P[i], 6))
