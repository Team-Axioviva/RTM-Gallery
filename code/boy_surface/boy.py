r"""Bryant-Kusner parametrization of Boy's surface.

Domain: closed unit disk |w| <= 1 (w = r e^{i phi}), with the antipodal map
w -> -w identifying boundary points (P(w) = P(-w) at |w| = 1): the quotient
is RP^2.

P(w) = (g1, g2, g3) / (g1^2 + g2^2 + g3^2)
g1 = -(3/2) Im[ w(1-w^4) / (w^6 + sqrt(5) w^3 - 1) ]
g2 = -(3/2) Re[ w(1+w^4) / (w^6 + sqrt(5) w^3 - 1) ]
g3 = Im[ (1+w^6) / (w^6 + sqrt(5) w^3 - 1) ] - 1/2

Structure (Wikipedia): the Moebius band lives in the annulus
{r0 <= r <= 1} (the boundary circle r = 1 is folded by the antipodal
identification; its boundary is the loop r = r0).  The disk r <= r0 caps
it: RP^2 = Moebius band + disk.  The triple point is at the origin, whose
three preimages are the poles of the rational functions, i.e. the roots of
w^6 + sqrt(5) w^3 - 1 = 0 with |w| < 1:

    r0 = ((3 - sqrt(5)) / 2)^(1/3)   ~ 0.7256
"""
import numpy as np

SQ5 = np.sqrt(5.0)

# |w| of the three triple-point preimages (roots of w^6+sqrt5 w^3-1=0)
R0 = ((3.0 - SQ5) / 2.0) ** (1.0 / 3.0)


def g_parts(w):
    """complex w (array) -> (g1, g2, g3) real arrays."""
    w2 = w * w
    w3 = w2 * w
    w4 = w2 * w2
    w6 = w3 * w3
    den = w6 + SQ5 * w3 - 1.0
    f1 = w * (1.0 - w4) / den
    f2 = w * (1.0 + w4) / den
    f3 = (1.0 + w6) / den
    g1 = -1.5 * f1.imag
    g2 = -1.5 * f2.real
    g3 = f3.imag - 0.5
    return g1, g2, g3


def P_of_w(w):
    g1, g2, g3 = g_parts(w)
    n = g1 * g1 + g2 * g2 + g3 * g3
    return np.stack([g1 / n, g2 / n, g3 / n], axis=-1)


def eval_grid(r, phi):
    """P on the r x phi grid, shape (len(r), len(phi), 3)."""
    R, PH = np.meshgrid(r, phi, indexing="ij")
    return P_of_w(R * np.exp(1j * PH))


def normals(r, phi, h=1e-6):
    """Per-grid-point normals by central differences (unnormalized).

    Edges: one-sided differences.  Near the triple-point preimages the
    chart derivative blows up; callers should clamp magnitudes.
    """
    P = eval_grid(r, phi)
    Pr = np.empty_like(P)
    for i in range(len(r)):
        if i == 0:
            Pr[i] = (eval_grid(r[:2], phi)[0][1] - P[0]) / (r[1] - r[0])
        elif i == len(r) - 1:
            Pr[i] = (P[-1] - eval_grid(r[-2:], phi)[0][0]) / (r[-1] - r[-2])
        else:
            Pr[i] = (eval_grid(np.array([r[i] + h]), phi)[0][0]
                     - eval_grid(np.array([r[i] - h]), phi)[0][0]) / (2 * h)
    Ph = np.empty_like(P)
    dphi = phi[1] - phi[0]
    for j in range(len(phi)):
        if j == 0:
            Ph[:, j] = (P[:, 1] - P[:, 0]) / dphi
        elif j == len(phi) - 1:
            Ph[:, j] = (P[:, -1] - P[:, -2]) / dphi
        else:
            Ph[:, j] = (eval_grid(r, np.array([phi[j] + h]))[:, 0]
                        - eval_grid(r, np.array([phi[j] - h]))[:, 0]) / (2 * h)
    return np.cross(Pr, Ph), P


def quad_indices(nr, nphi):
    """(nr*nphi, 4) index array of grid cells, corners (r_i, phi_j),
    (r_{i+1}, phi_j), (r_{i+1}, phi_{j+1}), (r_i, phi_{j+1})."""
    ii = np.arange(nr)
    jj = np.arange(nphi)
    I = np.broadcast_to(ii[:, None], (nr, nphi)).ravel()
    J = np.broadcast_to(jj[None, :], (nr, nphi)).ravel()
    Q = np.empty((nr * nphi, 4), dtype=np.int64)
    Q[:, 0] = I * (nphi + 1) + J
    Q[:, 1] = (I + 1) * (nphi + 1) + J
    Q[:, 2] = (I + 1) * (nphi + 1) + J + 1
    Q[:, 3] = I * (nphi + 1) + J + 1
    return Q
