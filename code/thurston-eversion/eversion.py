r"""Thurston's corrugation sphere eversion — numpy port of the `evert` geometry.

Source: Nathaniel Thurston's `evert` (Geometry Center, 1995), the program
behind the film *Outside In*.  This port is verified POINTWISE against the
original C++ `generateGeometry.cpp` (compiled and run; see gate.py).

The C++ structure, which this file reproduces exactly:

  * Coordinates: u in [0,1] maps pole -> equator; v in [0,1] maps to
    LONGITUDE [0, numStrips] -- v is the fraction ACROSS one strip of
    width 2*pi/N.  (A strip is a band, not a revolution.)
  * The base stage is evaluated at v = 0 with a dv seed of 1: a single
    MERIDIAN whose jet still carries the azimuthal derivative
    d/dv = 2*pi*(xsize*sin(phi), 0, 0).
  * Corrugate/AddFigureEight: the figure-eight profile runs ONCE across
    the strip (v in [0,1]); it vanishes at v = 0 and v = 1, so adjacent
    strips meet at uncorrugated guide meridians.  size = form*scale with
    the FFInterp/FSInterp envelope pair; the bend term keeps the loops
    attached; the final RotateZ(v/N) (= our qrot(-2*pi*v/N)) splays the
    band across its strip width.
  * The full sphere = 8 northern copies at RotZ(-s*2*pi/N) plus 8
    southern copies at RotZ((s+1)*2*pi/N) o RotY(pi) -- sixteen strips,
    the script's "sixteen rotated copies"; the south's +1 stagger makes
    the equator zipper.
  * Scene23's rotations use the C++ convention RotateZ(a) = qrot(-2*pi*a).

Jets are 6-tuples (f, fp, ft, fpp, fpt, ftt): value and derivatives up to
order 2 wrt the two parameters (phi, tau) with tau the strip parameter.
Everything is vectorized over a (Ntau, Nphi) grid.  A vjet is a 3-vector
of jets.
"""

import numpy as np

TAU = 2.0 * np.pi

# ----------------------------------------------------------------------
# second-order jet arithmetic (slots: fp = d/dphi, ft = d/dtau)
# ----------------------------------------------------------------------

def _asjet(x):
    if isinstance(x, tuple):
        return x
    a = np.asarray(x)
    z = np.zeros_like(a)
    return (a, z, z, z, z, z)

def jscale(a, c):
    a = _asjet(a)
    return (a[0] * c, a[1] * c, a[2] * c, a[3] * c, a[4] * c, a[5] * c)

def jadd(a, b):
    a, b = _asjet(a), _asjet(b)
    return tuple(a[k] + b[k] for k in range(6))

def jsub(a, b):
    a, b = _asjet(a), _asjet(b)
    return tuple(a[k] - b[k] for k in range(6))

def jmul(a, b):
    a, b = _asjet(a), _asjet(b)
    f = a[0] * b[0]
    fp = a[1] * b[0] + a[0] * b[1]
    ft = a[2] * b[0] + a[0] * b[2]
    fpp = a[3] * b[0] + 2 * a[1] * b[1] + a[0] * b[3]
    fpt = a[4] * b[0] + a[1] * b[2] + a[2] * b[1] + a[0] * b[4]
    ftt = a[5] * b[0] + 2 * a[2] * b[2] + a[0] * b[5]
    return (f, fp, ft, fpp, fpt, ftt)

def jpow(a, s):
    a = _asjet(a)
    safe = np.where(a[0] == 0, 1.0, a[0])
    f0 = a[0] ** s
    x1 = np.where(a[0] == 0, 0.0, s * f0 / safe)
    x2 = np.where(a[0] == 0, 0.0, (s - 1) * x1 / safe)
    x3 = np.where(a[0] == 0, 0.0, (s - 2) * x2 / safe)
    return (f0,
            x1 * a[1],
            x1 * a[2],
            x1 * a[3] + x2 * a[1] * a[1],
            x1 * a[4] + x2 * a[1] * a[2],
            x1 * a[5] + x2 * a[2] * a[2])

def jsin(a):
    a = _asjet(a)
    s, c = np.sin(a[0]), np.cos(a[0])
    return (s,
            c * a[1],
            c * a[2],
            c * a[3] - s * a[1] * a[1],
            c * a[4] - s * a[1] * a[2],
            c * a[5] - s * a[2] * a[2])

def jcos(a):
    a = _asjet(a)
    c, s = np.cos(a[0]), np.sin(a[0])
    return (c,
            -s * a[1],
            -s * a[2],
            -s * a[3] - c * a[1] * a[1],
            -s * a[4] - c * a[1] * a[2],
            -s * a[5] - c * a[2] * a[2])

def jmod(a, m):
    """modulo of the VALUE only (derivative slots untouched), as in evert."""
    a = _asjet(a)
    return (np.mod(a[0], m), a[1], a[2], a[3], a[4], a[5])

def jinterp(x0, x1, t):
    x0, x1, t = _asjet(x0), _asjet(x1), _asjet(t)
    return jadd(jmul(jsub(1.0, t), x0), jmul(t, x1))

def jpartial(a, idx):
    """partial derivative wrt slot idx (0 = phi, 1 = tau), carrying the
    second derivatives the way the C++ ThreeJet::D does."""
    a = _asjet(a)
    z = np.zeros_like(a[0])
    if idx == 0:
        return (a[1], a[3], a[4], z, z, z)
    return (a[2], a[4], a[5], z, z, z)


# ----------------------------------------------------------------------
# vjet (3-vector of jets)
# ----------------------------------------------------------------------

def vadd(a, b):
    return [jadd(a[i], b[i]) for i in range(3)]

def vsub(a, b):
    return [jsub(a[i], b[i]) for i in range(3)]

def vscale(s, v):
    return [jmul(s, v[i]) for i in range(3)]

def vdot(a, b):
    return jadd(jadd(jmul(a[0], b[0]), jmul(a[1], b[1])), jmul(a[2], b[2]))

def vcross(a, b):
    return [jsub(jmul(a[1], b[2]), jmul(a[2], b[1])),
            jsub(jmul(a[2], b[0]), jmul(a[0], b[2])),
            jsub(jmul(a[0], b[1]), jmul(a[1], b[0]))]

def vnormalize(v):
    s = vdot(v, v)
    val = s[0]
    ok = val > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(ok, val ** -0.5, 0.0)
        dinv = np.where(ok, -0.5 * val ** -1.5, 0.0)
        d2inv = np.where(ok, 0.75 * val ** -2.5, 0.0)
    inj = (inv,
           dinv * s[1],
           dinv * s[2],
           d2inv * s[1] * s[1] + dinv * s[3],
           d2inv * s[1] * s[2] + dinv * s[4],
           d2inv * s[2] * s[2] + dinv * s[5])
    return vscale(inj, v)

def vpartial(v, idx):
    return [jpartial(v[i], idx) for i in range(3)]

def vinterp(a, b, t):
    return [jinterp(a[i], b[i], t) for i in range(3)]


# ----------------------------------------------------------------------
# quaternion rotations (jet-valued, Hamilton product), right-handed +angle
# ----------------------------------------------------------------------

def qmul(q0, q1):
    a0, b0, c0, d0 = q0
    a1, b1, c1, d1 = q1
    return [jsub(jsub(jsub(jmul(a0, a1), jmul(b0, b1)), jmul(c0, c1)), jmul(d0, d1)),
            jsub(jadd(jadd(jmul(a0, b1), jmul(b0, a1)), jmul(c0, d1)), jmul(d0, c1)),
            jadd(jsub(jadd(jmul(a0, c1), jmul(c0, a1)), jmul(b0, d1)), jmul(d0, b1)),
            jadd(jsub(jadd(jmul(a0, d1), jmul(b0, c1)), jmul(c0, b1)), jmul(d0, a1))]

def qconj(q):
    return [q[0]] + [jscale(q[i], -1.0) for i in (1, 2, 3)]

def versor(angle, dirvec):
    h = jscale(angle, 0.5)
    n = vnormalize(dirvec)
    s = jsin(h)
    return [jcos(h), jmul(s, n[0]), jmul(s, n[1]), jmul(s, n[2])]

def qrot(v, q):
    """q * (0, v) * conj(q): right-handed rotation by +angle about the axis."""
    z = (np.zeros_like(v[0][0]),) * 6
    p = qmul(qmul(q, [z, v[0], v[1], v[2]]), qconj(q))
    return p[1:4]

_ZAXIS = [(np.zeros((1, 1)),) * 6 for _ in range(3)]
_ZAXIS[2] = (np.ones((1, 1)),) + (np.zeros((1, 1)),) * 5
_YAXIS = [(np.zeros((1, 1)),) * 6 for _ in range(3)]
_YAXIS[1] = (np.ones((1, 1)),) + (np.zeros((1, 1)),) * 5


def qrot_xy(v, angle):   # rotation about z
    return qrot(v, versor(angle, _ZAXIS))

def qrot_zx(v, angle):   # rotation about y
    return qrot(v, versor(angle, _YAXIS))


# ----------------------------------------------------------------------
# the eversion family: C++ stages and scenes
# ----------------------------------------------------------------------

def C(x):
    """A jet that is a constant (the C++ `ThreeJet(x,0,0)`)."""
    return _asjet(x)


def sphere_arc(phi, vjet, rx, ry, rz):
    """The C++ `Arc`: (rx s sin2pi v, ry s cos2pi v, rz cos), s = sin(phi')."""
    s = jsin(phi)
    c = jcos(phi)
    sv = jsin(jscale(vjet, TAU))
    cv = jcos(jscale(vjet, TAU))
    return [jmul(jscale(s, rx), sv), jmul(jscale(s, ry), cv), jscale(c, rz)]


def phi_deform0(phi):
    """C++ Param1 mapped to radians: (0..pi) -> (0..pi), C^1."""
    left = jadd(jscale(jpow(phi, 2), -2.0 / np.pi), jscale(phi, 2.0))
    right = jadd(jadd(jscale(jpow(phi, 2), -2.0 / np.pi), jscale(phi, 2.0)),
                 np.pi)
    return _piecewise(phi, left, right)


def phi_deform1(phi):
    """C++ Param2 mapped to radians: (0..pi/2)->(0..pi/2), (pi/2..pi)->(5pi/2..3pi)."""
    left = jscale(jpow(phi, 2), 2.0 / np.pi)
    right = jadd(jadd(jscale(jpow(phi, 2), -2.0 / np.pi), jscale(phi, 4.0)),
                 np.pi)
    return _piecewise(phi, left, right)


def _hemi_u(phi):
    """Per-hemisphere C++ u: 2*phi/pi on the north, mirrored on the south."""
    m = phi[0] > np.pi / 2
    v = np.where(m, 2.0 * (np.pi - phi[0]) / np.pi, 2.0 * phi[0] / np.pi)
    d = np.where(m, -2.0 / np.pi, 2.0 / np.pi)
    return (v, d * phi[1], d * phi[2], d * d * phi[3], d * d * phi[4],
            d * d * phi[5])


def _smooth3(x):
    """3x^2 - 2x^3 (the C++ cubic smootherstep half)."""
    return jsub(jscale(jpow(x, 2), 3.0), jscale(jpow(x, 3), 2.0))


def ff_interp(phi):
    """C++ FFInterp: smoothstep3(clamp(1.06u - 0.05)); profile envelope."""
    u = _hemi_u(phi)
    x = jsub(jscale(u, 1.06), 0.05)
    inside = (x[0] >= 0.0) & (x[0] <= 1.0)
    s = (np.clip(x[0], 0.0, 1.0),) + tuple(
        np.where(inside, x[k], 0.0) for k in range(1, 6))
    return _smooth3(s)


def fs_interp(phi):
    """C++ FSInterp: -0.2*smoothstep3(u); amplitude envelope (carries -0.2)."""
    return jscale(_smooth3(_hemi_u(phi)), -0.2)


def phi_deform2(phi):
    """C++ UInterp mapped to radians: bump, 0 at ends, 1 at pi/2."""
    m = phi[0] > np.pi / 2
    p = (np.where(m, np.pi - phi[0], phi[0]),) + tuple(
        np.where(m, -phi[k], phi[k]) for k in range(1, 6))
    return jadd(jscale(jpow(p, 3), -16.0 / np.pi ** 3),
                jscale(jpow(p, 2), 12.0 / np.pi ** 2))


def _piecewise(x, left, right):
    m = x[0] <= np.pi / 2
    return tuple(np.where(m, left[k], right[k]) for k in range(6))


def stage1(phi, vjet):
    return sphere_arc(phi, vjet, 1.0, 1.0, 1.0)


def stage2(phi, vjet):
    return vinterp(sphere_arc(phi_deform0(phi), vjet, 0.9, 0.9, -1.0),
                   sphere_arc(phi_deform1(phi), vjet, 1.0, 1.0, 0.5),
                   phi_deform2(phi))


def stage3(phi, vjet):
    return vinterp(sphere_arc(phi_deform0(phi), vjet, -0.9, -0.9, -1.0),
                   sphere_arc(phi_deform1(phi), vjet, -1.0, 1.0, -0.5),
                   phi_deform2(phi))


def stage4(phi, vjet):
    return sphere_arc(phi, vjet, -1.0, -1.0, -1.0)


def scene12(phi, vjet, t):
    return vinterp(stage1(phi, vjet), stage2(phi, vjet), t)


def scene23(phi, vjet, t):
    # C++: t = tmp.f*0.5; tt = (u<=1) ? t : -t; RotateZ(a) = qrot(-2pi a)
    th = jscale(t, 0.5)
    m = phi[0] <= np.pi / 2
    tt = (th[0],) + tuple(np.where(m, th[k], -th[k]) for k in range(1, 6))
    rot_z = qrot_xy(sphere_arc(phi_deform0(phi), vjet, 0.9, 0.9, -1.0),
                    jscale(tt, -TAU))
    rot_y = qrot_zx(sphere_arc(phi_deform1(phi), vjet, 1.0, 1.0, 0.5),
                    jscale(th, -TAU))
    return vinterp(rot_z, rot_y, phi_deform2(phi))


def scene34(phi, vjet, t):
    return vinterp(stage3(phi, vjet), stage4(phi, vjet), t)


# ----------------------------------------------------------------------
# the strip: meridian + figure-eight, splayed across one strip width
# ----------------------------------------------------------------------

def figure8(w, h, bend, form, tau):
    """C++ FigureEight: profile traced once across the strip (tau in [0,1])."""
    tp = jmod(tau, 1.0)
    height = jadd(jmul(jcos(jscale(tp, 2.0 * TAU)), -1.0), 1.0)
    m = (tp[0] > 0.25) & (tp[0] < 0.75)
    height = (np.where(m, 4.0 - height[0], height[0]),) + tuple(
        np.where(m, -height[k], height[k]) for k in range(1, 6))
    height = jscale(height, 0.6)
    h = vadd(h, vscale(jscale(jpow(height, 2), 1.0 / 64.0), bend))
    form = jsub(jscale(form, 2.0), jpow(form, 2))
    low = jscale(jsub(jcos(jscale(tp, TAU)), 1.0), -2.0)
    return vadd(vscale(jsin(jscale(tp, 2.0 * TAU)), w),
                vscale(jinterp(low, height, form), h))


def jdiv(a, b):
    return jmul(a, jpow(b, -1.0))


def phase_parts(t, phi, vjet):
    """(base p, figure-eight weight form, amplitude scale) for time t."""
    for name, tini, tdel in SCHEDULE:
        if t < tini + tdel or name == "UNCORRUGATE":
            tau = (t - tini) / tdel
            if name == "CORRUGATE":
                return stage1(phi, vjet), jmul(tau, ff_interp(phi)), fs_interp(phi)
            if name == "PUSH":
                return scene12(phi, vjet, tau), ff_interp(phi), fs_interp(phi)
            if name == "TWIST":
                return scene23(phi, vjet, tau), ff_interp(phi), fs_interp(phi)
            if name == "UNPUSH":
                return scene34(phi, vjet, tau), ff_interp(phi), fs_interp(phi)
            return stage4(phi, vjet), jmul(jsub(1.0, tau), ff_interp(phi)), fs_interp(phi)


def strip(t, tau, phj, nstrips, vjet=None):
    """One C++ strip: base meridian + figure-eight, splayed by qrot(-2pi tau/N).

    Jet slots are (dphi, dtau).  The C++ evaluates the base at v=0 with a dv
    seed, so its d/dv comes from the arc's x-components (the xsize radii) --
    reproduced here by passing the same vjet into every stage.
    """
    if vjet is None:
        z = np.zeros_like(phj[0])
        vjet = (z, z, np.ones_like(z), z, z, z)      # value 0, dtau = 1
    p, form, scale = phase_parts(t, phj, vjet)
    size = jmul(form, scale)
    du = vnormalize(vpartial(p, 0))
    dv = vnormalize(vpartial(p, 1))
    h = vscale(size, vnormalize(vcross(du, dv)))
    w = vscale(jscale(size, 1.1), vnormalize(vcross(h, du)))
    # C++ bend = du * D(size,0)/D(u,0) in ITS u-units (u in [0,1]);
    # our phi = u*pi/2, so d/du = (pi/2) d/dphi.
    bend = vscale(jscale(jdiv(jpartial(size, 0), jpartial(phj, 0)), np.pi / 2), du)
    fig8 = figure8(w, h, bend, form, tau)
    return qrot_xy(vadd(p, fig8), jscale(tau, -TAU / nstrips))


# phase schedule (seconds), as in the CUDA port
SCHEDULE = [
    ("CORRUGATE",   0.0,  1.0),
    ("PUSH",        1.0,  2.0),
    ("TWIST",       3.0,  6.0),
    ("UNPUSH",      9.0,  2.0),
    ("UNCORRUGATE", 11.0, 1.0),
]


# ----------------------------------------------------------------------
# mesh helpers: the strip over (tau, phi in [0, pi/2])
# ----------------------------------------------------------------------

def eval_strip(t, Ntau, Nphi, nstrips, phi_max=np.pi / 2):
    """Returns P (Nt,Np,3), normals, |dtau X x dphi X| on the strip chart."""
    tau = np.linspace(0.0, 1.0, Ntau)
    phi = np.linspace(0.0, phi_max, Nphi)
    TT, PP = np.meshgrid(tau, phi, indexing="ij")     # (Nt, Np)
    tauj = (TT, np.zeros_like(TT), np.ones_like(TT),
            np.zeros_like(TT), np.zeros_like(TT), np.zeros_like(TT))
    phj = (PP, np.ones_like(PP), np.zeros_like(PP),
           np.zeros_like(PP), np.zeros_like(PP), np.zeros_like(PP))
    X = strip(t, tauj, phj, nstrips)
    P = np.stack([X[i][0] for i in range(3)], axis=-1)
    dph = np.stack([X[i][1] for i in range(3)], axis=-1)
    dta = np.stack([X[i][2] for i in range(3)], axis=-1)
    nrm = np.cross(dta, dph)          # outward at t=0; flips through the eversion
    area = np.linalg.norm(nrm, axis=-1)
    nhat = nrm / np.maximum(area, 1e-300)[..., None]
    # pole row: dtau -> 0 there; use the limit from the adjacent row
    if Nphi > 2:
        nhat[:, 0] = nhat[:, 1]
    return P, nhat, area


def unit_sphere_check(P):
    R = np.linalg.norm(P, axis=-1)
    return np.max(np.abs(R - 1.0))


if __name__ == "__main__":
    P, n, a = eval_strip(0.0, 33, 49, 8)
    print("t=0   max||X|-1| =", unit_sphere_check(P))
    P, n, a = eval_strip(12.0, 33, 49, 8)
    print("t=12  max||X|-1| =", unit_sphere_check(P))
