"""
Build the textured mesh static data for the RTM scene.

Three stars, real SDO solar imagery (4096x4096):
  star 0 (m=3): AIA 193 Å  -> gold/bronze
  star 1 (m=4): AIA 304 Å  -> red
  star 2 (m=5): HMI intensitygram, blue colour grade -> blue star
                 (real white-light granulation texture, tinted)

Sphere = unit lat/lon grid; per-vertex colour = bilinear sample of the disk
image at the vertex's (x, y) — the disk projection wraps onto the sphere
(both hemispheres), so every viewing direction shows the solar face.

Outputs data/mesh_static.npz: unit verts, quad indices, per-vertex colours,
per-star arrow colours (mean texture colour, brightened).
"""
import os

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
DATA = os.path.join(HERE, "data")

NLAT, NLON = 36, 48
N_VERT = (NLAT + 1) * NLON
N_SPH_QUADS = NLAT * NLON

TINT_BLUE = np.array([0.55, 0.78, 1.50])   # HMIIC -> blue star
BRIGHTEN = {"193": 1.00, "304": 1.32, "hmiic": 1.10}


def build_grid():
    verts = np.zeros((N_VERT, 3))
    for i in range(NLAT + 1):
        th = np.pi * i / NLAT
        for j in range(NLON):
            ph = 2 * np.pi * j / NLON
            verts[i * NLON + j] = [np.sin(th) * np.cos(ph),
                                   np.sin(th) * np.sin(ph),
                                   np.cos(th)]
    quads = np.empty((N_SPH_QUADS, 4), dtype=np.int64)
    q = 0
    for i in range(NLAT):
        for j in range(NLON):
            jn = (j + 1) % NLON
            quads[q] = [i * NLON + j, i * NLON + jn,
                        (i + 1) * NLON + jn, (i + 1) * NLON + j]
            q += 1
    # (winding deliberately left as-is: the polar quads are degenerate and the
    #  mesh pass does not backface-cull; verified by stills before recording)
    return verts, quads


def sample_texture(img, verts):
    h, w = img.shape[:2]
    px = (verts[:, 0] * 0.5 + 0.5) * (w - 1)
    py = (verts[:, 1] * 0.5 + 0.5) * (h - 1)
    out = np.zeros((len(verts), 3))
    for c in range(3):
        out[:, c] = map_coordinates(img[:, :, c].astype(np.float64),
                                    [py, px], order=1, mode="nearest")
    return out / 255.0


def ascii_preview(name, rgb):
    """Coarse 44x22 luminance ASCII of the vertex disk (eye-free sanity)."""
    u = UNIT_XY  # noqa: F821  (set below in main)
    im = np.zeros((44, 44))
    px = ((u[:, 0] + 1) / 2 * 43).astype(int)
    py = ((u[:, 1] + 1) / 2 * 43).astype(int)
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    np.maximum.at(im, (py, px), lum)
    ramp = " .:-=+*#%@"
    print(f"--- {name} (mean rgb {rgb.mean(axis=0).round(3)}) ---")
    for row in im[::2]:
        print("".join(ramp[min(9, int(v * 10))] for v in row))


if __name__ == "__main__":
    UNIT_XY = None  # placeholder replaced below
    verts, quads = build_grid()

    def load(name):
        img = np.asarray(Image.open(os.path.join(ASSETS, name)).convert("RGB"),
                         dtype=np.float64)
        return img

    tex = np.zeros((3, N_VERT, 3))
    # star 0: 193 Å (gold)
    tex[0] = sample_texture(load("sun_0193.jpg"), verts) * BRIGHTEN["193"]
    # star 1: 304 Å (red)
    tex[1] = sample_texture(load("sun_0304.jpg"), verts) * BRIGHTEN["304"]
    # star 2: HMI white-light intensitygram -> blue star. The SDO colour table
    # for this image is warm (nearly zero blue), so tinting RGB yields yellow:
    # use the LUMINANCE (real granulation/sunspot structure) colour-graded blue.
    img_hmi = load("sun_HMIIC.jpg")
    lum = img_hmi @ np.array([0.299, 0.587, 0.114]) / 255.0
    px = (verts[:, 0] * 0.5 + 0.5) * (lum.shape[1] - 1)
    py = (verts[:, 1] * 0.5 + 0.5) * (lum.shape[0] - 1)
    lumv = map_coordinates(lum, [py, px], order=1, mode="nearest")
    tex[2] = np.clip((lumv[:, None] ** 0.85) * np.array([0.50, 0.72, 1.25]), 0.0, 1.0)
    tex = np.clip(tex, 0.0, 1.0)

    UNIT_XY = verts[:, :2]
    ascii_preview("193", tex[0])
    ascii_preview("304", tex[1])
    ascii_preview("HMIIC->blue", tex[2])

    # arrow colours: mean vertex colour, brightened
    arrow = np.clip(tex.mean(axis=1) * 1.35 + 0.06, 0.0, 1.0)
    print("arrow colors:", arrow.round(3))

    np.savez_compressed(
        os.path.join(DATA, "mesh_static.npz"),
        unit=verts, quads=quads, tex_colors=tex.astype(np.float32),
        arrow_colors=arrow.astype(np.float32),
    )
    print("saved", os.path.join(DATA, "mesh_static.npz"))
