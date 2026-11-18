# Copyright (C) 2026 David
# SPDX-License-Identifier: GPL-2.0-or-later
"""
Halftoning kernels on numpy arrays of darkness: 0 is paper (no burn),
1 is full ink (full burn). Every function returns a boolean array, True
where a dot is made.

Error diffusion is sequential by nature, so it is computed along
wavefronts: a pixel only needs the pixels to its left and in the rows
above, so all pixels with the same x + k*y (k chosen per kernel) can be
done at once. `error_diffusion_reference` is the plain pixel-by-pixel
version it is tested against; both add the errors in the same order and
give identical results.
"""

import numpy as np

# (dy, dx, weight) of the pixels a pixel's error goes to, and the divisor
KERNELS = {
    "floyd-steinberg": (((0, 1, 7), (1, -1, 3), (1, 0, 5), (1, 1, 1)), 16),
    "jarvis": (
        (
            (0, 1, 7), (0, 2, 5),
            (1, -2, 3), (1, -1, 5), (1, 0, 7), (1, 1, 5), (1, 2, 3),
            (2, -2, 1), (2, -1, 3), (2, 0, 5), (2, 1, 3), (2, 2, 1),
        ),
        48,
    ),
    "stucki": (
        (
            (0, 1, 8), (0, 2, 4),
            (1, -2, 2), (1, -1, 4), (1, 0, 8), (1, 1, 4), (1, 2, 2),
            (2, -2, 1), (2, -1, 2), (2, 0, 4), (2, 1, 2), (2, 2, 1),
        ),
        42,
    ),
    "sierra": (
        (
            (0, 1, 5), (0, 2, 3),
            (1, -2, 2), (1, -1, 4), (1, 0, 5), (1, 1, 4), (1, 2, 2),
            (2, -1, 2), (2, 0, 3), (2, 1, 2),
        ),
        32,
    ),
    "sierra-lite": (((0, 1, 2), (1, -1, 1), (1, 0, 1)), 4),
    # Atkinson passes on only 6/8 of the error: lighter, crisper
    "atkinson": (
        ((0, 1, 1), (0, 2, 1), (1, -1, 1), (1, 0, 1), (1, 1, 1), (2, 0, 1)),
        8,
    ),
}  # fmt: skip


def _sources(kernel):
    """The kernel seen from the receiving pixel: (dy, dx, weight) of the
    pixels whose error it takes, in the order they are finished."""
    taps, divisor = KERNELS[kernel]
    return sorted(
        ((dy, dx, w) for dy, dx, w in taps), key=lambda t: (-t[0], -t[1])
    ), divisor


def _slope(kernel):
    """k such that every source pixel of (y, x) has a smaller x + k*y:
    a source dy rows up and dx to the right needs k > dx/dy."""
    taps, _ = KERNELS[kernel]
    return max([1] + [(-dx) // dy + 1 for dy, dx, _ in taps if dy > 0])


def error_diffusion_reference(dark, kernel="floyd-steinberg"):
    """Plain version: each pixel takes the errors of the pixels before
    it, then is set where it is at least half dark."""
    sources, divisor = _sources(kernel)
    h, w = dark.shape
    err = np.zeros((h, w))
    out = np.zeros((h, w), bool)
    for y in range(h):
        for x in range(w):
            total = 0.0
            for dy, dx, wt in sources:
                sy, sx = y - dy, x - dx
                if sy >= 0 and 0 <= sx < w:
                    total += wt * err[sy, sx]
            value = dark[y, x] + total / divisor
            on = value >= 0.5
            out[y, x] = on
            err[y, x] = value - on
    return out


def error_diffusion(dark, kernel="floyd-steinberg"):
    """Same result as error_diffusion_reference, a wavefront at a time."""
    sources, divisor = _sources(kernel)
    k = _slope(kernel)
    h, w = dark.shape
    pad = 2  # the kernels reach at most 2 pixels
    err = np.zeros((h + pad, w + 2 * pad))
    out = np.zeros((h, w), bool)
    dark = np.asarray(dark, float)
    for t in range(w + k * (h - 1)):
        y0 = max(0, -((w - 1 - t) // k))  # smallest y with x = t - k*y < w
        y1 = min(h - 1, t // k)
        ys = np.arange(y0, y1 + 1)
        xs = t - k * ys
        total = np.zeros(len(ys))
        for dy, dx, wt in sources:
            total = total + wt * err[ys - dy + pad, xs - dx + pad]
        value = dark[ys, xs] + total / divisor
        on = value >= 0.5
        out[ys, xs] = on
        err[ys + pad, xs + pad] = value - on
    return out


# -- tone


def darkness(image):
    """A Pillow image as darkness 0..1, transparent parts as paper."""
    from PIL import Image

    if image.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(image, float)
        top = 65535.0 if image.mode.startswith("I;16") else max(float(arr.max()), 1.0)
        return 1 - np.clip(arr / top, 0, 1)
    if "A" in image.getbands() or image.mode == "P" and "transparency" in image.info:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba)
    return 1 - np.asarray(image.convert("L"), float) / 255


def adjust(dark, brightness=0.0, contrast=0.0, gamma=1.0, invert=False):
    """Brightness and contrast in -100..100; gamma above 1 lightens the
    middle tones (lasers tend to burn them too dark); invert for
    engraving light marks on a dark material."""
    light = 1 - np.asarray(dark, float)
    if gamma > 0 and gamma != 1:
        light = light ** (1 / gamma)
    light = light + brightness / 100
    if contrast:
        factor = np.tan(
            (contrast / 100 + 1) * np.pi / 4
        )  # 0 at -100, 1 at 0, inf at 100
        light = (light - 0.5) * factor + 0.5
    light = np.clip(light, 0, 1)
    return light if invert else 1 - light


# -- dispersed dots


def bayer(n):
    """The n x n Bayer matrix (n a power of 2), as thresholds in (0, 1)."""
    m = np.zeros((1, 1), int)
    while len(m) < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / m.size


def ordered(dark, n=8):
    h, w = dark.shape
    t = bayer(n)
    return dark > np.tile(t, (h // n + 1, w // n + 1))[:h, :w]


def threshold(dark, level=0.5):
    return dark >= level


# -- clustered dots (amplitude modulated screens)

SPOTS = {
    "round": lambda u, v: (np.cos(2 * np.pi * u) + np.cos(2 * np.pi * v)) / 2,
    "diamond": lambda u, v: -(np.abs(u - np.round(u)) + np.abs(v - np.round(v))),
    "square": lambda u, v: (
        -np.maximum(np.abs(u - np.round(u)), np.abs(v - np.round(v)))
    ),
    "line": lambda u, v: -np.abs(v - np.round(v)),
}


def _spot_cdf(spot, samples=256):
    """Sorted spot values over one cell: the share of the cell above a
    value. Ranking by it makes the screen tone-exact: a darkness d
    covers the share d of each cell, whatever the spot's shape."""
    g = (np.arange(samples) + 0.5) / samples
    u, v = np.meshgrid(g, g)
    return np.sort(SPOTS[spot](u, v).ravel())


def screen(dark, pixel_mm, frequency, angle=45.0, spot="round"):
    """Clustered-dot screen on a raster: `frequency` cells per mm at
    `angle` degrees; `pixel_mm` is the size of a pixel."""
    h, w = dark.shape
    y, x = np.mgrid[0:h, 0:w]
    x = (x + 0.5) * pixel_mm * frequency
    y = (y + 0.5) * pixel_mm * frequency
    a = np.radians(angle)
    u = x * np.cos(a) + y * np.sin(a)
    v = -x * np.sin(a) + y * np.cos(a)
    values = SPOTS[spot](u, v)
    ranks = _spot_cdf(spot)
    share_above = 1 - np.searchsorted(ranks, values, side="left") / ranks.size
    return dark > share_above - 0.5 / ranks.size


# -- vector shapes


def _circle_cover(r):
    """Share of a unit cell covered by circles of radius r on the grid."""
    r = np.asarray(r, float)
    h = 0.5
    out = np.pi * r**2
    over = r > h
    ro = r[over]
    segment = ro**2 * np.arccos(h / ro) - h * np.sqrt(ro**2 - h**2)
    out[over] -= 4 * segment
    return np.minimum(out, 1.0)


_R = np.linspace(0, np.sqrt(0.5), 2001)
_COVER = _circle_cover(_R)


def dot_radius(dark):
    """Radius (in cells) of the dot that covers the share `dark` of its
    cell, overlapping neighbours past 78.5 %."""
    return np.interp(np.clip(dark, 0, 1), _COVER, _R)


def grid_points(width, height, cell, angle):
    """Centres of a grid of `cell` spacing at `angle` degrees covering
    the rectangle (0, 0, width, height); arrays x, y and the grid's (i, j)."""
    a = np.radians(angle)
    c, s = np.cos(a), np.sin(a)
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    u = corners @ np.array([c, s]) / cell
    v = corners @ np.array([-s, c]) / cell
    i = np.arange(np.floor(u.min()), np.ceil(u.max()) + 1)
    j = np.arange(np.floor(v.min()), np.ceil(v.max()) + 1)
    ii, jj = np.meshgrid(i + 0.5, j + 0.5)
    x = (ii * c - jj * s) * cell
    y = (ii * s + jj * c) * cell
    inside = (x >= 0) & (x <= width) & (y >= 0) & (y <= height)
    return x[inside], y[inside]


def sample(dark, x, y, width, height):
    """Darkness at points (x, y) of a raster drawn over (0, 0, width,
    height), bilinear."""
    h, w = dark.shape
    fx = np.clip(x / width * w - 0.5, 0, w - 1)
    fy = np.clip(y / height * h - 0.5, 0, h - 1)
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    tx, ty = fx - x0, fy - y0
    top = dark[y0, x0] * (1 - tx) + dark[y0, x1] * tx
    bottom = dark[y1, x0] * (1 - tx) + dark[y1, x1] * tx
    return top * (1 - ty) + bottom * ty
