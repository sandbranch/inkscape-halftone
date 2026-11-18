# Copyright 2026 David
# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the halftoning kernels in dither.py."""

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dither

RNG = np.random.default_rng(7)


def classic_push(dark, kernel):
    """Textbook error diffusion: each pixel pushes its error forward."""
    taps, divisor = dither.KERNELS[kernel]
    value = np.array(dark, float)
    h, w = value.shape
    out = np.zeros((h, w), bool)
    for y in range(h):
        for x in range(w):
            on = value[y, x] >= 0.5
            out[y, x] = on
            e = value[y, x] - on
            for dy, dx, wt in taps:
                if y + dy < h and 0 <= x + dx < w:
                    value[y + dy, x + dx] += e * wt / divisor
    return out


@pytest.mark.parametrize("kernel", list(dither.KERNELS))
@pytest.mark.parametrize("shape", [(1, 9), (9, 1), (13, 29), (31, 12)])
def test_wavefront_equals_reference(kernel, shape):
    dark = RNG.random(shape)
    assert (
        dither.error_diffusion(dark, kernel)
        == dither.error_diffusion_reference(dark, kernel)
    ).all()


@pytest.mark.parametrize("kernel", list(dither.KERNELS))
def test_reference_is_textbook_error_diffusion(kernel):
    # the same up to the order floating point errors are added in
    dark = RNG.random((40, 50))
    differ = dither.error_diffusion_reference(dark, kernel) != classic_push(
        dark, kernel
    )
    assert differ.mean() < 0.002


@pytest.mark.parametrize(
    "kernel", ["floyd-steinberg", "jarvis", "stucki", "sierra", "sierra-lite"]
)
@pytest.mark.parametrize("level", [0.1, 0.37, 0.5, 0.82])
def test_diffusion_keeps_the_tone(kernel, level):
    # error pushed past the edges is lost, so not too small a picture
    dots = dither.error_diffusion(np.full((256, 256), level), kernel)
    assert dots.mean() == pytest.approx(level, abs=0.01)


def test_atkinson_drops_a_quarter_of_the_error():
    # so extremes clip: light greys go white, dark greys black
    assert dither.error_diffusion(np.full((64, 64), 0.1), "atkinson").mean() < 0.1


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_bayer(n):
    t = dither.bayer(n)
    assert sorted((t * n * n - 0.5).round().astype(int).ravel()) == list(range(n * n))
    for level in (0.25, 0.5, 0.75):
        assert dither.ordered(
            np.full((n * 4, n * 4), level), n
        ).mean() == pytest.approx(level)


@pytest.mark.parametrize("spot", list(dither.SPOTS))
@pytest.mark.parametrize("level", [0.05, 0.3, 0.5, 0.9])
def test_screens_are_tone_exact(spot, level):
    dots = dither.screen(np.full((300, 300), level), 0.05, 1.0, 30, spot)
    assert dots.mean() == pytest.approx(level, abs=0.01)


def test_screen_dots_are_clustered():
    # 15 % in 20 px cells: solid dots of about 60 pixels, so most dot
    # pixels are surrounded by dot; error diffusion scatters single pixels
    def solid(dots):
        inner = dots[1:-1, 1:-1]
        around = dots[:-2, 1:-1] & dots[2:, 1:-1] & dots[1:-1, :-2] & dots[1:-1, 2:]
        return (inner & around).sum() / inner.sum()

    flat = np.full((200, 200), 0.15)
    assert solid(dither.screen(flat, 0.05, 1.0, 0, "round")) > 0.6
    assert solid(dither.error_diffusion(flat)) < 0.05


def test_dot_radius_covers_the_cell():
    for level in (0.0, 0.2, np.pi / 4, 0.9, 1.0):
        r = dither.dot_radius(np.array([level]))
        assert dither._circle_cover(r)[0] == pytest.approx(level, abs=1e-4)


def test_grid_points_cover_the_rectangle():
    x, y = dither.grid_points(10, 5, 1, 0)
    assert len(x) == 50
    x, y = dither.grid_points(10, 5, 1, 45)
    assert 45 <= len(x) <= 55
    assert x.min() >= 0 and x.max() <= 10 and y.min() >= 0 and y.max() <= 5


def test_adjust():
    d = np.linspace(0, 1, 11)
    assert dither.adjust(d) == pytest.approx(d)
    assert dither.adjust(d, invert=True) == pytest.approx(1 - d)
    assert (dither.adjust(d, gamma=2)[1:-1] < d[1:-1]).all()  # lighter
    assert dither.adjust(d, brightness=10)[5] == pytest.approx(0.4)
    assert dither.adjust(d, contrast=-100) == pytest.approx(0.5)
    assert dither.adjust(d, contrast=50)[[0, 5, 10]] == pytest.approx([0, 0.5, 1])


def test_darkness_modes():
    assert dither.darkness(Image.new("L", (2, 2), 255)) == pytest.approx(0)
    assert dither.darkness(Image.new("RGB", (2, 2), (0, 0, 0))) == pytest.approx(1)
    # transparent is paper
    assert dither.darkness(Image.new("RGBA", (2, 2), (0, 0, 0, 0))) == pytest.approx(0)
    sixteen = Image.fromarray(np.full((2, 2), 16384, np.uint16))
    assert dither.darkness(sixteen) == pytest.approx(1 - 16384 / 65535)
