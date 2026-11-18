# Copyright 2026 David
# SPDX-License-Identifier: GPL-2.0-or-later
"""
Tests for halftone.py, run through inkex like Inkscape runs it. The test
document has 2 mm per user unit, so unit mistakes show. Needs inkex 1.4
(see the GitHub workflow); the vector-art test also needs Inkscape.
"""

import base64
import io
import os
import re
import shutil
import sys

import numpy as np
import pytest
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import inkex

from halftone import ATTR_FOR, Halftone

SVG = """<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
 width="200mm" height="100mm" viewBox="0 0 100 50">{body}</svg>"""


def png(size=(200, 100), value=128):
    out = io.BytesIO()
    Image.fromarray(np.full(size[::-1], value, np.uint8)).save(out, "PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def run(tmp_path, body, *args):
    path = tmp_path / "in.svg"
    path.write_text(SVG.format(body=body))
    out = io.BytesIO()
    Halftone().run([*args, str(path)], output=out)
    return inkex.load_svg(io.BytesIO(out.getvalue())).getroot()


@pytest.fixture(autouse=True)
def document_path(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCUMENT_PATH", str(tmp_path / "in.svg"))


def results(svg):
    return svg.xpath(f"//*[@{ATTR_FOR}]")


def picture(node):
    data = base64.b64decode(node.get("xlink:href").split(",", 1)[1])
    return Image.open(io.BytesIO(data))


# 40 x 20 user units = 80 x 40 mm
IMAGE = '<image id="im" x="5" y="5" width="40" height="20" xlink:href="{}"/>'


def test_raster_one_pixel_per_laser_line(tmp_path):
    svg = run(tmp_path, IMAGE.format(png()), "--method=floyd-steinberg", "--dpi=254")
    (node,) = results(svg)
    out = picture(node)
    assert out.mode == "1"
    assert out.size == (800, 400)  # 80 mm and 40 mm at 0.1 mm
    assert [float(node.get(a)) for a in ("x", "y", "width", "height")] == [5, 5, 40, 20]
    assert node.get(ATTR_FOR) == "im"
    dots = 1 - np.asarray(out, float)
    assert dots.mean() == pytest.approx(1 - 128 / 255, abs=0.02)


@pytest.mark.parametrize("method", ["ordered", "threshold", "screen", "atkinson"])
def test_raster_methods(tmp_path, method):
    svg = run(tmp_path, IMAGE.format(png(value=64)), f"--method={method}", "--dpi=100")
    out = picture(results(svg)[0])
    assert out.size == (315, 157)


def test_preserve_aspect_ratio(tmp_path):
    # a 200 x 100 picture in a 40 x 40 box: drawn 40 x 20, centred
    body = f'<image id="im" width="40" height="40" xlink:href="{png()}"/>'
    node = results(run(tmp_path, body, "--dpi=254"))[0]
    assert [float(node.get(a)) for a in ("x", "y", "width", "height")] == [
        0,
        10,
        40,
        20,
    ]


def test_original_hidden_kept_or_deleted(tmp_path):
    svg = run(tmp_path, IMAGE.format(png()), "--dpi=50")
    assert svg.getElementById("im").style.get("display") == "none"
    svg = run(tmp_path, IMAGE.format(png()), "--dpi=50", "--original=keep")
    assert svg.getElementById("im").style.get("display") is None
    svg = run(tmp_path, IMAGE.format(png()), "--dpi=50", "--original=delete")
    assert svg.getElementById("im") is None
    assert len(svg.xpath("//svg:image")) == 1


def test_again_on_the_result_replaces_it(tmp_path):
    svg = run(tmp_path, IMAGE.format(png()), "--dpi=50")
    path = tmp_path / "again.svg"
    path.write_bytes(svg.tostring())
    out = io.BytesIO()
    Halftone().run(
        [f"--id={results(svg)[0].get('id')}", "--method=dots", str(path)], output=out
    )
    again = inkex.load_svg(io.BytesIO(out.getvalue())).getroot()
    (node,) = results(again)
    assert node.get(ATTR_FOR) == "im"
    assert node.tag_name == "g"


def test_transform_and_crop_follow(tmp_path):
    body = (
        '<defs><clipPath id="crop"><rect x="10" y="10" width="10" height="10"/></clipPath></defs>'
        '<image id="im" x="5" y="5" width="40" height="20" transform="rotate(10)" '
        f'clip-path="url(#crop)" xlink:href="{png()}"/>'
    )
    node = results(run(tmp_path, body, "--dpi=50"))[0]
    assert node.get("clip-path") == "url(#crop)"
    assert node.transform == inkex.Transform("rotate(10)")


def circles(svg):
    d = results(svg)[0].findone(".//svg:path").get("d")
    return [float(r) for r in re.findall(r"a([\d.]+),", d)[::2]]


def test_dots_cover_the_image_darkness(tmp_path):
    # 30 % dark, 2 mm cells = 1 user unit: about 800 cells
    svg = run(
        tmp_path,
        IMAGE.format(png(value=round(255 * 0.7))),
        "--method=dots",
        "--cell=2",
        "--angle=0",
    )
    radii = np.array(circles(svg))
    assert len(radii) == 800
    assert (np.pi * radii**2).sum() / (40 * 20) == pytest.approx(0.3, abs=0.01)


def test_smallest_dot_left_out(tmp_path):
    # 2 % dark: dots of 0.32 mm across, kept at 0.3 mm smallest, not at 0.4
    body = IMAGE.format(png(value=250))
    assert circles(run(tmp_path, body, "--method=dots", "--cell=2", "--smallest=0.3"))
    assert (
        circles(run(tmp_path, body, "--method=dots", "--cell=2", "--smallest=0.4"))
        == []
    )


def test_lines_cover_the_image_darkness(tmp_path):
    svg = run(
        tmp_path,
        IMAGE.format(png(value=round(255 * 0.6))),
        "--method=lines",
        "--cell=2",
        "--angle=0",
    )
    path = inkex.PathElement.new(results(svg)[0].findone(".//svg:path").get("d"))
    # 10 lines across 20 units, each 40 % of 1 unit thick, 40 long
    area = sum(abs(_shoelace(sub)) for sub in path.path.to_superpath())
    assert area / (40 * 20) == pytest.approx(0.4, abs=0.01)


def _shoelace(sub):
    pts = np.array([p[1] for p in sub])
    x, y = pts[:, 0], pts[:, 1]
    return (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() / 2


@pytest.mark.skipif(not shutil.which("inkscape"), reason="needs Inkscape to render")
def test_vector_art_is_rendered_first(tmp_path):
    # a black square of 10 x 10 units (20 mm) in a moved group
    body = '<g transform="translate(30 10)"><rect id="sq" width="10" height="10" fill="#000"/></g>'
    svg = run(tmp_path, body, "--id=sq", "--method=threshold", "--dpi=127")
    (node,) = results(svg)
    assert node.getparent().get("transform") == "translate(30, 10)"
    box = node.bounding_box(node.getparent().composed_transform())
    assert (box.left, box.top, box.width, box.height) == pytest.approx(
        (30, 10, 10, 10), abs=0.01
    )
    out = picture(node)
    assert out.size == (100, 100)  # 20 mm at 127 dpi
    assert 1 - np.asarray(out, float).mean() > 0.98
