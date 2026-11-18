#!/usr/bin/env python3
# Copyright (C) 2026 David
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
"""
Halftone and dither images, for laser engraving and print effects.

Raster methods give a 1-bit picture with one pixel per laser line at the
chosen resolution, placed exactly over the image. Vector methods give
dots or lines of varying size as one path, for vector engraving, cutting
or plotting.
"""

import base64
import io
import os
import re
import sys
import tempfile
from urllib.parse import unquote, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import inkex
import inkex.command
import numpy as np
from inkex.localization import inkex_gettext as _
from PIL import Image

import dither

ATTR_FOR = "data-halftone-for"
DIFFUSION = tuple(dither.KERNELS)
RASTER = (*DIFFUSION, "ordered", "threshold", "screen")
VECTOR = ("dots", "lines")


def drawn_area(image, raster_w, raster_h):
    """(x, y, width, height) of the raster inside the image box, in the
    image's user units, following preserveAspectRatio."""
    x = image.to_dimensionless(image.get("x", "0"))
    y = image.to_dimensionless(image.get("y", "0"))
    width = image.to_dimensionless(image.get("width", str(raster_w)))
    height = image.to_dimensionless(image.get("height", str(raster_h)))
    aspect = (image.get("preserveAspectRatio") or "xMidYMid meet").split()
    if aspect[0] == "none":
        return x, y, width, height
    scale = min(width / raster_w, height / raster_h)
    if len(aspect) > 1 and aspect[1] == "slice":
        scale = max(width / raster_w, height / raster_h)
    dw, dh = raster_w * scale, raster_h * scale
    fx = {"xMin": 0.0, "xMid": 0.5, "xMax": 1.0}.get(aspect[0][:4], 0.5)
    fy = {"YMin": 0.0, "YMid": 0.5, "YMax": 1.0}.get(aspect[0][4:], 0.5)
    return x + (width - dw) * fx, y + (height - dh) * fy, dw, dh


def fmt(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


class Halftone(inkex.EffectExtension):
    """Halftone or dither the selected images"""

    def add_arguments(self, pars):
        pars.add_argument("--tab")
        pars.add_argument(
            "--method", default="floyd-steinberg", help=", ".join(RASTER + VECTOR)
        )
        pars.add_argument(
            "--dpi", type=float, default=254.0, help="raster: pixels per inch"
        )
        pars.add_argument(
            "--cell", type=float, default=0.5, help="screen, dots, lines: mm"
        )
        pars.add_argument("--angle", type=float, default=45.0, help="degrees")
        pars.add_argument("--spot", default="round", help=", ".join(dither.SPOTS))
        pars.add_argument("--bayer", type=int, default=8, help="ordered: 2, 4, 8 or 16")
        pars.add_argument("--smallest", type=float, default=0.1, help="dots, lines: mm")
        pars.add_argument("--brightness", type=float, default=0.0)
        pars.add_argument("--contrast", type=float, default=0.0)
        pars.add_argument("--gamma", type=float, default=1.0)
        pars.add_argument("--invert", type=inkex.Boolean, default=False)
        pars.add_argument("--color", type=inkex.Color, default=inkex.Color("black"))
        pars.add_argument("--original", default="hide", help="hide, keep or delete")

    # -- reading images

    def image_data(self, image):
        href = image.get("xlink:href") or image.get("href") or ""
        if href.startswith("data:"):
            match = re.match(r"data:[^,]*?(;base64)?,(.*)", href, re.DOTALL)
            if not match:
                return None
            payload = match.group(2)
            if match.group(1):
                return base64.b64decode(re.sub(r"\s+", "", payload))
            return unquote(payload).encode("latin-1")
        path = unquote(urlparse(href).path) if href.startswith("file:") else href
        if path and not os.path.isabs(path):
            path = os.path.join(self.svg_path() or "", path)
        absref = image.get("sodipodi:absref")
        for candidate in (path, absref):
            if candidate and os.path.isfile(candidate):
                with open(candidate, "rb") as f:
                    return f.read()
        return None

    def mm_per_unit(self, image):
        """Millimetres per user unit of the image's own coordinates."""
        t = image.composed_transform()
        scale = abs(t.a * t.d - t.b * t.c) ** 0.5
        return scale / self.svg.viewport_to_unit("1mm")  # unittouu ignores the viewBox

    # -- the effect

    def targets(self):
        """(node, is_image) for everything to halftone: images, and
        other objects, which Inkscape renders first."""
        if not self.svg.selection:
            return [(i, True) for i in self.svg.descendants().filter(inkex.Image)]
        found = []
        for node in self.svg.selection.values():
            if node.get(ATTR_FOR):  # an earlier result: redo its source
                node = self.svg.getElementById(node.get(ATTR_FOR))
                if node is None:
                    continue
            if isinstance(node, inkex.Image):
                found.append((node, True))
            elif isinstance(node, inkex.Group) and list(
                node.descendants().filter(inkex.Image)
            ):
                found.extend((i, True) for i in node.descendants().filter(inkex.Image))
            else:
                found.append((node, False))
        seen, out = set(), []
        for node, is_image in found:
            if node.get("id") not in seen:
                seen.add(node.get("id"))
                out.append((node, is_image))
        return out

    def effect(self):
        if self.options.method not in RASTER + VECTOR:
            raise inkex.AbortExtension(
                _("Unknown method: {}").format(self.options.method)
            )
        targets = self.targets()
        if not targets:
            raise inkex.AbortExtension(_("Select the images or objects to halftone."))
        failed = []
        for node, is_image in targets:
            self.remove_results(node)
            if is_image:
                picture = self.open_image(node)
                if picture is None:
                    failed.append(node.get("id"))
                    continue
                area = drawn_area(node, *picture.size)
                self.halftone(
                    node, picture, area, self.mm_per_unit(node), node.transform
                )
            else:
                rendered = self.render(node)
                if rendered is None:
                    failed.append(node.get("id"))
                    continue
                picture, area = rendered
                # the result is placed in the node's parent, in document
                # coordinates
                undo = -node.getparent().composed_transform()
                mm = 1 / self.svg.viewport_to_unit("1mm")
                self.halftone(node, picture, area, mm, undo)
        if failed:
            inkex.errormsg(_("Could not read or render: {}").format(", ".join(failed)))

    def remove_results(self, node):
        for old in self.svg.xpath(f"//*[@{ATTR_FOR}='{node.get('id')}']"):
            for ref in old.xpath(".//@clip-path"):
                clip = (
                    self.svg.getElementById(ref[5:-1])
                    if ref.startswith("url(#")
                    else None
                )
                if clip is not None and (clip.get("id") or "").startswith(
                    "halftone-edge"
                ):
                    clip.getparent().remove(clip)
            old.getparent().remove(old)

    def open_image(self, image):
        data = self.image_data(image)
        if not data:
            return None
        try:
            picture = Image.open(io.BytesIO(data))
            picture.load()
        except (OSError, ValueError):
            return None
        return picture

    def render_dpi(self):
        if self.options.method in RASTER:
            return self.options.dpi
        # about 8 pixels per cell is plenty to sample dots and lines from
        return min(1200.0, max(72.0, 8 * 25.4 / self.options.cell))

    def render(self, node):
        """Render one object (not an image) to a picture with Inkscape;
        returns the picture and its area in document coordinates."""
        box = node.bounding_box(
            node.getparent().composed_transform()
        )  # in document units
        if box is None or box.width <= 0 and box.height <= 0:
            return None
        # room for strokes, which bounding_box leaves out
        pad = max([self.stroke(n) for n in node.descendants()] + [0.0])
        x0, y0, x1, y1 = (
            box.left - pad,
            box.top - pad,
            box.right + pad,
            box.bottom + pad,
        )
        vx, vy = self.svg.get_viewbox()[:2]
        k = (
            self.svg.scale
        )  # Inkscape takes the area in px (1/96 in) from the page corner
        with tempfile.TemporaryDirectory(prefix="halftone-") as folder:
            doc = os.path.join(folder, "in.svg")
            out = os.path.join(folder, "out.png")
            copy = inkex.load_svg(io.BytesIO(self.svg.tostring())).getroot()
            target = copy.getElementById(node.get("id"))
            for each in [target, *target.ancestors()]:
                if (
                    isinstance(each, inkex.BaseElement)
                    and each.style.get("display") == "none"
                ):
                    each.style["display"] = None
            with open(doc, "wb") as f:
                f.write(copy.tostring())
            area = ":".join(
                f"{v:.4f}"
                for v in ((x0 - vx) * k, (y0 - vy) * k, (x1 - vx) * k, (y1 - vy) * k)
            )
            try:
                inkex.command.inkscape(
                    doc,
                    export_id=node.get("id"),
                    export_id_only=True,
                    export_area=area,
                    export_dpi=f"{self.render_dpi():.4f}",
                    export_background="white",
                    export_background_opacity="1",
                    export_type="png",
                    export_filename=out,
                )
            except inkex.command.ProgramRunError:
                return None
            if not os.path.isfile(out):
                return None
            picture = Image.open(out)
            picture.load()
        return picture, (x0, y0, x1 - x0, y1 - y0)

    def stroke(self, node):
        """Half the stroke width of `node`, in document units."""
        style = (
            node.specified_style() if hasattr(node, "specified_style") else node.style
        )
        if style.get("stroke") in (None, "none"):
            return 0.0
        width = self.svg.unittouu(style.get("stroke-width") or "1")
        t = node.composed_transform()
        return width * abs(t.a * t.d - t.b * t.c) ** 0.5 / 2

    def halftone(self, source, picture, area, mm, transform):
        x, y, width, height = area
        opts = self.options
        dark = dither.adjust(
            dither.darkness(picture),
            opts.brightness,
            opts.contrast,
            opts.gamma,
            opts.invert,
        )
        if opts.method in RASTER:
            node = self.raster(dark, x, y, width, height, mm)
        else:
            node = self.vector(dark, x, y, width, height, mm)
        node.set(ATTR_FOR, source.get("id"))
        node.set("id", self.svg.get_unique_id("halftone"))
        node.label = _("Halftone ({})").format(opts.method)
        if transform:
            node.transform = transform
        if isinstance(source, inkex.Image) and source.get("clip-path"):
            node.set("clip-path", source.get("clip-path"))
        source.addnext(node)
        if opts.original == "hide":
            source.style["display"] = "none"
        elif opts.original == "delete":
            node.set(ATTR_FOR, None)
            source.getparent().remove(source)
        elif source.style.get("display") == "none":
            source.style["display"] = None

    def raster(self, dark, x, y, width, height, mm):
        opts = self.options
        # one pixel per laser line: the picture's printed size at the dpi
        cols = max(1, round(width * mm / 25.4 * opts.dpi))
        rows = max(1, round(height * mm / 25.4 * opts.dpi))
        light = Image.fromarray(np.round((1 - dark) * 65535).astype(np.uint16))
        light = light.convert("F").resize((cols, rows), Image.Resampling.LANCZOS)
        grid = np.clip(1 - np.asarray(light) / 65535, 0, 1)
        method = opts.method
        if method in DIFFUSION:
            dots = dither.error_diffusion(grid, method)
        elif method == "ordered":
            n = opts.bayer if opts.bayer in (2, 4, 8, 16) else 8
            dots = dither.ordered(grid, n)
        elif method == "threshold":
            dots = dither.threshold(grid)
        else:
            dots = dither.screen(
                grid, 25.4 / opts.dpi, 1 / opts.cell, opts.angle, opts.spot
            )
        out = Image.fromarray(~dots)  # mode "1": white where no dot
        buffer = io.BytesIO()
        out.save(buffer, "PNG", optimize=True, dpi=(opts.dpi, opts.dpi))
        node = inkex.Image()
        node.set("x", fmt(x))
        node.set("y", fmt(y))
        node.set("width", fmt(width))
        node.set("height", fmt(height))
        node.set("preserveAspectRatio", "none")
        node.set(
            "xlink:href",
            "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
        )
        node.style["image-rendering"] = (
            "optimizeSpeed"  # no smoothing; inkex knows no "pixelated"
        )
        return node

    def vector(self, dark, x, y, width, height, mm):
        opts = self.options
        cell = opts.cell / mm  # in the image's units
        smallest = opts.smallest / mm
        if self.options.method == "dots":
            d = self.dots(dark, width, height, cell, smallest)
        else:
            d = self.lines(dark, width, height, cell, smallest)
        node = inkex.PathElement()
        node.set("d", d or "M 0,0")
        node.style = {"fill": str(opts.color), "stroke": "none"}
        node.transform = inkex.Transform(translate=(x, y))
        # dots at the edge reach out of the picture: cut them at its edge
        clip = self.svg.defs.add(inkex.ClipPath())
        clip.set("id", self.svg.get_unique_id("halftone-edge"))
        clip.add(inkex.Rectangle.new(x, y, width, height))
        inner = inkex.Group()
        inner.set("clip-path", f"url(#{clip.get('id')})")
        inner.add(node)
        group = inkex.Group()
        group.add(inner)
        return group

    def dots(self, dark, width, height, cell, smallest):
        px, py = dither.grid_points(width, height, cell, self.options.angle)
        r = (
            dither.dot_radius(
                dither.sample(self.blurred(dark, width, cell), px, py, width, height)
            )
            * cell
        )
        keep = 2 * r >= smallest
        parts = []
        for cx, cy, rr in zip(px[keep], py[keep], r[keep]):
            a = fmt(rr)
            parts.append(
                f"M{fmt(cx + rr)},{fmt(cy)}a{a},{a} 0 1 0 {fmt(-2 * rr)},0a{a},{a} 0 1 0 {fmt(2 * rr)},0z"
            )
        return "".join(parts)

    def lines(self, dark, width, height, cell, smallest):
        """Lines at `angle`, a cell apart, as thick as the image is dark."""
        a = np.radians(self.options.angle)
        c, s = np.cos(a), np.sin(a)
        corners = np.array([[0, 0], [width, 0], [0, height], [width, height]])
        along = corners @ np.array([c, s])
        across = corners @ np.array([-s, c])
        step = cell / 2
        t = np.arange(along.min(), along.max() + step, step)
        soft = self.blurred(dark, width, cell)
        parts = []
        for k in np.arange(
            np.floor(across.min() / cell), np.ceil(across.max() / cell) + 1
        ):
            v = (k + 0.5) * cell
            lx, ly = t * c - v * s, t * s + v * c
            inside = (lx >= 0) & (lx <= width) & (ly >= 0) & (ly <= height)
            half = dither.sample(soft, lx, ly, width, height) * cell / 2
            on = inside & (2 * half >= smallest)
            # runs of consecutive points with a line
            edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(int), [0]])))
            for start, end in zip(edges[::2], edges[1::2]):
                if end - start < 2:
                    continue
                # keep the ends and the points where the width bends
                h = half[start:end]
                bend = np.abs(h[:-2] - 2 * h[1:-1] + h[2:]) > 0.01 * cell
                i = start + np.flatnonzero(np.concatenate([[True], bend, [True]]))
                nx, ny = -s * half[i], c * half[i]
                top = [f"{fmt(px)},{fmt(py)}" for px, py in zip(lx[i] + nx, ly[i] + ny)]
                bottom = [
                    f"{fmt(px)},{fmt(py)}" for px, py in zip(lx[i] - nx, ly[i] - ny)
                ]
                parts.append("M" + " ".join(top + bottom[::-1]) + "z")
        return "".join(parts)

    @staticmethod
    def blurred(dark, width, cell):
        """The darkness averaged over about a cell, for sampling at dot
        or line centres."""
        h, w = dark.shape
        per_cell = cell / width * w  # raster pixels per cell
        if per_cell <= 2:
            return dark
        factor = per_cell / 2
        small = Image.fromarray(dark.astype(np.float32), "F").resize(
            (max(1, round(w / factor)), max(1, round(h / factor))), Image.Resampling.BOX
        )
        return np.asarray(small, float)


if __name__ == "__main__":
    Halftone().run()
