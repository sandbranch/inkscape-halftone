# Halftone for Inkscape

Halftone and dither pictures in Inkscape, for laser engraving, pen
plotters and print effects: **Extensions > Images > Halftone...**

Select images, or other objects (Inkscape renders them first), or
nothing for every image in the drawing. The result is drawn over each
one at the same size, cropped the same way, and the original is hidden
(or kept, or deleted). Run it again on the result, or the original, to
replace it with other settings.

## Methods

Raster: a black and white picture with **one pixel per laser line**. At
254 dpi (the default) a 200 mm wide image becomes exactly 2000 pixels,
one per 0.1 mm line, so the laser software has nothing left to resample.

| Method | Character |
|---|---|
| Error diffusion: Floyd-Steinberg, Jarvis-Judice-Ninke, Stucki, Sierra, Sierra Lite | Most detail; scattered single dots. The larger kernels (Jarvis, Stucki, Sierra) give fewer worm patterns |
| Error diffusion: Atkinson | Passes on only 3/4 of the error: crisper, lighter shadows and highlights clip |
| Ordered (Bayer 2, 4, 8, 16) | Regular cross-hatch texture; burns evenly |
| Threshold | Plain black and white at 50 % |
| Halftone screen | Clustered dots on a rotated grid: round, diamond, square or line shape |

Vector: one path, for vector engraving, cutting or plotting.

| Method | Character |
|---|---|
| Halftone dots | A dot per grid cell, its area the darkness of the image there |
| Halftone lines | Parallel lines, as thick as the image is dark |

Both leave out marks smaller than the smallest the laser or pen makes
(0.1 mm by default), and are cut at the picture's edge.

Every method keeps the tone: a 30 % grey becomes 30 % ink, on screens
of any dot shape too (tested). Error diffusion loses a little at the
picture's edges, where error falls outside it.

Tone: brightness, contrast, gamma (above 1 lightens the middle tones,
which lasers tend to burn too dark) and invert (light marks on a dark
material, such as slate or anodised aluminium).

## Install

    dev/install.sh            # symlinks into your Inkscape extensions folder
    dev/install.sh --uninstall

Or copy `halftone.inx`, `halftone.py` and `dither.py` into your Inkscape
extensions folder, then restart Inkscape. Needs Python with numpy and
Pillow.

From the command line, extensions get no selection, so every image is
done:
`inkscape in.svg --actions="org.sandbranch.filter.halftone.noprefs;export-filename:out.svg;export-do"`

## How error diffusion is fast in Python

Error diffusion is sequential: each pixel needs the errors of the ones
before it. But a pixel only needs pixels to its left and in rows above,
so every pixel on the same line x + k*y (k = 2 for Floyd-Steinberg, 3
for the larger kernels) can be done at once with numpy. A 2000 x 2000
picture takes 0.3 s (Floyd-Steinberg) to 0.8 s (Jarvis). The plain
pixel-by-pixel version is kept in `dither.py` as the reference; the
tests check that both give identical results for every kernel.
Serpentine scanning (alternate rows right to left) is not offered: it
cannot be split up this way.

## Tests

    python3 -m pytest tests/

`tests/test_dither.py` checks the kernels: wavefront against reference
against textbook error diffusion, tone kept by every method and screen,
Bayer matrices, dot sizes that cover the right area, tone adjustments.
`tests/test_halftone.py` runs the extension through inkex in a document
with 2 mm per user unit: raster size per laser line, placement with
`preserveAspectRatio`, transforms and crops, the original hidden, kept
or deleted, rerunning on a result, dot and line coverage, smallest
marks, and (when Inkscape is installed) rendering a vector object
first. They need inkex 1.4 (`pip install --no-deps inkex==1.4.0` plus
lxml, numpy, pillow, tinycss2, cssselect, pyparsing, packaging), as in
the GitHub workflow.

## Credits

The idea, and the choice of methods, come from Abhishek Sehgal's
[Inkscape extensions for halftone filters](https://github.com/abhishek-sehgal954/Inkscape_extensions_for_halftone_filters)
(Google Summer of Code 2017, Portland State University, mentor Bart
Massey), written for Inkscape 0.92 and Python 2. This is a new
implementation; none of that code is used.

Error diffusion kernels: Floyd and Steinberg (1976), Jarvis, Judice and
Ninke (1976), Stucki (1981), Frankie Sierra, Bill Atkinson.

## Licence

GPL-2.0-or-later, see `COPYING`.
