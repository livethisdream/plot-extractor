"""Raster fallback (stub).

Older and scanned datasheets embed plots as images. This module currently only
*detects* figure-sized images that weren't covered by a vector plot, so batch
runs can report them; extraction for them is not implemented yet.

Planned approach: render the image region, locate axes with OpenCV line
detection, OCR tick labels (Tesseract) to calibrate, then trace curves by
color/connected components. A local vision model can help label series.
"""
from __future__ import annotations

import pymupdf


def raster_figures(page: pymupdf.Page, frames, min_size: float = 100.0) -> list[dict]:
    found = []
    covered = [f.rect for f in frames]
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"])
        if r.width < min_size or r.height < min_size * 0.6:
            continue
        if any((r & c).get_area() > 0.3 * r.get_area() for c in covered):
            continue
        found.append({"page": page.number + 1, "bbox_pt": [round(v, 1) for v in r],
                      "status": "embedded image - not analyzed (could be a photo or a raster plot)"})
    return found
