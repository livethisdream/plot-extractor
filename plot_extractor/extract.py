"""Top-level pipeline: PDF -> plots -> labeled series."""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf

from .assign import Series, extract_series
from .axes import PlotFrame, find_plots
from .primitives import extract_primitives
from .raster import raster_figures


@dataclass
class PlotResult:
    source: str
    page: int  # 1-based
    index: int  # plot number within the document
    frame: PlotFrame
    series: list[Series]
    diagnostics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)  # optional LLM-normalized fields

    @property
    def slug(self) -> str:
        t = re.sub(r"[^A-Za-z0-9]+", "_", self.frame.title or "untitled").strip("_").lower()
        return f"p{self.page:02d}_{self.index:02d}_{t[:60]}"

    def to_dict(self) -> dict:
        f = self.frame
        return {
            "source": self.source,
            "page": self.page,
            "plot_index": self.index,
            "title": f.title,
            "x_axis": {"label": f.xlabel, "scale": f.xaxis.scale,
                       "ticks": [l.value for l in f.xaxis.labels],
                       "range": [float(f.xaxis.to_value(f.xaxis.lo)), float(f.xaxis.to_value(f.xaxis.hi))]},
            "y_axis": {"label": f.ylabel, "scale": f.yaxis.scale,
                       "ticks": [l.value for l in f.yaxis.labels],
                       "range": sorted([float(f.yaxis.to_value(f.yaxis.lo)), float(f.yaxis.to_value(f.yaxis.hi))])},
            "bbox_pt": [round(v, 2) for v in f.rect],
            "calibration_residual": {"x": round(f.xaxis.residual, 5), "y": round(f.yaxis.residual, 5)},
            "series": [{
                "label": s.label,
                "confidence": s.confidence,
                "n_points": int(len(s.points)),
                "ambiguous_with": s.ambiguous_with,
                "note": s.note,
                "points": [[_r(x), _r(y)] for x, y in s.points],
            } for s in self.series],
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        }


def env_flag(name: str, default: bool = True) -> bool:
    """On/off setting from the environment or plot-extractor.env (see config.py)."""
    from .config import flag
    return flag(name, default)


def _dumps(d: dict) -> str:
    """Indented JSON, but with each [x, y] point on one line."""
    txt = json.dumps(d, indent=1)
    return re.sub(r"\[\s*(-?[\d.eE+-]+),\s*(-?[\d.eE+-]+)\s*\]", r"[\1, \2]", txt)


def _r(v: float) -> float:
    return float(f"{v:.5g}")


@dataclass
class DocResult:
    source: str
    plots: list[PlotResult]
    raster_pages: list[dict]  # figures that need the raster path
    error: str = ""


def extract_pdf(path: str | Path) -> DocResult:
    path = Path(path)
    try:
        doc = pymupdf.open(path)
    except Exception as e:  # corrupt / encrypted
        return DocResult(str(path), [], [], f"open failed: {e}")
    plots: list[PlotResult] = []
    raster: list[dict] = []
    k = 0
    for pno in range(doc.page_count):
        page = doc[pno]
        prims = extract_primitives(page)
        frames = find_plots(page, prims.segs, pno)
        for f in frames:
            k += 1
            series, _, diag = extract_series(page, f, prims.segs, prims.boxes)
            plots.append(PlotResult(str(path), pno + 1, k, f, series, diag))
        raster += raster_figures(page, frames)
    return DocResult(str(path), plots, raster)


def write_outputs(res: DocResult, out_dir: str | Path, overlays: bool = True,
                  replots: bool = True) -> list[Path]:
    out_dir = Path(out_dir) / Path(res.source).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    summary = {"source": res.source, "error": res.error, "plots": [], "raster_figures": res.raster_pages}
    if overlays or replots:
        from .overlay import render_overlay, render_replot
    if overlays:
        doc = pymupdf.open(res.source)
    for p in res.plots:
        d = p.to_dict()
        j = out_dir / f"{p.slug}.json"
        j.write_text(_dumps(d))
        written.append(j)
        c = out_dir / f"{p.slug}.csv"
        with c.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["series", "x", "y"])
            for s in p.series:
                for x, y in s.points:
                    w.writerow([s.label, _r(x), _r(y)])
        written.append(c)
        if overlays:
            o = out_dir / f"{p.slug}_overlay.png"
            render_overlay(doc[p.page - 1], p.frame, p.series, str(o))
            written.append(o)
        if replots:
            rp = out_dir / f"{p.slug}_replot.png"
            render_replot(p.frame, p.series, str(rp))
            written.append(rp)
        summary["plots"].append({
            "file": j.name, "page": p.page, "title": p.frame.title,
            "series": {s.label: s.confidence for s in p.series},
            "min_confidence": min([s.confidence for s in p.series], default=0.0),
        })
    sm = out_dir / "summary.json"
    sm.write_text(json.dumps(summary, indent=1))
    written.append(sm)
    return written


def interpolate(series: Series, x) -> np.ndarray:
    """Convenience: y at given x (linear interpolation over the series points)."""
    P = series.points
    return np.interp(np.asarray(x, float), P[:, 0], P[:, 1])
