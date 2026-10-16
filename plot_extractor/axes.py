"""Find plots on a page from their numeric tick labels and calibrate the axes.

A plot is located by pairing a *column* of numeric labels (y axis) with a
*row* of numeric labels (x axis) that sits just below and to the right of it.
Label positions are snapped onto nearby gridlines/ticks, then a linear or
log10 mapping from PDF coordinates to data values is fitted.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pymupdf

from .primitives import Seg
from .text import PageText

_NUM_RE = re.compile(r"^([+\-−–]?)(\d+(?:\.\d*)?|\.\d+)([eE][+\-]?\d+)?([kMGmunpµ]?)$")
_SI = {"": 1.0, "k": 1e3, "M": 1e6, "G": 1e9, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12}


def parse_number(s: str) -> float | None:
    s = s.strip().replace(",", "")
    m = _NUM_RE.match(s)
    if not m:
        return None
    sign, mant, exp, si = m.groups()
    try:
        v = float(mant + (exp or ""))
    except ValueError:
        return None
    if sign in ("-", "−", "–"):
        v = -v
    return v * _SI[si]


@dataclass
class Label:
    value: float
    text: str
    rect: pymupdf.Rect

    @property
    def cx(self):
        return (self.rect.x0 + self.rect.x1) / 2

    @property
    def cy(self):
        return (self.rect.y0 + self.rect.y1) / 2


@dataclass
class Axis:
    orient: str  # "x" or "y"
    labels: list[Label]
    positions: list[float]  # snapped PDF coordinate of each tick
    scale: str = "linear"  # linear | log
    a: float = 0.0  # value(or log10 value) = a * pos + b
    b: float = 0.0
    residual: float = 0.0

    def to_value(self, pos):
        pos = np.asarray(pos, float)
        v = self.a * pos + self.b
        return np.power(10.0, v) if self.scale == "log" else v

    def to_pos(self, val):
        val = np.asarray(val, float)
        if self.scale == "log":
            val = np.log10(val)
        return (val - self.b) / self.a

    @property
    def lo(self):
        return min(self.positions)

    @property
    def hi(self):
        return max(self.positions)

    @property
    def spacing(self):
        p = sorted(self.positions)
        return float(np.median(np.diff(p))) if len(p) > 1 else 10.0


@dataclass
class PlotFrame:
    page: int
    xaxis: Axis
    yaxis: Axis
    rect: pymupdf.Rect = field(default=None)
    title: str = ""
    figure: str = ""
    ocr: bool = False  # tick/legend labels came from OCR
    xlabel: str = ""
    ylabel: str = ""

    def to_data(self, pts):
        P = np.asarray(pts, float).reshape(-1, 2)
        return np.c_[self.xaxis.to_value(P[:, 0]), self.yaxis.to_value(P[:, 1])]

    def to_pdf(self, data):
        D = np.asarray(data, float).reshape(-1, 2)
        return np.c_[self.xaxis.to_pos(D[:, 0]), self.yaxis.to_pos(D[:, 1])]


def numeric_labels(text: PageText) -> list[Label]:
    labels = []
    for w in text.words:
        if abs(w.dir[1]) > 0.1:
            continue
        v = parse_number(w.text)
        if v is not None:
            labels.append(Label(v, w.text, w.rect))
    return labels


def _skips_one(run, it) -> bool:
    """Does `it` continue `run`'s value progression with exactly one value missing?"""
    vals = [getattr(l, "value", None) for l in run[-2:]] + [getattr(it, "value", None)]
    if None in vals:
        return False
    a, b, c = vals
    step = b - a
    if step != 0 and abs((c - b) - 2 * step) <= 1e-6 * max(1.0, abs(step)):
        return True
    if a > 0 and b > 0 and c > 0:  # log axis: constant ratio
        return abs(c / b - (b / a) ** 2) <= 1e-6 * (b / a) ** 2
    return False


def _split_runs(items, key, tol=0.25):
    """Split a position-sorted list into evenly spaced runs (tick labels are evenly spaced)."""
    if len(items) < 2:
        return [items]
    runs, cur, g0 = [], [items[0]], None
    for it in items[1:]:
        g = key(it) - key(cur[-1])
        if g <= 0.5:
            runs.append(cur)
            cur, g0 = [it], None
            continue
        if g0 is None:
            if g > 200:  # too far apart to be neighbors on one axis
                runs.append(cur)
                cur = [it]
                continue
            g0 = g
            cur.append(it)
        elif abs(g - g0) <= tol * g0 or (len(cur) >= 3 and abs(g - 2 * g0) <= tol * g0
                                          and _skips_one(cur, it)):
            # (a gap of two spacings whose value also skips one step = one unreadable label)
            cur.append(it)
        elif len(cur) == 2:
            # the first label was a stray (e.g. a number in a heading): restart from
            # the second one, which still pairs with this label
            runs.append(cur[:1])
            cur, g0 = [cur[1], it], g
        else:
            runs.append(cur)
            cur, g0 = [it], None
    runs.append(cur)
    return runs


def _fit(positions, values):
    """Return (scale, a, b, normalized_residual) for the better of lin/log fits."""
    P = np.asarray(positions, float)
    V = np.asarray(values, float)
    best = None
    cands = [("linear", V)]
    if np.all(V > 0):
        cands.append(("log", np.log10(V)))
    for scale, Y in cands:
        if np.ptp(Y) == 0 or np.ptp(P) == 0:
            continue
        a, b = np.polyfit(P, Y, 1)
        res = float(np.max(np.abs(a * P + b - Y)) / np.ptp(Y))
        if best is None or res < best[3]:
            best = (scale, float(a), float(b), res)
    return best


def _drop_outlier(labels, positions, bad=0.03):
    """If one misread label spoils an otherwise clean axis fit, drop it."""
    f = _fit(positions, [l.value for l in labels])
    if (f is not None and f[3] <= bad) or len(labels) < 4:
        return labels, positions
    best = None
    for k in range(len(labels)):
        ls = labels[:k] + labels[k + 1:]
        ps = positions[:k] + positions[k + 1:]
        fk = _fit(ps, [l.value for l in ls])
        if fk is not None and fk[3] <= bad and (best is None or fk[3] < best[0]):
            best = (fk[3], ls, ps)
    return (best[1], best[2]) if best else (labels, positions)


def _monotone(values):
    d = np.diff(values)
    return bool(np.all(d > 0) or np.all(d < 0))


def _nearly_monotone(values):
    """Monotone, or monotone after removing one (misread) value."""
    if _monotone(values):
        return True
    return len(values) >= 4 and any(_monotone(values[:k] + values[k + 1:]) for k in range(len(values)))


def _axis_groups(labels: list[Label]):
    """Candidate y-axis columns and x-axis rows."""
    cols, rows = [], []
    # columns: share right edge
    used = sorted(labels, key=lambda l: l.rect.x1)
    clusters, cur = [], []
    for l in used:
        if cur and l.rect.x1 - cur[-1].rect.x1 > 1.5:
            clusters.append(cur)
            cur = []
        cur.append(l)
    clusters.append(cur)
    for c in clusters:
        c = sorted(c, key=lambda l: l.cy)
        for run in _split_runs(c, lambda l: l.cy):
            if len(run) >= 3 and _nearly_monotone([l.value for l in run]):
                cols.append(run)
    # rows: share vertical center
    used = sorted(labels, key=lambda l: l.cy)
    clusters, cur = [], []
    for l in used:
        if cur and l.cy - cur[-1].cy > 1.5:
            clusters.append(cur)
            cur = []
        cur.append(l)
    clusters.append(cur)
    for c in clusters:
        c = sorted(c, key=lambda l: l.cx)
        for run in _split_runs(c, lambda l: l.cx):
            if len(run) >= 3 and _nearly_monotone([l.value for l in run]):
                rows.append(run)
    return cols, rows


def _snap(labels, orient, segs, span):
    """Snap each label's tick position onto gridlines/ticks running across the plot."""
    lo, hi = span
    raw = [l.cy if orient == "y" else l.cx for l in labels]
    snapped: list[float | None] = []
    for r in raw:
        cands = []
        for s in segs:
            if s.length < 0.2 and s.src != "dot":
                continue
            if orient == "y":
                if abs(s.p0[1] - s.p1[1]) > 0.05 or abs(s.p0[1] - r) > 3.0:
                    continue
                if max(s.p0[0], s.p1[0]) < lo - 6 or min(s.p0[0], s.p1[0]) > hi + 6:
                    continue
                cands.append(s.p0[1])
            else:
                if abs(s.p0[0] - s.p1[0]) > 0.05 or abs(s.p0[0] - r) > 3.0:
                    continue
                if max(s.p0[1], s.p1[1]) < lo - 6 or min(s.p0[1], s.p1[1]) > hi + 6:
                    continue
                cands.append(s.p0[0])
        if cands:
            # most common coordinate near the label
            c = np.round(np.asarray(cands), 1)
            vals, counts = np.unique(c, return_counts=True)
            near = vals[np.argmax(counts - 0.01 * np.abs(vals - r))]
            snapped.append(float(np.median([x for x in cands if abs(x - near) < 0.15])))
        else:
            snapped.append(None)
    offs = [s - r for s, r in zip(snapped, raw) if s is not None]
    off = float(np.median(offs)) if offs else 0.0
    # labels sit at a consistent offset from their tick; reject snaps that disagree
    return [s if (s is not None and abs(s - r - off) < 0.4) else r + off
            for s, r in zip(snapped, raw)]


def find_plots(page: pymupdf.Page, segs: list[Seg], page_no: int = 0,
               text: PageText | None = None) -> list[PlotFrame]:
    text = text or PageText.from_page(page)
    labels = numeric_labels(text)
    cols, rows = _axis_groups(labels)
    pairs = []
    for ci, col in enumerate(cols):
        right = max(l.rect.x1 for l in col)
        bottom = max(l.cy for l in col)
        for ri, row in enumerate(rows):
            first = min(row, key=lambda l: l.cx)
            dy = row[0].cy - bottom
            dx = first.cx - right
            if 0 < dy < 25 and -2 < dx < 25:
                # the x row should not contain the column's own labels
                if set(map(id, col)) & set(map(id, row)):
                    continue
                pairs.append((dy + dx, ci, ri))
    pairs.sort()
    used_c, used_r, frames = set(), set(), []
    for _, ci, ri in pairs:
        if ci in used_c or ri in used_r:
            continue
        col, row = cols[ci], rows[ri]
        xspan = (min(l.cx for l in row), max(l.cx for l in row))
        yspan = (min(l.cy for l in col), max(l.cy for l in col))
        xpos = _snap(row, "x", segs, yspan)
        ypos = _snap(col, "y", segs, xspan)
        row, xpos = _drop_outlier(row, xpos)
        col, ypos = _drop_outlier(col, ypos)
        fx = _fit(xpos, [l.value for l in row])
        fy = _fit(ypos, [l.value for l in col])
        if fx is None or fy is None or fx[3] > 0.03 or fy[3] > 0.03:
            continue
        used_c.add(ci)
        used_r.add(ri)
        xa = Axis("x", row, xpos, *fx)
        ya = Axis("y", col, ypos, *fy)
        rect = pymupdf.Rect(xa.lo, ya.lo, xa.hi, ya.hi)
        frames.append(PlotFrame(page_no, xa, ya, rect))
    for f in frames:
        _attach_text(text, f)
    frames.sort(key=lambda f: (round(f.rect.y0 / 50), f.rect.x0))
    return frames


def _attach_text(text: PageText, f: PlotFrame):
    """Title (or figure caption), x label below the x ticks, y label left of y ticks."""
    lines = text.lines()
    r = f.rect
    ylab_x0 = min(l.rect.x0 for l in f.yaxis.labels)
    xlab_y1 = max(l.rect.y1 for l in f.xaxis.labels)
    horiz = [(rc, t) for rc, dr, t in lines if abs(dr[1]) < 0.1]
    vert = [(rc, t) for rc, dr, t in lines if abs(dr[0]) < 0.1]

    # title: horizontal lines just above the plot, overlapping it horizontally
    above = [(rc, t) for rc, t in horiz
             if r.y0 - 45 < rc.y1 <= r.y0 - 3 and rc.x1 > ylab_x0 - 10 and rc.x0 < r.x1
             and parse_number(t) is None]
    above.sort(key=lambda z: -z[0].y1)
    title_lines = []
    for rc, t in above:
        if not title_lines:
            title_lines.append((rc, t))
        elif title_lines[-1][0].y0 - rc.y1 < rc.height * 0.8:
            title_lines.append((rc, t))
    f.title = " ".join(t for _, t in reversed(title_lines))

    below = [(rc, t) for rc, t in horiz
             if xlab_y1 - 1 < rc.y0 < xlab_y1 + 20 and rc.x1 > r.x0 and rc.x0 < r.x1
             and parse_number(t) is None]
    below.sort(key=lambda z: z[0].y0)
    f.xlabel = below[0][1] if below else ""

    # "Figure N. ..." caption under the plot (common in newer datasheets)
    cap_top = (below[0][0].y1 if below else xlab_y1)
    caps = [(rc, t) for rc, t in horiz
            if cap_top - 1 < rc.y0 < cap_top + 45 and rc.x1 > ylab_x0 - 30 and rc.x0 < r.x1 + 30
            and re.match(r"^(Figure|Fig\.)\s*\d+", t)]
    if caps:
        caps.sort(key=lambda z: z[0].y0)
        rc, t = caps[0]
        parts = [t]
        nxt = [(rc2, t2) for rc2, t2 in horiz if 0 <= rc2.y0 - rc.y1 < 0.8 * rc.height
               and abs(rc2.x0 - rc.x0) < 120 and not re.match(r"^(Figure|Fig\.)", t2)]
        if nxt:
            parts.append(min(nxt, key=lambda z: z[0].y0)[1])
        cap = " ".join(parts)
        m = re.match(r"^((?:Figure|Fig\.)\s*\d+)\.?\s*(.*)$", cap)
        f.figure = m.group(1) if m else ""
        f.title = (m.group(2) if m else cap) or f.title

    left = [(rc, t) for rc, t in vert
            if ylab_x0 - 25 < rc.x1 <= ylab_x0 + 1 and rc.y1 > r.y0 - 5 and rc.y0 < r.y1 + 5]
    left.sort(key=lambda z: -z[0].x1)
    f.ylabel = left[0][1] if left else ""
