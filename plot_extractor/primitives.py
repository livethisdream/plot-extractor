"""Turn a PDF page's vector drawings into centerline line segments.

Datasheet plots reach the PDF in several encodings, and this module normalizes
all of them to ``Seg`` objects (a centerline from p0 to p1 with a pen width,
color and optional dash array):

* native stroked paths (lines, rects, Béziers) -- the easy case;
* "stroke-to-outline" fills, where each line segment was exported as the
  filled sweep of a square/round pen (a 4-8 vertex hexagon-ish polygon);
* thin filled rectangles (dotted gridlines, horizontal/vertical curve bits);
* tiny square fills (dots in dotted / dash-dot curves);
* longer filled outlines of whole polylines (centerline by averaging sides).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pymupdf


@dataclass
class Seg:
    p0: tuple[float, float]
    p1: tuple[float, float]
    width: float
    color: tuple[float, float, float]
    dashes: tuple[float, ...] | None = None  # explicit PDF dash array (None = solid/unknown)
    src: str = "stroke"  # stroke | sweep | rect | dot | outline
    wclass: float | None = None  # page-level width cluster (set by extract_primitives)

    @property
    def length(self) -> float:
        return math.dist(self.p0, self.p1)

    @property
    def mid(self) -> tuple[float, float]:
        return ((self.p0[0] + self.p1[0]) / 2, (self.p0[1] + self.p1[1]) / 2)

    @property
    def style(self) -> tuple:
        """Grouping key: segments of one plotted series share this."""
        c = tuple(round(v * 8) / 8 for v in self.color)
        w = self.wclass if self.wclass is not None else round(self.width * 4) / 4
        return (w, c)

    def axis_aligned(self, tol: float = 0.05) -> bool:
        dx = abs(self.p1[0] - self.p0[0])
        dy = abs(self.p1[1] - self.p0[1])
        if dx + dy < 1e-6:
            return False
        return min(dx, dy) <= tol * max(dx, dy) + 0.02


@dataclass
class Box:
    rect: pymupdf.Rect
    filled: bool
    stroked: bool


@dataclass
class PagePrimitives:
    segs: list[Seg] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)


_DASH_RE = re.compile(r"\[([^\]]*)\]")


def _parse_dashes(d) -> tuple[float, ...] | None:
    if not d:
        return None
    m = _DASH_RE.search(str(d))
    if not m:
        return None
    vals = [float(v) for v in m.group(1).split() if v.strip()]
    vals = [v for v in vals if v > 0]
    return tuple(vals) or None


def _color(c) -> tuple[float, float, float]:
    if c is None:
        return (0.0, 0.0, 0.0)
    c = tuple(float(v) for v in c)
    if len(c) == 1:
        return (c[0], c[0], c[0])
    if len(c) == 4:  # CMYK -> RGB
        cy, m, y, k = c
        return ((1 - cy) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    return c[:3]


def _bezier(p0, p1, p2, p3, n=8):
    pts = []
    for i in range(n + 1):
        t = i / n
        a = (1 - t) ** 3
        b = 3 * (1 - t) ** 2 * t
        c = 3 * (1 - t) * t ** 2
        d = t ** 3
        pts.append((a * p0.x + b * p1.x + c * p2.x + d * p3.x,
                    a * p0.y + b * p1.y + c * p2.y + d * p3.y))
    return pts


def _path_points(items) -> tuple[list[tuple[float, float]], bool]:
    """Flatten a drawing's items into one vertex list. Returns (pts, has_curves)."""
    pts: list[tuple[float, float]] = []
    curved = False
    for it in items:
        op = it[0]
        if op == "l":
            a, b = it[1], it[2]
            if not pts or math.dist(pts[-1], (a.x, a.y)) > 1e-3:
                pts.append((a.x, a.y))
            pts.append((b.x, b.y))
        elif op == "c":
            curved = True
            bz = _bezier(*it[1:5])
            if not pts or math.dist(pts[-1], bz[0]) > 1e-3:
                pts.append(bz[0])
            pts.extend(bz[1:])
        elif op == "qu":
            q = it[1]
            for p in (q.ul, q.ur, q.lr, q.ll):
                pts.append((p.x, p.y))
    # drop closing duplicate
    if len(pts) > 2 and math.dist(pts[0], pts[-1]) < 1e-3:
        pts.pop()
    # dedupe consecutive
    out = []
    for p in pts:
        if not out or math.dist(out[-1], p) > 1e-3:
            out.append(p)
    return out, curved


def _farthest_pair_mid(cluster: np.ndarray) -> np.ndarray:
    if len(cluster) == 1:
        return cluster[0]
    best, pair = -1.0, (0, 0)
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            d = float(np.sum((cluster[i] - cluster[j]) ** 2))
            if d > best:
                best, pair = d, (i, j)
    return (cluster[pair[0]] + cluster[pair[1]]) / 2


def _thin_polygon_to_segs(pts, color) -> list[Seg]:
    """Centerline of a filled polygon that is really a stroked line."""
    P = np.asarray(pts, float)
    if len(P) < 3:
        return []
    c = P.mean(axis=0)
    Q = P - c
    # principal axis
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    ax, nx = vt[0], vt[1]
    s = Q @ ax
    t = Q @ nx
    extent = float(s.max() - s.min())
    thick = float(t.max() - t.min())

    if len(P) <= 8:
        if thick > 4.0:
            return []
        if extent < 1.6 * max(thick, 0.3):
            # a dot: zero-length stroke at the centroid
            w = max(thick, extent)
            p = tuple(c)
            return [Seg(p, p, w, color, src="dot")]
        cap = max(thick, 0.3) * 1.05
        lo = P[s <= s.min() + cap]
        hi = P[s >= s.max() - cap]
        a = _farthest_pair_mid(lo)
        b = _farthest_pair_mid(hi)
        # pen width: each end cluster holds 2-3 corners of the (square) pen
        ws = [float(np.ptp(cl, axis=0).max()) for cl in (lo, hi) if len(cl) >= 2]
        w = float(np.median(ws)) if ws else thick
        return [Seg(tuple(a), tuple(b), w, color, src="sweep")]

    # long outline of a polyline: estimate width from area/perimeter
    x, y = P[:, 0], P[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    perim = float(np.sum(np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]]))))
    w = 2 * area / max(perim, 1e-6)
    if w > 3.0 or perim < 12 * max(w, 0.2):
        return []
    # endpoints = extreme vertices along principal axis
    i0, i1 = int(np.argmin(s)), int(np.argmax(s))
    n = len(P)
    side_a = [P[(i0 + k) % n] for k in range((i1 - i0) % n + 1)]
    side_b = [P[(i1 + k) % n] for k in range((i0 - i1) % n + 1)][::-1]

    def resample(side, m=40):
        S = np.asarray(side)
        dl = np.r_[0, np.cumsum(np.hypot(*np.diff(S, axis=0).T))]
        if dl[-1] <= 0:
            return np.repeat(S[:1], m, axis=0)
        u = np.linspace(0, dl[-1], m)
        return np.c_[np.interp(u, dl, S[:, 0]), np.interp(u, dl, S[:, 1])]

    mid = (resample(side_a) + resample(side_b)) / 2
    return [Seg(tuple(mid[k]), tuple(mid[k + 1]), w, color, src="outline")
            for k in range(len(mid) - 1)]


def extract_primitives(page: pymupdf.Page) -> PagePrimitives:
    out = PagePrimitives()
    for g in page.get_drawings():
        items = g["items"]
        typ = g.get("type", "")
        stroke_col = _color(g.get("color"))
        fill_col = _color(g.get("fill"))
        dashes = _parse_dashes(g.get("dashes"))
        width = float(g.get("width") or 0.0)

        # remember sizeable rectangles (legend boxes, frames)
        for it in items:
            if it[0] == "re":
                r = pymupdf.Rect(it[1])
                if r.width > 5 and r.height > 5:
                    out.boxes.append(Box(r, "f" in typ, "s" in typ))

        if "s" in typ:  # native stroke
            for it in items:
                op = it[0]
                if op == "l":
                    out.segs.append(Seg((it[1].x, it[1].y), (it[2].x, it[2].y), width, stroke_col, dashes))
                elif op == "c":
                    bz = _bezier(*it[1:5])
                    for a, b in zip(bz, bz[1:]):
                        out.segs.append(Seg(a, b, width, stroke_col, dashes))
                elif op == "re":
                    r = it[1]
                    cs = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
                    for k in range(4):
                        out.segs.append(Seg(cs[k], cs[(k + 1) % 4], width, stroke_col, dashes))
                elif op == "qu":
                    q = it[1]
                    cs = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)]
                    for k in range(4):
                        out.segs.append(Seg(cs[k], cs[(k + 1) % 4], width, stroke_col, dashes))
            continue

        if typ != "f":
            continue
        # filled shapes
        if len(items) == 1 and items[0][0] == "re":
            r = items[0][1]
            w, h = r.width, r.height
            lo, hi = min(w, h), max(w, h)
            if hi <= 0:
                continue
            if hi < 2.5 and hi / max(lo, 1e-6) < 1.6:
                c = ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
                out.segs.append(Seg(c, c, lo, fill_col, src="dot"))
            elif lo < 3.0:
                if w >= h:
                    y = (r.y0 + r.y1) / 2
                    out.segs.append(Seg((r.x0 + h / 2, y), (r.x1 - h / 2, y), h, fill_col, src="rect"))
                else:
                    x = (r.x0 + r.x1) / 2
                    out.segs.append(Seg((x, r.y0 + w / 2), (x, r.y1 - w / 2), w, fill_col, src="rect"))
            continue
        if any(it[0] == "re" for it in items):
            continue
        pts, _ = _path_points(items)
        out.segs.extend(_thin_polygon_to_segs(pts, fill_col))
    _cluster_widths(out.segs)
    return out


def _cluster_widths(segs: list[Seg], jump: float = 1.3):
    """Group pen widths into classes so small estimation noise doesn't split a curve."""
    ws = sorted({round(s.width, 3) for s in segs if s.width > 0})
    if not ws:
        return
    classes: list[list[float]] = [[ws[0]]]
    for w in ws[1:]:
        if w > classes[-1][-1] * 1.08 and w > classes[-1][0] * jump:
            classes.append([w])
        else:
            classes[-1].append(w)
    rep = {}
    for c in classes:
        r = round(float(np.median(c)), 2)
        for w in c:
            rep[w] = r
    for s in segs:
        s.wclass = rep.get(round(s.width, 3), round(s.width, 2))
