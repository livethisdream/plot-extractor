"""Separate the curves inside a plot frame into labeled series.

Steps:
1. keep curve-like segments inside the frame (drop grid, frame, ticks, text);
2. chain touching segments into *strokes* (a solid curve, or one dash);
3. read the legend: each non-numeric text line inside the frame plus the short
   horizontal run of strokes beside it gives a (label, line pattern) pair;
4. link strokes into *tracks* across dash gaps (global assignment on gap size
   and direction continuity);
5. match each track's dash signature (and color/width) to a legend entry.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pymupdf
from scipy.optimize import linear_sum_assignment

from .axes import PlotFrame, parse_number
from .primitives import Box, Seg
from .text import PageText


# --------------------------------------------------------------------------- strokes
@dataclass
class Stroke:
    pts: list[tuple[float, float]]
    style: tuple
    width: float
    color: tuple
    dashes: tuple | None = None

    @property
    def length(self) -> float:
        return float(sum(math.dist(a, b) for a, b in zip(self.pts, self.pts[1:])))

    @property
    def start(self):
        return self.pts[0]

    @property
    def end(self):
        return self.pts[-1]

    def orient(self):
        """Make the stroke run left to right (ties: top to bottom)."""
        if (self.pts[-1][0], self.pts[-1][1]) < (self.pts[0][0], self.pts[0][1]):
            self.pts.reverse()
        return self

    @property
    def is_dot(self) -> bool:
        """Dots are drawn as zero-length or roughly pen-width-long strokes."""
        return self.length < 1.3 * self.width + 0.1

    def tangent(self, at_end: bool):
        """Unit direction leaving the end (or arriving at the start)."""
        if self.is_dot:
            return None
        pts = self.pts if at_end else self.pts[::-1]
        a = pts[-1]
        for b in reversed(pts[:-1]):
            if math.dist(a, b) > 0.8:
                v = np.subtract(a, b)
                v = v / np.linalg.norm(v)
                return v if at_end else -v
        if len(pts) > 1 and math.dist(pts[0], pts[-1]) > 1e-6:
            v = np.subtract(pts[-1], pts[0])
            v = v / np.linalg.norm(v)
            return v if at_end else -v
        return None


def chain_segments(segs: list[Seg], tol_frac=0.35) -> list[Stroke]:
    """Join segments of the same style whose endpoints coincide."""
    by_style: dict[tuple, list[Seg]] = defaultdict(list)
    for s in segs:
        by_style[(s.style, s.dashes)].append(s)
    strokes: list[Stroke] = []
    for (style, dashes), group in by_style.items():
        w = float(np.median([s.width for s in group]))
        tol = max(0.12, tol_frac * w)
        # endpoint grid index
        def key(p):
            return (round(p[0] / tol), round(p[1] / tol))
        index: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
        for i, s in enumerate(group):
            index[key(s.p0)].append((i, 0))
            index[key(s.p1)].append((i, 1))

        def neighbors(p, exclude):
            kx, ky = key(p)
            out = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for i, e in index.get((kx + dx, ky + dy), []):
                        if i in exclude:
                            continue
                        q = group[i].p0 if e == 0 else group[i].p1
                        if math.dist(p, q) <= tol:
                            out.append((i, e))
            return out

        used: set[int] = set()
        for i, s in enumerate(group):
            if i in used:
                continue
            used.add(i)
            pts = [s.p0, s.p1] if s.length > 1e-6 else [s.p0]
            for forward in (True, False):
                while True:
                    tip = pts[-1] if forward else pts[0]
                    nb = neighbors(tip, used)
                    if not nb:
                        break
                    j, e = nb[0]
                    used.add(j)
                    sj = group[j]
                    other = sj.p1 if e == 0 else sj.p0
                    if sj.length <= 1e-6:
                        continue
                    if forward:
                        pts.append(other)
                    else:
                        pts.insert(0, other)
            strokes.append(Stroke(pts, style, w, style[1], dashes).orient())
    return strokes


# --------------------------------------------------------------------------- filtering
def curve_segments(frame: PlotFrame, segs: list[Seg], legend_rects: list[pymupdf.Rect]):
    """Segments inside the frame that can belong to data curves."""
    r = frame.rect
    pad = 1.5
    inside = [s for s in segs
              if r.x0 - pad <= s.mid[0] <= r.x1 + pad and r.y0 - pad <= s.mid[1] <= r.y1 + pad]
    # grid / frame / tick styles: nearly all of their length lies on a few long
    # horizontal or vertical lines. (Curves can be drawn as tiny H/V steps, so
    # "axis-aligned" alone isn't enough.)
    tot = defaultdict(float)
    hv = defaultdict(list)  # style -> [(orientation, coordinate, length)]
    for s in inside:
        L = s.length if s.length > 0 else 0.3
        tot[s.style] += L
        if s.axis_aligned() or s.length == 0:
            vertical = abs(s.p0[0] - s.p1[0]) < abs(s.p0[1] - s.p1[1])
            hv[s.style].append(("v", s.p0[0], L) if vertical else ("h", s.p0[1], L))
    curve_styles = set()
    for st, T in tot.items():
        lined = 0.0
        for o in ("h", "v"):
            span = r.width if o == "h" else r.height
            pts = sorted((c, L) for oo, c, L in hv[st] if oo == o)
            # bands of width 0.3 pt measured from the band's first coordinate
            # (not chained), so slowly sloping staircase curves don't merge
            band0, run_L = None, 0.0
            for c, L in pts + [(float("inf"), 0.0)]:
                if band0 is not None and c - band0 <= 0.3:
                    run_L += L
                else:
                    if run_L >= 0.15 * span:
                        lined += run_L
                    band0, run_L = c, L
        if T > 5.0 and lined < 0.85 * T:
            curve_styles.add(st)
    xs = set(np.round(frame.xaxis.positions, 1))
    ys = set(np.round(frame.yaxis.positions, 1))
    out = []
    for s in inside:
        if s.style not in curve_styles:
            continue
        if s.axis_aligned() and s.length > 0.3:
            vertical = abs(s.p0[0] - s.p1[0]) < abs(s.p0[1] - s.p1[1])
            c = round(s.p0[0], 1) if vertical else round(s.p0[1], 1)
            # on the frame border or exactly on a gridline, spanning a grid cell: drop
            if vertical and (abs(s.p0[0] - r.x0) < 0.6 or abs(s.p0[0] - r.x1) < 0.6):
                continue
            if not vertical and (abs(s.p0[1] - r.y0) < 0.6 or abs(s.p0[1] - r.y1) < 0.6):
                continue
            if (vertical and c in xs or not vertical and c in ys) and s.length > 0.9 * frame.xaxis.spacing:
                continue
        if any(lr.contains(pymupdf.Point(s.mid)) for lr in legend_rects):
            continue
        out.append(s)
    return out, curve_styles


# --------------------------------------------------------------------------- legend
@dataclass
class Pattern:
    solid: bool
    dash: float = 0.0  # median dash length (pt)
    gap: float = 0.0  # median gap length (pt)
    dot_frac: float = 0.0  # fraction of strokes that are dots
    dash_dot: bool = False
    color: tuple = (0, 0, 0)
    width: float = 1.0
    explicit: tuple | None = None

    def distance(self, o: "Pattern") -> float:
        d = 0.0
        d += 4.0 * float(np.linalg.norm(np.subtract(self.color, o.color)))
        d += abs(math.log(max(self.width, 0.1) / max(o.width, 0.1)))
        if self.explicit or o.explicit:
            return d + (0.0 if self.explicit == o.explicit else 2.0)
        if self.solid != o.solid:
            return d + 3.0
        if self.solid:
            return d
        d += abs(math.log1p(self.dash) - math.log1p(o.dash))
        d += abs(math.log1p(self.gap) - math.log1p(o.gap))
        d += 1.5 * abs(self.dot_frac - o.dot_frac)
        d += 0.7 * (self.dash_dot != o.dash_dot)
        return d



def pattern_of(strokes: list[Stroke], solid_min: float) -> Pattern:
    """Dash signature of an ordered run of strokes."""
    strokes = sorted(strokes, key=lambda s: s.start)
    lens = [s.length for s in strokes]
    gaps = [math.dist(a.end, b.start) for a, b in zip(strokes, strokes[1:])]
    s0 = strokes[0]
    if s0.dashes:
        return Pattern(False, color=s0.color, width=s0.width, explicit=s0.dashes)
    total = sum(lens)
    if (len(strokes) == 1 and total >= solid_min) or (gaps and max(lens) >= solid_min * 1.5):
        return Pattern(True, color=s0.color, width=s0.width)
    dots = [s.is_dot for s in strokes]
    dashes = [s.length for s in strokes if not s.is_dot]
    dot_frac = sum(dots) / len(lens)
    dash_dot = 0.15 < dot_frac < 0.85 and len(dashes) > 0
    return Pattern(False,
                   dash=float(np.median(dashes)) if dashes else 0.0,
                   gap=float(np.median(gaps)) if gaps else 0.0,
                   dot_frac=dot_frac, dash_dot=dash_dot,
                   color=s0.color, width=s0.width)


@dataclass
class LegendEntry:
    label: str
    rect: pymupdf.Rect
    strokes: list[Stroke]
    pattern: Pattern | None = None


def find_legend(page: pymupdf.Page, frame: PlotFrame, segs: list[Seg], boxes: list[Box],
                curve_styles: set, text: PageText | None = None):
    """Legend entries inside the frame, plus regions to exclude from curve data."""
    r = frame.rect
    text = text or PageText.from_page(page)
    # build label candidates from words grouped by text line, split on wide gaps
    words = [w.as_tuple() for w in text.words
             if abs(w.dir[1]) < 0.1 and r.contains(w.rect)]
    by_line = defaultdict(list)
    for w in words:
        by_line[(w[5], w[6])].append(w)
    labels = []
    for ws in by_line.values():
        ws.sort(key=lambda w: w[0])
        groups, cur = [], []
        for w in ws:
            if cur and w[0] - cur[-1][2] > max(6.0, 2.2 * (w[3] - w[1])):
                groups.append(cur)
                cur = []
            cur.append(w)
        if cur:
            groups.append(cur)
        for g in groups:
            txt = " ".join(w[4] for w in g)
            if parse_number(txt) is not None:
                continue
            gr = pymupdf.Rect(g[0][:4])
            for w in g[1:]:
                gr |= pymupdf.Rect(w[:4])
            labels.append((gr, txt))
    if not labels:
        return [], []

    # candidate sample strokes: everything inside the frame (curve-ish styles)
    inside = [s for s in segs if r.contains(pymupdf.Point(s.mid)) and s.style in curve_styles]
    strokes = chain_segments(inside)
    entries = []
    for gr, txt in labels:
        yc = (gr.y0 + gr.y1) / 2
        h = gr.height
        best = None
        for side in ("left", "right"):
            if side == "left":
                win = pymupdf.Rect(gr.x0 - 40, yc - 0.45 * h, gr.x0 + 0.5, yc + 0.45 * h)
            else:
                win = pymupdf.Rect(gr.x1 - 0.5, yc - 0.45 * h, gr.x1 + 40, yc + 0.45 * h)
            cand = [s for s in strokes
                    if all(win.contains(pymupdf.Point(p)) for p in (s.start, s.end))
                    and abs(s.end[1] - s.start[1]) < 0.6]
            if not cand:
                continue
            # keep the contiguous run nearest the label
            cand.sort(key=lambda s: s.start[0], reverse=(side == "left"))
            run = []
            edge = gr.x0 if side == "left" else gr.x1
            last = edge
            for s in cand:
                gap = (last - s.end[0]) if side == "left" else (s.start[0] - last)
                if gap > 9:
                    break
                run.append(s)
                last = s.start[0] if side == "left" else s.end[0]
            if not run:
                continue
            near = (edge - max(s.end[0] for s in run)) if side == "left" else (min(s.start[0] for s in run) - edge)
            if best is None or near < best[0]:
                best = (near, run)
        if best:
            entries.append(LegendEntry(txt, gr, best[1]))

    if not entries:
        return [], []
    # exclusion region: enclosing box if one exists, else padded union
    u = pymupdf.Rect(entries[0].rect)
    for e in entries:
        u |= e.rect
        for s in e.strokes:
            for q in s.pts:
                u.include_point(q)
    enclosing = [b.rect for b in boxes if b.rect.contains(u) and r.contains(b.rect)
                 and b.rect.get_area() < 0.5 * r.get_area()]
    region = min(enclosing, key=lambda x: x.get_area()) if enclosing else u + (-3, -3, 3, 3)
    return entries, [region]


# --------------------------------------------------------------------------- linking
def link_strokes(strokes: list[Stroke], max_gap: float, solid_min: float,
                 allow_dash_dot: bool = False) -> list[list[Stroke]]:
    """Link dashes into tracks with a global min-cost assignment."""
    n = len(strokes)
    if n == 0:
        return []
    BIG = 1e6
    C = np.full((n, n), BIG)
    ends = [s.end for s in strokes]
    starts = [s.start for s in strokes]
    t_end = [s.tangent(True) for s in strokes]
    t_start = [s.tangent(False) for s in strokes]
    lens = [s.length for s in strokes]
    for i in range(n):
        for j in range(n):
            if i == j or strokes[i].style != strokes[j].style:
                continue
            g = np.subtract(starts[j], ends[i])
            d = float(np.hypot(*g))
            if d > max_gap or g[0] < -0.3:
                continue
            cost = d / max_gap
            if d > 0.3:
                gu = g / d
                for t in (t_end[i], t_start[j]):
                    if t is not None:
                        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(t, gu))))))
                        if ang > 60:
                            cost += 10
                        cost += ang / 45.0
                if t_end[i] is not None and t_start[j] is not None:
                    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(t_end[i], t_start[j]))))))
                    cost += ang / 90.0
            # dashes of one series have similar lengths unless it is dash-dot
            li, lj = lens[i], lens[j]
            if strokes[i].is_dot != strokes[j].is_dot:
                cost += 0.0 if allow_dash_dot else 1.5
            else:
                cost += 0.6 * abs(math.log((li + 0.3) / (lj + 0.3)))
            C[i, j] = cost
    # add "no link" option
    thresh = 2.2
    M = np.full((2 * n, 2 * n), BIG)
    M[:n, :n] = C
    M[:n, n:] = np.where(np.eye(n) > 0, thresh, BIG)
    M[n:, :n] = np.where(np.eye(n) > 0, thresh, BIG)
    M[n:, n:] = 0
    rows, cols = linear_sum_assignment(M)
    nxt = {}
    for i, j in zip(rows, cols):
        if i < n and j < n and M[i, j] < thresh:
            nxt[i] = j
    has_prev = set(nxt.values())
    tracks = []
    seen = set()
    for i in range(n):
        if i in has_prev or i in seen:
            continue
        tr = []
        k = i
        while k is not None and k not in seen:
            seen.add(k)
            tr.append(strokes[k])
            k = nxt.get(k)
        tracks.append(tr)
    for i in range(n):  # cycles (shouldn't happen) – keep as singletons
        if i not in seen:
            tracks.append([strokes[i]])
    return tracks
