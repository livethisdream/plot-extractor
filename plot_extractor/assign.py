"""Assign curve strokes to legend entries using the legend's dash "grammar".

Every legend sample is a short sequence of *elements* (dot, a dash of a given
length, or a solid run) separated by gaps. We build a catalog of dash lengths
from all legend samples, classify every stroke in the plot by element type,
and only allow two strokes to be linked when some legend pattern contains
that element transition with a matching gap. Each linked track is then voted
into the legend group(s) whose transitions it used.

Legend entries whose patterns are identical (it happens -- e.g. two dotted
series) form one *group*; their tracks are split into separate curves and
reported as ambiguous rather than guessed.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pymupdf
from scipy.optimize import linear_sum_assignment

from .axes import PlotFrame
from .primitives import Box, Seg
from .series import (LegendEntry, Stroke, chain_segments, curve_segments, find_legend,
                     link_strokes, pattern_of)


# --------------------------------------------------------------------------- elements
@dataclass
class Catalog:
    dashes: list[float]  # catalog dash lengths (pt)
    has_solid: bool

    def classify(self, s: Stroke):
        if s.dashes:
            return ("x", s.dashes)
        if s.is_dot:
            return ("dot",)
        L = s.length
        if not self.dashes:
            return ("solid",)
        big = max(self.dashes)
        # dash patterns are often applied before a non-uniform x/y scale, so dashes
        # on steep stretches come out up to ~35% longer than the horizontal legend sample
        stretch = 1.0 + 0.4 * _sin2(s)
        if L > 1.5 * big * stretch:
            return ("solid",)
        D = self.dashes
        k = min(range(len(D)), key=lambda i: abs(math.log(L / D[i])))
        c = D[k]
        up = min(1.15 * stretch, math.sqrt(D[k + 1] / c) if k + 1 < len(D) else 9)
        lo = max(0.82, math.sqrt(D[k - 1] / c) if k > 0 else 0)
        if lo * c <= L <= up * c:
            return ("d", k)
        if L < c:
            return ("p", L)  # clipped dash: wildcard for any longer dash
        return ("?",)


def _sin2(s: Stroke) -> float:
    P = np.asarray(s.pts, float)
    if len(P) < 2:
        return 0.0
    d = np.diff(P, axis=0)
    L = np.hypot(*d.T)
    if L.sum() <= 0:
        return 0.0
    return float(np.sum(L * (d[:, 1] / np.maximum(L, 1e-9)) ** 2) / L.sum())


def _match(t, a) -> float | None:
    """Penalty for stroke element ``t`` standing in for pattern element ``a`` (None = no)."""
    if t == a:
        return 0.0
    if t[0] == "p" and a[0] == "d":
        return 0.4
    if t[0] == "p" and a[0] == "solid":
        return 0.6
    return None


def build_catalog(entries: list[LegendEntry]) -> Catalog:
    lens = []
    has_solid = False
    for e in entries:
        ss = [s for s in e.strokes if not s.is_dot and not s.dashes]
        if len(e.strokes) == 1 and ss:
            has_solid = True
            continue
        vals = [s.length for s in ss]
        reliable = [v for i, v in enumerate(vals)
                    if any(abs(v - w) <= 0.12 * v for j, w in enumerate(vals) if j != i)]
        if not reliable and vals:
            reliable = [max(vals)]
        lens += reliable
    lens.sort()
    centers: list[list[float]] = []
    for v in lens:
        if centers and abs(v - np.mean(centers[-1])) <= 0.12 * v:
            centers[-1].append(v)
        else:
            centers.append([v])
    return Catalog([float(np.median(c)) for c in centers], has_solid)


@dataclass
class Group:
    labels: list[str]
    style: tuple
    elements: set
    trans: dict  # (a, b) -> expected gap
    solid: bool = False

    @property
    def name(self):
        return " | ".join(self.labels)


def build_groups(entries: list[LegendEntry], cat: Catalog) -> list[Group]:
    groups: list[Group] = []
    for e in entries:
        ss = sorted(e.strokes, key=lambda s: s.start)
        types = [cat.classify(s) for s in ss]
        if len(ss) == 1 and types[0][0] in ("d", "p", "?"):
            types = [("solid",)]
        style = ss[0].style
        trans: dict = defaultdict(list)
        for (a, sa), (b, sb) in zip(zip(types, ss), list(zip(types, ss))[1:]):
            if a[0] in ("p", "?") or b[0] in ("p", "?"):
                continue
            g = math.dist(sa.end, sb.start)
            trans[(a, b)].append(g)
            trans[(b, a)].append(g)
        elements = {t for t in types if t[0] not in ("p", "?")}
        solid = elements <= {("solid",)} or any(t[0] == "x" for t in elements)
        tr = {k: float(np.median(v)) for k, v in trans.items()}
        for t in elements:
            if t[0] in ("solid", "x"):
                tr[(t, t)] = 0.0
        # merge with an identical group
        for g in groups:
            if g.style == style and g.elements == elements and set(g.trans) == set(tr) and all(
                    abs(g.trans[k] - tr[k]) <= 0.15 * max(tr[k], 1.0) for k in tr):
                g.labels.append(e.label)
                break
        else:
            groups.append(Group([e.label], style, elements, tr, solid))
    return groups


# --------------------------------------------------------------------------- linking
def _angle(u, v):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u, v))))))


def grammar_link(strokes: list[Stroke], types: list, groups: list[Group]):
    n = len(strokes)
    BIG = 1e6
    C = np.full((n, n), BIG)
    allowed: dict[tuple[int, int], set[int]] = {}
    t_end = [s.tangent(True) for s in strokes]
    t_start = [s.tangent(False) for s in strokes]
    gmax = max([v for g in groups for v in g.trans.values()] + [1.0])
    reach = 1.6 * gmax + 1.0
    starts = np.asarray([s.start for s in strokes])
    for i in range(n):
        e = np.asarray(strokes[i].end)
        dd = np.hypot(*(starts - e).T)
        for j in np.nonzero(dd <= reach)[0]:
            j = int(j)
            if i == j or strokes[i].style != strokes[j].style:
                continue
            g = starts[j] - e
            d = float(dd[j])
            if g[0] < -0.3 and d > 0.3:
                continue
            ang_cost = 0.0
            if d > 0.5:
                gu = g / d
                bad = False
                for t in (t_end[i], t_start[j]):
                    if t is not None:
                        a = _angle(t, gu)
                        if a > 65:
                            bad = True
                        ang_cost += a / 40.0
                if bad:
                    continue
            elif t_end[i] is not None and t_start[j] is not None:
                if _angle(t_end[i], t_start[j]) > 70:
                    continue
            per_group = {}
            for gi, G in enumerate(groups):
                if G.style != strokes[i].style:
                    continue
                best = None
                for (a, b), ge in G.trans.items():
                    pa, pb = _match(types[i], a), _match(types[j], b)
                    if pa is None or pb is None:
                        continue
                    if ge == 0.0:
                        if d > 1.0:
                            continue
                        c = d / 0.5
                    else:
                        if d > 1.6 * ge + 1.0:
                            continue
                        c = abs(d - ge) / (0.3 * ge + 0.4)
                    c += pa + pb
                    best = c if best is None else min(best, c)
                if best is not None:
                    per_group[gi] = best + ang_cost
            if not per_group:
                continue
            m = min(per_group.values())
            C[i, j] = m
            allowed[(i, j)] = {gi for gi, c in per_group.items() if c <= m + 0.25}
    thresh = 2.2
    M = np.full((2 * n, 2 * n), BIG)
    M[:n, :n] = C
    M[:n, n:] = np.where(np.eye(n) > 0, thresh, BIG)
    M[n:, :n] = np.where(np.eye(n) > 0, thresh, BIG)
    M[n:, n:] = 0
    rows, cols = linear_sum_assignment(M)
    nxt = {i: j for i, j in zip(rows, cols) if i < n and j < n and M[i, j] < thresh}
    has_prev = set(nxt.values())
    tracks, seen = [], set()
    for i in range(n):
        if i in has_prev or i in seen:
            continue
        tr, links, k = [i], [], i
        seen.add(i)
        while k in nxt and nxt[k] not in seen:
            links.append((k, nxt[k]))
            k = nxt[k]
            seen.add(k)
            tr.append(k)
        tracks.append((tr, links))
    for i in range(n):
        if i not in seen:
            tracks.append(([i], []))
    return tracks, allowed, C


# --------------------------------------------------------------------------- growth
def _track_dir(strokes, idx, at_end: bool, span=5.0):
    """Direction of a track near one end, measured over ~span pt of its points."""
    pts = [p for i in idx for p in strokes[i].pts]
    if not at_end:
        pts = pts[::-1]
    a = np.asarray(pts[-1])
    for p in reversed(pts[:-1]):
        if math.dist(a, p) >= span:
            v = a - np.asarray(p)
            return v / np.linalg.norm(v)
    if len(pts) > 1 and math.dist(pts[0], pts[-1]) > 0.8:
        v = a - np.asarray(pts[0])
        return v / np.linalg.norm(v)
    return None


def _grow(strokes, types, groups, by_group, max_cost=1.6):
    """Greedily extend assigned tracks with unassigned strokes that continue them."""
    used = {i for ts in by_group.values() for t in ts for i in t}
    free = set(range(len(strokes))) - used
    added = 0
    while free:
        best = None
        for gi, ts in by_group.items():
            G = groups[gi]
            gaps = [v for v in G.trans.values() if v > 0]
            reach = 1.6 * max(gaps) + 1.0 if gaps else 1.5
            for ti, t in enumerate(ts):
                for at_end in (True, False):
                    tip = np.asarray(strokes[t[-1]].end if at_end else strokes[t[0]].start)
                    tdir = _track_dir(strokes, t, at_end)
                    for j in free:
                        sj = strokes[j]
                        if sj.style != G.style:
                            continue
                        q = np.asarray(sj.start if at_end else sj.end)
                        g = q - tip
                        d = float(np.hypot(*g))
                        if d > reach:
                            continue
                        if (g[0] < -0.3 if at_end else g[0] > 0.3) and d > 0.3:
                            continue
                        cost = d / reach
                        if d > 0.5:
                            gu = g / d
                            # tdir points outward from the track tip; the candidate's own
                            # direction must continue along the gap too
                            tj = sj.tangent(False) if at_end else sj.tangent(True)
                            if tj is not None and not at_end:
                                tj = -tj
                            ok = True
                            for tv in (tdir, tj):
                                if tv is None:
                                    continue
                                a = _angle(tv, gu)
                                if a > 50:
                                    ok = False
                                cost += a / 45.0
                            if not ok:
                                continue
                        el_ok = any(_match(types[j], el) is not None for el in G.elements) or types[j][0] == "?"
                        cost += 0.0 if el_ok else 0.9
                        if best is None or cost < best[0]:
                            best = (cost, gi, ti, j, at_end)
        if best is None or best[0] > max_cost:
            break
        _, gi, ti, j, at_end = best
        if at_end:
            by_group[gi][ti].append(j)
        else:
            by_group[gi][ti].insert(0, j)
        free.discard(j)
        added += 1
    return added


# --------------------------------------------------------------------------- output
@dataclass
class Series:
    label: str
    points: np.ndarray  # data coordinates, sorted by x
    pdf_points: np.ndarray
    tracks: list = field(default_factory=list)
    confidence: float = 1.0
    ambiguous_with: list[str] = field(default_factory=list)
    note: str = ""


def _split_curves(tracks: list[list[Stroke]], reach: float) -> list[list[list[Stroke]]]:
    """Greedily chain tracks left-to-right into separate, non-overlapping curves."""
    tracks = sorted(tracks, key=lambda t: t[0].start[0])
    curves: list[list[list[Stroke]]] = []
    for t in tracks:
        s0 = t[0].start
        best, bd = None, None
        for c in curves:
            e = c[-1][-1].end
            if e[0] > s0[0] + 1.0:
                continue
            d = math.dist(e, s0)
            if d <= reach and (bd is None or d < bd):
                best, bd = c, d
        if best is None:
            curves.append([t])
        else:
            best.append(t)
    return curves


def _x_overlap(tracks) -> float:
    """Fraction of a series' x-span covered by more than one fragment (should be ~0)."""
    iv = sorted((min(p[0] for s in t for p in s.pts), max(p[0] for s in t for p in s.pts)) for t in tracks)
    if len(iv) < 2:
        return 0.0
    span = max(b for _, b in iv) - iv[0][0]
    over, reach = 0.0, iv[0][1]
    for a, b in iv[1:]:
        if a < reach:
            over += min(b, reach) - a
        reach = max(reach, b)
    return over / span if span > 0 else 0.0


def _points(frame, tracks):
    pts = [p for t in tracks for s in t for p in s.pts]
    P = np.asarray(pts, float).reshape(-1, 2)
    if len(P):
        P = P[np.argsort(P[:, 0], kind="stable")]
        keep = np.r_[True, np.any(np.abs(np.diff(P, axis=0)) > 1e-3, axis=1)]
        P = P[keep]
    return frame.to_data(P) if len(P) else np.empty((0, 2)), P


def extract_series(page: pymupdf.Page, frame: PlotFrame, segs: list[Seg], boxes: list[Box]):
    _, styles = curve_segments(frame, segs, [])
    entries, excl = find_legend(page, frame, segs, boxes, styles)
    csegs, _ = curve_segments(frame, segs, excl)
    strokes = chain_segments(csegs)
    diag = {"legend": [e.label for e in entries], "unassigned_length_pt": 0.0}

    if not entries:
        return _no_legend(frame, strokes), entries, diag

    cat = build_catalog(entries)
    groups = build_groups(entries, cat)
    types = [cat.classify(s) for s in strokes]
    raw_tracks, allowed, C = grammar_link(strokes, types, groups)

    by_group: dict[int, list] = defaultdict(list)
    costs: dict[int, list] = defaultdict(list)
    unassigned = 0.0
    for idx, links in raw_tracks:
        votes = Counter()
        for (i, j) in links:
            a = allowed[(i, j)]
            for gi in a:
                votes[gi] += 1.0 / len(a)
        for i in idx:
            cands = [gi for gi, G in enumerate(groups)
                     if G.style == strokes[i].style and any(_match(types[i], el) is not None for el in G.elements)]
            for gi in cands:
                votes[gi] += 0.5 / len(cands)
        length = sum(strokes[i].length for i in idx)
        if not votes:
            unassigned += length
            continue
        (gi, v), *rest = votes.most_common()
        total = sum(votes.values())
        # single strokes must be unambiguous; tiny specks are noise
        if (len(idx) == 1 and (rest and rest[0][1] >= v)) or length < 0.5 and len(idx) < 3:
            unassigned += length
            continue
        if v < 0.55 * total:
            unassigned += length
            continue
        by_group[gi].append(list(idx))
        lc = [C[i, j] for i, j in links]
        solo = 0.0 if types[idx[0]][0] in ("solid", "x") else 1.0
        costs[gi].append((float(np.mean(lc)) if lc else solo, length, v / total))

    grown = _grow(strokes, types, groups, by_group)
    used = {i for ts in by_group.values() for t in ts for i in t}
    unassigned = sum(s.length for i, s in enumerate(strokes) if i not in used)
    diag["unassigned_length_pt"] = round(unassigned, 1)
    diag["grown_strokes"] = grown
    by_group = {gi: [[strokes[i] for i in t] for t in ts] for gi, ts in by_group.items()}

    gmax = max([v for g in groups for v in g.trans.values()] + [1.0])
    out: list[Series] = []
    for gi, G in enumerate(groups):
        ts = by_group.get(gi, [])
        cs = costs.get(gi, [])
        w = sum(L for _, L, _ in cs) or 1.0
        conf = float(np.exp(-0.5 * sum(c * L for c, L, _ in cs) / w) *
                     (sum(s * L for _, L, s in cs) / w)) if cs else 0.0
        if len(G.labels) == 1:
            data, P = _points(frame, ts)
            ov = _x_overlap(ts)
            note = ""
            if ov > 0.08:
                note = f"fragments overlap over {ov:.0%} of the x-span; series may include pieces of a neighboring curve"
                conf *= max(0.2, 1 - 2 * ov)
            out.append(Series(G.labels[0], data, P, ts, round(conf, 3), note=note))
            continue
        curves = _split_curves(ts, reach=4 * gmax + 4)
        # keep the longest curves, one per label
        curves.sort(key=lambda c: -sum(s.length for t in c for s in t))
        main = curves[:len(G.labels)]
        main.sort(key=lambda c: float(np.mean([p[1] for t in c for s in t for p in s.pts])))  # top first
        note = (f"legend entries {G.labels} use an identical line style; curves are listed "
                f"top-to-bottom on the page and cannot be labeled from the legend alone")
        for k, c in enumerate(main):
            data, P = _points(frame, c)
            out.append(Series(f"{G.name} [curve {k + 1} of {len(main)}, top to bottom]", data, P, c,
                              round(conf * 0.5, 3), list(G.labels), note))
        for lab in G.labels[len(main):]:
            out.append(Series(lab, np.empty((0, 2)), np.empty((0, 2)), [], 0.0, list(G.labels),
                              "not found"))
    return out, entries, diag


def _no_legend(frame, strokes):
    """No legend: link generically, then group tracks by line style."""
    tracks = link_strokes(strokes, max_gap=6.0, solid_min=12.0)
    tracks = [t for t in tracks if sum(s.length for s in t) > 2.0]
    curves = _split_curves(tracks, reach=10.0)
    curves = [c for c in curves if sum(s.length for t in c for s in t) > 0.1 * frame.rect.width]
    out = []
    for k, c in enumerate(sorted(curves, key=lambda c: c[0][0].start[1])):
        data, P = _points(frame, c)
        p = pattern_of([s for t in c for s in t], 12.0)
        kind = "solid" if p.solid else f"dash~{p.dash:.1f}pt"
        out.append(Series(f"series_{k + 1} ({kind})", data, P, c, 0.5))
    return out
