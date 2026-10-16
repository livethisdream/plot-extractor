"""Read text that a PDF draws as vector outlines (no text layer), via Tesseract.

Some datasheets (e.g. newer ADI ones) export every tick label, axis title and
legend entry as filled glyph shapes. We:

1. collect small dark filled shapes (glyphs) and group them into words/lines
   by adjacency (horizontal lines, plus rotated y-axis titles);
2. render each group cleanly (everything that isn't one of its glyphs is
   masked to white), stack them into one image and OCR it in a single
   Tesseract call (a second call with a digits-only whitelist for numbers);
3. rebuild numbers geometrically: the minus sign and decimal point are
   identified from glyph shape and position, OCR only supplies the digits.

Requires the ``tesseract`` executable (apt install tesseract-ocr / brew install
tesseract). If it's missing, OCR is skipped and a warning is recorded.
"""
from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf

from .text import Word

DPI = 600
MAX_GLYPH = 12.0  # pt; larger filled shapes aren't text


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


@dataclass
class Glyph:
    rect: pymupdf.Rect

    @property
    def w(self):
        return self.rect.width

    @property
    def h(self):
        return self.rect.height

    def is_dash(self, ref_h: float) -> bool:
        return self.h < 0.35 * ref_h and self.w > 1.8 * self.h

    def is_dot(self, ref_h: float) -> bool:
        return max(self.w, self.h) < 0.32 * ref_h


def _contours(items) -> list[pymupdf.Rect]:
    """Bounding boxes of the separate closed outlines in one drawing."""
    boxes, cur, last = [], None, None
    for it in items:
        op = it[0]
        if op == "re":
            boxes.append(pymupdf.Rect(it[1]))
            last = None
            continue
        if op == "qu":
            boxes.append(it[1].rect)
            last = None
            continue
        pts = [it[1], it[-1]] if op == "l" else list(it[1:5])
        start = pts[0]
        if last is None or abs(start.x - last.x) > 1e-3 or abs(start.y - last.y) > 1e-3:
            if cur is not None:
                boxes.append(cur)
            cur = pymupdf.Rect(start, start)
        for q in pts:
            cur.include_point(q)
        last = pts[-1]
    if cur is not None:
        boxes.append(cur)
    # holes (the inside of 0, 6, 8, ...) belong to the glyph that contains them
    boxes.sort(key=lambda r: -r.get_area())
    out: list[pymupdf.Rect] = []
    for b in boxes:
        if not any(o.contains(b) for o in out):
            out.append(b)
    return out


def collect_glyphs(page: pymupdf.Page) -> list[Glyph]:
    out = []
    for g in page.get_drawings():
        if "f" not in (g.get("type") or ""):
            continue
        r = g["rect"]
        if r.width <= 0 or r.height <= 0 or r.height > MAX_GLYPH:
            continue
        fill = g.get("fill") or (0, 0, 0)
        if len(fill) == 3 and sum(fill) / 3 > 0.6:  # white/light fills are backgrounds
            continue
        for b in _contours(g["items"]):
            if b.width > 0 and b.height > 0 and max(b.width, b.height) <= MAX_GLYPH:
                out.append(Glyph(b))
    return out


def _overlap(a0, a1, b0, b1):
    return min(a1, b1) - max(a0, b0)


def group_glyphs(glyphs: list[Glyph]) -> list[tuple[list[Glyph], bool]]:
    """Union-find glyphs into lines. Returns (glyphs, rotated) per group."""
    n = len(glyphs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    rotated_link = [False] * n
    order = sorted(range(n), key=lambda i: glyphs[i].rect.x0)
    for ii, i in enumerate(order):
        a = glyphs[i].rect
        for j in order[ii + 1:]:
            b = glyphs[j].rect
            if b.x0 - a.x1 > MAX_GLYPH:
                break
            hmax = max(a.height, b.height)
            # same text row
            vo = _overlap(a.y0, a.y1, b.y0, b.y1)
            gap = max(a.x0, b.x0) - min(a.x1, b.x1)
            if vo >= 0.5 * min(a.height, b.height) and gap <= 0.45 * hmax:
                parent[find(i)] = find(j)
                continue
            # same rotated column (glyphs wider than tall)
            ho = _overlap(a.x0, a.x1, b.x0, b.x1)
            wmax = max(a.width, b.width)
            vgap = max(a.y0, b.y0) - min(a.y1, b.y1)
            big = a if a.width * a.height >= b.width * b.height else b
            if big.width > 1.05 * big.height and ho >= 0.5 * min(a.width, b.width) and vgap <= 0.45 * wmax:
                parent[find(i)] = find(j)
                rotated_link[i] = rotated_link[j] = True
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in groups.values():
        gs = [glyphs[i] for i in idx]
        rot = any(rotated_link[i] for i in idx)
        out.append((gs, rot))
    return out


def _group_rect(gs):
    r = pymupdf.Rect(gs[0].rect)
    for g in gs[1:]:
        r |= g.rect
    return r


def _render_group(page, gs, rotated) -> np.ndarray:
    pad = 1.0
    clip = _group_rect(gs) + (-pad, -pad, pad, pad)
    pix = page.get_pixmap(dpi=DPI, clip=clip, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).copy()
    s = pix.w / clip.width
    mask = np.zeros_like(img, bool)
    for g in gs:
        r = g.rect + (-0.25, -0.25, 0.25, 0.25)
        x0, y0 = int((r.x0 - clip.x0) * s), int((r.y0 - clip.y0) * s)
        x1, y1 = int(np.ceil((r.x1 - clip.x0) * s)), int(np.ceil((r.y1 - clip.y0) * s))
        mask[max(y0, 0):y1, max(x0, 0):x1] = True
    img[~mask] = 255
    if rotated:
        img = np.rot90(img, k=-1)  # bottom-to-top text -> left-to-right
    return img


def _tesseract_rows(imgs: list[np.ndarray], digits_only: bool) -> list[str]:
    """OCR many single-line images in one call; returns text per image."""
    if not imgs:
        return []
    # normalize height so tesseract sees consistent text size
    from PIL import Image
    target = 64
    rows = []
    for im in imgs:
        h, w = im.shape
        sc = target / max(h, 1)
        pim = Image.fromarray(im).resize((max(1, int(w * sc)), target), Image.LANCZOS)
        rows.append(np.asarray(pim))
    gap = 40
    W = max(r.shape[1] for r in rows) + 2 * gap
    H = sum(r.shape[0] + gap for r in rows) + gap
    canvas = np.full((H, W), 255, np.uint8)
    tops = []
    y = gap
    for r in rows:
        canvas[y:y + r.shape[0], gap:gap + r.shape[1]] = r
        tops.append((y, y + r.shape[0]))
        y += r.shape[0] + gap
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rows.png"
        Image.fromarray(canvas).save(p)
        cmd = ["tesseract", str(p), "-", "--psm", "6"]
        if digits_only:
            cmd += ["-c", "tessedit_char_whitelist=0123456789"]
        cmd += ["tsv"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    texts = [[] for _ in imgs]
    for rec in csv.DictReader(io.StringIO(out), delimiter="\t"):
        t = (rec.get("text") or "").strip()
        if not t:
            continue
        top, h = int(rec["top"]), int(rec["height"])
        cy = top + h / 2
        for k, (a, b) in enumerate(tops):
            if a - gap / 2 <= cy <= b + gap / 2:
                texts[k].append((int(rec["left"]), t))
                break
    return [" ".join(t for _, t in sorted(ws)) for ws in texts]


_CONFUSE = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1",
                          "i": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "g": "9"})


def _numeric_from(gs: list[Glyph], general: str, digits: str) -> str | None:
    """Rebuild a number from glyph geometry + OCR'd digits, or None."""
    gs = sorted(gs, key=lambda g: g.rect.x0)
    ref_h = max(g.h for g in gs)
    kinds = []
    for g in gs:
        if g.is_dash(ref_h):
            kinds.append("-")
        elif g.is_dot(ref_h):
            kinds.append(".")
        else:
            kinds.append("d")
    nd = kinds.count("d")
    if nd == 0 or nd > 7:
        return None
    # the general reading must look numeric: at most one stray letter
    core = re.sub(r"[\s\-−–—.,+~]", "", general).translate(_CONFUSE)
    if not core or sum(c.isalpha() for c in core) > 1 or not any(c.isdigit() for c in core + digits):
        return None
    cands = [re.sub(r"\D", "", digits), re.sub(r"\D", "", core)]
    ds = next((c for c in cands if len(c) == nd), None)
    if ds is None:
        return None
    out, k = [], 0
    for i, kind in enumerate(kinds):
        if kind == "d":
            out.append(ds[k])
            k += 1
        elif kind == ".":
            if 0 < i < len(kinds) - 1:
                out.append(".")
        elif kind == "-" and i == 0:
            out.append("-")
    s = "".join(out)
    if general.strip().startswith("+"):
        s = "+" + s
    return s


def ocr_words(page: pymupdf.Page, min_glyphs: int = 12) -> tuple[list[Word], str]:
    """OCR outlined text on a page. Returns (words, status message)."""
    glyphs = collect_glyphs(page)
    if len(glyphs) < min_glyphs:
        return [], ""
    if not tesseract_available():
        return [], "page has outlined (non-text) labels but tesseract is not installed; plots skipped"
    groups = group_glyphs(glyphs)
    groups = [(gs, rot) for gs, rot in groups if _group_rect(gs).height > 1.2 or len(gs) > 1]
    imgs = [_render_group(page, gs, rot) for gs, rot in groups]
    general = _tesseract_rows(imgs, digits_only=False)
    digits = _tesseract_rows(imgs, digits_only=True)
    words = []
    for k, ((gs, rot), g_txt, d_txt) in enumerate(zip(groups, general, digits)):
        r = _group_rect(gs)
        num = None if rot else _numeric_from(gs, g_txt, d_txt)
        text = num if num is not None else re.sub(r"\s+", " ", g_txt).strip()
        text = re.sub(r"[\u2014\u2013\u2212](?=\d)", "-", text)  # dashes before digits are minus signs
        if not text:
            continue
        words.append(Word(r.x0, r.y0, r.x1, r.y1, text, k, 0,
                          (0.0, -1.0) if rot else (1.0, 0.0), "ocr"))
    return _merge_rotated(words), f"read {len(words)} outlined labels with OCR"


def _merge_rotated(words: list[Word]) -> list[Word]:
    """Join word pieces of a rotated (bottom-to-top) title into one line."""
    rot = sorted([w for w in words if w.dir[1]], key=lambda w: (round(w.x0), -w.y1))
    flat = [w for w in words if not w.dir[1]]
    merged: list[Word] = []
    for w in rot:
        m = merged[-1] if merged else None
        if m and abs(m.x0 - w.x0) < 1.5 and 0 <= m.y0 - w.y1 < 2.5 * (w.x1 - w.x0):
            merged[-1] = Word(min(m.x0, w.x0), w.y0, max(m.x1, w.x1), m.y1, m.text + " " + w.text,
                              m.block, 0, m.dir, "ocr")
        else:
            merged.append(w)
    return flat + merged
