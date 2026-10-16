"""Page text as a flat word list, from the PDF text layer and/or OCR.

Everything downstream (axis detection, legends, titles) reads words through
``PageText`` so it doesn't care whether a word was real PDF text or was read
by OCR from outlined glyphs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pymupdf


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block: int
    line: int
    dir: tuple = (1.0, 0.0)  # (1,0) horizontal; (0,-1) reads bottom-to-top
    source: str = "pdf"  # pdf | ocr

    @property
    def rect(self) -> pymupdf.Rect:
        return pymupdf.Rect(self.x0, self.y0, self.x1, self.y1)

    def as_tuple(self):
        """Same layout as PyMuPDF's get_text('words') tuples."""
        return (self.x0, self.y0, self.x1, self.y1, self.text, self.block, self.line, 0)


@dataclass
class PageText:
    words: list[Word] = field(default_factory=list)
    ocr_used: bool = False

    @classmethod
    def from_page(cls, page: pymupdf.Page) -> "PageText":
        # text direction per word, matched by position (block/line numbers from
        # get_text("words") and get_text("dict") don't always correspond)
        lines = []
        for b in page.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                d = tuple(round(v, 3) for v in ln["dir"])
                if abs(d[1]) > 0.1:
                    lines.append((pymupdf.Rect(ln["bbox"]), d))
        words = []
        for w in page.get_text("words"):
            c = pymupdf.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)
            d = next((dd for r, dd in lines if r.contains(c)), (1.0, 0.0))
            words.append(Word(w[0], w[1], w[2], w[3], w[4], w[5], w[6], d))
        return cls(words)

    def lines(self) -> list[tuple[pymupdf.Rect, tuple, str]]:
        """(rect, dir, text) per text line, in reading order within the line."""
        groups = defaultdict(list)
        for w in self.words:
            groups[(w.block, w.line)].append(w)
        out = []
        for ws in groups.values():
            horiz = abs(ws[0].dir[1]) < 0.1
            ws.sort(key=(lambda w: w.x0) if horiz else (lambda w: -w.y1))
            r = pymupdf.Rect(ws[0].rect)
            for w in ws[1:]:
                r |= w.rect
            out.append((r, ws[0].dir, " ".join(w.text for w in ws)))
        return out

    def extend(self, words: list[Word]):
        """Add OCR'd words, grouping horizontal ones into lines by position."""
        base = max([w.block for w in self.words] + [0]) + 1000
        horiz = sorted([w for w in words if abs(w.dir[1]) < 0.1], key=lambda w: w.x0)
        lines: list[list[Word]] = []
        for w in horiz:
            h = w.y1 - w.y0
            cy = (w.y0 + w.y1) / 2
            best = None
            for ln in lines:
                last = ln[-1]
                lh = max(last.y1 - last.y0, h)
                lcy = (ln[0].y0 + ln[0].y1) / 2
                gap = w.x0 - last.x1
                # same baseline band, and close enough to be the same phrase
                if abs(cy - lcy) < 0.45 * lh and -0.5 < gap < 2.2 * lh:
                    best = ln
                    break
            if best is None:
                lines.append([w])
            else:
                best.append(w)
        for k, ln in enumerate(lines):
            for w in ln:
                w.block, w.line = base + k, 0
        for k, w in enumerate(w for w in words if abs(w.dir[1]) >= 0.1):
            w.block, w.line = base + len(lines) + k, 0
        self.words += words
