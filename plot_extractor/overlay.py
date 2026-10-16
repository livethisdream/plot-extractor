"""Draw extracted series over a render of the original plot, for spot-checking."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pymupdf  # noqa: E402

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
          "#f032e6", "#9a6324", "#469990", "#808000"]


def render_overlay(page: pymupdf.Page, frame, series, path: str, dpi: int = 200,
                   show_tracks: bool = False):
    clip = pymupdf.Rect(frame.rect) + (-45, -40, 15, 30)
    clip &= page.rect
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
    # fade the original so the overlay stands out
    img = (255 - (255 - img.astype(float)) * 0.45).astype(np.uint8)
    sx = pix.w / clip.width
    sy = pix.h / clip.height

    fig, ax = plt.subplots(figsize=(pix.w / 100, pix.h / 100), dpi=100)
    ax.imshow(img)
    ax.set_axis_off()
    for k, s in enumerate(series):
        col = COLORS[k % len(COLORS)]
        lab = f"{s.label} (conf {s.confidence:.2f}, n={len(s.points)})"
        if show_tracks and s.tracks:
            for t in s.tracks:
                P = np.asarray([p for st in t for p in st.pts])
                ax.plot((P[:, 0] - clip.x0) * sx, (P[:, 1] - clip.y0) * sy, "-", color=col, lw=1.2)
        if len(s.pdf_points):
            P = s.pdf_points
            ax.plot((P[:, 0] - clip.x0) * sx, (P[:, 1] - clip.y0) * sy, ".", color=col, ms=3.5,
                    label=lab)
        else:
            ax.plot([], [], ".", color=col, label=lab + " – NOT FOUND")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    ax.set_title(frame.title or "(untitled)", fontsize=9)
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def render_replot(frame, series, path: str):
    """Re-draw the plot from extracted data only, using the detected axes.

    If this looks like the datasheet plot, the extraction round-tripped.
    """
    xa, ya = frame.xaxis, frame.yaxis
    xt = [l.value for l in xa.labels]
    yt = [l.value for l in ya.labels]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
    if xa.scale == "log":
        ax.set_xscale("log")
    if ya.scale == "log":
        ax.set_yscale("log")
    for k, s in enumerate(series):
        if not len(s.points):
            continue
        col = COLORS[k % len(COLORS)]
        P = s.points
        ls = "--" if s.ambiguous_with else "-"
        lab = s.label + ("" if s.confidence >= 0.5 else f"  (low conf {s.confidence:.2f})")
        # break the line where the curve has a real hole, so gaps aren't bridged
        span = max(xt) - min(xt) if xa.scale == "linear" else None
        if span:
            breaks = np.nonzero(np.diff(P[:, 0]) > 0.08 * span)[0] + 1
            parts = np.split(P, breaks)
        else:
            parts = [P]
        for i, part in enumerate(parts):
            ax.plot(part[:, 0], part[:, 1], ls, color=col, lw=1.4, label=lab if i == 0 else None)
    ax.set_xlim(min(xt), max(xt))
    ax.set_ylim(min(yt), max(yt))
    ax.set_xticks(sorted(xt))
    ax.set_yticks(sorted(yt))
    ax.set_xticklabels([l.text.replace("−", "-") for l in sorted(xa.labels, key=lambda l: l.value)])
    ax.set_yticklabels([l.text.replace("−", "-") for l in sorted(ya.labels, key=lambda l: l.value)])
    ax.minorticks_off()
    ax.grid(True, ls=":", lw=0.6, color="0.6")
    ax.tick_params(labelsize=8)
    ax.set_xlabel(frame.xlabel, fontsize=8)
    ax.set_ylabel(frame.ylabel, fontsize=8)
    ax.set_title((frame.title or "(untitled)") + "\n[re-plotted from extracted data]", fontsize=9)
    if any(len(s.points) for s in series):
        ax.legend(fontsize=7, loc="best", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
