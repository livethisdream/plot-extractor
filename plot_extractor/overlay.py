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
