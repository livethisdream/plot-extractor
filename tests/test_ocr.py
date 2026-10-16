"""Plots whose labels are drawn as outlines (no PDF text layer) are read via OCR."""
import shutil

import numpy as np
import pytest

from plot_extractor import extract_pdf, interpolate

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


@pytest.fixture(scope="module")
def outlined_pdf(tmp_path_factory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    fp = FontProperties(family="DejaVu Sans")
    fig = plt.figure(figsize=(4, 3.2))
    ax = fig.add_axes([0.18, 0.18, 0.75, 0.72])
    x = np.linspace(0, 25, 200)
    ax.plot(x, -30 + 0.8 * x, color="tab:blue", lw=1.2)
    ax.plot(x, -45 + 1.2 * x, color="tab:red", lw=1.2)
    ax.set_xlim(0, 25)
    ax.set_ylim(-50, 0)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks(range(-50, 1, 10))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="0.7", lw=0.5)

    def outline(text, x_fig, y_fig, size=7, ha="center"):
        tp = TextPath((0, 0), text, size=size, prop=fp)
        bb = tp.get_extents()
        dx = {"center": -bb.width / 2, "right": -bb.width, "left": 0}[ha]
        tr = Affine2D().translate(dx, -bb.height / 2).scale(1 / 72) \
            .scale(1 / fig.get_figwidth(), 1 / fig.get_figheight()).translate(x_fig, y_fig)
        fig.add_artist(PathPatch(tr.transform_path(tp), color="black", lw=0, transform=fig.transFigure))

    to_fig = lambda xd, yd: fig.transFigure.inverted().transform(ax.transData.transform((xd, yd)))
    for v in range(0, 26, 5):
        fx, fy = to_fig(v, -50)
        outline(str(v), fx, fy - 0.04)
    for v in range(-50, 1, 10):
        fx, fy = to_fig(0, v)
        outline(str(v).replace("-", "−"), fx - 0.015, fy, ha="right")
    for k, (lab, col) in enumerate([("RF1", "tab:blue"), ("RF2", "tab:red")]):
        fx, fy = to_fig(3, -6 - 5 * k)
        ax.plot([1, 2.5], [-6 - 5 * k] * 2, color=col, lw=1.2)
        outline(lab, fx, fy, ha="left")

    path = tmp_path_factory.mktemp("ocr") / "outlined.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def test_outlined_labels_are_read(outlined_pdf):
    import pymupdf
    page = pymupdf.open(outlined_pdf)[0]
    assert not any(w[4].lstrip("−-").isdigit() for w in page.get_text("words"))  # really no text

    res = extract_pdf(outlined_pdf)
    assert len(res.plots) == 1
    p = res.plots[0]
    assert p.frame.ocr
    assert [l.value for l in p.frame.yaxis.labels] == [0, -10, -20, -30, -40, -50]
    s = {x.label: x for x in p.series}
    assert set(s) == {"RF1", "RF2"}
    xs = np.linspace(4, 24, 9)
    assert np.abs(interpolate(s["RF1"], xs) - (-30 + 0.8 * xs)).max() < 0.5
    assert np.abs(interpolate(s["RF2"], xs) - (-45 + 1.2 * xs)).max() < 0.5
