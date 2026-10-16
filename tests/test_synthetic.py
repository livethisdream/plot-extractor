"""End-to-end test on a generated datasheet-style plot with known ground truth."""
import math

import numpy as np
import pytest

from plot_extractor import extract_pdf, interpolate
from plot_extractor.axes import parse_number
from plot_extractor.primitives import _thin_polygon_to_segs


def f_a(x):
    return 14 + 4 * np.sin((x - 17) / 13 * math.pi)


def f_b(x):
    return 12 + 0.5 * (x - 17)


def f_c(x):
    return 2 + 10 * np.exp(-((x - 24) / 3) ** 2)


@pytest.fixture(scope="module")
def synthetic_pdf(tmp_path_factory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42

    x = np.linspace(17, 30, 60)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    ax = axes[0]
    ax.plot(x, f_a(x), "k-", lw=1, label="+25C")
    ax.plot(x, f_b(x), "k--", lw=1, label="+85C")
    ax.plot(x, f_c(x), "k:", lw=1, label="-55C")
    ax.set_xlim(17, 30)
    ax.set_ylim(0, 20)
    ax.set_xticks(range(17, 31))
    ax.set_yticks(range(0, 21, 2))
    ax.grid(True, ls=":", lw=0.4, color="0.5")
    ax.legend(loc="lower center", fontsize=6)
    ax.set_title("Output Power vs. Temperature", fontsize=8)
    ax.set_xlabel("OUTPUT FREQUENCY (GHz)", fontsize=6)
    ax.set_ylabel("OUTPUT POWER (dBm)", fontsize=6)
    ax.tick_params(labelsize=6)

    # log-x plot, colored series
    ax = axes[1]
    f = np.logspace(1, 4, 50)
    ax.semilogx(f, -20 + 5 * np.log10(f), color="tab:red", label="VDD=3.3V")
    ax.semilogx(f, -25 + 5 * np.log10(f), color="tab:blue", label="VDD=5V")
    ax.set_xlim(10, 10000)
    ax.set_ylim(-30, 0)
    ax.set_yticks(range(-30, 1, 5))
    ax.grid(True, which="major", ls=":", lw=0.4, color="0.5")
    ax.legend(loc="upper left", fontsize=6)
    ax.set_title("Noise vs. Frequency", fontsize=8)
    ax.tick_params(labelsize=6)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    fig.tight_layout()
    path = tmp_path_factory.mktemp("pdf") / "synthetic.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def test_finds_both_plots(synthetic_pdf):
    res = extract_pdf(synthetic_pdf)
    titles = [p.frame.title for p in res.plots]
    assert "Output Power vs. Temperature" in titles
    assert "Noise vs. Frequency" in titles


def test_linear_plot_values(synthetic_pdf):
    res = extract_pdf(synthetic_pdf)
    p = next(p for p in res.plots if p.frame.title.startswith("Output Power"))
    assert p.frame.xaxis.scale == "linear" and p.frame.yaxis.scale == "linear"
    s = {x.label: x for x in p.series}
    assert set(s) == {"+25C", "+85C", "-55C"}
    xs = np.linspace(17.5, 29.5, 25)
    for lab, fn in [("+25C", f_a), ("+85C", f_b), ("-55C", f_c)]:
        err = np.abs(interpolate(s[lab], xs) - fn(xs))
        assert err.max() < 0.15, (lab, err.max())  # 20 dB span -> <0.75 %


def test_log_axis_and_color(synthetic_pdf):
    res = extract_pdf(synthetic_pdf)
    p = next(p for p in res.plots if p.frame.title.startswith("Noise"))
    assert p.frame.xaxis.scale == "log"
    s = {x.label: x for x in p.series}
    assert set(s) == {"VDD=3.3V", "VDD=5V"}
    for lab, off in [("VDD=3.3V", -20), ("VDD=5V", -25)]:
        P = s[lab].points
        err = np.abs(P[:, 1] - (off + 5 * np.log10(P[:, 0])))
        assert err.max() < 0.3, (lab, err.max())


def test_parse_number():
    assert parse_number("−5") == -5
    assert parse_number("10k") == 10e3
    assert parse_number("2.5") == 2.5
    assert parse_number("1E-3") == 1e-3
    assert parse_number("dBm") is None


def test_square_pen_sweep_centerline():
    # a segment from (0,0) to (10,5) drawn as the sweep of a 1x1 square pen
    pts = [(-0.5, -0.5), (0.5, -0.5), (10.5, 4.5), (10.5, 5.5), (9.5, 5.5), (-0.5, 0.5)]
    segs = _thin_polygon_to_segs(pts, (0, 0, 0))
    assert len(segs) == 1
    s = segs[0]
    ends = sorted([s.p0, s.p1])
    assert np.allclose(ends[0], (0, 0), atol=1e-6) and np.allclose(ends[1], (10, 5), atol=1e-6)
    assert abs(s.width - 1.0) < 1e-6
