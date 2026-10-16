"""End-to-end test on a generated datasheet-style plot with known ground truth."""
import numpy as np

from plot_extractor import extract_pdf, interpolate
from plot_extractor.axes import parse_number
from plot_extractor.primitives import _thin_polygon_to_segs
from conftest import f_a, f_b, f_c


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
