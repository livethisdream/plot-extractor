"""Regression test against the HMC576 datasheet (ADI).

The PDF isn't committed. Put it at tests/data/hmc576.pdf to enable this test
(download it from analog.com, product page HMC576).
"""
from pathlib import Path

import numpy as np
import pytest

from plot_extractor import extract_pdf, interpolate

PDF = Path(__file__).parent / "data" / "hmc576.pdf"
pytestmark = pytest.mark.skipif(not PDF.exists(), reason="tests/data/hmc576.pdf not present")


@pytest.fixture(scope="module")
def plots():
    res = extract_pdf(PDF)
    return {p.frame.title: {s.label: s for s in p.series} for p in res.plots}


def test_all_seven_plots(plots):
    assert len(plots) == 7
    assert "Isolation @ 3 dBm Drive Level" in plots


def test_return_loss_minima(plots):
    irl = plots["Input Return Loss vs. Temperature"]["+25C"].points
    x, y = irl[np.argmin(irl[:, 1])]
    assert abs(x - 10.5) < 0.1 and abs(y + 27.7) < 0.3
    orl = plots["Output Return Loss vs. Temperature"]["+25C"].points
    x, y = orl[np.argmin(orl[:, 1])]
    assert abs(x - 22.0) < 0.2 and abs(y + 13.4) < 0.3


def test_temperature_series(plots):
    t = plots["Output Power vs. Temperature @ 3 dBm Drive Level"]
    assert set(t) == {"+25C", "+85C", "-55C"}
    assert t["+25C"].confidence > 0.9
    v = interpolate(t["+25C"], [24.0])[0]
    assert abs(v - 18.2) < 0.3
    # -55C runs highest around 27 GHz
    assert interpolate(t["-55C"], [27.0])[0] > interpolate(t["+25C"], [27.0])[0] + 1.0


def test_identical_legend_styles_flagged(plots):
    d = plots["Output Power vs. Drive Level"]
    amb = [s for s in d.values() if s.ambiguous_with]
    assert len(amb) == 2
    assert set(amb[0].ambiguous_with) == {"-6dBm", "0dBm"}
