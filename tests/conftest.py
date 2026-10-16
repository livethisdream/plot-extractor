"""Shared fixtures."""
import math

import numpy as np
import pytest


def f_a(x):
    return 14 + 4 * np.sin((x - 17) / 13 * math.pi)


def f_b(x):
    return 12 + 0.5 * (x - 17)


def f_c(x):
    return 2 + 10 * np.exp(-((x - 24) / 3) ** 2)


@pytest.fixture(scope="session")
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
