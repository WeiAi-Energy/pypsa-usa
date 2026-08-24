"""Distribution of the SSSC-to-line capacity ratio (sssc_nom_opt / s_nom_opt).

Panel (a): histogram of the ratio on a log axis with the cumulative share.
Panel (b): ratio against optimal AC line capacity.
Style follows AC_add_dis.ipynb.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pypsa

NETWORK_PATH = Path(
    r"D:\Research\sssc_project\pypsa-usa\workflow\results\test_tr\network\network.nc"
)
OUT_PATH = Path(
    r"D:\Research\sssc_project\pypsa-usa\workflow\results\test_tr\figure\sssc_ratio_dist.png"
)

COLOR = "#374151"
COLOR_CUM = "#008080"
FIG_WIDTH = 3.5
FIG_HEIGHT = 1.8

plt.style.use("default")
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 6,
        "axes.labelsize": 7,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.linewidth": 0.3,
        "grid.color": "#B0B0B0",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }
)

n = pypsa.Network(str(NETWORK_PATH))
df = n.line_xs if len(n.lines) == 0 else n.lines
df = df.loc[df["carrier"].eq("AC")].copy()
df = df.loc[(df["s_nom_opt"] > 0) & (df["sssc_nom_opt"] > 0)]
df["ratio"] = df["sssc_nom_opt"] / df["s_nom_opt"]

ratio_pct = df["ratio"] * 100.0
print(f"n = {len(df)}, median = {ratio_pct.median():.2f}%, mean = {ratio_pct.mean():.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)

hist_ax, scatter_ax = axes

bins = np.logspace(-4, 2, 37)
hist_ax.hist(ratio_pct, bins=bins, color=COLOR, alpha=0.9, edgecolor="none", zorder=3)
hist_ax.set_xscale("log")
hist_ax.set_xlabel("SSSC / line capacity (%)")
hist_ax.set_ylabel("Number of AC lines")

cum_ax = hist_ax.twinx()
ordered = np.sort(ratio_pct.values)
cum_ax.plot(
    ordered,
    np.arange(1, len(ordered) + 1) / len(ordered) * 100.0,
    color=COLOR_CUM,
    linewidth=0.8,
    zorder=4,
)
cum_ax.set_ylim(0, 100)
cum_ax.set_ylabel("Cumulative share (%)", color=COLOR_CUM)
cum_ax.tick_params(axis="y", colors=COLOR_CUM, length=2, width=0.6)
cum_ax.spines["top"].set_visible(False)
cum_ax.spines["right"].set_color(COLOR_CUM)

scatter_ax.scatter(
    df["s_nom_opt"] / 1000.0,
    ratio_pct,
    s=3,
    color=COLOR,
    linewidths=0,
    alpha=0.35,
    marker="o",
    zorder=3,
)
scatter_ax.set_yscale("log")
scatter_ax.set_ylim(1e-4, 200)
scatter_ax.set_xlim(left=0)
scatter_ax.set_xlabel("Optimal AC line capacity (GW)")
scatter_ax.set_ylabel("SSSC / line capacity (%)")

for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle=":", linewidth=0.3, color="#D0D0D0")
    ax.spines["top"].set_visible(False)
    ax.tick_params(length=2, width=0.6)
scatter_ax.spines["right"].set_visible(False)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight", pad_inches=0.01)
print(f"saved -> {OUT_PATH}")
