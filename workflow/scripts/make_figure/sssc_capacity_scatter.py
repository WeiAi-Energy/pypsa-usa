"""Subplot (a) of AC_add_dis: optimal SSSC capacity vs optimal AC line capacity.

Reads the solved network directly (LineX component carries `s_nom_opt` and
`sssc_nom_opt`) instead of the post-processed statistics/lines.csv.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pypsa

NETWORK_PATH = Path(
    r"D:\Research\sssc_project\pypsa-usa\workflow\results\test_tr\network\network.nc"
)
OUT_PATH = Path(
    r"D:\Research\sssc_project\pypsa-usa\workflow\results\test_tr\figure\sssc_capacity_scatter.png"
)

COLOR_SSSC = "#374151"
FIG_WIDTH = 1.75
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


def load_lines(network_path: Path):
    n = pypsa.Network(str(network_path))
    df = n.line_xs if len(n.lines) == 0 else n.lines
    df = df.copy()
    for column in ["s_nom", "s_nom_opt", "sssc_nom_opt", "length"]:
        if column in df.columns:
            df[column] = df[column].astype(float)
    df["capacity_add"] = df["s_nom_opt"] - df["s_nom"]
    return df


lines = load_lines(NETWORK_PATH)
if "carrier" in lines.columns:
    lines = lines.loc[lines["carrier"].eq("AC")]

sssc_df = lines.loc[(lines["s_nom_opt"] > 0) & (lines["sssc_nom_opt"] > 0)]
print(f"{len(lines)} AC lines, {len(sssc_df)} with SSSC installed")

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)

ax.scatter(
    sssc_df["s_nom_opt"] / 1000.0,
    sssc_df["sssc_nom_opt"] / 1000.0,
    s=18,
    color=COLOR_SSSC,
    edgecolors="#333333",
    linewidths=0,
    alpha=0.9,
    marker="o",
    zorder=3,
)
ax.set_xlabel("Optimal AC line capacity (GW)")
ax.set_ylabel("Optimal SSSC capacity (GVAr)")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_axisbelow(True)
ax.grid(True, which="major", linestyle=":", linewidth=0.3, color="#D0D0D0")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(length=2, width=0.6)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight", pad_inches=0.01)
print(f"saved -> {OUT_PATH}")
