"""County map of the ReEDS 500 kV AC transmission cost field.

Colours every CONUS county by ``regional_cost.county_unit_cost_field`` -- the
per-county 500 kV AC line cost in 2004 USD/MW per great-circle km -- expressed as
a multiplier on the national median. This is model *input* data, so the figure
does not depend on a solved network.

Map conventions (EqualEarth projection, white zone boundaries, Times New Roman)
follow plot_network_maps.py. The colour ramp is a single hue light->dark, the
correct encoding for a magnitude field; a distribution strip sits on the colorbar
because the field is strongly right-skewed (~37% of counties sit at the floor).
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs as ccrs
from matplotlib.colors import LinearSegmentedColormap, Normalize

# regional_cost lives one level up, in workflow/scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regional_cost import (  # noqa: E402
    NEAR_NEIGHBOUR_PAIRS,
    REEDS_TRANSMISSION_COST_YEAR,
    REFERENCE_VOLTAGE_KV,
    county_unit_cost_field,
    load_transmission_pair_costs,
)

PAIR_COST_PATH = Path(
    r"D:\Research\sssc_project\pypsa-usa\workflow\repo_data\ReEDS_Constraints\transmission"
    r"\transmission_distance_cost_500kVac_county.csv"
)
COUNTY_SHAPES_PATH = Path(
    r"D:\Research\sssc_project\pypsa-usa\workflow\resources\shared\Geospatial\county_shapes.geojson"
)
STATE_SHAPES_PATH = Path(
    r"D:\Research\sssc_project\pypsa-usa\workflow\resources\shared\Geospatial\state_boundaries.geojson"
)
OUT_PATH = Path(
    r"D:\Research\sssc_project\pypsa-usa\workflow\results\test_tr\figure\county_transmission_cost_field.png"
)

# plot_network_maps.py conventions
TITLE_SIZE = 16
MAP_REGION_BOUNDARY_COLOR = "#ffffff"
NO_DATA_COLOR = "#EEEEEE"  # MAP_BACKGROUND_COLOR

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

# Sequential ramp: one hue, light -> dark. Never a rainbow for magnitude.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

# Territories the model never covers; dropping them keeps the extent tight.
NON_CONUS_STATE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}
CONUS_EXTENT = (-125.0, -66.5, 24.0, 49.5)

FIG_WIDTH = 13
FIG_HEIGHT = 7.6

plt.style.use("default")
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }
)


# --- data -------------------------------------------------------------------
field = county_unit_cost_field(load_transmission_pair_costs(PAIR_COST_PATH))
national_median = float(field.median())

counties = gpd.read_file(COUNTY_SHAPES_PATH)
counties = counties[~counties["STATEFP"].isin(NON_CONUS_STATE_FIPS)].copy()
counties["unit_cost"] = counties["GEOID"].map(field)

# Connecticut: ReEDS keys its post-2022 planning regions (p09110-p09190) while
# these census shapes carry the legacy county FIPS (p09001-p09015). Fill from the
# state median, exactly as transmission_unit_costs() step 3 does, so the map shows
# what the model actually uses rather than a hole.
state_median = field.groupby(field.index.str.slice(1, 3)).median()
filled = counties["unit_cost"].isna()
counties.loc[filled, "unit_cost"] = counties.loc[filled, "STATEFP"].map(state_median)
n_filled = int(filled.sum())

counties["multiplier"] = counties["unit_cost"] / national_median
values = counties["multiplier"].dropna()
vmin, vmax = float(values.min()), float(values.max())

cmap = LinearSegmentedColormap.from_list("reeds_blue", SEQUENTIAL_BLUE)
norm = Normalize(vmin=vmin, vmax=vmax)
states = gpd.read_file(STATE_SHAPES_PATH)


# --- figure -----------------------------------------------------------------
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
fig.patch.set_facecolor("white")

lon_center = 0.5 * (CONUS_EXTENT[0] + CONUS_EXTENT[1])
ax = fig.add_axes([0.02, 0.20, 0.96, 0.68], projection=ccrs.EqualEarth(lon_center))
ax.set_facecolor("white")
ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())

counties.plot(
    ax=ax, column="multiplier", cmap=cmap, norm=norm,
    transform=ccrs.PlateCarree(), aspect="equal",
    edgecolor="none", zorder=1,
    missing_kwds={"color": NO_DATA_COLOR, "edgecolor": "none"},
)
# State outlines in the map's own boundary white: geography without competing
# with the data layer.
states.plot(
    ax=ax, facecolor="none", edgecolor=MAP_REGION_BOUNDARY_COLOR,
    linewidth=0.6, transform=ccrs.PlateCarree(), aspect="equal", zorder=2,
)
ax.set_axis_off()

# Colorbar with a distribution strip: the field is strongly right-skewed, so the
# strip says where counties actually are on a scale the eye reads as uniform.
CB_LEFT, CB_WIDTH = 0.30, 0.40
CB_BOTTOM, CB_HEIGHT = 0.125, 0.022

hax = fig.add_axes([CB_LEFT, CB_BOTTOM + CB_HEIGHT, CB_WIDTH, 0.045])
counts, edges = np.histogram(values, bins=56, range=(vmin, vmax))
centers = 0.5 * (edges[:-1] + edges[1:])
hax.bar(
    centers, counts, width=(edges[1] - edges[0]),
    color=[cmap(norm(c)) for c in centers], linewidth=0,
)
hax.set_xlim(vmin, vmax)
hax.set_ylim(0, counts.max() * 1.08)
hax.set_axis_off()

cax = fig.add_axes([CB_LEFT, CB_BOTTOM, CB_WIDTH, CB_HEIGHT])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
cb.outline.set_visible(False)
cb.set_ticks([round(v, 2) for v in np.linspace(vmin, vmax, 6)])
cb.ax.tick_params(labelsize=11, length=3, color=INK_MUTED, labelcolor=INK_SECONDARY)
cb.set_label("Transmission cost multiplier  (× national median)", size=12,
             color=INK_PRIMARY, labelpad=8)

fig.text(0.5, 0.972, "County Transmission Cost Field", ha="center", va="top",
         fontsize=TITLE_SIZE, color=INK_PRIMARY)
fig.text(
    0.5, 0.936,
    f"500 kV AC line cost, national median = {national_median:,.0f} "
    f"USD{REEDS_TRANSMISSION_COST_YEAR}/MW per great-circle km",
    ha="center", va="top", fontsize=11.5, color=INK_SECONDARY,
)
fig.text(
    0.5, 0.052,
    f"Each county takes the median unit cost of its {NEAR_NEIGHBOUR_PAIRS} shortest incident "
    "ReEDS county pairs (transmission_distance_cost_500kVac_county).\n"
    f"Connecticut ({n_filled} counties) is filled from the state median: ReEDS keys its "
    "post-2022 planning regions, these census shapes use legacy FIPS.",
    ha="center", va="top", fontsize=9, color=INK_MUTED, linespacing=1.5,
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight", pad_inches=0.01, facecolor="white")
plt.close(fig)

print(
    f"{len(counties)} counties mapped ({n_filled} filled from the state median) | "
    f"multiplier {vmin:.2f}-{vmax:.2f} at {REFERENCE_VOLTAGE_KV:.0f} kV AC"
)
print(f"saved -> {OUT_PATH}")
