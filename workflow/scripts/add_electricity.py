# PyPSA USA Authors
"""
Adds existing conventional generators, renewable generators, and storage devices to the network.

This script will add all generator unit availabilities (capacity-factors) to the network, for all investment horizons.
"""

import logging
import os

import constants as const
import dill as pickle
import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import (
    LINK_FIXED_COST_COL,
    LINK_UNIT_COST_COL,
    calculate_annuity,
    configure_logging,
    export_network_for_gis_mapping,
    get_currency_conversion_factor,
    get_multiindex_snapshots,
    recompute_link_transmission_costs,
    update_p_nom_max,
    weighted_avg,
)
from regional_cost import (
    REEDS_TRANSMISSION_COST_YEAR,
    county_unit_cost_field,
    dc_ac_line_cost_ratio,
    load_transmission_basecost,
    load_transmission_pair_costs,
    pair_unit_costs,
    renewable_interconnection_capital_cost,
    transmission_unit_costs,
    voltage_cost_anchors,
    voltage_cost_exponent,
    voltage_ratio,
)
from select_representative_periods import (
    read_representative_snapshots,
    reindex_calendar_timeseries_to_snapshots,
    reindex_source_timeseries_to_snapshots,
)
from sklearn.neighbors import BallTree

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def sanitize_carriers(n, config):
    """
    Sanitize the carrier information in a PyPSA Network object.

    The function ensures that all unique carrier names are present in the network's
    carriers attribute, and adds nice names and colors for each carrier according
    to the provided configuration dictionary.

    Parameters
    ----------
    n : pypsa.Network
        A PyPSA Network object that represents an electrical power system.
    config : dict
        A dictionary containing configuration information, specifically the
        "plotting" key with "nice_names" and "tech_colors" keys for carriers.

    Returns
    -------
    None
        The function modifies the 'n' PyPSA Network object in-place, updating the
        carriers attribute with nice names and colors.

    Warnings
    --------
    Raises a warning if any carrier's "tech_colors" are not defined in the config dictionary.
    """
    for c in n.iterate_components():
        if "carrier" in c.df:
            add_missing_carriers(n, c.df.carrier)

    carrier_i = n.carriers.index
    nice_names = (
        pd.Series(config["plotting"]["nice_names"]).reindex(carrier_i).fillna(carrier_i.to_series().str.title())
    )
    n.carriers["nice_name"] = n.carriers.nice_name.where(
        n.carriers.nice_name != "",
        nice_names,
    )
    colors = pd.Series(config["plotting"]["tech_colors"]).reindex(carrier_i)
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(f"tech_colors for carriers {missing_i} not defined in config.")
    n.carriers["color"] = n.carriers.color.where(n.carriers.color != "", colors)


def add_missing_carriers(n, carriers):
    """Function to add missing carriers to the network without raising errors."""
    missing_carriers = set(carriers) - set(n.carriers.index)
    if len(missing_carriers) > 0:
        n.madd("Carrier", missing_carriers)


def _annualization_factor(costs: pd.DataFrame, technology: str) -> float:
    """Annuity plus FOM share, applied to a per-line overnight capex.

    FOM for transmission is a fixed percentage of capex, so once capex varies by
    line the FOM has to vary with it -- hence one combined factor rather than the
    single pre-annualized ``annualized_capex_per_mw_km_fom`` figure this used to
    read out of the cost table.
    """
    needed = ["cost_recovery_period_years", "wacc_real", "opex_fixed_pct_of_capex"]
    values = {}
    for parameter in needed:
        try:
            value = float(costs.at[technology, parameter])
        except KeyError as exc:
            raise KeyError(f"Cost table has no {parameter!r} for {technology!r}.") from exc
        # A blank cell pivots to NaN, which would silently make every line free
        # rather than failing, so it is rejected here instead.
        if not np.isfinite(value):
            raise ValueError(f"Cost table has no finite {parameter!r} for {technology!r}.")
        values[parameter] = value

    annuity = calculate_annuity(values["cost_recovery_period_years"], values["wacc_real"])
    return float(annuity + values["opex_fixed_pct_of_capex"])


def update_transmission_costs(
    n,
    costs,
    length_factor: float = 1.0,
    *,
    distance_cost_fn: str,
    basecost_fn: str,
):
    """Price AC lines and DC links by voltage class and by route region.

    Both components are costed per **great-circle km**: ``n.lines.length`` already
    carries ``length_factor`` from ``build_base_network.assign_line_length``, so it
    is divided back out rather than multiplied again, and the ReEDS unit costs
    already embed route detour in the price (their ``length_miles`` is a
    centroid-to-centroid great-circle distance). Applying ``length_factor`` twice --
    as this did, giving AC lines 1.25^2 while links got 1.25 -- double counts it.

    ``capital_cost`` stays exactly proportional to ``length``, which is what makes
    the three aggregation passes downstream preserve it: parallel merging averages
    it by capacity, series merging sums cost and length together, and pypsa's
    ``length_capacity_weighted_average`` rebuilds it as
    ``new_length * capacity-weighted mean unit cost``.

    Links are handled by :func:`recompute_link_transmission_costs` off two stored
    columns instead, because their endpoints move under aggregation while pypsa
    does not rescale their cost.
    """
    basecost = load_transmission_basecost(basecost_fn)
    voltage_exponent = voltage_cost_exponent(voltage_cost_anchors(basecost))
    pair_table = load_transmission_pair_costs(distance_cost_fn)
    pair_costs = pair_unit_costs(pair_table)
    field = county_unit_cost_field(pair_table)

    # ReEDS quotes both tables in 2004 USD; the rest of the workflow is 2022 USD.
    to_usd2022 = get_currency_conversion_factor(REEDS_TRANSMISSION_COST_YEAR, "USD")
    bus_county = n.buses.get("county")
    if bus_county is None:
        raise ValueError(
            "n.buses has no 'county' column; transmission costs need it to resolve the "
            "route region. It is assigned in build_base_network and dropped later in "
            "simplify_network, so update_transmission_costs must run before that.",
        )

    def unit_cost_usd2022(branches):
        return (
            transmission_unit_costs(
                bus_county.reindex(branches["bus0"]).set_axis(branches.index),
                bus_county.reindex(branches["bus1"]).set_axis(branches.index),
                pair_costs,
                field,
            )
            * to_usd2022
        )

    if not n.lines.empty:
        ac_factor = _annualization_factor(costs, "HVAC overhead")
        n.lines["capital_cost"] = (
            unit_cost_usd2022(n.lines)
            * voltage_ratio(n.lines["v_nom"], voltage_exponent)
            * ac_factor
            * (n.lines["length"] / length_factor)
        )
        per_km = n.lines["capital_cost"] / n.lines["length"].where(n.lines["length"] > 0)
        logger.info(
            "AC line annualized cost per network-km: min=%.0f median=%.0f max=%.0f USD2022/MW/km.",
            per_km.min(),
            per_km.median(),
            per_km.max(),
        )

    if n.links.empty:
        return

    transmission = n.links.index[n.links.carrier.isin(["AC", "DC"])]
    if transmission.empty:
        return

    links = n.links.loc[transmission]
    dc_ratio = dc_ac_line_cost_ratio(basecost)
    unit = unit_cost_usd2022(links)

    # DC corridors are 500 kV bipoles, so the AC voltage ratio is 1.0 by
    # construction and only the DC/AC line ratio applies. An AC-carrier transport
    # link is priced as a 500 kV AC line with no converters.
    is_dc = links.carrier == "DC"
    line_ratio = pd.Series(1.0, index=transmission).where(~is_dc, dc_ratio)
    factor = pd.Series(_annualization_factor(costs, "HVAC overhead"), index=transmission).where(
        ~is_dc,
        _annualization_factor(costs, "HVDC overhead"),
    )

    n.links[LINK_UNIT_COST_COL] = 0.0
    n.links[LINK_FIXED_COST_COL] = 0.0
    n.links.loc[transmission, LINK_UNIT_COST_COL] = unit * line_ratio * factor
    n.links.loc[transmission, LINK_FIXED_COST_COL] = np.where(
        is_dc,
        costs.at["HVDC inverter pair", "annualized_capex_fom"],
        0.0,
    )
    recompute_link_transmission_costs(n)


def load_powerplants(
    plants_fn,
    investment_periods: list[int],
    interconnect: str | None = None,
) -> pd.DataFrame:
    plants = pd.read_csv(
        plants_fn,
    )
    plants = plants.set_index("generator_name")

    # Convert date columns to datetime
    plants["current_planned_generator_operating_date"] = pd.to_datetime(
        plants["current_planned_generator_operating_date"],
    )

    plants["generator_retirement_date"] = pd.to_datetime(
        plants["generator_retirement_date"],
    )

    # if operational_status is proposed replace build_year with year of current_planned_generator_operating_date
    plants.loc[plants.operational_status == "proposed", "build_year"] = plants.loc[
        plants.operational_status == "proposed",
        "current_planned_generator_operating_date",
    ].dt.year

    # If operational_status is existing or proposed, replace generator_retirement_date with 1/1/2100
    retirement_date = pd.to_datetime("2100-01-01")
    plants.loc[plants.operational_status.isin(["existing", "proposed"]), "generator_retirement_date"] = retirement_date

    # Handle NaT values
    plants.loc[plants.generator_retirement_date.isna(), "generator_retirement_date"] = pd.to_datetime("1900-01-01")

    # Filter out plants that are not built by first investment period and retired before the first investment period.
    plants = plants[plants.build_year <= investment_periods[0]]
    plants = plants[plants.generator_retirement_date.dt.year > investment_periods[0]]

    # Filter out non-conus plants
    plants = plants[plants.nerc_region != "non-conus"]
    if (interconnect is not None) & (interconnect != "usa"):
        plants["interconnection"] = plants["nerc_region"].map(const.NERC_REGION_MAPPER)
        plants = plants[plants.interconnection == interconnect]
    return plants


def match_nearest_bus(plants_subset, buses_subset):
    """Assign the nearest bus to each plant in the given subsets."""
    if plants_subset.empty or buses_subset.empty:
        return plants_subset

    # Create a BallTree for the given subset of buses
    tree = BallTree(buses_subset[["x", "y"]].values, leaf_size=2)

    # Find nearest bus for each plant in the subset
    distances, indices = tree.query(
        plants_subset[["longitude", "latitude"]].values,
        k=1,
    )

    # Map the nearest bus information back to the plants subset
    plants_subset["bus_assignment"] = buses_subset.index.to_numpy()[indices.flatten()]
    plants_subset["distance_nearest"] = distances.flatten()

    return plants_subset


def match_plant_to_bus(n, plants):
    """
    Matches each plant to it's corresponding bus in the network, keeping the
    match inside the region the plant was located in.

    Efficient matching taken from:
    https://stackoverflow.com/questions/58893719/find-nearest-point-in-other-dataframe-with-a-lot-of-data
    """
    plants_matched = plants.copy()
    plants_matched["bus_assignment"] = None
    plants_matched["distance_nearest"] = None

    buses = n.buses.copy()

    # First pass: the raw bus-region GeoJSON is named by substation ID, while
    # the unsimplified network is indexed by individual bus ID.  Select one
    # existing bus per substation so a plant remains inside the region that
    # owns its ground.  Once transformers/substations are collapsed downstream,
    # all such representatives map to the same final bus.
    n_in_region = 0
    region_bus = plants_matched.get("name")
    if region_bus is not None:
        if "sub_id" in buses.columns:
            sub_to_bus = (
                pd.DataFrame(
                    {
                        "sub_id": normalize_bus_keys(buses.sub_id).to_numpy(),
                        "bus_id": buses.index.to_numpy(),
                    },
                )
                .drop_duplicates("sub_id")
                .set_index("sub_id")
                .bus_id
            )
            assigned = normalize_bus_keys(region_bus).map(sub_to_bus)
        else:
            assigned = region_bus.astype(str)
        in_region = assigned.isin(buses.index).to_numpy()
        n_in_region = int(in_region.sum())
        assigned = assigned[in_region]
        plants_matched.loc[in_region, "bus_assignment"] = assigned.to_numpy()
        plants_matched.loc[in_region, "distance_nearest"] = np.hypot(
            plants_matched.loc[in_region, "longitude"].to_numpy(dtype=float) - buses.loc[assigned.to_numpy(), "x"].to_numpy(),
            plants_matched.loc[in_region, "latitude"].to_numpy(dtype=float) - buses.loc[assigned.to_numpy(), "y"].to_numpy(),
        )

    # Second pass: use a county/FIPS constraint whenever the source provides
    # one.  Existing PHS has plant coordinates and county FIPS, while buses
    # carry the same `p`-prefixed FIPS code in their `county` field.
    n_in_county = 0
    if "county" in plants_matched.columns and "county" in buses.columns:
        county = plants_matched.county.astype(str).str.strip()
        available_counties = set(buses.county.dropna().astype(str))
        for county_id in county[county.isin(available_counties)].unique():
            buses_in_county = buses[buses.county.astype(str) == county_id]
            plants_in_county = plants_matched[
                (county == county_id) & plants_matched.bus_assignment.isnull()
            ].copy()
            matched = match_nearest_bus(plants_in_county, buses_in_county)
            n_in_county += len(matched)
            plants_matched.update(matched)

    # Third pass: inputs that arrive with a ReEDS zone label instead of a
    # region -- PHS uses this as a fallback when its county has no bus.
    n_in_zone = 0
    if "country" in plants_matched.columns:
        for zone_id in buses["reeds_zone"].unique():
            buses_in_zone = buses[buses["reeds_zone"] == zone_id]
            plants_in_zone = plants_matched[
                (plants_matched["country"] == zone_id) & (plants_matched["bus_assignment"].isnull())
            ].copy()

            matched = match_nearest_bus(plants_in_zone, buses_in_zone)
            n_in_zone += len(matched)
            plants_matched.update(matched)

    # Final pass: whatever is left -- offshore units pulled in by the ReEDS shape
    # fallback, mostly -- goes to the nearest bus regardless of region or zone.
    unmatched_plants = plants_matched[plants_matched["bus_assignment"].isnull()].copy()
    if not unmatched_plants.empty:
        plants_matched.update(match_nearest_bus(unmatched_plants, buses))

    logger.info(
        "Matched %s of %s plants to the bus owning their region, %s to the nearest bus in their county, %s to the nearest bus in their zone, %s to the nearest bus anywhere.",
        n_in_region,
        len(plants_matched),
        n_in_county,
        n_in_zone,
        len(unmatched_plants),
    )

    return plants_matched


def filter_plants_by_region(
    plants: pd.DataFrame,
    regions_onshore: gpd.GeoDataFrame,
    regions_offshore: gpd.GeoDataFrame,
    reeds_shapes: gpd.GeoDataFrame,
    all_reeds_shapes: gpd.GeoDataFrame,
    reeds_memberships: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filters the plants dataframe to remove plants not within the onshore and
    offshore geometries.
    """
    plants = plants.copy()
    plants["geometry"] = gpd.points_from_xy(
        plants.longitude,
        plants.latitude,
        crs="EPSG:4326",
    )
    gdf_plants = gpd.GeoDataFrame(plants, geometry="geometry")
    plants_onshore = gpd.sjoin(gdf_plants, regions_onshore, how="inner")
    plants_offshore = gpd.sjoin(gdf_plants, regions_offshore, how="inner")
    if not plants_offshore.empty:
        logger.warning(f"Offshore plants: {plants_offshore}")
    plants_filt = pd.concat([plants_onshore, plants_offshore])

    # Some plants like Diablo Canyon near oceans don't have region due to
    # imprecise ReEDS Shapes. We filter plants that have no reeds regions,
    # then search these points again.
    plants_in_regions = gpd.sjoin(
        gdf_plants,
        reeds_shapes,
        how="inner",
        predicate="intersects",
    )
    plants_no_region = gdf_plants[~gdf_plants.index.isin(plants_in_regions.index)]
    if not plants_no_region.empty:
        # identify the plants for which the interconnection according to the reeds membership is different than the interconnection according to the EIA data. We need to include these plants since these are plants where the reeds shapes are not precise enough to assign a region.
        plants_no_region = plants_no_region.to_crs(epsg=3857)
        plants_no_region_all_shapes = gpd.sjoin(
            plants_no_region.reset_index(),
            all_reeds_shapes,
            how="inner",
            predicate="intersects",
        )
        plants_no_region_all_shapes = plants_no_region_all_shapes.to_crs(epsg=4326)
        reeds_memberships.loc[reeds_memberships.interconnect == "ercot", "interconnect"] = "texas"
        plants_no_region_all_shapes = plants_no_region_all_shapes.merge(
            reeds_memberships[["ba", "interconnect"]],
            left_on="rb",
            right_on="ba",
            how="left",
        )
        # Handles non-US wide interconnection cases (western, eastern, texas)
        if "interconnection" in plants_no_region_all_shapes.columns:
            plants_must_add = plants_no_region_all_shapes[
                plants_no_region_all_shapes.interconnect != plants_no_region_all_shapes.interconnection
            ]
            remaining_plants = plants_no_region_all_shapes[
                plants_no_region_all_shapes.interconnect == plants_no_region_all_shapes.interconnection
            ]
        # Handles US wide interconnection cases
        else:
            plants_must_add = plants_no_region_all_shapes
            remaining_plants = pd.DataFrame()
        plants_must_add.set_index("generator_name", inplace=True)

        if not remaining_plants.empty:
            remaining_clean = remaining_plants.drop(columns=["index_right"], errors="ignore")
            plants_nearshore = gpd.sjoin_nearest(
                remaining_clean,
                regions_onshore.to_crs(epsg=3857),
                how="inner",
                max_distance=2000,
                distance_col="distance",
            )
            plants_nearshore = plants_nearshore.to_crs(epsg=4326)
            plants_filt = pd.concat([plants_filt, plants_nearshore, plants_must_add])
        else:
            plants_filt = pd.concat([plants_filt, plants_must_add])

    plants_filt = plants_filt.drop(columns=["geometry"])
    plants_filt = plants_filt[~plants_filt.index.duplicated()]
    return pd.DataFrame(plants_filt)


def attach_existing_renewable_capacities(
    n: pypsa.Network,
    plants_df: pd.DataFrame,
    renewable_carriers: list,
    costs: pd.DataFrame,
):
    plants = plants_df.query(
        "bus_assignment in @n.buses.index",
    )
    planning_horizon = n.investment_periods[0]
    for tech in renewable_carriers:
        plants_filt = plants.query("carrier == @tech").copy()
        if plants_filt.empty:
            continue

        # Existing plants whose technical lifetime ends at or before the planning
        # horizon won't survive to it, so they shouldn't count as capacity the
        # model must keep -- only carry forward those still operating past it.
        lifetime = costs.at[tech, "lifetime"]
        plants_filt = plants_filt[plants_filt.build_year + lifetime > planning_horizon]
        if plants_filt.empty:
            continue

        # `simplify_network` aggregates the grid onto its substations, so the bus
        # index *is* the substation id from that point on and the separate
        # `sub_id` column is gone. Fall back to the bus itself so existing
        # capacity still lands on the same key the generators carry.
        sub_of_bus = (
            n.buses["sub_id"]
            if "sub_id" in n.buses.columns
            else pd.Series(n.buses.index, index=n.buses.index)
        )

        generators_tech = n.generators[n.generators.carrier == tech].copy()
        generators_tech["sub_assignment"] = generators_tech.bus.map(sub_of_bus)
        plants_filt["sub_assignment"] = plants_filt.bus_assignment.map(sub_of_bus)

        build_year_avg = plants_filt.groupby(["sub_assignment"])[plants_filt.columns].apply(
            lambda x: pd.Series(
                {field: weighted_avg(x, field, "p_nom") for field in ["build_year"]},
            ),
        )

        caps_per_bus = (
            plants_filt[["sub_assignment", "p_nom"]].groupby("sub_assignment").sum().p_nom
        )  # namplate capacity per sub_id

        if caps_per_bus[~caps_per_bus.index.isin(generators_tech.sub_assignment)].sum() > 0:
            # p_all = plants_filt[["sub_assignment", "p_nom", "latitude", "longitude"]]
            # missing_plants = p_all[~p_all.sub_assignment.isin(generators_tech.sub_assignment)]
            missing_capacity = caps_per_bus[~caps_per_bus.index.isin(generators_tech.sub_assignment)].sum()
            # missing_plants.to_csv(f"missing_{tech}_plants.csv",)

            logger.info(
                f"There are {np.round(missing_capacity / 1000, 4)} GW of {tech} plants that are not in the network. See git issue #16.",
            )

        logger.info(
            f"{np.round(caps_per_bus.sum() / 1000, 2)} GW of {tech} capacity added.",
        )
        mapped_values = generators_tech.sub_assignment.map(caps_per_bus).dropna()
        n.generators.loc[mapped_values.index, "p_nom"] = mapped_values
        n.generators.loc[mapped_values.index, "p_nom_min"] = mapped_values
        # Installed capacity can exceed the atlite-derived resource potential
        # (data mismatch/clustering artifacts); widen p_nom_max to match so the
        # generator stays feasible (p_nom_min must never exceed p_nom_max).
        n.generators.loc[mapped_values.index, "p_nom_max"] = n.generators.loc[
            mapped_values.index,
            "p_nom_max",
        ].clip(lower=mapped_values)
        mapped_values = generators_tech.sub_assignment.map(
            build_year_avg.build_year,
        ).dropna()
        n.generators.loc[mapped_values.index, "build_year"] = mapped_values.astype(int)


def attach_conventional_generators(
    n: pypsa.Network,
    costs: pd.DataFrame,
    plants: pd.DataFrame,
    conventional_carriers: list,
    extendable_carriers: list,
    conventional_params,
    renewable_carriers: list,
    conventional_inputs,
    unit_commitment=None,
    fuel_price=None,
):
    carriers = [
        carrier
        for carrier in set(conventional_carriers) | set(extendable_carriers["Generator"])
        if carrier not in renewable_carriers
    ]
    add_missing_carriers(n, carriers)

    plants = (
        plants.query("carrier in @carriers")
        .join(costs, on="carrier", rsuffix="_r")
        .rename(index=lambda s: "C" + str(s))
    )

    plants["efficiency"] = plants.efficiency.astype(float).fillna(plants.efficiency_r)

    committable_fields = ["start_up_cost", "min_down_time", "min_up_time"]
    defaults = pypsa.components.component_attrs["Generator"].default
    if unit_commitment:
        for attr in committable_fields:
            plants[attr] = plants[attr].astype(float).fillna(defaults[attr])
        plants["p_min_pu"] = (
            (plants.minimum_load_mw / plants.p_nom)
            .clip(
                upper=np.minimum(plants.summer_derate, plants.winter_derate),
                lower=0,
            )
            .astype(float)
            .fillna(0)
            .mul(0.95)
        )
    else:
        for attr in committable_fields:
            plants[attr] = defaults[attr]
    committable_attrs = {attr: plants[attr] for attr in committable_fields}

    # Define generators using modified ppl DataFrame
    caps = plants.groupby("carrier").p_nom.sum().div(1e3).round(2)
    logger.info(f"Adding {len(plants)} generators with capacities [GW] \n{caps}")
    n.madd(
        "Generator",
        plants.index,
        carrier=plants.carrier,
        bus=plants.bus_assignment,
        p_nom_min=plants.p_nom.where(
            plants.carrier.isin(conventional_carriers),
            0,
        ),  # enforces that plants cannot be retired/sold-off at their capital cost
        p_nom=plants.p_nom.where(plants.carrier.isin(conventional_carriers), 0),
        p_nom_extendable=plants.carrier.isin(extendable_carriers["Generator"]),
        efficiency=plants.efficiency.round(3),
        marginal_cost=plants.marginal_cost,
        capital_cost=plants.annualized_capex_fom,
        build_year=plants.build_year.astype(int).fillna(0),
        lifetime=plants.carrier.map(costs.lifetime),
        committable=unit_commitment,
        **committable_attrs,
    )

    # Add fuel and VOM costs to the network
    n.generators.loc[plants.index, "vom_cost"] = plants.carrier.map(
        costs.opex_variable_per_mwh,
    )
    n.generators.loc[plants.index, "fuel_cost"] = plants.fuel_cost
    n.generators.loc[plants.index, "heat_rate"] = plants.heat_rate_mmbtu_per_mwh
    n.generators.loc[plants.index, "ba_eia"] = plants.balancing_authority_code


def add_existing_phs(n: pypsa.Network, data_file: str):
    """
    Add existing pumped hydro storage units (ReEDS/NEMS generator database)
    to the network, matching each unit to its nearest bus by coordinates.
    """
    data = pd.read_csv(data_file)

    phs = data[data["tech"] == "pumped-hydro"][
        ["cap", "reeds_ba", "FIPS", "T_LAT", "T_LONG", "T_FOM"]
    ].copy()
    phs = phs.rename(
        columns={
            "reeds_ba": "country",
            "FIPS": "county",
            "T_LAT": "latitude",
            "T_LONG": "longitude",
        },
    )
    for col in ["cap", "T_FOM", "latitude", "longitude"]:
        phs[col] = pd.to_numeric(phs[col], errors="coerce")
    phs["county"] = phs.county.astype(str).str.strip()
    phs = phs.dropna(subset=["cap", "T_FOM", "latitude", "longitude"])
    if phs.empty:
        logger.info("No existing PHS units found in %s.", data_file)
        return

    phs = match_plant_to_bus(n, phs)
    phs = phs.dropna(subset=["bus_assignment"])
    if phs.empty:
        logger.info("No existing PHS units could be matched to a bus.")
        return

    # Aggregate units sharing a bus: sum capacity, capacity-weighted average T_FOM.
    phs_aggregated = phs.groupby("bus_assignment").agg(
        cap=("cap", "sum"),
        T_FOM=(
            "T_FOM",
            lambda x: (
                (x * phs.loc[x.index, "cap"]).sum() / phs.loc[x.index, "cap"].sum()
                if phs.loc[x.index, "cap"].sum() > 0
                else x.mean()
            ),
        ),
    )

    efficiency_store = 0.894427191  # 0.894427191^2 = 0.8
    efficiency_dispatch = 0.894427191  # 0.894427191^2 = 0.8

    capital_cost = phs_aggregated["T_FOM"].values * 1000

    add_missing_carriers(n, ["PHS"])
    n.carriers.loc["PHS", "co2_emissions"] = 0

    n.madd(
        "StorageUnit",
        names=phs_aggregated.index,
        suffix=" PHS",
        bus=phs_aggregated.index,
        carrier="PHS",
        p_nom_extendable=False,
        p_nom=phs_aggregated["cap"].values,
        p_nom_max=phs_aggregated["cap"].values,
        capital_cost=capital_cost,
        marginal_cost=0,
        efficiency_store=efficiency_store,
        efficiency_dispatch=efficiency_dispatch,
        max_hours=553.0 / 23.0 / efficiency_dispatch,
        cyclic_state_of_charge=True,
        lifetime=np.inf,
    )

    logger.info(
        "Added %d existing PHS storage units (%.1f MW total) matched to buses by coordinates.",
        len(phs_aggregated),
        phs_aggregated["cap"].sum(),
    )


def normed(s):
    return s / s.sum()


def normalize_bus_keys(values) -> pd.Series:
    """Canonicalize numeric bus/substation IDs read from CSV and NetCDF.

    GeoPandas writes the region name as a float in this dataset (``1.0``),
    while the base-network and bus2sub CSV use integral strings (``1``).
    Keeping that distinction would make every profile generator reference an
    undefined bus.
    """
    keys = pd.Series(values, copy=True).astype(str).str.strip()
    numeric = pd.to_numeric(keys, errors="coerce")
    integral = numeric.notna() & np.isclose(numeric % 1, 0.0)
    keys.loc[integral] = numeric.loc[integral].astype(np.int64).astype(str)
    return keys


def load_sub_to_bus(input_profiles) -> pd.DataFrame:
    """Map raw substation ids to raw buses before topology reduction."""
    bus2sub = pd.read_csv(input_profiles.bus2sub, dtype=str)
    bus_column = "bus_id" if "bus_id" in bus2sub.columns else "Bus"
    required = {"sub_id", bus_column}
    if not required.issubset(bus2sub.columns):
        raise ValueError(f"bus2sub must contain {sorted(required)}.")
    return (
        pd.DataFrame(
            {
                "sub_id": normalize_bus_keys(bus2sub["sub_id"]),
                "bus_id": bus2sub[bus_column].astype(str),
            },
        )
        .drop_duplicates("sub_id")
    )


def attach_wind_and_solar(
    n: pypsa.Network,
    costs: pd.DataFrame,
    input_profiles: str,
    carriers: list[str],
    extendable_carriers: dict[str, list[str]],
    source_timesteps=None,
    cost_multipliers: dict | None = None,
):
    add_missing_carriers(n, carriers)

    # Wind/solar get no ReEDS reg_cap_cost_diff multiplier (that table has no
    # wind/UPV column). Their regional cost differentiation is the per-site
    # interconnection cost the profiles already carry; see regional_cost.py.
    use_reeds_interconnection = bool((cost_multipliers or {}).get("enable", False))

    for car in carriers:
        if car in ["hydro", "EGS"]:
            continue

        capital_cost = costs.at[car, "annualized_capex_fom"]
        sub_to_bus = load_sub_to_bus(input_profiles).set_index("sub_id").bus_id

        with xr.open_dataset(getattr(input_profiles, "profile_" + car)) as ds:
            if ds.indexes["bus"].empty:
                continue
            profile_sub_ids = normalize_bus_keys(ds.bus.values)
            bus_list = profile_sub_ids.map(sub_to_bus)
            missing = profile_sub_ids[bus_list.isna()]
            if not missing.empty:
                raise ValueError(
                    f"{car} profile has {len(missing)} substation IDs absent from bus2sub "
                    f"(first: {missing.iloc[0]}).",
                )
            # `Network.madd` treats a Series as an indexed static attribute;
            # use an Index so its *values*, not its 0..N positional index, are
            # supplied as the Generator.bus values.
            bus_list = pd.Index(bus_list.astype(str).to_numpy())
            p_nom_max_bus = ds["p_nom_max"].to_pandas().set_axis(bus_list)
            weight_bus = ds["weight"].to_pandas().set_axis(bus_list)
            bus_profiles = ds["profile"].transpose("time", "bus").to_pandas()
            bus_profiles.columns = bus_list
            # Slice to representative hours, or broadcast across all horizons
            bus_profiles = align_timeseries_to_snapshots(n, bus_profiles, source_timesteps)
            # Per-site ReEDS interconnection cost. Taken as a bare array so it
            # aligns positionally with bus_list, which was built from this same
            # `bus` dimension just above -- no index alignment involved.
            cost_trans = (
                np.nan_to_num(ds["cost_trans_usd_per_mw"].to_pandas().to_numpy(), nan=0.0)
                if "cost_trans_usd_per_mw" in ds
                else None
            )

        if use_reeds_interconnection and cost_trans is not None:
            annuity_factor = calculate_annuity(
                costs.at[car, "cost_recovery_period_years"],
                costs.at[car, "wacc_real"],
            )
            capital_cost = renewable_interconnection_capital_cost(
                costs.at[car, "capex_overnight_per_kw"],
                costs.at[car, "capex_construction_finance_factor"],
                costs.at[car, "opex_fixed_per_kw"],
                cost_trans,
                annuity_factor,
            )
            logger.info(
                "%s: replaced ATB grid connection (%.1f USD/kW) with ReEDS per-site "
                "interconnection cost (min=%.0f mean=%.0f max=%.0f USD/MW).",
                car,
                costs.at[car, "capex_grid_connection_per_kw"],
                cost_trans.min(),
                cost_trans.mean(),
                cost_trans.max(),
            )
        elif use_reeds_interconnection:
            logger.warning(
                "%s profile has no cost_trans_usd_per_mw; keeping the uniform ATB capital cost.",
                car,
            )

        logger.info(f"Adding {car} capacity-factor profiles to the network.")

        n.madd(
            "Generator",
            bus_list,
            " " + car,
            bus=bus_list,
            carrier=car,
            p_nom_extendable=car in extendable_carriers["Generator"],
            p_nom_max=p_nom_max_bus,
            weight=weight_bus,
            marginal_cost=costs.at[car, "marginal_cost"],
            capital_cost=capital_cost,
            efficiency=1,
            build_year=n.investment_periods[0],
            lifetime=costs.at[car, "lifetime"],
            p_max_pu=bus_profiles,
        )


def attach_egs(
    n: pypsa.Network,
    costs: pd.DataFrame,
    input_profiles: str,
    carriers: list[str],
    extendable_carriers: dict[str, list[str]],
    line_length_factor=1,
    source_timesteps=None,
):
    """
    Attached STM Calculated wind and solar capacity factor profiles to the
    network.
    """
    car = "EGS"
    if (car not in carriers) and (car not in extendable_carriers["Generator"]):
        return
    if source_timesteps is not None:
        raise NotImplementedError(
            "EGS profiles are indexed by (year, Date) rather than a plain weather-hour index, so they "
            "cannot yet be sliced to representative hours. Disable EGS or "
            "clustering.temporal.representative_periods.",
        )
    add_missing_carriers(n, carriers)
    capital_recovery_period = 25  # Following EGS supply curves by Aljubran et al. (2024)
    discount_rate = 0.07  # load_costs(snakemake.input.tech_costs).loc["geothermal", "wacc_real"]
    drilling_cost = snakemake.config["renewable"]["EGS"]["drilling_cost"]

    with (
        xr.open_dataset(
            getattr(input_profiles, "specs_egs"),
        ) as ds_specs,
        xr.open_dataset(
            getattr(input_profiles, "profile_egs"),
        ) as ds_profile,
    ):
        bus2sub = load_sub_to_bus(input_profiles)

        # IGNORE: Remove dropna(). Rather, apply dropna when creating the original dataset
        specs = ds_specs.to_dataframe().reset_index().dropna()
        specs["sub_id"] = normalize_bus_keys(specs["sub_id"])
        df_specs = pd.merge(
            specs,
            bus2sub,
            on="sub_id",
            how="left",
        )
        df_specs["bus_id"] = df_specs["bus_id"].astype(str)

        # bus_id must be in index for pypsa to read it
        df_specs = df_specs.set_index("bus_id")

        # columns must be renamed to refer to the right quantities for pypsa to read it correctly
        logger.info(f"Using {drilling_cost} EGS drilling costs.")
        df_specs = df_specs.rename(
            columns={
                ("advanced_capex_usd_kw" if drilling_cost == "advanced" else "capex_usd_kw"): "capital_cost",
                "avail_capacity_mw": "p_nom_max",
                "fixed_om": "fixed_om",
            },
        )

        # TODO: come up with proper values for these params

        df_specs["capital_cost"] = 1000 * (
            df_specs["capital_cost"] * calculate_annuity(capital_recovery_period, discount_rate) + df_specs["fixed_om"]
        )  # convert and annualize USD/kW to USD/MW-year
        df_specs["efficiency"] = 1.0

        df_specs = df_specs.loc[~(df_specs.index == "nan")]

        # TODO: review what qualities need to be included. Currently limited for speedup.
        qualities = [1]  # df_specs.Quality.unique()

        for q in qualities:
            suffix = " " + car  # + f" Q{q}"
            df_q = df_specs[df_specs["Quality"] == q]

            bus_list = df_q.index.values
            capital_cost = df_q["capital_cost"]
            p_nom_max_bus = df_q["p_nom_max"]
            efficiency = df_q["efficiency"]  # for now.

            # IGNORE: Remove dropna(). Rather, apply dropna when creating the original dataset
            profile = ds_profile.sel(Quality=q).to_dataframe().dropna().reset_index()
            profile["sub_id"] = normalize_bus_keys(profile["sub_id"])
            df_q_profile = pd.merge(
                profile,
                bus2sub,
                on="sub_id",
                how="left",
            )
            bus_profiles = pd.pivot_table(
                df_q_profile,
                columns="bus_id",
                index=["year", "Date"],
                values="capacity_factor",
            )

            logger.info(
                f"Adding EGS (Resource Quality-{q}) capacity-factor profiles to the network.",
            )

            n.madd(
                "Generator",
                bus_list,
                suffix,
                bus=bus_list,
                carrier=car,
                p_nom_extendable=car in extendable_carriers["Generator"],
                p_nom_max=p_nom_max_bus,
                capital_cost=capital_cost,
                efficiency=efficiency,
                p_max_pu=bus_profiles,
                build_year=n.investment_periods[0],
                lifetime=capital_recovery_period,
            )


def attach_battery_storage(
    n: pypsa.Network,
    costs: pd.DataFrame,
    plants: pd.DataFrame,
):
    """Attaches Existing Battery Energy Storage Systems To the Network."""
    plants_filt = plants.query("carrier == 'battery' ")
    plants_filt.index = plants_filt.index.astype(str) + "_" + plants_filt.generator_id.astype(str)
    plants_filt.loc[:, "energy_storage_capacity_mwh"] = plants_filt.energy_storage_capacity_mwh.astype(float)
    plants_filt = plants_filt.dropna(subset=["energy_storage_capacity_mwh"])

    logger.info(
        f"Added Batteries as Storage Units to the network.\n{np.round(plants_filt.p_nom.sum() / 1000, 2)} GW Power Capacity \n{np.round(plants_filt.energy_storage_capacity_mwh.sum() / 1000, 2)} GWh Energy Capacity",
    )

    plants_filt = plants_filt.dropna(subset=["energy_storage_capacity_mwh"])
    n.madd(  # Adds storage units which can retire economically or at their lifetime
        "StorageUnit",
        plants_filt.index,
        carrier="battery",
        bus=plants_filt.bus_assignment,
        p_nom=plants_filt.p_nom,
        p_nom_max=plants_filt.p_nom,
        p_nom_min=0,
        p_nom_extendable=False,  # Only Allow lifetime retirments for existing BESS
        capital_cost=costs.at["4hr_battery_storage", "opex_fixed_per_kw"] * 1e3,
        max_hours=plants_filt.energy_storage_capacity_mwh / plants_filt.p_nom,
        build_year=plants_filt.build_year,
        lifetime=costs.at["4hr_battery_storage", "lifetime"],
        efficiency_store=0.85**0.5,
        efficiency_dispatch=0.85**0.5,
        cyclic_state_of_charge=True,
    )


def broadcast_investment_horizons_index(n: pypsa.Network, df: pd.DataFrame):
    """
    Broadcast the index of a dataframe to match the potentially multi-indexed
    investment periods of a PyPSA network.
    """
    sns = n.snapshots
    if isinstance(sns, pd.MultiIndex):
        source_index = pd.DatetimeIndex(df.index)
        target_times = pd.DatetimeIndex(sns.get_level_values("timestep"))
        period_count = len(n.investment_periods)
        if (
            len(df) * period_count == len(sns)
            and all(
                target_times[sns.get_level_values("period") == period].equals(
                    source_index,
                )
                for period in n.investment_periods
            )
        ):
            broadcast = pd.concat([df.copy() for _ in n.investment_periods], ignore_index=True)
            broadcast.index = sns
            return broadcast

    if not len(df.index) == len(sns):  # if broadcasting is necessary
        df.index = pd.to_datetime(df.index)
        dfs = []
        for planning_horizon in n.investment_periods.to_list():
            period_data = df.copy()
            period_data.index = df.index.map(lambda x: x.replace(year=planning_horizon))
            dfs.append(period_data)
        df = pd.concat(dfs)
        df = pd.merge(
            df,
            sns.to_frame().droplevel(0),
            left_index=True,
            right_index=True,
        ).drop(columns=["period", "timestep"])
        assert len(df.index) == len(sns)
    df.index = sns
    return df


def align_timeseries_to_snapshots(
    n: pypsa.Network,
    df: pd.DataFrame,
    source_timesteps=None,
    calendar_year_profile: bool = False,
):
    """
    Attach a raw source time series to the network snapshots.

    When representative periods are active, ``source_timesteps`` gives the real
    weather hour behind each (synthetic) snapshot, so the series is sliced down to
    those hours instead of broadcast across a full timeline.

    Set ``calendar_year_profile`` for inputs published on one canonical year
    (e.g. the Breakthrough hydro profiles) rather than on the real weather years:
    those are matched by (month, day, hour), since they carry no weather-hour
    identity to slice on.
    """
    if source_timesteps is not None:
        if calendar_year_profile:
            return reindex_calendar_timeseries_to_snapshots(n, df, source_timesteps)
        return reindex_source_timeseries_to_snapshots(n, df, source_timesteps)
    return broadcast_investment_horizons_index(n, df)


def apply_must_run_ratings(
    n: pypsa.Network,
    plants: pd.DataFrame,
    conventional_carriers: list,
    sns: pd.DatetimeIndex,
):
    """Applies Minimum Loading Capacities only to WECC ADS designated Plants."""
    conv_plants = plants.query("carrier in @conventional_carriers").copy()
    conv_plants.index = "C" + conv_plants.index

    conv_plants.loc[:, "ads_mustrun"] = conv_plants.ads_mustrun.infer_objects(
        copy=False,
    ).fillna(False)

    conv_plants.loc[:, "minimum_load_pu"] = conv_plants.minimum_load_mw / conv_plants.p_nom
    conv_plants.loc[:, "minimum_load_pu"] = (
        conv_plants.minimum_load_pu.clip(
            upper=np.minimum(conv_plants.summer_derate, conv_plants.winter_derate),
            lower=0,
        )
        .astype(float)
        .fillna(0)
    )
    must_run = conv_plants.query("ads_mustrun == True")
    n.generators.loc[must_run.index, "p_min_pu"] = must_run.minimum_load_pu.round(3) * 0.95


def clean_bus_data(n: pypsa.Network):
    """Drops data from the network that are no longer needed in workflow."""
    col_list = [
        # "Pd",
        "load_dissag",
        "LAF",
    ]
    n.buses = n.buses.drop(columns=[col for col in col_list if col in n.buses])


def attach_breakthrough_renewable_plants(
    n,
    fn_plants,
    renewable_carriers,
    extendable_carriers,
    costs,
    source_timesteps=None,
):
    add_missing_carriers(n, renewable_carriers)

    plants = pd.read_csv(fn_plants, dtype={"bus_id": str}, index_col=0).query(
        "bus_id in @n.buses.index",
    )
    plants = plants.replace(["wind_offshore"], ["offwind"])

    for tech in renewable_carriers:
        assert tech == "hydro"
        tech_plants = plants.query("type == @tech")
        tech_plants.index = tech_plants.index.astype(str)
        logger.info(f"Adding {len(tech_plants)} {tech} generators to the network.")

        p_nom_be = pd.read_csv(snakemake.input[f"{tech}_breakthrough"], index_col=0)

        intersection = set(p_nom_be.columns).intersection(
            tech_plants.index,
        )  # filters by plants ID for the plants of type tech
        p_nom_be = p_nom_be[list(intersection)]

        p_nom_be.columns = p_nom_be.columns.astype(str)

        if (tech_plants.Pmax == 0).any():
            # p_nom is the maximum of {Pmax, dispatch}
            p_nom = pd.concat([p_nom_be.max(axis=0), tech_plants["Pmax"]], axis=1).max(
                axis=1,
            )
            p_max_pu = (p_nom_be[p_nom.index] / p_nom).astype(float).fillna(0)  # some values remain 0
        else:
            p_nom = tech_plants.Pmax
            p_max_pu = p_nom_be[tech_plants.index] / p_nom

        leap_day = p_max_pu.loc["2016-02-29 00:00:00":"2016-02-29 23:00:00"]
        p_max_pu = p_max_pu.drop(leap_day.index)
        # Breakthrough profiles live on a fixed 2016 calendar, not on the weather years.
        p_max_pu = align_timeseries_to_snapshots(
            n,
            p_max_pu,
            source_timesteps,
            calendar_year_profile=True,
        )

        n.madd(
            "Generator",
            tech_plants.index,
            bus=tech_plants.bus_id,
            p_nom_min=p_nom,
            p_nom=p_nom,
            marginal_cost=0,
            p_max_pu=p_max_pu,  # timeseries of max power output pu
            p_nom_extendable=False,
            carrier=tech,
            weight=1.0,
            build_year=n.investment_periods[0],
            lifetime=np.inf,
        )
    return n


def main(snakemake):
    params = snakemake.params
    interconnection = snakemake.params.interconnect

    n = pypsa.Network(snakemake.input.base_network)

    # With representative periods active the network is built directly on the
    # selected snapshots, so the full 15-weather-year timeline is never created.
    # `source_timesteps` carries the real weather hour behind each snapshot and is
    # threaded into every time-series attachment below.
    source_timesteps = None
    representative_snapshots = getattr(snakemake.input, "representative_snapshots", None)
    if representative_snapshots is not None:
        snapshots, snapshot_weightings, source_timesteps = read_representative_snapshots(
            representative_snapshots,
        )
        n.snapshots = snapshots
        n.set_investment_periods(periods=params.planning_horizons)
        n.snapshot_weightings = snapshot_weightings
        logger.info(
            "Using %s representative snapshots from %s.",
            len(snapshots),
            representative_snapshots,
        )
    else:
        n.snapshots = get_multiindex_snapshots(
            params.snapshots,
            params.planning_horizons,
            params.renewable_weather_years,
        )
        n.set_investment_periods(periods=params.planning_horizons)
        n.snapshot_weightings.loc[:, :] = 1.0 / len(params.renewable_weather_years)

    regions_onshore = gpd.read_file(snakemake.input.regions_onshore)
    regions_offshore = gpd.read_file(snakemake.input.regions_offshore)
    reeds_shapes = gpd.read_file(snakemake.input.reeds_shapes)
    all_reeds_shapes = gpd.read_file(snakemake.input.all_reeds_shapes)
    reeds_memberships = pd.read_csv(snakemake.input.reeds_memberships)

    costs = pd.read_csv(snakemake.input.tech_costs)
    costs = costs.pivot(index="pypsa-name", columns="parameter", values="value")
    update_transmission_costs(
        n,
        costs,
        params.length_factor,
        distance_cost_fn=snakemake.input.transmission_distance_cost,
        basecost_fn=snakemake.input.transmission_basecost,
    )

    renewable_carriers = set(params.renewable_carriers)
    extendable_carriers = params.extendable_carriers
    conventional_carriers = params.conventional_carriers
    conventional_inputs = {k: v for k, v in snakemake.input.items() if k.startswith("conventional_")}

    plants = load_powerplants(
        snakemake.input["powerplants"],
        n.investment_periods,
        interconnect=interconnection,
    )
    plants = filter_plants_by_region(
        plants,
        regions_onshore,
        regions_offshore,
        reeds_shapes,
        all_reeds_shapes,
        reeds_memberships,
    )
    plants = match_plant_to_bus(n, plants)

    attach_egs(
        n,
        costs,
        snakemake.input,
        renewable_carriers,
        extendable_carriers,
        params.length_factor,
        source_timesteps=source_timesteps,
    )

    attach_conventional_generators(
        n,
        costs,
        plants,
        conventional_carriers,
        extendable_carriers,
        params.conventional,
        renewable_carriers,
        conventional_inputs,
        unit_commitment=params.conventional["unit_commitment"],
        fuel_price=None,  # update fuel prices later
    )
    if params.conventional.get("must_run", False):
        # TODO (@ktehranchi): In the future the plants that are must-run should
        # not be clustered and instead retire according to lifetime
        apply_must_run_ratings(
            n,
            plants,
            conventional_carriers,
            n.snapshots,
        )

    if params.add_existing_phs:
        add_existing_phs(n, snakemake.input.existing_PHS)

    # attach_battery_storage(
    #     n,
    #     costs,
    #     plants,
    # )

    attach_wind_and_solar(
        n,
        costs,
        snakemake.input,
        renewable_carriers,
        extendable_carriers,
        source_timesteps=source_timesteps,
        cost_multipliers=snakemake.params.cost_multipliers,
    )
    renewable_carriers = list(
        set(params.renewable_carriers).intersection(
            {"onwind", "solar", "offwind", "offwind_floating"},
        ),
    )

    attach_existing_renewable_capacities(
        n,
        plants,
        renewable_carriers,
        costs,
    )

    # temporarily adding hydro with breakthrough only data until I can correctly import hydro_data
    n = attach_breakthrough_renewable_plants(
        n,
        snakemake.input["plants_breakthrough"],
        list(set(params.renewable_carriers).intersection({"hydro"})),
        extendable_carriers,
        costs,
        source_timesteps=source_timesteps,
    )

    update_p_nom_max(n)

    # fix p_nom_min for extendable generators
    # The "- 0.001" is just to avoid numerical issues
    n.generators["p_nom_min"] = n.generators.apply(
        lambda x: x["p_nom"] if (x["p_nom_extendable"] and x["p_nom_min"] == 0) else x["p_nom_min"],
        axis=1,
    )

    output_folder = os.path.dirname(snakemake.output[0]) + "/base_network"
    export_network_for_gis_mapping(n, output_folder)

    clean_bus_data(n)
    sanitize_carriers(n, snakemake.config)
    n.meta = snakemake.config

    # n.export_to_netcdf(snakemake.output[0])
    pickle.dump(n, open(snakemake.output[0], "wb"))


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("add_electricity", demand_level="High")
    configure_logging(snakemake)
    main(snakemake)
