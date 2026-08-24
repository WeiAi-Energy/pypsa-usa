(data-costs)=
# Costs
## Costs and Candidate Resources

 In PyPSA-USA, candidate resource forecasted capital and operating costs are defined by the NREL Annual Technology Baseline (ATB) accessed through the PUDL project. The model currently uses the 2024 ATB which provides data for expected costs across the years 2025 - 2050. Users are able to configure which ATB model case and scenario to reference:

 ```yaml
   atb:
    model_case: "Market" # Market, R&D
    scenario: "Moderate" # Advanced, Conservative, Moderate
```

### Regional Cost Differentiation

Capital costs vary geographically through two complementary mechanisms, both of which
scale **overnight capital cost only** and never touch fixed O&M:

- **Dispatchable technologies and storage** use the county-level `reg_cap_cost_diff`
  table from [NREL ReEDS](https://nrel.github.io/ReEDS-2.0/), which gives relative
  cost differences centred on zero per technology group (applied as `1 + x`). Because
  the model's buses are finer than county resolution and lose their `county` attribute
  during network simplification, each bus is matched to the geographically nearest
  county the table covers.
- **Wind and solar** are deliberately excluded from that table, which publishes no
  wind or utility-PV column. Their regional differentiation instead comes from the
  per-site interconnection cost (`cost_total_trans_usd_per_mw`) in the ReEDS supply
  curves, which *replaces* ATB's uniform `capex_grid_connection_per_kw` to avoid
  double counting interconnection.

The technology mapping is configuration-driven, so the covered carriers can be changed
without touching code:

```yaml
cost_multipliers:
  enable: true
  column_to_tech:
    BATTERY|PVB: [2hr_battery_storage, 4hr_battery_storage, 6hr_battery_storage]
    COMBINED_CYCLE|FUEL_CELL: [CCGT, CCGT-95CCS, CCGT-97CCS, hydrogen_ct, tes]
    COMBUSTION_TURBINE|CONSUME|LFILL|OGS: [OCGT, electrolysis]
    NUCLEAR: [nuclear]
```

Each technology is adjusted where it is created: wind and solar in `add_electricity`,
everything else in `add_extra_components`. Setting `enable: false` reproduces
undifferentiated national costs exactly. Technologies whose costs bypass the ATB table
entirely (EGS, pumped hydro) receive no multiplier.

### Transmission Expansion Costs

AC line and DC link expansion costs are resolved per branch by **voltage class** and by
**route region**, rather than from a single national USD/MW-km. Both dimensions come from
[NREL ReEDS](https://nrel.github.io/ReEDS-2.0/) tables, in 2004 USD, converted to 2022 USD:

| File | Supplies |
| --- | --- |
| `transmission_distance_cost_500kVac_county.csv` | 500 kV AC cost for ~799k county pairs |
| `rev_transmission_basecost.csv` | USD/MW-mile by voltage class and by four ReEDS cost regions |

**Route region.** The county-pair table's `length_miles` is the great-circle distance
between county centroids, so ReEDS's terrain and route-detour costs sit entirely in the
*price*. Dividing cost by distance therefore gives a USD/MW **per great-circle km** that
already carries regional base cost, terrain and detour. Each branch resolves its value in
four steps:

1. the exact `(county0, county1)` pair, which carries the measured corridor;
2. the mean of the two endpoint counties' **county field** — the median over each county's
   8 shortest incident pairs. This is the main path, not an exception: 78% of lines have
   both ends in one county and ReEDS publishes no same-county pair. The field is restricted
   to near neighbours because a county carries ~514 incident pairs reaching out to ~700 km
   and the per-km cost falls with pair length, so a whole-county average both blurs the
   local corridor and biases short lines low;
3. the median field value across the same state, which covers Connecticut (ReEDS uses its
   post-2022 planning regions while the county shapes here use the legacy FIPS codes);
4. the national median, for buses with no county at all.

**Voltage class.** Each voltage's cost ratio is taken relative to that region's *own*
500 kV AC entry first, and the median across the four regions is used. Taking the ratio
inside the region isolates the voltage relationship from the regional level — the level
is already carried by the county table, so a regional voltage factor would double count
it. Cost then falls with voltage as a single power law, `ratio = (kV / 500)^b`, fitted
by log-log least squares to the four published anchors (69 kV 4.72, 138 kV 3.01,
230 kV 2.11, 500 kV 1.00) and constrained through the 500 kV reference so that
`ratio(500 kV) = 1` exactly — the county field is quoted at 500 kV, so a free intercept
would rescale every line. This gives `b = -0.82` and covers the classes ReEDS does not
publish but the network carries (100/115/161/345/765 kV) on the same footing as the
measured ones. The anchors are not exactly collinear in log-log, so one exponent cannot
hit all four: the worst residual is -10.4% at 230 kV, and -0.9% on the length-weighted
network total.

**Line length and impedance.** `lines.length_factor` is the routed-length / great-circle
ratio, applied once in `assign_line_length` and reused unchanged by every later
aggregation. It must be the same value throughout: PyPSA rescales `x`, `r` and
`capital_cost` by `new_length / old_length` on each aggregation pass, so a consistent
factor cancels out of both impedance and cost per great-circle km, while a mismatched one
silently biases them. Lines carry no standard `type`; their `r`/`x`/`b` come from the
TAMU data and, after `convert_to_voltage_level`, from the per-unit conversion to the
common base voltage. Assigning a type would make PyPSA recompute impedance as
`per_length × length`, discarding those values and re-coupling impedance to
`length_factor`.

**DC.** DC corridors are 500 kV bipoles, priced as the AC line cost times a single
DC/AC ratio of 0.4137 — the four cost regions agree on it to within rounding — plus the
converter-pair cost, which does not scale with distance. Each physical corridor is modelled
as a `_fwd`/`_rev` Link pair whose expansions are constrained equal, so each direction
carries half the corridor cost.

At the 500 kV reference voltage this reproduces the national figure it replaces to within
about 11%; at 138 kV it is roughly 5x higher, which is the point of the change.

### Candidate Resources

- **Coal Plants**: With and without Carbon Capture Storage (CCS) at 95% and 99% capture rates.
- **Natural Gas**: Combustion Turbines and Combined Cycle plants, with and without 95% CCS.
- **Hydrogen Combustion Turbines**: Hydrogen Combusion Turbines are implemented under the assumption of market-available hydrogen drop-in fuel. Following the default assumptions in the [ReEDS Hydrogen implementation](https://nrel.github.io/ReEDS-2.0/model_documentation.html#drop-in-renewable-fuel). This implementation does not account for the energy or costs required to produce or transport the fuel. Future work will implement a more detailed production, transport, and storage model of hydrogen.
- **Nuclear Reactors**: Large Nuclear Reactors (AP1000) and Small Modular Reactors
- **Renewable Energy**: Utility-scale onshore wind, fixed-bottom and floating offshore wind, utility-scale solar.
- **Battery Energy Storage**: 2-10 hour Battery Energy Storage Systems (BESS).
- **Pumped Hydro Storage (PHS)**: Supply curves for 8-12 hour PHS are integrated from the [NREL Closed-Loop PHS dataset](https://www2.nrel.gov/gis/psh-supply-curves).
- **Enhanced Geothermal Systems (EGS**): Methods for implementation will be released in a forthcoming paper.

## Fuel Costs

PyPSA-USA integrates fuel costs that varry across spatial scopes and temporal scales. For more information, see [here](./data-generators.md#fuel-costs-and-heat-rates)

## Sector Costs

Running sector studies will use the same power system costs as electrical only studies. Costs specific to each sector can be found in the [service sector](./data-services.md), [transportation sector](./data-transportation.md), and [industrial sector](./data-industrial.md) pages accordingly.
