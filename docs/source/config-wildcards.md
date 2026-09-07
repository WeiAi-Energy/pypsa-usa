(wildcards)=
# Wildcards

It is easy to run PyPSA-USA for multiple scenarios using the wildcards feature of `snakemake`.
Wildcards generalise a rule to produce all files that follow a regular expression pattern
which e.g. defines one particular scenario. One can think of a wildcard as a parameter that shows
up in the input/output file names of the `Snakefile` and thereby determines which rules to run,
what data to retrieve and what files to produce.

```{note}
Detailed explanations of how wildcards work in ``snakemake`` can be found in the
`relevant section of the [documentation](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#wildcards).
```

(interconnect)=
## The `{interconnect}` wildcard

The `{interconnect}` wildcard sets the geographc scope of the model run. Models
can be run for the `western`, `eastern`, `texas`, or `usa` grid. The interconnects
follow the representation described by [Breakthrough Energy](https://breakthroughenergy.org/).

A visual representation of each `{interconnect}` is shown below:

```{eval-rst}
.. image:: _static/cutouts/cutouts.png
    :scale: 100 %
```

(simpl)=
## The ``{simpl}`` wildcard

The ``{simpl}`` wildcard specifies number of buses a detailed
network model should be pre-clustered to in the rule
:mod:`simplify_network`.

(clusters)=
## The `{clusters}` wildcard

The `{clusters}` wildcard specifies the number of buses a detailed network model should be reduced to. Every aggregation pass — substation aggregation, electrical-distance clustering and low-degree reduction — runs in the rule :mod:`simplify_network`; there is no separate clustering rule.
The number of clusters must be lower than the total number of nodes and higher than the number of balancing authoritites.

If an `m` is placed behind the number of clusters (e.g. `100m`), generators are only moved to the clustered buses but not aggregated by carrier; i.e. the clustered bus may have more than one e.g. wind generator.

(ll)=
## The `{ll}` wildcard

The `{ll}` wildcard specifies what limits on
line expansion are set for the optimisation model.
It is handled in the rule :mod:`prepare_network`.

We reccomend using the line volume limit for constraining
transission expansion. Use ``lv`` (for setting a limit on line volume)

After ``lv`` you can specify two type of limits:

       ``opt`` or a float bigger than one (e.g. 1.25).

       (a) If ``opt`` is chosen line expansion is optimised
           according to its capital cost.

       (b) ``v1.25`` will limit the total volume of line expansion
           to 25 % of currently installed capacities weighted by
           individual line lengths; investment costs are neglected.


(opts)=
## The `{opts}` wildcard

The `{opts}` wildcard is used for electricity-only studies. It triggers
optional constraints, which are activated in either :mod:`prepare_network` or
the :mod:`solve_network` step. It may hold multiple triggers separated by `-`,
i.e. `REM-3H` contains the `REM` regional emissions limit trigger and the `3H` switch.

The REM, ERM, RPS can be defined using either the reeds zone name 'p##',
the state code (eg, TX, CA, MT), pypsa-usa interconnect name (western, eastern, texas, usa),
nerc region name, transmission region name (trans_reg, eg PJM, MISO), or transmission
group name (trans_grp, eg PJM_East, MISO_Central).

```{warning}
TCT Targets can only be used with renewable generators and utility scale batteries in sector studies.
```

There are currently:

```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 10,20,10,10
   :file: configtables/opts.csv
```

### Energy Reserve Margin (ERM) Configuration

The ERM constraint ensures the system can serve demand plus a reserve margin at every timestep. Unlike traditional planning reserve margins that only consider peak demand, ERM enforces the constraint across all snapshots.

It is formulated as one **aggregated adequacy inequality per region and snapshot**. For each region `R` and snapshot `t`:

```
  Σ_{g∈R}  p_nom_g · availability_{g,t}          generator capacity × availability factor
+ Σ_{s∈R}  discharge_potential_{s,t}             energy-backed storage discharge potential
+ F_{R,t}                                        net actual power flow into R across its boundary
≥ (1 + erm_R) · demand_{R,t}
```

Only the **dischargers** carry a variable of their own. Every other term is either a linear expression in base-state variables the model already builds or a constant, so the ERM adds no second copy of the dispatch, no second nodal balance and no second voltage law. On a county-resolution USA network at 8 snapshots this is the difference between roughly +3.4 M constraint rows and +0.8 M, and it removes a duplicated Kirchhoff voltage law worth about 13 M matrix non-zeros.

**Contributions are gross.** Nothing is netted out for what the base state is already doing with the same asset: a generator counts its full available capacity even while it is dispatched, and a storage unit counts its discharge potential even while it charges. The one place the base state does enter is the boundary flow, which is deliberately conservative — a region that exports in the base state is assumed to keep exporting when the reserve is called.

**Key Features:**

- **Availability factor is `p_max_pu`.** It already carries the VRE resource profile *and* the temperature-dependent thermal derates applied in `capacity_derates.py`, so no separate capacity-credit table is involved. Extendable capacity enters the row on its `p_nom` variable; fixed capacity is a constant on the right-hand side
- Resources must be "energy-backed" — reserve discharge from a storage unit is capped both by its rating against the shared `p_nom` and by the state of charge the base-state dispatch leaves in the device, and reserve flow on a `Store` discharge link is capped by the energy in that store. A discharge link reaches the electricity bus through its efficiency, so it contributes `efficiency · p`
- **Inter-regional transmission contributes to adequacy through the base state's actual flow.** A branch with exactly one end in the region enters with the sign of the in-region end: `+s` at `bus1`, `−s` at `bus0` for a passive branch, `+efficiency · p` at `bus1` and `−p` at `bus0` for a link. Where losses are modelled, half the branch loss is charged at each end, matching PyPSA's own nodal balance. Only electric branches count — a `tes` or `H2` bus carries no region label, so a TES or electrolysis link is not a boundary crossing
- Because the boundary term is the base-state flow rather than an independently redispatchable reserve flow, an SSSC no longer earns adequacy value directly; it affects the requirement only through the base-state flows it steers
- No ramping coupling to the base state is imposed, and the constraint is applied at every snapshot rather than only at stress hours
- Supports overlapping regions with different reserve margins; each region gets its own constraint row, so overlaps are additive in the requirement rather than resolved by taking a maximum

**Requirement granularity.** One row per region and snapshot. `erm: {all: X}` is shorthand for "apply `X` to every transmission group (`trans_grp`)" rather than a single nationwide row: a nationwide region has no boundary, so its flow term would vanish and the requirement would collapse into a nationwide capacity sum. A transmission-region key (`trans_reg`, e.g. `PJM`, `MISO`) expands the same way, into one row per transmission group it contains. Transmission group is the unit because that is the granularity at which reserve margins are actually planned and enforced — each ISO/RTO, and each of MISO's and SPP's sub-regions, carries its own requirement. A NERC-region key (`nerc_reg`, e.g. `WECC_NW`) is *not* expanded this way, because `nerc_reg` and `trans_grp` cross-cut each other with no unambiguous mapping between them; it is passed through unchanged as a single pooled row, like a state or interconnect key. Explicit region keys override the value inherited from `all` or a transmission-region key.

The flip side of aggregating is worth stating plainly: within a region the requirement is one sum, so intra-regional transmission earns no adequacy value. A region containing both a large generator and an electrically unreachable load pocket will pass. Use a finer region definition (transmission group rather than transmission region, or ReEDS zone, say) where that matters.

One dual is stored after solving:

- `n.erm_region_price` — a `DataFrame` indexed by snapshot with one column per ERM region, the dual of the regional requirement: the marginal cost of raising that region's reserve margin.

**Configuration:**

To enable ERM, add `ERM` to the `opts` wildcard in your scenario configuration:

```yaml
scenario:
  opts: [ERM-3h]  # or [REM-ERM-3h] to combine with other opts
```

To customize the ERM values per region, add an `erm` section under `electricity` in your config file:

```yaml
electricity:
  erm:
    all: 0.15        # 15% reserve margin for all transmission groups (default)
    # Or specify per region:
    # PJM: 0.186          # transmission region -> expands to PJM_East, PJM_West
    # MISO_Central: 0.081 # explicit transmission group
    # WECC_NW: 0.178      # NERC region -> passed through unchanged, one pooled row
```

If no `erm` configuration is provided, a default of `{'all': 0.15}` is used, i.e. a 15% reserve margin on every transmission group.

**Valid region identifiers:**
- `all` - applies the same margin to every transmission group present in the network
- State codes: `TX`, `CA`, `MT`, etc.
- Interconnect names: `western`, `eastern`, `texas`
- NERC region names (`nerc_reg`, e.g. `WECC_NW`) - passed through unchanged as one pooled row; not expanded, since `nerc_reg` and `trans_grp` cross-cut each other
- Transmission region names (`trans_reg`, e.g. `PJM`, `MISO`, `SPP`) - expanded into one row per transmission group it contains
- Transmission group names (`trans_grp`, e.g. `PJM_East`, `MISO_Central`) - the finest, and default, granularity
- ReEDS zone names: `p1`, `p2`, etc.

(sector)=
## The `{sector}` wildcard

The `{sector}` wildcard is used to specify what sectors to include. If `None`
is provided, an electrical only study is completed.

| Sector      | Code | Description                                    | Status      |
|-------------|------|------------------------------------------------|-------------|
| Electricity | E    | Electrical sector. Will always be run.         | Runs        |
| Natural Gas | G    | All sectors added                              | Development |


(cutout_wc)=
## The `{cutout}` wildcard

The `{cutout}` wildcard facilitates running the rule :mod:`build_cutout`
for all cutout configurations specified under `atlite: cutouts:`. Each cutout
is descibed in the form `{dataset}_{year}`. These cutouts will be stored in a
folder specified by `{cutout}`.

Valid dataset names include: `era5`
Valid years can be from `1940` to `2022`

```{note}
Data for `era5_2019` has been pre-pared for the user and will be automatically downloaded
during the workflow. If other years are needed, the user will need to prepaer the
cutout themself.
```
