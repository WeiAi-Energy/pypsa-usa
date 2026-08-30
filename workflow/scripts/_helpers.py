# By PyPSA-USA Authors

import copy
import hashlib
import logging
import os
import re
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import requests
import yaml
from pypsa.geo import haversine_pts
from snakemake.utils import update_config

logger = logging.getLogger(__name__)

REGION_COLS = ["geometry", "name", "x", "y", "country"]


def configure_logging(snakemake, skip_handlers=False):
    """
    Configure the basic behaviour for the logging module.

    Note: Must only be called once from the __main__ section of a script.

    The setup includes printing log messages to STDERR and to a log file defined
    by either (in priority order): snakemake.log.python, snakemake.log[0] or "logs/{rulename}.log".
    Additional keywords from logging.basicConfig are accepted via the snakemake configuration
    file under snakemake.config.logging.

    Parameters
    ----------
    snakemake : snakemake object
        Your snakemake object containing a snakemake.config and snakemake.log.
    skip_handlers : True | False (default)
        Do (not) skip the default handlers created for redirecting output to STDERR and file.
    """
    kwargs = snakemake.config.get("logging", dict()).copy()
    kwargs.setdefault("level", "INFO")

    if skip_handlers is False:
        fallback_path = Path(__file__).parent.joinpath(
            "..",
            "logs",
            f"{snakemake.rule}.log",
        )
        logfile = snakemake.log.get(
            "python",
            snakemake.log[0] if snakemake.log else fallback_path,
        )
        kwargs.update(
            {
                "handlers": [
                    # Prefer the 'python' log, otherwise take the first log for each
                    # Snakemake rule
                    logging.FileHandler(logfile),
                    logging.StreamHandler(),
                ],
            },
        )
    logging.basicConfig(**kwargs)


def configure_cds_api(snakemake):
    """
    Expose CDS API credentials from the Snakemake config to cdsapi/atlite.

    Does nothing when no credentials are configured, in which case ``cdsapi``
    falls back to ``~/.cdsapirc``.
    """
    api = snakemake.config.get("api", {})
    cds = api.get("cds") or api.get("cdsapi") or api.get("copernicus")

    if not cds:
        return

    if isinstance(cds, str):
        key = cds
        url = "https://cds.climate.copernicus.eu/api"
    else:
        key = cds.get("key") or cds.get("token") or cds.get("personal_access_token")
        url = cds.get("url", "https://cds.climate.copernicus.eu/api")

    if key and not os.environ.get("CDSAPI_KEY"):
        os.environ["CDSAPI_KEY"] = str(key)
    if url and not os.environ.get("CDSAPI_URL"):
        os.environ["CDSAPI_URL"] = str(url)


def setup_custom_logger(name):
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    # logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


def register_topology_carriers(n):
    """
    Declare the carriers the raw topology already refers to.

    `add_electricity` is what normally populates `n.carriers`, and it runs after
    the network has been reduced and clustered, so the AC and DC carriers that
    buses, lines and links were born with are undeclared until then. PyPSA's
    consistency check flags every one of them on each export in between, which
    buries any real warning under tens of thousands of lines of noise.
    """
    referenced = set()
    for component in ("Bus", "Line", "Link"):
        df = n.df(component)
        if "carrier" in df.columns:
            referenced |= set(df.carrier.dropna().astype(str))
    missing = sorted(carrier for carrier in referenced - set(n.carriers.index) if carrier)
    if missing:
        n.madd("Carrier", missing)
        logging.getLogger(__name__).info(
            "Registered topology carriers: %s.",
            ", ".join(missing),
        )
    return n


def read_network(path):
    """
    Read a network written either as netCDF or as a pickle.

    `add_electricity` hands its result on as a pickle so that the custom
    columns this workflow carries on buses, lines and generators survive
    intact; everything else in the pipeline uses netCDF. Rules that sit
    downstream of the hand-off should not have to care which they were given.
    """
    if str(path).endswith((".pkl", ".pickle")):
        import dill

        with open(path, "rb") as stream:
            return dill.load(stream)
    return pypsa.Network(path)


def load_network(import_name=None, custom_components=None):
    """
    Helper for importing a pypsa.Network with additional custom components.

    Parameters
    ----------
    import_name : str
        As in pypsa.Network(import_name)
    custom_components : dict
        Dictionary listing custom components.
        For using ``snakemake.config['override_components']``
        in ``config.yaml`` define:

        .. code:: yaml

            override_components:
                ShadowPrice:
                    component: ["shadow_prices","Shadow price for a global constraint.",np.nan]

    Attributes
    ----------
                    name: ["string","n/a","n/a","Unique name","Input (required)"]
                    value: ["float","n/a",0.,"shadow value","Output"]

    Returns
    -------
    pypsa.Network
    """
    from pypsa.descriptors import Dict

    override_components = None
    override_component_attrs = None

    if custom_components is not None:
        override_components = pypsa.components.components.copy()
        override_component_attrs = Dict(
            {k: v.copy() for k, v in pypsa.components.component_attrs.items()},
        )
        for k, v in custom_components.items():
            override_components.loc[k] = v["component"]
            override_component_attrs[k] = pd.DataFrame(
                columns=["type", "unit", "default", "description", "status"],
            )
            for attr, val in v["attributes"].items():
                override_component_attrs[k].loc[attr] = val

    return pypsa.Network(
        import_name=import_name,
        override_components=override_components,
        override_component_attrs=override_component_attrs,
    )


def pdbcast(v, h):
    return pd.DataFrame(
        v.values.reshape((-1, 1)) * h.values,
        index=v.index,
        columns=h.index,
    )


def calculate_annuity(n, r):
    """
    Calculate the annuity factor for an asset with lifetime n years and.

    discount rate of r, e.g. annuity(20, 0.05) * 20 = 1.6
    """
    if isinstance(r, pd.Series):
        return pd.Series(1 / n, index=r.index).where(
            r == 0,
            r / (1.0 - 1.0 / (1.0 + r) ** n),
        )
    elif r > 0:
        return r / (1.0 - 1.0 / (1.0 + r) ** n)
    else:
        return 1 / n


def load_costs(tech_costs: str) -> pd.DataFrame:
    df = pd.read_csv(tech_costs)
    return df.pivot(index="pypsa-name", columns="parameter", values="value").fillna(0)


def get_complete_bidirectional_link_pairs(links: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Return complete `_fwd`/`_rev` link pairs keyed by shared base name.

    Every physical DC corridor is built as two directional Links (``add_dclines``
    in ``build_base_network``), and ``opts.bidirectional_link`` forces the two
    directions to expand by the same amount. Callers that price or measure a
    corridor therefore have to know which Links are two halves of one asset --
    and which are unpaired survivors that must carry the full figure (a
    direction can be dropped as a self-loop during topology reduction).
    """
    # Keyed off the index alone, so a frame carrying names but no columns -- which
    # pandas reports as `empty` -- still resolves.
    if links.index.empty:
        return {}

    directional = links.index[links.index.str.contains(r"_fwd$|_rev$", regex=True, case=True)]
    pairs: dict[str, dict[str, str]] = {}

    for link_name in directional:
        if link_name.endswith("_fwd"):
            pairs.setdefault(link_name[: -len("_fwd")], {})["fwd"] = link_name
        elif link_name.endswith("_rev"):
            pairs.setdefault(link_name[: -len("_rev")], {})["rev"] = link_name

    return {base_name: pair for base_name, pair in pairs.items() if {"fwd", "rev"} <= set(pair)}


#: Link columns written by ``add_electricity.update_transmission_costs``: the
#: annualized per-MW-km line cost (already voltage- and region-resolved) and the
#: annualized converter-pair cost that does not scale with distance.
LINK_UNIT_COST_COL = "capex_per_mw_km_annual"
LINK_FIXED_COST_COL = "capex_fixed_per_mw_annual"


def recompute_link_transmission_costs(n: pypsa.Network) -> None:
    """Rebuild transmission ``Link.capital_cost`` from the stored unit costs.

    ``add_electricity`` resolves each transmission Link's per-MW-km cost from its
    endpoint counties and stashes it in :data:`LINK_UNIT_COST_COL`. Every later
    topology step moves those endpoints -- ``simplify_network`` relocates Links
    off eliminated buses, and pypsa's clustering re-points them at cluster
    centroids -- so the distance the cost was based on goes stale. pypsa only
    rescales Link costs when ``scale_link_capital_costs=True``, which would also
    (wrongly) scale the converter-pair term, so this recomputes both terms
    explicitly from the *current* bus coordinates instead.

    Call it at the end of every rule that changes bus positions.
    """
    if n.links.empty or LINK_UNIT_COST_COL not in n.links.columns:
        return

    transmission = n.links.index[n.links.carrier.isin(["AC", "DC"])]
    if transmission.empty:
        return

    coords = n.buses[["x", "y"]].astype(float)
    great_circle_km = pd.Series(
        haversine_pts(
            coords.loc[n.links.loc[transmission, "bus0"]].to_numpy(),
            coords.loc[n.links.loc[transmission, "bus1"]].to_numpy(),
        ),
        index=transmission,
    )

    unit = pd.to_numeric(n.links.loc[transmission, LINK_UNIT_COST_COL], errors="coerce").fillna(0.0)
    fixed = pd.to_numeric(n.links.loc[transmission, LINK_FIXED_COST_COL], errors="coerce").fillna(0.0)

    # A complete pair is one physical corridor represented twice, so each
    # direction carries half the cost; an unpaired survivor carries all of it.
    paired = [
        link_name
        for pair in get_complete_bidirectional_link_pairs(n.links.loc[transmission]).values()
        for link_name in pair.values()
    ]
    directions = pd.Series(1.0, index=transmission)
    directions.loc[paired] = 2.0

    n.links.loc[transmission, "capital_cost"] = (unit * great_circle_km + fixed) / directions
    logger.info(
        "Recomputed capital_cost for %s transmission Links (%s of them paired) from current bus coordinates.",
        len(transmission),
        len(paired),
    )


def is_transport_model(transmission_network):
    if transmission_network != "tamu":
        raise ValueError(
            "This workflow only supports the non-transport TAMU power-flow model; "
            f"received {transmission_network!r}.",
        )
    return False


def update_p_nom_max(n):
    # if extendable carriers (solar/onwind/...) have capacity >= 0,
    # e.g. existing assets from the OPSD project are included to the network,
    # the installed capacity might exceed the expansion limit.
    # Hence, we update the assumptions.

    n.generators.p_nom_max = n.generators[["p_nom_min", "p_nom_max"]].max(1)


def aggregate_p_nom(n):
    return pd.concat(
        [
            n.generators.groupby("carrier").p_nom_opt.sum(),
            n.storage_units.groupby("carrier").p_nom_opt.sum(),
            n.links.groupby("carrier").p_nom_opt.sum(),
            n.loads_t.p.groupby(n.loads.carrier, axis=1).sum().mean(),
        ],
    )


def aggregate_p(n):
    return pd.concat(
        [
            n.generators_t.p.sum().groupby(n.generators.carrier).sum(),
            n.storage_units_t.p.sum().groupby(n.storage_units.carrier).sum(),
            n.stores_t.p.sum().groupby(n.stores.carrier).sum(),
            -n.loads_t.p.sum().groupby(n.loads.carrier).sum(),
        ],
    )


def aggregate_e_nom(n):
    return pd.concat(
        [
            (n.storage_units["p_nom_opt"] * n.storage_units["max_hours"]).groupby(n.storage_units["carrier"]).sum(),
            n.stores["e_nom_opt"].groupby(n.stores.carrier).sum(),
        ],
    )


def aggregate_p_curtailed(n):
    return pd.concat(
        [
            (
                (n.generators_t.p_max_pu.sum().multiply(n.generators.p_nom_opt) - n.generators_t.p.sum())
                .groupby(n.generators.carrier)
                .sum()
            ),
            ((n.storage_units_t.inflow.sum() - n.storage_units_t.p.sum()).groupby(n.storage_units.carrier).sum()),
        ],
    )


def aggregate_costs(n, flatten=False, opts=None, existing_only=False):
    components = dict(
        Link=("p_nom", "p0"),
        Generator=("p_nom", "p"),
        StorageUnit=("p_nom", "p"),
        Store=("e_nom", "p"),
        Line=("s_nom", None),
        Transformer=("s_nom", None),
    )

    costs = {}
    for c, (p_nom, p_attr) in zip(
        n.iterate_components(components.keys(), skip_empty=False),
        components.values(),
    ):
        if c.df.empty:
            continue
        if not existing_only:
            p_nom += "_opt"
        costs[(c.list_name, "capital")] = (c.df[p_nom] * c.df.capital_cost).groupby(c.df.carrier).sum()
        if p_attr is not None:
            p = c.pnl[p_attr].sum()
            if c.name == "StorageUnit":
                p = p.loc[p > 0]
            costs[(c.list_name, "marginal")] = (p * c.df.marginal_cost).groupby(c.df.carrier).sum()
    costs = pd.concat(costs)

    if flatten:
        assert opts is not None
        conv_techs = opts["conv_techs"]

        costs = costs.reset_index(level=0, drop=True)
        costs = costs["capital"].add(
            costs["marginal"].rename({t: t + " marginal" for t in conv_techs}),
            fill_value=0.0,
        )

    return costs


def progress_retrieve(url, file):
    import urllib

    from progressbar import ProgressBar

    pbar = ProgressBar(0, 100)

    def dlProgress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        pbar.update(min(percent, 100))

    urllib.request.urlretrieve(url, file, reporthook=dlProgress)


PUDL_S3_ENDPOINT = "https://s3.us-west-2.amazonaws.com"


def _http_proxy_from_env():
    """Return the ``host:port`` of the configured HTTP proxy, or None."""
    for var in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        url = os.environ.get(var)
        if url:
            return url.split("://", 1)[-1].rstrip("/")
    return None


def _duckdb_accepts_proxy(duckdb) -> bool:
    """DuckDB grew the ``http_proxy`` setting in 0.10.3."""
    try:
        version = tuple(int(part) for part in duckdb.__version__.split(".")[:3])
    except ValueError:
        return False
    return version >= (0, 10, 3)


class PudlSource:
    """
    Hands DuckDB a readable location for the PUDL parquet files.

    Interpolate it into a query the way a plain path would be --
    ``read_parquet('{pudl}/some_table.parquet')`` -- and run that query through
    :meth:`query`.  Which files a run needs is then stated once, in the SQL.

    DuckDB's httpfs extension ignores the ``http_proxy`` environment variables,
    and before 0.10.3 it cannot be pointed at a proxy at all.  Where the only
    route out is a proxy -- UMich Great Lakes, for instance -- reading ``s3://``
    blocks until the scheduler kills the job rather than raising.  ``requests``
    does honour those variables, so there the files a query names are mirrored
    to local disk first and the query rewritten to read the mirror.  Set
    ``PUDL_CACHE_DIR`` to place that mirror (default ``~/.cache/pudl``).
    """

    def __init__(self, pudl_path: str):
        import duckdb

        self.pudl_path = pudl_path.rstrip("/")
        self.remote = self.pudl_path.startswith("s3://")
        duckdb.connect(database=":memory:", read_only=False)

        proxy = _http_proxy_from_env()
        self.mirror = self.remote and bool(proxy) and not _duckdb_accepts_proxy(duckdb)
        if self.mirror:
            logger.warning(
                f"DuckDB {duckdb.__version__} cannot be pointed at a proxy ({proxy}); "
                "mirroring the PUDL parquet files this run needs to local disk.",
            )
        elif self.remote:
            duckdb.query("INSTALL httpfs;")
            duckdb.query("LOAD httpfs;")
            duckdb.query("SET s3_region='us-west-2';")
            if proxy:
                duckdb.query(f"SET http_proxy='{proxy}';")

    def __str__(self) -> str:
        return self.pudl_path

    def query(self, sql: str):
        """Run ``sql``, first making every parquet file it names reachable."""
        import duckdb

        if self.mirror:
            sql = re.sub(
                re.escape(self.pudl_path) + r"/([^'\"\s]+\.parquet)",
                lambda match: self._localize(match.group(1)),
                sql,
            )
        return duckdb.query(sql)

    def _localize(self, filename: str) -> str:
        """Download ``filename`` unless it is already cached; return its local path."""
        root = os.environ.get("PUDL_CACHE_DIR") or Path.home() / ".cache" / "pudl"
        cache = Path(root) / self.pudl_path.rsplit("/", 1)[-1]
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / filename

        if not (target.exists() and target.stat().st_size):
            url = f"{PUDL_S3_ENDPOINT}/{self.pudl_path[len('s3://') :]}/{filename}"
            logger.info(f"Caching {filename} from {url}")
            # A unique suffix keeps parallel jobs off each other's partial files.
            partial = target.with_name(f"{filename}.{os.getpid()}.part")
            try:
                with requests.get(url, stream=True, timeout=(30, 300)) as response:
                    response.raise_for_status()
                    with open(partial, "wb") as handle:
                        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                            handle.write(chunk)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)

        return target.as_posix()


def get_aggregation_strategies(aggregation_strategies):
    # default aggregation strategies that cannot be defined in .yaml format must be specified within
    # the function, otherwise (when defaults are passed in the function's definition) they get lost
    # when custom values are specified in the config.

    import numpy as np
    from pypsa.clustering.spatial import _make_consense

    bus_strategies = dict(country=_make_consense("Bus", "country"))
    bus_strategies.update(aggregation_strategies.get("buses", {}))

    generator_strategies = {"build_year": lambda x: 0, "lifetime": lambda x: np.inf}
    generator_strategies.update(aggregation_strategies.get("generators", {}))

    return bus_strategies, generator_strategies


def export_network_for_gis_mapping(n, output_path):
    # Creating GIS Table for Mapping Lines in QGIS
    lines_gis = n.lines.copy()
    lines_gis["latitude1"] = n.buses.loc[lines_gis.bus0].y.values
    lines_gis["longitude1"] = n.buses.loc[lines_gis.bus0].x.values
    lines_gis["latitude2"] = n.buses.loc[lines_gis.bus1].y.values
    lines_gis["longitude2"] = n.buses.loc[lines_gis.bus1].x.values
    lines_gis["v_nom"] = n.buses.loc[lines_gis.bus0].v_nom.values
    lines_gis["wkt_geom"] = (
        "LINESTRING ("
        + lines_gis.longitude1.astype(str)
        + " "
        + lines_gis.latitude1.astype(str)
        + ", "
        + lines_gis.longitude2.astype(str)
        + " "
        + lines_gis.latitude2.astype(str)
        + ")"
    )

    lines_gis.to_csv(output_path + "_lines_GIS.csv")

    # Creating GIS Table for Mapping Buses in QGIS
    buses_gis = n.buses.copy()
    buses_gis.to_csv(output_path + "_buses_GIS.csv")


def mock_snakemake(rulename, **wildcards):
    """
    Function is expected to be executed from the 'scripts'-directory of '
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards.

    Parameters
    ----------
    rulename: str
        name of the rule for which the snakemake object should be generated
    **wildcards:
        keyword arguments fixing the wildcards. Only necessary if wildcards are
        needed.
    """
    import os

    import snakemake as sm
    from packaging.version import Version, parse
    from pypsa.descriptors import Dict
    from snakemake.script import Snakemake

    script_dir = Path(__file__).parent.resolve()
    assert Path.cwd().resolve() == script_dir, (
        f"mock_snakemake has to be run from the repository scripts directory {script_dir}"
    )
    os.chdir(script_dir.parent)
    for p in sm.SNAKEFILE_CHOICES:
        if os.path.exists(p):
            snakefile = p
            break
    kwargs = dict(rerun_triggers=[]) if parse(sm.__version__) > Version("7.7.0") else {}
    workflow = sm.Workflow(snakefile, overwrite_configfiles=[], **kwargs)
    workflow.include(snakefile)
    workflow.global_resources = {}
    rule = workflow.get_rule(rulename)
    dag = sm.dag.DAG(workflow, rules=[rule])
    wc = Dict(wildcards)
    job = sm.jobs.Job(rule, dag, wc)

    def make_accessable(*ios):
        for io in ios:
            for i in range(len(io)):
                io[i] = os.path.abspath(io[i])

    make_accessable(job.input, job.output, job.log)
    snakemake = Snakemake(
        job.input,
        job.output,
        job.params,
        job.wildcards,
        job.threads,
        job.resources,
        job.log,
        job.dag.workflow.config,
        job.rule.name,
        None,
    )
    # create log and output dir if not existent
    for path in list(snakemake.log) + list(snakemake.output):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    os.chdir(script_dir)
    return snakemake


def validate_checksum(file_path, zenodo_url=None, checksum=None):
    """
    Validate file checksum against provided or Zenodo-retrieved checksum.
    Calculates the hash of a file using 64KB chunks. Compares it against a
    given checksum or one from a Zenodo URL.

    Parameters
    ----------
    file_path : str
        Path to the file for checksum validation.
    zenodo_url : str, optional
        URL of the file on Zenodo to fetch the checksum.
    checksum : str, optional
        Checksum (format 'hash_type:checksum_value') for validation.

    Raises
    ------
    AssertionError
        If the checksum does not match, or if neither `checksum` nor `zenodo_url` is provided.


    Examples
    --------
    >>> validate_checksum("/path/to/file", checksum="md5:abc123...")
    >>> validate_checksum(
    ...     "/path/to/file",
    ...     zenodo_url="https://zenodo.org/record/12345/files/example.txt",
    ... )

    If the checksum is invalid, an AssertionError will be raised.
    """
    assert checksum or zenodo_url, "Either checksum or zenodo_url must be provided"
    if zenodo_url:
        checksum = get_checksum_from_zenodo(zenodo_url)
    hash_type, checksum = checksum.split(":")
    hasher = hashlib.new(hash_type)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):  # 64kb chunks
            hasher.update(chunk)
    calculated_checksum = hasher.hexdigest()
    assert calculated_checksum == checksum, (
        "Checksum is invalid. This may be due to an incomplete download. Delete the file and re-execute the rule."
    )


def get_checksum_from_zenodo(file_url):
    parts = file_url.split("/")
    record_id = parts[parts.index("record") + 1]
    filename = parts[-1]

    response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=30)
    response.raise_for_status()
    data = response.json()

    for file in data["files"]:
        if file["key"] == filename:
            return file["checksum"]
    return None


### Config Related Helpers ###


def set_case_config(snakemake):
    """Merge case-specific overrides into snakemake.config."""
    cases = snakemake.config.get("cases", {})
    if cases and "case" in snakemake.wildcards.keys():
        case_name = snakemake.wildcards.case
        if case_name not in cases:
            raise KeyError(f"Case '{case_name}' not found in config/cases_backup.yaml.")
        update_config(snakemake.config, copy.deepcopy(cases[case_name] or {}))


def set_scenario_config(snakemake):
    """Backward-compatible alias for case-aware config overrides."""
    set_case_config(snakemake)


def get_opt(opts, expr, flags=None):
    """
    Return the first option matching the regular expression.

    The regular expression is case-insensitive by default.
    """
    if flags is None:
        flags = re.IGNORECASE
    for o in opts:
        match = re.match(expr, o, flags=flags)
        if match:
            return match.group(0)
    return None


def find_opt(opts, expr):
    """Return if available the float after the expression."""
    for o in opts:
        if expr in o:
            m = re.findall(r"m?\d+(?:[\.p]\d+)?", o)
            if len(m) > 0:
                return True, float(m[-1].replace("p", ".").replace("m", "-"))
            else:
                return True, None
    return False, None


def update_config_from_wildcards(config, w, inplace=True):
    """Parses configuration settings from wildcards and updates the config."""
    from packaging.version import parse

    if not inplace:
        config = copy.deepcopy(config)

    if w.get("opts"):
        opts = w.opts.split("-")

        if nhours := get_opt(opts, r"^\d+(h|seg)$"):
            config["clustering"]["temporal"]["resolution_elec"] = nhours

        co2l_enable, co2l_value = find_opt(opts, "Co2L")
        if co2l_enable:
            config["electricity"]["co2limit_enable"] = True
            if co2l_value is not None:
                config["electricity"]["co2limit"] = co2l_value * config["electricity"]["co2base"]

        gasl_enable, gasl_value = find_opt(opts, "CH4L")
        if gasl_enable:
            config["electricity"]["gaslimit_enable"] = True
            if gasl_value is not None:
                config["electricity"]["gaslimit"] = gasl_value * 1e6

        attr_lookup = {
            "p": "p_nom_max",
            "e": "e_nom_max",
            "c": "capital_cost",
            "m": "marginal_cost",
        }
        for o in opts:
            flags = ["+e", "+p", "+m", "+c"]
            if all(flag not in o for flag in flags):
                continue
            carrier, attr_factor = o.split("+")
            attr = attr_lookup[attr_factor[0]]
            factor = float(attr_factor[1:])
            if not isinstance(config["adjustments"]["electricity"], dict):
                config["adjustments"]["electricity"] = dict()
            update_config(
                config["adjustments"]["electricity"],
                {attr: {carrier: factor}},
            )

    if w.get("sector_opts"):
        opts = w.sector_opts.split("-")

        if "T" in opts:
            config["sector"]["transport"] = True

        if "H" in opts:
            config["sector"]["heating"] = True

        if "B" in opts:
            config["sector"]["biomass"] = True

        if "I" in opts:
            config["sector"]["industry"] = True

        if "A" in opts:
            config["sector"]["agriculture"] = True

        if "TCT" in opts:
            config["solving"]["constraints"]["TCT"] = True

        eq_value = get_opt(opts, r"^EQ+\d*\.?\d+(c|)")
        for o in opts:
            if eq_value is not None:
                config["solving"]["constraints"]["EQ"] = eq_value
            elif "EQ" in o:
                config["solving"]["constraints"]["EQ"] = True
            break

        if "BAU" in opts:
            config["solving"]["constraints"]["BAU"] = True

        if "SAFE" in opts:
            config["solving"]["constraints"]["SAFE"] = True

        if nhours := get_opt(opts, r"^\d+(h|sn|seg)$"):
            config["clustering"]["temporal"]["resolution_sector"] = nhours

        if "decentral" in opts:
            config["sector"]["electricity_transmission_grid"] = False

        if "noH2network" in opts:
            config["sector"]["H2_network"] = False

        if "nowasteheat" in opts:
            config["sector"]["use_fischer_tropsch_waste_heat"] = False
            config["sector"]["use_methanolisation_waste_heat"] = False
            config["sector"]["use_haber_bosch_waste_heat"] = False
            config["sector"]["use_methanation_waste_heat"] = False
            config["sector"]["use_fuel_cell_waste_heat"] = False
            config["sector"]["use_electrolysis_waste_heat"] = False

        if "nodistrict" in opts:
            config["sector"]["district_heating"]["progress"] = 0.0

        dg_enable, dg_factor = find_opt(opts, "dist")
        if dg_enable:
            config["sector"]["electricity_distribution_grid"] = True
            if dg_factor is not None:
                config["sector"]["electricity_distribution_grid_cost_factor"] = dg_factor

        if "biomasstransport" in opts:
            config["sector"]["biomass_transport"] = True

        _, maxext = find_opt(opts, "linemaxext")
        if maxext is not None:
            config["lines"]["max_extension"] = maxext * 1e3
            config["links"]["max_extension"] = maxext * 1e3

        _, co2l_value = find_opt(opts, "Co2L")
        if co2l_value is not None:
            config["co2_budget"] = float(co2l_value)

        if co2_distribution := get_opt(opts, r"^(cb)\d+(\.\d+)?(ex|be)$"):
            config["co2_budget"] = co2_distribution

        if co2_budget := get_opt(opts, r"^(cb)\d+(\.\d+)?$"):
            config["co2_budget"] = float(co2_budget[2:])

        attr_lookup = {
            "p": "p_nom_max",
            "e": "e_nom_max",
            "c": "capital_cost",
            "m": "marginal_cost",
        }
        for o in opts:
            flags = ["+e", "+p", "+m", "+c"]
            if all(flag not in o for flag in flags):
                continue
            carrier, attr_factor = o.split("+")
            attr = attr_lookup[attr_factor[0]]
            factor = float(attr_factor[1:])
            if not isinstance(config["adjustments"]["sector"], dict):
                config["adjustments"]["sector"] = dict()
            update_config(config["adjustments"]["sector"], {attr: {carrier: factor}})

        _, sdr_value = find_opt(opts, "sdr")
        if sdr_value is not None:
            config["costs"]["social_discountrate"] = sdr_value / 100

        _, seq_limit = find_opt(opts, "seq")
        if seq_limit is not None:
            config["sector"]["co2_sequestration_potential"] = seq_limit

        # any config option can be represented in wildcard
        for o in opts:
            if o.startswith("CF+"):
                infix = o.split("+")[1:]
                update_config(config, parse(infix))

    if not inplace:
        return config


def get_scenarios(run):
    scenario_config = run.get("scenarios", {})
    if run["name"] and scenario_config.get("enable"):
        fn = Path(scenario_config["file"])
        if fn.exists():
            scenarios = yaml.safe_load(fn.read_text())
            if run["name"] == "all":
                run["name"] = list(scenarios.keys())
            return scenarios
    return {}


def get_rdir(run):
    scenario_config = run.get("scenarios", {})
    if run["name"] and scenario_config.get("enable"):
        rdir = "{run}/"
    elif run["name"]:
        rdir = run["name"] + "/"
    else:
        rdir = ""

    prefix = run.get("prefix", "")
    if prefix:
        rdir = f"{prefix}/{rdir}"

    return rdir


def get_run_path(fn, dir, rdir, shared_resources):
    """
    Dynamically provide paths based on shared resources and filename.

    Use this function for snakemake rule inputs or outputs that should be
    optionally shared across runs or created individually for each run.

    Parameters
    ----------
    fn : str
        The filename for the path to be generated.
    dir : str
        The base directory.
    rdir : str
        Relative directory for non-shared resources.
    shared_resources : str or bool
        Specifies which resources should be shared.
        - If string is "base", special handling for shared "base" resources (see notes).
        - If random string other than "base", this folder is used instead of the `rdir` keyword.
        - If boolean, directly specifies if the resource is shared.

    Returns
    -------
    str
        Full path where the resource should be stored.

    Notes
    -----
    Special case for "base" allows no wildcards other than "technology", "year"
    and "scope" and excludes filenames starting with "networks/elec" or
    "add_electricity". All other resources are shared.
    """
    if shared_resources == "base":
        pattern = r"\{([^{}]+)\}"
        existing_wildcards = set(re.findall(pattern, fn))
        irrelevant_wildcards = {"technology", "year", "scope", "kind"}
        no_relevant_wildcards = not existing_wildcards - irrelevant_wildcards
        no_elec_rule = not fn.startswith("networks/elec") and not fn.startswith(
            "add_electricity",
        )
        is_shared = no_relevant_wildcards and no_elec_rule
        rdir = "" if is_shared else rdir
    elif isinstance(shared_resources, str):
        rdir = shared_resources + "/"
    elif isinstance(shared_resources, bool):
        rdir = "" if shared_resources else rdir
    else:
        raise ValueError(
            "shared_resources must be a boolean, str, or 'base' for special handling.",
        )

    return f"{dir}{rdir}{fn}"


def path_provider(dir, rdir, shared_resources):
    """
    Returns a partial function that dynamically provides paths based on shared
    resources and the filename.

    Returns
    -------
    partial function
        A partial function that takes a filename as input and
        returns the path to the file based on the shared_resources parameter.
    """
    return partial(get_run_path, dir=dir, rdir=rdir, shared_resources=shared_resources)


def get_snapshots(
    snapshots: dict[str, str],
    drop_leap_day: bool = True,
    freq: str = "h",
    **kwargs,
) -> pd.date_range:
    """
    Returns pandas DateTimeIndex potentially without leap days.

    Taken from PyPSA-Eur implementation
    """
    time = pd.date_range(freq=freq, **snapshots, **kwargs)
    if drop_leap_day and time.is_leap_year.any():
        time = time[~((time.month == 2) & (time.day == 29))]

    return time


def get_weather_year_snapshots(
    weather_years: list[int] | tuple[int, ...],
    *,
    drop_leap_day: bool = True,
    freq: str = "h",
) -> pd.DatetimeIndex:
    """Return concatenated, timezone-naive snapshots for explicit weather years.

    Leap days are removed with the same ``get_snapshots`` implementation used by
    the reference workflow.  Gaps in the requested weather-year sequence are
    deliberately retained; callers can therefore distinguish 2013 from 2016.
    """
    snapshots = pd.DatetimeIndex([], name="timestep")
    for weather_year in weather_years:
        year = int(weather_year)
        snapshots = snapshots.append(
            get_snapshots(
                {
                    "start": f"{year}-01-01 00:00:00",
                    "end": f"{year + 1}-01-01 00:00:00",
                    "inclusive": "left",
                },
                drop_leap_day=drop_leap_day,
                freq=freq,
            ),
        )
    return snapshots


def weighted_avg(df, values, weights):
    """
    Return the weighted average of a DataFrame column(s) `values` with weights
    `weights`.
    """
    valid = df[values].notna()
    if valid.sum() == 0:
        return np.nan  # Return NaN if no valid entries
    return np.average(df[values][valid], weights=df[weights][valid])


def get_multiindex_snapshots(
    sns_config: dict[str, str],
    invest_periods: list[int],
    weather_years: list[int] | tuple[int, ...] | None = None,
) -> pd.MultiIndex:
    if weather_years:
        weather_snapshots = get_weather_year_snapshots(weather_years)
        periods = np.repeat([int(year) for year in invest_periods], len(weather_snapshots))
        timesteps = weather_snapshots.to_numpy().tolist() * len(invest_periods)
        return pd.MultiIndex.from_arrays(
            [periods, pd.DatetimeIndex(timesteps)],
            names=["period", "timestep"],
        )

    sns = pd.DatetimeIndex([])
    for year in invest_periods:
        sns = sns.append(
            get_snapshots(sns_config).map(lambda x: x.replace(year=year)),
        )
    return pd.MultiIndex.from_arrays(
        [sns.year, sns],
        names=["period", "timestep"],
    )


def get_currency_conversion_factor(year, currency="EUR"):
    """Convert nominal EUR/USD costs to 2022 USD using reference factors."""
    eur_usd_avg = {
        2010: 1.3261,
        2011: 1.3931,
        2012: 1.2859,
        2013: 1.3281,
        2014: 1.3297,
        2015: 1.1096,
        2016: 1.1072,
        2017: 1.1301,
        2018: 1.1817,
        2019: 1.1194,
        2020: 1.1410,
        2021: 1.1830,
        2022: 1.0534,
        2023: 1.0817,
        2024: 1.0820,
        2025: 1.1687,
    }
    cpi_us = {
        # ReEDS reports transmission costs in 2004 USD, so the index reaches back
        # that far even though no EUR exchange rate is available for 2004.
        # CPI-U annual averages 188.9 (2004) / 292.655 (2022), rebased to 2022=100,
        # which makes the 2004 -> 2022 factor 1.549.
        2004: 64.547,
        2010: 74.5,
        2011: 76.9,
        2012: 78.5,
        2013: 79.6,
        2014: 80.9,
        2015: 81.0,
        2016: 82.0,
        2017: 83.8,
        2018: 85.8,
        2019: 87.4,
        2020: 88.4,
        2021: 92.6,
        2022: 100.0,
        2023: 104.1,
        2024: 107.2,
        2025: 110.1,
    }
    year = int(year)
    if year not in cpi_us:
        raise ValueError(f"Year {year} has no US CPI reference; supported years are {sorted(cpi_us)}.")
    currency = currency.upper()
    if currency == "EUR":
        if year not in eur_usd_avg:
            raise ValueError(
                f"No EUR/USD exchange rate for {year}; supported years are {sorted(eur_usd_avg)}.",
            )
        exchange_rate = eur_usd_avg[year]
    elif currency == "USD":
        exchange_rate = 1.0
    else:
        raise ValueError(f"Unsupported currency: {currency}. Use 'EUR' or 'USD'.")
    return exchange_rate * cpi_us[2022] / cpi_us[year]
