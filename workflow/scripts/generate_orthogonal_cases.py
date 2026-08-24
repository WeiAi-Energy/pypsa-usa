"""Generate orthogonal cases from workflow/config/cases_backup.yaml templates."""

import argparse
import copy
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
WORKFLOW_DIR = SCRIPT_DIR.parent

YEARS = [2050]
DEMAND_LEVELS = ["low", "mid", "high"]
DEMAND_NAME = {"low": "Low", "mid": "Mid", "high": "High"}
OPTS_BY_YEAR = {
    2030: ["TCT-RPS-ERM-3h"],
    2050: ["TCT-RPS-ERM-3h"],
}
LL_BY_YEAR = {
    2030: ["v1.0", "v1.035", "v1.07", "v1.14", "vinf"],
    2050: ["v1.0", "v1.15", "v1.30", "v1.60", "vinf"],
}
DC_SETTINGS = {
    "NoDCNet": {"new_dc_network": False},
    "DCNet": {"new_dc_network": True},
}
SSSC_SETTINGS = {
    "NoSSSC": False,
    "SSSC": True,
}
DC_BY_YEAR = {
    2030: ["NoDCNet"],
    2050: list(DC_SETTINGS),
}
SSSC_TOTAL_MAX_SETTINGS = {
    "1GVA": 1000,
    "5GVA": 5000,
    "10GVA": 10000,
    "50GVA": 50000,
    "InfGVA": float("inf"),
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class InlineListDumper(yaml.SafeDumper):
    """Dumper that always writes Python lists in flow style: [a, b, c]."""


def _represent_list_inline(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


InlineListDumper.add_representer(list, _represent_list_inline)


def dump_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep two blank lines between each top-level case entry for readability.
    if list(data.keys()) == ["cases"] and isinstance(data["cases"], dict):
        blocks = ["cases:"]
        items = list(data["cases"].items())
        for i, (case_name, case_value) in enumerate(items):
            block = yaml.dump(
                {case_name: case_value},
                sort_keys=False,
                default_flow_style=False,
                width=100000,
                Dumper=InlineListDumper,
            ).rstrip("\n")
            blocks.extend(f"  {line}" for line in block.splitlines())
            if i < len(items) - 1:
                blocks.append("")
                blocks.append("")
        text = "\n".join(blocks) + "\n"
        path.write_text(text, encoding="utf-8")
        return

    with path.open("w", encoding="utf-8") as file:
        yaml.dump(
            data,
            file,
            sort_keys=False,
            default_flow_style=False,
            width=100000,
            Dumper=InlineListDumper,
        )


def extract_year_templates(cases: dict) -> dict[int, dict]:
    templates: dict[int, dict] = {}
    for case_config in cases.values():
        planning_horizons = case_config.get("scenario", {}).get("planning_horizons", [])
        if not planning_horizons:
            continue

        year = int(planning_horizons[0])
        if year in YEARS and year not in templates:
            templates[year] = case_config

    missing_years = [year for year in YEARS if year not in templates]
    if missing_years:
        raise ValueError(
            f"Could not find template case for year(s): {missing_years}. "
            "Please keep at least one 2030 case and one 2050 case in the input yaml.",
        )
    return templates


def build_case_name(
    year: int,
    demand_level: str,
    ll: str,
    dc: str,
    sssc: str,
    sssc_tot_max: str | None = None,
) -> str:
    name_parts = [f"{year}_{DEMAND_NAME[demand_level]}Dmd", ll, dc, sssc]
    if sssc == "SSSC":
        name_parts.append(sssc_tot_max)
    return "_".join(name_parts)


def build_single_case(
    template: dict,
    year: int,
    demand_level: str,
    ll: str,
    dc: str,
    sssc: str,
    sssc_tot_max: str | None = None,
) -> dict:
    case = copy.deepcopy(template)

    scenario = case.setdefault("scenario", {})
    scenario["planning_horizons"] = [year]
    scenario["ll"] = [ll]
    scenario["opts"] = copy.deepcopy(OPTS_BY_YEAR[year])
    scenario["demand_level"] = DEMAND_NAME[demand_level]

    links = case.setdefault("links", {})
    links.update(copy.deepcopy(DC_SETTINGS[dc]))

    lines = case.setdefault("lines", {})
    convert_lines = lines.setdefault("convert_lines_to_line_x", {})
    convert_lines["enable"] = SSSC_SETTINGS[sssc]
    if sssc == "SSSC":
        convert_lines["sssc_tot_max"] = SSSC_TOTAL_MAX_SETTINGS[sssc_tot_max]
    else:
        convert_lines.pop("sssc_tot_max", None)

    return case


def generate_cases(templates: dict[int, dict]) -> dict:
    generated_cases = {}
    for year in YEARS:
        template = templates[year]
        for demand_level in DEMAND_LEVELS:
            for ll in LL_BY_YEAR[year]:
                    for dc in DC_BY_YEAR[year]:
                        for sssc in SSSC_SETTINGS:
                            sssc_tot_max_options = SSSC_TOTAL_MAX_SETTINGS if sssc == "SSSC" else [None]
                            for sssc_tot_max in sssc_tot_max_options:
                                case_name = build_case_name(
                                    year,
                                    demand_level,
                                    ll,
                                    dc,
                                    sssc,
                                    sssc_tot_max=sssc_tot_max,
                                )
                                generated_cases[case_name] = build_single_case(
                                    template=template,
                                    year=year,
                                    demand_level=demand_level,
                                    ll=ll,
                                    dc=dc,
                                    sssc=sssc,
                                    sssc_tot_max=sssc_tot_max,
                                )
    return generated_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate orthogonal case yaml.")
    parser.add_argument(
        "--input",
        type=Path,
        default=WORKFLOW_DIR / "config" / "cases_backup.yaml",
        help="Input cases yaml used to extract 2030/2050 templates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKFLOW_DIR / "config" / "cases.generated.yaml",
        help="Output yaml path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.input)

    if "cases" not in config:
        raise KeyError(f"Input file {args.input} does not contain top-level key: 'cases'")

    templates = extract_year_templates(config["cases"])
    generated = {"cases": generate_cases(templates)}
    dump_yaml(generated, args.output)

    print(f"Generated {len(generated['cases'])} cases -> {args.output}")


if __name__ == "__main__":
    main()
