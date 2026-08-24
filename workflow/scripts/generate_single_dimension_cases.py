"""Generate one-factor-at-a-time cases from workflow/config/cases_backup.yaml templates."""

import argparse
from pathlib import Path

from generate_orthogonal_cases import (
    DC_BY_YEAR,
    DC_SETTINGS,
    DEMAND_LEVELS,
    DEMAND_NAME,
    LL_BY_YEAR,
    SSSC_SETTINGS,
    SSSC_TOTAL_MAX_SETTINGS,
    WORKFLOW_DIR,
    build_case_name,
    build_single_case,
    dump_yaml,
    extract_year_templates,
    load_yaml,
)

def _matches_subset(actual: dict, expected: dict) -> bool:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                return False
            if not _matches_subset(actual_value, expected_value):
                return False
        elif actual_value != expected_value:
            return False
    return True


def infer_demand_level(template: dict, year: int) -> str:
    demand_level = template.get("scenario", {}).get("demand_level")
    for name, label in DEMAND_NAME.items():
        if label == demand_level:
            return name
    raise ValueError(f"Could not infer baseline demand level for {year} template.")


def infer_ll(template: dict, year: int) -> str:
    ll_values = template.get("scenario", {}).get("ll", [])
    if len(ll_values) != 1:
        raise ValueError(f"Expected exactly one baseline ll for {year} template, got: {ll_values}")

    ll = ll_values[0]
    if ll not in LL_BY_YEAR[year]:
        raise ValueError(f"Baseline ll '{ll}' is not in configured ll list for {year}.")
    return ll


def infer_dc(template: dict, year: int) -> str:
    links = template.get("links", {})
    for dc_name, dc_settings in DC_SETTINGS.items():
        if _matches_subset(links, dc_settings):
            return dc_name
    raise ValueError(f"Could not infer baseline DC setting for {year} template.")


def infer_sssc_tot_max(template: dict) -> str:
    sssc_tot_max = template.get("lines", {}).get("convert_lines_to_line_x", {}).get("sssc_tot_max", float("inf"))
    for sssc_tot_max_name, sssc_tot_max_value in SSSC_TOTAL_MAX_SETTINGS.items():
        if sssc_tot_max == sssc_tot_max_value:
            return sssc_tot_max_name
    raise ValueError(f"Could not infer baseline sssc_tot_max setting from value '{sssc_tot_max}'.")


def infer_baseline_dimensions(template: dict, year: int) -> dict:
    return {
        "demand_level": infer_demand_level(template, year),
        "ll": infer_ll(template, year),
        "dc": infer_dc(template, year),
        "sssc_tot_max": infer_sssc_tot_max(template),
    }


def build_case_from_dimensions(template: dict, year: int, dimensions: dict) -> tuple[str, dict]:
    case_name = build_case_name(
        year=year,
        demand_level=dimensions["demand_level"],
        ll=dimensions["ll"],
        dc=dimensions["dc"],
        sssc=dimensions["sssc"],
        sssc_tot_max=dimensions["sssc_tot_max"],
    )
    case = build_single_case(
        template=template,
        year=year,
        demand_level=dimensions["demand_level"],
        ll=dimensions["ll"],
        dc=dimensions["dc"],
        sssc=dimensions["sssc"],
        sssc_tot_max=dimensions["sssc_tot_max"],
    )
    return case_name, case


def add_sssc_variants(
    generated_cases: dict,
    template: dict,
    year: int,
    dimensions: dict,
    sssc_options: tuple[str, ...] = tuple(SSSC_SETTINGS.keys()),
) -> None:
    for sssc in sssc_options:
        variant = dict(dimensions, sssc=sssc)
        case_name, case = build_case_from_dimensions(template, year, variant)
        generated_cases[case_name] = case


def add_high_cost_variant(
    generated_cases: dict,
    template: dict,
    year: int,
    baseline_dimensions: dict,
) -> None:
    variant = dict(baseline_dimensions, sssc="SSSC")
    case_name, case = build_case_from_dimensions(template, year, variant)
    convert_lines = case.setdefault("lines", {}).setdefault("convert_lines_to_line_x", {})
    convert_lines["capex_per_kva"] = 80
    convert_lines["cost_recovery_period_years"] = 15
    generated_cases[f"{case_name}_HighCost"] = case


def generate_cases(templates: dict[int, dict]) -> dict:
    generated_cases = {}

    for year, template in templates.items():
        baseline = infer_baseline_dimensions(template, year)
        add_sssc_variants(generated_cases, template, year, baseline)
        add_high_cost_variant(generated_cases, template, year, baseline)

        for demand_level in DEMAND_LEVELS:
            if demand_level == baseline["demand_level"]:
                continue
            variant = dict(baseline, demand_level=demand_level)
            add_sssc_variants(generated_cases, template, year, variant)

        for ll in LL_BY_YEAR[year]:
            if ll == baseline["ll"]:
                continue
            variant = dict(baseline, ll=ll)
            add_sssc_variants(generated_cases, template, year, variant)

        for dc in DC_BY_YEAR[year]:
            if dc == baseline["dc"]:
                continue
            variant = dict(baseline, dc=dc)
            add_sssc_variants(generated_cases, template, year, variant)

        for sssc_tot_max in SSSC_TOTAL_MAX_SETTINGS:
            if sssc_tot_max == baseline["sssc_tot_max"]:
                continue
            variant = dict(baseline, sssc_tot_max=sssc_tot_max)
            add_sssc_variants(generated_cases, template, year, variant, sssc_options=("SSSC",))

    return generated_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate single-dimension case yaml.")
    parser.add_argument(
        "--input",
        type=Path,
        default=WORKFLOW_DIR / "config" / "cases_backup.yaml",
        help="Input cases yaml used to extract 2030/2050 templates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKFLOW_DIR / "config" / "cases.yaml",
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
