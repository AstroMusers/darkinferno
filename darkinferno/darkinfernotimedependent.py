#!/usr/bin/python

import argparse
import json
import math
from pathlib import Path
from tdpy import narrate, write_text

from scenarios import SCENARIOS, get_scenario


def positive_float(value):
    """Parse a positive floating-point argument."""

    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def finite_float(value):
    """Parse a finite floating-point argument."""

    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def positive_integer(value):
    """Parse a positive integer argument."""

    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def parse_args():
    """Parse time-dependent command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evolve a planetary thermal domain under dark matter heating."
    )
    parser.add_argument(
        "log10_dark_matter_mass",
        type=finite_float,
        help="base-10 logarithm of the dark matter mass in GeV",
    )
    parser.add_argument("heat_flow", type=positive_float, help="heat injection in TW")
    parser.add_argument(
        "initial_temperature", type=positive_float, help="initial temperature in K"
    )
    parser.add_argument(
        "conductivity", type=positive_float, help="thermal conductivity in W m^-1 K^-1"
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="earth-inner-core",
        help="body and thermal scenario (default: %(default)s)",
    )
    parser.add_argument(
        "--end-time-years",
        type=positive_float,
        default=4.5e9,
        help="maximum evolution time in years (default: %(default)s)",
    )
    parser.add_argument(
        "--time-step-years",
        type=positive_float,
        default=1e6,
        help="implicit-solver timestep in years (default: %(default)s)",
    )
    parser.add_argument(
        "--radial-cells",
        type=positive_integer,
        default=300,
        help="number of radial grid cells (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-every",
        type=positive_integer,
        default=10,
        help="record every N timesteps (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-width-k",
        type=positive_float,
        default=50.0,
        help="Gaussian latent-heat smoothing width in K (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, help="write the structured result to this JSON file"
    )
    parser.add_argument(
        "--typeverb",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="narration level: 0 critical, 1 nominal, 2 verbose (default: %(default)s)",
    )
    arguments = parser.parse_args()

    return arguments


args = parse_args()

from physics import (
    GEV_IN_KG,
    dark_matter_scale_radius,
    evolve_temperature,
    gaussian_enclosed_fraction,
)

scenario = get_scenario(args.scenario)
mdm = 10**args.log10_dark_matter_mass * GEV_IN_KG  # [kg]
heatflow = args.heat_flow * 1e12  # [W]
t0 = args.initial_temperature  # [K]
k0 = args.conductivity  # [W m^-1 K^-1]
initial_scale_radius = dark_matter_scale_radius(t0, scenario.density_kg_m3, mdm)  # [m]
initial_enclosed_fraction = gaussian_enclosed_fraction(
    scenario.thermal_radius_m, initial_scale_radius
)
narrate(args.typeverb, f"Preparing scenario {scenario.name}...")
narrate(args.typeverb, f"  Total deposited power = {heatflow:g} W", level=2)
narrate(args.typeverb, f"  Initial temperature = {t0:g} K", level=2)
narrate(args.typeverb, "Evolving the temperature profile...")

result = evolve_temperature(
    scenario,
    mdm,
    heatflow,
    t0,
    k0,
    args.end_time_years,
    args.time_step_years,
    radial_cells=args.radial_cells,
    sample_every=args.sample_every,
    typeverb=args.typeverb,
    latent_width_k=args.latent_width_k,
)
narrate(args.typeverb, "Thermal evolution complete.")

if result.phase_crossing_years is None:
    print("The diagnostic radius did not reach the melting temperature.")
else:
    print(
        f"Temperature at {scenario.diagnostic_radius_m / 1000.0} km reaches "
        f"{scenario.phase_temperature_k} K after {result.phase_crossing_years} years"
    )
print(f"Final phase-proxy exceedance radius is {result.final_phase_radius_m} meters")
print(
    f"Initial deposited-power fraction inside thermal domain = {initial_enclosed_fraction:.6g}"
)
if initial_enclosed_fraction < 0.9:
    print(
        "Warning: less than 90% of the Gaussian source lies inside the thermal domain."
    )
for warning in scenario.warnings:
    print(f"warning: {warning}")

record = {
    "scenario": scenario.name,
    "body": scenario.body_name,
    "log10_dark_matter_mass_gev": args.log10_dark_matter_mass,
    "heat_flow_tw": args.heat_flow,
    "total_deposited_power_w": heatflow,
    "initial_thermal_domain_power_fraction": initial_enclosed_fraction,
    "initial_thermal_domain_deposited_power_w": initial_enclosed_fraction * heatflow,
    "initial_temperature_k": t0,
    "conductivity_w_m_k": k0,
    "phase_temperature_k": scenario.phase_temperature_k,
    "diagnostic_radius_m": scenario.diagnostic_radius_m,
    "phase_crossing_years": result.phase_crossing_years,
    "final_phase_radius_m": result.final_phase_radius_m,
    "final_central_temperature_k": result.final_central_temperature_k,
    "radial_step_m": result.radial_step_m,
    "time_step_years": result.time_step_years,
    "latent_width_k": args.latent_width_k,
    "sample_years": result.sample_years,
    "diagnostic_temperatures_k": result.diagnostic_temperatures_k,
    "provenance": scenario.provenance,
    "warnings": scenario.warnings,
}

if args.output:
    write_text(args.output, json.dumps(record, indent=2) + "\n")
elif scenario.name == "earth-inner-core":
    if result.phase_crossing_years is not None:
        write_text("timehistogram5tw.txt", f"{result.phase_crossing_years}\n", "a")
    output_text = (
        " ".join(str(value) for value in result.sample_years)
        + "\n"
        + " ".join(str(value) for value in result.diagnostic_temperatures_k)
        + "\n"
    )
    write_text("temperaturearray.txt", output_text, "a")
else:
    write_text(f"{scenario.name}-time.json", json.dumps(record, indent=2) + "\n")
narrate(args.typeverb, "Time-dependent calculation complete.")
