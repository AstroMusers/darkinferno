#!/usr/bin/python

# Use ASTERIA capture to solve the spherical heat equation for phase-proxy exceedance.
# The solution assumes the system is in steady state
# Iterate because the dark-matter distribution depends on the body's temperature.

import argparse
import json
import math
from pathlib import Path
from tdpy import narrate, write_text

from scenarios import SCENARIOS, get_scenario


def finite_float(value):
    """Parse a finite floating-point argument."""

    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def positive_float(value):
    """Parse a positive floating-point argument."""

    number = finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def fraction(value):
    """Parse a fraction in the interval (0, 1]."""

    number = finite_float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be greater than zero and at most one")
    return number


def parse_args():
    """Parse steady-state command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Estimate phase-boundary heating from captured dark matter."
    )
    parser.add_argument(
        "log10_dark_matter_mass",
        type=finite_float,
        help="base-10 logarithm of the dark matter mass in GeV",
    )
    parser.add_argument(
        "log10_cross_section",
        type=finite_float,
        help="base-10 logarithm of the dark matter-nucleon cross section in cm^2",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="earth-inner-core",
        help="body and thermal scenario (default: %(default)s)",
    )
    parser.add_argument(
        "--halo-density",
        type=positive_float,
        default=0.4,
        help="dark-matter halo density in GeV cm^-3 (default: %(default)s)",
    )
    parser.add_argument(
        "--halo-speed",
        type=positive_float,
        default=270000.0,
        help="dark-matter halo speed in m s^-1 (default: %(default)s)",
    )
    parser.add_argument(
        "--deposition-efficiency",
        type=fraction,
        default=1.0,
        help="fraction of annihilation energy deposited as heat (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, help="append a structured result to this JSON Lines file"
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

import asteria
from physics import (
    GEV_IN_JOULE,
    GEV_IN_KG,
    gaussian_enclosed_fraction,
    solve_steady_state,
)

scenario = get_scenario(args.scenario)
mdm_gev = 10**args.log10_dark_matter_mass  # [GeV]
mdm = mdm_gev * GEV_IN_KG  # [kg]
sigma = 10**args.log10_cross_section  # [cm^2]

narrate(args.typeverb, f"Preparing scenario {scenario.name}...")
narrate(args.typeverb, f"  Dark-matter mass = {mdm_gev:g} GeV", level=2)
narrate(args.typeverb, f"  Scattering cross section = {sigma:g} cm^2", level=2)

narrate(args.typeverb, "Calculating the dark-matter capture rate with ASTERIA...")
mnucasteria = 0.94  # [GeV]
asteria.set_elements(scenario.asteria_elements)
capturerate = asteria.multicapture(
    scenario.radius_m,
    scenario.mass_kg,
    args.halo_density,
    args.halo_speed,
    mnucasteria,
    sigma,
    mdm_gev,
)  # [s^-1]

heatflow = (
    float(capturerate) * mdm_gev * GEV_IN_JOULE * args.deposition_efficiency
)  # [W]
narrate(args.typeverb, f"Capture calculation complete: {heatflow:g} W deposited.")

narrate(args.typeverb, "Solving the steady-state heat equation...")
result = solve_steady_state(scenario, mdm, heatflow)
enclosed_fraction = gaussian_enclosed_fraction(
    scenario.thermal_radius_m, result.scale_radius_m
)
print(f"Phase-proxy exceedance radius = {result.phase_radius_m:g} m")
print(f"Deposited-power fraction inside thermal domain = {enclosed_fraction:.6g}")
if enclosed_fraction < 0.9:
    print(
        "Warning: less than 90% of the Gaussian source lies inside the thermal domain."
    )
for warning in scenario.warnings:
    print(f"Warning: {warning}")

record = {
    "scenario": scenario.name,
    "body": scenario.body_name,
    "log10_dark_matter_mass_gev": args.log10_dark_matter_mass,
    "log10_cross_section_cm2": args.log10_cross_section,
    "halo_density_gev_cm3": args.halo_density,
    "halo_speed_m_s": args.halo_speed,
    "deposition_efficiency": args.deposition_efficiency,
    "capture_rate_s-1": float(capturerate),
    "heat_flow_w": heatflow,
    "total_deposited_power_w": heatflow,
    "thermal_domain_power_fraction": enclosed_fraction,
    "thermal_domain_deposited_power_w": enclosed_fraction * heatflow,
    "central_temperature_k": result.central_temperature_k,
    "diagnostic_temperature_k": result.diagnostic_temperature_k,
    "phase_temperature_k": scenario.phase_temperature_k,
    "phase_radius_m": result.phase_radius_m,
    "scale_radius_m": result.scale_radius_m,
    "assumes_capture_annihilation_equilibrium": True,
    "models_evaporation": False,
    "provenance": scenario.provenance,
    "warnings": scenario.warnings,
}

if args.output or scenario.name != "earth-inner-core":
    output_path = args.output or Path(f"{scenario.name}-steady.jsonl")
    output_text = json.dumps(record) + "\n"
else:
    output_path = Path("parameterscan.txt")
    output_text = (
        f"{args.log10_dark_matter_mass} {args.log10_cross_section} "
        f"{result.phase_radius_m}\n"
    )
write_text(output_path, output_text, "a")
narrate(args.typeverb, "Steady-state calculation complete.")
