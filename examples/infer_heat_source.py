#!/usr/bin/env python3
"""Fit a synthetic radial temperature profile.

The example assumes the Mars reference scenario, a 100 GeV dark-matter mass,
and a 1e10 W Gaussian heat source. Five radial temperatures with independent
3 K Gaussian errors are generated from DarkInferno's steady solver. This is a
sampler demonstration, not a Mars measurement or dark-matter constraint.
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from darkinferno.inference import infer_heat_source_power
from darkinferno.physics import GEV_IN_KG, steady_temperature_profile
from darkinferno.scenarios import get_scenario


SCENARIO_NAME = "mars-core-reference"
DARK_MATTER_MASS_GEV = 100.0  # [GeV]
TRUE_HEAT_SOURCE_POWER_W = 1.0e10  # [W]
TEMPERATURE_UNCERTAINTY_K = 3.0  # [K]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=800)
    parser.add_argument("--burn-count", type=int, default=400)
    parser.add_argument("--chain-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Run a short sampling check.")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    scenario = get_scenario(SCENARIO_NAME)
    radius_m = np.array([0.0, 250_000.0, 500_000.0, 900_000.0, 1_400_000.0])  # [m]
    truth = steady_temperature_profile(
        scenario,
        DARK_MATTER_MASS_GEV * GEV_IN_KG,
        TRUE_HEAT_SOURCE_POWER_W,
        radius_m,
    ).temperature_k
    random_generator = np.random.default_rng(arguments.seed)
    observed_temperature_k = random_generator.normal(
        truth, TEMPERATURE_UNCERTAINTY_K
    )  # [K]
    uncertainty_k = np.full(radius_m.size, TEMPERATURE_UNCERTAINTY_K)  # [K]

    posterior = infer_heat_source_power(
        scenario,
        DARK_MATTER_MASS_GEV,
        radius_m,
        observed_temperature_k,
        uncertainty_k,
        chain_count=1 if arguments.smoke else arguments.chain_count,
        sample_count=80 if arguments.smoke else arguments.sample_count,
        burn_count=40 if arguments.smoke else arguments.burn_count,
        seed=arguments.seed,
    )
    lower_power_w, median_power_w, upper_power_w = posterior.power_interval_w
    print(
        f"Synthetic heat power: {TRUE_HEAT_SOURCE_POWER_W:.3g} W; posterior median: "
        f"{median_power_w:.3g} (+{upper_power_w - median_power_w:.2g}/"
        f"-{median_power_w - lower_power_w:.2g}) W."
    )

    output_path = arguments.output or (
        Path(__file__).resolve().parents[1]
        / "visuals"
        / f"heat_source_posterior_{datetime.now():%Y%m%d_%H%M%S}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure, axis = plt.subplots(figsize=(6.0, 3.6))
    axis.hist(
        np.log10(posterior.power_samples_w.ravel()),
        bins=35,
        density=True,
        color="#007360",
        alpha=0.8,
    )
    axis.axvline(
        np.log10(TRUE_HEAT_SOURCE_POWER_W),
        color="#A51417",
        linestyle="--",
        label="Assumed synthetic truth",
    )
    axis.set_xlabel(r"$\log_{10}(P_{\rm source}/\mathrm{W})$")
    axis.set_ylabel("Posterior density")
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(False)
    figure.savefig(output_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()