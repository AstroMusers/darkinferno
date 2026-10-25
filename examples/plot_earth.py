#!/usr/bin/env python3

from _plotting import (
    EARTH_COLORS,
    foreground_color,
    configure_plotting,
    figure_already_exists,
    finish_plot,
    narrate,
    parse_plot_args,
    single_plot,
)
import numpy as np

from darkinferno.physics import (
    GEV_IN_KG,
    evolve_temperature,
    steady_temperature_profile,
)
from darkinferno.scenarios import get_scenario


def main():
    """Generate Earth steady-state and time-dependent figures."""

    args = parse_plot_args("Generate the numbered DarkInferno Earth plots.")
    configure_plotting(args.typeplotback)
    stems = ("01_earth_steady_temperature_profile", "02_earth_temperature_evolution")
    exists = [
        figure_already_exists(args.output_dir, stem, args.typefileplot)
        for stem in stems
    ]
    if all(exists):
        return
    narrate(args.typeverb, "Preparing the Earth heating plots...")
    # Earth thermal model: https://arxiv.org/abs/2505.24070
    # Pure-iron melting at the inner-core boundary: https://doi.org/10.1126/science.1233514
    scenario = get_scenario("earth-inner-core")
    dark_matter_mass = 1e3 * GEV_IN_KG  # [kg]

    if not exists[0]:
        figure, axis = single_plot(height_in=6.0)
        radius = np.linspace(0.0, scenario.thermal_radius_m, 500)  # [m]
        narrate(args.typeverb, "Calculating steady-state temperature profiles...")
        for power_tw, color in zip((0.0, 0.05, 0.1, 0.25), EARTH_COLORS):  # [TW]
            narrate(args.typeverb, f"  Power = {power_tw:g} TW", level=2)
            profile = steady_temperature_profile(
                scenario, dark_matter_mass, power_tw * 1e12, radius
            )
            axis.plot(
                radius / 1000.0,
                profile.temperature_k,
                color=color,
                label=f"Power = {power_tw:g} TW",
            )
            if power_tw == 0.25 and profile.result.phase_radius_m > 0:
                axis.axvspan(
                    0.0,
                    profile.result.phase_radius_m / 1000.0,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                    label="Above threshold at 0.25 TW",
                )
        axis.axhline(
            scenario.phase_temperature_k,
            color=foreground_color(),
            linestyle="--",
            linewidth=1.4,
            label="ICB melting estimate\n(Anzellini et al. 2013)",
        )
        finish_plot(
            figure,
            axis,
            args.output_dir,
            stems[0],
            args.typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.18),
            xlabel="Radius from center [km]",
            ylabel="Conductive reference temperature [K]",
            ylim=(5300.0, 7500.0),
        )
        narrate(args.typeverb, "Steady-state temperature profiles complete.")

    if not exists[1]:
        figure, axis = single_plot(height_in=6.4)
        crossing_labeled = False
        narrate(args.typeverb, "Evolving the time-dependent temperature profiles...")
        for power_tw, color in zip((0.5, 1.0, 3.0, 5.0), EARTH_COLORS[1:]):  # [TW]
            narrate(args.typeverb, f"  Power = {power_tw:g} TW", level=2)
            evolution = evolve_temperature(
                scenario,
                dark_matter_mass,
                power_tw * 1e12,
                scenario.initial_temperature_k,
                scenario.conductivity_w_m_k,
                end_time_years=1e9,
                time_step_years=2e6,
                radial_cells=180,
                sample_every=5,
                typeverb=args.typeverb,
                latent_width_k=50.0,
            )
            axis.plot(
                np.asarray(evolution.sample_years) / 1e6,
                evolution.diagnostic_temperatures_k,
                color=color,
                label=f"Power = {power_tw:g} TW",
            )
            if evolution.phase_crossing_years is not None:
                axis.scatter(
                    evolution.phase_crossing_years / 1e6,
                    scenario.phase_temperature_k,
                    color=color,
                    edgecolor=foreground_color(),
                    linewidth=0.6,
                    s=48,
                    zorder=4,
                    label="First criterion crossing" if not crossing_labeled else None,
                )
                crossing_labeled = True
        axis.axhline(
            scenario.phase_temperature_k,
            color=foreground_color(),
            linestyle="--",
            linewidth=1.4,
            label="ICB melting estimate\n(Anzellini et al. 2013)",
        )
        finish_plot(
            figure,
            axis,
            args.output_dir,
            stems[1],
            args.typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.18),
            xlabel="Elapsed time [Myr]",
            ylabel="Temperature at $r=400$ km [K]",
            xlim=(0.0, 1000.0),
        )
        narrate(args.typeverb, "Time-dependent temperature profiles complete.")


if __name__ == "__main__":
    main()
