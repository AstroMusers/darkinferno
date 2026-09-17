#!/usr/bin/env python3

from _plotting import (
    INK,
    MOON_COLORS,
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
    """Generate lunar steady-state and time-dependent figures."""

    args = parse_plot_args("Generate the numbered DarkInferno Moon plots.")
    stems = ("01_moon_steady_temperature_profile", "02_moon_temperature_evolution")
    exists = [
        figure_already_exists(args.output_dir, stem, args.typefileplot)
        for stem in stems
    ]
    if all(exists):
        return
    narrate(args.typeverb, "Preparing the Moon heating plots...")
    configure_plotting()
    # Lunar core structure: https://doi.org/10.1126/science.1199375
    # Elemental-iron melting point: https://doi.org/10.1201/9781315380476
    central = get_scenario("moon-inner-core")
    dark_matter_mass = 1e3 * GEV_IN_KG  # [kg]

    if not exists[0]:
        figure, axis = single_plot(height_in=7.0)
        radius = np.linspace(0.0, central.thermal_radius_m, 500)  # [m]
        narrate(args.typeverb, "Calculating steady-state temperature profiles...")
        for power_gw, color in zip((0.0, 0.3, 1.0, 3.0, 10.0), MOON_COLORS):  # [GW]
            narrate(args.typeverb, f"  Power = {power_gw:g} GW", level=2)
            profile = steady_temperature_profile(
                central, dark_matter_mass, power_gw * 1e9, radius
            )
            axis.plot(
                radius / 1000.0,
                profile.temperature_k,
                color=color,
                label=f"Power = {power_gw:g} GW",
            )
        axis.axhline(
            central.phase_temperature_k,
            color=INK,
            linestyle="--",
            linewidth=1.4,
            label="Iron melting proxy\n(Haynes et al. 2016)",
        )
        axis.axvline(
            central.diagnostic_radius_m / 1000.0,
            ymax=0.46,
            color=MOON_COLORS[1],
            linestyle=":",
            linewidth=1.3,
            label="Diagnostic radius [km] = 120",
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
            ylim=(1744.0, 2200.0),
        )
        narrate(args.typeverb, "Steady-state temperature profiles complete.")

    if not exists[1]:
        figure, axis = single_plot(height_in=6.4)
        crossing_labeled = False
        narrate(args.typeverb, "Evolving the time-dependent temperature profiles...")
        for power_gw, color in zip((0.3, 1.0, 3.0, 10.0), MOON_COLORS[1:]):  # [GW]
            narrate(args.typeverb, f"  Power = {power_gw:g} GW", level=2)
            evolution = evolve_temperature(
                central,
                dark_matter_mass,
                power_gw * 1e9,
                central.initial_temperature_k,
                central.conductivity_w_m_k,
                end_time_years=5e8,
                time_step_years=1e6,
                radial_cells=180,
                sample_every=2,
                typeverb=args.typeverb,
                latent_width_k=50.0,
            )
            axis.plot(
                np.asarray(evolution.sample_years) / 1e6,
                evolution.diagnostic_temperatures_k,
                color=color,
                label=f"Power = {power_gw:g} GW",
            )
            if evolution.phase_crossing_years is not None:
                axis.scatter(
                    evolution.phase_crossing_years / 1e6,
                    central.phase_temperature_k,
                    color=color,
                    edgecolor=INK,
                    linewidth=0.6,
                    s=48,
                    zorder=4,
                    label="First criterion crossing" if not crossing_labeled else None,
                )
                crossing_labeled = True
        axis.axhline(
            central.phase_temperature_k,
            color=INK,
            linestyle="--",
            linewidth=1.4,
            label="Iron melting proxy\n(Haynes et al. 2016)",
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
            ylabel="Temperature at $r=120$ km [K]",
            xlim=(0.0, 500.0),
            ylim=(1725.0, 1820.0),
        )
        narrate(args.typeverb, "Time-dependent temperature profiles complete.")


if __name__ == "__main__":
    main()
