#!/usr/bin/env python3

from _plotting import (
    EARTH_COLORS,
    MOON_COLORS,
    configure_plotting,
    figure_already_exists,
    finish_plot,
    narrate,
    parse_plot_args,
    single_plot,
)
import numpy as np

from darkinferno.physics import GEV_IN_KG, evolve_temperature
from darkinferno.scenarios import get_scenario


def crossing_times(
    scenario, power_w, end_time_years, widths_k, radial_cells, time_step_years
):
    """Calculate diagnostic-radius crossing times across latent widths."""

    times_years = []
    for width_k in widths_k:
        evolution = evolve_temperature(
            scenario,
            1e3 * GEV_IN_KG,
            power_w,
            scenario.initial_temperature_k,
            scenario.conductivity_w_m_k,
            end_time_years=end_time_years,
            time_step_years=time_step_years,
            radial_cells=radial_cells,
            sample_every=10,
            latent_width_k=width_k,
        )
        times_years.append(evolution.phase_crossing_years or np.nan)

    return np.asarray(times_years)


def plot_body_convergence(
    scenario,
    power_w,
    power_label,
    end_time_years,
    base_time_step_years,
    colors,
    output_dir,
    typefileplot,
    typeverb,
):
    """Plot latent-width, radial-grid, and timestep sensitivity for one body."""

    body = scenario.body_name.lower()
    stem = f"09_{body}_transient_convergence"
    if figure_already_exists(output_dir, stem, typefileplot):
        return

    narrate(typeverb, f"Calculating {scenario.body_name} transient convergence...")
    widths_k = np.array((50.0, 100.0, 200.0, 500.0))  # [K]
    configurations = (
        (180, base_time_step_years, "180 cells, $\\Delta t$"),
        (360, base_time_step_years, "360 cells, $\\Delta t$"),
        (180, base_time_step_years / 2.0, "180 cells, $\\Delta t/2$"),
        (360, base_time_step_years / 2.0, "360 cells, $\\Delta t/2$"),
    )

    figure, axis = single_plot(height_in=5.5)
    for (radial_cells, time_step_years, label), color, marker in zip(
        configurations, colors[:4], ("o", "s", "^", "D")
    ):
        times_years = crossing_times(
            scenario,
            power_w,
            end_time_years,
            widths_k,
            radial_cells,
            time_step_years,
        )
        axis.semilogx(
            widths_k,
            times_years / 1e6,
            color=color,
            marker=marker,
            label=label,
        )
    axis.tick_params(axis="x", which="minor", labelbottom=False)

    finish_plot(
        figure,
        axis,
        output_dir,
        stem,
        typefileplot,
        "upper center",
        legend_anchor=(0.5, -0.2),
        xlabel="Latent-heat width $\\Delta T$ [K]",
        ylabel="First crossing time [Myr]",
        xlim=(40.0, 620.0),
        xticks=widths_k,
        xticklabels=("50", "100", "200", "500"),
    )
    narrate(
        typeverb,
        f"{scenario.body_name} transient convergence complete ({power_label}).",
    )


def main():
    """Generate Earth and lunar transient-convergence figures."""

    args = parse_plot_args("Plot DarkInferno transient numerical convergence.")
    configure_plotting(args.typeplotback)
    plot_body_convergence(
        get_scenario("earth-inner-core"),
        5e12,
        "5 TW",
        1e9,
        2e6,
        EARTH_COLORS,
        args.output_dir,
        args.typefileplot,
        args.typeverb,
    )
    plot_body_convergence(
        get_scenario("moon-inner-core"),
        1e10,
        "10 GW",
        5e8,
        1e6,
        MOON_COLORS,
        args.output_dir,
        args.typefileplot,
        args.typeverb,
    )


if __name__ == "__main__":
    main()
