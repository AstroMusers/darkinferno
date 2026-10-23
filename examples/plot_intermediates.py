#!/usr/bin/env python3

from _plotting import (
    EARTH_COLORS,
    foreground_color,
    MOON_COLORS,
    configure_plotting,
    figure_already_exists,
    finish_plot,
    narrate,
    parse_plot_args,
    single_plot,
)
import numpy as np
from scipy import optimize, special

from darkinferno.physics import (
    GEV_IN_KG,
    SECONDS_PER_YEAR,
    dark_matter_scale_radius,
    gaussian_enclosed_fraction,
    solve_steady_state,
    volumetric_heating,
)
from darkinferno.scenarios import get_scenario


def plot_intermediates(
    scenario, power_w, power_label, colors, output_dir, typefileplot, typeverb
):
    """Plot the major intermediate variables for one body."""

    body = scenario.body_name.lower()
    stems = tuple(
        f"{number:02d}_{body}_{name}"
        for number, name in enumerate(
            (
                "thermal_scale_radius",
                "volumetric_heating",
                "steady_convergence",
                "specific_heat_capacity",
                "volumetric_heat_capacity",
                "thermal_diffusivity",
                "diffusion_number",
                "source_temperature_increment",
            ),
            1,
        )
    )
    exists = [figure_already_exists(output_dir, stem, typefileplot) for stem in stems]
    if all(exists):
        return
    narrate(typeverb, f"Calculating {scenario.body_name} intermediate variables...")
    dark_matter_mass = 1e3 * GEV_IN_KG  # [kg]
    radius = np.linspace(0.0, scenario.thermal_radius_m, 500)  # [m]
    latent_width = 50.0  # [K]
    time_step_years = 2e6 if scenario.body_name == "Earth" else 1e6  # [yr]
    radial_step = scenario.thermal_radius_m / 180  # [m]
    melting_label = (
        "ICB melting estimate\n(Anzellini et al. 2013)"
        if scenario.body_name == "Earth"
        else "Iron melting proxy\n(Haynes et al. 2016)"
    )

    if not exists[0]:
        masses_gev = np.logspace(1.0, 6.0, 200)  # [GeV]
        scale_radii_km = np.array(
            [
                dark_matter_scale_radius(
                    scenario.initial_temperature_k,
                    scenario.density_kg_m3,
                    mass * GEV_IN_KG,
                )
                / 1000.0
                for mass in masses_gev
            ]
        )  # [km]
        enclosed_fractions = np.array(
            [
                gaussian_enclosed_fraction(
                    scenario.thermal_radius_m, radius_km * 1000.0
                )
                for radius_km in scale_radii_km
            ]
        )
        valid_mass_gev = np.interp(0.9, enclosed_fractions, masses_gev)  # [GeV]
        radius_90_factor = optimize.brentq(
            lambda ratio: special.erf(np.sqrt(2.0) * ratio)
            - 2.0 * np.sqrt(2.0 / np.pi) * ratio * np.exp(-2.0 * ratio**2)
            - 0.9,
            0.0,
            5.0,
        )
        figure, axis = single_plot(height_in=7.0)
        axis.loglog(
            masses_gev, scale_radii_km, color=colors[0], label=r"Scale radius $r_\chi$"
        )
        axis.loglog(
            masses_gev,
            radius_90_factor * scale_radii_km,
            color=colors[2],
            linestyle="--",
            label=r"90% radius $r_{90}$",
        )
        axis.axhline(
            scenario.thermal_radius_m / 1000.0,
            color=colors[3],
            linestyle="-.",
            linewidth=1.6,
            label=r"Thermal-domain $R$",
        )
        axis.axhline(
            scenario.diagnostic_radius_m / 1000.0,
            color=foreground_color(),
            linestyle=":",
            linewidth=1.6,
            label="Diagnostic radius",
        )
        axis.axvline(1e3, color=foreground_color(), linewidth=1.0, alpha=0.45, label=r"$m_\chi=1$ TeV")
        axis.axvspan(
            masses_gev[0],
            valid_mass_gev,
            color=foreground_color(),
            alpha=0.08,
            linewidth=0,
            label=r"$f_R<0.9$",
        )
        finish_plot(
            figure,
            axis,
            output_dir,
            stems[0],
            typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.12),
            xlabel=r"Dark-matter mass $m_\chi$ [GeV]",
            ylabel="Radius [km]",
        )

    scale_radius = dark_matter_scale_radius(
        scenario.initial_temperature_k, scenario.density_kg_m3, dark_matter_mass
    )  # [m]
    if not exists[1]:
        figure, axis = single_plot(height_in=4.8)
        axis.semilogy(
            radius / 1000.0,
            volumetric_heating(
                radius,
                scenario.initial_temperature_k,
                scenario,
                dark_matter_mass,
                power_w,
            ),
            color=colors[1],
            label=f"Heating; {power_label}",
        )
        axis.axvline(
            scale_radius / 1000.0,
            ymax=0.72,
            color=foreground_color(),
            linestyle=":",
            linewidth=1.3,
            label=r"$r_\chi$; 1 TeV",
        )
        finish_plot(
            figure,
            axis,
            output_dir,
            stems[1],
            typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.2),
            xlabel="Radius from center [km]",
            ylabel=r"Heating [W m$^{-3}$]",
        )

    if not exists[2]:
        iterations = np.arange(1, 11)
        steady_results = [
            solve_steady_state(
                scenario, dark_matter_mass, power_w, iterations=int(iteration)
            )
            for iteration in iterations
        ]
        figure, axis = single_plot(height_in=4.8)
        axis.plot(
            iterations,
            [result.characteristic_temperature_k for result in steady_results],
            marker="o",
            color=colors[0],
            label="Characteristic; circles",
        )
        axis.plot(
            iterations,
            [result.diagnostic_temperature_k for result in steady_results],
            marker="s",
            color=colors[2],
            label="Diagnostic; squares",
        )
        finish_plot(
            figure,
            axis,
            output_dir,
            stems[2],
            typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.2),
            xlabel="Steady-state iteration",
            ylabel="Temperature [K]",
            xticks=iterations,
        )

    if not all(exists[3:7]):
        temperature = np.linspace(
            max(1.0, scenario.initial_temperature_k - 1000.0),
            scenario.phase_temperature_k + 1000.0,
            500,
        )  # [K]
        latent_capacity = (
            scenario.latent_heat_j_kg
            / (np.sqrt(2.0 * np.pi) * latent_width)
            * np.exp(
                -((temperature - scenario.phase_temperature_k) ** 2)
                / (2.0 * latent_width**2)
            )
        )  # [J kg^-1 K^-1]
        total_capacity = (
            scenario.heat_capacity_j_kg_k + latent_capacity
        )  # [J kg^-1 K^-1]
        volumetric_capacity = scenario.density_kg_m3 * total_capacity  # [J m^-3 K^-1]
        diffusivity = scenario.conductivity_w_m_k / volumetric_capacity  # [m^2 s^-1]
        diffusion_number = (
            diffusivity * time_step_years * SECONDS_PER_YEAR / radial_step**2
        )

    if not exists[3]:
        figure, axis = single_plot(height_in=6.2)
        axis.plot(
            temperature,
            np.full_like(temperature, scenario.heat_capacity_j_kg_k),
            color=colors[0],
            linestyle="--",
            label="Sensible heat capacity",
        )
        axis.plot(
            temperature,
            latent_capacity,
            color=colors[2],
            linestyle=":",
            label="Latent contribution",
        )
        axis.plot(
            temperature,
            total_capacity,
            color=colors[3],
            label="Effective heat capacity",
        )
        axis.axvline(
            scenario.phase_temperature_k,
            ymax=0.72,
            color=foreground_color(),
            linestyle="-.",
            linewidth=1.3,
            label=melting_label,
        )
        finish_plot(
            figure,
            axis,
            output_dir,
            stems[3],
            typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.18),
            xlabel="Temperature [K]",
            ylabel=r"Specific heat capacity [J kg$^{-1}$ K$^{-1}$]",
            ylim=(0.0, 700.0 if scenario.body_name == "Earth" else 1350.0),
        )

    if not exists[4]:
        figure, axis = single_plot(height_in=5.0)
        axis.plot(
            temperature,
            volumetric_capacity,
            color=colors[3],
            label="Volumetric heat capacity",
        )
        axis.axvline(
            scenario.phase_temperature_k,
            ymax=0.72,
            color=foreground_color(),
            linestyle="-.",
            linewidth=1.3,
            label=melting_label,
        )
        finish_plot(
            figure,
            axis,
            output_dir,
            stems[4],
            typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.2),
            xlabel="Temperature [K]",
            ylabel=r"Volumetric heat capacity [J m$^{-3}$ K$^{-1}$]",
        )

    if not exists[5]:
        figure, axis = single_plot(height_in=5.0)
        axis.semilogy(
            temperature, diffusivity, color=colors[1], label="Thermal diffusivity"
        )
        axis.axvline(
            scenario.phase_temperature_k,
            ymax=0.72,
            color=foreground_color(),
            linestyle="-.",
            linewidth=1.3,
            label=melting_label,
        )
        finish_plot(
            figure,
            axis,
            output_dir,
            stems[5],
            typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.2),
            xlabel="Temperature [K]",
            ylabel=r"Thermal diffusivity [m$^2$ s$^{-1}$]",
        )

    if not exists[6]:
        figure, axis = single_plot(height_in=4.8)
        axis.semilogy(
            temperature,
            diffusion_number,
            color=colors[2],
            label=rf"Diffusion number; $\Delta t$ [yr] = {time_step_years:.0e}",
        )
        axis.axhline(
            1.0, color=foreground_color(), linestyle="--", linewidth=1.3, label="Diffusion number = 1"
        )
        axis.axvline(
            scenario.phase_temperature_k,
            ymax=0.72,
            color=foreground_color(),
            linestyle="-.",
            linewidth=1.3,
            label=melting_label,
        )
        finish_plot(
            figure,
            axis,
            output_dir,
            stems[6],
            typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.12),
            xlabel="Temperature [K]",
            ylabel="Diffusion number",
        )

    if not exists[7]:
        initial_capacity = scenario.density_kg_m3 * (
            scenario.heat_capacity_j_kg_k
            + scenario.latent_heat_j_kg
            / (np.sqrt(2.0 * np.pi) * latent_width)
            * np.exp(
                -((scenario.initial_temperature_k - scenario.phase_temperature_k) ** 2)
                / (2.0 * latent_width**2)
            )
        )  # [J m^-3 K^-1]
        figure, axis = single_plot()
        axis.semilogy(
            radius / 1000.0,
            volumetric_heating(
                radius,
                scenario.initial_temperature_k,
                scenario,
                dark_matter_mass,
                power_w,
            )
            * time_step_years
            * SECONDS_PER_YEAR
            / initial_capacity,
            color=colors[3],
            label=rf"Increment; $\Delta t={time_step_years:.0e}$ yr",
        )
        axis.axvline(
            scale_radius / 1000.0,
            ymax=0.72,
            color=foreground_color(),
            linestyle=":",
            linewidth=1.3,
            label=r"$r_\chi$; 1 TeV",
        )
        finish_plot(
            figure,
            axis,
            output_dir,
            stems[7],
            typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.2),
            xlabel="Radius from center [km]",
            ylabel="Source increment [K per step]",
        )
    narrate(typeverb, f"{scenario.body_name} intermediate-variable plots complete.")


def main():
    """Generate intermediate-variable figures for Earth and the Moon."""

    args = parse_plot_args(
        "Plot the major intermediate variables in the DarkInferno calculations."
    )
    configure_plotting(args.typeplotback)
    plot_intermediates(
        get_scenario("earth-inner-core"),
        1e12,
        "power [TW] = 1",
        EARTH_COLORS,
        args.output_dir,
        args.typefileplot,
        args.typeverb,
    )  # [W]
    plot_intermediates(
        get_scenario("moon-inner-core"),
        1e9,
        "power [GW] = 1",
        MOON_COLORS,
        args.output_dir,
        args.typefileplot,
        args.typeverb,
    )  # [W]


if __name__ == "__main__":
    main()
