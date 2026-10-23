#!/usr/bin/env python3

import csv
from dataclasses import replace
from pathlib import Path

from _plotting import (
    EARTH_COLORS,
    PAPER_COLUMN_WIDTH_IN,
    configure_plotting,
    finish_plot,
    narrate,
    parse_plot_args,
    save_figure,
    single_plot,
)
import matplotlib.pyplot as plt
import numpy as np

from darkinferno.constraints import (
    ParticleHeatingModel,
    annihilation_cross_section_for_equilibrium_cm3_s,
    cross_section_boundary_cm2,
    earth_core_drift_ceiling_cm2,
    gordion_capture_rate_s,
    logarithmic_capture_slope,
    minimum_deposition_efficiency,
)
from darkinferno.observations import (
    get_heat_flow_constraint,
    get_radial_heat_flow_constraint,
)
from darkinferno.physics import (
    GEV_IN_KG,
    dark_matter_scale_radius,
    diagnostic_temperature_power,
    gaussian_enclosed_fraction,
)
from darkinferno.scenarios import earth_elements_with_iron_fraction, get_scenario


def main():
    """Generate the Earth radial-calorimetry particle exclusion."""

    args = parse_plot_args("Calculate Earth's radial dark-matter constraint.")
    configure_plotting(args.typeplotback)
    scenario = get_scenario("earth-inner-core")
    cmb = get_radial_heat_flow_constraint("Earth core-mantle boundary")
    surface = get_heat_flow_constraint("Earth")
    masses_gev = np.logspace(np.log10(5.0), 6.0, 31)
    boundaries = {
        "cmb_13_tw": [],
        "cmb_15_tw": [],
        "surface_49_tw": [],
        "inner_core_10000_k": [],
    }
    halo_models = {
        "rho0p3_v230": ParticleHeatingModel(
            halo_density_gev_cm3=0.3, halo_speed_m_s=230_000.0
        ),
        "rho0p3_v310": ParticleHeatingModel(
            halo_density_gev_cm3=0.3, halo_speed_m_s=310_000.0
        ),
        "rho0p6_v230": ParticleHeatingModel(
            halo_density_gev_cm3=0.6, halo_speed_m_s=230_000.0
        ),
        "rho0p6_v310": ParticleHeatingModel(
            halo_density_gev_cm3=0.6, halo_speed_m_s=310_000.0
        ),
    }
    halo_cmb_boundaries = {name: [] for name in halo_models}
    halo_boundary_ratios = {name: [] for name in halo_models}
    composition_scenarios = {
        "iron_0p27": replace(
            scenario, elements=earth_elements_with_iron_fraction(0.27)
        ),
        "iron_0p37": replace(
            scenario, elements=earth_elements_with_iron_fraction(0.37)
        ),
    }
    composition_cmb_boundaries = {name: [] for name in composition_scenarios}
    composition_boundary_ratios = {name: [] for name in composition_scenarios}
    capture_slopes = []
    minimum_efficiencies = []
    equilibrium_cross_sections = []
    narrate(args.typeverb, "Solving the Earth particle-constraint boundaries...")

    for mass_gev in masses_gev:
        for name, radius_m, ceiling_w in (
            ("cmb_13_tw", cmb.boundary_radius_m, cmb.outward_power_min_w),
            ("cmb_15_tw", cmb.boundary_radius_m, cmb.outward_power_max_w),
            ("surface_49_tw", scenario.radius_m, surface.reported_total_max_w),
        ):
            boundaries[name].append(
                cross_section_boundary_cm2(
                    scenario,
                    mass_gev,
                    radius_m,
                    ceiling_w,
                    log10_cross_section_bounds=(-45.0, -16.0),
                )
            )
        inner_core_power_w = diagnostic_temperature_power(
            scenario,
            mass_gev * GEV_IN_KG,
            10_000.0,
        )
        boundaries["inner_core_10000_k"].append(
            cross_section_boundary_cm2(
                scenario,
                mass_gev,
                scenario.radius_m,
                inner_core_power_w,
                log10_cross_section_bounds=(-45.0, -16.0),
            )
        )
        for name, model in halo_models.items():
            cmb_boundary_cm2 = cross_section_boundary_cm2(
                scenario,
                mass_gev,
                cmb.boundary_radius_m,
                cmb.outward_power_max_w,
                model=model,
                log10_cross_section_bounds=(-45.0, -16.0),
            )
            surface_boundary_cm2 = cross_section_boundary_cm2(
                scenario,
                mass_gev,
                scenario.radius_m,
                surface.reported_total_max_w,
                model=model,
                log10_cross_section_bounds=(-45.0, -16.0),
            )
            halo_cmb_boundaries[name].append(cmb_boundary_cm2)
            halo_boundary_ratios[name].append(
                cmb_boundary_cm2 / surface_boundary_cm2
            )
        for name, composition_scenario in composition_scenarios.items():
            cmb_boundary_cm2 = cross_section_boundary_cm2(
                composition_scenario,
                mass_gev,
                cmb.boundary_radius_m,
                cmb.outward_power_max_w,
                log10_cross_section_bounds=(-45.0, -16.0),
            )
            surface_boundary_cm2 = cross_section_boundary_cm2(
                composition_scenario,
                mass_gev,
                composition_scenario.radius_m,
                surface.reported_total_max_w,
                log10_cross_section_bounds=(-45.0, -16.0),
            )
            composition_cmb_boundaries[name].append(cmb_boundary_cm2)
            composition_boundary_ratios[name].append(
                cmb_boundary_cm2 / surface_boundary_cm2
            )
        capture_slopes.append(
            logarithmic_capture_slope(
                scenario,
                mass_gev,
                boundaries["cmb_15_tw"][-1],
            )
        )
        drift_ceiling_cm2 = earth_core_drift_ceiling_cm2(mass_gev)
        capture_rate_s = gordion_capture_rate_s(
            scenario,
            mass_gev,
            drift_ceiling_cm2,
        )
        scale_radius_m = dark_matter_scale_radius(
            scenario.initial_temperature_k,
            scenario.density_kg_m3,
            mass_gev * GEV_IN_KG,
        )
        enclosed_fraction = gaussian_enclosed_fraction(
            cmb.boundary_radius_m,
            scale_radius_m,
        )
        minimum_efficiencies.append(
            minimum_deposition_efficiency(
                capture_rate_s,
                mass_gev,
                enclosed_fraction,
                cmb.outward_power_max_w,
            )
        )
        equilibrium_cross_sections.append(
            annihilation_cross_section_for_equilibrium_cm3_s(
                capture_rate_s,
                scale_radius_m,
                equilibrium_fraction=0.9,
            )
        )

    output_data = Path(args.output_dir) / "08_earth_particle_constraint.csv"
    output_data.parent.mkdir(parents=True, exist_ok=True)
    with output_data.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "dark_matter_mass_gev",
                "cmb_13_tw_cm2",
                "cmb_15_tw_cm2",
                "surface_49_tw_cm2",
                "inner_core_10000_k_cm2",
                "halo_cmb_min_cm2",
                "halo_cmb_max_cm2",
                "halo_boundary_ratio_min",
                "halo_boundary_ratio_max",
                "composition_cmb_min_cm2",
                "composition_cmb_max_cm2",
                "composition_boundary_ratio_min",
                "composition_boundary_ratio_max",
                "capture_logarithmic_slope",
                "drift_ceiling_cm2",
                "minimum_deposition_efficiency",
                "annihilation_cross_section_90_equilibrium_cm3_s",
            )
        )
        for index, mass_gev in enumerate(masses_gev):
            writer.writerow(
                (
                    mass_gev,
                    boundaries["cmb_13_tw"][index],
                    boundaries["cmb_15_tw"][index],
                    boundaries["surface_49_tw"][index],
                    boundaries["inner_core_10000_k"][index],
                    min(values[index] for values in halo_cmb_boundaries.values()),
                    max(values[index] for values in halo_cmb_boundaries.values()),
                    min(values[index] for values in halo_boundary_ratios.values()),
                    max(values[index] for values in halo_boundary_ratios.values()),
                    min(
                        values[index]
                        for values in composition_cmb_boundaries.values()
                    ),
                    max(
                        values[index]
                        for values in composition_cmb_boundaries.values()
                    ),
                    min(
                        values[index]
                        for values in composition_boundary_ratios.values()
                    ),
                    max(
                        values[index]
                        for values in composition_boundary_ratios.values()
                    ),
                    capture_slopes[index],
                    earth_core_drift_ceiling_cm2(mass_gev),
                    minimum_efficiencies[index],
                    equilibrium_cross_sections[index],
                )
            )

    figure, axis = single_plot(height_in=5.1)
    drift_ceiling = np.array(
        [earth_core_drift_ceiling_cm2(mass) for mass in masses_gev]
    )
    cmb_lower = np.array(boundaries["cmb_13_tw"])
    cmb_upper = np.array(boundaries["cmb_15_tw"])
    surface_boundary = np.array(boundaries["surface_49_tw"])
    inner_core_boundary = np.array(boundaries["inner_core_10000_k"])
    axis.fill_between(
        masses_gev,
        cmb_upper,
        drift_ceiling,
        color=EARTH_COLORS[0],
        alpha=0.16,
        linewidth=0,
        label="Above 15 TW CMB benchmark",
    )
    axis.fill_between(
        masses_gev,
        cmb_lower,
        cmb_upper,
        color=EARTH_COLORS[1],
        alpha=0.22,
        linewidth=0,
        label="CMB: 13--15 TW preferred",
    )
    axis.loglog(
        masses_gev,
        cmb_upper,
        color=EARTH_COLORS[0],
        linewidth=2.4,
        label="CMB: 15 TW",
    )
    axis.loglog(
        masses_gev,
        surface_boundary,
        color="#505b62",
        linestyle="--",
        label="Surface: 49 TW",
    )
    axis.loglog(
        masses_gev,
        inner_core_boundary,
        color=EARTH_COLORS[2],
        linestyle="-.",
        label="Inner-core 10,000 K reproduction",
    )
    axis.loglog(
        masses_gev,
        drift_ceiling,
        color="#172126",
        linestyle=":",
        label="1 Gyr drift ceiling",
    )
    finish_plot(
        figure,
        axis,
        args.output_dir,
        "08_earth_particle_constraint",
        args.typefileplot,
        "upper center",
        legend_anchor=(0.5, -0.22),
        xlabel="DM mass $m_\\chi$\n[GeV]",
        ylabel="DM--nucleon cross section\n[cm$^2$]",
        xlim=(5.0, 1e6),
        ylim=(1e-39, 1e-16),
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(PAPER_COLUMN_WIDTH_IN, 6.0),
        layout="constrained",
        sharex=True,
    )
    axes[0].loglog(
        masses_gev,
        np.array(minimum_efficiencies) * 100.0,
        color=EARTH_COLORS[2],
    )
    axes[0].axhline(1.0, color="#737b82", linestyle=":", linewidth=1.4)
    axes[0].set(
        ylabel="Minimum deposited\nenergy fraction [%]",
        ylim=(0.5, 3.0),
    )
    axes[1].loglog(
        masses_gev,
        equilibrium_cross_sections,
        color=EARTH_COLORS[0],
    )
    axes[1].axhline(
        3e-26,
        color="#505b62",
        linestyle="--",
        linewidth=1.6,
    )
    axes[1].text(7.0, 6e-27, "Thermal relic", color="#505b62")
    axes[1].text(9.0, 1.2e-31, "90% equilibrium", color=EARTH_COLORS[0])
    axes[1].set(
        xlabel="DM mass $m_\\chi$ [GeV]",
        ylabel="Annihilation coefficient\n[cm$^3$ s$^{-1}$]",
        xlim=(5.0, 1e6),
        ylim=(1e-36, 1e-24),
    )
    save_figure(
        figure,
        args.output_dir,
        "10_earth_particle_robustness",
        typefileplot=args.typefileplot,
    )

    halo_cmb_values = np.array(list(halo_cmb_boundaries.values()))
    halo_ratio_values = np.array(list(halo_boundary_ratios.values()))
    composition_cmb_values = np.array(list(composition_cmb_boundaries.values()))
    composition_ratio_values = np.array(
        list(composition_boundary_ratios.values())
    )
    reference_ratio = cmb_upper / surface_boundary
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(PAPER_COLUMN_WIDTH_IN, 7.2),
        layout="constrained",
        sharex=True,
    )
    axes[0].fill_between(
        masses_gev,
        np.min(halo_cmb_values, axis=0),
        np.max(halo_cmb_values, axis=0),
        color=EARTH_COLORS[1],
        alpha=0.35,
        linewidth=0,
        label="Halo envelope",
    )
    axes[0].loglog(masses_gev, cmb_upper, color=EARTH_COLORS[0], label="Reference")
    axes[0].fill_between(
        masses_gev,
        np.min(composition_cmb_values, axis=0),
        np.max(composition_cmb_values, axis=0),
        facecolor="none",
        edgecolor=EARTH_COLORS[2],
        hatch="////",
        linewidth=0.8,
        label="Bulk Fe proxy",
    )
    axes[0].set(ylabel="CMB boundary\n[cm$^2$]")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].fill_between(
        masses_gev,
        np.min(halo_ratio_values, axis=0),
        np.max(halo_ratio_values, axis=0),
        color=EARTH_COLORS[1],
        alpha=0.35,
        linewidth=0,
    )
    axes[1].semilogx(masses_gev, reference_ratio, color=EARTH_COLORS[0])
    axes[1].fill_between(
        masses_gev,
        np.min(composition_ratio_values, axis=0),
        np.max(composition_ratio_values, axis=0),
        facecolor="none",
        edgecolor=EARTH_COLORS[2],
        hatch="////",
        linewidth=0.8,
    )
    axes[1].axhline(1.0, color="#737b82", linestyle=":", linewidth=1.4)
    axes[1].set(ylabel=r"$\sigma_{\rm CMB}/\sigma_{\rm surface}$")
    axes[2].semilogx(masses_gev, capture_slopes, color=EARTH_COLORS[2])
    axes[2].axhline(1.0, color="#737b82", linestyle=":", linewidth=1.4)
    axes[2].set(
        xlabel="DM mass $m_\\chi$ [GeV]",
        ylabel=r"$d\ln C/d\ln\sigma_{\chi N}$",
        xlim=(5.0, 1e6),
    )
    save_figure(
        figure,
        args.output_dir,
        "11_earth_particle_systematics",
        typefileplot=args.typefileplot,
    )


if __name__ == "__main__":
    main()