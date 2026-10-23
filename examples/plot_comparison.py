#!/usr/bin/env python3

from dataclasses import replace

from _plotting import (
    EARTH_COLORS,
    MARS_COLORS,
    MOON_COLORS,
    PAPER_COLUMN_WIDTH_IN,
    configure_plotting,
    figure_already_exists,
    finish_plot,
    narrate,
    parse_plot_args,
    save_figure,
    single_plot,
)
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from darkinferno.observations import (
    get_heat_flow_constraint,
    get_radial_heat_flow_constraint,
)
from darkinferno.physics import (
    GEV_IN_KG,
    dark_matter_scale_radius,
    gaussian_enclosed_fraction,
    radial_temperature_shape_ratio,
    solve_steady_state,
    thermal_response_completeness,
)
from darkinferno.scenarios import get_scenario


def onset_power(scenario, dark_matter_mass_kg):
    """Calculate the deposited power that erases the imposed central margin."""

    power_w = 10.0 ** optimize.brentq(
        lambda log10_power: solve_steady_state(
            scenario, dark_matter_mass_kg, 10.0**log10_power
        ).central_temperature_k
        - scenario.phase_temperature_k,
        3.0,
        16.0,
    )  # [W]

    return power_w


def main():
    """Generate nonredundant rocky-body comparison figures."""

    args = parse_plot_args("Compare the DarkInferno rocky-body calculations.")
    configure_plotting(args.typeplotback)
    stems = (
        "01_comparison_phase_radius",
        "02_comparison_spatial_heating",
        "03_comparison_phase_onset_power",
        "04_comparison_dimensionless_validity",
        "05_comparison_radial_fingerprint",
        "06_comparison_observed_heat_flow",
        "07_earth_radial_calorimetry",
    )
    exists = [
        figure_already_exists(args.output_dir, stem, args.typefileplot)
        for stem in stems
    ]
    if all(exists):
        return
    narrate(args.typeverb, "Preparing the rocky-body comparison plots...")
    # Earth thermal model: https://arxiv.org/abs/2505.24070
    # Lunar core structure: https://doi.org/10.1126/science.1199375
    earth, moon, mars = (
        get_scenario(name)
        for name in ("earth-inner-core", "moon-inner-core", "mars-core-reference")
    )
    diagnostic_scenarios = (
        (earth, "Earth", EARTH_COLORS[0], "o"),
        (moon, "Moon benchmark", MOON_COLORS[2], "s"),
        (mars, "Mars reference", MARS_COLORS[0], "^"),
    )
    dark_matter_mass = 1e3 * GEV_IN_KG  # [kg]

    if not exists[0]:
        figure, axis = single_plot(height_in=7.2)
        powers = np.logspace(7.0, 14.2, 125)  # [W]
        narrate(args.typeverb, "Comparing phase radii at equal heating power...")
        for scenario, label, color in (
            (earth, "Earth: 6230 K", EARTH_COLORS[0]),
            (moon, "Moon: 1811 K", MOON_COLORS[2]),
        ):
            phase_fraction = np.array(
                [
                    solve_steady_state(scenario, dark_matter_mass, power).phase_radius_m
                    / scenario.thermal_radius_m
                    for power in powers
                ]
            )
            axis.semilogx(powers, phase_fraction, color=color, label=label)
        earth_bounds = tuple(
            np.array(
                [
                    solve_steady_state(
                        replace(earth, phase_temperature_k=proxy_k),
                        dark_matter_mass,
                        power,
                    ).phase_radius_m
                    / earth.thermal_radius_m
                    for power in powers
                ]
            )
            for proxy_k in (5730.0, 6730.0)
        )
        axis.fill_between(
            powers,
            np.minimum(*earth_bounds),
            np.maximum(*earth_bounds),
            color=EARTH_COLORS[0],
            alpha=0.14,
            linewidth=0,
            label="Earth: 5730--6730 K\n(Anzellini et al. 2013)",
        )
        for proxy_k, linestyle in zip((1800.0, 1900.0, 2000.0), ("--", "-.", ":")):
            axis.semilogx(
                powers,
                [
                    solve_steady_state(
                        replace(moon, phase_temperature_k=proxy_k),
                        dark_matter_mass,
                        power,
                    ).phase_radius_m
                    / moon.thermal_radius_m
                    for power in powers
                ],
                color=MOON_COLORS[1],
                linestyle=linestyle,
                label=f"Moon proxy choice: {proxy_k:.0f} K",
            )
        finish_plot(
            figure,
            axis,
            args.output_dir,
            stems[0],
            args.typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.12),
            xlabel="Deposited DM power\n[W]",
            ylabel="Domain fraction above\nmelting proxy",
            ylim=(0.0, 1.02),
        )

    if not exists[1]:
        figure, axis = single_plot(height_in=4.8)
        normalized_radius = np.linspace(0.0, 1.0, 500)
        narrate(args.typeverb, "Comparing the spatial heating profiles...")
        for scenario, label, color, _ in diagnostic_scenarios:
            scale_radius = dark_matter_scale_radius(
                scenario.initial_temperature_k, scenario.density_kg_m3, dark_matter_mass
            )  # [m]
            enclosed_fraction = gaussian_enclosed_fraction(
                scenario.thermal_radius_m, scale_radius
            )
            axis.semilogy(
                normalized_radius,
                np.exp(
                    -2.0
                    * (normalized_radius * scenario.thermal_radius_m) ** 2
                    / scale_radius**2
                ),
                color=color,
                label=(
                    f"{label}\n$R/r_\\chi$ = {scenario.thermal_radius_m / scale_radius:.2f}; "
                    f"$f_R$ = {enclosed_fraction:.3f}"
                ),
            )
        finish_plot(
            figure,
            axis,
            args.output_dir,
            stems[1],
            args.typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.22),
            xlabel="Radius / thermal-domain radius",
            ylabel="Heating / central heating",
            ylim=(1e-8, 2.0),
        )

    if not exists[2]:
        figure, axis = single_plot(height_in=4.8)
        masses_gev = np.logspace(1.0, 6.0, 45)  # [GeV]
        narrate(
            args.typeverb, "Calculating melting-onset powers across dark-matter mass..."
        )
        onset = tuple(
            np.array([onset_power(scenario, mass * GEV_IN_KG) for mass in masses_gev])
            for scenario in (earth, moon)
        )  # [W]
        for scenario, powers_w, color, label in (
            (
                earth,
                onset[0],
                EARTH_COLORS[0],
                "Earth\n(Cappiello & Daylan 2025)",
            ),
            (
                moon,
                onset[1],
                MOON_COLORS[2],
                "Moon benchmark\n(Weber et al. 2011)",
            ),
        ):
            scale_radii = np.array(
                [
                    solve_steady_state(scenario, mass * GEV_IN_KG, power).scale_radius_m
                    for mass, power in zip(masses_gev, powers_w)
                ]
            )  # [m]
            valid = np.array(
                [
                    gaussian_enclosed_fraction(scenario.thermal_radius_m, scale_radius)
                    >= 0.9
                    for scale_radius in scale_radii
                ]
            )
            axis.loglog(masses_gev, powers_w, color=color, linestyle=":", alpha=0.65)
            axis.loglog(
                masses_gev,
                np.where(valid, powers_w, np.nan),
                color=color,
                label=label,
            )
        axis.plot([], [], color="#172126", linestyle=":", label=r"$f_R<0.9$")
        finish_plot(
            figure,
            axis,
            args.output_dir,
            stems[2],
            args.typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.22),
            xlabel="DM mass $m_\\chi$\n[GeV]",
            ylabel="Central-proxy onset power\n[W]",
            ylim=(1e6, None),
        )

    if not exists[3]:
        narrate(args.typeverb, "Mapping finite-domain regimes...")
        domain_to_source = np.logspace(-1.3, 1.2, 500)
        response_completeness = np.array(
            [thermal_response_completeness(ratio) for ratio in domain_to_source]
        )
        containment = np.array(
            [gaussian_enclosed_fraction(ratio, 1.0) for ratio in domain_to_source]
        )
        containment_boundary = optimize.brentq(
            lambda ratio: gaussian_enclosed_fraction(ratio, 1.0) - 0.9,
            0.1,
            5.0,
        )
        response_boundary = optimize.brentq(
            lambda ratio: thermal_response_completeness(ratio) - 0.9,
            1.0,
            10.0,
        )
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(PAPER_COLUMN_WIDTH_IN, 6.2),
            layout="constrained",
            sharex=True,
        )
        axes[0].semilogx(
            domain_to_source,
            response_completeness,
            color=EARTH_COLORS[0],
            label=r"$g_R(x)$",
        )
        axes[0].axhline(0.9, color="#737b82", linestyle=":", linewidth=1.4)
        axes[0].set_ylabel("Dirichlet-domain\nconvergence $g_R$")
        axes[0].set_ylim(-0.03, 1.03)
        axes[1].semilogx(
            domain_to_source,
            containment,
            color=MOON_COLORS[2],
            label=r"$f_R(x)$",
        )
        axes[1].axhline(0.9, color="#737b82", linestyle="--", linewidth=1.4)
        for axis in axes:
            axis.axvspan(
                domain_to_source[0],
                containment_boundary,
                color="#d1495b",
                alpha=0.09,
                linewidth=0,
            )
            axis.axvline(
                containment_boundary,
                color="#737b82",
                linestyle="--",
                linewidth=1.4,
            )
            axis.axvspan(
                containment_boundary,
                response_boundary,
                color="#e9a23b",
                alpha=0.10,
                linewidth=0,
            )
            axis.axvline(
                response_boundary,
                color="#737b82",
                linestyle=":",
                linewidth=1.4,
            )
        for scenario, _, color, marker in diagnostic_scenarios:
            for mass_gev, filled in ((10.0, False), (1e3, True)):
                scale_radius = dark_matter_scale_radius(
                    scenario.initial_temperature_k,
                    scenario.density_kg_m3,
                    mass_gev * GEV_IN_KG,
                )
                ratio = scenario.thermal_radius_m / scale_radius
                axes[0].plot(
                    ratio,
                    thermal_response_completeness(ratio),
                    color=color,
                    marker=marker,
                    markerfacecolor=color if filled else "white",
                    markersize=6.5,
                    linestyle="none",
                )
                axes[1].plot(
                    ratio,
                    gaussian_enclosed_fraction(ratio, 1.0),
                    color=color,
                    marker=marker,
                    markerfacecolor=color if filled else "white",
                    markersize=6.5,
                    linestyle="none",
                    label=(
                        f"{scenario.body_name}: " f"{mass_gev / 1e3:g} TeV"
                        if filled
                        else f"{scenario.body_name}: {mass_gev:g} GeV"
                    ),
                )
        axes[1].set(
            xlabel=r"Domain-to-source ratio $x=R/r_\chi$",
            ylabel=r"Contained power $f_R$",
            ylim=(-0.03, 1.03),
        )
        axes[1].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncols=2,
        )
        save_figure(
            figure,
            args.output_dir,
            stems[3],
            typefileplot=args.typefileplot,
        )

    if not exists[4]:
        narrate(args.typeverb, "Mapping the normalization-free radial fingerprint...")
        figure, axis = single_plot(height_in=4.8)
        masses_gev = np.logspace(0.0, 6.0, 300)
        for scenario, label, color, marker in diagnostic_scenarios:
            radii = np.array([0.0, 0.3, 0.7]) * scenario.thermal_radius_m
            scale_radii = np.array(
                [
                    dark_matter_scale_radius(
                        scenario.initial_temperature_k,
                        scenario.density_kg_m3,
                        mass_gev * GEV_IN_KG,
                    )
                    for mass_gev in masses_gev
                ]
            )
            shape_ratios = np.array(
                [radial_temperature_shape_ratio(scale_radius, *radii) for scale_radius in scale_radii]
            )
            valid = np.array(
                [
                    gaussian_enclosed_fraction(
                        scenario.thermal_radius_m, scale_radius
                    )
                    >= 0.9
                    for scale_radius in scale_radii
                ]
            )
            axis.loglog(
                masses_gev,
                shape_ratios,
                color=color,
                linestyle=":",
                alpha=0.65,
            )
            axis.loglog(
                masses_gev,
                np.where(valid, shape_ratios, np.nan),
                color=color,
                label=label,
            )
            reference_scale = dark_matter_scale_radius(
                scenario.initial_temperature_k,
                scenario.density_kg_m3,
                dark_matter_mass,
            )
            axis.plot(
                1e3,
                radial_temperature_shape_ratio(reference_scale, *radii),
                color=color,
                marker=marker,
                linestyle="none",
            )
        axis.plot([], [], color="#172126", linestyle=":", label=r"$f_R<0.9$")
        finish_plot(
            figure,
            axis,
            args.output_dir,
            stems[4],
            args.typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.22),
            xlabel="DM mass $m_\\chi$\n[GeV]",
            ylabel=r"Radial fingerprint $\mathcal{Q}_{0,0.3,0.7}$",
        )

    if not exists[5]:
        narrate(args.typeverb, "Comparing observation-based heat-flow ceilings...")
        figure, axis = single_plot(height_in=4.8)
        for body_index, (scenario, label, color, marker) in enumerate(
            diagnostic_scenarios
        ):
            constraint = get_heat_flow_constraint(scenario.body_name)
            lower_w, upper_w = constraint.total_heat_loss_range_w(scenario.radius_m)
            center_w = 0.5 * (lower_w + upper_w)
            axis.errorbar(
                body_index,
                center_w,
                yerr=np.array([[center_w - lower_w], [upper_w - center_w]]),
                color=color,
                marker=marker,
                markersize=7.0,
                capsize=5.0,
                linestyle="none",
                label=f"{scenario.body_name}: {constraint.evidence}",
            )
        axis.set_yscale("log")
        axis.set_xticks(range(len(diagnostic_scenarios)))
        axis.set_xticklabels(
            [scenario.body_name for scenario, *_ in diagnostic_scenarios]
        )
        finish_plot(
            figure,
            axis,
            args.output_dir,
            stems[5],
            args.typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.20),
            ylabel="Global surface heat loss\n[W]",
            ylim=(1e11, 1e14),
        )

    if not exists[6]:
        narrate(args.typeverb, "Comparing Earth's radial calorimeters...")
        figure, axis = single_plot(height_in=4.8)
        masses_gev = np.logspace(-1.0, 6.0, 500)
        scale_radii = np.array(
            [
                dark_matter_scale_radius(
                    earth.initial_temperature_k,
                    earth.density_kg_m3,
                    mass_gev * GEV_IN_KG,
                )
                for mass_gev in masses_gev
            ]
        )
        surface_constraint = get_heat_flow_constraint("Earth")
        _, surface_upper_w = surface_constraint.total_heat_loss_range_w(
            earth.radius_m
        )
        surface_fraction = np.array(
            [
                gaussian_enclosed_fraction(earth.radius_m, scale_radius)
                for scale_radius in scale_radii
            ]
        )
        cmb_constraint = get_radial_heat_flow_constraint(
            "Earth core-mantle boundary"
        )
        cmb_fraction = np.array(
            [
                gaussian_enclosed_fraction(
                    cmb_constraint.boundary_radius_m, scale_radius
                )
                for scale_radius in scale_radii
            ]
        )
        surface_ceiling_w = surface_upper_w / surface_fraction
        cmb_ceiling_w = cmb_constraint.outward_power_max_w / cmb_fraction
        axis.loglog(
            masses_gev,
            surface_ceiling_w,
            color=EARTH_COLORS[1],
            label="Surface: 49 TW",
        )
        axis.loglog(
            masses_gev,
            cmb_ceiling_w,
            color=EARTH_COLORS[0],
            label="CMB: 15 TW",
        )
        axis.fill_between(
            masses_gev,
            cmb_constraint.outward_power_min_w / cmb_fraction,
            cmb_ceiling_w,
            color=EARTH_COLORS[0],
            alpha=0.14,
            linewidth=0,
            label="CMB: 13--15 TW preferred",
        )
        axis.plot(
            masses_gev[cmb_fraction >= 0.9],
            cmb_ceiling_w[cmb_fraction >= 0.9],
            color=EARTH_COLORS[0],
            linewidth=3.0,
        )
        finish_plot(
            figure,
            axis,
            args.output_dir,
            stems[6],
            args.typefileplot,
            "upper center",
            legend_anchor=(0.5, -0.22),
            xlabel="DM mass $m_\\chi$\n[GeV]",
            ylabel="Allowed total DM power\n[W]",
        )


if __name__ == "__main__":
    main()
