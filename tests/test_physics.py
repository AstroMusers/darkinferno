import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import scipy.integrate

from darkinferno.physics import (
    GEV_IN_KG,
    central_temperature_response,
    containment_transition_mass,
    dark_matter_scale_radius,
    dimensionless_central_response,
    evolve_temperature,
    fixed_source_margin_power,
    diagnostic_temperature_power,
    gaussian_enclosed_fraction,
    phase_radius,
    radial_temperature_shape_ratio,
    response_transition_mass,
    solve_steady_state,
    steady_temperature_profile,
    thermal_response_completeness,
    volumetric_heating,
)
from darkinferno.scenarios import SCENARIOS, get_scenario


class ScenarioTests(unittest.TestCase):
    def test_proxy_compositions_are_normalized(self):
        for scenario in SCENARIOS.values():
            total = sum(element.mass_fraction for element in scenario.elements)
            if scenario.body_name != "Earth":
                self.assertAlmostEqual(total, 1.0)
            self.assertLessEqual(scenario.thermal_radius_m, scenario.radius_m)
            self.assertLessEqual(
                scenario.diagnostic_radius_m,
                scenario.thermal_radius_m,
            )


class SteadyStateTests(unittest.TestCase):
    def test_conductivity_override_rejects_invalid_values(self):
        scenario = get_scenario("moon-inner-core")
        for conductivity in (0.0, -1.0, math.nan, math.inf, -math.inf):  # [W m^-1 K^-1]
            for solver, extra in ((solve_steady_state, {}), (steady_temperature_profile, {"radius_m": np.array([0.0])})):
                with self.subTest(conductivity=conductivity, solver=solver.__name__):
                    with self.assertRaisesRegex(ValueError, "conductivity must be finite and positive"):
                        solver(scenario, 1e3 * GEV_IN_KG, 1e8, conductivity_w_m_k=conductivity, **extra)

    def test_conductivity_override_preserves_defaults_and_valid_solutions(self):
        for name in ("earth-inner-core", "moon-inner-core", "mars-core-reference"):
            scenario = get_scenario(name)
            with self.subTest(scenario=name, conductivity=None):
                default = solve_steady_state(scenario, 1e3 * GEV_IN_KG, 1e8)
                explicit = solve_steady_state(scenario, 1e3 * GEV_IN_KG, 1e8, conductivity_w_m_k=None)
                self.assertEqual(default, explicit)
            for conductivity in (10.0, 1000.0):  # [W m^-1 K^-1]
                with self.subTest(scenario=name, conductivity=conductivity):
                    override = solve_steady_state(scenario, 1e3 * GEV_IN_KG, 1e8, conductivity_w_m_k=conductivity)
                    configured = replace(scenario, conductivity_w_m_k=conductivity)
                    expected = solve_steady_state(configured, 1e3 * GEV_IN_KG, 1e8)
                    self.assertEqual(override, expected)
                    radii = np.array([0.0, scenario.diagnostic_radius_m, scenario.thermal_radius_m])  # [m]
                    profile = steady_temperature_profile(scenario, 1e3 * GEV_IN_KG, 1e8, radii, conductivity_w_m_k=conductivity)
                    reference = steady_temperature_profile(configured, 1e3 * GEV_IN_KG, 1e8, radii)
                    self.assertEqual(profile.result, reference.result)
                    np.testing.assert_array_equal(profile.temperature_k, reference.temperature_k)
                    np.testing.assert_array_equal(profile.radius_m, reference.radius_m)

    def test_mars_reference_benchmarks(self):
        scenario = get_scenario("mars-core-reference")
        containment_mass_gev = containment_transition_mass(
            scenario.initial_temperature_k,
            scenario.density_kg_m3,
            scenario.thermal_radius_m,
        ) / GEV_IN_KG
        response_mass_gev = response_transition_mass(
            scenario.initial_temperature_k,
            scenario.density_kg_m3,
            scenario.thermal_radius_m,
        ) / GEV_IN_KG
        scale_radius = dark_matter_scale_radius(
            scenario.initial_temperature_k,
            scenario.density_kg_m3,
            1e3 * GEV_IN_KG,
        )
        radii = np.array([0.0, 0.3, 0.7]) * scenario.thermal_radius_m

        self.assertAlmostEqual(containment_mass_gev, 8.633119, places=6)
        self.assertAlmostEqual(response_mass_gev, 216.925740, places=6)
        self.assertAlmostEqual(
            radial_temperature_shape_ratio(scale_radius, *radii),
            9.522072,
            places=6,
        )

    def test_phase_radius_handles_none_partial_and_complete(self):
        temperature = lambda radius_m: 10.0 - radius_m

        self.assertEqual(phase_radius(temperature, 10.0, 11.0), 0.0)
        self.assertAlmostEqual(phase_radius(temperature, 10.0, 5.0), 5.0)
        self.assertEqual(phase_radius(temperature, 10.0, 0.0), 10.0)

    def test_zero_heating_returns_boundary_temperature(self):
        scenario = get_scenario("moon-inner-core")

        result = solve_steady_state(scenario, 1e3 * GEV_IN_KG, 0.0)

        self.assertEqual(result.central_temperature_k, scenario.boundary_temperature_k)
        self.assertEqual(
            result.diagnostic_temperature_k, scenario.boundary_temperature_k
        )
        self.assertEqual(result.phase_radius_m, 0.0)

    def test_profile_rejects_invalid_radii_before_solving(self):
        scenario = get_scenario("moon-inner-core")
        invalid_radii = (
            [math.nan], [math.inf], [-math.inf], [-1.0],
            [scenario.thermal_radius_m + 1.0], 0.0, np.zeros((1, 2)),
            np.ma.array([0.0, scenario.diagnostic_radius_m], mask=[True, False]),
            np.ma.masked_all(2), np.ma.masked,
        )
        for heat_flow in (0.0, 1e8):  # [W]
            for radius in invalid_radii:  # [m]
                with self.subTest(heat_flow=heat_flow, radius=radius):
                    with patch("darkinferno.physics.solve_steady_state", side_effect=AssertionError("Invalid radii reached solver")) as solver:
                        with self.assertRaisesRegex(ValueError, "radius_m"):
                            steady_temperature_profile(scenario, 1e3 * GEV_IN_KG, heat_flow, radius)
                        solver.assert_not_called()

    def test_profile_preserves_valid_radius_order_containers_and_empty_arrays(self):
        scenario = get_scenario("moon-inner-core")
        radii = np.array([scenario.thermal_radius_m, 0.0, scenario.diagnostic_radius_m])  # [m]
        reference = steady_temperature_profile(scenario, 1e3 * GEV_IN_KG, 1e8, radii)
        for values in (radii.tolist(), tuple(radii), radii, np.ma.asarray(radii)):
            with self.subTest(container=type(values).__name__):
                profile = steady_temperature_profile(scenario, 1e3 * GEV_IN_KG, 1e8, values)
                np.testing.assert_array_equal(profile.radius_m, radii)
                np.testing.assert_array_equal(profile.temperature_k, reference.temperature_k)
                self.assertEqual(profile.result, reference.result)
        empty = steady_temperature_profile(scenario, 1e3 * GEV_IN_KG, 1e8, np.array([]))
        self.assertEqual(empty.radius_m.shape, (0,))
        self.assertEqual(empty.temperature_k.shape, (0,))
        self.assertEqual(empty.result, reference.result)

    def test_profile_matches_steady_state_summary(self):
        scenario = get_scenario("moon-inner-core")
        radius = np.array(
            [0.0, scenario.diagnostic_radius_m, scenario.thermal_radius_m]
        )

        profile = steady_temperature_profile(
            scenario,
            1e3 * GEV_IN_KG,
            1e10,
            radius,
        )

        self.assertAlmostEqual(
            profile.temperature_k[0],
            profile.result.central_temperature_k,
        )
        self.assertAlmostEqual(
            profile.temperature_k[1],
            profile.result.diagnostic_temperature_k,
        )
        self.assertAlmostEqual(
            profile.temperature_k[-1],
            scenario.boundary_temperature_k,
        )

    def test_analytic_response_matches_fixed_source_profile(self):
        scenario = get_scenario("moon-inner-core")
        dark_matter_mass = 1e3 * GEV_IN_KG
        heat_flow = 1e9
        scale_radius = dark_matter_scale_radius(
            scenario.initial_temperature_k,
            scenario.density_kg_m3,
            dark_matter_mass,
        )
        profile = steady_temperature_profile(
            scenario,
            dark_matter_mass,
            heat_flow,
            np.array([0.0]),
            iterations=1,
        )
        expected_rise = heat_flow * central_temperature_response(
            scale_radius,
            scenario.conductivity_w_m_k,
            scenario.thermal_radius_m,
        )

        self.assertAlmostEqual(
            profile.temperature_k[0] - scenario.boundary_temperature_k,
            expected_rise,
        )

    def test_radial_shape_ratio_cancels_power_and_conductivity(self):
        scenario = get_scenario("moon-inner-core")
        dark_matter_mass = 1e3 * GEV_IN_KG
        radii = np.array([0.0, 0.3, 0.7]) * scenario.thermal_radius_m
        expected_ratios = []

        for heat_flow, conductivity in [(1e8, 10.0), (1e11, 1_000.0)]:
            profile = steady_temperature_profile(
                scenario,
                dark_matter_mass,
                heat_flow,
                radii,
                iterations=1,
                conductivity_w_m_k=conductivity,
            )
            temperatures = profile.temperature_k
            expected_ratios.append(
                (temperatures[0] - temperatures[1])
                / (temperatures[1] - temperatures[2])
            )

        scale_radius = dark_matter_scale_radius(
            scenario.boundary_temperature_k,
            scenario.density_kg_m3,
            dark_matter_mass,
        )
        shape_ratio = radial_temperature_shape_ratio(scale_radius, *radii)

        self.assertAlmostEqual(expected_ratios[0], shape_ratio)
        self.assertAlmostEqual(expected_ratios[1], shape_ratio)

    def test_fixed_source_margin_power_benchmarks(self):
        expected_power_gw = {
            "earth-inner-core": 95.74829,
            "moon-inner-core": 2.831743,
        }

        for scenario_name, benchmark_gw in expected_power_gw.items():
            scenario = get_scenario(scenario_name)
            scale_radius = dark_matter_scale_radius(
                scenario.initial_temperature_k,
                scenario.density_kg_m3,
                1e3 * GEV_IN_KG,
            )
            power_gw = (
                fixed_source_margin_power(
                    scenario.phase_temperature_k - scenario.boundary_temperature_k,
                    scale_radius,
                    scenario.conductivity_w_m_k,
                    scenario.thermal_radius_m,
                )
                / 1e9
            )

            self.assertAlmostEqual(power_gw, benchmark_gw, places=5)

    def test_diagnostic_temperature_power_reaches_target(self):
        scenario = get_scenario("earth-inner-core")
        dark_matter_mass_kg = 1e3 * GEV_IN_KG
        target_temperature_k = 10_000.0

        power_w = diagnostic_temperature_power(
            scenario,
            dark_matter_mass_kg,
            target_temperature_k,
        )
        result = solve_steady_state(scenario, dark_matter_mass_kg, power_w)

        self.assertAlmostEqual(
            result.diagnostic_temperature_k,
            target_temperature_k,
            places=6,
        )

    def test_dimensionless_response_matches_dimensional_response(self):
        scale_radius_m = 150_000.0
        conductivity_w_m_k = 100.0
        boundary_radius_m = 1_200_000.0
        dimensional_response = central_temperature_response(
            scale_radius_m, conductivity_w_m_k, boundary_radius_m
        )
        recovered_dimensionless_response = (
            4.0
            * math.pi
            * conductivity_w_m_k
            * boundary_radius_m
            * dimensional_response
        )

        self.assertAlmostEqual(
            recovered_dimensionless_response,
            dimensionless_central_response(boundary_radius_m / scale_radius_m),
        )

    def test_transition_mass_produces_requested_containment(self):
        for scenario in SCENARIOS.values():
            transition_mass = containment_transition_mass(
                scenario.initial_temperature_k,
                scenario.density_kg_m3,
                scenario.thermal_radius_m,
            )
            scale_radius = dark_matter_scale_radius(
                scenario.initial_temperature_k,
                scenario.density_kg_m3,
                transition_mass,
            )

            self.assertAlmostEqual(
                gaussian_enclosed_fraction(
                    scenario.thermal_radius_m,
                    scale_radius,
                ),
                0.9,
            )

    def test_containment_does_not_imply_response_completeness(self):
        containment_boundary = scipy.optimize.brentq(
            lambda ratio: gaussian_enclosed_fraction(ratio, 1.0) - 0.9,
            0.1,
            5.0,
        )

        self.assertAlmostEqual(containment_boundary, 1.250139, places=6)
        self.assertAlmostEqual(
            thermal_response_completeness(containment_boundary),
            0.504951,
            places=6,
        )

    def test_response_transition_mass_produces_requested_completeness(self):
        for scenario in SCENARIOS.values():
            transition_mass = response_transition_mass(
                scenario.initial_temperature_k,
                scenario.density_kg_m3,
                scenario.thermal_radius_m,
            )
            scale_radius = dark_matter_scale_radius(
                scenario.initial_temperature_k,
                scenario.density_kg_m3,
                transition_mass,
            )
            ratio = scenario.thermal_radius_m / scale_radius

            self.assertAlmostEqual(ratio, 6.266571, places=6)
            self.assertAlmostEqual(thermal_response_completeness(ratio), 0.9)

    def test_lunar_melting_temperature_exceeds_reference_temperature(self):
        scenario = get_scenario("moon-inner-core")
        self.assertGreater(scenario.phase_temperature_k, scenario.initial_temperature_k)

    def test_volumetric_heating_integrates_to_input_power(self):
        scenario = get_scenario("moon-inner-core")
        dark_matter_mass = 1e3 * GEV_IN_KG
        heat_flow = 1e10
        scale_radius = dark_matter_scale_radius(
            scenario.initial_temperature_k,
            scenario.density_kg_m3,
            dark_matter_mass,
        )

        integrated_power = scipy.integrate.quad(
            lambda radius_m: float(
                volumetric_heating(
                    np.array(radius_m),
                    scenario.initial_temperature_k,
                    scenario,
                    dark_matter_mass,
                    heat_flow,
                )
            )
            * 4.0
            * math.pi
            * radius_m**2,
            0.0,
            8.0 * scale_radius,
        )[0]

        self.assertAlmostEqual(integrated_power / heat_flow, 1.0, places=8)


class TimeEvolutionTests(unittest.TestCase):
    def test_zero_source_is_invariant_and_heating_raises_center(self):
        scenario = get_scenario("moon-inner-core")
        arguments = {
            "scenario": scenario,
            "dark_matter_mass_kg": 1e3 * GEV_IN_KG,
            "initial_temperature_k": scenario.initial_temperature_k,
            "conductivity_w_m_k": scenario.conductivity_w_m_k,
            "end_time_years": 1e6,
            "time_step_years": 1e5,
            "radial_cells": 60,
            "sample_every": 1,
        }

        zero = evolve_temperature(heat_flow_w=0.0, **arguments)
        heated = evolve_temperature(heat_flow_w=1e10, **arguments)

        self.assertTrue(
            all(
                math.isclose(
                    temperature,
                    scenario.initial_temperature_k,
                    abs_tol=1e-9,
                )
                for temperature in zero.diagnostic_temperatures_k
            )
        )
        self.assertGreater(
            heated.final_central_temperature_k,
            zero.final_central_temperature_k,
        )


if __name__ == "__main__":
    unittest.main()
