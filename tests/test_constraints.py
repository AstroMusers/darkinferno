import builtins
import math
import unittest
from unittest.mock import patch

from gordion.capture import CaptureModel

from darkinferno.capture import gordion_capture_rate_s
from darkinferno.constraints import (
    ParticleHeatingModel,
    annihilation_cross_section_for_equilibrium_cm3_s,
    annihilation_equilibrium_fraction,
    cross_section_boundary_cm2,
    earth_core_drift_ceiling_cm2,
    effective_annihilation_volume_cm3,
    logarithmic_capture_slope,
    minimum_deposition_efficiency,
    particle_heating_point,
)
from darkinferno.physics import GEV_IN_JOULE
from darkinferno.scenarios import EARTH_ELEMENTS, earth_elements_with_iron_fraction, get_scenario


def linear_capture_rate(
    scenario,
    dark_matter_mass_gev,
    cross_section_cm2,
    halo_density_gev_cm3,
    halo_speed_m_s,
):
    del scenario, dark_matter_mass_gev, halo_density_gev_cm3, halo_speed_m_s
    return 1e25 * cross_section_cm2 / 1e-36


class ParticleConstraintTests(unittest.TestCase):
    def setUp(self):
        self.scenario = get_scenario("earth-inner-core")
        self.equilibrium_model = ParticleHeatingModel(
            annihilation_cross_section_cm3_s=1e-20
        )

    def test_gaussian_effective_volume(self):
        scale_radius_m = 1000.0
        self.assertAlmostEqual(
            effective_annihilation_volume_cm3(scale_radius_m),
            (2.0 * math.pi) ** 1.5 * (100.0 * scale_radius_m) ** 3,
        )

    def test_gordion_capture_rate_uses_scenario_composition_without_asteria(self):
        mass_gev = 10.0
        cross_section_cm2 = 1e-36
        expected_rate = CaptureModel(self.scenario.capture_composition).calculate(
            self.scenario.radius_m,
            self.scenario.mass_kg,
            0.4,
            270_000.0,
            mass_gev,
            cross_section_cm2,
            0.94,
            scale_from_nucleon=True,
        ).rate_per_second
        original_import = builtins.__import__

        def import_without_asteria(name, *args, **kwargs):
            if name.split(".", maxsplit=1)[0].lower() == "asteria":
                raise AssertionError("DarkInferno must not import Asteria.")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_asteria):
            actual_rate = gordion_capture_rate_s(
                self.scenario,
                mass_gev,
                cross_section_cm2,
            )

        self.assertEqual(actual_rate, expected_rate)

    def test_equilibrium_fraction_has_correct_limits(self):
        self.assertEqual(annihilation_equilibrium_fraction(0.0, 1e5), 0.0)
        self.assertAlmostEqual(
            annihilation_equilibrium_fraction(1e30, 1e5), 1.0
        )

    def test_required_annihilation_cross_section_inverts_equilibrium(self):
        capture_rate_s = 1e20
        scale_radius_m = 1e5
        target_fraction = 0.9
        annihilation_cross_section = (
            annihilation_cross_section_for_equilibrium_cm3_s(
                capture_rate_s,
                scale_radius_m,
                target_fraction,
            )
        )

        self.assertAlmostEqual(
            annihilation_equilibrium_fraction(
                capture_rate_s,
                scale_radius_m,
                annihilation_cross_section,
            ),
            target_fraction,
        )

    def test_minimum_deposition_efficiency_reaches_ceiling(self):
        efficiency = minimum_deposition_efficiency(
            capture_rate_s=1e25,
            dark_matter_mass_gev=10.0,
            enclosed_fraction=0.5,
            power_ceiling_w=1e12,
            equilibrium_fraction=0.8,
        )
        recovered_power_w = (
            efficiency * 1e25 * 10.0 * GEV_IN_JOULE * 0.5 * 0.8
        )

        self.assertAlmostEqual(recovered_power_w / 1e12, 1.0)

    def test_equilibrium_limit_recovers_capture_power(self):
        mass_gev = 10.0
        point = particle_heating_point(
            self.scenario,
            mass_gev,
            1e-36,
            self.scenario.radius_m,
            self.equilibrium_model,
            linear_capture_rate,
        )

        self.assertAlmostEqual(point.equilibrium_fraction, 1.0)
        self.assertAlmostEqual(point.enclosed_fraction, 1.0)
        self.assertAlmostEqual(point.deposited_power_w, 1e25 * mass_gev * GEV_IN_JOULE)
        self.assertTrue(point.evaporation_safe)

    def test_deposition_efficiency_scales_power(self):
        full = particle_heating_point(
            self.scenario,
            10.0,
            1e-36,
            self.scenario.radius_m,
            self.equilibrium_model,
            linear_capture_rate,
        )
        half = particle_heating_point(
            self.scenario,
            10.0,
            1e-36,
            self.scenario.radius_m,
            ParticleHeatingModel(
                annihilation_cross_section_cm3_s=1e-20,
                deposition_efficiency=0.5,
            ),
            linear_capture_rate,
        )
        self.assertAlmostEqual(half.deposited_power_w, 0.5 * full.deposited_power_w)

    def test_cross_section_boundary_reaches_ceiling(self):
        ceiling_w = 1e12
        cross_section_cm2 = cross_section_boundary_cm2(
            self.scenario,
            10.0,
            self.scenario.radius_m,
            ceiling_w,
            self.equilibrium_model,
            capture_rate_function=linear_capture_rate,
        )
        point = particle_heating_point(
            self.scenario,
            10.0,
            cross_section_cm2,
            self.scenario.radius_m,
            self.equilibrium_model,
            linear_capture_rate,
        )
        self.assertAlmostEqual(point.boundary_power_w / ceiling_w, 1.0, places=7)

    def test_discontinuous_capture_boundary_returns_first_excluded_cross_section(self):
        transition_cross_section = 1e-36

        def step_capture_rate(
            scenario,
            dark_matter_mass_gev,
            cross_section_cm2,
            halo_density_gev_cm3,
            halo_speed_m_s,
        ):
            del scenario, dark_matter_mass_gev, halo_density_gev_cm3, halo_speed_m_s
            return 1e25 if cross_section_cm2 >= transition_cross_section else 1e23

        ceiling_w = 1e15
        cross_section_cm2 = cross_section_boundary_cm2(
            self.scenario,
            10.0,
            self.scenario.radius_m,
            ceiling_w,
            self.equilibrium_model,
            log10_cross_section_bounds=(-40.0, -32.0),
            capture_rate_function=step_capture_rate,
        )
        point = particle_heating_point(
            self.scenario,
            10.0,
            cross_section_cm2,
            self.scenario.radius_m,
            self.equilibrium_model,
            step_capture_rate,
        )

        self.assertGreaterEqual(point.boundary_power_w, ceiling_w)
        self.assertAlmostEqual(cross_section_cm2 / transition_cross_section, 1.0, places=7)

    def test_logarithmic_capture_slope_recovers_linear_scaling(self):
        slope = logarithmic_capture_slope(
            self.scenario,
            10.0,
            1e-36,
            self.equilibrium_model,
            capture_rate_function=linear_capture_rate,
        )

        self.assertAlmostEqual(slope, 1.0)

    def test_evaporation_mask_is_conservative(self):
        point = particle_heating_point(
            self.scenario,
            4.9,
            1e-36,
            self.scenario.radius_m,
            self.equilibrium_model,
            linear_capture_rate,
        )
        self.assertFalse(point.evaporation_safe)

    def test_drift_ceiling_has_published_high_mass_limit(self):
        mass_gev = 1e8
        coefficient = earth_core_drift_ceiling_cm2(mass_gev) / mass_gev
        self.assertEqual(coefficient, 2.5e-23)

    def test_earth_iron_proxy_preserves_represented_mass(self):
        elements = earth_elements_with_iron_fraction(0.27)

        self.assertAlmostEqual(elements[0].mass_fraction, 0.27)
        self.assertAlmostEqual(
            sum(element.mass_fraction for element in elements),
            sum(element.mass_fraction for element in EARTH_ELEMENTS),
        )


if __name__ == "__main__":
    unittest.main()