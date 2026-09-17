import math
import unittest

from darkinferno.observations import (
    HEAT_FLOW_CONSTRAINTS,
    RADIAL_HEAT_FLOW_CONSTRAINTS,
    HeatFlowConstraint,
    RadialHeatFlowConstraint,
)
from darkinferno.scenarios import SCENARIOS


class HeatFlowConstraintTests(unittest.TestCase):
    def test_all_scenario_bodies_have_heat_flow_constraints(self):
        self.assertEqual(
            {scenario.body_name for scenario in SCENARIOS.values()},
            set(HEAT_FLOW_CONSTRAINTS),
        )

    def test_total_heat_loss_uses_spherical_surface_area(self):
        mars = SCENARIOS["mars-core-reference"]
        constraint = HEAT_FLOW_CONSTRAINTS["Mars"]
        total_min_w, total_max_w = constraint.total_heat_loss_range_w(mars.radius_m)

        self.assertAlmostEqual(
            total_min_w,
            4.0 * math.pi * mars.radius_m**2 * constraint.surface_flux_min_w_m2,
        )
        self.assertAlmostEqual(
            total_max_w,
            4.0 * math.pi * mars.radius_m**2 * constraint.surface_flux_max_w_m2,
        )

    def test_reported_total_heat_loss_takes_precedence(self):
        moon = SCENARIOS["moon-inner-core"]
        constraint = HEAT_FLOW_CONSTRAINTS["Moon"]

        self.assertEqual(
            constraint.total_heat_loss_range_w(moon.radius_m),
            (2.8e11, 4.1e11),
        )

    def test_invalid_constraint_is_rejected(self):
        with self.assertRaises(ValueError):
            HeatFlowConstraint(
                body_name="Example",
                evidence="model inference",
                provenance="example",
                caveat="example",
                reported_total_min_w=2.0,
                reported_total_max_w=1.0,
            )

    def test_radial_constraint_corrects_for_enclosed_dm_power(self):
        constraint = RADIAL_HEAT_FLOW_CONSTRAINTS["Earth core-mantle boundary"]

        self.assertAlmostEqual(
            constraint.total_dm_power_ceiling_w(constraint.boundary_radius_m),
            constraint.outward_power_max_w
            / 0.7385358700508894,
        )

    def test_radial_constraint_recovers_boundary_power_for_compact_source(self):
        constraint = RADIAL_HEAT_FLOW_CONSTRAINTS["Earth core-mantle boundary"]

        self.assertAlmostEqual(
            constraint.total_dm_power_ceiling_w(
                constraint.boundary_radius_m / 20.0
            ),
            constraint.outward_power_max_w,
        )

    def test_cmb_range_uses_modern_preferred_interval(self):
        constraint = RADIAL_HEAT_FLOW_CONSTRAINTS["Earth core-mantle boundary"]

        self.assertEqual(constraint.outward_power_min_w, 13e12)
        self.assertEqual(constraint.outward_power_max_w, 15e12)
        self.assertIn("held fixed", constraint.caveat)

    def test_invalid_radial_constraint_is_rejected(self):
        with self.assertRaises(ValueError):
            RadialHeatFlowConstraint(
                body_name="Example",
                boundary_name="example boundary",
                boundary_radius_m=1.0,
                outward_power_min_w=2.0,
                outward_power_max_w=1.0,
                evidence="example",
                provenance="example",
                caveat="example",
            )


if __name__ == "__main__":
    unittest.main()
