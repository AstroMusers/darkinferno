import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import numpy as np

from darkinferno.inference import (
    infer_heat_source_power,
    radial_temperature_log_likelihood,
)
from darkinferno.physics import GEV_IN_KG, steady_temperature_profile
from darkinferno.scenarios import get_scenario


class PCATInferenceTests(unittest.TestCase):
    def setUp(self):
        self.scenario = get_scenario("mars-core-reference")
        self.mass_gev = 100.0
        self.radius_m = np.array([0.0, 250_000.0, 500_000.0, 900_000.0, 1_400_000.0])
        self.true_log10_power_w = 10.0
        profile = steady_temperature_profile(
            self.scenario,
            self.mass_gev * GEV_IN_KG,
            10.0**self.true_log10_power_w,
            self.radius_m,
        )
        self.temperature_k = profile.temperature_k
        self.uncertainty_k = np.full(self.radius_m.size, 3.0)

    def test_profile_likelihood_prefers_injected_source_power(self):
        state = {
            "scenario": self.scenario,
            "dark_matter_mass_kg": self.mass_gev * GEV_IN_KG,
            "radius_m": self.radius_m,
            "temperature_k": self.temperature_k,
            "uncertainty_k": self.uncertainty_k,
            "iterations": 10,
        }

        at_truth = radial_temperature_log_likelihood([10.0], state)
        away_from_truth = radial_temperature_log_likelihood([11.0], state)

        self.assertGreater(at_truth, away_from_truth)
        self.assertTrue(np.isfinite(at_truth))

    def test_profile_likelihood_preserves_unequal_uncertainty_normalization(self):
        observed = self.temperature_k + np.array([0.5, -1.0, 2.0, -3.0, 4.0])  # [K]
        uncertainty = np.array([0.2, 0.5, 1.0, 2.0, 3.0])  # [K]
        state = {
            "scenario": self.scenario,
            "dark_matter_mass_kg": self.mass_gev * GEV_IN_KG,
            "radius_m": self.radius_m,
            "temperature_k": observed,
            "uncertainty_k": uncertainty,
            "iterations": 10,
        }
        residual = (observed - self.temperature_k) / uncertainty
        expected = -0.5 * np.sum(residual**2 + np.log(2.0 * np.pi * uncertainty**2))
        actual = radial_temperature_log_likelihood([self.true_log10_power_w], state)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_profile_likelihood_retains_invalid_power_rejection(self):
        for log_power in (np.nan, np.inf, 301.0):
            self.assertEqual(radial_temperature_log_likelihood([log_power], {}), -np.inf)

    def test_inference_uses_fixed_chain_sampler_and_returns_power_samples(self):
        fake_pcat = ModuleType("pcat")
        fake_pcat.__path__ = []
        fake_fixed = ModuleType("pcat.fixed")
        captured = {}

        def sample_fixed_chains(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return np.full((2, 5, 1), 10.0), np.zeros((2, 5))

        fake_fixed.sample_fixed_chains = sample_fixed_chains
        with patch.dict(sys.modules, {"pcat": fake_pcat, "pcat.fixed": fake_fixed}):
            posterior = infer_heat_source_power(
                self.scenario,
                self.mass_gev,
                self.radius_m,
                self.temperature_k,
                self.uncertainty_k,
                chain_count=2,
                sample_count=10,
                burn_count=5,
            )

        self.assertEqual(captured["args"][3], ("log10_heat_source_power_w",))
        self.assertEqual(captured["kwargs"]["typeseed"], 0)
        self.assertEqual(posterior.scenario_name, self.scenario.name)
        self.assertEqual(posterior.log10_power_chain.shape, (2, 5))
        np.testing.assert_allclose(posterior.power_samples_w, 1.0e10)
        np.testing.assert_allclose(posterior.power_interval_w, 1.0e10)

    def test_invalid_temperature_uncertainty_is_rejected(self):
        uncertainty = self.uncertainty_k.copy()
        uncertainty[0] = 0.0
        with self.assertRaisesRegex(ValueError, "uncertainties finite and positive"):
            infer_heat_source_power(
                self.scenario,
                self.mass_gev,
                self.radius_m,
                self.temperature_k,
                uncertainty,
            )


if __name__ == "__main__":
    unittest.main()