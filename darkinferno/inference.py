"""Posterior inference for DarkInferno thermal profiles."""

from dataclasses import dataclass

import numpy as np

from darkinferno.physics import GEV_IN_KG, steady_temperature_profile


@dataclass(frozen=True)
class HeatSourcePosterior:
    """Posterior heat-source power samples conditional on a thermal scenario."""

    log10_power_chain: np.ndarray
    log_probability: np.ndarray
    dark_matter_mass_gev: float
    scenario_name: str

    @property
    def power_samples_w(self) -> np.ndarray:
        """Return posterior source-power samples [W]."""

        return 10.0**self.log10_power_chain

    @property
    def power_interval_w(self) -> tuple[float, float, float]:
        """Return median and central 68% source-power interval [W]."""

        quantiles = np.percentile(self.power_samples_w, [16.0, 50.0, 84.0])
        return tuple(float(value) for value in quantiles)


def radial_temperature_log_likelihood(parameters, state) -> float:
    """Evaluate independent Gaussian radial-temperature measurement likelihood."""

    log10_power_w = float(parameters[0])
    if not np.isfinite(log10_power_w) or log10_power_w > 300.0:
        return -np.inf
    heat_flow_w = 10.0**log10_power_w  # [W]
    try:
        profile = steady_temperature_profile(
            state["scenario"],
            state["dark_matter_mass_kg"],
            heat_flow_w,
            state["radius_m"],
            iterations=state["iterations"],
        )
    except (ValueError, OverflowError, FloatingPointError):
        return -np.inf
    residual = (
        state["temperature_k"] - profile.temperature_k
    ) / state["uncertainty_k"]
    if not np.isfinite(residual).all():
        return -np.inf
    return float(-0.5 * np.sum(residual**2 + np.log(2.0 * np.pi * state["uncertainty_k"] ** 2)))


def infer_heat_source_power(
    scenario,
    dark_matter_mass_gev: float,
    radius_m,
    temperature_k,
    uncertainty_k,
    *,
    log10_power_bounds=(6.0, 14.0),
    chain_count: int = 4,
    sample_count: int = 1200,
    burn_count: int = 600,
    seed: int = 0,
    pathbase: str | None = None,
    iterations: int = 10,
) -> HeatSourcePosterior:
    """Fit source power to radial temperatures with Bayesian sampling.

    The fit is conditional on the supplied body scenario, dark-matter mass,
    fixed scenario boundary temperature, and independent Gaussian temperature
    errors. A uniform prior in log10(power [W]) is used. This API is intended
    for calibrated radial-temperature data; the included example uses an
    explicitly synthetic profile and is not an observational constraint.
    """

    radius_m = np.asarray(radius_m, dtype=float)
    temperature_k = np.asarray(temperature_k, dtype=float)
    uncertainty_k = np.asarray(uncertainty_k, dtype=float)
    if (radius_m.ndim != 1 or temperature_k.shape != radius_m.shape
            or uncertainty_k.shape != radius_m.shape or radius_m.size < 3):
        raise ValueError("radius, temperature, and uncertainty must be matching 1D arrays with at least 3 points")
    if (not np.isfinite(radius_m).all() or np.any(radius_m < 0.0)
            or np.any(radius_m > scenario.thermal_radius_m)):
        raise ValueError("radii must be finite and lie within the scenario thermal domain")
    if (not np.isfinite(temperature_k).all() or not np.isfinite(uncertainty_k).all()
            or np.any(uncertainty_k <= 0.0)):
        raise ValueError("temperatures must be finite and uncertainties finite and positive")
    if not np.isfinite(dark_matter_mass_gev) or dark_matter_mass_gev <= 0.0:
        raise ValueError("dark_matter_mass_gev must be finite and positive")
    lower_power, upper_power = log10_power_bounds
    if not np.isfinite([lower_power, upper_power]).all() or lower_power >= upper_power:
        raise ValueError("log10_power_bounds must be finite and increasing")
    if chain_count < 1 or sample_count <= burn_count or burn_count < 0 or iterations < 1:
        raise ValueError("chain_count and iterations must be positive; sample_count must exceed burn_count")

    try:
        from pcat.fixed import sample_fixed_chains
    except ImportError as exception:
        raise ImportError(
            "DarkInferno inference dependencies are missing; install requirements.txt."
        ) from exception

    state = {
        "scenario": scenario,
        "dark_matter_mass_kg": float(dark_matter_mass_gev) * GEV_IN_KG,  # [kg]
        "radius_m": radius_m,
        "temperature_k": temperature_k,
        "uncertainty_k": uncertainty_k,
        "iterations": int(iterations),
    }
    chains, log_probability = sample_fixed_chains(
        state,
        radial_temperature_log_likelihood,
        None,
        ("log10_heat_source_power_w",),
        ("self",),
        np.asarray([lower_power]),
        np.asarray([upper_power]),
        None,
        None,
        None,
        chain_count,
        sample_count,
        burn_count,
        pathbase=pathbase,
        typeverb=0,
        typeseed=seed,
        boolmakeplot=False,
        boolmakeplotinit=False,
        makeanim=False,
        booldiag=False,
    )
    return HeatSourcePosterior(
        log10_power_chain=np.asarray(chains, dtype=float)[:, :, 0],
        log_probability=np.asarray(log_probability, dtype=float),
        dark_matter_mass_gev=float(dark_matter_mass_gev),
        scenario_name=scenario.name,
    )