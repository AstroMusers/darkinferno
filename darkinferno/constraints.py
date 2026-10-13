from dataclasses import dataclass
import math

import scipy.optimize

from darkinferno.capture import gordion_capture_rate_s
from darkinferno.physics import (
    GEV_IN_JOULE,
    GEV_IN_KG,
    dark_matter_scale_radius,
    gaussian_enclosed_fraction,
)


EARTH_AGE_S = 4.54e9 * 365.25 * 24.0 * 3600.0
EARTH_EVAPORATION_MASS_GEV = 5.0
THERMAL_RELIC_ANNIHILATION_CM3_S = 3.0e-26


@dataclass(frozen=True)
class ParticleHeatingModel:
    halo_density_gev_cm3: float = 0.4
    halo_speed_m_s: float = 270_000.0
    annihilation_cross_section_cm3_s: float = THERMAL_RELIC_ANNIHILATION_CM3_S
    deposition_efficiency: float = 1.0
    system_age_s: float = EARTH_AGE_S
    evaporation_mass_floor_gev: float = EARTH_EVAPORATION_MASS_GEV

    def __post_init__(self):
        for field_name in (
            "halo_density_gev_cm3",
            "halo_speed_m_s",
            "annihilation_cross_section_cm3_s",
            "system_age_s",
            "evaporation_mass_floor_gev",
        ):
            if not math.isfinite(getattr(self, field_name)) or getattr(
                self, field_name
            ) <= 0:
                raise ValueError(f"{field_name} must be finite and greater than zero")
        if not math.isfinite(self.deposition_efficiency) or not (
            0 < self.deposition_efficiency <= 1
        ):
            raise ValueError("deposition_efficiency must be in (0, 1]")


@dataclass(frozen=True)
class ParticleConstraintPoint:
    dark_matter_mass_gev: float
    cross_section_cm2: float
    capture_rate_s: float
    equilibrium_fraction: float
    enclosed_fraction: float
    deposited_power_w: float
    boundary_power_w: float
    evaporation_safe: bool


def effective_annihilation_volume_cm3(scale_radius_m):
    """Return (integral n)^2 / integral n^2 for a Gaussian DM cloud."""

    if not math.isfinite(scale_radius_m) or scale_radius_m <= 0:
        raise ValueError("scale_radius_m must be finite and greater than zero")
    return (2.0 * math.pi) ** 1.5 * (100.0 * scale_radius_m) ** 3


def annihilation_equilibrium_fraction(
    capture_rate_s,
    scale_radius_m,
    annihilation_cross_section_cm3_s=THERMAL_RELIC_ANNIHILATION_CM3_S,
    system_age_s=EARTH_AGE_S,
):
    """Return tanh^2(t sqrt(C C_A)) for capture and annihilation."""

    if capture_rate_s < 0 or not math.isfinite(capture_rate_s):
        raise ValueError("capture_rate_s must be finite and non-negative")
    if capture_rate_s == 0:
        return 0.0
    if annihilation_cross_section_cm3_s <= 0 or system_age_s <= 0:
        raise ValueError("annihilation cross section and system age must be positive")

    annihilation_coefficient_s = (
        annihilation_cross_section_cm3_s
        / effective_annihilation_volume_cm3(scale_radius_m)
    )
    age_in_equilibration_times = system_age_s * math.sqrt(
        capture_rate_s * annihilation_coefficient_s
    )
    return math.tanh(age_in_equilibration_times) ** 2


def annihilation_cross_section_for_equilibrium_cm3_s(
    capture_rate_s,
    scale_radius_m,
    equilibrium_fraction=0.9,
    system_age_s=EARTH_AGE_S,
):
    """Return the annihilation rate coefficient needed for a target equilibrium fraction."""

    if capture_rate_s <= 0 or not math.isfinite(capture_rate_s):
        raise ValueError("capture_rate_s must be finite and greater than zero")
    if not 0 < equilibrium_fraction < 1:
        raise ValueError("equilibrium_fraction must lie between zero and one")
    if system_age_s <= 0 or not math.isfinite(system_age_s):
        raise ValueError("system_age_s must be finite and greater than zero")

    age_in_equilibration_times = math.atanh(math.sqrt(equilibrium_fraction))
    return (
        effective_annihilation_volume_cm3(scale_radius_m)
        * (age_in_equilibration_times / system_age_s) ** 2
        / capture_rate_s
    )


def minimum_deposition_efficiency(
    capture_rate_s,
    dark_matter_mass_gev,
    enclosed_fraction,
    power_ceiling_w,
    equilibrium_fraction=1.0,
):
    """Return the deposited energy fraction required to reach a radial heat ceiling."""

    for field_name, value in (
        ("capture_rate_s", capture_rate_s),
        ("dark_matter_mass_gev", dark_matter_mass_gev),
        ("power_ceiling_w", power_ceiling_w),
    ):
        if value <= 0 or not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite and greater than zero")
    if not 0 < enclosed_fraction <= 1:
        raise ValueError("enclosed_fraction must lie in (0, 1]")
    if not 0 < equilibrium_fraction <= 1:
        raise ValueError("equilibrium_fraction must lie in (0, 1]")

    available_boundary_power_w = (
        capture_rate_s
        * dark_matter_mass_gev
        * GEV_IN_JOULE
        * enclosed_fraction
        * equilibrium_fraction
    )
    return power_ceiling_w / available_boundary_power_w


def earth_core_drift_ceiling_cm2(dark_matter_mass_gev):
    """Return Mack et al.'s published high-mass 1 Gyr core-drift ceiling."""

    if dark_matter_mass_gev <= 0:
        raise ValueError("dark-matter mass must be positive")
    return 2.5e-23 * dark_matter_mass_gev


def logarithmic_capture_slope(
    scenario,
    dark_matter_mass_gev,
    cross_section_cm2,
    model=ParticleHeatingModel(),
    log10_step=0.01,
    capture_rate_function=gordion_capture_rate_s,
):
    """Return the local derivative d ln(C) / d ln(sigma)."""

    if dark_matter_mass_gev <= 0 or cross_section_cm2 <= 0:
        raise ValueError("dark-matter mass and cross section must be positive")
    if not math.isfinite(log10_step) or log10_step <= 0:
        raise ValueError("log10_step must be finite and greater than zero")

    lower_rate_s = capture_rate_function(
        scenario,
        dark_matter_mass_gev,
        cross_section_cm2 / 10.0**log10_step,
        model.halo_density_gev_cm3,
        model.halo_speed_m_s,
    )
    upper_rate_s = capture_rate_function(
        scenario,
        dark_matter_mass_gev,
        cross_section_cm2 * 10.0**log10_step,
        model.halo_density_gev_cm3,
        model.halo_speed_m_s,
    )
    if lower_rate_s <= 0 or upper_rate_s <= 0:
        raise ValueError("capture rates must be greater than zero")
    return math.log(upper_rate_s / lower_rate_s) / math.log(
        10.0 ** (2.0 * log10_step)
    )


def particle_heating_point(
    scenario,
    dark_matter_mass_gev,
    cross_section_cm2,
    boundary_radius_m,
    model=ParticleHeatingModel(),
    capture_rate_function=gordion_capture_rate_s,
):
    """Map particle parameters to deposited power crossing a radial boundary."""

    if dark_matter_mass_gev <= 0 or cross_section_cm2 <= 0:
        raise ValueError("dark-matter mass and cross section must be positive")
    if not 0 < boundary_radius_m <= scenario.radius_m:
        raise ValueError("boundary_radius_m must lie inside the body")

    scale_radius_m = dark_matter_scale_radius(
        scenario.initial_temperature_k,
        scenario.density_kg_m3,
        dark_matter_mass_gev * GEV_IN_KG,
    )
    capture_rate_s = capture_rate_function(
        scenario,
        dark_matter_mass_gev,
        cross_section_cm2,
        model.halo_density_gev_cm3,
        model.halo_speed_m_s,
    )
    equilibrium_fraction = annihilation_equilibrium_fraction(
        capture_rate_s,
        scale_radius_m,
        model.annihilation_cross_section_cm3_s,
        model.system_age_s,
    )
    deposited_power_w = (
        capture_rate_s
        * dark_matter_mass_gev
        * GEV_IN_JOULE
        * equilibrium_fraction
        * model.deposition_efficiency
    )
    enclosed_fraction = gaussian_enclosed_fraction(
        boundary_radius_m, scale_radius_m
    )

    return ParticleConstraintPoint(
        dark_matter_mass_gev=dark_matter_mass_gev,
        cross_section_cm2=cross_section_cm2,
        capture_rate_s=capture_rate_s,
        equilibrium_fraction=equilibrium_fraction,
        enclosed_fraction=enclosed_fraction,
        deposited_power_w=deposited_power_w,
        boundary_power_w=enclosed_fraction * deposited_power_w,
        evaporation_safe=dark_matter_mass_gev >= model.evaporation_mass_floor_gev,
    )


def cross_section_boundary_cm2(
    scenario,
    dark_matter_mass_gev,
    boundary_radius_m,
    power_ceiling_w,
    model=ParticleHeatingModel(),
    log10_cross_section_bounds=(-50.0, -20.0),
    capture_rate_function=gordion_capture_rate_s,
):
    """Find the first cross section whose deposited power reaches the ceiling.

    At a discontinuous capture-regime transition, return the first value on the
    high-power side of the skipped ceiling.
    """

    if power_ceiling_w <= 0:
        raise ValueError("power_ceiling_w must be positive")

    def residual(log10_cross_section):
        point = particle_heating_point(
            scenario,
            dark_matter_mass_gev,
            10.0**log10_cross_section,
            boundary_radius_m,
            model,
            capture_rate_function,
        )
        return math.log10(point.boundary_power_w / power_ceiling_w)

    lower, upper = log10_cross_section_bounds
    previous_log_cross_section = lower
    previous_residual = residual(lower)
    if previous_residual >= 0:
        raise ValueError("lower cross-section bracket already exceeds the heat ceiling")
    root_bracket = None
    number_steps = max(2, math.ceil((upper - lower) / 0.25))
    for step in range(1, number_steps + 1):
        log_cross_section = lower + (upper - lower) * step / number_steps
        current_residual = residual(log_cross_section)
        if current_residual >= 0:
            root_bracket = (previous_log_cross_section, log_cross_section)
            break
        previous_log_cross_section = log_cross_section
        previous_residual = current_residual
    if root_bracket is None:
        raise ValueError("capture saturation does not reach the heat ceiling")

    root = scipy.optimize.brentq(residual, *root_bracket, xtol=1e-8)
    if abs(residual(root)) > 1e-5:
        lower_log_cross_section, upper_log_cross_section = root_bracket
        for _ in range(80):
            midpoint = 0.5 * (lower_log_cross_section + upper_log_cross_section)
            if residual(midpoint) >= 0.0:
                upper_log_cross_section = midpoint
            else:
                lower_log_cross_section = midpoint
        root = upper_log_cross_section
        if residual(root) < 0.0:
            raise ValueError("capture model does not reach the heat ceiling")
    return 10.0**root