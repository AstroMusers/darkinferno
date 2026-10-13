"""Gordion-backed capture rates for DarkInferno body scenarios."""

from functools import lru_cache


@lru_cache(maxsize=32)
def _capture_model(composition):
    from gordion.capture import CaptureModel

    return CaptureModel(composition)


def gordion_capture_rate_s(
    scenario,
    dark_matter_mass_gev,
    cross_section_cm2,
    halo_density_gev_cm3=0.4,
    halo_speed_m_s=270_000.0,
):
    """Calculate spin-independent nucleon capture using Gordion."""

    composition = tuple(
        (element.mass_fraction, element.atomic_mass) for element in scenario.elements
    )
    result = _capture_model(composition).calculate(
        radius_m=scenario.radius_m,
        mass_kg=scenario.mass_kg,
        density_gev_cm3=halo_density_gev_cm3,
        speed_m_s=halo_speed_m_s,
        dm_mass_gev=dark_matter_mass_gev,
        sigma_cm2=cross_section_cm2,
        nucleon_mass_gev=0.94,
        scale_from_nucleon=True,
    )
    return result.rate_per_second