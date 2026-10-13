from dataclasses import dataclass


@dataclass(frozen=True)
class Element:
    mass_fraction: float
    atomic_mass: float


@dataclass(frozen=True)
class BodyScenario:
    name: str
    body_name: str
    radius_m: float
    mass_kg: float
    elements: tuple[Element, ...]
    thermal_radius_m: float
    density_kg_m3: float
    conductivity_w_m_k: float
    heat_capacity_j_kg_k: float
    initial_temperature_k: float
    boundary_temperature_k: float
    phase_temperature_k: float
    latent_heat_j_kg: float
    diagnostic_radius_m: float
    provenance: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self):
        """Validate fast physical invariants of a body scenario."""

        for field_name in (
            "radius_m",
            "mass_kg",
            "thermal_radius_m",
            "density_kg_m3",
            "conductivity_w_m_k",
            "heat_capacity_j_kg_k",
            "initial_temperature_k",
            "boundary_temperature_k",
            "phase_temperature_k",
            "latent_heat_j_kg",
            "diagnostic_radius_m",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than zero")
        if self.thermal_radius_m > self.radius_m:
            raise ValueError("thermal_radius_m cannot exceed radius_m")
        if self.diagnostic_radius_m > self.thermal_radius_m:
            raise ValueError("diagnostic_radius_m cannot exceed thermal_radius_m")
        if not self.elements:
            raise ValueError("elements cannot be empty")
        if any(
            element.mass_fraction <= 0 or element.atomic_mass <= 0
            for element in self.elements
        ):
            raise ValueError(
                "element mass fractions and atomic masses must be positive"
            )
        if sum(element.mass_fraction for element in self.elements) > 1.0 + 1e-8:
            raise ValueError("element mass fractions cannot sum to more than one")

    @property
    def capture_composition(self):
        """Return mass fractions and nuclear masses for capture models."""

        return tuple(
            (element.mass_fraction, element.atomic_mass) for element in self.elements
        )


# Earth bulk composition used by the Asteria v1.0 reference model.
EARTH_ELEMENTS = tuple(
    Element(*values)
    for values in (
        (0.32, 55.8),
        (0.29, 16.0),
        (0.15, 24.0),
        (0.14, 28.0),
        (0.017, 40.0),
        (0.015, 27.0),
    )
)  # mass fractions and atomic masses [u]


def earth_elements_with_iron_fraction(iron_mass_fraction):
    """Return a bulk-Earth proxy with rescaled non-iron abundances."""

    represented_mass_fraction = sum(
        element.mass_fraction for element in EARTH_ELEMENTS
    )
    if not 0 < iron_mass_fraction < represented_mass_fraction:
        raise ValueError("iron_mass_fraction must lie within the represented mass")
    reference_iron_fraction = EARTH_ELEMENTS[0].mass_fraction
    noniron_scale = (
        represented_mass_fraction - iron_mass_fraction
    ) / (represented_mass_fraction - reference_iron_fraction)
    return (Element(iron_mass_fraction, EARTH_ELEMENTS[0].atomic_mass),) + tuple(
        Element(element.mass_fraction * noniron_scale, element.atomic_mass)
        for element in EARTH_ELEMENTS[1:]
    )
MOON_PROXY_ELEMENTS = tuple(
    Element(*values)
    for values in (
        (0.43, 16.0),
        (0.21, 28.0),
        (0.20, 24.0),
        (0.10, 55.8),
        (0.03, 40.0),
        (0.03, 27.0),
    )
)  # assumed mass fractions and atomic masses [u]
MARS_PROXY_ELEMENTS = tuple(
    Element(*values)
    for values in (
        (0.36, 16.0),
        (0.25, 55.8),
        (0.17, 24.0),
        (0.17, 28.0),
        (0.03, 32.1),
        (0.02, 40.0),
    )
)  # assumed whole-body proxy mass fractions and atomic masses [u]

COMMON_MOON_WARNINGS = (
    "The 1754 K boundary temperature is an imposed benchmark, not a temperature "
    "evaluated at the 240 km model boundary from a radial lunar geotherm.",
    "Lunar capture lies below the escape-velocity range used to calibrate Gordion's "
    "reflection fit.",
    "Gordion assumes homogeneous whole-body composition and does not model "
    "dark-matter evaporation.",
    "Capture-annihilation equilibrium and complete local energy deposition are assumed.",
    "The lunar elemental abundances are a bulk-silicate proxy, not a validated "
    "whole-Moon composition.",
    "The 1811 K melting temperature is an elemental-iron proxy; lunar-core pressure "
    "raises melting temperatures while light elements lower them.",
)

# Lunar mean radius: https://doi.org/10.1007/s10569-017-9805-5
# Lunar mass from Lunar Prospector gravity: https://doi.org/10.1006/icar.2000.6573
# Lunar core structure: https://doi.org/10.1126/science.1199375
MOON_COMMON = {
    "radius_m": 1.7374e6,  # [m]
    "mass_kg": 7.346e22,  # [kg]
    "elements": MOON_PROXY_ELEMENTS,
    "thermal_radius_m": 240_000.0,  # [m]
    "density_kg_m3": 7800.0,  # [kg m^-3]
    "heat_capacity_j_kg_k": 800.0,  # [J kg^-1 K^-1]
    "latent_heat_j_kg": 247_000.0,  # [J kg^-1]
    "diagnostic_radius_m": 120_000.0,  # [m]
}
MOON_STRUCTURE_PROVENANCE = (
    "Archinal et al. (2018), doi:10.1007/s10569-017-9805-5",
    "Konopliv (2001), doi:10.1006/icar.2000.6573",
    "Weber et al. (2011), doi:10.1126/science.1199375",
)


def moon_scenario(
    name, conductivity_w_m_k, temperature_k, phase_temperature_k, provenance
):
    """Construct the lunar inner-core scenario."""

    scenario = BodyScenario(
        name=name,
        body_name="Moon",
        conductivity_w_m_k=conductivity_w_m_k,
        initial_temperature_k=temperature_k,
        boundary_temperature_k=temperature_k,
        phase_temperature_k=phase_temperature_k,
        provenance=MOON_STRUCTURE_PROVENANCE + (provenance,),
        warnings=COMMON_MOON_WARNINGS,
        **MOON_COMMON,
    )

    return scenario


SCENARIOS = {
    # Earth thermal model: https://arxiv.org/abs/2505.24070
    # Pure-iron melting at the inner-core boundary: https://doi.org/10.1126/science.1233514
    "earth-inner-core": BodyScenario(
        name="earth-inner-core",
        body_name="Earth",
        radius_m=6.38e6,  # [m]
        mass_kg=5.97e24,  # [kg]
        elements=EARTH_ELEMENTS,
        thermal_radius_m=1.2e6,  # [m]
        density_kg_m3=13_000.0,  # [kg m^-3]
        conductivity_w_m_k=100.0,  # [W m^-1 K^-1]
        heat_capacity_j_kg_k=(100.0 / 24e-6) / 13_000.0,  # [J kg^-1 K^-1]
        initial_temperature_k=5500.0,  # [K]
        boundary_temperature_k=5500.0,  # [K]
        phase_temperature_k=6230.0,  # [K]
        latent_heat_j_kg=247_000.0,  # [J kg^-1]
        diagnostic_radius_m=400_000.0,  # [m]
        provenance=(
            "Cappiello & Daylan (2025), arXiv:2505.24070",
            "Anzellini et al. (2013), doi:10.1126/science.1233514",
        ),
        warnings=(
            "The 6230 K melting temperature is a pure-iron ICB estimate with a reported "
            "500 K uncertainty; light elements lower the alloy melting temperature.",
            "The capture composition models 93.2% of Earth's mass and omits minor elements.",
        ),
    ),
    # Elemental-iron melting point: https://doi.org/10.1201/9781315380476
    "moon-inner-core": moon_scenario(
        "moon-inner-core",
        40.0,
        1754.0,
        1811.0,
        "Imposed 1754 K benchmark; Haynes et al. (2016), doi:10.1201/9781315380476",
    ),  # [W m^-1 K^-1], [K], [K]
    # Mean radius: https://doi.org/10.1007/s10569-017-9805-5
    # Core radius: https://doi.org/10.1126/science.abi7730
    "mars-core-reference": BodyScenario(
        name="mars-core-reference",
        body_name="Mars",
        radius_m=3.3895e6,  # [m]
        mass_kg=6.4171e23,  # [kg]
        elements=MARS_PROXY_ELEMENTS,
        thermal_radius_m=1.83e6,  # [m]
        density_kg_m3=6000.0,  # [kg m^-3]
        conductivity_w_m_k=40.0,  # [W m^-1 K^-1]
        heat_capacity_j_kg_k=800.0,  # [J kg^-1 K^-1]
        initial_temperature_k=2000.0,  # [K]
        boundary_temperature_k=2000.0,  # [K]
        phase_temperature_k=2100.0,  # [K]
        latent_heat_j_kg=247_000.0,  # [J kg^-1]
        diagnostic_radius_m=915_000.0,  # [m]
        provenance=(
            "Archinal et al. (2018), doi:10.1007/s10569-017-9805-5",
            "Stahler et al. (2021), doi:10.1126/science.abi7730",
            "Plesa et al. (2018), doi:10.1029/2018GL080728",
        ),
        warnings=(
            "The 1830 km thermal radius is an InSight-scale core reference, not a "
            "fixed-temperature boundary inferred from seismology.",
            "The uniform 2000 K temperature and constant thermal properties are "
            "reference assumptions, not direct Martian observations.",
            "The 2100 K phase threshold is a required diagnostic placeholder and is "
            "not used to claim Martian melting.",
            "The elemental abundances are a whole-body proxy and are not a validated "
            "radial Martian composition.",
            "Gordion assumes homogeneous whole-body composition and does not model "
            "dark-matter evaporation.",
            "Capture-annihilation equilibrium and complete local energy deposition are assumed.",
        ),
    ),
}


def get_scenario(name):
    """Retrieve a named body scenario."""

    scenario = SCENARIOS[name]

    return scenario
