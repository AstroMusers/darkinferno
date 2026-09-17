from dataclasses import dataclass
import math

import numpy as np
from scipy import linalg, optimize, special


# CODATA constants: https://physics.nist.gov/cuu/Constants/
GRAVITATIONAL_CONSTANT = 6.67e-11  # [m^3 kg^-1 s^-2]
BOLTZMANN_CONSTANT = 1.38e-23  # [J K^-1]
GEV_IN_KG = 1.78e-27  # [kg GeV^-1]
GEV_IN_JOULE = 1.602e-10  # [J GeV^-1]
SECONDS_PER_YEAR = 3.15e7  # [s yr^-1]


@dataclass(frozen=True)
class SteadyStateResult:
    characteristic_temperature_k: float
    central_temperature_k: float
    diagnostic_temperature_k: float
    phase_radius_m: float
    scale_radius_m: float
    iterations: int


@dataclass(frozen=True)
class SteadyStateProfile:
    radius_m: np.ndarray
    temperature_k: np.ndarray
    result: SteadyStateResult


@dataclass(frozen=True)
class TimeEvolutionResult:
    sample_years: tuple[float, ...]
    diagnostic_temperatures_k: tuple[float, ...]
    phase_crossing_years: float | None
    final_phase_radius_m: float
    final_central_temperature_k: float
    radial_step_m: float
    time_step_years: float


def dark_matter_scale_radius(temperature_k, density_kg_m3, dark_matter_mass_kg):
    """Calculate the isothermal dark-matter scale radius."""

    if min(temperature_k, density_kg_m3, dark_matter_mass_kg) <= 0:
        raise ValueError("temperature, density, and dark-matter mass must be positive")
    scale_radius_m = math.sqrt(
        3.0
        * BOLTZMANN_CONSTANT
        * temperature_k
        / (2.0 * math.pi * GRAVITATIONAL_CONSTANT * density_kg_m3 * dark_matter_mass_kg)
    )  # [m]

    return scale_radius_m


def gaussian_enclosed_fraction(radius_m, scale_radius_m):
    """Calculate the Gaussian annihilation power enclosed within a radius."""

    if radius_m < 0 or scale_radius_m <= 0:
        raise ValueError("radius must be nonnegative and scale radius must be positive")
    ratio = radius_m / scale_radius_m
    fraction = special.erf(math.sqrt(2.0) * ratio) - (
        2.0 * math.sqrt(2.0 / math.pi) * ratio * math.exp(-2.0 * ratio**2)
    )

    return fraction


def dimensionless_central_response(domain_to_source_ratio):
    """Calculate the reference model's finite-domain central response."""

    if domain_to_source_ratio <= 0:
        raise ValueError("domain-to-source ratio must be positive")
    response = 2.0 * math.sqrt(2.0) * domain_to_source_ratio / math.sqrt(
        math.pi
    ) - special.erf(math.sqrt(2.0) * domain_to_source_ratio)

    return response


def thermal_response_completeness(domain_to_source_ratio):
    """Calculate convergence toward the formal infinite Dirichlet domain."""

    if domain_to_source_ratio <= 0:
        raise ValueError("domain-to-source ratio must be positive")
    completeness = 1.0 - (
        math.sqrt(math.pi)
        * special.erf(math.sqrt(2.0) * domain_to_source_ratio)
        / (2.0 * math.sqrt(2.0) * domain_to_source_ratio)
    )

    return completeness


def containment_transition_mass(
    temperature_k, density_kg_m3, boundary_radius_m, enclosed_fraction=0.9
):
    """Calculate the DM mass at a specified Gaussian containment boundary."""

    if min(temperature_k, density_kg_m3, boundary_radius_m) <= 0:
        raise ValueError("temperature, density, and boundary radius must be positive")
    if not 0 < enclosed_fraction < 1:
        raise ValueError("enclosed fraction must lie between zero and one")
    domain_to_source_ratio = optimize.brentq(
        lambda ratio: gaussian_enclosed_fraction(ratio, 1.0) - enclosed_fraction,
        1e-8,
        20.0,
    )
    mass_kg = (
        3.0
        * BOLTZMANN_CONSTANT
        * temperature_k
        * domain_to_source_ratio**2
        / (
            2.0
            * math.pi
            * GRAVITATIONAL_CONSTANT
            * density_kg_m3
            * boundary_radius_m**2
        )
    )

    return mass_kg


def response_transition_mass(
    temperature_k, density_kg_m3, boundary_radius_m, completeness=0.9
):
    """Calculate the DM mass at specified Dirichlet-domain convergence."""

    if min(temperature_k, density_kg_m3, boundary_radius_m) <= 0:
        raise ValueError("temperature, density, and boundary radius must be positive")
    if not 0 < completeness < 1:
        raise ValueError("completeness must lie between zero and one")
    domain_to_source_ratio = optimize.brentq(
        lambda ratio: thermal_response_completeness(ratio) - completeness,
        1e-8,
        1e4,
    )
    mass_kg = (
        3.0
        * BOLTZMANN_CONSTANT
        * temperature_k
        * domain_to_source_ratio**2
        / (
            2.0
            * math.pi
            * GRAVITATIONAL_CONSTANT
            * density_kg_m3
            * boundary_radius_m**2
        )
    )

    return mass_kg


def central_temperature_response(scale_radius_m, conductivity_w_m_k, boundary_radius_m):
    """Calculate the fixed-source central temperature rise per unit power."""

    response_k_w = dimensionless_central_response(
        boundary_radius_m / scale_radius_m
    ) / (
        4.0 * math.pi * conductivity_w_m_k * boundary_radius_m
    )  # [K W^-1]

    return response_k_w


def fixed_source_margin_power(
    temperature_margin_k, scale_radius_m, conductivity_w_m_k, boundary_radius_m
):
    """Calculate the power required to erase an imposed temperature margin."""

    power_w = temperature_margin_k / central_temperature_response(
        scale_radius_m, conductivity_w_m_k, boundary_radius_m
    )  # [W]

    return power_w


def diagnostic_temperature_power(
    scenario,
    dark_matter_mass_kg,
    target_temperature_k,
    log10_power_bounds=(3.0, 16.0),
):
    """Return the power that reaches a target temperature at the reporting radius."""

    if dark_matter_mass_kg <= 0 or target_temperature_k <= 0:
        raise ValueError("dark-matter mass and target temperature must be positive")
    lower, upper = log10_power_bounds
    if lower >= upper:
        raise ValueError("power bounds must be increasing")
    root = optimize.brentq(
        lambda log10_power: solve_steady_state(
            scenario, dark_matter_mass_kg, 10.0**log10_power
        ).diagnostic_temperature_k
        - target_temperature_k,
        lower,
        upper,
    )
    return 10.0**root


def radial_temperature_shape_ratio(
    scale_radius_m, inner_radius_m, middle_radius_m, outer_radius_m
):
    """Calculate a normalization-independent ratio of radial temperature drops."""

    if scale_radius_m <= 0:
        raise ValueError("scale radius must be positive")
    if not 0 <= inner_radius_m < middle_radius_m < outer_radius_m:
        raise ValueError("radii must be nonnegative and strictly increasing")

    def profile_factor(radius_m):
        if radius_m == 0:
            return 2.0 * math.sqrt(2.0) / (math.sqrt(math.pi) * scale_radius_m)
        return special.erf(math.sqrt(2.0) * radius_m / scale_radius_m) / radius_m

    inner_drop = profile_factor(inner_radius_m) - profile_factor(middle_radius_m)
    outer_drop = profile_factor(middle_radius_m) - profile_factor(outer_radius_m)

    return inner_drop / outer_drop


def _steady_temperature_function(
    scale_radius_m,
    heat_flow_w,
    conductivity_w_m_k,
    boundary_radius_m,
    boundary_temperature_k,
):
    """Construct the analytic steady-state temperature function."""

    if heat_flow_w == 0:
        temperature = lambda radius_m: boundary_temperature_k

        return temperature

    coefficient = heat_flow_w / (4.0 * math.pi * conductivity_w_m_k)  # [K m]

    def unshifted_temperature(radius_m):
        """Calculate temperature before imposing the boundary offset."""

        profile_factor = (
            2.0 * math.sqrt(2.0) / (math.sqrt(math.pi) * scale_radius_m)
            if radius_m == 0
            else special.erf(math.sqrt(2.0) * radius_m / scale_radius_m) / radius_m
        )  # [m^-1]
        temperature_k = coefficient * profile_factor  # [K]

        return temperature_k

    boundary_offset = boundary_temperature_k - unshifted_temperature(
        boundary_radius_m
    )  # [K]
    temperature = lambda radius_m: unshifted_temperature(radius_m) + boundary_offset

    return temperature


def phase_radius(temperature, outer_radius_m, phase_temperature_k):
    """Calculate the outermost radius above the phase temperature."""

    central_temperature = temperature(0.0)  # [K]
    boundary_temperature = temperature(outer_radius_m)  # [K]
    if central_temperature < phase_temperature_k:
        phase_radius_m = 0.0  # [m]
    elif boundary_temperature >= phase_temperature_k:
        phase_radius_m = outer_radius_m  # [m]
    else:
        phase_radius_m = optimize.brentq(
            lambda radius_m: temperature(radius_m) - phase_temperature_k,
            0.0,
            outer_radius_m,
        )  # [m]

    return phase_radius_m


def solve_steady_state(
    scenario, dark_matter_mass_kg, heat_flow_w, iterations=10, conductivity_w_m_k=None
):
    """Solve the steady spherical conduction model."""

    if heat_flow_w < 0 or iterations < 1:
        raise ValueError(
            "heat flow must be nonnegative and iterations must be positive"
        )
    conductivity = conductivity_w_m_k or scenario.conductivity_w_m_k  # [W m^-1 K^-1]
    characteristic_temperature = scenario.boundary_temperature_k  # [K]
    temperature = None
    scale_radius = None

    for _ in range(iterations):
        scale_radius = dark_matter_scale_radius(
            characteristic_temperature, scenario.density_kg_m3, dark_matter_mass_kg
        )  # [m]
        temperature = _steady_temperature_function(
            scale_radius,
            heat_flow_w,
            conductivity,
            scenario.thermal_radius_m,
            scenario.boundary_temperature_k,
        )
        characteristic_temperature = (
            characteristic_temperature + temperature(scenario.diagnostic_radius_m)
        ) / 2.0

    result = SteadyStateResult(
        characteristic_temperature_k=characteristic_temperature,
        central_temperature_k=temperature(0.0),
        diagnostic_temperature_k=temperature(scenario.diagnostic_radius_m),
        phase_radius_m=phase_radius(
            temperature, scenario.thermal_radius_m, scenario.phase_temperature_k
        ),
        scale_radius_m=scale_radius,
        iterations=iterations,
    )

    return result


def steady_temperature_profile(
    scenario,
    dark_matter_mass_kg,
    heat_flow_w,
    radius_m,
    iterations=10,
    conductivity_w_m_k=None,
):
    """Calculate a sampled steady-state temperature profile."""

    result = solve_steady_state(
        scenario,
        dark_matter_mass_kg,
        heat_flow_w,
        iterations=iterations,
        conductivity_w_m_k=conductivity_w_m_k,
    )
    conductivity = conductivity_w_m_k or scenario.conductivity_w_m_k  # [W m^-1 K^-1]
    temperature = _steady_temperature_function(
        result.scale_radius_m,
        heat_flow_w,
        conductivity,
        scenario.thermal_radius_m,
        scenario.boundary_temperature_k,
    )  # [K callable]
    radius = np.asarray(radius_m, dtype=float)  # [m]
    if radius.ndim != 1:
        raise ValueError("radius_m must be one-dimensional")
    if np.any(radius < 0) or np.any(radius > scenario.thermal_radius_m):
        raise ValueError("radius_m must lie within the thermal domain")
    temperatures = np.fromiter(
        (temperature(float(value)) for value in radius), dtype=float, count=radius.size
    )  # [K]
    profile = SteadyStateProfile(
        radius_m=radius, temperature_k=temperatures, result=result
    )

    return profile


def volumetric_heating(
    radius_m, temperature_k, scenario, dark_matter_mass_kg, heat_flow_w
):
    """Calculate Gaussian volumetric dark-matter heating."""

    scale_radius = dark_matter_scale_radius(
        temperature_k, scenario.density_kg_m3, dark_matter_mass_kg
    )  # [m]
    normalization = math.pi**1.5 * scale_radius**3 / (2.0 * math.sqrt(2.0))  # [m^3]
    heating_w_m3 = (
        heat_flow_w * np.exp(-2.0 * radius_m**2 / scale_radius**2) / normalization
    )  # [W m^-3]

    return heating_w_m3


def sampled_phase_radius(radius_m, temperature_k, phase_temperature_k):
    """Interpolate the phase radius from a sampled temperature profile."""

    above_phase = np.flatnonzero(temperature_k >= phase_temperature_k)
    if above_phase.size == 0:
        phase_radius_m = 0.0  # [m]

        return phase_radius_m
    last_above = int(above_phase[-1])
    if last_above == len(radius_m) - 1:
        phase_radius_m = float(radius_m[-1])  # [m]

        return phase_radius_m
    first_below = last_above + 1
    temperature_span = temperature_k[last_above] - temperature_k[first_below]
    if temperature_span == 0:
        phase_radius_m = float(radius_m[last_above])  # [m]

        return phase_radius_m
    fraction = (temperature_k[last_above] - phase_temperature_k) / temperature_span
    phase_radius_m = float(
        radius_m[last_above] + fraction * (radius_m[first_below] - radius_m[last_above])
    )  # [m]

    return phase_radius_m


def evolve_temperature(
    scenario,
    dark_matter_mass_kg,
    heat_flow_w,
    initial_temperature_k,
    conductivity_w_m_k,
    end_time_years,
    time_step_years,
    radial_cells=300,
    sample_every=10,
    typeverb=0,
    latent_width_k=50.0,
):
    """Evolve the spherical temperature profile with an implicit solver."""

    if (
        min(
            dark_matter_mass_kg,
            initial_temperature_k,
            conductivity_w_m_k,
            latent_width_k,
        )
        <= 0
    ):
        raise ValueError(
            "mass, temperature, conductivity, and latent width must be positive"
        )
    if (
        heat_flow_w < 0
        or min(end_time_years, time_step_years, radial_cells, sample_every) <= 0
    ):
        raise ValueError(
            "heat flow must be nonnegative and grid/time inputs must be positive"
        )
    radius = np.linspace(0.0, scenario.thermal_radius_m, radial_cells + 1)  # [m]
    radial_step = radius[1] - radius[0]  # [m]
    diagnostic_index = int(round(scenario.diagnostic_radius_m / radial_step))
    temperature = np.full(radial_cells + 1, initial_temperature_k, dtype=float)  # [K]
    boundary_temperature = initial_temperature_k  # [K]
    seconds_per_step = time_step_years * SECONDS_PER_YEAR  # [s]
    number_of_steps = int(math.ceil(end_time_years / time_step_years))
    sample_years = [0.0]  # [yr]
    diagnostic_temperatures = [float(temperature[diagnostic_index])]  # [K]
    phase_crossing_years = None
    progress_interval = max(1, number_of_steps // 10)

    for step in range(1, number_of_steps + 1):
        current_time_years = min(step * time_step_years, end_time_years)  # [yr]
        previous_time_years = (step - 1) * time_step_years  # [yr]
        previous_diagnostic_temperature = float(temperature[diagnostic_index])  # [K]
        effective_seconds = seconds_per_step  # [s]
        if typeverb >= 2 and (step == 1 or step % progress_interval == 0):
            print(
                f"  Thermal evolution: {current_time_years:g} / {end_time_years:g} yr"
            )
        if current_time_years == end_time_years:
            effective_seconds = (
                end_time_years - previous_time_years
            ) * SECONDS_PER_YEAR  # [s]

        latent_heat_capacity = (
            scenario.latent_heat_j_kg
            / (math.sqrt(2.0 * math.pi) * latent_width_k)
            * np.exp(
                -((temperature - scenario.phase_temperature_k) ** 2)
                / (2.0 * latent_width_k**2)
            )
        )  # [J kg^-1 K^-1]
        volumetric_heat_capacity = scenario.density_kg_m3 * (
            scenario.heat_capacity_j_kg_k + latent_heat_capacity
        )  # [J m^-3 K^-1]
        thermal_diffusivity = (
            conductivity_w_m_k / volumetric_heat_capacity
        )  # [m^2 s^-1]
        diffusion_number = (
            thermal_diffusivity * effective_seconds / radial_step**2
        )  # [1]
        if not np.all(np.isfinite(diffusion_number)) or np.any(diffusion_number < 0):
            raise RuntimeError("thermal diffusion number became nonfinite or negative")

        matrix = np.zeros((3, radial_cells + 1))  # [1]
        matrix[1, :] = 1.0 + 2.0 * diffusion_number
        matrix[1, 0] = 1.0 + 6.0 * diffusion_number[0]
        matrix[0, 1] = -6.0 * diffusion_number[0]

        interior = np.arange(1, radial_cells)
        radial_ratio = radial_step / radius[interior]
        matrix[2, interior - 1] = -diffusion_number[interior] * (1.0 - radial_ratio)
        matrix[0, interior + 1] = -diffusion_number[interior] * (1.0 + radial_ratio)

        matrix[1, -1] = 1.0
        matrix[2, -2] = 0.0
        heating_temperature = float(temperature[diagnostic_index])  # [K]
        heating = volumetric_heating(
            radius, heating_temperature, scenario, dark_matter_mass_kg, heat_flow_w
        )  # [W m^-3]
        right_hand_side = (
            temperature + heating * effective_seconds / volumetric_heat_capacity
        )  # [K]
        right_hand_side[-1] = boundary_temperature
        temperature = linalg.solve_banded((1, 1), matrix, right_hand_side)  # [K]
        if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0):
            raise RuntimeError(
                "temperature solver produced nonfinite or nonpositive values"
            )

        diagnostic_temperature = float(temperature[diagnostic_index])  # [K]
        crossed_this_step = False
        if (
            phase_crossing_years is None
            and diagnostic_temperature >= scenario.phase_temperature_k
        ):
            temperature_change = (
                diagnostic_temperature - previous_diagnostic_temperature
            )  # [K]
            crossing_fraction = (
                (scenario.phase_temperature_k - previous_diagnostic_temperature)
                / temperature_change
                if temperature_change > 0
                else 1.0
            )
            phase_crossing_years = previous_time_years + crossing_fraction * (
                current_time_years - previous_time_years
            )
            crossed_this_step = True
        if step % sample_every == 0 or step == number_of_steps or crossed_this_step:
            sample_years.append(current_time_years)
            diagnostic_temperatures.append(diagnostic_temperature)
        if phase_crossing_years is not None:
            break

    result = TimeEvolutionResult(
        sample_years=tuple(sample_years),
        diagnostic_temperatures_k=tuple(diagnostic_temperatures),
        phase_crossing_years=phase_crossing_years,
        final_phase_radius_m=sampled_phase_radius(
            radius, temperature, scenario.phase_temperature_k
        ),
        final_central_temperature_k=float(temperature[0]),
        radial_step_m=radial_step,
        time_step_years=time_step_years,
    )

    return result
