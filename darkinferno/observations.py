from dataclasses import dataclass
import math

from darkinferno.physics import gaussian_enclosed_fraction


@dataclass(frozen=True)
class HeatFlowConstraint:
    body_name: str
    evidence: str
    provenance: str
    caveat: str
    surface_flux_min_w_m2: float | None = None
    surface_flux_max_w_m2: float | None = None
    reported_total_min_w: float | None = None
    reported_total_max_w: float | None = None

    def __post_init__(self):
        """Validate a planetary surface heat-flow constraint."""

        if self.evidence not in {
            "global measurement",
            "regional inference",
            "model inference",
        }:
            raise ValueError("unsupported evidence classification")
        if (self.surface_flux_min_w_m2 is None) != (
            self.surface_flux_max_w_m2 is None
        ):
            raise ValueError("surface heat-flow bounds must be supplied together")
        if self.surface_flux_min_w_m2 is not None and not (
            0 < self.surface_flux_min_w_m2 <= self.surface_flux_max_w_m2
        ):
            raise ValueError("surface heat-flow bounds must be positive and ordered")
        if (self.reported_total_min_w is None) != (
            self.reported_total_max_w is None
        ):
            raise ValueError("reported total heat-loss bounds must be supplied together")
        if self.reported_total_min_w is not None and not (
            0 < self.reported_total_min_w <= self.reported_total_max_w
        ):
            raise ValueError(
                "reported total heat-loss bounds must be positive and ordered"
            )
        if self.surface_flux_min_w_m2 is None and self.reported_total_min_w is None:
            raise ValueError("a surface-flux or reported-total interval is required")

    def total_heat_loss_range_w(self, radius_m):
        """Convert a surface heat-flux interval to a global power interval."""

        if radius_m <= 0:
            raise ValueError("radius_m must be greater than zero")
        if self.reported_total_min_w is not None:
            return self.reported_total_min_w, self.reported_total_max_w

        area_m2 = 4.0 * math.pi * radius_m**2
        return (
            area_m2 * self.surface_flux_min_w_m2,
            area_m2 * self.surface_flux_max_w_m2,
        )


@dataclass(frozen=True)
class RadialHeatFlowConstraint:
    body_name: str
    boundary_name: str
    boundary_radius_m: float
    outward_power_min_w: float
    outward_power_max_w: float
    evidence: str
    provenance: str
    caveat: str

    def __post_init__(self):
        """Validate a heat-flow constraint at an internal spherical boundary."""

        if self.boundary_radius_m <= 0:
            raise ValueError("boundary_radius_m must be greater than zero")
        if not 0 < self.outward_power_min_w <= self.outward_power_max_w:
            raise ValueError("outward-power bounds must be positive and ordered")

    def total_dm_power_ceiling_w(self, scale_radius_m):
        """Return the total DM power allowed by heating enclosed at the boundary."""

        enclosed_fraction = gaussian_enclosed_fraction(
            self.boundary_radius_m, scale_radius_m
        )
        if enclosed_fraction == 0:
            return math.inf

        return self.outward_power_max_w / enclosed_fraction


HEAT_FLOW_CONSTRAINTS = {
    "Earth": HeatFlowConstraint(
        body_name="Earth",
        evidence="global measurement",
        provenance="Davies (2013), doi:10.1002/ggge.20271",
        caveat=(
            "The interval represents the reported global surface heat loss and is a "
            "conservative ceiling on any additional steady heat source."
        ),
        reported_total_min_w=45e12,
        reported_total_max_w=49e12,
    ),
    "Moon": HeatFlowConstraint(
        body_name="Moon",
        evidence="regional inference",
        provenance="Siegler & Smrekar (2014), doi:10.1002/2013JE004453",
        caveat=(
            "The mantle flux and total mantle power are inferred from two Apollo "
            "regions after modeling crustal radioactivity; neither is a global "
            "surface heat-flow measurement."
        ),
        reported_total_min_w=2.8e11,
        reported_total_max_w=4.1e11,
    ),
    "Mars": HeatFlowConstraint(
        body_name="Mars",
        surface_flux_min_w_m2=0.014,
        surface_flux_max_w_m2=0.025,
        evidence="model inference",
        provenance="Parro et al. (2017), doi:10.1038/srep45629",
        caveat=(
            "InSight HP3 did not obtain a direct heat-flow measurement; this interval "
            "comes from thermal and geophysical modeling."
        ),
    ),
}


RADIAL_HEAT_FLOW_CONSTRAINTS = {
    "Earth core-mantle boundary": RadialHeatFlowConstraint(
        body_name="Earth",
        boundary_name="core-mantle boundary",
        boundary_radius_m=3.48e6,
        outward_power_min_w=13e12,
        outward_power_max_w=15e12,
        evidence="modern multidisciplinary synthesis",
        provenance=(
            "Frost et al. (2022), doi:10.1029/2021GC009764; historical broader "
            "estimates reviewed by Lay et al. (2008), doi:10.1038/ngeo.2007.44"
        ),
        caveat=(
            "The preferred 13--15 TW interval is an indirect, model-dependent "
            "geophysical inference rather than a direct calorimetric measurement. "
            "The 15 TW endpoint bounds steady DM heating only with the conventional "
            "present-day geophysical background held fixed."
        ),
    ),
}


def get_heat_flow_constraint(body_name):
    """Retrieve the heat-flow constraint for a named body."""

    return HEAT_FLOW_CONSTRAINTS[body_name]


def get_radial_heat_flow_constraint(name):
    """Retrieve a named internal-boundary heat-flow constraint."""

    return RADIAL_HEAT_FLOW_CONSTRAINTS[name]
