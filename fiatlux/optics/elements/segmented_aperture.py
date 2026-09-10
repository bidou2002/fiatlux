from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from fiatlux.config.registry import register_type
from fiatlux.core.grid import Grid
from fiatlux.optics.elements.mask import Mask


AxialCoordinate = tuple[int, int]


def _hexagonal_coordinates(rings: int) -> tuple[AxialCoordinate, ...]:
    """Return axial coordinates ordered by ring, then lexicographically."""
    coordinates = [
        (q, r)
        for q in range(-rings, rings + 1)
        for r in range(-rings, rings + 1)
        if max(abs(q), abs(r), abs(-q - r)) <= rings
    ]
    return tuple(
        sorted(
            coordinates,
            key=lambda coordinate: (
                max(abs(coordinate[0]), abs(coordinate[1]), abs(-sum(coordinate))),
                coordinate,
            ),
        )
    )


@register_type("HexagonalSegmentedAperture")
class HexagonalSegmentedAperture(Mask):
    """Regular, flat-top hexagonal segmented aperture.

    Parameters are expressed in metres. ``segment_circumradius`` is the
    centre-to-corner distance of one reflective segment. ``gap`` is the
    edge-to-edge separation between neighbouring segments. Segments are
    identified by deterministic axial ``(q, r)`` coordinates.
    """

    def __init__(
        self,
        grid: Grid,
        segment_circumradius: float,
        rings: int,
        *,
        gap: float = 0.0,
        inactive_segments: Iterable[AxialCoordinate] = (),
        rotation: float = 0.0,
        center: tuple[float, float] = (0.0, 0.0),
    ):
        if not math.isfinite(segment_circumradius) or segment_circumradius <= 0:
            raise ValueError("segment_circumradius must be positive and finite.")
        if not isinstance(rings, int) or isinstance(rings, bool) or rings < 0:
            raise ValueError("rings must be a non-negative integer.")
        if not math.isfinite(gap) or gap < 0:
            raise ValueError("gap must be finite and non-negative.")
        if not math.isfinite(rotation):
            raise ValueError("rotation must be finite.")
        if len(center) != 2 or not all(math.isfinite(value) for value in center):
            raise ValueError("center must contain two finite coordinates in metres.")

        all_coordinates = _hexagonal_coordinates(rings)
        inactive = frozenset(tuple(coordinate) for coordinate in inactive_segments)
        unknown = inactive.difference(all_coordinates)
        if unknown:
            raise ValueError(f"inactive_segments contains unknown coordinates: {sorted(unknown)}")

        self.segment_circumradius = float(segment_circumradius)
        self.rings = rings
        self.gap = float(gap)
        self.inactive_segments = inactive
        self.rotation = float(rotation)
        self.center = (float(center[0]), float(center[1]))
        self.segment_coordinates = tuple(
            coordinate for coordinate in all_coordinates if coordinate not in inactive
        )
        self.segment_index = torch.full(
            grid.shape, -1, device=grid.device, dtype=torch.int64
        )
        super().__init__(grid=grid)

    @property
    def number_of_segments(self) -> int:
        return len(self.segment_coordinates)

    @property
    def segment_centers(self) -> torch.Tensor:
        """Physical ``(x, y)`` centres, shaped ``(number_of_segments, 2)``."""
        effective_radius = self.segment_circumradius + self.gap / math.sqrt(3.0)
        centers = torch.tensor(
            [
                (
                    1.5 * effective_radius * q,
                    math.sqrt(3.0) * effective_radius * (r + 0.5 * q),
                )
                for q, r in self.segment_coordinates
            ],
            device=self.grid.device,
            dtype=self.grid.dtype,
        )
        angle = torch.as_tensor(
            math.radians(self.rotation), device=self.grid.device, dtype=self.grid.dtype
        )
        cosine, sine = torch.cos(angle), torch.sin(angle)
        x = cosine * centers[:, 0] - sine * centers[:, 1] + self.center[0]
        y = sine * centers[:, 0] + cosine * centers[:, 1] + self.center[1]
        return torch.stack((x, y), dim=-1)

    def _build_transmission(self) -> None:
        x, y = self.grid.meshgrid()
        segment_index = torch.full(
            self.grid.shape, -1, device=self.grid.device, dtype=torch.int64
        )
        radius = self.segment_circumradius
        sqrt_three = math.sqrt(3.0)

        for index, center in enumerate(self.segment_centers):
            local_x = x - center[0]
            local_y = y - center[1]
            inside = (
                (local_y.abs() <= 0.5 * sqrt_three * radius)
                & ((sqrt_three * local_x + local_y).abs() <= sqrt_three * radius)
                & ((sqrt_three * local_x - local_y).abs() <= sqrt_three * radius)
            )
            segment_index[inside] = index

        self.segment_index = segment_index
        self.transmission = (segment_index >= 0).to(dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        self.opd = torch.zeros(
            self.grid.shape, device=self.grid.device, dtype=self.grid.dtype
        )

    def index_of(self, coordinate: AxialCoordinate) -> int:
        """Return the stable integer label associated with an active segment."""
        try:
            return self.segment_coordinates.index(tuple(coordinate))
        except ValueError as error:
            raise KeyError(f"No active segment at axial coordinate {coordinate}.") from error

    def segment_mask(self, coordinate: AxialCoordinate) -> torch.Tensor:
        """Boolean pixel support of one active segment after :meth:`build`."""
        return self.segment_index == self.index_of(coordinate)


def _rotate_axial(coordinate: AxialCoordinate) -> AxialCoordinate:
    q, r = coordinate
    return -r, q + r


def _axial_orbit(coordinate: AxialCoordinate) -> tuple[AxialCoordinate, ...]:
    orbit = []
    current = coordinate
    for _ in range(6):
        orbit.append(current)
        current = _rotate_axial(current)
    return tuple(sorted(set(orbit)))


def _elt_coordinates(
    segment_circumradius: float,
    gap: float,
    inner_radius: float,
    segment_count: int,
) -> tuple[AxialCoordinate, ...]:
    """Select complete sixfold axial orbits nearest the ELT annulus."""
    if segment_count <= 0 or segment_count % 6:
        raise ValueError("segment_count must be a positive multiple of six.")
    effective_radius = segment_circumradius + gap / math.sqrt(3.0)
    candidates = _hexagonal_coordinates(24)
    orbits: dict[tuple[AxialCoordinate, ...], float] = {}
    for coordinate in candidates:
        if coordinate == (0, 0):
            continue
        orbit = _axial_orbit(coordinate)
        q, r = coordinate
        x = 1.5 * effective_radius * q
        y = math.sqrt(3.0) * effective_radius * (r + 0.5 * q)
        radius = math.hypot(x, y)
        if radius >= inner_radius - segment_circumradius:
            orbits[orbit] = radius
    selected = sorted(orbits, key=lambda orbit: (orbits[orbit], orbit))[
        : segment_count // 6
    ]
    return tuple(sorted(coordinate for orbit in selected for coordinate in orbit))


@register_type("ELTHarmoniPupil")
class ELTHarmoniPupil(HexagonalSegmentedAperture):
    """Self-contained analytical ELT pupil for HARMONI simulations.

    Lengths are metres in the entrance-pupil plane. Spider and M4 inter-petal
    widths are projected widths in that plane. Defaults describe a 37 m filled
    M1 annulus made from 798 reflective 1.45 m corner-to-corner segments.
    """

    def __init__(
        self,
        grid: Grid,
        *,
        outer_radius: float = 18.5,
        central_obscuration_radius: float = 5.5,
        segment_circumradius: float = 0.725,
        segment_gap: float = 0.004,
        segment_count: int = 798,
        spider_width: float = 0.5,
        spider_angles: tuple[float, ...] = (0.0, 60.0, 120.0),
        petal_gap: float = 0.0,
        petal_rotation: float = 0.0,
        inactive_segments: Iterable[AxialCoordinate] = (),
        rotation: float = 0.0,
        center: tuple[float, float] = (0.0, 0.0),
    ):
        for name, value in (
            ("outer_radius", outer_radius),
            ("central_obscuration_radius", central_obscuration_radius),
            ("spider_width", spider_width),
            ("petal_gap", petal_gap),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if central_obscuration_radius >= outer_radius:
            raise ValueError("central_obscuration_radius must be smaller than outer_radius.")
        if not spider_angles or not all(math.isfinite(angle) for angle in spider_angles):
            raise ValueError("spider_angles must contain finite angles in degrees.")
        if not math.isfinite(petal_rotation):
            raise ValueError("petal_rotation must be finite.")

        active = _elt_coordinates(
            segment_circumradius,
            segment_gap,
            central_obscuration_radius,
            segment_count,
        )
        all_coordinates = set(_hexagonal_coordinates(24))
        inactive = all_coordinates.difference(active).union(
            tuple(coordinate) for coordinate in inactive_segments
        )
        self.outer_radius = float(outer_radius)
        self.central_obscuration_radius = float(central_obscuration_radius)
        self.spider_width = float(spider_width)
        self.spider_angles = tuple(float(angle) for angle in spider_angles)
        self.petal_gap = float(petal_gap)
        self.petal_rotation = float(petal_rotation)
        self.petal_index = torch.full(
            grid.shape, -1, device=grid.device, dtype=torch.int64
        )
        super().__init__(
            grid,
            segment_circumradius,
            24,
            gap=segment_gap,
            inactive_segments=inactive,
            rotation=rotation,
            center=center,
        )

    def _build_transmission(self) -> None:
        super()._build_transmission()
        x, y = self.grid.meshgrid()
        x = x - self.center[0]
        y = y - self.center[1]
        angle = math.radians(self.rotation)
        local_x = math.cos(angle) * x + math.sin(angle) * y
        local_y = -math.sin(angle) * x + math.cos(angle) * y
        radius = torch.sqrt(local_x.square() + local_y.square())
        support = (radius >= self.central_obscuration_radius) & (
            radius <= self.outer_radius
        )

        if self.spider_width > 0:
            for spider_angle in self.spider_angles:
                theta = math.radians(spider_angle)
                perpendicular_distance = (
                    -math.sin(theta) * local_x + math.cos(theta) * local_y
                ).abs()
                support &= perpendicular_distance > 0.5 * self.spider_width

        polar_angle = torch.remainder(
            torch.atan2(local_y, local_x) - math.radians(self.petal_rotation),
            2 * math.pi,
        )
        petal_index = torch.floor(polar_angle / (math.pi / 3)).to(torch.int64)
        if self.petal_gap > 0:
            for boundary in range(3):
                theta = math.radians(self.petal_rotation) + boundary * math.pi / 3
                distance = (-math.sin(theta) * local_x + math.cos(theta) * local_y).abs()
                support &= distance > 0.5 * self.petal_gap

        self.transmission *= support.to(self.grid.dtype)
        self.segment_index = torch.where(
            support, self.segment_index, torch.full_like(self.segment_index, -1)
        )
        self.petal_index = torch.where(
            self.transmission > 0, petal_index, torch.full_like(petal_index, -1)
        )

    def petal_mask(self, index: int) -> torch.Tensor:
        """Boolean pixel support for one of the six M4 petals."""
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 6:
            raise ValueError("petal index must be an integer from 0 to 5.")
        return self.petal_index == index
