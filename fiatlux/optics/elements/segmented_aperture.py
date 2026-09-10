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
    ):
        if not math.isfinite(segment_circumradius) or segment_circumradius <= 0:
            raise ValueError("segment_circumradius must be positive and finite.")
        if not isinstance(rings, int) or isinstance(rings, bool) or rings < 0:
            raise ValueError("rings must be a non-negative integer.")
        if not math.isfinite(gap) or gap < 0:
            raise ValueError("gap must be finite and non-negative.")
        if not math.isfinite(rotation):
            raise ValueError("rotation must be finite.")

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
        x = cosine * centers[:, 0] - sine * centers[:, 1]
        y = sine * centers[:, 0] + cosine * centers[:, 1]
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
