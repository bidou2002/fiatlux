from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np

from fiatlux.core.field import Field
from fiatlux.core.grid import Grid


def _grid_description(grid: Grid) -> str:
    return (
        f"shape={grid.shape}, spacing=({grid.dy}, {grid.dx}) m, "
        f"device={grid.device}, dtype={grid.dtype}"
    )


def validate_field_grid(field: Field, expected_grid: Grid, component: str) -> None:
    """Reject a field that is incompatible with a fixed-grid component."""
    if not isinstance(field, Field):
        raise TypeError(f"{component} expects a Field, got {type(field).__name__}.")
    if field.grid != expected_grid:
        raise ValueError(
            f"{component} grid must match the incoming field grid; expected "
            f"{_grid_description(expected_grid)}, got "
            f"{_grid_description(field.grid)}."
        )


@dataclass
class OpticalElement(ABC):
    """Pure same-plane field transformation.

    Unless a concrete element explicitly documents otherwise, ``apply``
    returns a new Field with the incoming grid, spectrum, shape, device, dtype,
    and physical amplitude units. An element may change amplitude or phase but
    does not change spatial sampling; sampling changes belong to Propagator.
    """

    @abstractmethod
    def apply(self, field: Field) -> Field:
        """Apply the same-plane optical transformation."""
        ...

    @abstractmethod
    def build(self, grid: Grid) -> None:
        """Build any cached tensor representation required by the element."""
        ...
