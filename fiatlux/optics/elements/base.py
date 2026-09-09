from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np

from fiatlux.core.field import Field
from fiatlux.core.grid import Grid


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
