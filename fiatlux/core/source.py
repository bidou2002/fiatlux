from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod

import torch

from fiatlux.optics.elements.base import OpticalElement
from fiatlux.core.field import Field
from fiatlux.core.spectrum import Spectrum
from fiatlux.utils.resolution import Resolution
from fiatlux.core.grid import Grid


@dataclass
class Source(ABC):
    """
    Active element — generates a Field, does not receive one.
    Grid and spectrum are known at construction time.
    The field is generated once and cached.
    """

    def __init__(self, spectrum: Spectrum):
        self.spectrum = spectrum

    def _get_fluxes(self, grid: Grid) -> torch.Tensor:
        return self.spectrum.fluxes * grid.dx * grid.dy

    @abstractmethod
    def generate_field(self, grid: Grid) -> Field: ...

    @property
    def _symbol(self) -> str: ...


@dataclass
class IncidenceAngle:
    tip: float = 0.0  # rad
    tilt: float = 0.0  # rad


class PlaneWave(Source):

    def __init__(self, spectrum: Spectrum):
        super().__init__(spectrum)

    def generate_field(self, grid: Grid) -> Field:
        return Field(
            self._get_fluxes(grid=grid)[:, None, None] * torch.ones((grid.ny, grid.nx)),
            grid,
            self.spectrum,
        )

    @property
    def _symbol(self) -> str:
        return "|"


class GaussianSource(Source):

    def __init__(
        self,
        spectrum: Spectrum,
        waist: float,
    ):
        self.waist = waist
        super().__init__(spectrum)

    def generate_field(self, grid: Grid) -> Field:
        x, y = grid.meshgrid()
        n_wavelengths = len(self.spectrum.wavelengths)

        envelope = (
            (torch.exp(-(x**2 + y**2) / self.waist**2))
            .to(torch.complex64)
            .unsqueeze(0)
            .expand(n_wavelengths, -1, -1)
        )
        return Field(envelope, grid, self.spectrum)

    @property
    def _symbol(self) -> str:
        return "G"
