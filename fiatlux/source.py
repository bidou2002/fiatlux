from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod

import torch

from fiatlux.optical_elements.base import OpticalElement
from fiatlux.field import Field
from fiatlux.spectrum import Spectrum
from fiatlux.utils.resolution import Resolution
from fiatlux.grid import Grid


class Source(ABC):
    """Source active : génère un champ, n'en reçoit pas."""

    def __init__(self, spectrum: Spectrum):
        self.spectrum = spectrum

    @abstractmethod
    def generate_field(self) -> Field:
        """Génère le champ initial — ne prend pas de Field en entrée."""
        raise NotImplementedError(
            "La méthode generate doit être implémentée par les sous-classes."
        )


@dataclass  # ← dataclass, pas une classe normale
class IncidenceAngle:
    tip: float = 0.0
    tilt: float = 0.0


class PlaneWave(Source):

    def __init__(
        self,
        spectrum: Spectrum,
        incidence_angle: IncidenceAngle = None,
    ):
        super().__init__(spectrum)
        self.incidence_angle = incidence_angle or IncidenceAngle()

    def generate_field(self, grid: Grid) -> Field:
        x_grid, y_grid = grid.meshgrid()

        complex_amplitude = []
        for wl in self.spectrum.wavelengths:
            t = torch.exp(
                1j * self.incidence_angle.tip * x_grid / wl
                + 1j * self.incidence_angle.tilt * y_grid / wl
            ).to(torch.complex64)
            complex_amplitude.append(t)

        complex_amplitude = torch.stack(complex_amplitude, dim=0)

        return Field(complex_amplitude, grid, self.spectrum)
