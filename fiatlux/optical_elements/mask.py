from __future__ import annotations
from abc import ABC, abstractmethod

import torch

from fiatlux.optical_elements.base import OpticalElement
from fiatlux.grid import Grid
from fiatlux.field import Field
from fiatlux.spectrum import Spectrum


class Mask(OpticalElement, ABC):
    """
    Base mask class.
    Grid (and spectrum for chromatic masks) known at construction.
    Transmission built once at construction, cached.
    """

    def __init__(self):
        self.transmission: torch.Tensor | None = None
        self.opd: torch.Tensor | None = None
        self.complex_transmission: torch.Tensor | None = None
        self.recompute = False

    @abstractmethod
    def _build_transmission(self, grid: Grid, spectrum: Spectrum) -> None: ...

    @abstractmethod
    def _build_opd(self, grid: Grid, spectrum: Spectrum) -> None: ...

    def build(self, grid: Grid, spectrum: Spectrum) -> None:
        self._build_transmission(grid, spectrum)
        self._build_opd(grid, spectrum)
        self.complex_transmission = self.transmission * torch.exp(
            1j * 2 * torch.pi * self.opd / spectrum.wavelengths[:, None, None]
        )

    def apply(self, field: Field) -> Field:
        if self.complex_transmission is None or self.recompute:
            self.build(field.grid, field.spectrum)

        return Field(
            field.complex_amplitude * self.complex_transmission,
            field.grid,
            field.spectrum,
        )


class CircularAperture(Mask):
    def __init__(self, radius: float):
        self.radius = radius
        super().__init__()

    def _build_transmission(self, grid: Grid, spectrum: Spectrum) -> None:
        x, y = grid.meshgrid()
        r = torch.sqrt(x**2 + y**2)
        self.transmission = r <= self.radius

    def _build_opd(self, grid: Grid, spectrum: Spectrum) -> None:
        self.opd = torch.zeros((grid.ny, grid.nx))

    @property
    def _symbol(self) -> str:
        return "O"


class ZeldaMask(Mask):
    def __init__(self, radius: float, well_depth: float):
        self.radius = radius
        self.well_depth = well_depth
        super().__init__()

    def _build_transmission(self, grid: Grid, spectrum: Spectrum) -> None:
        self.transmission = torch.ones((grid.ny, grid.nx))

    def _build_opd(self, grid: Grid, spectrum: Spectrum) -> None:
        x, y = grid.meshgrid()
        r = torch.sqrt(x**2 + y**2)
        self.opd = self.well_depth * (r <= self.radius)


class ZeldaStop(Mask):

    def __init__(self, radius: float):
        self.radius = radius
        super().__init__()

    def _build_transmission(self, grid: Grid, spectrum: Spectrum) -> None:
        x, y = grid.meshgrid()
        r = torch.sqrt(x**2 + y**2)
        self.transmission = r <= self.radius

    def _build_opd(self, grid: Grid, spectrum: Spectrum) -> None:
        self.opd = torch.zeros((grid.ny, grid.nx))


class Piston(Mask):

    def __init__(self, piston: float):
        self.piston = piston
        super().__init__()

    def _build_transmission(self, grid: Grid, spectrum: Spectrum) -> None:
        self.transmission = torch.ones((grid.ny, grid.nx))

    def _build_opd(self, grid: Grid, spectrum: Spectrum) -> None:
        self.opd = self.piston * torch.ones((grid.ny, grid.nx))


class TipTilt(Mask):

    def __init__(self, tip: float, tilt: float):
        self.tip = tip
        self.tilt = tilt
        super().__init__()

    def _build_transmission(self, grid: Grid, spectrum: Spectrum) -> None:
        self.transmission = torch.ones((grid.ny, grid.nx))

    def _build_opd(self, grid: Grid, spectrum: Spectrum) -> None:
        x, y = grid.meshgrid()
        self.opd = self.tip * x + self.tilt * y


class ADC(Mask):

    def __init__(self, amplitude: float, angle: float = torch.tensor(0.0)):
        self.amplitude = amplitude
        self.angle = angle
        super().__init__()

    def _build_transmission(self, grid: Grid, spectrum: Spectrum) -> None:
        self.transmission = torch.ones((grid.ny, grid.nx))

    def _build_opd(self, grid: Grid, spectrum: Spectrum) -> None:
        x, y = grid.meshgrid()
        self.opd = (
            self.amplitude
            * (
                x * torch.cos(torch.tensor([self.angle]))
                + y * torch.sin(torch.tensor([self.angle]))
            )
            * self._opd_factor(spectrum.wavelengths)[:, None, None]
        )

    def _opd_factor(self, wavelengths: torch.Tensor) -> torch.Tensor:
        return (wavelengths / wavelengths.max()) ** 2
