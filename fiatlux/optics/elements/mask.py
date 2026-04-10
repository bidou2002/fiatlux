from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from fiatlux.optics.elements.base import OpticalElement
from fiatlux.core.grid import Grid
from fiatlux.core.field import Field
from fiatlux.core.spectrum import Spectrum

from fiatlux.config.registry import register_type


@dataclass
class Mask(OpticalElement, ABC):
    """
    Base mask class.
    Grid (and spectrum for chromatic masks) known at construction.
    Transmission built once at construction, cached.
    """

    transmission: torch.Tensor | None = None
    opd: torch.Tensor | None = None
    complex_transmission: torch.Tensor | None = None
    recompute = False

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

    def __repr__(self):
        return "".join(
            (
                self.__class__.__name__,
                "(" f"transmission={self.transmission.type()}, ",
                f"opd={self.opd.type()}, ",
                f"complex_transmission={self.complex_transmission.type()}, ",
                f"recompute={self.recompute}",
                ")",
            )
        )


@register_type("CircularAperture")
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


@register_type("ArbitraryAperture")
class ArbitraryAperture(Mask):
    def __init__(self, transmission: torch.Tensor):
        super().__init__()
        self._input_transmission = transmission

    def _build_transmission(self, grid: Grid, spectrum: Spectrum) -> None:
        # Validate size compatibility
        if self._input_transmission.shape != (grid.ny, grid.nx):
            raise ValueError(
                f"Transmission shape {self._input_transmission.shape} "
                f"does not match grid shape {(grid.ny, grid.nx)}"
            )

        # Use provided tensor
        self.transmission = self._input_transmission

    def _build_opd(self, grid: Grid, spectrum: Spectrum) -> None:
        self.opd = torch.zeros((grid.ny, grid.nx))

    @property
    def _symbol(self) -> str:
        return "O"


@register_type("ZeldaMask")
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


@register_type("ZeldaStop")
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


@register_type("Piston")
class Piston(Mask):

    def __init__(self, piston: float):
        self.piston = piston
        super().__init__()

    def _build_transmission(self, grid: Grid, spectrum: Spectrum) -> None:
        self.transmission = torch.ones((grid.ny, grid.nx))

    def _build_opd(self, grid: Grid, spectrum: Spectrum) -> None:
        self.opd = self.piston * torch.ones((grid.ny, grid.nx))


@register_type("TipTilt")
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


@register_type("ADC")
class ADC(Mask):

    def __init__(self, amplitude: torch.Tensor, angle: float = torch.tensor(0.0)):
        self.amplitude = amplitude
        self.angle = angle
        super().__init__()

    def _build_transmission(self, grid: Grid, spectrum: Spectrum) -> None:
        self.transmission = torch.ones((grid.ny, grid.nx))

    def _build_opd(self, grid: Grid, spectrum: Spectrum) -> None:
        x, y = grid.meshgrid()

        self.opd = self.amplitude[:, None, None] * (
            x * torch.cos(torch.deg2rad(torch.as_tensor(self.angle)))
            + y * torch.sin(torch.deg2rad(torch.as_tensor(self.angle)))
        )
