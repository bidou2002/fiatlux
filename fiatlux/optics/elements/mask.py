from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from fiatlux.optics.elements.base import OpticalElement
from fiatlux.core.grid import Grid
from fiatlux.core.field import Field
from fiatlux.core.spectrum import Spectrum
from fiatlux.optics.atmosphere import AtmosphereModel

from fiatlux.config.registry import register_type


@dataclass
class Mask(OpticalElement, ABC):
    """
    Base mask class.
    Grid (and spectrum for chromatic masks) known at construction.
    Transmission built once at construction, cached.
    """

    grid: Grid
    transmission: torch.Tensor | None = None
    opd: torch.Tensor | None = None
    complex_transmission: torch.Tensor | None = None
    recompute = False

    @abstractmethod
    def _build_transmission(self) -> None: ...

    @abstractmethod
    def _build_opd(self) -> None: ...

    def build(self, spectrum: Spectrum) -> None:
        self._build_transmission()
        self._build_opd()
        self.complex_transmission = self.transmission * torch.exp(
            1j * 2 * torch.pi * self.opd / spectrum.wavelengths[:, None, None]
        )

    def apply(self, field: Field) -> Field:
        if self.complex_transmission is None or self.recompute:
            self.build(field.spectrum)

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
    def __init__(self, grid: Grid, radius: float):
        self.radius = radius
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        x, y = self.grid.meshgrid()
        r = torch.sqrt(x**2 + y**2)
        self.transmission = r <= self.radius

    def _build_opd(self) -> None:
        self.opd = torch.zeros((self.grid.ny, self.grid.nx))

    @property
    def _symbol(self) -> str:
        return "O"


@register_type("ArbitraryAperture")
class ArbitraryAperture(Mask):
    def __init__(self, grid: Grid, transmission: torch.Tensor):
        super().__init__(grid=grid)
        self._input_transmission = transmission

    def _build_transmission(self) -> None:
        # Validate size compatibility
        if self._input_transmission.shape != (self.grid.ny, self.grid.nx):
            raise ValueError(
                f"Transmission shape {self._input_transmission.shape} "
                f"does not match grid shape {(self.grid.ny, self.grid.nx)}"
            )

        # Use provided tensor
        self.transmission = self._input_transmission

    def _build_opd(self) -> None:
        self.opd = torch.zeros((self.grid.ny, self.grid.nx))

    @property
    def _symbol(self) -> str:
        return "O"


@register_type("ZeldaMask")
class ZeldaMask(Mask):
    def __init__(self, grid: Grid, radius: float, well_depth: float):
        self.radius = radius
        self.well_depth = well_depth
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones((self.grid.ny, self.grid.nx))

    def _build_opd(self) -> None:
        x, y = self.grid.meshgrid()
        r = torch.sqrt(x**2 + y**2)
        self.opd = self.well_depth * (r <= self.radius)


@register_type("ZeldaStop")
class ZeldaStop(Mask):

    def __init__(self, grid: Grid, radius: float):
        self.radius = radius
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        x, y = self.grid.meshgrid()
        r = torch.sqrt(x**2 + y**2)
        self.transmission = r <= self.radius

    def _build_opd(self) -> None:
        self.opd = torch.zeros((self.grid.ny, self.grid.nx))


@register_type("Piston")
class Piston(Mask):

    def __init__(self, grid: Grid, piston: float):
        self.piston = piston
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones((self.grid.ny, self.grid.nx))

    def _build_opd(self) -> None:
        self.opd = self.piston * torch.ones((self.grid.ny, self.grid.nx))


@register_type("Step")
class Step(Mask):

    def __init__(self, grid: Grid, piston: float):
        self.piston = piston
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones((self.grid.ny, self.grid.nx))

    def _build_opd(self) -> None:
        tmp = torch.ones((self.grid.ny, self.grid.nx))
        tmp[:, : self.grid.nx // 2] = 0
        self.opd = self.piston * (torch.ones((self.grid.ny, self.grid.nx)) - tmp)


@register_type("TipTilt")
class TipTilt(Mask):

    def __init__(self, grid: Grid, tip: float, tilt: float):
        self.tip = tip
        self.tilt = tilt
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones((self.grid.ny, self.grid.nx))

    def _build_opd(self) -> None:
        x, y = self.grid.meshgrid()
        self.opd = self.tip * x + self.tilt * y


@register_type("ADC")
class ADC(Mask):

    def __init__(
        self, grid: Grid, amplitude: torch.Tensor, angle: float = torch.tensor(0.0)
    ):
        self.amplitude = amplitude
        self.angle = angle
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones((self.grid.ny, self.grid.nx))

    def _build_opd(self) -> None:
        x, y = self.grid.meshgrid()

        self.opd = self.amplitude[:, None, None] * (
            x * torch.cos(torch.deg2rad(torch.as_tensor(self.angle)))
            + y * torch.sin(torch.deg2rad(torch.as_tensor(self.angle)))
        )


class Atmosphere(Mask):
    def __init__(self, grid: Grid, atmosphere_model: AtmosphereModel):
        self.atmosphere_model = atmosphere_model
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones((self.grid.ny, self.grid.nx))

    def _build_opd(self) -> None:
        cn = (
            torch.randn(*self.atmosphere_model.psd.shape)
            + 1j * torch.randn(*self.atmosphere_model.psd.shape)
        ) * torch.sqrt(self.atmosphere_model.psd)

        self.opd = torch.real(
            torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(cn)))
            * self.atmosphere_model.psd.numel()
        )
