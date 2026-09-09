from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from fiatlux.optics.elements.base import OpticalElement
from fiatlux.core.grid import Grid
from fiatlux.core.field import Field
from fiatlux.core.spectrum import Spectrum
from fiatlux.optics.atmosphere import AtmosphereModel, NCPAModel

from fiatlux.config.registry import register_type

import os

from itertools import cycle

from astropy.io import fits


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
    recompute: bool = False

    @abstractmethod
    def _build_transmission(self) -> None: ...

    @abstractmethod
    def _build_opd(self) -> None: ...

    def build(self, spectrum: Spectrum) -> None:
        self._build_transmission()
        self._build_opd()
        expected_spatial_shape = (self.grid.ny, self.grid.nx)
        allowed_shapes = {
            expected_spatial_shape,
            (len(spectrum.wavelengths), *expected_spatial_shape),
        }
        for name in ("transmission", "opd"):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) not in allowed_shapes
            ):
                raise ValueError(
                    f"Mask {name} must have shape (ny, nx) or "
                    f"(n_wavelengths, ny, nx); got "
                    f"{None if not isinstance(value, torch.Tensor) else tuple(value.shape)}."
                )
        self.complex_transmission = self.transmission * torch.exp(
            1j * 2 * torch.pi * self.opd / spectrum.wavelengths[:, None, None]
        )

    def apply(self, field: Field) -> Field:
        if field.grid != self.grid:
            raise ValueError("Mask grid must match the incoming field grid.")
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
        self.opd = torch.zeros(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

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
        self.transmission = self._input_transmission.to(device=self.grid.device)

    def _build_opd(self) -> None:
        self.opd = torch.zeros(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

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
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

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
        self.opd = torch.zeros(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)


@register_type("Piston")
class Piston(Mask):

    def __init__(self, grid: Grid, piston: float):
        self.piston = piston
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        self.opd = self.piston * torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)


@register_type("Step")
class Step(Mask):

    def __init__(self, grid: Grid, piston: float):
        self.piston = piston
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        tmp = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)
        tmp[:, : self.grid.nx // 2] = 0
        self.opd = self.piston * (torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype) - tmp)


@register_type("TipTilt")
class TipTilt(Mask):

    def __init__(self, grid: Grid, tip: float, tilt: float):
        self.tip = tip
        self.tilt = tilt
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        x, y = self.grid.meshgrid()
        self.opd = self.tip * x + self.tilt * y


class ProuhetThueMorse(Mask):

    def __init__(self, grid: Grid):
        super().__init__(grid=grid)

    def parity_popcount(self, x):
        p = torch.zeros_like(x)

        while x.any():
            p ^= x & 1
            x >>= 1

        return p

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        y, x = torch.meshgrid(
            torch.arange(self.grid.ny, device=self.grid.device),
            torch.arange(self.grid.nx, device=self.grid.device),
            indexing="ij",
        )

        # XOR-based 2D PTM
        Z = x ^ y

        self.opd = 600e-9 * self.parity_popcount(Z).to(self.grid.dtype)


@register_type("ADC")
class ADC(Mask):

    def __init__(
        self, grid: Grid, amplitude: torch.Tensor, angle: float = 0.0
    ):
        self.amplitude = amplitude
        self.angle = angle
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        x, y = self.grid.meshgrid()

        amplitude = self.amplitude.to(device=self.grid.device, dtype=self.grid.dtype)
        angle = torch.as_tensor(self.angle, device=self.grid.device, dtype=self.grid.dtype)
        self.opd = amplitude[:, None, None] * (
            x * torch.cos(torch.deg2rad(angle))
            + y * torch.sin(torch.deg2rad(angle))
        )


class Atmosphere(Mask):
    def __init__(
        self,
        grid: Grid,
        atmosphere_model: AtmosphereModel,
        recompute: bool = True,
        remove_piston: bool = True,
    ):
        self.atmosphere_model = atmosphere_model
        self.remove_piston = remove_piston
        super().__init__(grid=grid, recompute=recompute)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(
            (self.grid.ny, self.grid.nx),
            device=self.grid.device,
            dtype=self.atmosphere_model.dtype,
        )

    def _build_opd(self) -> None:
        self.opd = self.atmosphere_model.sample_opd(
            remove_piston=self.remove_piston
        )


class NCPA(Mask):
    def __init__(
        self,
        grid: Grid,
        ncpa_model: NCPAModel,
        recompute: bool = True,
    ):
        self.ncpa_model = ncpa_model
        super().__init__(grid=grid, recompute=recompute)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(
            (self.grid.ny, self.grid.nx),
            device=self.grid.device,
            dtype=self.ncpa_model.dtype,
        )

    def _build_opd(self) -> None:
        self.opd = self.ncpa_model.sample_opd()


class Random(Mask):
    def __init__(
        self,
        grid: Grid,
        amplitude: float,
        recompute: bool = True,
    ):
        self.amplitude = amplitude
        super().__init__(grid=grid, recompute=recompute)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        self.opd = self.amplitude * torch.randn(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)


class HarmoniResiduals(Mask):
    def __init__(self, grid: Grid):
        self.grid = grid
        self.load_datacube(
            path="/Users/janinpop/Documents/code/use_fiatlux/data/harmoni_residuals"
        )

        r = 1009
        N = len(self.datacube)
        idx = torch.arange(N).repeat(r)
        idx = idx[torch.randperm(idx.numel())]
        self.iterator = iter(cycle(idx.tolist()))

        super().__init__(grid=grid, recompute=True)

    def load_datacube(self, path: str) -> None:
        datacube = []
        for file in os.listdir(path):
            if file.endswith(".fits"):
                hdul = fits.open(os.path.join(path, file))
                arr = hdul[0].data[1, ...]
                datacube.append(
                    torch.rot90(
                        torch.tensor(
                            arr.astype(arr.dtype.newbyteorder("="), copy=True)
                        ),
                        dims=[1, 2],
                    ).to(torch.float)
                    / 2
                )
        self.datacube = torch.cat(datacube, dim=0)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        self.opd = self.datacube[next(self.iterator), ...].to(device=self.grid.device, dtype=self.grid.dtype)
