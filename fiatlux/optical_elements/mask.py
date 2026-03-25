from __future__ import annotations
from abc import ABC, abstractmethod

import torch

from fiatlux.optical_elements.base import OpticalElement
from fiatlux.grid import Grid
from fiatlux.field import Field
from fiatlux.spectrum import Spectrum


class Mask(OpticalElement, ABC):

    complex_transmission: torch.Tensor = None

    @abstractmethod
    def build(self, grid: Grid, spectrum: Spectrum | None = None) -> None:
        raise NotImplementedError(
            "La méthode build doit être implémentée par les sous-classes."
        )

    def apply(self, field: Field) -> Field:
        raise NotImplementedError(
            "La méthode build doit être implémentée par les sous-classes."
        )


class CircularAperture(Mask):
    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius

    def build(self, grid: Grid, spectrum=None):
        x_grid, y_grid = grid.meshgrid()
        r_grid = torch.sqrt(x_grid**2 + y_grid**2)
        self.complex_transmission = (r_grid <= self.radius).to(torch.complex64)
        self.complex_transmission /= self.complex_transmission.sum()

    def apply(self, field: Field) -> Field:
        return Field(
            field.complex_amplitude * self.complex_transmission,
            field.grid,
            field.spectrum,
        )


class ZeldaMask(Mask):
    def __init__(self, radius: float, well_depth: float):
        super().__init__()
        self.radius = radius
        self.well_depth = well_depth

    def build(self, grid: Grid, spectrum: Spectrum):
        x_grid, y_grid = grid.meshgrid()
        r_grid = torch.sqrt(x_grid**2 + y_grid**2)
        aperture = r_grid <= self.radius

        transmissions = []
        for wl in spectrum.wavelengths:
            t = torch.exp(1j * 2 * torch.pi * (self.well_depth / wl) * aperture).to(
                torch.complex64
            )
            transmissions.append(t)

        self.complex_transmission = torch.stack(transmissions, dim=0)

    def apply(self, field: Field) -> Field:

        wavelengths = field.spectrum.wavelengths

        def apply_one(
            amplitude: torch.Tensor, wavelength: torch.Tensor
        ) -> torch.Tensor:
            return field.complex_amplitude * self.complex_transmission

        # field.amplitude : (n_wavelengths, nx, ny)
        amplitude = torch.vmap(apply_one)(field.complex_amplitude, wavelengths)
        return Field(
            field.complex_amplitude * self.complex_transmission,
            field.grid,
            field.spectrum,
        )


class ZeldaStop(Mask):

    def __init__(self, radius: float, well_depth: float):
        super().__init__()
        self.radius = radius
        self.well_depth = well_depth

    def build(self, grid: Grid) -> None:

        x_grid, y_grid = grid.meshgrid()
        r_grid = torch.sqrt(x_grid**2 + y_grid**2)
        aperture = r_grid <= self.radius

        self.complex_transmission = aperture.to(torch.complex64)

    def apply(self, field: Field) -> Field:

        return Field(
            field.complex_amplitude * self.complex_transmission,
            field.grid,
            field.spectrum,
        )


class FieldStop(Mask):

    def __init__(self):
        super().__init__()

    def build(self, grid: Grid, spectrum: Spectrum) -> None:
        lambda_max = spectrum.wavelengths.max()

        x_max = grid.nx
        y_max = grid.ny

        x, y = torch.meshgrid(
            torch.linspace(-x_max, x_max, x_max),
            torch.linspace(-y_max, y_max, y_max),
        )

        self.complex_transmission = torch.stack(
            [
                (
                    (x.abs() <= x_max * (wl / lambda_max))
                    & (y.abs() <= y_max * (wl / lambda_max))
                ).to(torch.complex64)
                for wl in spectrum.wavelengths
            ],
            dim=0,
        )  # (n_λ, nx, ny)

    def apply(self, field: Field) -> Field:

        return Field(
            field.complex_amplitude * self.complex_transmission,
            field.grid,
            field.spectrum,
        )
