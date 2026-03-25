from abc import ABC, abstractmethod
from dataclasses import dataclass

from fiatlux.field import Field
from fiatlux.grid import Grid

import torch


class Propagator(ABC):
    @abstractmethod
    def propagate(self, field: Field, grid: Grid) -> list[Field, Grid]:
        raise NotImplementedError(
            "La méthode propagate doit être implémentée par les sous-classes."
        )


@dataclass
class MFTPropagator(Propagator):
    focal_length: float

    @staticmethod
    def _mft_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-2j * torch.pi * torch.outer(a, b))

    @staticmethod
    def _dft_matrix(
        x: torch.Tensor,
        x_out: torch.Tensor,
        wavelength: torch.Tensor,
        focal_length: float,
    ) -> torch.Tensor:
        """wavelength doit être un scalaire (vmap s'occupe de la dimension)"""
        u = x_out / (wavelength * focal_length)
        return torch.exp(-2j * torch.pi * torch.outer(x, u))  # (nx, mx) — 2D

    def propagate(self, field: Field, output_grid: Grid) -> Field:
        input_grid = field.grid
        x, y = input_grid.x, input_grid.y
        u, v = output_grid.x, output_grid.y

        wavelengths = field.spectrum.wavelengths  # (n_wavelengths,)

        def propagate_one(
            amplitude: torch.Tensor, wavelength: torch.Tensor
        ) -> torch.Tensor:
            M1 = self._dft_matrix(v, y, wavelength, self.focal_length)
            M2 = self._dft_matrix(x, u, wavelength, self.focal_length)
            return M2.T @ amplitude @ M1.T

        # field.amplitude : (n_wavelengths, nx, ny)
        amplitude = torch.vmap(propagate_one)(field.complex_amplitude, wavelengths)

        return Field(amplitude, output_grid, field.spectrum)
