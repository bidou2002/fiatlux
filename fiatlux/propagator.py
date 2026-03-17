from abc import ABC, abstractmethod

from fiatlux.field import Field
from fiatlux.grid import Grid

import torch


class Propagator(ABC):
    @abstractmethod
    def propagate(self, field: Field, grid: Grid) -> list[Field, Grid]:
        raise NotImplementedError(
            "La méthode propagate doit être implémentée par les sous-classes."
        )


class MFTPropagator(Propagator):

    @staticmethod
    def _mft_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-2j * torch.pi * torch.outer(a, b))

    def propagate(self, field: Field, grid: Grid) -> Field:
        x = field.grid.x
        y = field.grid.y

        u = grid.x
        v = grid.y

        M1 = self._mft_matrix(v, y)  # (ny, ny)
        M2 = self._mft_matrix(x, u)  # (nx, nx)

        complex_amplitude = M2.T @ field.complex_amplitude @ M1.T

        return Field(complex_amplitude, grid)
