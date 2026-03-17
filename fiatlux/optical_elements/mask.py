from abc import ABC, abstractmethod

import torch

from fiatlux.optical_elements.base import OpticalElement
from fiatlux.grid import Grid
from fiatlux.field import Field


class Mask(OpticalElement):
    def __init__(self, grid: Grid):
        self.grid = grid
        self.complex_transmission = None

    @abstractmethod
    def build(self, grid: Grid):
        raise NotImplementedError(
            "La méthode build doit être implémentée par les sous-classes."
        )

    def apply(self, field: Field) -> Field:
        raise NotImplementedError(
            "La méthode build doit être implémentée par les sous-classes."
        )


class CircularAperture(Mask):
    def __init__(self, radius: float, grid: Grid):
        super().__init__(grid)
        self.radius = radius

    def build(self):
        x_grid, y_grid = self.grid.meshgrid()
        r_grid = torch.sqrt(x_grid**2 + y_grid**2)
        self.complex_transmission = (r_grid <= self.radius).to(torch.complex64)

    def apply(self, field: Field) -> Field:
        return Field(field.complex_amplitude * self.complex_transmission, self.grid)
