import torch

from fiatlux.core.grid import Grid
from fiatlux.core.field import Field


class AtmosphereModel:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.psd: torch.Tensor | None = None

        self.compute_psd()

    def compute_psd(self) -> None:
        x, y = self.grid.meshgrid()
        self.psd = (x**2 + y**2) ** -3
        self.psd[(x == 0) & (y == 0)] = 0
        self.psd *= (20e-9) ** 2 / self.psd.sum()
