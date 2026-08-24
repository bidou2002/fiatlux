import torch

from fiatlux.core.grid import Grid
from fiatlux.core.field import Field

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class AtmosphereModel:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.psd: torch.Tensor | None = None

        self.compute_psd()

    def compute_psd(self) -> None:
        raise NotImplementedError("compute_psd must be implemented in subclasses.")


class KolmogorovAtmosphereModel(AtmosphereModel):
    def compute_psd(self) -> None:
        fx = torch.fft.fftfreq(self.grid.nx, d=self.grid.dx)
        fy = torch.fft.fftfreq(self.grid.ny, d=self.grid.dy)
        fx, fy = torch.meshgrid(fx, fy, indexing="ij")
        f = (fx**2 + fy**2) ** 0.5

        d = 20
        fc = 1 / (2 * d)
        hp = (f > fc).float()

        r0 = 50e-2
        self.psd = 0.023 * r0 ** (-5 / 3) * f ** (-11 / 3)
        self.psd *= hp
        self.psd[0, 0] = 0
        self.psd = torch.fft.fftshift(self.psd)
        self.psd *= 500e-9 / (2 * torch.pi)

        plt.imshow(self.psd)


class NCPAModel(AtmosphereModel):
    def compute_psd(self) -> None:
        fx = torch.fft.fftfreq(self.grid.nx, d=self.grid.dx)
        fy = torch.fft.fftfreq(self.grid.ny, d=self.grid.dy)
        fx, fy = torch.meshgrid(fx, fy, indexing="ij")
        f = (fx**2 + fy**2) ** 0.5

        self.psd = f ** (-3)
        self.psd[0, 0] = 0
        self.psd = torch.fft.fftshift(self.psd)
        self.psd *= (5e-1) ** 2 / (self.psd.sum())

        plt.imshow(self.psd, norm=LogNorm())
