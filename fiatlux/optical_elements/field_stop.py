import torch

from fiatlux.field import Field
from fiatlux.grid import Grid
from fiatlux.spectrum import Spectrum
from fiatlux.optical_elements.mask import Mask


class ShanonFieldStop(Mask):

    def __init__(self):
        super().__init__()

    def _build_transmission(self, grid, spectrum):
        lambda_max = spectrum.wavelengths.max()

        x_max = grid.nx
        y_max = grid.ny

        x, y = torch.meshgrid(
            torch.linspace(-x_max, x_max, x_max),
            torch.linspace(-y_max, y_max, y_max),
        )

        self.transmission = torch.stack(
            [
                (
                    (x.abs() <= x_max * (wl / lambda_max))
                    & (y.abs() <= y_max * (wl / lambda_max))
                ).to(torch.complex64)
                for wl in spectrum.wavelengths
            ],
            dim=0,
        )  # (n_λ, nx, ny)

    def _build_opd(self, grid, spectrum):
        self.opd = torch.zeros((grid.ny, grid.nx))
