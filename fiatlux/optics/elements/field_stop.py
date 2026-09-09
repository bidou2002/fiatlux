import torch

from fiatlux.core.grid import Grid
from fiatlux.core.spectrum import Spectrum
from fiatlux.optics.elements.mask import Mask


class ShanonFieldStop(Mask):
    """Wavelength-scaled rectangular field stop on a ``(ny, nx)`` grid."""

    def __init__(self, grid: Grid):
        self._spectrum: Spectrum | None = None
        super().__init__(grid=grid)

    def build(self, spectrum: Spectrum) -> None:
        self._spectrum = spectrum
        try:
            super().build(spectrum)
        finally:
            self._spectrum = None

    def _build_transmission(self) -> None:
        if self._spectrum is None:
            raise RuntimeError("ShanonFieldStop.build() requires a Spectrum.")
        lambda_max = self._spectrum.wavelengths.max()
        x_limit = self.grid.nx / 2
        y_limit = self.grid.ny / 2

        y, x = torch.meshgrid(
            torch.arange(self.grid.ny, device=self.grid.device) - self.grid.ny // 2,
            torch.arange(self.grid.nx, device=self.grid.device) - self.grid.nx // 2,
            indexing="ij",
        )

        self.transmission = torch.stack(
            [
                (
                    (x.abs() <= x_limit * (wl / lambda_max))
                    & (y.abs() <= y_limit * (wl / lambda_max))
                ).to(torch.complex64)
                for wl in self._spectrum.wavelengths
            ],
            dim=0,
        )

    def _build_opd(self) -> None:
        self.opd = torch.zeros(
            self.grid.shape,
            device=self.grid.device,
        )
