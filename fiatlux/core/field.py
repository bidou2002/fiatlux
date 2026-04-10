# field.py
from __future__ import annotations
from dataclasses import dataclass
import torch

from fiatlux.core.grid import BaseGrid, Grid
from fiatlux.core.spectrum import Spectrum

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fiatlux.optics.elements.mask import Mask


@dataclass
class Field:
    complex_amplitude: torch.Tensor
    grid: BaseGrid
    spectrum: Spectrum

    def intensity(self) -> torch.Tensor:
        return self.complex_amplitude.abs() ** 2

    def phase(self) -> torch.Tensor:
        return self.complex_amplitude.angle()

    # def is_spatial(self) -> bool:
    #     return isinstance(self.grid, Grid)

    # def is_frequency(self) -> bool:
    #     return isinstance(self.grid, FrequencyGrid)

    def to(self, device: torch.device) -> Field:
        return Field(self.amplitude.to(device), self.grid.to(device), self.wavelength)

    def __rmul__(self, tensor: torch.Tensor) -> Field:
        return Field(tensor * self.complex_amplitude, self.grid, self.spectrum)

    def __sub__(self, tensor: torch.Tensor) -> Field:
        return Field(self.complex_amplitude - tensor, self.grid, self.spectrum)

    def __sub__(self, field: Field) -> Field:
        return Field(
            self.complex_amplitude - field.complex_amplitude, self.grid, self.spectrum
        )

    def __add__(self, field: Field) -> Field:
        return Field(
            self.complex_amplitude + field.complex_amplitude, self.grid, self.spectrum
        )

    def plot(self, wavelength_index: int = -1):
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        pcm = axs[0].imshow(
            self.intensity()[wavelength_index],
            extent=[
                self.grid.x.min(),
                self.grid.x.max(),
                self.grid.y.min(),
                self.grid.y.max(),
            ],
            norm=colors.PowerNorm(1),
        )
        axs[0].set_xlabel("x (m)")
        axs[0].set_ylabel("y (m)")
        axs[0].set_title(
            f"Intensity at λ={self.spectrum.wavelengths[wavelength_index]:.2e} m"
        )
        fig.colorbar(pcm, ax=axs[0], label="Intensity (in photon)")

        pcm = axs[1].imshow(
            self.phase()[wavelength_index],
            extent=[
                self.grid.x.min(),
                self.grid.x.max(),
                self.grid.y.min(),
                self.grid.y.max(),
            ],
            vmin=-torch.pi,
            vmax=torch.pi,
        )
        axs[1].set_xlabel("x (m)")
        axs[1].set_ylabel("y (m)")
        axs[1].set_title(
            f"Phase at λ={self.spectrum.wavelengths[wavelength_index]:.2e} m"
        )
        fig.colorbar(pcm, ax=axs[1], label="Phase (in rad)")
