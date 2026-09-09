# field.py
from __future__ import annotations
from dataclasses import dataclass
from numbers import Number
import torch

from fiatlux.core.grid import BaseGrid
from fiatlux.core.spectrum import Spectrum

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fiatlux.optics.elements.mask import Mask


@dataclass
class Field:
    """Polychromatic scalar optical field sampled on a physical grid.

    ``complex_amplitude`` has shape ``(n_wavelengths, ny, nx)``. In a spatial
    plane its units are ``sqrt(photons / s / m²)``, so :meth:`intensity`
    returns a photon-rate density in ``photons / s / m²`` for each wavelength
    channel. Consequently, the photon rate in channel ``k`` is
    ``intensity()[k].sum() * grid.dx * grid.dy``.
    """

    complex_amplitude: torch.Tensor
    grid: BaseGrid
    spectrum: Spectrum

    def intensity(self) -> torch.Tensor:
        """Return spectral photon-rate density ``|E|²`` in photons / s / m²."""
        return self.complex_amplitude.abs() ** 2

    def phase(self) -> torch.Tensor:
        return self.complex_amplitude.angle()

    # def is_spatial(self) -> bool:
    #     return isinstance(self.grid, Grid)

    # def is_frequency(self) -> bool:
    #     return isinstance(self.grid, FrequencyGrid)

    def to(self, device: torch.device | str) -> Field:
        """Return a new field with all tensor state moved to ``device``.

        The complex-amplitude, wavelength and flux dtypes are preserved.
        Neither this field nor its associated grid and spectrum are mutated.
        """
        device = torch.device(device)
        return Field(
            self.complex_amplitude.to(device),
            self.grid.to(device),
            self.spectrum.to(device),
        )

    def _validate_compatible_field(self, other: Field) -> None:
        if self.grid != other.grid:
            raise ValueError("Field grids must be identical for arithmetic.")
        if self.complex_amplitude.shape != other.complex_amplitude.shape:
            raise ValueError(
                "Field amplitudes must have identical shapes for arithmetic."
            )
        for name in ("wavelengths", "fluxes"):
            left = getattr(self.spectrum, name)
            right = getattr(other.spectrum, name)
            if (
                left.shape != right.shape
                or left.device != right.device
                or left.dtype != right.dtype
                or not torch.equal(left, right)
            ):
                raise ValueError(
                    f"Field spectra must have identical {name} for arithmetic."
                )

    def _operand(self, other: Field | torch.Tensor | Number):
        if isinstance(other, Field):
            self._validate_compatible_field(other)
            return other.complex_amplitude
        if isinstance(other, (torch.Tensor, Number)):
            return other
        return NotImplemented

    def __add__(self, other: Field | torch.Tensor | Number) -> Field:
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented
        return Field(self.complex_amplitude + operand, self.grid, self.spectrum)

    def __radd__(self, other: torch.Tensor | Number) -> Field:
        return self + other

    def __sub__(self, other: Field | torch.Tensor | Number) -> Field:
        operand = self._operand(other)
        if operand is NotImplemented:
            return NotImplemented
        return Field(self.complex_amplitude - operand, self.grid, self.spectrum)

    def __rsub__(self, other: torch.Tensor | Number) -> Field:
        if not isinstance(other, (torch.Tensor, Number)):
            return NotImplemented
        return Field(other - self.complex_amplitude, self.grid, self.spectrum)

    def __mul__(self, other: torch.Tensor | Number) -> Field:
        if not isinstance(other, (torch.Tensor, Number)):
            return NotImplemented
        return Field(self.complex_amplitude * other, self.grid, self.spectrum)

    def __rmul__(self, other: torch.Tensor | Number) -> Field:
        return self * other

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
        fig.colorbar(pcm, ax=axs[0], label="Photon rate density (photons/s/m²)")

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
