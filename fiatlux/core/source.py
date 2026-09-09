from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import torch

from fiatlux.core.field import Field
from fiatlux.core.spectrum import Spectrum
from fiatlux.core.grid import Grid


@dataclass
class Source(ABC):
    """
    Active element — generates a Field, does not receive one.
    Grid and spectrum are known at construction time.
    The field is generated once and cached.
    """

    def __init__(self, spectrum: Spectrum):
        self.spectrum = spectrum

    def _normalize_spatial_amplitude(
        self,
        spatial_amplitude: torch.Tensor,
        grid: Grid,
        spectrum: Spectrum,
    ) -> torch.Tensor:
        """Normalize a spatial amplitude to each channel's photon flux.

        The returned field satisfies
        ``sum(abs(E[channel])**2) * dx * dy == spectrum.fluxes[channel]``.
        """
        integrated_intensity = (
            spatial_amplitude.abs().square().sum() * grid.dx * grid.dy
        )
        if not torch.isfinite(integrated_intensity) or integrated_intensity <= 0:
            raise ValueError("The source spatial profile must have finite positive power.")
        if torch.any(spectrum.fluxes < 0) or not torch.isfinite(spectrum.fluxes).all():
            raise ValueError("Spectrum fluxes must be finite and non-negative.")

        channel_amplitudes = spectrum.fluxes.sqrt().to(
            device=grid.device,
            dtype=spatial_amplitude.real.dtype,
        )
        normalized_profile = spatial_amplitude / integrated_intensity.sqrt()
        return channel_amplitudes[:, None, None] * normalized_profile

    @abstractmethod
    def generate_field(self, grid: Grid) -> Field: ...

    @property
    def _symbol(self) -> str: ...


@dataclass
class IncidenceAngle:
    tip: float = 0.0  # rad
    tilt: float = 0.0  # rad


class PlaneWave(Source):

    def __init__(self, spectrum: Spectrum):
        super().__init__(spectrum)

    def generate_field(self, grid: Grid) -> Field:
        spectrum = self.spectrum.to(device=grid.device, dtype=grid.dtype)
        complex_dtype = (
            torch.complex64 if grid.dtype == torch.float32 else torch.complex128
        )
        spatial_amplitude = torch.ones(
            (grid.ny, grid.nx),
            dtype=complex_dtype,
            device=grid.device,
        )
        return Field(
            self._normalize_spatial_amplitude(spatial_amplitude, grid, spectrum),
            grid,
            spectrum,
        )

    @property
    def _symbol(self) -> str:
        return "|"


class GaussianSource(Source):

    def __init__(
        self,
        spectrum: Spectrum,
        waist: float,
    ):
        if not math.isfinite(waist) or waist <= 0:
            raise ValueError("waist must be a positive finite length.")
        self.waist = waist
        super().__init__(spectrum)

    def generate_field(self, grid: Grid) -> Field:
        x, y = grid.meshgrid()
        spectrum = self.spectrum.to(device=grid.device, dtype=grid.dtype)
        complex_dtype = (
            torch.complex64 if grid.dtype == torch.float32 else torch.complex128
        )
        spatial_amplitude = torch.exp(
            -(x**2 + y**2) / self.waist**2
        ).to(complex_dtype)
        amplitude = self._normalize_spatial_amplitude(
            spatial_amplitude,
            grid,
            spectrum,
        )
        return Field(amplitude, grid, spectrum)

    @property
    def _symbol(self) -> str:
        return "G"
