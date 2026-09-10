"""Optical formation of Shack-Hartmann lenslet-array spots."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fiatlux.core.field import Field
from fiatlux.core.grid import Grid
from fiatlux.optics.elements.base import validate_field_grid


def _positive_finite(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number in metres.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


@dataclass(frozen=True)
class ShackHartmannImage:
    """Spectral lenslet focal fields and their wavelength-dependent sampling.

    ``complex_amplitude`` is shaped
    ``(n_wavelengths, n_lenslets_y, n_lenslets_x, spot_ny, spot_nx)`` and has
    units ``sqrt(photons / s / m²)``. Each wavelength has its own natural
    detector sampling, stored in ``pixel_scale_x`` and ``pixel_scale_y``.
    """

    complex_amplitude: torch.Tensor
    wavelengths: torch.Tensor
    pixel_scale_x: torch.Tensor
    pixel_scale_y: torch.Tensor
    valid_subapertures: torch.Tensor

    @property
    def intensity(self) -> torch.Tensor:
        """Spectral photon-rate density in photons / s / m²."""
        return self.complex_amplitude.abs().square()

    @property
    def pixel_flux(self) -> torch.Tensor:
        """Spectral photon rate in every detector pixel."""
        pixel_area = self.pixel_scale_x * self.pixel_scale_y
        return self.intensity * pixel_area[:, None, None, None, None]

    def mosaic(self, wavelength_index: int | None = None) -> torch.Tensor:
        """Tile spots into a detector image expressed as photons / s / pixel."""
        flux = self.pixel_flux
        if wavelength_index is None:
            flux = flux.sum(dim=0)
        else:
            flux = flux[wavelength_index]
        nly, nlx, spot_ny, spot_nx = flux.shape
        return flux.permute(0, 2, 1, 3).reshape(nly * spot_ny, nlx * spot_nx)


class ShackHartmannLensletArray:
    """Form independent focal spots behind a registered square lenslet array.

    The lenslet pitch must contain an integer number of input samples. The
    covered array is centered on the pupil grid, then shifted by the registered
    integer-pixel offsets. Each lenslet is propagated to its own focal plane by
    the Fraunhofer transform equivalent to a thin-lens phase followed by a
    distance equal to ``focal_length``.
    """

    def __init__(
        self,
        grid: Grid,
        *,
        pitch: float,
        focal_length: float,
        n_lenslets_x: int | None = None,
        n_lenslets_y: int | None = None,
        registration_x: float = 0.0,
        registration_y: float = 0.0,
        valid_subapertures: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(grid, Grid):
            raise TypeError("grid must be a fiatlux Grid.")
        self.grid = grid
        self.pitch = _positive_finite(pitch, "pitch")
        self.focal_length = _positive_finite(focal_length, "focal_length")
        self.samples_x = self._integer_samples(self.pitch, grid.dx, "pitch / dx")
        self.samples_y = self._integer_samples(self.pitch, grid.dy, "pitch / dy")
        maximum_x = grid.nx // self.samples_x
        maximum_y = grid.ny // self.samples_y
        self.n_lenslets_x = self._lenslet_count(n_lenslets_x, maximum_x, "x")
        self.n_lenslets_y = self._lenslet_count(n_lenslets_y, maximum_y, "y")
        offset_x = self._integer_offset(registration_x, grid.dx, "registration_x")
        offset_y = self._integer_offset(registration_y, grid.dy, "registration_y")
        covered_x = self.n_lenslets_x * self.samples_x
        covered_y = self.n_lenslets_y * self.samples_y
        self.start_x = (grid.nx - covered_x) // 2 + offset_x
        self.start_y = (grid.ny - covered_y) // 2 + offset_y
        if self.start_x < 0 or self.start_x + covered_x > grid.nx:
            raise ValueError("registered lenslet array exceeds the grid along x.")
        if self.start_y < 0 or self.start_y + covered_y > grid.ny:
            raise ValueError("registered lenslet array exceeds the grid along y.")

        expected = (self.n_lenslets_y, self.n_lenslets_x)
        if valid_subapertures is None:
            valid_subapertures = torch.ones(expected, dtype=torch.bool, device=grid.device)
        elif not isinstance(valid_subapertures, torch.Tensor):
            raise TypeError("valid_subapertures must be a torch.Tensor.")
        elif tuple(valid_subapertures.shape) != expected:
            raise ValueError(
                f"valid_subapertures must have shape {expected}, got "
                f"{tuple(valid_subapertures.shape)}."
            )
        self.valid_subapertures = valid_subapertures.to(
            device=grid.device, dtype=torch.bool
        )

    @staticmethod
    def _integer_samples(length: float, spacing: float, name: str) -> int:
        samples = round(length / spacing)
        if samples < 1 or not math.isclose(
            length, samples * spacing, rel_tol=1e-9, abs_tol=1e-15
        ):
            raise ValueError(f"{name} must be a positive integer.")
        return samples

    @staticmethod
    def _integer_offset(offset: float, spacing: float, name: str) -> int:
        if not isinstance(offset, (int, float)) or isinstance(offset, bool):
            raise TypeError(f"{name} must be a real number in metres.")
        pixels = round(offset / spacing)
        if not math.isfinite(offset) or not math.isclose(
            offset, pixels * spacing, rel_tol=1e-9, abs_tol=1e-15
        ):
            raise ValueError(f"{name} must be an integer multiple of grid spacing.")
        return pixels

    @staticmethod
    def _lenslet_count(value: int | None, maximum: int, axis: str) -> int:
        value = maximum if value is None else value
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
            raise ValueError(
                f"n_lenslets_{axis} must be an integer between 1 and {maximum}."
            )
        return value

    def lenslet_phase(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """Return the local thin-lens phase in radians for every wavelength."""
        if not isinstance(wavelengths, torch.Tensor) or wavelengths.ndim != 1:
            raise TypeError("wavelengths must be a one-dimensional torch.Tensor.")
        wavelengths = wavelengths.to(
            device=self.grid.device, dtype=self.grid.dtype
        )
        if torch.any(~torch.isfinite(wavelengths)) or torch.any(wavelengths <= 0):
            raise ValueError("wavelengths must be positive and finite metres.")
        x = (
            torch.arange(self.samples_x, device=self.grid.device, dtype=self.grid.dtype)
            - (self.samples_x - 1) / 2
        ) * self.grid.dx
        y = (
            torch.arange(self.samples_y, device=self.grid.device, dtype=self.grid.dtype)
            - (self.samples_y - 1) / 2
        ) * self.grid.dy
        radius_squared = y[:, None].square() + x[None, :].square()
        return -torch.pi * radius_squared[None] / (
            wavelengths[:, None, None] * self.focal_length
        )

    def propagate(self, field: Field) -> ShackHartmannImage:
        """Window the pupil and form all spectral lenslet spots in one batch."""
        validate_field_grid(field, self.grid, self.__class__.__name__)
        stop_x = self.start_x + self.n_lenslets_x * self.samples_x
        stop_y = self.start_y + self.n_lenslets_y * self.samples_y
        cropped = field.complex_amplitude[
            :, self.start_y:stop_y, self.start_x:stop_x
        ]
        n_wavelengths = cropped.shape[0]
        subapertures = cropped.reshape(
            n_wavelengths,
            self.n_lenslets_y,
            self.samples_y,
            self.n_lenslets_x,
            self.samples_x,
        ).permute(0, 1, 3, 2, 4)
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.ifftshift(subapertures, dim=(-2, -1)),
                dim=(-2, -1),
            ),
            dim=(-2, -1),
        )
        wavelengths = field.spectrum.wavelengths.to(
            device=self.grid.device, dtype=self.grid.dtype
        )
        scale = self.grid.dx * self.grid.dy / (
            wavelengths * self.focal_length
        )
        amplitude = spectrum * scale[:, None, None, None, None]
        amplitude = amplitude * self.valid_subapertures[
            None, :, :, None, None
        ]
        pixel_scale_x = wavelengths * self.focal_length / (
            self.samples_x * self.grid.dx
        )
        pixel_scale_y = wavelengths * self.focal_length / (
            self.samples_y * self.grid.dy
        )
        return ShackHartmannImage(
            complex_amplitude=amplitude,
            wavelengths=wavelengths,
            pixel_scale_x=pixel_scale_x,
            pixel_scale_y=pixel_scale_y,
            valid_subapertures=self.valid_subapertures,
        )
