"""Optical formation of Shack-Hartmann lenslet-array spots."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from fiatlux.core.field import Field
from fiatlux.core.grid import Grid
from fiatlux.optics.elements.base import validate_field_grid
from fiatlux.utils.random import RandomGeneratorMixin


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
    illumination_fractions: torch.Tensor | None = None

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


@dataclass(frozen=True)
class ShackHartmannDetectorFrame:
    """Common-grid detector exposure in electrons per pixel."""

    electrons: torch.Tensor
    expected_electrons: torch.Tensor
    pixel_scale_x: torch.Tensor
    pixel_scale_y: torch.Tensor
    valid_subapertures: torch.Tensor
    saturated_subapertures: torch.Tensor
    illumination_fractions: torch.Tensor


class ShackHartmannDetector(RandomGeneratorMixin):
    """Expose lenslet spots on a common physical detector grid.

    Spectral intensity densities are bilinearly sampled at the configured
    detector-pixel centres before wavelength channels are summed. Noise is
    therefore applied once to the physically integrated broadband exposure.
    """

    def __init__(
        self,
        *,
        exposure_time: float,
        pixel_scale_x: float,
        pixel_scale_y: float | None = None,
        quantum_efficiency: float = 1.0,
        photon_noise: bool = True,
        dark_current: float = 0.0,
        read_noise: float = 0.0,
        full_well: float | None = None,
        minimum_electrons: float = 0.0,
        random_seed: int | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.exposure_time = _positive_finite(exposure_time, "exposure_time")
        self.pixel_scale_x = _positive_finite(pixel_scale_x, "pixel_scale_x")
        self.pixel_scale_y = _positive_finite(
            pixel_scale_x if pixel_scale_y is None else pixel_scale_y,
            "pixel_scale_y",
        )
        if not isinstance(quantum_efficiency, (int, float)) or isinstance(
            quantum_efficiency, bool
        ) or not 0 <= quantum_efficiency <= 1:
            raise ValueError("quantum_efficiency must be in [0, 1].")
        self.quantum_efficiency = float(quantum_efficiency)
        self.photon_noise = bool(photon_noise)
        self.dark_current = self._non_negative(dark_current, "dark_current")
        self.read_noise = self._non_negative(read_noise, "read_noise")
        self.minimum_electrons = self._non_negative(
            minimum_electrons, "minimum_electrons"
        )
        self.full_well = (
            None if full_well is None else _positive_finite(full_well, "full_well")
        )
        self._configure_generator(device, seed=random_seed, generator=generator)

    @staticmethod
    def _non_negative(value: float, name: str) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number.")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite.")
        return value

    @staticmethod
    def _resample_density(
        density: torch.Tensor,
        input_scale_x: torch.Tensor,
        input_scale_y: torch.Tensor,
        output_scale_x: float,
        output_scale_y: float,
    ) -> torch.Tensor:
        nw, nly, nlx, ny, nx = density.shape
        dtype = density.dtype
        device = density.device
        x = (torch.arange(nx, device=device, dtype=dtype) - nx // 2) * output_scale_x
        y = (torch.arange(ny, device=device, dtype=dtype) - ny // 2) * output_scale_y
        gx = (
            torch.zeros((nw, 1), device=device, dtype=dtype)
            if nx == 1
            else 2
            * (x[None] / input_scale_x[:, None] + nx // 2)
            / (nx - 1)
            - 1
        )
        gy = (
            torch.zeros((nw, 1), device=device, dtype=dtype)
            if ny == 1
            else 2
            * (y[None] / input_scale_y[:, None] + ny // 2)
            / (ny - 1)
            - 1
        )
        grid_x = gx[:, None, :].expand(nw, ny, nx)
        grid_y = gy[:, :, None].expand(nw, ny, nx)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        batch = density.reshape(nw, nly * nlx, ny, nx)
        sampled = functional.grid_sample(
            batch,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.reshape(nw, nly, nlx, ny, nx)

    def expose(self, image: ShackHartmannImage) -> ShackHartmannDetectorFrame:
        """Generate one broadband noisy exposure from ideal spectral spots."""
        if not isinstance(image, ShackHartmannImage):
            raise TypeError("image must be a ShackHartmannImage.")
        if image.complex_amplitude.device != self.generator.device:
            raise ValueError("image and detector random generator must share a device.")
        density = self._resample_density(
            image.intensity,
            image.pixel_scale_x,
            image.pixel_scale_y,
            self.pixel_scale_x,
            self.pixel_scale_y,
        )
        photon_rate = density.sum(dim=0) * self.pixel_scale_x * self.pixel_scale_y
        expected_signal = photon_rate * self.exposure_time * self.quantum_efficiency
        expected_dark = torch.full_like(
            expected_signal, self.dark_current * self.exposure_time
        )
        electrons = (
            torch.poisson(expected_signal, generator=self.generator)
            if self.photon_noise
            else expected_signal.clone()
        )
        if self.dark_current:
            electrons = electrons + torch.poisson(
                expected_dark, generator=self.generator
            )
        expected = expected_signal + expected_dark
        if self.read_noise:
            electrons = electrons + torch.normal(
                mean=torch.zeros_like(electrons),
                std=self.read_noise,
                generator=self.generator,
            )
        electrons = electrons.clamp_min(0)
        saturated_pixels = torch.zeros_like(electrons, dtype=torch.bool)
        if self.full_well is not None:
            saturated_pixels = electrons >= self.full_well
            electrons = electrons.clamp(max=self.full_well)
        saturated = saturated_pixels.any(dim=(-2, -1))
        enough_signal = expected.sum(dim=(-2, -1)) >= self.minimum_electrons
        valid = image.valid_subapertures & enough_signal & ~saturated
        fractions = image.illumination_fractions
        if fractions is None:
            fractions = torch.ones_like(valid, dtype=electrons.dtype)
        return ShackHartmannDetectorFrame(
            electrons=electrons,
            expected_electrons=expected,
            pixel_scale_x=torch.as_tensor(
                self.pixel_scale_x, dtype=electrons.dtype, device=electrons.device
            ),
            pixel_scale_y=torch.as_tensor(
                self.pixel_scale_y, dtype=electrons.dtype, device=electrons.device
            ),
            valid_subapertures=valid,
            saturated_subapertures=saturated,
            illumination_fractions=fractions,
        )


@dataclass(frozen=True)
class ShackHartmannMeasurement:
    """Calibrated Shack-Hartmann centroids and wavefront slopes.

    The last axis of ``centroids`` and ``slopes`` is ordered ``(x, y)``.
    Centroids are detector-plane positions in metres and slopes are angles in
    radians (dimensionless in SI). Invalid subapertures contain zeros and are
    identified by ``valid_subapertures``.
    """

    centroids: torch.Tensor
    reference_centroids: torch.Tensor
    slopes: torch.Tensor
    valid_subapertures: torch.Tensor
    saturated_subapertures: torch.Tensor
    weights: torch.Tensor

    @property
    def slope_vector(self) -> torch.Tensor:
        """Return valid slopes as ``[x row-major, y row-major]``."""
        valid = self.valid_subapertures
        return torch.cat((self.slopes[..., 0][valid], self.slopes[..., 1][valid]))


class ShackHartmannSlopeEstimator:
    """Extract flux-weighted centroids and calibrated slopes from lenslet spots.

    ``window_radius`` is an optional half-width in detector pixels, either one
    integer for both axes or ``(y, x)``. ``threshold`` is a fraction of each
    spectral spot's peak pixel flux. Centroids are first evaluated in physical
    detector coordinates for every wavelength and then combined by spectral
    flux; this is required because the natural pixel scale changes with
    wavelength.
    """

    def __init__(
        self,
        *,
        focal_length: float,
        window_radius: int | tuple[int, int] | None = None,
        threshold: float = 0.0,
        reference_centroids: torch.Tensor | None = None,
    ) -> None:
        self.focal_length = _positive_finite(focal_length, "focal_length")
        self.window_radius = self._window_radius(window_radius)
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(threshold)
            or not 0 <= threshold < 1
        ):
            raise ValueError("threshold must be finite and in [0, 1).")
        self.threshold = float(threshold)
        if reference_centroids is not None and not isinstance(
            reference_centroids, torch.Tensor
        ):
            raise TypeError("reference_centroids must be a torch.Tensor.")
        self.reference_centroids = reference_centroids

    @staticmethod
    def _window_radius(
        value: int | tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        if value is None:
            return None
        if isinstance(value, int) and not isinstance(value, bool):
            value = (value, value)
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or any(
                not isinstance(v, int) or isinstance(v, bool) or v < 0
                for v in value
            )
        ):
            raise ValueError("window_radius must be a non-negative integer or (y, x).")
        return value

    def measure(
        self,
        image: ShackHartmannImage | ShackHartmannDetectorFrame,
        *,
        reference_centroids: torch.Tensor | None = None,
    ) -> ShackHartmannMeasurement:
        """Measure centroids and reference-subtracted wavefront slopes."""
        if isinstance(image, ShackHartmannImage):
            flux = image.pixel_flux
            scales_x = image.pixel_scale_x
            scales_y = image.pixel_scale_y
            saturated = torch.zeros_like(image.valid_subapertures)
            fractions = image.illumination_fractions
        elif isinstance(image, ShackHartmannDetectorFrame):
            flux = image.electrons[None]
            scales_x = image.pixel_scale_x.reshape(1)
            scales_y = image.pixel_scale_y.reshape(1)
            saturated = image.saturated_subapertures
            fractions = image.illumination_fractions
        else:
            raise TypeError("image must be a ShackHartmannImage or detector frame.")
        _, nly, nlx, spot_ny, spot_nx = flux.shape
        work = flux

        if self.window_radius is not None:
            radius_y, radius_x = self.window_radius
            iy = torch.arange(spot_ny, device=flux.device)
            ix = torch.arange(spot_nx, device=flux.device)
            window = (
                (iy[:, None] - spot_ny // 2).abs() <= radius_y
            ) & ((ix[None, :] - spot_nx // 2).abs() <= radius_x)
            work = work * window[None, None, None]

        if self.threshold:
            peak = work.amax(dim=(-2, -1), keepdim=True)
            work = torch.where(work >= self.threshold * peak, work, 0)

        x = (
            torch.arange(spot_nx, device=flux.device, dtype=flux.dtype)
            - spot_nx // 2
        ) * scales_x[:, None]
        y = (
            torch.arange(spot_ny, device=flux.device, dtype=flux.dtype)
            - spot_ny // 2
        ) * scales_y[:, None]
        total_flux = work.sum(dim=(0, -2, -1))
        numerator_x = (work * x[:, None, None, None, :]).sum(dim=(0, -2, -1))
        numerator_y = (work * y[:, None, None, :, None]).sum(dim=(0, -2, -1))
        illuminated = total_flux > 0
        safe_flux = torch.where(illuminated, total_flux, torch.ones_like(total_flux))
        centroids = torch.stack(
            (numerator_x / safe_flux, numerator_y / safe_flux), dim=-1
        )

        reference = reference_centroids
        if reference is None:
            reference = self.reference_centroids
        expected = (nly, nlx, 2)
        if reference is None:
            reference = torch.zeros(expected, device=flux.device, dtype=flux.dtype)
        elif tuple(reference.shape) != expected:
            raise ValueError(
                f"reference_centroids must have shape {expected}, got "
                f"{tuple(reference.shape)}."
            )
        else:
            reference = reference.to(device=flux.device, dtype=flux.dtype)

        valid = image.valid_subapertures.to(device=flux.device) & illuminated
        slopes = (centroids - reference) / self.focal_length
        slopes = torch.where(valid[..., None], slopes, torch.zeros_like(slopes))
        centroids = torch.where(
            valid[..., None], centroids, torch.zeros_like(centroids)
        )
        return ShackHartmannMeasurement(
            centroids=centroids,
            reference_centroids=reference,
            slopes=slopes,
            valid_subapertures=valid,
            saturated_subapertures=saturated,
            weights=(
                torch.ones_like(valid, dtype=flux.dtype)
                if fractions is None
                else fractions.to(device=flux.device, dtype=flux.dtype)
            ),
        )


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
        pupil_transmission: torch.Tensor | None = None,
        minimum_illumination: float = 0.0,
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
        if not isinstance(minimum_illumination, (int, float)) or isinstance(
            minimum_illumination, bool
        ) or not 0 <= minimum_illumination <= 1:
            raise ValueError("minimum_illumination must be in [0, 1].")
        self.minimum_illumination = float(minimum_illumination)
        self.illumination_fractions = (
            None
            if pupil_transmission is None
            else self.compute_illumination_fractions(pupil_transmission)
        )
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
        if self.illumination_fractions is not None:
            self.valid_subapertures &= (
                self.illumination_fractions >= self.minimum_illumination
            )

    def compute_illumination_fractions(
        self, pupil_transmission: torch.Tensor
    ) -> torch.Tensor:
        """Return mean transmitted power in each registered subaperture."""
        if not isinstance(pupil_transmission, torch.Tensor):
            raise TypeError("pupil_transmission must be a torch.Tensor.")
        if tuple(pupil_transmission.shape) != (self.grid.ny, self.grid.nx):
            raise ValueError(
                "pupil_transmission must have shape "
                f"{(self.grid.ny, self.grid.nx)}."
            )
        transmission = pupil_transmission.to(device=self.grid.device)
        power = transmission.abs().square().to(dtype=self.grid.dtype)
        stop_x = self.start_x + self.n_lenslets_x * self.samples_x
        stop_y = self.start_y + self.n_lenslets_y * self.samples_y
        cropped = power[self.start_y:stop_y, self.start_x:stop_x]
        return cropped.reshape(
            self.n_lenslets_y,
            self.samples_y,
            self.n_lenslets_x,
            self.samples_x,
        ).permute(0, 2, 1, 3).mean(dim=(-2, -1))

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
            illumination_fractions=self.illumination_fractions,
        )
