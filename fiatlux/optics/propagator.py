from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import math

from fiatlux.core.field import Field
from fiatlux.core.grid import Grid
from fiatlux.config.registry import register_type
from fiatlux.optics.elements.base import validate_field_grid

import torch


class Propagator(ABC):
    """Transformation between sampled optical planes.

    A propagator may replace the spatial grid and spatial shape, but preserves
    wavelength ordering, spectral flux metadata, device, compatible precision,
    and the leading ``n_wavelengths`` dimension.
    """


class PropagationSamplingError(ValueError):
    """A propagation request violates a documented sampling condition."""


class PropagationRegime(str, Enum):
    """Physical approximation used independently of the numerical transform."""

    FRAUNHOFER = "fraunhofer"
    FRESNEL = "fresnel"


def _coerce_regime(value: PropagationRegime | str) -> PropagationRegime:
    try:
        return PropagationRegime(value)
    except ValueError as error:
        choices = ", ".join(regime.value for regime in PropagationRegime)
        raise ValueError(
            f"propagation must be one of {choices}; got {value!r}."
        ) from error


def _validate_scale(value: float | None, name: str, *, allow_zero: bool) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number in metres.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if not allow_zero and value == 0:
        raise ValueError(f"{name} must be non-zero.")
    return value


@dataclass(init=False)
@register_type("MFTPropagator")
class MFTPropagator(Propagator):
    """MFT propagation in the Fraunhofer or Fresnel regime.

    Input amplitudes have shape ``(n_wavelengths, ny_in, nx_in)`` and output
    amplitudes have shape ``(n_wavelengths, output_grid.ny, output_grid.nx)``.
    Input and output grids must share device and real dtype. Wavelength
    channels are propagated independently and retain their original order.
    Fraunhofer is the default and preserves the historical Fiatlux behavior.
    """
    focal_length: float | None
    output_grid: Grid
    propagation: PropagationRegime
    distance: float | None

    def __init__(
        self,
        focal_length: float | None = None,
        output_grid: Grid | None = None,
        *,
        propagation: PropagationRegime | str = PropagationRegime.FRAUNHOFER,
        distance: float | None = None,
    ) -> None:
        if not isinstance(output_grid, Grid):
            raise TypeError("output_grid must be a fiatlux Grid.")
        self.propagation = _coerce_regime(propagation)
        self.output_grid = output_grid
        self.distance = distance
        self.focal_length = focal_length

        if self.propagation is PropagationRegime.FRAUNHOFER:
            self.focal_length = _validate_scale(
                focal_length, "focal_length", allow_zero=False
            )
            if distance is not None:
                raise ValueError("distance is only valid for Fresnel propagation.")
        else:
            self.distance = _validate_scale(distance, "distance", allow_zero=True)

    @property
    def _scale(self) -> float:
        if self.propagation is PropagationRegime.FRAUNHOFER:
            assert self.focal_length is not None
            return self.focal_length
        assert self.distance is not None
        return self.distance

    @staticmethod
    def _mft_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-2j * torch.pi * torch.outer(a, b))

    @staticmethod
    def _dft_matrix(
        x: torch.Tensor,
        x_out: torch.Tensor,
        wavelength: torch.Tensor,
        focal_length: float,
    ) -> torch.Tensor:
        """wavelength doit être un scalaire (vmap s'occupe de la dimension)"""
        u = x_out / (wavelength * focal_length)
        return torch.exp(-2j * torch.pi * torch.outer(x, u))  # (nx, mx) — 2D

    def apply(self, field: Field) -> Field:
        if not isinstance(field, Field):
            raise TypeError(
                f"MFTPropagator expects a Field, got {type(field).__name__}."
            )
        if field.grid.device != self.output_grid.device:
            raise ValueError(
                "MFTPropagator output_grid device must match the incoming field "
                f"grid device; got {self.output_grid.device} and {field.grid.device}."
            )
        if field.grid.dtype != self.output_grid.dtype:
            raise ValueError(
                "MFTPropagator input and output grids must have the same dtype; "
                "output_grid dtype must match the incoming field "
                f"grid dtype; got {self.output_grid.dtype} and {field.grid.dtype}."
            )

        if self.propagation is PropagationRegime.FRESNEL and self._scale == 0:
            validate_field_grid(field, self.output_grid, "MFTPropagator")
            return Field(field.complex_amplitude, self.output_grid, field.spectrum)

        x, y = field.grid.x, field.grid.y
        u, v = self.output_grid.x, self.output_grid.y

        real_dtype = field.complex_amplitude.real.dtype
        x = x.to(dtype=real_dtype)
        y = y.to(dtype=real_dtype)
        u = u.to(dtype=real_dtype)
        v = v.to(dtype=real_dtype)
        wavelengths = field.spectrum.wavelengths.to(dtype=real_dtype)

        def propagate_one(
            amplitude: torch.Tensor, wavelength: torch.Tensor
        ) -> torch.Tensor:
            scale = self._scale
            if self.propagation is PropagationRegime.FRESNEL:
                coefficient = torch.pi / (wavelength * scale)
                input_chirp = torch.exp(
                    1j * coefficient * (y[:, None].square() + x[None, :].square())
                )
                amplitude = amplitude * input_chirp

            mx = self._dft_matrix(x, u, wavelength, scale)
            my = self._dft_matrix(y, v, wavelength, scale)
            propagated = (
                (my.T @ amplitude @ mx)
                * field.grid.dx
                * field.grid.dy
                / (wavelength * scale)
            )

            if self.propagation is PropagationRegime.FRESNEL:
                coefficient = torch.pi / (wavelength * scale)
                output_chirp = torch.exp(
                    1j * coefficient * (v[:, None].square() + u[None, :].square())
                )
                carrier = torch.exp(2j * torch.pi * scale / wavelength)
                propagated = carrier / 1j * output_chirp * propagated
            return propagated

        # field.complex_amplitude: (n_wavelengths, ny_in, nx_in)
        # amplitude: (n_wavelengths, ny_out, nx_out)
        amplitude = torch.vmap(propagate_one)(field.complex_amplitude, wavelengths)

        return Field(amplitude, self.output_grid, field.spectrum)

    @property
    def _symbol(self) -> str:
        return ">"


@dataclass
@register_type("FFTPropagator")
class FFTPropagator(Propagator):
    """FFT propagation on the wavelength-dependent natural output grid.

    The current ``Field`` model stores one spatial grid for all wavelength
    channels, so this propagator deliberately accepts monochromatic fields
    only. MFT propagation remains available for polychromatic fields on a
    common explicitly sampled output grid.
    """

    focal_length: float | None = None
    propagation: PropagationRegime | str = PropagationRegime.FRAUNHOFER
    distance: float | None = None

    def __post_init__(self) -> None:
        self.propagation = _coerce_regime(self.propagation)
        if self.propagation is PropagationRegime.FRAUNHOFER:
            self.focal_length = _validate_scale(
                self.focal_length, "focal_length", allow_zero=False
            )
            if self.distance is not None:
                raise ValueError("distance is only valid for Fresnel propagation.")
        else:
            self.distance = _validate_scale(
                self.distance, "distance", allow_zero=True
            )

    @property
    def _scale(self) -> float:
        if self.propagation is PropagationRegime.FRAUNHOFER:
            assert self.focal_length is not None
            return self.focal_length
        assert self.distance is not None
        return self.distance

    def output_grid_for(self, field: Field) -> Grid:
        """Return the natural physical output grid for a monochromatic field."""
        if not isinstance(field, Field):
            raise TypeError(
                f"FFTPropagator expects a Field, got {type(field).__name__}."
            )
        if field.complex_amplitude.shape[0] != 1:
            raise PropagationSamplingError(
                "FFTPropagator currently requires a monochromatic Field because "
                "its physical output sampling depends on wavelength; use "
                "MFTPropagator for a polychromatic common output grid."
            )
        if self.propagation is PropagationRegime.FRESNEL and self._scale == 0:
            return field.grid

        wavelength = float(field.spectrum.wavelengths[0])
        scale = abs(self._scale)
        return Grid(
            nx=field.grid.nx,
            ny=field.grid.ny,
            dx=wavelength * scale / (field.grid.nx * field.grid.dx),
            dy=wavelength * scale / (field.grid.ny * field.grid.dy),
            device=field.grid.device,
            dtype=field.grid.dtype,
        )

    def apply(self, field: Field) -> Field:
        output_grid = self.output_grid_for(field)
        if self.propagation is PropagationRegime.FRESNEL and self._scale == 0:
            return field

        amplitude = field.complex_amplitude[0]
        wavelength = field.spectrum.wavelengths[0].to(
            dtype=amplitude.real.dtype, device=amplitude.device
        )
        scale = self._scale

        if self.propagation is PropagationRegime.FRESNEL:
            x, y = field.grid.meshgrid()
            coefficient = torch.pi / (wavelength * scale)
            amplitude = amplitude * torch.exp(
                1j * coefficient * (x.square() + y.square())
            )

        if scale > 0:
            transformed = torch.fft.fftshift(
                torch.fft.fft2(torch.fft.ifftshift(amplitude), norm="backward")
            )
        else:
            transformed = torch.fft.fftshift(
                torch.fft.ifft2(torch.fft.ifftshift(amplitude), norm="backward")
            ) * (field.grid.nx * field.grid.ny)

        propagated = (
            transformed
            * field.grid.dx
            * field.grid.dy
            / (wavelength * scale)
        )

        if self.propagation is PropagationRegime.FRESNEL:
            x_out, y_out = output_grid.meshgrid()
            coefficient = torch.pi / (wavelength * scale)
            output_chirp = torch.exp(
                1j * coefficient * (x_out.square() + y_out.square())
            )
            carrier = torch.exp(2j * torch.pi * scale / wavelength)
            propagated = carrier / 1j * output_chirp * propagated

        return Field(propagated.unsqueeze(0), output_grid, field.spectrum)

    @property
    def _symbol(self) -> str:
        return ">"

@dataclass
@register_type("IdentityPropagator")
class IdentityPropagator(Propagator):
    grid: Grid

    def apply(self, field: Field) -> Field:
        validate_field_grid(field, self.grid, "IdentityPropagator")
        return field
