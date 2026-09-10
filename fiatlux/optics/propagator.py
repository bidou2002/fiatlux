from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

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


@dataclass
@register_type("MFTPropagator")
class MFTPropagator(Propagator):
    """Matrix Fourier transform from a Field grid to ``output_grid``.

    Input amplitudes have shape ``(n_wavelengths, ny_in, nx_in)`` and output
    amplitudes have shape ``(n_wavelengths, output_grid.ny, output_grid.nx)``.
    Input and output grids must share device and real dtype. Wavelength
    channels are propagated independently and retain their original order.
    """
    focal_length: float
    output_grid: Grid

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
            mx = self._dft_matrix(x, u, wavelength, self.focal_length)
            my = self._dft_matrix(y, v, wavelength, self.focal_length)
            return (
                (my.T @ amplitude @ mx)
                * field.grid.dx
                * field.grid.dy
                / (wavelength * self.focal_length)
            )

        # field.complex_amplitude: (n_wavelengths, ny_in, nx_in)
        # amplitude: (n_wavelengths, ny_out, nx_out)
        amplitude = torch.vmap(propagate_one)(field.complex_amplitude, wavelengths)

        return Field(amplitude, self.output_grid, field.spectrum)

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
