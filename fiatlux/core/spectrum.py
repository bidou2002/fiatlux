from __future__ import annotations
from copy import copy
from dataclasses import dataclass

import torch


@dataclass
class Band:
    central_wavelength: float
    delta_wavelength: float
    f0: float

    def photon_flux(self, magnitude: float) -> float:
        """Photon flux for a given magnitude."""
        return (1 / 368) * self.f0 * 10 ** (-magnitude / 2.5)


class PhotometricBand:
    U = Band(0.360e-6, 0.070e-6, 2.0e12)
    V0 = Band(0.500e-6, 0.090e-6, 3.3e12)
    B = Band(0.440e-6, 0.100e-6, 5.4e12)
    V = Band(0.550e-6, 0.090e-6, 3.3e12)
    R = Band(0.640e-6, 0.150e-6, 4.0e12)
    I = Band(0.790e-6, 0.150e-6, 2.7e12)
    I1 = Band(0.700e-6, 0.033e-6, 2.7e12)
    I2 = Band(0.750e-6, 0.033e-6, 2.7e12)
    I3 = Band(0.800e-6, 0.033e-6, 2.7e12)
    I4 = Band(0.700e-6, 0.100e-6, 2.7e12)
    I5 = Band(0.850e-6, 0.100e-6, 2.7e12)
    I6 = Band(1.000e-6, 0.100e-6, 2.7e12)
    I7 = Band(0.850e-6, 0.300e-6, 2.7e12)
    R2 = Band(0.650e-6, 0.300e-6, 7.92e12)
    R3 = Band(0.600e-6, 0.300e-6, 7.92e12)
    R4 = Band(0.670e-6, 0.300e-6, 7.92e12)
    I8 = Band(0.750e-6, 0.100e-6, 2.7e12)
    I9 = Band(0.850e-6, 0.300e-6, 7.36e12)
    I10 = Band(0.900e-6, 0.300e-6, 2.7e12)
    J = Band(1.215e-6, 0.260e-6, 1.9e12)
    H = Band(1.654e-6, 0.290e-6, 1.1e12)
    Kp = Band(2.1245e-6, 0.351e-6, 6e11)
    Ks = Band(2.157e-6, 0.320e-6, 5.5e11)
    K = Band(2.179e-6, 0.410e-6, 7.0e11)
    K0 = Band(2.000e-6, 0.410e-6, 7.0e11)
    K1 = Band(2.400e-6, 0.410e-6, 7.0e11)
    L = Band(3.547e-6, 0.570e-6, 2.5e11)
    M = Band(4.769e-6, 0.450e-6, 8.4e10)
    Na = Band(0.589e-6, 0, 3.3e12)
    EOS = Band(1.064e-6, 0, 3.3e12)
    HCM = Band(1.200e-6, 0.02e-6, 1.1e12)
    HCM2 = Band(1.100e-6, 0.05e-6, 1.1e12)
    Mono = Band(1.200e-6, 0.5e-9, 1.1e12)

    @classmethod
    def __class_getitem__(cls, key):
        """
        Enables:
            PhotometricBand["R"]
        """

        try:
            return getattr(cls, key)

        except AttributeError:

            raise KeyError(f"Unknown photometric band '{key}'")


class Spectrum:
    def __init__(
        self,
        magnitude: float,
        band: Band,
        samples: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
            raise ValueError("samples must be a positive integer.")

        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Spectrum dtype must be torch.float32 or torch.float64.")
        self.magnitude = magnitude
        self._set_wavelengths(band, samples, device=torch.device(device), dtype=dtype)
        self._set_fluxes(
            self.magnitude,
            band,
            samples,
            device=torch.device(device),
            dtype=dtype,
        )

    def _set_wavelengths(
        self,
        band: Band,
        samples: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if samples == 1:
            self.wavelengths = torch.tensor(
                [band.central_wavelength], device=device, dtype=dtype
            )
            return
        self.wavelengths = band.central_wavelength + band.delta_wavelength * (
            torch.linspace(0, 1, samples, device=device, dtype=dtype) - 0.5
        )

    def _set_fluxes(
        self,
        magnitude: float,
        band: Band,
        samples: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.fluxes = torch.full(
            (samples,),
            band.photon_flux(magnitude) / samples,
            device=device,
            dtype=dtype,
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Spectrum:
        """Return a new spectrum with the requested device and real dtype."""
        if dtype is not None and dtype not in (torch.float32, torch.float64):
            raise ValueError("Spectrum dtype must be torch.float32 or torch.float64.")
        moved = copy(self)
        moved.wavelengths = self.wavelengths.to(device=device, dtype=dtype)
        moved.fluxes = self.fluxes.to(device=device, dtype=dtype)
        return moved

    @classmethod
    def from_sampling(
        cls,
        magnitude: float,
        band: Band,
        Nu: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Spectrum:
        """Build a spectrum sampled at approximately one focal-plane pixel.

        At least one channel is always returned. Monochromatic bands and bands
        narrower than the requested sampling are represented by one channel at
        the central wavelength.
        """
        if not isinstance(Nu, int) or isinstance(Nu, bool) or Nu < 1:
            raise ValueError("Nu must be a positive integer.")

        lambda_max = band.central_wavelength + band.delta_wavelength / 2
        d_lambda = 2 * lambda_max / Nu
        n_lambda = max(1, round(band.delta_wavelength / d_lambda))

        spectrum = cls(
            magnitude=magnitude,
            band=band,
            samples=n_lambda,
            device=device,
            dtype=dtype,
        )
        if n_lambda > 1:
            spectrum.wavelengths = torch.sort(
                lambda_max
                - torch.arange(n_lambda, device=device, dtype=dtype) * d_lambda
            ).values

        return spectrum
