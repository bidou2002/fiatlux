from dataclasses import dataclass

from astropy.units import Quantity
from astropy import units as u

import numpy as np


@dataclass
class Band:
    central_wavelength: Quantity
    delta_wavelength: Quantity
    f0: Quantity

    def __post_init__(self):
        self.central_wavelength = self.central_wavelength.to(u.m)
        self.delta_wavelength = self.delta_wavelength.to(u.m)
        self.f0 = self.f0.to(u.photon / (u.m**2 * u.s))

    def photon_flux(self, magnitude: float) -> Quantity:
        """Photon flux for a given magnitude."""
        return self.f0 * 10 ** (-magnitude / 2.5)


class PhotometricBand:
    U = Band(0.360e-6 * u.m, 0.070e-6 * u.m, 2.0e12 * u.photon / (u.m**2 * u.s))
    V0 = Band(0.500e-6 * u.m, 0.090e-6 * u.m, 3.3e12 * u.photon / (u.m**2 * u.s))
    B = Band(0.440e-6 * u.m, 0.100e-6 * u.m, 5.4e12 * u.photon / (u.m**2 * u.s))
    V = Band(0.550e-6 * u.m, 0.090e-6 * u.m, 3.3e12 * u.photon / (u.m**2 * u.s))
    R = Band(0.640e-6 * u.m, 0.150e-6 * u.m, 4.0e12 * u.photon / (u.m**2 * u.s))
    I = Band(0.790e-6 * u.m, 0.150e-6 * u.m, 2.7e12 * u.photon / (u.m**2 * u.s))
    I1 = Band(0.700e-6 * u.m, 0.033e-6 * u.m, 2.7e12 * u.photon / (u.m**2 * u.s))
    I2 = Band(0.750e-6 * u.m, 0.033e-6 * u.m, 2.7e12 * u.photon / (u.m**2 * u.s))
    I3 = Band(0.800e-6 * u.m, 0.033e-6 * u.m, 2.7e12 * u.photon / (u.m**2 * u.s))
    I4 = Band(0.700e-6 * u.m, 0.100e-6 * u.m, 2.7e12 * u.photon / (u.m**2 * u.s))
    I5 = Band(0.850e-6 * u.m, 0.100e-6 * u.m, 2.7e12 * u.photon / (u.m**2 * u.s))
    I6 = Band(1.000e-6 * u.m, 0.100e-6 * u.m, 2.7e12 * u.photon / (u.m**2 * u.s))
    I7 = Band(0.850e-6 * u.m, 0.300e-6 * u.m, 2.7e12 * u.photon / (u.m**2 * u.s))
    R2 = Band(0.650e-6 * u.m, 0.300e-6 * u.m, 7.92e12 * u.photon / (u.m**2 * u.s))
    R3 = Band(0.600e-6 * u.m, 0.300e-6 * u.m, 7.92e12 * u.photon / (u.m**2 * u.s))
    R4 = Band(0.670e-6 * u.m, 0.300e-6 * u.m, 7.92e12 * u.photon / (u.m**2 * u.s))
    I8 = Band(0.750e-6 * u.m, 0.100e-6 * u.m, 2.7e12 * u.photon / (u.m**2 * u.s))
    I9 = Band(0.850e-6 * u.m, 0.300e-6 * u.m, 7.36e12 * u.photon / (u.m**2 * u.s))
    I10 = Band(0.900e-6 * u.m, 0.300e-6 * u.m, 2.7e12 * u.photon / (u.m**2 * u.s))
    J = Band(1.215e-6 * u.m, 0.260e-6 * u.m, 1.9e12 * u.photon / (u.m**2 * u.s))
    H = Band(1.654e-6 * u.m, 0.290e-6 * u.m, 1.1e12 * u.photon / (u.m**2 * u.s))
    Kp = Band(2.1245e-6 * u.m, 0.351e-6 * u.m, 6e11 * u.photon / (u.m**2 * u.s))
    Ks = Band(2.157e-6 * u.m, 0.320e-6 * u.m, 5.5e11 * u.photon / (u.m**2 * u.s))
    K = Band(2.179e-6 * u.m, 0.410e-6 * u.m, 7.0e11 * u.photon / (u.m**2 * u.s))
    K0 = Band(2.000e-6 * u.m, 0.410e-6 * u.m, 7.0e11 * u.photon / (u.m**2 * u.s))
    K1 = Band(2.400e-6 * u.m, 0.410e-6 * u.m, 7.0e11 * u.photon / (u.m**2 * u.s))
    L = Band(3.547e-6 * u.m, 0.570e-6 * u.m, 2.5e11 * u.photon / (u.m**2 * u.s))
    M = Band(4.769e-6 * u.m, 0.450e-6 * u.m, 8.4e10 * u.photon / (u.m**2 * u.s))
    Na = Band(0.589e-6 * u.m, 0 * u.m, 3.3e12 * u.photon / (u.m**2 * u.s))
    EOS = Band(1.064e-6 * u.m, 0 * u.m, 3.3e12 * u.photon / (u.m**2 * u.s))


class Spectrum:
    def __init__(self, band: Band, samples: int):
        self.wavelengths: np.ndarray = None
        self.fluxes: np.ndarray = None

        self._set_wavelengths(band, samples)
        self._set_fluxes(band, samples)

    def _set_wavelengths(self, band: Band, samples: int):
        self.wavelengths = band.central_wavelength + band.delta_wavelength * (
            np.linspace(0, 1, samples) - 0.5
        )

    def _set_fluxes(self, band: Band, samples: int):
        self.fluxes = band.photon_flux(0) * np.ones(samples)
