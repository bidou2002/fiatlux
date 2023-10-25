from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np

import astropy.constants as C
import astropy.units as u


class SpectralBand(Enum):
    # spectral bands with caracteristics λeff, Δλ and f_0
    U = [0.360e-6, 0.070e-6, 2.0e12]
    B = [0.500e-6, 0.090e-6, 3.3e12]
    V = [0.550e-6, 0.090e-6, 3.3e12]
    R = [0.640e-6, 0.150e-6, 4.0e12]


class SpectralFilter(Enum):
    U = [360e-9, 550e-9]
    B = [360e-9, 550e-9]


@dataclass
class Spectrum:
    _photon_number: float = 0.0

    # def __post_init__(self):
    #     # need typing ???
    #     self._is_filtered: bool = False
    #     self._photon_number: float = 0.0
    @abstractmethod
    def compute_photon_flux(self, **kwargs):
        pass


@dataclass(kw_only=True)
class Monochromatic(Spectrum):
    """
    A monochromatic radiation at wavelength λ with irradiance I
    """

    wavelength: float
    irradiance: float

    def compute_photon_flux(self):
        print(self._photon_number)
        c = C.c  # speed of light
        h = C.h  # Plank's constant
        self._photon_number = self.irradiance / ((h * c) / self.wavelength)


@dataclass(kw_only=True)
class BlackBody(Spectrum):
    """
    A black body at temperature T.
    """

    temperature: float
    filter: SpectralFilter

    def compute_photon_flux(self, filter: SpectralFilter):
        # define constants
        c = C.c  # speed of light
        h = C.h  # Plank's constant
        k_B = C.k_B  # Boltzman's constant

        c1 = 2 * h * c**2
        c2 = h * c / k_B
        nu = 10
        I = 0
        for n in range(1, 10):
            I += (
                (c1 / k_B**4)
                * (np.exp(-n * k_B.value * nu) / (n**4))
                * ((n * k_B * nu) + 3 * (n * k_B * nu) ** 2 + 6 * (n * k_B * nu) + 6)
            )
            print(I)


@dataclass(kw_only=True)
class Photometric(Spectrum):
    """
    A spectrally filtered radiation with with magnitude m and spectral band B.
    """

    magnitude: float
    spectral_band: SpectralBand

    def compute_photon_flux(self):
        lambda_eff, delta_lambda, f_0 = self.spectral_band.value
        return (f_0 / 368) * 10 ** (-self.magnitude / 2.5)
