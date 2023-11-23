from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np

import astropy.constants as C
import astropy.units as u

import astropy


@dataclass
class SpectralBandCharacteristics:
    lambda_eff: float
    delta_lambda: float
    f_0: astropy.units.quantity.Quantity


class SpectralBand(Enum):
    # spectral bands with caracteristics λeff, Δλ and f_0
    U = SpectralBandCharacteristics(
        lambda_eff=360e-9 * u.meter,
        delta_lambda=70e-9 * u.meter,
        f_0=2.0e12 * u.photon / u.meter**2 / u.second,
    )
    B = SpectralBandCharacteristics(
        lambda_eff=500e-9 * u.meter,
        delta_lambda=90e-9 * u.meter,
        f_0=3.3e12 * u.photon / u.meter**2 / u.second,
    )
    V = SpectralBandCharacteristics(
        lambda_eff=550e-9 * u.meter,
        delta_lambda=90e-9 * u.meter,
        f_0=3.3e12 * u.photon / u.meter**2 / u.second,
    )
    R = SpectralBandCharacteristics(
        lambda_eff=640e-9 * u.meter,
        delta_lambda=150e-9 * u.meter,
        f_0=4.0e12 * u.photon / u.meter**2 / u.second,
    )


class SpectralFilter(Enum):
    U = [360e-9, 550e-9]
    B = [360e-9, 550e-9]


class Spectrum:
    def __init__(self):
        self.flux: float = 0.0

    # def __post_init__(self):
    #     # need typing ???
    #     self._is_filtered: bool = False
    #     self._photon_number: float = 0.0
    @abstractmethod
    def compute_photon_flux(self, **kwargs):
        pass


@dataclass
class Monochromatic(Spectrum):
    """
    A monochromatic radiation at wavelength λ with irradiance I
    """

    wavelength: float
    irradiance: float

    def __post_init__(self):
        self.wavelength *= u.meter
        self.irradiance *= u.watt / u.meter**2
        self.flux = self.compute_photon_flux().decompose()

    def compute_photon_flux(self):
        c = C.c  # speed of light
        h = C.h  # Plank's constant
        photon_energy = ((h * c) / self.wavelength) * 1 / u.photon
        return self.irradiance / photon_energy


@dataclass
class BlackBody(Spectrum):
    """
    A black body at temperature T.
    """

    temperature: float
    filter: SpectralFilter

    def compute_photon_flux(self):
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


@dataclass
class Photometric(Spectrum):
    """
    A spectrally filtered radiation with with magnitude m and spectral band B.
    """

    magnitude: float
    spectral_band: SpectralBand

    def __post_init__(self):
        self.flux = self.compute_photon_flux()

    def compute_photon_flux(self):
        f_0 = self.spectral_band.value.f_0
        return (f_0 / 368) * 10 ** (-self.magnitude / 2.5)
