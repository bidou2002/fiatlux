from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np

from ..physical_object.spectrum_dev import Spectrum, SpectralFilter, Monochromatic
from ..physical_object.complex_amplitude import ComplexAmplitude


@dataclass
class Source:
    spectrum: Spectrum = Monochromatic(wavelength=550e-9, irradiance=1)

    @abstractmethod
    def compute_field(self, field_size: int) -> ComplexAmplitude:
        pass


@dataclass(kw_only=True)
class PointSource(Source):
    incidence_angles: list[float]

    def compute_field(self, field_size: int) -> ComplexAmplitude:
        x_grid, y_grid = np.meshgrid(
            np.linspace(start=0, stop=field_size, num=field_size),
            np.linspace(start=0, stop=field_size, num=field_size),
        )
        complex_amplitude = np.zeros(
            shape=(field_size, field_size, 0),
            dtype=complex,
        )
        complex_amplitude = np.exp(
            1j * (self.incidence_angles[0] * x_grid + self.incidence_angles[1] * y_grid)
        )
        return ComplexAmplitude(
            complex_amplitude=complex_amplitude, photon=self.spectrum.photon_number
        )


@dataclass(kw_only=True)
class ExtendedSource(Source):
    incidence_angles_list: list[list[float]]

    def compute_field(self, field_size: int) -> ComplexAmplitude:
        x_grid, y_grid = np.meshgrid(
            np.linspace(start=0, stop=field_size, num=field_size),
            np.linspace(start=0, stop=field_size, num=field_size),
        )
        complex_amplitude = np.zeros(
            shape=(field_size, field_size, len(self.incidence_angles_list)),
            dtype=complex,
        )
        for i, incidence_angles in enumerate(self.incidence_angles_list):
            complex_amplitude[:, :, i] = np.exp(
                1j * (incidence_angles[0] * x_grid + incidence_angles[1] * y_grid)
            )

        return ComplexAmplitude(
            complex_amplitude=complex_amplitude, photon=self.spectrum.photon_number
        )
