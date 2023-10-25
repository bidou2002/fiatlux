from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np

from .spectrum_dev import Spectrum, SpectralFilter, Monochromatic
from .complex_amplitude import ComplexAmplitude


@dataclass
class Source:
    spectrum: Spectrum = Monochromatic(wavelength=550e-9, irradiance=1)

    @abstractmethod
    def compute_field(self, **kwargs) -> np.ndarray:
        pass


@dataclass(kw_only=True)
class PointSource:
    incidence_angles: [float, float]

    def compute_field(self, field_size) -> np.ndarray:
        print("cul")
        x_grid, y_grid = np.meshgrid(
            np.linspace(start=0, stop=field_size, num=field_size),
            np.linspace(start=0, stop=field_size, num=field_size),
        )
        complex_amplitude = np.exp(
            1j * (self.incidence_angles[0] * x_grid + self.incidence_angles[1] * y_grid)
        )
        return ComplexAmplitude(_complex_amplitude=complex_amplitude, _dimension=0)


@dataclass(kw_only=True)
class ExtendedSource:
    incidence_angles_list: list([float, float])

    def compute_field(self, field_size) -> np.ndarray:
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

        return ComplexAmplitude(_complex_amplitude=complex_amplitude, _dimension=1)
