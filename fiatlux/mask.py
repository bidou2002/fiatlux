from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np
import astropy.units as u

from .spectrum_dev import Spectrum, SpectralFilter, Monochromatic
from .complex_amplitude import ComplexAmplitude


@dataclass
class Mask:
    @abstractmethod
    def compute_complex_transparency(self, size: int):
        pass


@dataclass
class CircularPupil(Mask):
    physical_diameter: float
    resolution: float  # resolution in px/(λ/D)

    def __post_init__(self):
        self.physical_diameter *= u.meter
        self.resolution *= u.pixel / u.meter

        # self._sampling = self._field_size / (self.physical_diameter * self.resolution)

    def compute_complex_transparency(self, field_size):
        # compute the number of pixel in the pupil diameter from res = field_size / (physical_diameter * sampling)
        pupil_diameter_px = (field_size / (self.resolution)).value

        x_grid, y_grid = np.meshgrid(
            np.linspace(start=0, stop=field_size, num=field_size),
            np.linspace(start=0, stop=field_size, num=field_size),
        )
        complex_amplitude = np.zeros((field_size, field_size), dtype=complex)
        complex_amplitude[
            ((x_grid - field_size // 2) ** 2 + (y_grid - field_size // 2) ** 2)
            <= (pupil_diameter_px // 2) ** 2
        ] = 1
        complex_amplitude = complex_amplitude / np.sum(complex_amplitude) ** 0.5
        return ComplexAmplitude(_complex_amplitude=complex_amplitude)
