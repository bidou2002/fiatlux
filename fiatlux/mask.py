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
    def compute_complex_transparency(self, size):
        pass


@dataclass
class Pupil(Mask):
    physical_diameter: float
    resolution: float

    def __post_init__(self):
        self.physical_diameter *= u.meters
        self.resolution *= u.pixel / u.meter

    def compute_complex_transparency(self, size):
        pass
