from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np


@dataclass
class ComplexAmplitude:
    complex_amplitude: np.ndarray
    photon: float

    def compute_amplitude(self):
        return np.abs(self.complex_amplitude)

    def compute_phase(self):
        return np.angle(self.complex_amplitude)

    def compute_intensity(self):
        return np.abs(self.complex_amplitude) ** 2

    def field_size(self):
        return self.complex_amplitude.shape[0]
