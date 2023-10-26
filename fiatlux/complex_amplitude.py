from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np


@dataclass
class ComplexAmplitude:
    _complex_amplitude: np.ndarray

    def compute_amplitude(self):
        return np.abs(self._complex_amplitude)
    
    def compute_phase(self):
        return np.angle(self._complex_amplitude)
    
    def compute_intensity(self):
        return np.abs(self._complex_amplitude)**2