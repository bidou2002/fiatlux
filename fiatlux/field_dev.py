from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np

from .source_dev import Source, PointSource, ExtendedSource
from .propagators import Propagator
from .complex_amplitude import ComplexAmplitude


@dataclass
class Field:
    field_size: int
    source_list: list[Source]
    scale: float = 1.0

    def __post_init__(self):
        print("bite")
        self._complex_amplitude_list: list[ComplexAmplitude] = []
        for source in self.source_list:
            self._complex_amplitude_list += [source.compute_field(self.field_size)]

    def __mul__(self, vararg):
        match vararg():
            case Propagator():
                self.propagate(vararg)
            case Mask():
                pass

    def propagate(self, propagator: Propagator):
        tmp_list = []
        for complex_amplitude in self._complex_amplitude_list:
            tmp_list += [
                propagator.compute_transformation(complex_amplitude=complex_amplitude)
            ]
        self._complex_amplitude_list = tmp_list

    def compute_intensity(self):
        intensity = np.zeros((self.field_size, self.field_size))
        for complex_amplitude in self._complex_amplitude_list:
            n_dim = complex_amplitude._complex_amplitude.ndim
            if n_dim > 2:
                print(type(complex_amplitude.compute_intensity()))
                intensity += np.sum(
                    complex_amplitude.compute_intensity(),
                    axis=tuple(i for i in range(2, n_dim)),
                )
            else:
                intensity += complex_amplitude.compute_intensity()
        return intensity
