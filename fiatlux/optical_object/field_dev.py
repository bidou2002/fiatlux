from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np

from ..optical_object.source_dev import Source, PointSource, ExtendedSource
from ..physical_object.complex_amplitude import ComplexAmplitude

from .propagators import Propagator
from .mask import Mask, SpectralFilter
from .detector import Detector


class Field:
    def __init__(
        self,
        field_size: int,
        sources: list[Source],
        optical_path: list[OpticalObject] = [],
    ):
        self.field_size = field_size
        self.sources = sources
        self.optical_states: list[OpticalState] = []
        for optical_obj in optical_path:
            self.optical_states.append(
                OpticalState(transformation=optical_obj, complex_amplitudes=[])
            )

    # add vararg on optical path
    def __mul__(self, optical_obj: OpticalObject) -> Field:
        self.optical_states.append(
            OpticalState(transformation=optical_obj, complex_amplitudes=[])
        )
        return self

    def __pos__(self) -> None:
        self.resolve()

    def resolve(self) -> None:
        # used a the end of the building process to resolve and propagate along each module in the optical_path
        current_complex_amplitudes = []
        # Initialize field with complex amplitudes from sources
        for source in self.sources:
            current_complex_amplitudes += [source.compute_field(self.field_size)]

        # propagate the complex amplitudes through each module of the optical states
        for optical_state in self.optical_states:
            current_complex_amplitudes = (
                optical_state.transformation.compute_transformation(
                    current_complex_amplitudes
                )
            )

            optical_state.complex_amplitudes = current_complex_amplitudes

    def compute_intensity(self) -> np.ndarray:
        # intensity = np.zeros((self.field_size, self.field_size))
        # for complex_amplitude in self._complex_amplitude_list:
        #     n_dim = complex_amplitude._complex_amplitude.ndim
        #     if n_dim > 2:
        #         print(type(complex_amplitude.compute_intensity()))
        #         intensity += np.sum(
        #             complex_amplitude.compute_intensity(),
        #             axis=tuple(i for i in range(2, n_dim)),
        #         )
        #     else:
        #         intensity += complex_amplitude.compute_intensity()
        # return intensity
        raise NotImplementedError


@dataclass
class OpticalState:
    transformation: OpticalObject
    complex_amplitudes: list[ComplexAmplitude]


@dataclass
class OpticalObject:
    @abstractmethod
    def compute_transformation(
        self,
        complex_amplitudes: list[ComplexAmplitude],
    ) -> list[ComplexAmplitude]:
        pass
