from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from fiatlux.optical_object import OpticalObject

import numpy as np
import pyfftw
import os

from .source_dev import Source, PointSource, ExtendedSource
from ..physical_object.complex_amplitude import ComplexAmplitude


class Propagator(OpticalObject):
    def __init__(self):
        self.n_threads = os.cpu_count() if os.cpu_count() else 1

    @abstractmethod
    def compute_LCT(self):
        pass


class FFT(Propagator):
    def compute_LCT(self):
        """linear canonical transformation (https://en.wikipedia.org/wiki/Linear_canonical_transformation)"""
        self._LCT = [[0, 1], [-1, 0]]

    def compute_transformation(
        self,
        complex_amplitudes: list[ComplexAmplitude],
    ) -> list[ComplexAmplitude]:
        tmp = []
        for complex_amplitude in complex_amplitudes:
            shape = np.shape(complex_amplitude.complex_amplitude)
            a = pyfftw.empty_aligned(shape, dtype="complex128")
            b = pyfftw.empty_aligned(shape, dtype="complex128")
            fft = pyfftw.FFTW(a, b, axes=(0, 1), threads=self.n_threads)
            tmp += [
                ComplexAmplitude(
                    complex_amplitude=np.fft.fftshift(
                        fft(np.fft.fftshift(complex_amplitude.complex_amplitude))
                    ),
                    photon=complex_amplitude.photon,
                )
            ]
        return tmp


class DFT(Propagator):
    pass


class Fraunhofer(Propagator):
    pass


class Fresnel(Propagator):
    pass
