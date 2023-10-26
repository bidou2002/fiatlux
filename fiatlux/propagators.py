from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np
import pyfftw
import os

from .source_dev import Source, PointSource, ExtendedSource
from .complex_amplitude import ComplexAmplitude


@dataclass
class Propagator:
    n_threads: int = os.cpu_count()

    @abstractmethod
    def compute_LCT(self):
        pass

    @abstractmethod
    def compute_transformation(self, complex_amplitude: ComplexAmplitude):
        pass


@dataclass(kw_only=True)
class FFT(Propagator):
    def compute_LCT(self):
        """linear canonical transformation (https://en.wikipedia.org/wiki/Linear_canonical_transformation)"""
        self._LCT = [[0, 1], [-1, 0]]

    def compute_transformation(self, complex_amplitude: ComplexAmplitude):
        shape = np.shape(complex_amplitude._complex_amplitude)
        a = pyfftw.empty_aligned(shape, dtype="complex128")
        b = pyfftw.empty_aligned(shape, dtype="complex128")
        fft = pyfftw.FFTW(a, b, axes=(0, 1), threads=self.n_threads)
        print(shape)
        return ComplexAmplitude(
            np.fft.fftshift(fft(np.fft.fftshift(complex_amplitude._complex_amplitude)))
        )


@dataclass(kw_only=True)
class DFT(Propagator):
    pass


@dataclass(kw_only=True)
class Fresnel(Propagator):
    pass
