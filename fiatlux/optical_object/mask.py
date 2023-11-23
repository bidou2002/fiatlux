from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from fiatlux.optical_object import OpticalObject

import numpy as np
import astropy.units as u

from ..physical_object.spectrum_dev import Spectrum, SpectralFilter, Monochromatic
from ..physical_object.complex_amplitude import ComplexAmplitude


class Mask(OpticalObject):
    @abstractmethod
    def compute_complex_transparency(self, field_size: int) -> np.ndarray:
        pass

    @abstractmethod
    def compute_transformation(
        self, complex_amplitudes: list[ComplexAmplitude]
    ) -> list[ComplexAmplitude]:
        pass


class CircularPupil(Mask):
    def __init__(self, physical_diameter: float, resolution: float):
        self.physical_diameter = physical_diameter * u.meter
        self.resolution = resolution * u.pixel / u.meter
        self._surface = np.pi * (self.physical_diameter / 2) ** 2

        # self._sampling = self._field_size / (self.physical_diameter * self.resolution)

    def compute_complex_transparency(self, field_size) -> np.ndarray:
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
        # normalize the pupil so that the integral of squared values is 1
        complex_amplitude = complex_amplitude / np.sum(complex_amplitude) ** 0.5
        return complex_amplitude

    def compute_transformation(
        self, complex_amplitudes: list[ComplexAmplitude]
    ) -> list[ComplexAmplitude]:
        # get shape of the field
        shape = complex_amplitudes[0].field_size()

        # compute the complex transparency of the mask
        complex_transparency = self.compute_complex_transparency(shape)

        # define the output list
        res = []
        for complex_amplitude in complex_amplitudes:
            # compute the number of photons
            n_photon = complex_amplitude.flux * self._surface

            # check if the complex_amplitude is a 2D or 3D array
            if complex_amplitude.complex_amplitude.ndim == complex_transparency.ndim:
                masked_complex_amplitude = (
                    complex_amplitude.complex_amplitude * complex_transparency
                )
            else:
                # Reshape the array to the new shape
                complex_transparency_new_axis = complex_transparency[:, :, np.newaxis]

                masked_complex_amplitude = (
                    complex_amplitude.complex_amplitude * complex_transparency_new_axis
                )

            res += [
                ComplexAmplitude(
                    complex_amplitude=masked_complex_amplitude,
                    flux=n_photon,
                )
            ]

        return res


class Zernike(Mask):
    def __init__(self, physical_diameter: float, resolution: float):
        self.physical_diameter = physical_diameter * u.meter
        self.resolution = resolution * u.pixel / u.meter
        self._surface = np.pi * (self.physical_diameter / 2) ** 2
