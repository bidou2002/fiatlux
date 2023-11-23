from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from fiatlux.optical_object import OpticalObject

import numpy as np
import astropy.units as u

from ..physical_object.spectrum_dev import Spectrum, SpectralFilter, Monochromatic
from ..physical_object.complex_amplitude import ComplexAmplitude

import matplotlib.pyplot as plt


class Detector(OpticalObject):
    def __init__(
        self,
        quantum_efficiency: float = 1,
        photon_noise: bool = False,
        readout_noise_variance: float = 0,
        dark_current: float = 0,
        exposure_time: float = 1,
        offset: int = 0,
        bitdepth: int = 16,
        sensitivity: float = 1,
        random_state_generator: np.random.RandomState = np.random.RandomState(
            seed=31415
        ),
        name: str = "default_camera",
    ) -> None:
        self.quantum_efficiency = quantum_efficiency * u.electron / u.photon
        self.photon_noise = photon_noise
        self.readout_noise_variance = readout_noise_variance * u.electron
        self.dark_current = dark_current * u.electron / u.second
        self.exposure_time = exposure_time * u.second
        self.offset = offset * u.adu
        self.bitdepth = bitdepth
        self.sensitivity = sensitivity * u.adu / u.electron
        self.random_state_generator = random_state_generator

    def compute_transformation(
        self,
        complex_amplitudes_list: list[ComplexAmplitude],
    ) -> list[ComplexAmplitude]:
        """Compute the transformation, i.e calculate the intensity, store it in self.intensity and return unmodified complex_amplitude_list"""
        self.intensity = self.compute_intensity(complex_amplitudes_list)
        return complex_amplitudes_list

    def compute_intensity(
        self, complex_amplitudes: list[ComplexAmplitude]
    ) -> np.ndarray:
        """Compute the intensity from a list of ComplexAmplitudes"""
        # get array shape
        shape = complex_amplitudes[0].field_size()
        # get unit
        unit = complex_amplitudes[0].flux.unit

        # initialize intensity array
        intensity = u.quantity.Quantity(
            np.zeros((shape, shape), dtype=float),
            unit=unit,
        )

        # go through all the complex amplitudes of the list
        for complex_amplitude in complex_amplitudes:
            n_dim = complex_amplitude.complex_amplitude.ndim
            if n_dim > 2:
                intensity += complex_amplitude.flux * np.sum(
                    complex_amplitude.intensity(),
                    axis=tuple(i for i in range(2, n_dim)),
                )
            else:
                intensity += complex_amplitude.flux * complex_amplitude.intensity()

        # take the exposure time into account
        intensity *= self.exposure_time

        self.noise_free_intensity = intensity

        # manage photon noise
        if self.photon_noise == True:
            photons = (
                self.random_state_generator.poisson(
                    intensity.astype(float).value,
                    size=(shape, shape),
                )
                * intensity.unit
            )
        else:
            photons = intensity

        # convert photons to electrons
        electrons = self.quantum_efficiency * photons

        # compute dark noise from the RON and dark current
        variance_dark_noise = (
            self.readout_noise_variance + self.dark_current * self.exposure_time
        )

        # add the computed noise to electrons
        electrons_out = (
            self.random_state_generator.normal(
                scale=variance_dark_noise.value, size=(shape, shape)
            )
            * u.electron
            + electrons
        )

        # Convert to ADU and add baseline
        max_adu = int(2**self.bitdepth - 1) * u.adu
        adu = (electrons_out * self.sensitivity).astype(
            int
        )  # Convert to discrete numbers

        adu += (self.offset).astype(int)
        # models pixel saturation
        adu[adu > max_adu] = max_adu

        return adu

    def display_intensity(self):
        plt.set_cmap("pink")
        plt.imshow(self.intensity.value)
        plt.title(f"Intensity in {self.intensity.unit}")
        plt.colorbar()
