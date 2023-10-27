from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np
import astropy.units as u

from .spectrum_dev import Spectrum, SpectralFilter, Monochromatic
from .complex_amplitude import ComplexAmplitude


@dataclass
class Detector:
    quantum_efficiency: float = 1
    photon_noise: bool = False
    readout_noise_variance: float = 0
    dark_current: float = 0
    exposure_time: float = 1
    offset: int = 0
    bitdepth: int = 16
    sensitivity: float = 1
    random_state_generator: np.random.RandomState = np.random.RandomState(seed=31415)
    name: str = "default_camera"

    def compute_intensity(
        self, complex_amplitude_list: [ComplexAmplitude], photon_per_second: float
    ) -> np.ndarray:
        # initialize intensity
        intensity = np.zeros(complex_amplitude_list[0].shape, dtype=float)
        for complex_amplitude in complex_amplitude_list:
            n_dim = complex_amplitude.ndim
            if n_dim > 2:
                print(type(complex_amplitude.compute_intensity()))
                intensity += np.sum(
                    complex_amplitude.compute_intensity(),
                    axis=tuple(i for i in range(2, n_dim)),
                )
            else:
                intensity += complex_amplitude.compute_intensity()

        # determine the number of photon per pixel
        intensity *= photon_per_second * self.exposure_time

        if self.photon_noise == True:
            photons = (
                self.random_state_generator.poisson(
                    intensity,
                    size=intensity.shape,
                )
                * u.photon
            )
        else:
            photons = intensity * u.photon

        electrons = self.quantum_efficiency * photons

        # Add dark noise
        variance_dark_noise += (
            self.readout_noise_variance + self.dark_current * self.exposure_time
        )

        electrons_out = (
            self.random_state_generator.normal(
                scale=variance_dark_noise.value, size=electrons.shape
            )
            * u.electron
            + electrons
        )

        # Convert to ADU and add baseline
        max_adu = int(2**self._bitdepth - 1) * u.adu
        adu = (electrons_out * self._sensitivity).astype(
            int
        )  # Convert to discrete numbers

        adu += self._offset.astype(int)

        # models pixel saturation
        adu[adu > max_adu] = max_adu

        return adu

    def display_intensity(self):
        pass
