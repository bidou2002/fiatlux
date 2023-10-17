# -*- coding: utf-8 -*-
"""
 ______
(______)         _                 _
 _     _ _____ _| |_ _____  ____ _| |_ ___   ____
| |   | | ___ (_   _) ___ |/ ___|_   _) _ \ / ___)
| |__/ /| ____| | |_| ____( (___  | || |_| | |
|_____/ |_____)  \__)_____)\____)  \__)___/|_|

Created on Fri Jun 15 15:19:33 2018

@author: pjanin
"""

import numpy as np
import scipy as sc
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import astropy.units as u
import astropy.constants as C
from .arrow3D import Arrow3D
from .polarizer import Polarizer
from .mirror import Mirror
from .mask import Mask

plt.rcParams["image.cmap"] = "pink"


class Detector(object):
    """"""

    """####################################################################"""
    """####################### INIT AND OVERLOADING #######################"""
    """####################################################################"""

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):
        # FIELD_SIZE IS A MANDATORY PARAMETER
        self._field_size = field_size
        # FIELD_SIZE IS A MANDATORY PARAMETER
        self._complex_amplitude = np.zeros((self._field_size, self._field_size, 1))
        self.intensity = np.zeros((self._field_size, self._field_size, 1))

        self._display_id = id(self)
        self._camera_name = kwargs.get("camera_name", "Default camera")
        self._display_intensity = kwargs.get("display_intensity", True)

        # Camera noise integration
        # Adapted from http://kmdouglass.github.io/posts/modeling-noise-for-image-simulations/

        # Initialize random state
        self._random_state_generator = np.random.RandomState(seed=31415)

        # Choose if noise is included or not
        self._noise = kwargs.get("noise", False)

        # Quantum efficiency
        self._quantum_efficiency = kwargs.get("quantum_efficiency", 1)
        self._quantum_efficiency = self._quantum_efficiency * (u.electron / u.photon)

        # Choose if readout noise is included or not
        self._readout_noise_variance = kwargs.get("readout_noise_variance", 0)
        self._readout_noise_variance = self._readout_noise_variance * (u.electron)

        # Choose if dark current is included or not
        self._dark_current = kwargs.get("dark_current", 0)
        self._dark_current = self._dark_current * (u.electron / u.second)

        # Choose if dark current is included or not
        self._exposure_time = kwargs.get("exposure_time", 0)
        self._exposure_time = self._exposure_time * (u.second)

        #
        self._offset = kwargs.get("offset", 100)
        self._offset = self._offset * (u.adu)

        self._bitdepth = kwargs.get("bitdepth", 12)

        self._sensitivity = kwargs.get("sensitivity", 5)
        self._sensitivity = self._sensitivity * (u.adu / u.electron)

        self._observing_wavelength = None

    # DESTRUCTOR
    def __del__(self):
        del self

    # REPRESENTER
    def __str__(self):
        return "* field_size : {} (px)\n".format(self._field_size)

    """####################################################################"""
    """####################### GET / SET DEFINITION #######################"""
    """####################################################################"""

    # GET/SET FIELD SIZE
    def _get_field_size(self):
        # print("Complex transparency is {}"
        #       .format(self._complex_transparency))
        return self._field_size

    def _set_field_size(self, field_size):
        self._field_size = field_size

    def _del_field_size(self):
        print("Undeletable type property")

    # GET/SET COMPLEX AMPLITUDE
    def _get_complex_amplitude(self):
        # print("Complex transparency is {}"
        #       .format(self._complex_transparency))
        return self._complex_amplitude

    def _set_complex_amplitude(self, complex_amplitude):
        self._complex_amplitude = complex_amplitude

    def _del_complex_amplitude(self):
        print("Undeletable type property")

    # GET/SET COMPLEX AMPLITUDE
    def _get_camera_name(self):
        # print("Complex transparency is {}"
        #       .format(self._complex_transparency))
        return self._camera_name

    def _set_camera_name(self, camera_name):
        self._camera_name = camera_name

    def _del_camera_name(self):
        print("Undeletable type property")

    # GET/SET DISPLAY INTENSITY
    def _get_display_intensity(self):
        # print("Complex transparency is {}"
        #       .format(self._complex_transparency))
        return self._display_intensity

    def _set_display_intensity(self, display_intensity):
        self._display_intensity = display_intensity

    def _del_display_intensity(self):
        print("Undeletable type property")

    """####################################################################"""
    """####################### FUNCTIONS DEFINITION #######################"""
    """####################################################################"""

    def object_ID(self):
        return [self]

    def compute_intensity(self):
        self.intensity = (
            self._exposure_time
            * np.sum(np.abs((self._complex_amplitude) ** 2), axis=2)
            * (u.pixel**2)
        )
        if self._noise is True:
            self.add_noise()

    def disp_intensity(self):
        fig = plt.figure(self._display_id)
        if not ("inline" in matplotlib.get_backend()):
            fig.canvas.set_window_title(self._camera_name)
        self.compute_intensity()
        plt.imshow(self.intensity.value)
        plt.title(f"Intensity (in {self.intensity.unit})")
        plt.colorbar()
        plt.show()

    def add_noise(self):
        photons = (
            self._random_state_generator.poisson(
                (self.intensity * (self._observing_wavelength / (C.h * C.c))).value,
                size=self.intensity.shape,
            )
            * u.photon
        )
        electrons = self._quantum_efficiency * photons

        # Add dark noise
        variance_dark_noise = (
            self._readout_noise_variance + self._dark_current * self._exposure_time
        )
        electrons_out = (
            self._random_state_generator.normal(
                scale=variance_dark_noise.value, size=electrons.shape
            )
            + electrons
        )

        # Convert to ADU and add baseline
        max_adu = int(2**self._bitdepth - 1) * u.adu
        adu = (electrons_out * self._sensitivity).astype(
            int
        )  # Convert to discrete numbers
        
        print(adu.unit, max_adu.unit)

        adu += self._offset.astype(int)

        # models pixel saturation
        adu[adu > max_adu] = max_adu

        self.intensity = adu
        

    """####################################################################"""
    """####################### PROPERTIES DEFINITION ######################"""
    """####################################################################"""

    field_size = property(
        _get_field_size,
        _set_field_size,
        _del_field_size,
        "The 'field_size' property defines the field size " "(in px)",
    )

    complex_amplitude = property(
        _get_complex_amplitude,
        _set_complex_amplitude,
        _del_complex_amplitude,
        "The 'amplitude' property defines the " "amplitude of the wave (in ???)",
    )

    camera_name = property(
        _get_camera_name,
        _set_camera_name,
        _del_camera_name,
        "The 'camera_name' property defines the camera name ",
    )

    display_intensity = property(
        _get_display_intensity,
        _set_display_intensity,
        _del_display_intensity,
        "The 'display_intensity' property defines the" " display_intensity ",
    )
