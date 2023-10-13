# -*- coding: utf-8 -*-
"""
 ______
(_____ \                                    _
 _____) )___ ___  ____  _____  ____ _____ _| |_ ___   ____
|  ____/ ___) _ \|  _ \(____ |/ _  (____ (_   _) _ \ / ___)
| |   | |  | |_| | |_| / ___ ( (_| / ___ | | || |_| | |
|_|   |_|   \___/|  __/\_____|\___ \_____|  \__)___/|_|
                 |_|         (_____|

Created on Fri Aug  3 09:38:13 2018

@author: pjanin
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from .polarizer import Polarizer
from .mirror import Mirror
from .mask import Mask
from .detector import Detector


class Propagator(object):

    """####################################################################"""

    """####################### INIT AND OVERLOADING #######################"""
    """####################################################################"""

    # CONSTRUCTOR
    def __init__(self, LCT_type, field, **kwargs):
        self._LCT_type = LCT_type
        self._wavelength = field._wavelength

        # FAST FOURIER TRANSFORMATION
        if self._LCT_type is "FFT" or self._LCT_type is "IFFT":
            # LCT
            self._LCT = [[0, 1], [-1, 0]]
        # FOURIER TRANSFORMATION
        elif self._LCT_type is "FT":
            self._focal_length = kwargs.get("focal_length", 300e-3)
            # LCT
            self._LCT = [
                [0, self._wavelength * self._f],
                [-1 / (self._wavelength * self._f), 0],
            ]
        # FRESNEL TRANSFORMATION
        elif self._LCT_type is "FST":
            self._z = kwargs.get("z", 300e-3)
            # LCT
            self._LCT = [[1, self._wavelength * self._z], [0, 1]]
        # FRACTIONNAL FOURIER TRANSFORMATION
        elif self._LCT_type is "FRT":
            self._p = kwargs.get("p", 0)
            # LCT
            self._LCT = [
                [
                    np.cos(self._p * np.pi / 2),
                    self._wavelength * self._q * np.sin(self._p * np.pi / 2),
                ],
                [
                    -np.sin(self._p * np.pi / 2) / (self._wavelength * self._q),
                    np.cos(self._p * np.pi / 2),
                ],
            ]
        #
        elif self._LCT_type is "CMT":
            self._focal_length = kwargs.get("focal_length", 300e-3)
            # LCT
            self._LCT = [[1, 0], [-1 / (self._wavelength * self._f), 1]]

    """####################################################################"""
    """####################### GET / SET DEFINITION #######################"""
    """####################################################################"""

    # GET/SET LCT_TYPE
    def _get_LCT_type(self):
        return self._LCT_type

    def _set_LCT_type(self, polar_type):
        print("Immutable property. Please create a new propagator instance.")

    def _del_LCT_type(self):
        print("Undeletable type property")

    """####################################################################"""
    """####################### FUNCTIONS DEFINITION #######################"""
    """####################################################################"""

    def propagate(self, field):
        unit = field.complex_amplitude.unit
        if self._LCT_type is "FFT":
            FFTCube = np.zeros(
                (
                    field.field_size,
                    field.field_size,
                    np.shape(field.complex_amplitude)[2],
                ),
                dtype=complex,
            )
            for k in range(0, np.shape(field.complex_amplitude)[2]):
                FFTCube[:, :, k] = (
                    np.fft.fftshift(
                        np.fft.fft2(np.fft.fftshift(field.complex_amplitude[:, :, k]))
                    )
                    / field.field_size
                )
            return FFTCube * unit
        elif self._LCT_type is "IFFT":
            IFFTCube = np.zeros(
                (
                    field.field_size,
                    field.field_size,
                    np.shape(field.complex_amplitude)[2],
                ),
                dtype=complex,
            )
            for k in range(0, np.shape(field.complex_amplitude)[2]):
                IFFTCube[:, :, k] = (
                    np.fft.fftshift(
                        np.fft.fft2(np.fft.fftshift(field.complex_amplitude[:, :, k]))
                    )
                    / field.field_size
                )
            return IFFTCube * unit
        # THE THING TO IMPLEMENT NOW !
        elif self._LCT_type is "FT":
            return 0
        # FST
        elif self._LCT_type is "FST":
            FSTCube = np.zeros(
                (
                    field.field_size,
                    field.field_size,
                    np.shape(field.complex_amplitude)[2],
                ),
                dtype=complex,
            )
            # NUMBER OF PIXELS
            N = field.field_size
            # FROM GARCIA ET AL. 96 (EQ. 7)
            Lx = N * field.scale
            wavelength = field.wavelength
            z = self._z
            x = np.linspace(-N / 2 + 0.5, N / 2 - 0.5, N)
            y = np.linspace(-N / 2 + 0.5, N / 2 - 0.5, N)
            x_grid, y_grid = np.meshgrid(x, y)
            if z < Lx**2 / (wavelength * N):
                for k in range(0, np.shape(field.complex_amplitude)[2]):
                    # FROM GARCIA ET AL. 96 (EQ. 10)
                    FSTCube[:, :, k] = np.fft.fftshift(
                        np.fft.ifft2(
                            np.fft.fftshift(
                                np.fft.fftshift(
                                    np.fft.fft2(
                                        np.fft.fftshift(
                                            field.complex_amplitude[:, :, k]
                                        )
                                    )
                                )
                                * np.exp(
                                    -1j
                                    * (
                                        np.pi
                                        * (wavelength * z)
                                        / (Lx**2)
                                        * x_grid**2
                                    )
                                )
                                * np.exp(
                                    -1j
                                    * (
                                        np.pi
                                        * (wavelength * z)
                                        / (Lx**2)
                                        * y_grid**2
                                    )
                                )
                            )
                        )
                    )
            else:
                for k in range(0, np.shape(field.complex_amplitude)[2]):
                    # FROM GARCIA ET AL. 96 (EQ. 6)
                    FSTCube[:, :, k] = np.fft.fftshift(
                        np.fft.fft2(
                            np.fft.fftshift(
                                field.complex_amplitude[:, :, k]
                                * np.exp(
                                    1j
                                    * (np.pi / (wavelength * z))
                                    * (Lx / N) ** 2
                                    * x_grid**2
                                )
                                * np.exp(
                                    1j
                                    * (np.pi / (wavelength * z))
                                    * (Lx / N) ** 2
                                    * y_grid**2
                                )
                            )
                        )
                    )
            return FSTCube
        elif self._LCT_type is "FRT":
            return 0

    def object_ID(self):
        return [self]

    """####################################################################"""
    """####################### PROPERTIES DEFINITION ######################"""
    """####################################################################"""

    LCT_type = property(
        _get_LCT_type,
        _set_LCT_type,
        _del_LCT_type,
        "The 'LCT_type' property defines the type of LCT. ",
    )
