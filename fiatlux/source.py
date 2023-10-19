# -*- coding: utf-8 -*-
"""
  ______                              
 / _____)                             
( (____   ___  _   _  ____ ____ _____ 
 \____ \ / _ \| | | |/ ___) ___) ___ |
 _____) ) |_| | |_| | |  ( (___| ____|
(______/ \___/|____/|_|   \____)_____)

Created on Fri Oct 13 10:05:00 2023

@author: pjanin
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import astropy.units as u

from .field import Field
from .spectrum import Spectrum


class Source:
    """The Source class define the physical properties of a source."""

    # Constructor
    def __init__(self, **kwargs):
        # Name of the source
        self.name = kwargs.get("name", "unamed_source")

        # Define type of field - default is plane wave
        self.type = kwargs.get("type", "plane_wave")

        # Define type-specific arguments
        match self.type:
            case "plane_wave":
                self.type_arguments = {
                    "incidence_angles": kwargs.get("incidence_angles", [0.0, 0.0]),
                }
            case "gaussian_wave":
                self.type_arguments = {
                    "center": kwargs.get("center", [0.0, 0.0]),
                    "variance": kwargs.get("variance", 1.0),
                }
            case "point_source":
                self.type_arguments = {
                    "center": kwargs.get("center", [0.0, 0.0])
                }

        # Magnitude of the source
        self.magnitude = kwargs.get("magnitude", 0)

        # Optical band of observation
        self.optical_band = kwargs.get("optical_band", "V")

        # Parameters corresponding to the spectrum
        lambda_eff, delta_lambda, f_0 = Spectrum.photometry(self.optical_band)

        # Effective wavelength
        self._wavelength = lambda_eff

        # Photon flux at the pupil in [photon / m2 / s]
        self.photon_flux = f_0 * 10 ** (-self.magnitude / 2.5)
