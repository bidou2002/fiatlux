# -*- coding: utf-8 -*-
"""
 ______             _ _
(_____ \           (_) |
 _____) )   _ ____  _| |
|  ____/ | | |  _ \| | |
| |    | |_| | |_| | | |
|_|    |____/|  __/|_|\_)
             |_|

Created on Fri Jun 15 15:19:33 2018
@author: pjanin
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import astropy.units as u
from .mask import Mask


class Pupil(Mask):
    """Une pupille hérite d'un masque et est caractérisée par:
    C'est un masque en amplitude !
    - sa forme (carrée/circulaire)
    - sa taille en pixels (côté/diamètre)"""

    """Ies méthodes :

    """

    """####################################################################"""
    """####################### INIT AND OVERLOADING #######################"""
    """####################################################################"""

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):
        # Initialization as a mask
        Mask.__init__(self, field_size)

        # Define aperture shape - by default set to circular
        self._aperture_shape = kwargs.get("aperture_shape", "circular")

        # Define aperture size in [m]
        self._aperture_size = kwargs.get("aperture_size", 1)
        self._aperture_size = self._aperture_size * (u.meter)

        # Define aperture resolution as the number of pixel in aperture_size [px]
        self.resolution = kwargs.get("resolution", field_size / 4)
        self.resolution = self.resolution * (u.pixel)

        self.scale = self.resolution / self._aperture_size
        print(self)

        # Prepare meshgrid to build the pupil
        x = np.linspace(
            -self._field_size / 2 + 0.5, self._field_size / 2 - 0.5, self._field_size
        )
        y = np.linspace(
            -self._field_size / 2 + 0.5, self._field_size / 2 - 0.5, self._field_size
        )
        x_grid, y_grid = np.meshgrid(x, y)
        if self._aperture_shape == "circular":
            d = np.sqrt(x_grid**2 + y_grid**2) * (u.pixel)
            self._complex_transparency[d < self.resolution / 2] = 1
            surface = np.pi * (self._aperture_size / 2) ** 2
        elif self._aperture_shape == "square":
            self._complex_transparency[
                (np.abs(x_grid) * (u.pixel) < self.resolution / 2)
                & (np.abs(y_grid) * (u.pixel) < self.resolution / 2)
            ] = 1
            surface = self._aperture_size**2

        # Define telescope surface
        self._surface = surface



    # REPRESENTERS
    def __str__(self):
        return (
            "<pupil>\n"
            "  * field_size : {} (px)\n"
            "  * scale : {} (px/m)\n"
            "  * aperture_size : {} (m)".format(
                self._field_size, self.scale, self._aperture_size
            )
        )