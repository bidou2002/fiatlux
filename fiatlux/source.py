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


class Source(Field):
    # Constructor
    def __init__(self, field_size, **kwargs):
        self.name = kwargs.get("name", "unamed_source")

        Field.__init__(self, field_size=field_size)

        self._create_plane_wave(incidence_angles=[0, 0])

    def _create_point_source(self):
        pass

    def _create_plane_wave(self, incidence_angles):
        self._incidence_angles = incidence_angles
        x = np.linspace(
            -self._field_size / 2, self._field_size / 2 - 1, self._field_size
        )
        y = np.linspace(
            -self._field_size / 2, self._field_size / 2 - 1, self._field_size
        )
        x_grid, y_grid = np.meshgrid(x, y)
        self._complex_amplitude[:, :, 0] = np.exp(
            1j
            * (self._incidence_angles[0] * x_grid + self._incidence_angles[1] * y_grid)
        )

    def _create_extended_object(
        self,
    ):
        pass
