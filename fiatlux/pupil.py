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
from .mask import Mask


class Pupil(Mask):
    """Une pupille hérite d'un masque et est caractérisée par:
    C'est un masque en amplitude !
    - sa forme (carrée/circulaire)
    - sa taille en pixels (côté/diamètre)"""

    """Ies méthodes :

    """

    '''####################################################################'''
    '''####################### INIT AND OVERLOADING #######################'''
    '''####################################################################'''

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):
        # MASK INITIALIZATION
        Mask.__init__(self, field_size)
        # GET APERTURE SHAPE
        self._aperture_shape = kwargs.get('aperture_shape', 'circular')
        # GET APERTURE SIZE
        self._aperture_size = kwargs.get('aperture_size', field_size/4)
        x = np.linspace(-self._field_size/2 + 0.5, self._field_size/2 - 0.5,
                        self._field_size)
        y = np.linspace(-self._field_size/2 + 0.5, self._field_size/2 - 0.5,
                        self._field_size)
        x_grid, y_grid = np.meshgrid(x, y)
        if self._aperture_shape is 'circular':
            d = np.sqrt(x_grid**2 + y_grid**2)
            self._complex_transparency[d < self._aperture_size/2] = 1
        elif self._aperture_shape is 'square':
            self._complex_transparency[
                    (np.abs(x_grid) < self._aperture_size/2) &
                    (np.abs(y_grid) < self._aperture_size/2)] = 1
