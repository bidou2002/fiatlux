# -*- coding: utf-8 -*-
"""
 _
(_)
 _       _____ ____   ___
| |     | ___ |  _ \ /___)
| |_____| ____| | | |___ |
|_______)_____)_| |_(___/

Created on Fri Jun 15 15:19:33 2018

La classe lentille pour une simulation de banc optique

@author: ybrule
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


class Lens(object):
    """Une lentille est caractérisée par:
    - sa focale [m]
    - son ouverture numérique
    - son ouverture géométrique(OG)
    - son diamètre [m] (défaut 5 cm)
    - sa longueur d'onde (defaut 589 nm gamme de longueurs d'onde) [m] """
    def __init__(self, field_size, focal_length, **kwargs):
        #FIELD SIZE
        self._field_size = field_size
        # FOCAL LENGTH
        self._focal_length = focal_length
        # LENS DIAMETER
        self._wavelength = kwargs.get('wavelength', 589e-9)
        # LENS DIAMETER
        self._diameter = kwargs.get('diameter', 5e-2)
        # LENS INDEX (BK7 DEFAULT)
        self._index = kwargs.get('index', 1.52)

    # DESTRUCTOR
    def __del__(self):
        del(self)

    '''####################################################################'''
    '''####################### GET / SET DEFINITION #######################'''
    '''####################################################################'''

    # GET/SET FOCAL LENGTH
    def _get_focal_length(self):
        return self._focal_length

    def _set_focal_length(self, focal_length):
        self._focal_length = focal_length

    def _del_focal_length(self):
        print("Undeletable type property")

    # GET/SET DIAMETER
    def _get_diameter(self):
        return self._diameter

    def _set_diameter(self, diameter):
        self._diameter = diameter

    def _del_diameter(self):
        print("Undeletable type property")

    # GET/SET INDEX
    def _get_index(self):
        return self._index

    def _set_index(self, index):
        self._index = index

    def _del_index(self):
        print("Undeletable type property")

    # GET/SET NUMERICAL APERTURE
    def _get_numerical_aperture(self):
        return self._index*np.sin(np.arctan(
                self._diameter/(2*self._focal_length)))

    def _set_numerical_aperture(self, geometrical_aperture):
        print('PLease set \'focal_length\' or \'diameter\' accordingly')

    def _del_numerical_aperture(self):
        print("Undeletable type property")

    # GET/SET GEOMETRICAL APERTURE
    def _get_geometrical_aperture(self):
        return self._focal_length/self._diameter

    def _set_geometrical_aperture(self, geometrical_aperture):
        print('PLease set \'focal_length\' or \'diameter\' accordingly')

    def _del_geometrical_aperture(self):
        print("Undeletable type property")

        # GET/SET COMPLEX TRANSMISSION
    def _get_complex_transparency(self):
        x = np.linspace(-self._field_size/2 + 0.5, self._field_size/2 - 0.5,
                        self._field_size)
        y = np.linspace(-self._field_size/2 + 0.5, self._field_size/2 - 0.5,
                        self._field_size)
        x_grid, y_grid = np.meshgrid(x, y)
        return np.exp(-1j*np.pi*(x_grid**2 + y_grid**2) /
                      (self._wavelength*self._focal_length))

    def _set_complex_transparency(self, geometrical_aperture):
        print('PLease set \'focal_length\' accordingly')

    def _del_complex_transparency(self):
        print("Undeletable type property")

    '''####################################################################'''
    '''####################### FUNCTIONS DEFINITION #######################'''
    '''####################################################################'''

    def object_ID(self):
        return [self]

    '''####################################################################'''
    '''####################### PROPERTIES DEFINITION ######################'''
    '''####################################################################'''

    focal_length = property(_get_focal_length, _set_focal_length,
                            _del_focal_length, "The 'geometrical_aperture' "
                            "property defines the geometrical aperture of the "
                            "lens")
    diameter = property(_get_diameter, _set_diameter, _del_diameter,
                        "The 'geometrical_aperture' property defines the "
                        "geometrical aperture of the lens")
    index = property(_get_index, _set_index, _del_index,
                     "The 'geometrical_aperture' property defines the "
                     "geometrical aperture of the lens")
    numerical_aperture = property(_get_numerical_aperture,
                                  _set_numerical_aperture,
                                  _del_numerical_aperture,
                                  "The 'geometrical_aperture' property "
                                  "defines the geometrical aperture of the "
                                  "lens")
    geometrical_aperture = property(_get_geometrical_aperture,
                                    _set_geometrical_aperture,
                                    _del_geometrical_aperture,
                                    "The 'geometrical_aperture' property "
                                    "defines the geometrical aperture of the "
                                    "lens")
    complex_transparency = property(_get_complex_transparency,
                                    _set_complex_transparency,
                                    _del_complex_transparency,
                                    "The 'geometrical_aperture' property "
                                    "defines the geometrical aperture of the "
                                    "lens")
