# -*- coding: utf-8 -*-
"""
 _______ _
(_______|_)
 _  _  _ _  ____ ____ ___   ____
| ||_|| | |/ ___) ___) _ \ / ___)
| |   | | | |  | |  | |_| | |
|_|   |_|_|_|  |_|   \___/|_|

Created on Tue Jul 31 10:26:58 2018

@author: pjanin
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from arrow3D import Arrow3D
from polarizer import Polarizer


class Mirror(Polarizer):

    '''####################################################################'''
    '''####################### INIT AND OVERLOADING #######################'''
    '''####################################################################'''

    # CONSTRUCTOR
    def __init__(self, **kwargs):

        # FIELD_SIZE IS A MANDATORY PARAMETER
        self._reflectance = kwargs.get('reflectance', 1)

        Polarizer.__init__(self, 'linear', angle=kwargs.get('angle', 0),
                           phase_retardation=kwargs.get('phase_retardation',
                                                        np.pi))

    # DESTRUCTOR
    def __del__(self):
        # print("Deleted mirror")
        del(self)

    # REPRESENTER
    def __str__(self):
        return ("<mirror>\n"
                "  * reflectance : {} \n"
                "  * angle : {} (rad)\n"
                "  * phase_retardation : {} (rad)"
                .format(self._reflectance, self._angle,
                        self._phase_retardation))

    # MULTIPLACATION SURCHARGE
    def __mul__(self, vararg):
        return self

    '''####################################################################'''
    '''####################### GET / SET DEFINITION #######################'''
    '''####################################################################'''

    # GET/SET MUELLER MATRIX
    def _get_reflectance(self):
        # print("Reflectance is {}".format(self._reflectance))
        return self._reflectance

    def _set_reflectance(self, reflectance):
        self._reflectance = reflectance
        # print("Reflectance set")

    def _del_reflectance(self):
        print("Undeletable type property")

    '''####################################################################'''
    '''####################### FUNCTIONS DEFINITION #######################'''
    '''####################################################################'''

    def object_ID(self):
        return [self]

    # MAKE_MUELLER_MATRIX
    def _make_mueller_matrix(self):
        t = self._angle
        d = self._phase_retardation
        r = self._reflectance
        a = (r + 1)/2
        b = (r - 1)/2
        self._mueller_matrix = [
                                # First line
                                [a,
                                 b*np.cos(2*t),
                                 b*np.sin(2*t),
                                 0],
                                # Second line
                                [b*np.cos(2*t),
                                 a*np.cos(2*t)**2 +
                                 np.sqrt(r)*np.cos(d)*np.sin(2*t)**2,
                                 (a - np.sqrt(r)*np.cos(d))*np.sin(4*t)/2,
                                 -np.sqrt(r)*np.sin(d)*np.sin(2*t)],
                                # Third line
                                [b*np.sin(2*t),
                                 (a - np.sqrt(r)*np.cos(d))*np.sin(4*t)/2,
                                 a*np.sin(2*t)**2 +
                                 np.sqrt(r)*np.cos(d)*np.cos(2*t)**2,
                                 np.sqrt(r)*np.sin(d)*np.cos(2*t)],
                                # Fourth line
                                [0,
                                 np.sqrt(r)*np.sin(d)*np.sin(2*t),
                                 -np.sqrt(r)*np.sin(d)*np.cos(2*t),
                                 np.sqrt(r)*np.cos(d)]]

    '''####################################################################'''
    '''####################### PROPERTIES DEFINITION ######################'''
    '''####################################################################'''

    reflectance = property(_get_reflectance, _set_reflectance,
                           _del_reflectance)
