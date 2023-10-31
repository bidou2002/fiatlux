# -*- coding: utf-8 -*-
"""
 ______  _______ ______
(______)(_______|______)
 _     _ _  _  _ _     _
| |   | | ||_|| | |   | |
| |__/ /| |   | | |__/ /
|_____/ |_|   |_|_____/

Created on Fri Jun 15 15:19:33 2018

@author: pjanin
"""
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


class DMD(object):

    '''####################################################################'''
    '''####################### INIT AND OVERLOADING #######################'''
    '''####################################################################'''

    def __init__(self, field_size, **kwargs):
        self._field_size = field_size
        self._facets = kwargs.get('facets', np.int(self._field_size/2))
        self._complex_amplitude = kwargs.get('pistons', np.ones(
                (self._field_size, self._field_size,
                 1), dtype=complex))
        self._build_waffer()
        self._build_DMD()

    '''####################################################################'''
    '''####################### GET / SET DEFINITION #######################'''
    '''####################################################################'''

    def _get_field_size(self):
        return self._field_size

    def _set_field_size(self, field_size):
        self._field_size = field_size

    def _del_field_size(self):
        print("Undeletable property")

    '''####################################################################'''
    '''####################### FUNCTIONS DEFINITION #######################'''
    '''####################################################################'''

    def _build_waffer(self):
        facets = self._facets
        waffer = np.zeros((self._field_size, self._field_size))
        for i in range(facets-1):
            waffer[i*facets:(i+1)*facets] = 1
        self._waffer = waffer

    def _build_DMD(self):
        self._complex_amplitude = np.exp(1)

    '''####################################################################'''
    '''####################### PROPERTIES DEFINITION ######################'''
    '''####################################################################'''

    field_size = property(_set_field_size, _get_field_size, _del_field_size)
