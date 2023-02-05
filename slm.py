# -*- coding: utf-8 -*-
"""
  ______ _       _______
 / _____|_)     (_______)
( (____  _       _  _  _
 \____ \| |     | ||_|| |
 _____) ) |_____| |   | |
(______/|_______)_|   |_|

Created on Fri Jul 13 12:48:32 2018

@author: pjanin
"""
import matplotlib.pyplot as plt
import numpy as np
from polarizer import Polarizer


class SLM(Polarizer):

    '''####################################################################'''
    '''####################### INIT AND OVERLOADING #######################'''
    '''####################################################################'''

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):
        self._field_size = field_size
        self._complex_transparency = np.ones((self._field_size,
                                              self._field_size))
        self._filling_factor = kwargs.get('filling_factor', 1)
        Polarizer.__init__(self, 'retarder', angle=kwargs.get('angle', 0),
                           phase_retardation=kwargs.get(
                                   'phase_retardation', 0))
        self._polarization_bool = kwargs.get('polarization_bool', False)
        print(self._polarization_bool)
        shape = kwargs.get('shape', None)
        if shape is 'pyramid':
            pyramid_angle = kwargs.get('pyramid_angle', self._field_size)
            pyramid_faces = kwargs.get('pyramid_faces', 4)
            self.Pyramid(pyramid_angle=pyramid_angle,
                         pyramid_faces=pyramid_faces)
        elif shape is 'fqpm':
            self.FQPM()
        elif shape is 'vortex':
            vortex_charge = kwargs.get('vortex_charge', 1)
            self.Vortex(vortex_charge=vortex_charge)
        self._make_polarization()

    # DESTRUCTOR
    def __del__(self):
        del(self)

    '''####################################################################'''
    '''####################### GET / SET DEFINITION #######################'''
    '''####################################################################'''

    # GET/SET SIZE
    def _get_field_size(self):
        return self._field_size

    def _set_field_size(self, field_size):
        self._field_size = field_size

    def _del_field_size(self, field_size):
        print("Undeletable type property")

    # GET/SET COMPLEX_TRANSPARENCY
    def _get_complex_transparency(self):
        return self._complex_transparency

    def _set_complex_transparency(self, complex_transparency):
        if (isinstance(complex_transparency, np.ndarray) and
            (complex_transparency.shape == (self._field_size,
                                            self._field_size))):
            self._complex_transparency = complex_transparency
            # print("Complex amplitude changed to {}"
            #       .format(self._complex_transparency))
        else:
            print("Complex transparency is not of the right format or does not"
                  " have the right shape ({}x{}x{}px)"
                  .format(self._field_size, self._field_size))

    def _del_complex_transparency(self):
        print("Undeletable property")

    # GET/SET AMPLITUDE
    def _get_amplitude(self):
        return np.abs(self._complex_transparency)

    def _set_amplitude(self, amplitude):
        if isinstance(amplitude, np.ndarray) and (amplitude.shape ==
                                                  (self._field_size,
                                                   self._field_size)):
            phase = np.angle(self._complex_transparency)
            self._complex_transparency = amplitude*np.exp(1j*phase)
            # print("Amplitude changed to {}"
            #       .format(np.abs(self._complex_transparency)))
        else:
            print("Amplitude is not of the right format or does not have the "
                  "right shape ({}x{}x{}px)".format(self._field_size,
                                                    self._field_size))

    def _del_amplitude(self):
        print("Undeletable property")

    # GET/SET PHASE
    def _get_phase(self):
        return np.angle(self._complex_transparency)

    def _set_phase(self, phase):
        if isinstance(phase, np.ndarray) and (phase.shape ==
                                              (self._field_size,
                                               self._field_size)):
            amplitude = np.abs(self._complex_transparency)
            self._complex_transparency = amplitude*np.exp(1j*phase)
            # print("Phase changed to {}"
            #       .format(np.angle(self._complex_transparency)))
        else:
            print("Phase is not of the right format or does not have the "
                  "right shape ({}x{}x{}px)".format(self._field_size,
                                                    self._field_size))

    def _del_phase(self):
        print("Undeletable property")

    # GET/SET FILLING_FACTOR
    def _get_filling_factor(self):
        print("SLM filling_factor is {}".format(self._filling_factor))
        return self._filling_factor

    def _set_filling_factor(self, filling_factor):
        self._filling_factor = filling_factor
        print("SLM filling_factor changed")

    def _del_filling_factor(self, filling_factor):
        print("Undeletable type property")

    '''####################################################################'''
    '''####################### FUNCTIONS DEFINITION #######################'''
    '''####################################################################'''

    def _make_polarization(self):
        if self._polarization_bool is True:
            self._polarization = (
                    [[Polarizer('retarder',
                                phase_retardation=(
                                        np.angle(
                                                self.complex_transparency[i, j]
                                                )))
                        for j in range(self._field_size)]
                        for i in range(self._field_size)])

    def FQPM(self):
        N = self._field_size
        FQPM = np.ones((N, N), dtype=complex)
        FQPM[0:N//2, 0:N//2] = -1
        FQPM[N//2:N, N//2:N] = -1
        self._complex_transparency = FQPM
        self._make_polarization()
        return self

    def Pyramid(self, **kwargs):
        # GET PARAMETERS
        pyramid_angle = kwargs.get('pyramid_angle', 1)
        pyramid_faces = kwargs.get('pyramid_faces', 4)
        N = self._field_size
        # DEFINE X AND Y GRIDS
        x = np.linspace(-N//2 + 0.5, N//2 - 0.5, N)
        y = np.linspace(-N//2 + 0.5, N//2 - 0.5, N)
        x_grid, y_grid = np.meshgrid(x, y)
        # DEFINE THE SLOPE GRID
        angle_grid = np.arctan2(y_grid, x_grid)
        # INITIALIZE PYRAMID MASK
        Pyramid = np.zeros((N, N))
        for i in range(pyramid_faces):
            theta = np.pi*(1./pyramid_faces - 1) + i*2*np.pi/pyramid_faces
            slope = np.cos(theta)*x_grid + np.sin(theta)*y_grid
            slope[(-np.pi+i*2*np.pi/pyramid_faces > angle_grid) |
                  (angle_grid > (-np.pi+(i + 1)*2*np.pi/pyramid_faces))] = 0
            Pyramid = Pyramid + pyramid_angle*slope
        self._complex_transparency = np.exp(-1j*np.pi*Pyramid/N)
        self._make_polarization()
        return self

    def Vortex(self, **kwargs):
        vortex_charge = kwargs.get('vortex_charge', 1)
        N = self._field_size
        x = np.linspace(-N//2 + 0.5, N//2 - 0.5, N)
        y = np.linspace(-N//2 + 0.5, N//2 - 0.5, N)
        x_grid, y_grid = np.meshgrid(x, y)
        Vortex = np.exp(1j*vortex_charge*np.arctan2(y_grid, x_grid))
        self._complex_transparency = Vortex
        self._make_polarization()
        return self

    '''####################################################################'''
    '''####################### PROPERTIES DEFINITION ######################'''
    '''####################################################################'''

    # DEFINE THE PROPERTIES
    field_size = property(_get_field_size, _set_field_size, _del_field_size,
                          "The 'size' property defines the size of the SLM.")
    complex_transparency = property(_get_complex_transparency,
                                    _set_complex_transparency,
                                    _del_complex_transparency,
                                    "The 'amplitude' property defines the "
                                    "amplitude map on the SLM.")
    amplitude = property(_get_amplitude, _set_amplitude, _del_amplitude,
                         "The 'amplitude' property defines the amplitude map "
                         "on the SLM.")
    phase = property(_get_phase, _set_phase, _del_phase,
                     "The 'phase' property defines the phase map on "
                     "the SLM.")
    filling_factor = property(_get_filling_factor, _set_filling_factor,
                              _del_filling_factor,
                              "The 'filling_factor' property defines the "
                              "filling factor of the SLM.")
