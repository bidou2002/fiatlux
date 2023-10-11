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
from .arrow3D import Arrow3D
from .polarizer import Polarizer
from .mirror import Mirror
from .masque import Mask


class Detector(object):

    '''####################################################################'''
    '''####################### INIT AND OVERLOADING #######################'''
    '''####################################################################'''

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):

        # FIELD_SIZE IS A MANDATORY PARAMETER
        self._field_size = field_size
        # FIELD_SIZE IS A MANDATORY PARAMETER
        self._complex_amplitude = np.zeros((self._field_size,
                                            self._field_size, 1))
        self._display_id = id(self)
        self._camera_name = kwargs.get('camera_name', 'Default camera')
        self._display_intensity = kwargs.get('display_intensity', True)

    # DESTRUCTOR
    def __del__(self):
        del(self)

    # REPRESENTER
    def __str__(self):
        return ("* field_size : {} (px)\n"
                .format(self._field_size))

    '''####################################################################'''
    '''####################### GET / SET DEFINITION #######################'''
    '''####################################################################'''

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

    '''####################################################################'''
    '''####################### FUNCTIONS DEFINITION #######################'''
    '''####################################################################'''

    def object_ID(self):
        return [self]

    def disp_intensity(self):
        fig = plt.figure(self._display_id)
        if not ('inline' in matplotlib.get_backend()):
            fig.canvas.set_window_title(self._camera_name)
        plt.imshow(np.sum(np.abs((self._complex_amplitude)**2), axis=2))
        plt.title('Intensity (in ???)')
        plt.show()

    '''####################################################################'''
    '''####################### PROPERTIES DEFINITION ######################'''
    '''####################################################################'''

    field_size = property(_get_field_size, _set_field_size, _del_field_size,
                          "The 'field_size' property defines the field size "
                          "(in px)")

    complex_amplitude = property(_get_complex_amplitude,
                                 _set_complex_amplitude,
                                 _del_complex_amplitude,
                                 "The 'amplitude' property defines the "
                                 "amplitude of the wave (in ???)")

    camera_name = property(_get_camera_name, _set_camera_name,
                           _del_camera_name,
                          "The 'camera_name' property defines the camera name ")

    display_intensity= property(_get_display_intensity, _set_display_intensity,
                           _del_display_intensity,
                          "The 'display_intensity' property defines the"
                          " display_intensity ")
