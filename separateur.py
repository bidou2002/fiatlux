# -*- coding: utf-8 -*-
"""
  ______
 / _____)                                _
( (____  _____ ____  _____  ____ _____ _| |_ ___   ____
 \____ \| ___ |  _ \(____ |/ ___|____ (_   _) _ \ / ___)
 _____) ) ____| |_| / ___ | |   / ___ | | || |_| | |
(______/|_____)  __/\_____|_|   \_____|  \__)___/|_|
              |_|

Created on Fri Jun 15 15:19:33 2018

@author: ybrule
"""
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from arrow3D import Arrow3D
from polarizer import Polarizer
from mirror import Mirror


class Separator(Mirror):

    def __init__(self, **kwargs):

        Mirror.__init__(self, reflectance=kwargs.get('reflectance', 0.5),
                        angle=kwargs.get('angle', 0),
                        phase_retardation=kwargs.get('phase_retardation',
                                                     np.pi))
        self._transmitance = kwargs.get('transmitance', 0.5)
        self._mode = 'transmission'
        self.make_mueller_matrix()

    # GET/SET TRANSMITANCE
    def _get_transmitance(self):
        return self._transmitance

    def _set_transmitance(self, transmitance):
        self._transmitance = transmitance

    def _del_transmitance(self):
        print("Undeletable type property")

    # GET/SET REFLECTANCE
    def _get_reflectance(self):
        return self._reflectance

    def _set_reflectance(self, reflectance):
        self._reflectance = reflectance

    def _del_reflectance(self):
        print("Undeletable type property")

    # GET/SET MODE
    def _get_mode(self):
        return self._mode

    def _set_mode(self, mode):
        self._mode = mode

    def _del_mode(self):
        print("Undeletable type property")

    def object_ID(self):
        return [self]

    def R(self):
        self._mode = 'reflection'
        self.make_mueller_matrix()
        return self

    def T(self):
        self._mode = 'transmission'
        self.make_mueller_matrix()
        return self

    def make_mueller_matrix(self):
        if self._mode is 'transmission':
            self._mueller_matrix = [
                                    # First line
                                    [self._transmitance,
                                     0,
                                     0,
                                     0],
                                    # Second line
                                    [0,
                                     self._transmitance,
                                     0,
                                     0],
                                    # Third line
                                    [0,
                                     0,
                                     self._transmitance,
                                     0],
                                    # Fourth line
                                    [0,
                                     0,
                                     0,
                                     self._transmitance]]
        elif self._mode is 'reflection':
            self._mueller_matrix = [
                                    # First line
                                    [self._reflectance,
                                     0,
                                     0,
                                     0],
                                    # Second line
                                    [0,
                                     self._reflectance,
                                     0,
                                     0],
                                    # Third line
                                    [0,
                                     0,
                                     self._reflectance,
                                     0],
                                    # Fourth line
                                    [0,
                                     0,
                                     0,
                                     self._reflectance]]

    transmitance = property(_get_transmitance, _set_transmitance,
                            _del_transmitance)
    mode = property(_get_mode, _set_mode, _del_mode)
