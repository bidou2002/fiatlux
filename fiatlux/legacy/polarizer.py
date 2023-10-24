# -*- coding: utf-8 -*-
"""
 ______      _
(_____ \    | |            (_)
 _____) )__ | | _____  ____ _ _____ _____  ____
|  ____/ _ \| |(____ |/ ___) (___  ) ___ |/ ___)
| |   | |_| | |/ ___ | |   | |/ __/| ____| |
|_|   \___/ \_)_____|_|    |_(_____)_____)_|

Created on Fri Jul 13 12:48:32 2018

@author: pjanin
"""

import matplotlib.pyplot as plt
import numpy as np


class Polarizer(object):

    '''####################################################################'''
    '''####################### INIT AND OVERLOADING #######################'''
    '''####################################################################'''

    # CONSTRUCTOR
    def __init__(self, polar_type, **kwargs):
        if polar_type not in ('linear', 'retarder', 'rotator'):
            raise ValueError("polar_type should be either 'linear' / "
                             "'retarder' / 'rotator'")
        self._polar_type = polar_type
        self._angle = kwargs.get('angle', 0)
        self._phase_retardation = kwargs.get('phase_retardation', 0)
        self._jones_matrix = np.ones((2, 2))
        self._mueller_matrix = np.eye(4)
        # PROCESS THE JONES MATRIX
        self._make_jones_matrix()
        self._make_mueller_matrix()

    # DESTRUCTOR
    def __del__(self):
        # print("Deleted polarizer")
        del(self)

    # REPRESENTER
    def __str__(self):
        return ("<polarizer>\n"
                "  * polarizer_type : {}\n"
                "  * angle : {} (rad)\n"
                "  * jones_matrix : {}"
                .format(self._polar_type, self._angle, self._jones_matrix))

    '''####################################################################'''
    '''####################### GET / SET DEFINITION #######################'''
    '''####################################################################'''

    # GET/SET POLARIZER_TYPE
    def _get_polar_type(self):
        # print("Polarizer type is {}".format(self._polar_type))
        return self._polar_type

    def _set_polar_type(self, polar_type):
        self._polar_type = polar_type
        # print("Polarizer type changed")
        self._make_jones_matrix()
        self._make_mueller_matrix()

    def _del_polar_type(self):
        print("Undeletable type property")

    # GET/SET ANGLE
    def _get_angle(self):
        # print("Polarizer angle is {}".format(self._angle))
        return self._angle

    def _set_angle(self, angle):
        self._angle = angle
        # print("Polarizer angle changed")
        self._make_jones_matrix()
        self._make_mueller_matrix()

    def _del_angle(self):
        print("Undeletable type property")

    # GET/SET PHASE_RETARDATION
    def _get_phase_retardation(self):
        # print("Polarizer phase retardation is {}"
        #       .format(self._phase_retardation))
        return self._phase_retardation

    def _set_phase_retardation(self, phase_retardation):
        self._phase_retardation = phase_retardation
        # print("Polarizer phase retardation changed")
        self._make_jones_matrix()
        self._make_mueller_matrix()

    def _del_phase_retardation(self):
        print("Undeletable type property")

    # GET/SET JONES MATRIX
    def _get_jones_matrix(self):
        # print("Jones matrix is {}".format(self._jones_matrix))
        return self._jones_matrix

    def _set_jones_matrix(self, jones_matrix):
        self._jones_matrix = jones_matrix
        # print("Jones matrix set")

    def _del_jones_matrix(self):
        print("Undeletable type property")

    # GET/SET MUELLER MATRIX
    def _get_mueller_matrix(self):
        # print("Mueller matrix is \n{}".format(self._mueller_matrix))
        return self._mueller_matrix

    def _set_mueller_matrix(self, mueller_matrix):
        self._mueller_matrix = mueller_matrix
        # print("Mueller matrix set")

    def _del_mueller_matrix(self):
        print("Undeletable type property")

    '''####################################################################'''
    '''####################### FUNCTIONS DEFINITION #######################'''
    '''####################################################################'''

    # MAKE_JONES_MATRIX
    def _make_jones_matrix(self):
        if self._polar_type == 'linear':
            self._jones_matrix = [[np.cos(self._angle)**2,
                                   np.cos(self._angle)*np.sin(self._angle)],
                                  [np.cos(self._angle)*np.sin(self._angle),
                                   np.sin(self._angle)**2]]
        elif self._polar_type == 'retarder':
            # EQUATION 3.14 OF FUNDAMENTALS OF LIQUID CRYSTALS DEVICES
            self._jones_matrix = [[np.cos(self._angle)**2 *
                                   np.exp(-1j*self._phase_retardation/2) +
                                   np.sin(self._angle)**2 *
                                   np.exp(1j*self._phase_retardation/2),
                                   np.cos(self._angle)*np.sin(self._angle) *
                                   np.exp(-1j*self._phase_retardation/2 -
                                          np.exp(1j *
                                                 self._phase_retardation/2))],
                                  [np.cos(self._angle)*np.sin(self._angle) *
                                   np.exp(-1j*self._phase_retardation/2 -
                                          np.exp(1j *
                                                 self._phase_retardation/2)),
                                   np.cos(self._angle)**2 *
                                   np.exp(-1j*self._phase_retardation/2) +
                                   np.exp(1j*self._phase_retardation/2)]]
        elif self._polar_type == 'rotator':
            self._jones_matrix = [[np.cos(self._angle), -np.sin(self._angle)],
                                  [np.sin(self._angle), np.cos(self._angle)]]

    # MAKE_MUELLER_MATRIX
    def _make_mueller_matrix(self):
        alpha = self._angle
        delta = self._phase_retardation
        if self._polar_type == 'linear':
            self._mueller_matrix = np.multiply(.5,
                                               [[1,
                                                 np.cos(2*alpha),
                                                 np.sin(2*alpha),
                                                 0],
                                                # Second line
                                                [np.cos(2*alpha),
                                                 np.cos(2*alpha)**2,
                                                 np.cos(2*alpha) *
                                                 np.sin(2*alpha),
                                                 0],
                                                # Third line
                                                [np.sin(2*alpha),
                                                 np.cos(2*alpha) *
                                                 np.sin(2*alpha),
                                                 np.sin(2*alpha)**2,
                                                 0],
                                                # Fourth line
                                                [0,
                                                 0,
                                                 0,
                                                 0]])
        elif self._polar_type == 'retarder':
            # EQUATION 3.14 OF FUNDAMENTALS OF LIQUID CRYSTALS DEVICES
            self._mueller_matrix = [
                                    # First line
                                    [1,
                                     0,
                                     0,
                                     0],
                                    # Second line
                                    [0,
                                     np.cos(delta)*np.sin(2*alpha)**2 +
                                     np.cos(2*alpha)**2,
                                     (1 - np.cos(delta)) *
                                     np.cos(2*alpha)*np.sin(2*alpha),
                                     np.sin(delta)*np.sin(2*alpha)],
                                    # Third line
                                    [0,
                                     (1 - np.cos(delta)) *
                                     np.cos(2*alpha)*np.sin(2*alpha),
                                     np.cos(delta)*np.cos(2*alpha)**2 +
                                     np.sin(2*alpha)**2,
                                     -np.sin(delta)*np.cos(2*alpha)],
                                    # Fourth line
                                    [0,
                                     -np.sin(delta)*np.sin(2*alpha),
                                     np.sin(delta)*np.cos(2*alpha),
                                     np.cos(delta)]]
        elif self._polar_type == 'rotator':
            self._mueller_matrix = [
                                    # First line
                                    [1,
                                     0,
                                     0,
                                     0],
                                    # Second line
                                    [0,
                                     1,
                                     0,
                                     0],
                                    # Third line
                                    [0,
                                     0,
                                     np.cos(alpha),
                                     -np.sin(alpha)],
                                    # Fourth line
                                    [0,
                                     0,
                                     np.sin(alpha),
                                     np.cos(alpha)]]

    # DISPLAY POLARIZER
    def disp(self):
        fig, ax = plt.subplots(1)
        plt.axis('equal')
        # IF LINEAR POLARIZER -> PLOT A SEGMENT ORIENTED LIKE THE POLARIZER
        if self._polar_type == 'linear':
            x1, y1 = ([1 - np.cos(self._angle), 1 + np.cos(self._angle)],
                      [1 - np.sin(self._angle), 1 + np.sin(self._angle)])
            plt.plot(x1, y1)
        # IF RETARDER POLARIZER -> PLOT TWO SEGMENTS: ONE ORIENTED ALONG THE
        # FAST AXIS ONE ORIENTED ALONG THE SLOW AXIS
        elif self._polar_type == 'retarder':
            ret = (self._phase_retardation % (2*np.pi))/(2*np.pi)
            x1, y1 = ([1-np.cos(self._angle), 1+np.cos(self._angle)],
                      [1 - np.sin(self._angle), 1 + np.sin(self._angle)])
            x2, y2 = ([1 - ret*np.cos(self._angle+np.pi/2),
                       1 + ret*np.cos(self._angle+np.pi/2)],
                      [1 - ret*np.sin(self._angle + np.pi/2),
                       1 + ret*np.sin(self._angle + np.pi/2)])
            plt.plot(x1, y1, x2, y2)
        # IF ROTATOR POLARIZER -> PLOT TWO SEGMENTS: ONE HORIZONTAL ONE
        # ORIENTED ALONG THE ANGLE OF ROTATION VALUE
        elif self._polar_type == 'rotator':
            x1, y1 = ([1, 2],
                      [1, 1])
            x2, y2 = ([1, 1 + np.cos(self._angle)],
                      [1, 1 + np.sin(self._angle)])
            plt.plot(x1, y1, x2, y2)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])

    def object_ID(self):
            return [self]

    '''####################################################################'''
    '''####################### PROPERTIES DEFINITION ######################'''
    '''####################################################################'''

    # DEFINE THE PROPERTIES
    polar_type = property(_get_polar_type, _set_polar_type, _del_polar_type,
                          "The 'type' property defines the type of polarizer. "
                          "It can be set as 'linear' / 'retarder' / "
                          "'rotator'.")
    angle = property(_get_angle, _set_angle, _del_angle,
                     "The 'angle' property defines the orientation of the "
                     "polarizer for linear and retardor types of polarizers. "
                     "It is expressed in radian. For rotator polarizer the "
                     "'angle' property defines the amount of rotation the "
                     "Jones vector undergoes.")
    phase_retardation = property(_get_phase_retardation,
                                 _set_phase_retardation,
                                 _del_phase_retardation)
    jones_matrix = property(_get_jones_matrix, _set_jones_matrix,
                            _del_jones_matrix)
    mueller_matrix = property(_get_mueller_matrix, _set_mueller_matrix,
                              _del_mueller_matrix)
