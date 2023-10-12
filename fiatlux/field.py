# -*- coding: utf-8 -*-
"""
 _______ _       _     _
(_______|_)     | |   | |
 _____   _ _____| | __| |
|  ___) | | ___ | |/ _  |
| |     | | ____| ( (_| |
|_|     |_|_____)\_)____|

Created on Fri Jun 15 15:19:33 2018

@author: pjanin
"""
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D
from .arrow3D import Arrow3D
from .polarizer import Polarizer
from .mirror import Mirror
from .mask import *
from .detector import Detector
from .propagator import Propagator
from .lens import Lens
from .pupil import Pupil
from .splitter import Separator
from .slm import SLM


class Field(object):

    """field_size
    scale: default is 4e-5
    wavelength: default is 550e-9
    verbose: default is False
    field_map: default is None
    incidence_angles: default is [0, 0]
    gaussian_center: default is 0
    gaussian_variance: default is 1
    stokes_parameters: default is None
    spherical_stokes_parameters: default is None

    Un champ est caractérisée par:
    - la taille de la carte [en pixel] ... (l'enfer en 256 points centré sur
                                            2x2 pixels)
    - l'échelle du pixel [m/pixel]
    - son chemin optique (historique des transformations du champ /
                          propagations / éléments optiqe)
    - ses cartes de champ complexe psi(x,y), tout le long du chemin optique,
    avec la polarisation (Horizontale / Verticale soient deux cartes de champ):
    - ses cartes d'amplitude (x,y) (attribut dependent) tout le long du
    chemin optique
    - ses cartes de phase (x,y) [rad] (attribut dependent) tout le long du
    chemin optique
    - sa longueur d'onde[m]
    - degré de polarisation horizontale/verticale (incertitude sur la
    représentation valable de la polarisation vis à vis du domaine
    d'approximation de la lumière)
    """

    """Ies méthodes :
        - carte de champ complexe de base:
        - onde plane à incidence oblique
        - gaussienne

        - propagation libre de Fresnel (distance de propagation z)
        - propagation libre de Rayleigh-Sommerfeld (RS)
        (distance de propagation z)

        - passage masque (x transparence complexe)

        - Opérateurs de propagation: (simplification des propagations
                                      physiques)
        - TF
        - Hartley
        - ...
    """

    '''####################################################################'''
    '''####################### INIT AND OVERLOADING #######################'''
    '''####################################################################'''

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):

        # FIELD_SIZE IS A MANDATORY PARAMETER
        self._field_size = field_size

        # SCALE SET TO 1 PX/M BY DEFAULT
        self._scale = kwargs.get('scale', 4e-5)

        # WAVELENGTH SET TO 550 NM BY DEFAULT
        self._wavelength = kwargs.get('wavelength', 550e-9)

        # DEFINE VERBOSE MODE
        self._verbose = kwargs.get('verbose', False)

        # COMPLEX AMPLITUDE INITIALIZATION
        self._complex_amplitude = np.ones((self._field_size, self.field_size,
                                          1), dtype=complex)
        # FIELD MAP BUILDING
        x = np.linspace(-self._field_size/2, self._field_size/2-1,
                        self._field_size)
        y = np.linspace(-self._field_size/2, self._field_size/2-1,
                        self._field_size)
        x_grid, y_grid = np.meshgrid(x, y)
        self._field_map = kwargs.get('field_map', None)
        # PLAN WAVE
        if self._field_map == 'plan_wave':
            self._incidence_angles = kwargs.get('incidence_angles', [0, 0])
            self._complex_amplitude[:, :, 0] = np.exp(1j*(
                    self._incidence_angles[0]*x_grid +
                    self._incidence_angles[1]*y_grid))
        # GAUSSIAN
        elif self._field_map == 'gaussian':
            self._center = kwargs.get('gaussian_center', [0,0])
            self._variance = kwargs.get('gaussian_variance', 1)
            self._complex_amplitude[:, :, 0] = np.exp(-(
                    (x_grid-self._center[0])**2 + (y_grid-self._center[1])**2) /
                (2*self._variance))
        else:
            self._complex_amplitude[:, :, 0] = np.ones((self._field_size,
                                                       self._field_size))

        # DEFINITION OF THE STOKES_PARAMETER OR ITS EQUIVALENT IN SPHERICAL
        # COORDINATES SPHERICAL_STOKES_PARAMETERS
        stokes_parameters = kwargs.get('stokes_parameters', None)
        spherical_stokes_parameters = kwargs.get(
                'spherical_stokes_parameters', None)
        if stokes_parameters is not None:
            self._stokes_parameters = stokes_parameters
            if spherical_stokes_parameters is not None:
                print("WARNING : spherical_stokes_parameters have been "
                      "overwritted by stokes_parameters !")
        elif spherical_stokes_parameters is not None:
            self._stokes_parameters = (
                    self._process_stokes_parameters(
                            spherical_stokes_parameters))
        else:
            # print("Default polarization => unpolarized light !")
            self._stokes_parameters = [1, 0, 0, 0]

        # OPTICAL PATH INITIALIZATION WITH FIELD AT PLAN 0
        self._optical_path_position = 0
        self._optical_path = []
        # F***ING DEEP COPY OF THE COMPLEX AMPLITUDE SO THAT [TMP] DOES NOT
        # BEHAVE LIKE A POINTER TO SELF._COMPLEX_AMPLITUDE
        tmp = copy.deepcopy(self.complex_amplitude)
        self._optical_path.append([self._optical_path_position] +
                                  self.object_ID() + [tmp])
        self._optical_path_position += 1

        # COMPLEX AMPLITUDE ALONG THE PATH
        self._stokes_parameters_path = np.zeros((1, 4, 1))
        self._stokes_parameters_path[0, :, 0] = self._stokes_parameters

    # DESTRUCTOR
    def __del__(self):
        # print("Deleted field")
        del(self)

    # REPRESENTERS
    def __str__(self):
        return ("<field>\n"
                "  * field_size : {} (px)\n"
                "  * scale : {} (px/m)\n"
                "  * wavelength : {} (m)"
                .format(self._field_size, self._scale, self._wavelength))

    # FIELD*VARARG - MAKES FIELD GOING THROUGH VARARG
    def __mul__(self, vararg):
        # IF VARARG IS A FIELD WE CANNOT DO MUCH ...
        if isinstance(vararg, Field):
            print('Field cannot go through field')
        # IF VARARG IS NOT A FIELD
        else:
            # IF VARARG IS A POLARIZER
            if isinstance(vararg, Polarizer):
                self._stokes_parameters = np.dot(vararg._mueller_matrix,
                                                 self._stokes_parameters)

                # IF VARARG IS A SEPARATOR
                if isinstance(vararg, Separator):
                    if vararg.mode == 'transmission':
                        if self._verbose is True:
                            print('Transmission through separator')
                    else:
                        if self._verbose is True:
                            print('Reflection on separator')

                # IF VARARG IS A MIRROR
                elif isinstance(vararg, Mirror):
                    self._complex_amplitude = (
                            np.fliplr(self._complex_amplitude))
                    if self._verbose is True:
                        print('Reflecting on mirror')

                # IF VARARG IS A SLM
                elif isinstance(vararg, SLM):
                    if self._verbose is True:
                        print('Reflecting on SLM')
                    for k in range(0, np.shape(self._complex_amplitude)[2]):
                        self._complex_amplitude[:, :, k] = (
                                np.multiply(self.complex_amplitude[:, :, k],
                                            vararg.complex_transparency))
                else:
                    if self._verbose is True:
                        print('Going through polarizer')

            # IF VARARG IS A MASK
            elif isinstance(vararg, Mask):
                # IF VARARG IS A PUPIL
                if isinstance(vararg, Pupil):
                    if self._verbose is True:
                        print('Going through pupil')
                    for k in range(0, np.shape(self._complex_amplitude)[2]):
                        self._complex_amplitude[:, :, k] = (
                                np.multiply(self._complex_amplitude[:, :, k],
                                            vararg.complex_transparency))

                # IF THE MASK IS A PYRAMID, A LOT HAS TO BE DONE TO PROCESS THE
                # MODULATION
                elif isinstance(vararg, Pyramid):
                    if self._verbose is True:
                        print('Going through pyramid')
                    for k in range(0, np.shape(self._complex_amplitude)[2]):
                        self._complex_amplitude[:, :, k] = (np.multiply(
                                self._complex_amplitude[:, :, k],
                                vararg.complex_transparency))

                # IF THE MASK IS A TIP-TILT MIRROR
                elif isinstance(vararg, TipTiltMirror):
                    # VERBOSE MODE
                    if self._verbose is True:
                        print('Reflecting on tip-tilt mirror')
                    tmp = np.zeros((self.field_size, self.field_size,
                                    np.shape(vararg._complex_transparency)[2]),
                                   dtype=complex)
                    for k in range(0,
                                   np.shape(vararg._complex_transparency)[2]):
                        tmp[:, :, k] = (
                                np.multiply(self._complex_amplitude[:, :, 0],
                                            (vararg.complex_transparency
                                             [:, :, k])))
                    self._complex_amplitude = tmp

                # IF VARARG IS ANOTHER KIND OF MASK
                else:
                    if self._verbose is True:
                        print('Going through mask')
                    for k in range(0, np.shape(self._complex_amplitude)[2]):
                        self._complex_amplitude[:, :, k] = (
                                np.multiply(self._complex_amplitude[:, :, k],
                                            vararg.complex_transparency))

            # IF VARARG IS A DETECTOR
            elif isinstance(vararg, Detector):
                if self._verbose is True:
                    print("Arriving on detector")
                vararg.complex_amplitude = self._complex_amplitude
                if vararg.display_intensity is True:
                    vararg.disp_intensity()

            # IF VARARG IS A PROPAGATOR
            elif isinstance(vararg, Propagator):
                if self._verbose is True:
                    print("Propagating following propagator")
                self._complex_amplitude = vararg.propagate(self)

            # IF VARARG IS A LENS
            elif isinstance(vararg, Lens):
                if self._verbose is True:
                    print('Going through lens')
                self._complex_amplitude = (
                        np.multiply(self.complex_amplitude,
                                    vararg.complex_transparency))

            # SAVE THE FIELD AND STOKES VECTOR AT CURRENT POSITION
            tmp = copy.deepcopy(self.complex_amplitude)
            self._optical_path.append([self._optical_path_position] +
                                      vararg.object_ID() +
                                      [tmp])
            self._optical_path_position += 1
            self._stokes_parameters_path = np.dstack(
                    (self._stokes_parameters_path, self._stokes_parameters))
        # RETURN THE FIELD
        return self

    # FIELD>VARARG - PROPAGATES THE FIELD BY A FOURIER TRANSFORM AND GO THROUGH
    # VARARG
    def __gt__(self, vararg):
        if self._verbose is True:
            print("Propagate field from pupil to focal plan")
        # DEFINE A NEW PROPAGATOR
        FFT = Propagator('FFT', self)
        # DO THE PROPAGATION
        self*FFT
        self._stokes_parameters_path = np.dstack(
                (self._stokes_parameters_path, self._stokes_parameters))
        # PROPAGATES THROUGH VARARG
        self*vararg
        return self

    # FIELD<VARARG - PROPAGATES THE FIELD BY AN INVERSE FOURIER TRANSFORM AND
    # GO THROUGH VARARG
    def __lt__(self, vararg):
        if self._verbose is True:
            print("Propagate field from focal to pupil plan")
        # DEFINE A NEW PROPAGATOR
        IFFT = Propagator('IFFT', self)
        # DO THE PROPAGATION
        self*IFFT
        # SAVE THE FIELD AND STOKES VECTOR AT CURRENT POSITION
#        tmp = np.zeros((self.field_size, self.field_size, 1),
#                       dtype=complex)
#        tmp[:, :, 0] = self._complex_amplitude
#        self._complex_amplitude_path = np.concatenate(
#                (self._complex_amplitude_path,
#                 tmp), axis=2)
        self._stokes_parameters_path = np.dstack(
                (self._stokes_parameters_path, self._stokes_parameters))
        # PROPAGATES THROUGH VARARG
        self*vararg
        return self

    # FIELD_1 + FIELD_2
    def __add__(self, vararg):
        self.complex_amplitude = (self.complex_amplitude +
                                  vararg.complex_amplitude)
        return self

    # FIELD_1 & FIELD_2
    def __and__(self, vararg):
        self.complex_amplitude = (self.complex_amplitude +
                                  vararg.complex_amplitude)
        return self

    # +FIELD
    def __pos__(self):
        size = np.shape(self._optical_path)[0]
        f = copy.deepcopy(self@0)
        for k in range(1, size):
            f*self._optical_path[k][1]
        self._complex_amplitude = f._complex_amplitude
        self._stokes_parameters = f._stokes_parameters
        return f

    # FIELD@VARARG - RETURN THE FIELD AT POSITION VARARG
    def __matmul__(self, vararg):
        # CREATE A NEW FIELD INSTANCE (FIELD = SELF MAKES FIELD==SELF RETURNING
        # TRUE MAKING IT THE SAME INSTANCE? I.E. NOT WHAT WE NEED)
        field = Field(self._field_size, scale=self._scale,
                      wavelength=self._wavelength)
        # REASIGN THE FIELD MAP PARAMETERS TO FIELD
        if self._field_map == 'plan_wave':
            field._incidence_angles = self._incidence_angles
        elif self._field_map == 'gaussian':
            field._center = self._center
            field._variance = self._variance
        # ASSIGN THE VARARG-TH COMPLEXE AMPLITUDE PATH TO FIELD
        field._complex_amplitude = (
                copy.deepcopy(self._optical_path[vararg][2][:, :, :]))
        # OPTICAL PATH
        field._optical_path_position = 0
        field._optical_path = []
        field._optical_path.append([field._optical_path_position] +
                                   field.object_ID() +
                                   [field._complex_amplitude])
        field._optical_path_position += 1
        # ASSIGN THE VARARG-TH STOKES PARAMETERS PATH TO FIELD
        field._stokes_parameters = copy.deepcopy(
                self._stokes_parameters_path[0, :, vararg])
        field._stokes_parameters_path = field._stokes_parameters
        return field

    '''####################################################################'''
    '''####################### GET / SET DEFINITION #######################'''
    '''####################################################################'''

    # GET/SET FIELD_SIZE
    def _get_field_size(self):
        # print("Field size is {} px".format(self._field_size))
        return self._field_size

    def _set_field_size(self, field_size):
        print("Immutable property. Please create a new field instance.")

    def _del_field_size(self):
        print("Undeletable property")

    # GET/SET SCALE
    def _get_scale(self):
        # print("Scale is {} px/m".format(self._scale))
        return self._scale

    def _set_scale(self, scale):
        self._scale = scale
        # print("Scale changed to {}".format(self._scale))

    def _del_scale(self):
        print("Undeletable property")

    # GET/SET WAVELENGTH
    def _get_wavelength(self):
        # print("Wavelength is {} m".format(self._wavelength))
        return self._wavelength

    def _set_wavelength(self, wavelength):
        self._wavelength = wavelength
        # print("Wavelength changed to {}".format(self._wavelength))

    def _del_wavelength(self):
        print("Undeletable property")

    # GET/SET COMPLEX_AMPLITUDE
    def _get_complex_amplitude(self):
        return self._complex_amplitude

    def _set_complex_amplitude(self, complex_amplitude):
        print(np.shape(complex_amplitude))
        if isinstance(complex_amplitude, np.ndarray):
            self._complex_amplitude = complex_amplitude
            # MODIFY THE VALUE OF THE COMPLEX AMPLITUDE IN THE OPTICAL PATH
            self._optical_path[self._optical_path_position - 1][2][:, :, :] = (
                    complex_amplitude)
            # print("Complex amplitude changed to {}"
            #       .format(self._complex_amplitude))
        else:
            print("Complex amplitude is not of the right format or does not "
                  "have the right shape ({}x{}x{}px)"
                  .format(self._field_size, self._field_size))

    def _del_complex_amplitude(self):
        print("Undeletable property")

    # GET/SET AMPLITUDE
    def _get_amplitude(self):
        return np.abs(self._complex_amplitude)

    def _set_amplitude(self, amplitude):
        if isinstance(amplitude, np.ndarray) and (amplitude.shape ==
                                                  (self._field_size,
                                                   self._field_size)):
            phase = np.angle(self._complex_amplitude)
            self._complex_amplitude = amplitude*np.exp(1j*phase)
            # print("Amplitude changed to {}"
            #       .format(np.abs(self._complex_amplitude)))
        else:
            print("Amplitude is not of the right format or does not have the "
                  "right shape ({}x{}x{}px)".format(self._field_size,
                                                    self._field_size))

    def _del_amplitude(self):
        print("Undeletable property")

    # GET/SET PHASE
    def _get_phase(self):
        return np.angle(self._complex_amplitude)

    def _set_phase(self, phase):
        if isinstance(phase, np.ndarray) and (phase.shape ==
                                              (self._field_size,
                                               self._field_size)):
            amplitude = np.abs(self._complex_amplitude)
            self._complex_amplitude = amplitude*np.exp(1j*phase)
            # print("Phase changed to {}"
            #       .format(np.angle(self._complex_amplitude)))
        else:
            print("Phase is not of the right format or does not have the "
                  "right shape ({}x{}x{}px)".format(self._field_size,
                                                    self._field_size))

    def _del_phase(self):
        print("Undeletable property")

    # GET/SET STOKES_PARAMETERS
    def _get_stokes_parameters(self):
        # print("Stokes parameters are {}".format(self._stokes_parameters))
        return self._stokes_parameters

    def _set_stokes_parameters(self, stokes_parameters):
        self._stokes_parameters = stokes_parameters
        # print("Stokes parameters changed to {}"
        #       .format(self._stokes_parameters))

    def _del_stokes_parameters(self):
        print("Undeletable property")

    # GET/SET SPHERICAL_STOKES_PARAMETERS
    def _get_spherical_stokes_parameters(self):
        spherical_stokes_parameter = (
                self._process_spherical_stokes_parameters(
                        self._stokes_parameters))
        # print("Spherical Stokes parameters are {}"
        #       .format(spherical_stokes_parameter))
        return spherical_stokes_parameter

    def _set_spherical_stokes_parameters(self, spherical_stokes_parameters):
        self._stokes_parameters = (
                self._process_stokes_parameters(spherical_stokes_parameters))
        # print("Spherical Stokes parameters changed to {}"
        #       .format(spherical_stokes_parameters))

    def _del_spherical_stokes_parameters(self):
        print("Undeletable property")

    # GET/SET OPTICAL PATH
    def _get_optical_path(self):
        return self._optical_path

    def _set_optical_path(self, optical_path):
        print("Unsetable property for now on ...")

    def _del_optical_path(self):
        print("Undeletable property")

    '''####################################################################'''
    '''####################### FUNCTIONS DEFINITION #######################'''
    '''####################################################################'''

    # REPRESENT THE OPTICAL PATH
    def disp_optical_path(self):
        for k in range(0, np.shape(self.optical_path)[0]):
            print(self.optical_path[k][0:2])

    # REPRESENT THE POLARIZATION STATE ON POINCARE SPHERE
    def disp_polarization(self):

        # GET THE SPHERICAL POLARIZATION COORDINATES
        I0, p, Psi, Chi = self._process_spherical_stokes_parameters(
                self._stokes_parameters)
        Psi = Psi/2
        Chi = Chi/2

        # GET THE COOLEST COLORMAP
        cmap = plt.get_cmap("tab10")

        """
        DEFINING ALL THE NECESSARY PARAMETERS
        """

        # DEFINE FIGURE AND AXIS
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # MAKE DATA FOR (U,V) SPHERE
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # DEFINE THE UNITARY VECTORS (R = 1.25 SO THE TIP OF THE ARROW GOES OUT
        #                             OUT THE SPHERE)
        x_arrow = Arrow3D([0, 1.25], [0, 0],
                          [0, 0], mutation_scale=10,
                          lw=2, arrowstyle="-|>", color=cmap(3))
        y_arrow = Arrow3D([0, 0], [0, 1.25],
                          [0, 0], mutation_scale=10,
                          lw=2, arrowstyle="-|>", color=cmap(3))
        z_arrow = Arrow3D([0, 0], [0, 0],
                          [0, 1.25], mutation_scale=10,
                          lw=2, arrowstyle="-|>", color=cmap(3))

        # DEFINE THE POLARIZATION ARROW
        p_arrow = Arrow3D([0, p*np.cos(2*Psi)*np.cos(2*Chi)],
                          [0, p*np.sin(2*Psi)*np.cos(2*Chi)],
                          [0, p*np.sin(2*Chi)], mutation_scale=10,
                          lw=4, arrowstyle="-|>", color=cmap(2))

        # PROJECTION OF THE POLARIZATION ON THE (X,Y) PLAN
        x0 = [0, p*np.cos(2*Psi)*np.cos(2*Chi)]
        y0 = [0, p*np.sin(2*Psi)*np.cos(2*Chi)]
        z0 = [0, 0]

        # PROJECTION OF THE POLARIZATION ON THE (Y,Z) PLAN
        x1 = [p*np.cos(2*Psi)*np.cos(2*Chi), p*np.cos(2*Psi)*np.cos(2*Chi)]
        y1 = [p*np.sin(2*Psi)*np.cos(2*Chi), p*np.sin(2*Psi)*np.cos(2*Chi)]
        z1 = [0, p*np.sin(2*Chi)]

        """
        PLOTING
        """

        # PLOT THE SPHERE OF RADIUS 1
        ax.plot_surface(x, y, z, color=cmap(0), alpha=0.1)

        # PLOT THE CIRCLE CONTOUR LINES OF THE SPHERE OF RADIUS 1
        ax.plot(np.cos(u), np.sin(u), color=cmap(0))
        ax.plot(np.cos(-np.pi/4)*np.cos(u), np.sin(-np.pi/4)*np.cos(u),
                np.sin(u), color=cmap(0))

        # PLOT THE UNITARY ARROWS FOR AXIS
        ax.add_artist(x_arrow)
        ax.add_artist(y_arrow)
        ax.add_artist(z_arrow)

        # PLOT THE CIRCLE CONTOUR LINES OF THE SPHERE OF RADIUS R
        ax.plot(p*np.cos(u), p*np.sin(u), color=cmap(1), linestyle=':')
        ax.plot(p*np.cos(-np.pi/4)*np.cos(u), p*np.sin(-np.pi/4)*np.cos(u),
                p*np.sin(u), color=cmap(1), linestyle=':')

        # PLOT THE POLARIZATION ARROW
        ax.add_artist(p_arrow)

        # PLOT OF THE PROJECTIONS
        ax.plot(x0, y0, z0, color=cmap(4), linestyle=':')
        ax.plot(x1, y1, z1, color=cmap(4), linestyle=':')

        # SET AND SHOW PLOTS
        plt.axis('scaled')
        plt.title('p = {:.3g}\n'
                  '\u03A8 = {:.3g} rad\n'
                  '\u03C7 = {:.3g} rad'.format(p, Psi, Chi))
        ax.view_init(20, 45)
        plt.show()

    # REPRESENT THE AMPLITUDE AND PHASE OF THE FIELD
    def disp_complex_amplitude(self):
        plt.figure()
        plt.subplot(121)
        plt.imshow(np.abs(self._complex_amplitude[:, :, 0]))
        plt.title('Amplitude')
        plt.subplot(122)
        plt.imshow(np.angle(self._complex_amplitude[:, :, 0]))
        plt.title('Phase (in rad)')
        plt.suptitle('Field complex amplitude')
        plt.show()

    # PROCESS STOKES_PARAMETERS
    def _process_stokes_parameters(self, spherical_stokes_parameters):
        I0, p, Psi, Chi = spherical_stokes_parameters
        Psi = Psi/2
        Chi = Chi/2
        S0 = I0
        S1 = I0*p*np.cos(2*Psi)*np.cos(2*Chi)
        S2 = I0*p*np.sin(2*Psi)*np.cos(2*Chi)
        S3 = I0*p*np.sin(2*Chi)
        stokes_parameters = [S0, S1, S2, S3]
        return stokes_parameters

    # PROCESS SPHERICAL_STOKES_PARAMETERS
    def _process_spherical_stokes_parameters(self, stokes_parameters):
        # GET THE STOKES PARAMETER FROM SELF
        S0, S1, S2, S3 = stokes_parameters
        # TRANSPOSE IT TO SPHERICAL COORDINATES
        I0 = S0
        p = np.sqrt(S1**2 + S2**2 + S3**2)/S0
        # CALCULATE PSI
        if S1 == 0:
            if S2 == 0:
                Psi = 0.5*np.arctan(1)
            else:
                Psi = 0.5*np.sign(S2)*np.pi/2
        else:
            if S1 >= 0:
                if S2 >= 0:
                    Psi = 0.5*np.arctan(S2/S1)
                else:
                    Psi = 0.5*(2*np.pi - np.arctan(-S2/S1))
            else:
                if S2 >= 0:
                    Psi = 0.5*(np.pi - np.arctan(-S2/S1))
                else:
                    Psi = 0.5*(np.pi + np.arctan(S2/S1))
        # CALCULATE CHI
        if np.sqrt(S1**2+S2**2) == 0:
            if S3 == 0:
                Chi = 0.5*np.arctan(1)
            else:
                Chi = 0.5*np.sign(S3)*np.pi/2
        else:
            Chi = 0.5*np.arctan(S3/np.sqrt(S1**2 + S2**2))
        spherical_stokes_parameters = [I0, p, 2*Psi, 2*Chi]
        return spherical_stokes_parameters

    def object_ID(self):
        return [self]

    '''####################################################################'''
    '''####################### PROPERTIES DEFINITION ######################'''
    '''####################################################################'''

    # DEFINE THE PROPERTIES
    field_size = property(_get_field_size, _set_field_size, _del_field_size,
                          "The 'field_size' property defines the size of the "
                          "matrix containing the field data.")

    scale = property(_get_scale, _set_scale, _del_scale,
                     "The 'scale' property defines the correspondance "
                     "between pixels units (in px) and physical distance "
                     "(in m)")

    wavelength = property(_get_wavelength, _set_wavelength, _del_wavelength,
                          "The 'wavelength' property defines the wavelength "
                          "of the wave (in m)")

    complex_amplitude = property(_get_complex_amplitude,
                                 _set_complex_amplitude,
                                 _del_complex_amplitude,
                                 "The 'amplitude' property defines the "
                                 "amplitude of the wave (in ???)")

    amplitude = property(_get_amplitude, _set_amplitude, _del_amplitude,
                         "The 'amplitude' property defines the amplitude of "
                         "the wave (in ???)")

    phase = property(_get_phase, _set_phase, _del_phase,
                     "The 'phase' property defines the phase of "
                     "the wave (in rad)")

    stokes_parameters = property(_get_stokes_parameters,
                                 _set_stokes_parameters,
                                 _del_stokes_parameters,
                                 "The 'stokes_parameters' property "
                                 "defines the four Stokes parameter "
                                 "necessary to describe the polarization")

    spherical_stokes_parameters = property(_get_spherical_stokes_parameters,
                                           _set_spherical_stokes_parameters,
                                           _del_spherical_stokes_parameters,
                                           "The 'spherical_stokes_parameters' "
                                           "property defines the four Stokes "
                                           "parameter necessary to describe "
                                           "the polarization in spherical "
                                           "coordinates")

    optical_path = property(_get_optical_path, _set_optical_path,
                            _del_optical_path, "The 'optical_path' property"
                            "define the optical path undergone by the field")
