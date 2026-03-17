# -*- coding: utf-8 -*-
"""
 _______            _
(_______)          | |
 _  _  _ _____  ___| |  _
| ||_|| (____ |/___) |_/ )
| |   | / ___ |___ |  _ (
|_|   |_\_____(___/|_| \_)

Created on Fri Jun 15 15:19:33 2018

@author: pjanin
"""

import numpy as np
import scipy as sc
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .arrow3D import Arrow3D
from .polarizer import Polarizer
import time


class Mask(object):

    """####################################################################"""

    """####################### INIT AND OVERLOADING #######################"""
    """####################################################################"""

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):
        self._field_size = field_size
        # FIELD_SIZE IS A MANDATORY PARAMETER
        self._complex_transparency = np.zeros((self._field_size, self._field_size))

    # DESTRUCTOR
    def __del__(self):
        del self

    """####################################################################"""
    """####################### GET / SET DEFINITION #######################"""
    """####################################################################"""

    # GET/SET JONES COMPLEX TRANSPARENCY
    def _get_complex_transparency(self):
        # print("Complex transparency is {}"
        #       .format(self._complex_transparency))
        return self._complex_transparency

    def _set_complex_transparency(self, complex_transparency):
        self._complex_transparency = complex_transparency

    def _del_complex_transparency(self):
        print("Undeletable type property")

    """####################################################################"""
    """####################### FUNCTIONS DEFINITION #######################"""
    """####################################################################"""

    def object_ID(self):
        """
        M.object_ID() : return the object's identifier.

        Returns
        -------
        [M]
        """
        return [self]

    """####################################################################"""
    """####################### PROPERTIES DEFINITION ######################"""
    """####################################################################"""

    complex_transparency = property(
        _get_complex_transparency,
        _set_complex_transparency,
        _del_complex_transparency,
        "complex transparency of the " "mask",
    )


class Blank(Mask):

    """####################################################################"""

    """####################### INIT AND OVERLOADING #######################"""
    """####################################################################"""

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):
        Mask.__init__(self, field_size)
        self._build_blank()

    """####################################################################"""
    """####################### GET / SET DEFINITION #######################"""
    """####################################################################"""

    """####################################################################"""
    """####################### FUNCTIONS DEFINITION #######################"""
    """####################################################################"""

    def _build_blank(self):
        """
        M._build_blank() : changes M.complex_transparency to a blank mask
        (i.e. complex_transparency = 1).

        Returns
        -------
        Mask object with modified complex_transparency
        """
        self._complex_transparency = np.ones(
            (self._field_size, self._field_size), dtype=complex
        )

    """####################################################################"""
    """####################### PROPERTIES DEFINITION ######################"""
    """####################################################################"""


class Fourier(Mask):

    """####################################################################"""

    """####################### INIT AND OVERLOADING #######################"""
    """####################################################################"""

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):
        Mask.__init__(self, field_size)
        self._n_modes = kwargs.get("n_modes", field_size // 4)
        self._modes = np.zeros((field_size, field_size, (self._n_modes + 1) ** 2))
        self._coefficients = kwargs.get("coefficients", np.ones(self._n_modes))
        self._pupil = kwargs.get("pupil", np.ones((field_size, field_size)))
        self._build_modes()
        self._build_fourier()

    """####################################################################"""
    """####################### GET / SET DEFINITION #######################"""
    """####################################################################"""

    # GET/SET CHARGE
    def _get_coefficients(self):
        return self._coefficients

    def _set_coefficients(self, coefficients):
        self._coefficients = coefficients
        self._build_fourier()

    def _del_coefficients(self):
        print("Undeletable type property")

    """####################################################################"""
    """####################### FUNCTIONS DEFINITION #######################"""
    """####################################################################"""

    def _build_fourier(self):
        """
        M.Vortex(charge=C) : changes M.complex_transparency to a vortex mask
        with charge C.

        Parameters
        ----------
        charge : int
            Charge of the vortex mask.

        Returns
        -------
        Mask object with modified complex_transparency
        """
        coefficients = self._coefficients
        fourier = np.zeros((self._field_size, self._field_size))
        for i in range(0, len(coefficients)):
            fourier = fourier + coefficients[i] * self._modes[:, :, i]
        self._complex_transparency = np.exp(1j * fourier)

    def _build_modes(self):
        """
        M.FQPM() : changes M.complex_transparency to a Four Quadrant Phase
        Mask.

        Returns
        -------
        Mask object with modified complex_transparency
        """
        N = self._field_size
        idx = 0
        for k in range(0, self._n_modes):
            for l in range(0, self._n_modes):
                tmp = np.zeros([N, N])
                tmp[N // 2 - 4 * k, N // 2 - 4 * l] = 1 / 2
                tmp[N // 2 + 4 * k, N // 2 + 4 * l] = 1 / 2
                self._modes[:, :, idx] = np.real(np.fft.fft2(np.fft.fftshift(tmp)))
                idx += 1

    """####################################################################"""
    """####################### PROPERTIES DEFINITION ######################"""
    """####################################################################"""

    coefficients = property(
        _get_coefficients,
        _set_coefficients,
        _del_coefficients,
        "shape of the mask. None if no defined shape",
    )


class FQPM(Mask):

    """####################################################################"""

    """####################### INIT AND OVERLOADING #######################"""
    """####################################################################"""

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):
        Mask.__init__(self, field_size)
        self._shift = kwargs.get("shift", np.pi)
        self._build_FQPM()

    """####################################################################"""
    """####################### GET / SET DEFINITION #######################"""
    """####################################################################"""

    """####################################################################"""
    """####################### FUNCTIONS DEFINITION #######################"""
    """####################################################################"""

    def _build_FQPM(self):
        """
        M.FQPM() : changes M.complex_transparency to a Four Quadrant Phase
        Mask.

        Returns
        -------
        Mask object with modified complex_transparency
        """
        N = self._field_size
        phi = self._shift
        FQPM = np.ones((N, N), dtype=complex)
        FQPM[0 : N // 2, 0 : N // 2] = np.exp(1j * phi)
        FQPM[N // 2 : N, N // 2 : N] = np.exp(1j * phi)
        self._complex_transparency = FQPM

    """####################################################################"""
    """####################### PROPERTIES DEFINITION ######################"""
    """####################################################################"""


class Pyramid(Mask):

    """####################################################################"""

    """####################### INIT AND OVERLOADING #######################"""
    """####################################################################"""

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):
        Mask.__init__(self, field_size)
        # PYRAMID ANGLE
        self._angle = kwargs.get("angle", self._field_size)
        # PYRAMID FACES
        self._faces = kwargs.get("faces", 4)
        # Pyramid ORIENTATION
        self._orientation = kwargs.get("orientation", 0)
        self._modulation = kwargs.get("modulation", 0)
        if self._modulation is True:
            self._modulation_shape = kwargs.get("modulation_shape", None)
            self._modulation_path = kwargs.get("modulation_path", None)
            if self._modulation_shape == "circular":
                self._modulation_radius = kwargs.get(
                    "modulation_radius", field_size / 4
                )
                self._x_mod = self._modulation_radius * np.cos(
                    np.linspace(0, 2 * np.pi, 100)
                )
                self._y_mod = self._modulation_radius * np.sin(
                    np.linspace(0, 2 * np.pi, 100)
                )
            else:
                if self._modulation_path is not None:
                    self._x_mod = self._modulation_path[0]
                    self._y_mod = self._modulation_path[1]

        self._tip_tilt_mirror = kwargs.get("tip_tilt_mirror", None)
        self._tip_tilt_mirror_position = 0
        self._tip_tilt_mirror_position_init = False
        self._pyramid_position = 0
        # BUILD PYRAMID
        self._build_pyramid()

    """####################################################################"""
    """####################### GET / SET DEFINITION #######################"""
    """####################################################################"""

    # GET/SET ANGLE
    def _get_angle(self):
        return self._angle

    def _set_angle(self, angle):
        self._angle = angle
        self._build_pyramid()

    def _del_angle(self):
        print("Undeletable type property")

    # GET/SET FACES
    def _get_faces(self):
        return self._faces

    def _set_faces(self, faces):
        self._faces = faces
        self._build_pyramid()

    def _del_faces(self):
        print("Undeletable type property")

    # GET/SET MODULATION
    def _get_modulation(self):
        return self._modulation

    def _set_modulation(self, modulation):
        self._modulation = modulation

    def _del_modulation(self):
        print("Undeletable type property")

    # GET/SET MODULATION
    def _get_orientation(self):
        return self._orientation

    def _set_orientation(self, orientation):
        self._orientation = orientation
        self._build_pyramid()

    def _del_orientation(self):
        print("Undeletable type property")

    """####################################################################"""
    """####################### FUNCTIONS DEFINITION #######################"""
    """####################################################################"""

    def _build_pyramid(self):
        """
        P._build_pyramid() : changes P.complex_transparency to a
        pyramid mask with self._faces faces oriented at self._angle angle.

        Parameters
        ----------
        angle : float
            Orientation of the faces.
        faces : int
            Number of faces comprising the mask.

        Returns
        -------
        Mask object with modified complex_transparency
        """
        # GET PARAMETERS
        pyramid_angle = self._angle
        pyramid_faces = self._faces
        pyramid_orientation = self._orientation
        N = self._field_size
        # DEFINE X AND Y GRIDS
        x = np.linspace(-N // 2 + 0.5, N // 2 - 0.5, N)
        y = np.linspace(-N // 2 + 0.5, N // 2 - 0.5, N)
        x_grid, y_grid = np.meshgrid(x, y)
        # DEFINE THE SLOPE GRID
        angle_grid = np.arctan2(
            x_grid * np.sin(-pyramid_orientation)
            + y_grid * np.cos(-pyramid_orientation),
            x_grid * np.cos(-pyramid_orientation)
            - y_grid * np.sin(-pyramid_orientation),
        )
        # angle_grid = np.arctan2(y_grid,x_grid)
        # INITIALIZE PYRAMID MASK
        Pyramid = np.zeros((N, N))
        for i in range(pyramid_faces):
            theta = (
                np.pi * (1.0 / pyramid_faces - 1)
                + i * 2 * np.pi / pyramid_faces
                + pyramid_orientation
            )
            slope = np.cos(theta) * x_grid + np.sin(theta) * y_grid
            slope[
                (-np.pi + i * 2 * np.pi / pyramid_faces > angle_grid)
                | (angle_grid > (-np.pi + (i + 1) * 2 * np.pi / pyramid_faces))
            ] = 0
            Pyramid = Pyramid + pyramid_angle * slope
        Pyramid = np.exp(-1j * np.pi * Pyramid / N)
        self._complex_transparency = Pyramid
        return self

    """####################################################################"""
    """####################### PROPERTIES DEFINITION ######################"""
    """####################################################################"""

    angle = property(
        _get_angle,
        _set_angle,
        _del_angle,
        "shape of the mask. " "None if no defined shape",
    )
    faces = property(
        _get_faces,
        _set_faces,
        _del_faces,
        "shape of the mask. " "None if no defined shape",
    )
    modulation = property(
        _get_modulation,
        _set_modulation,
        _del_modulation,
        "shape of the mask. None if no defined shape",
    )
    orientation = property(
        _get_orientation,
        _set_orientation,
        _del_orientation,
        "shape of the mask. None if no defined shape",
    )


class Tilt(Mask):

    """####################################################################"""

    """####################### INIT AND OVERLOADING #######################"""
    """####################################################################"""

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):
        Mask.__init__(self, field_size)
        self._angles = kwargs.get("angles", [-1, -1])
        self._build_tilt()

    """####################################################################"""
    """####################### GET / SET DEFINITION #######################"""
    """####################################################################"""

    # GET/SET CHARGE
    def _get_angles(self):
        return self._angles

    def _set_angles(self, angles):
        self._angles = angles
        self._build_tilt()

    def _del_angles(self):
        print("Undeletable type property")

    """####################################################################"""
    """####################### FUNCTIONS DEFINITION #######################"""
    """####################################################################"""

    def _build_tilt(self):
        """
        M.Tilt(angles=[t, T]) : changes M.complex_transparency to a tip-tilt
        mask with tip t and tilt T.

        Parameters
        ----------
        angles : [float, float]
            Tip and tilt angles.

        Returns
        -------
        Mask object with modified complex_transparency
        """
        tip, tilt = self._angles
        N = self._field_size
        x = np.linspace(-N // 2 + 0.5, N // 2 - 0.5, N)
        y = np.linspace(-N // 2 + 0.5, N // 2 - 0.5, N)
        x_grid, y_grid = np.meshgrid(x, y)
        self._complex_transparency = np.exp(
            1j * np.pi * (tip * x_grid + tilt * y_grid) / N
        )

    """####################################################################"""
    """####################### PROPERTIES DEFINITION ######################"""
    """####################################################################"""

    angles = property(
        _get_angles,
        _set_angles,
        _del_angles,
        "shape of the " "mask. None if no defined shape",
    )


class TipTiltMirror(Mask):

    """####################################################################"""

    """####################### INIT AND OVERLOADING #######################"""
    """####################################################################"""

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):
        Mask.__init__(self, field_size)
        self._angles = kwargs.get("angles", [np.zeros((1)), np.zeros((1))])
        self._sampling = 100
        self._modulation_shape = kwargs.get("modulation_shape", None)
        self._modulation_size = kwargs.get("modulation_size", field_size / 4)
        if self._modulation_shape == "circular":
            x_mod = self._modulation_size * np.cos(
                np.linspace(0, 2 * np.pi, self._sampling)
            )
            y_mod = self._modulation_size * np.sin(
                np.linspace(0, 2 * np.pi, self._sampling)
            )
            self._angles = [x_mod, y_mod]
        elif self._modulation_shape == "square":
            x_mod = np.zeros((self._sampling))
            y_mod = np.zeros((self._sampling))
            r = self._modulation_size
            s = self._sampling // 4
            x_mod[0:s] = np.linspace(-r, r, s)
            x_mod[s : 2 * s] = np.linspace(r, r, s)
            x_mod[2 * s : 3 * s] = np.linspace(r, -r, s)
            x_mod[3 * s : 4 * s] = np.linspace(-r, -r, s)
            y_mod[0:s] = np.linspace(-r, -r, s)
            y_mod[s : 2 * s] = np.linspace(-r, r, s)
            y_mod[2 * s : 3 * s] = np.linspace(r, r, s)
            y_mod[3 * s : 4 * s] = np.linspace(r, -r, s)
            self._angles = [x_mod, y_mod]
        self._build_tip_tilt_mirror()

    """####################################################################"""
    """####################### GET / SET DEFINITION #######################"""
    """####################################################################"""

    # GET/SET ANGLE
    def _get_angles(self):
        return self._angles

    def _set_angles(self, angles):
        self._angles = angles
        self._build_tip_tilt_mirror()

    def _del_angles(self):
        print("Undeletable type property")

    # GET/SET MODULATION SHAPE
    def _get_modulation_shape(self):
        return self._modulation_shape

    def _set_modulation_shape(self, modulation_shape):
        self._modulation_shape = modulation_shape
        if self._modulation_shape == "circular":
            self._build_circular_modulation()
        elif self._modulation_shape == "square":
            self._build_square_modulation()
        self._build_tip_tilt_mirror()

    def _del_modulation_shape(self):
        print("Undeletable type property")

    # GET/SET MODULATION SIZE
    def _get_modulation_size(self):
        return self._modulation_size

    def _set_modulation_size(self, modulation_size):
        self._modulation_size = modulation_size
        if self._modulation_shape == "circular":
            self._build_circular_modulation()
        elif self._modulation_shape == "square":
            self._build_square_modulation()
        self._build_tip_tilt_mirror()

    def _del_modulation_size(self):
        print("Undeletable type property")

    """####################################################################"""
    """####################### FUNCTIONS DEFINITION #######################"""
    """####################################################################"""

    def _build_square_modulation(self):
        x_mod = np.zeros((self._sampling))
        y_mod = np.zeros((self._sampling))
        r = self._modulation_size
        s = self._sampling // 4
        x_mod[0:s] = np.linspace(-r, r, s)
        x_mod[s : 2 * s] = np.linspace(r, r, s)
        x_mod[2 * s : 3 * s] = np.linspace(r, -r, s)
        x_mod[3 * s : 4 * s] = np.linspace(-r, -r, s)
        y_mod[0:s] = np.linspace(-r, -r, s)
        y_mod[s : 2 * s] = np.linspace(-r, r, s)
        y_mod[2 * s : 3 * s] = np.linspace(r, r, s)
        y_mod[3 * s : 4 * s] = np.linspace(r, -r, s)
        self._angles = [x_mod, y_mod]

    def _build_circular_modulation(self):
        x_mod = self._modulation_size * np.cos(
            np.linspace(0, 2 * np.pi, self._sampling)
        )
        y_mod = self._modulation_size * np.sin(
            np.linspace(0, 2 * np.pi, self._sampling)
        )
        self._angles = [x_mod, y_mod]

    def _build_tip_tilt_mirror(self):
        """
        M.Tilt(angles=[t, T]) : changes M.complex_transparency to a tip-tilt
        mask with tip t and tilt T.

        Parameters
        ----------
        angles : [float, float]
            Tip and tilt angles.

        Returns
        -------
        Mask object with modified complex_transparency
        """
        [tip, tilt] = self._angles
        #        tip = tip
        #        tilt = tilt
        length = np.shape(tip)[0]
        N = self._field_size
        tmp = np.zeros((N, N, length), dtype=complex)
        x = np.linspace(-N // 2 + 0.5, N // 2 - 0.5, N)
        y = np.linspace(-N // 2 + 0.5, N // 2 - 0.5, N)
        x_grid, y_grid = np.meshgrid(x, y)
        for k in range(0, length):
            tmp[:, :, k] = np.exp(1j * np.pi * (tip[k] * x_grid + tilt[k] * y_grid) / N)
        self._complex_transparency = tmp

    def disp_modulation_path(self):
        plt.scatter(self.angles[0], self.angles[1])

    """####################################################################"""
    """####################### PROPERTIES DEFINITION ######################"""
    """####################################################################"""

    angles = property(
        _get_angles,
        _set_angles,
        _del_angles,
        "shape of the " "mask. None if no defined shape",
    )
    modulation_shape = property(
        _get_modulation_shape,
        _set_modulation_shape,
        _del_modulation_shape,
        "shape of the mask. None if no defined shape",
    )
    modulation_size = property(
        _get_modulation_size,
        _set_modulation_size,
        _del_modulation_size,
        "shape of the mask. None if no defined shape",
    )


class Vortex(Mask):

    """####################################################################"""

    """####################### INIT AND OVERLOADING #######################"""
    """####################################################################"""

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):
        Mask.__init__(self, field_size)
        self._charge = kwargs.get("charge", 2)
        self._build_vortex()

    """####################################################################"""
    """####################### GET / SET DEFINITION #######################"""
    """####################################################################"""

    # GET/SET CHARGE
    def _get_charge(self):
        return self._charge

    def _set_charge(self, charge):
        self._charge = charge
        self._build_vortex()

    def _del_charge(self):
        print("Undeletable type property")

    """####################################################################"""
    """####################### FUNCTIONS DEFINITION #######################"""
    """####################################################################"""

    def _build_vortex(self):
        """
        M.Vortex(charge=C) : changes M.complex_transparency to a vortex mask
        with charge C.

        Parameters
        ----------
        charge : int
            Charge of the vortex mask.

        Returns
        -------
        Mask object with modified complex_transparency
        """
        charge = self._charge
        N = self._field_size
        x = np.linspace(-N // 2 + 0.5, N // 2 - 0.5, N)
        y = np.linspace(-N // 2 + 0.5, N // 2 - 0.5, N)
        x_grid, y_grid = np.meshgrid(x, y)
        Vortex = np.exp(1j * charge * np.arctan2(y_grid, x_grid))
        self._complex_transparency = Vortex
        return self


class Zelda(Mask):

    """####################################################################"""

    """####################### INIT AND OVERLOADING #######################"""
    """####################################################################"""

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):
        Mask.__init__(self, field_size)
        self._patch_size = kwargs.get("patch_size", 4)
        self._patch_shift = kwargs.get("patch_shift", np.pi / 2)
        self._build_zelda()

    """####################################################################"""
    """####################### GET / SET DEFINITION #######################"""
    """####################################################################"""

    # GET/SET PATCH SIZE
    def _get_patch_size(self):
        return self._patch_size

    def _set_patch_size(self, patch_size):
        self._patch_size = patch_size
        self._build_zelda()

    def _del_patch_size(self):
        print("Undeletable type property")

    # GET/SET PATCH SHIFT
    def _get_patch_shift(self):
        return self._patch_shift

    def _set_patch_shift(self, patch_shift):
        self._patch_shift = patch_shift
        self._build_zelda()

    def _del_patch_shift(self):
        print("Undeletable type property")

    """####################################################################"""
    """####################### FUNCTIONS DEFINITION #######################"""
    """####################################################################"""

    def _build_zelda(self):
        """
        M.Vortex(charge=C) : changes M.complex_transparency to a vortex mask
        with charge C.

        Parameters
        ----------
        charge : int
            Charge of the vortex mask.

        Returns
        -------
        Mask object with modified complex_transparency
        """
        N = self._field_size
        x = np.linspace(-N // 2 + 0.5, N // 2 - 0.5, N)
        y = np.linspace(-N // 2 + 0.5, N // 2 - 0.5, N)
        x_grid, y_grid = np.meshgrid(x, y)
        rho = np.sqrt(x_grid**2 + y_grid**2)
        tmp = np.ones((N, N), dtype=complex)
        tmp[rho < self._patch_size / 2] = np.exp(1j * self._patch_shift)
        self._complex_transparency = tmp

    """####################################################################"""
    """####################### PROPERTIES DEFINITION ######################"""
    """####################################################################"""

    patch_size = property(
        _get_patch_size,
        _set_patch_size,
        _del_patch_size,
        "shape of the mask. None if no defined shape",
    )

    patch_shift = property(
        _get_patch_shift,
        _set_patch_shift,
        _del_patch_shift,
        "shape of the mask. None if no defined shape",
    )


class Zernike(Mask):

    """####################################################################"""

    """####################### INIT AND OVERLOADING #######################"""
    """####################################################################"""

    # CONSTRUCTOR
    def __init__(self, field_size, **kwargs):
        Mask.__init__(self, field_size)
        self._indices = kwargs.get("indices", [1])
        self._coefficients = kwargs.get("coefficients", np.ones(len(self._indices)))
        self._modes = np.zeros([field_size, field_size, len(self._indices)])
        self._pupil = kwargs.get("pupil", np.ones((field_size, field_size)))
        self._build_modes()
        self._build_mean_matrix()
        self._build_zernike()

    """####################################################################"""
    """####################### GET / SET DEFINITION #######################"""
    """####################################################################"""

    # GET/SET CHARGE
    def _get_coefficients(self):
        return self._coefficients

    def _set_coefficients(self, coefficients):
        self._coefficients = coefficients
        self._build_zernike()

    def _del_coefficients(self):
        print("Undeletable type property")

    # GET/SET MEAN MATRIX
    def _get_mean_matrix(self):
        return self._mean_matrix

    def _set_mean_matrix(self, mean_matrix):
        self._mean_matrix = mean_matrix

    def _del_mean_matrix(self):
        print("Undeletable type property")

    """####################################################################"""
    """####################### FUNCTIONS DEFINITION #######################"""
    """####################################################################"""

    def _build_zernike(self):
        """
        M.Vortex(charge=C) : changes M.complex_transparency to a vortex mask
        with charge C.

        Parameters
        ----------
        charge : int
            Charge of the vortex mask.

        Returns
        -------
        Mask object with modified complex_transparency
        """
        coefficients = self._coefficients
        zernike = np.zeros((self._field_size, self._field_size))
        for i in range(0, len(coefficients)):
            zernike = zernike + coefficients[i] * self._modes[:, :, i]
        self._complex_transparency = np.exp(1j * zernike)

    def _build_modes(self):
        # FIELD MAP BUILDING
        N = self._field_size
        x = np.linspace(-4, 4, N)
        y = np.linspace(-4, 4, N)
        x_grid, y_grid = np.meshgrid(x, y)
        rho = np.sqrt(x_grid**2 + y_grid**2)
        phi = np.arctan2(y_grid, x_grid)
        idx = 0
        for l in self._indices:
            n, m = self._noll_to_radial(l)
            r = np.zeros((N, N))
            for k in range(0, (n - abs(m)) // 2 + 1):
                r = r + (-1) ** k * math.factorial(n - k) * rho ** (n - 2 * k) / (
                    math.factorial(k)
                    * math.factorial((n + m) // 2 - k)
                    * math.factorial((n - m) // 2 - k)
                )
            if m > 0:
                tmp = r * np.cos(m * phi) * self._pupil
                self._modes[:, :, idx] = tmp / np.sqrt(np.sum(tmp**2))
            elif m < 0:
                tmp = r * np.sin(abs(m) * phi) * self._pupil
                self._modes[:, :, idx] = tmp / np.sqrt(np.sum(tmp**2))
            else:
                tmp = r * self._pupil
                self._modes[:, :, idx] = tmp / np.sqrt(np.sum(tmp**2))
            idx += 1

    def _noll_to_radial(self, j):
        n = int(np.round(np.sqrt(2 * j + 1) - 1))
        m = int(2 * j - n**2 - 2 * n)
        return n, m

    def _radial_to_noll(self, n, m):
        return (n * (n + 2) + m) / 2

    def _build_mean_matrix(self):
        modes_number = len(self._indices)
        N, M = self._noll_to_radial(self._indices[modes_number - 1])
        n0, m0 = self._noll_to_radial(self._indices[0])
        self._mean_matrix = np.zeros((N - n0, modes_number))
        offset = -2 * n0 + (n0 + 1) * (n0 + 2) // 2 - 1
        for z in range(n0, N - n0):
            self._mean_matrix[
                z - n0,
                -n0
                + (z + 1) * (z + 2) // 2
                - 1
                - z
                - offset : (z + 1) * (z + 2) // 2
                - n0
                - offset,
            ] = (
                1 / z
            )

    """####################################################################"""
    """####################### PROPERTIES DEFINITION ######################"""
    """####################################################################"""

    coefficients = property(
        _get_coefficients,
        _set_coefficients,
        _del_coefficients,
        "shape of the mask. None if no defined shape",
    )

    mean_matrix = property(
        _get_mean_matrix,
        _set_mean_matrix,
        _del_mean_matrix,
        "shape of the mask. None if no defined shape",
    )
