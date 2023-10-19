# -*- coding: utf-8 -*-
"""
  ______                                           
 / _____)                    _                     
( (____  ____  _____  ____ _| |_  ____ _   _ ____  
 \____ \|  _ \| ___ |/ ___|_   _)/ ___) | | |    \ 
 _____) ) |_| | ____( (___  | |_| |   | |_| | | | |
(______/|  __/|_____)\____)  \__)_|   |____/|_|_|_|
        |_|                                        

Created on Thu Oct 19 10:02:00 2023

@author: pjanin
"""


import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import astropy.units as u

from .field import Field


class Spectrum:
    # Photometry from OOPAO
    def photometry(arg):
        # photometry object [wavelength, bandwidth, zeroPoint]
        class phot:
            pass

        phot.U = [0.360e-6, 0.070e-6, 2.0e12]
        phot.V0 = [0.500e-6, 0.090e-6, 3.3e12]
        phot.B = [0.440e-6, 0.100e-6, 5.4e12]
        phot.V = [0.550e-6, 0.090e-6, 3.3e12]
        phot.R = [0.640e-6, 0.150e-6, 4.0e12]
        phot.I = [0.790e-6, 0.150e-6, 2.7e12]
        phot.I1 = [0.700e-6, 0.033e-6, 2.7e12]
        phot.I2 = [0.750e-6, 0.033e-6, 2.7e12]
        phot.I3 = [0.800e-6, 0.033e-6, 2.7e12]
        phot.I4 = [0.700e-6, 0.100e-6, 2.7e12]
        phot.I5 = [0.850e-6, 0.100e-6, 2.7e12]
        phot.I6 = [1.000e-6, 0.100e-6, 2.7e12]
        phot.I7 = [0.850e-6, 0.300e-6, 2.7e12]
        phot.R2 = [0.650e-6, 0.300e-6, 7.92e12]
        phot.R3 = [0.600e-6, 0.300e-6, 7.92e12]
        phot.R4 = [0.670e-6, 0.300e-6, 7.92e12]
        phot.I8 = [0.750e-6, 0.100e-6, 2.7e12]
        phot.I9 = [0.850e-6, 0.300e-6, 7.36e12]
        phot.I10 = [0.900e-6, 0.300e-6, 2.7e12]
        phot.J = [1.215e-6, 0.260e-6, 1.9e12]
        phot.H = [1.654e-6, 0.290e-6, 1.1e12]
        phot.Kp = [2.1245e-6, 0.351e-6, 6e11]
        phot.Ks = [2.157e-6, 0.320e-6, 5.5e11]
        phot.K = [2.179e-6, 0.410e-6, 7.0e11]
        phot.K0 = [2.000e-6, 0.410e-6, 7.0e11]
        phot.K1 = [2.400e-6, 0.410e-6, 7.0e11]

        phot.L = [3.547e-6, 0.570e-6, 2.5e11]
        phot.M = [4.769e-6, 0.450e-6, 8.4e10]
        phot.Na = [0.589e-6, 0, 3.3e12]
        phot.EOS = [1.064e-6, 0, 3.3e12]

        if isinstance(arg, str):
            if hasattr(phot, arg):
                wavelength, delta_lambda, f0 = getattr(phot, arg)
                wavelength *= u.meter
                delta_lambda *= u.meter
                f0 *= u.photon / (u.meter**2 * u.second)
                return [wavelength, delta_lambda, f0]
            else:
                print("Error: Wrong name for the photometry object")
                return -1
        else:
            print("Error: The photometry object takes a scalar as an input")
            return -1
