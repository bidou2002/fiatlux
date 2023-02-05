#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:14:36 2018

@author: pjanin
"""

import numpy as np
from champ import Field
from pupille import Pupil
from polarizer import Polarizer
from separateur import Separator
from mirror import Mirror
from detecteur import Detector
from slm import SLM
from propagator import Propagator
from masque import *
import matplotlib.pyplot as plt
import time
from FindCaptureRange import FindCaptureRange

#%%

# DEFINE FIELD'S SIZE
n = 256
# DEFINE MODES NUMBER
Z = 100
F = 20

# CREATE THE INCIDENT FIELD
f = Field(n, field_map='plan_wave', incidence_angles=[-np.pi/(n),
                                                      -np.pi/(n)])
# CREATE THE ENTRANCE PUPIL
P = Pupil(n, aperture_shape='circular')
# CREATE THE SLM WITH A 4-SIDED PYRAMID
Slm = SLM(n, shape='pyramid', pyramid_angle=n/(2*np.sqrt(2)), pyramid_faces=4)
# CREATE THE PYRAMID MASK
Pyr = Pyramid(n, angle=n/(2*np.sqrt(2)), faces=4)
FlatPyr1 = Pyramid(n, angle=n/20, faces=4)
FlatPyr2 = Pyramid(n, angle=n/100, faces=4)
iFQ = FQPM(n,shift=np.pi/2)
iFQm = FQPM(n,shift=-np.pi/2)
# FQ = FQPM(n,shift=np.pi)
Zel = Zelda(256)

blank = Blank(n)

# CREATE THE TWO OCAM DETECTORS AND THE THORLABS
PyrCam = Detector(n, camera_name='Pyramid', display_intensity=False)
FlatPyrCam1 = Detector(n, camera_name='FlatPyr', display_intensity=False)
FlatPyrCam2 = Detector(n, camera_name='FlatPyr', display_intensity=False)
iFQCam = Detector(n, camera_name='iFQPM', display_intensity=False)
iFQmCam = Detector(n, camera_name='iFQPMm', display_intensity=False)
ZelCam = Detector(n, camera_name='Zelda', display_intensity=False)

# CREATE THE ZERNIKE FOR CALIBRATION PURPOSES
zern = Zernike(256, indices=list(range(1,Z+1)), coefficients=np.zeros(Z),
               pupil=np.abs(P.complex_transparency))

four = Fourier(256, n_modes = F, coefficients=np.zeros(Z),
               pupil=np.abs(P.complex_transparency))

FFT = Propagator('FFT', f)
IFFT = Propagator('IFFT', f)

f1 = Field(n, field_map='plan_wave', incidence_angles=[-np.pi/(n),
                                                      -np.pi/(n)])
f2 = Field(n, field_map='plan_wave', incidence_angles=[-np.pi/(n),
                                                      -np.pi/(n)])
f3 = Field(n, field_map='plan_wave', incidence_angles=[-np.pi/(n),
                                                      -np.pi/(n)])
f4 = Field(n, field_map='plan_wave', incidence_angles=[-np.pi/(n),
                                                      -np.pi/(n)])
f5 = Field(n, field_map='plan_wave', incidence_angles=[-np.pi/(n),
                                                      -np.pi/(n)])
f6 = Field(n, field_map='plan_wave', incidence_angles=[-np.pi/(n),
                                                      -np.pi/(n)])
# PROPAGATE THE FIELDS
f1 * P * blank * zern * four * FFT * Pyr *IFFT * PyrCam
f2 * P * blank * zern * four * FFT * FlatPyr1 *IFFT * FlatPyrCam1
f3 * P * blank * zern * four * FFT * FlatPyr2 *IFFT * FlatPyrCam2
f4 * P * blank * zern * four * FFT * iFQ *IFFT * iFQCam
f5 * P * blank * zern * four * FFT * Zel *IFFT * ZelCam
f6 * P * blank * zern * four * FFT * iFQm *IFFT * iFQmCam

# CALCULATE INTENSITIES AT 0 ABERRATIONS
I1_0 = np.abs(PyrCam.complex_amplitude)**2
I2_0 = np.abs(FlatPyrCam1.complex_amplitude)**2
I3_0 = np.abs(FlatPyrCam2.complex_amplitude)**2
I4_0 = np.abs(iFQCam.complex_amplitude)**2
I5_0 = np.abs(ZelCam.complex_amplitude)**2
I6_0 = np.abs(iFQmCam.complex_amplitude)**2

#%%
amp = 0.001

# PRE-ALLOCATE THE IMAGES
I1_m = np.zeros((n, n, F**2 + 1))
I2_m = np.zeros((n, n, F**2 + 1))
I3_m = np.zeros((n, n, F**2 + 1))
I4_m = np.zeros((n, n, F**2 + 1))
I5_m = np.zeros((n, n, F**2 + 1))
I6_m = np.zeros((n, n, F**2 + 1))
I7_m = np.zeros((n, n, F**2 + 1))

# INITIALIZE THE INDEX
idx = 0
for f in range(0, F**2 + 1):
    # SET THE AMPLITUDE OF THE ZERNIKE MODE
    a = np.zeros(F**2 + 1)
    a[f] = amp
    four.coefficients = a
    # PROPAGATE THE WAVEFRONTS FOR EACH SENSOR
    +f1
    I1 = np.abs(PyrCam.complex_amplitude)**2 - I1_0
    +f2
    I2 = np.abs(FlatPyrCam1.complex_amplitude)**2 - I2_0
    +f3
    I3 = np.abs(FlatPyrCam2.complex_amplitude)**2 - I3_0
    +f4
    I4 = np.abs(iFQCam.complex_amplitude)**2 - I4_0
    +f5
    I5 = np.abs(ZelCam.complex_amplitude)**2 - I5_0
    +f6
    I6 = np.abs(iFQmCam.complex_amplitude)**2 - I6_0
    I7 = (I4)/2-(I6)/2
    # SAVE IMAGES
    I1_m[:, :, idx] = I1[:, :, 0]
    I2_m[:, :, idx] = I2[:, :, 0]
    I3_m[:, :, idx] = I3[:, :, 0]
    I4_m[:, :, idx] = I4[:, :, 0]
    I5_m[:, :, idx] = I5[:, :, 0]
    I6_m[:, :, idx] = I6[:, :, 0]
    I7_m[:, :, idx] = I7[:, :, 0]
    print('four : {:d} %'.format(np.int(100*f/(F**2 + 1))))
    idx += 1