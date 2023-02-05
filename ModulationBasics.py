# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:06:25 2019

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
from masque import Pyramid, Tilt, TipTiltMirror


# DEFINE FIELD'S SIZE
n = 256

# CREATE FIELD
#f = Field(n, field_map='gaussian', gaussian_variance=2)
f = Field(n, field_map='plan_wave', incidence_angles=[-np.pi/(n),
                                                      -np.pi/(n)])

# DEFINE THE PROPAGATORS
FFT = Propagator('FFT', f)
IFFT = Propagator('IFFT', f)

# CREATE THE ENTRANCE PUPIL
P = Pupil(n, aperture_shape='circular')

# CREATED TIP-TILT MASK TO MODULATE ON THE SLM TOP
mod = TipTiltMirror(n)


# CREATE THE TWO OCAM DETECTORS
Cam = Detector(n, camera_name='Thorlabs Cam')

#%%

# LIGHT PROPAGATION FROM INFINITE SOURCE -> PUPIL
# -> MODULATION MIRROR -> CAMERA
f * P * mod * FFT * Cam

#%%
# ROTATION
theta = 0

# AMPLITUDE
amp = 100

# PHASE
phi = np.pi/2

# CREATE THE MODULATION VECTORS
x_mod = amp * np.cos(theta) * np.cos(
        np.linspace(2*np.pi/200,
                    2*np.pi, 200) + phi)
y_mod = amp * np.sin(theta) * np.sin(
        np.linspace(2*np.pi/200,
                    2*np.pi, 200))

x_mod = amp * np.cos(theta) * np.linspace(1, 1, 100)
y_mod = amp * np.sin(theta) * np.linspace(1, 1, 100)

x_mod = amp * np.cos(theta) * np.linspace(-1, 1, 100)
y_mod = amp * np.sin(theta) * np.linspace(1, 1, 100)

# CHANGE THE MODULATION
mod.angles = [x_mod, y_mod]
# ACTUALIZE THE FIELD
+f
# DISPLAY THE CAMERA INTENSITY
Cam.disp_intensity()
