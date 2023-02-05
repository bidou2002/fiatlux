# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 11:26:39 2018

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

# =============================================================================
# DEFINE EVERYTHING WE NEED
# =============================================================================

# DEFINE FIELD'S SIZE
n = 256

# CREATE FIELD
#f = Field(n, field_map='gaussian', gaussian_variance=2)
f = Field(n, field_map='plan_wave', incidence_angles=[-np.pi/(n),
                                                      -np.pi/(n)])

# CREATED TIP-TILT MASK TO CENTER ON THE SLM TOP
tip = Tilt(n, angles=[-1, -1])

# DEFINE THE PROPAGATORS
FFT = Propagator('FFT', f)
IFFT = Propagator('IFFT', f)

# CREATE THE ENTRANCE PUPIL
P = Pupil(n, aperture_shape='circular')

# CREATED TIP-TILT MASK TO MODULATE ON THE SLM TOP
mod = TipTiltMirror(n)

# CREATE THE PYRAMID MASK
Pyr = Pyramid(n, angle=n/(2*np.sqrt(2)), faces=4)

# CREATE THE TWO OCAM DETECTORS
OCAM1 = Detector(n)

# %%
# DO THE PROPAGATION
(((((((f * tip) * IFFT) * P) * mod) * FFT) * Pyr) * IFFT) * OCAM1
# DISPLAY THE CAMERA INTENSITY
OCAM1.disp_intensity()

# %%
# CREATE THE MODULATION VECTORS
x_mod = np.cos(1)*100*np.cos(
        np.linspace(2*np.pi/100,
                    2*np.pi, 100))
y_mod = np.sin(1)*100*np.sin(
        np.linspace(2*np.pi/100,
                    2*np.pi, 100))

# CHANGE THE MODULATION
mod.angles = [x_mod, y_mod]
# ACTUALIZE THE FIELD
+f
# DISPLAY THE CAMERA INTENSITY
OCAM1.disp_intensity()

# %%
# CREATE THE MODULATION VECTORS
x_mod = np.zeros((200))
y_mod = np.zeros((200))

r = 5
x_mod[0:50] = np.linspace(-r, r)
x_mod[50:100] = np.linspace(r, r)
x_mod[100:150] = np.linspace(r, -r)
x_mod[150:200] = np.linspace(-r, -r)

y_mod[0:50] = np.linspace(-r, -r)
y_mod[50:100] = np.linspace(-r, r)
y_mod[100:150] = np.linspace(r, r)
y_mod[150:200] = np.linspace(r, -r)

# CHANGE THE PYRAMID NUMBER OF FACES
Pyr.faces = 5
Pyr.angle = n/2
# CHANGE THE MODULATION
mod.angles = [x_mod, y_mod]
# ACTUALIZE THE FIELD
+f
# DISPLAY THE CAMERA INTENSITY
OCAM1.disp_intensity()
