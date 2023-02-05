# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 11:58:03 2018

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
from masque import Mask

n = 256
f = Field(n, field_map='plan_wave', incidence_angles=[-np.pi/n, -np.pi/n])
pupille = Pupil(n, aperture_shape='circular')
polar = Polarizer('linear', angle=np.pi/5)
separator = Separator(transmitance=0.55, reflectance=0.4)
slm = SLM(n)
camera = Detector(n)
FFT = Propagator('FFT', f)
IFFT = Propagator('IFFT', f)

# FLAT FIELD F THROUGH CIRUCLAR PUPIL
# PASSING THROUGH SEPARATOR IN TRANSMISSION
# FOCALIZING ON SLM WITH A PYRAMID MASK GIVING A .75xPUPIL DIAMETER SEPARATION
# PASSING THROUGH SEPARATOR IN REFLECTION
# ARRIVING ON CAMERA
((((f * pupille) * separator.T()) > slm.Pyramid(angle=n, faces=4)) <
separator.R()) * camera
camera.disp_intensity()

# CHANGE THE MASK ON THE SLM BY A FQPM
slm.FQPM()
# ACTUALIZE THE FIELD
+f
# DISPLAY THE CAMERA IMAGE
camera.disp_intensity()

# CHANGE THE MASK ON THE SLM BY A VORTEX
slm.Vortex(charge=2)
# ACTUALIZE THE FIELD
+f
# DISPLAY THE CAMERA IMAGE
camera.disp_intensity()
