# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 18:58:50 2018

@author: pjanin
"""

import numpy as np
import matplotlib.pyplot as plt
from champ import Field
from polarizer import Polarizer
from mirror import Mirror
from masque import Mask
from detecteur import Detector
from propagator import Propagator

# CREATE A FLAT TITLTED FIELD WITH ARBITRARY TIP AND TILT
f = Field(256, wavelength=632.8e-9, field_map='plan_wave',
          incidence_angles=[.51, .255])

# CREATE A PUPIL MASK
pup = Mask(256, diameter=32)

# CREATE A LINEAR POLARIZER PARALLEL TO THE HORIZONTAL AXIS
p_linear = Polarizer('linear', angle=0)
# CREATE A BIREFRIGENT MATERIAL POLARIZER WITH FAST AXIS ORIENTED AT PI/4 AND A
# PI/2 PHASE RETARDATION (TO TRANSFORM AN HORIZONTAL POLARIZATION TO A
# LEFT-HANDED CIRCULAR POLARIZATION)
p_retarder = Polarizer('retarder', angle=-np.pi/4, phase_retardation=np.pi/2)

# CREATE A MIRROR
mirror = Mirror(reflectance=1, angle=np.pi/3)

# CREATE A DETECTOR
cam = Detector(256)

'''#####################################################################'''
'''#################### > AND < FOURIER PROPAGATION ####################'''
'''#####################################################################'''

# MAKE FOURIER PROPAGATION WITH > AND INVERSE FOURIER PROPAGATION WITH < FROM
# PLAN TO PLAN
(((((f * pup) > p_linear) < mirror) * p_retarder) > mirror) * cam

# PRINT THE PATH FOLLOWED BY THE LIGHT
print('\n'.join(map(str, f.optical_path)))

# PRINT FIELD AND POLARIZATION :
# 0 = ENTRANCE FIELD
# 1 = FIELD AFTER PUPIL
# 2 = FIELD AFTER FOURIER TRANSFORM
# 3 = FIELD AFTER LINEAR POLARIZER
# 4 = FIELD BEFORE MIRROR AFTER FOURIER TRANSFORM
# 5 = FIELD AFTER MIRROR
# 6 = FIELD AFTER BIREFRINGENT PLATE
# 7 = FIELD AFTER FOURIER TRASNFORM
# 8 = FIELD AFTER MIRROR
# 9 = FIELD AT CAMERA
BoolPlot = True
Plan2Plot = (0, 1, 2)
if BoolPlot is True:
    for i in Plan2Plot:
        (f@i).disp_complex_amplitude()

'''#####################################################################'''
'''############# FOURIER PROPAGATION WITH PROPAGATOR CLASS #############'''
'''#####################################################################'''

# PROPAGATE VIA THE PROPAGATOR OBJECT
FFT = Propagator('FFT', f)
IFFT = Propagator('IFFT', f)

f = f@0
(((((((f * pup) * FFT) * p_linear * IFFT) * mirror) * p_retarder) * FFT) *
 mirror) * cam
# PRINT THE PATH FOLLOWED BY THE LIGHT
print('\n'.join(map(str, f.optical_path)))

# PRINT FIELD AND POLARIZATION AT POSITIONS 0 (ENTRANCE FIELD),
# 1 (FIELD AFTER PUPIL), 2 (FIELD AFTER FOURIER TRANSFORM),
# 4 (FIELD BEFORE MIRROR), 5 (FIELD AFTER MIRROR), 6 (FIELD AFTER BIREFRINGENT
# PLATE)
BoolPlot = True
if BoolPlot is True:
    for i in Plan2Plot:
        (f@i).disp_complex_amplitude()
