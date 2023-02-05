# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:02:03 2018

@author: pjanin
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from arrow3D import Arrow3D
from polarizer import Polarizer
from mirror import Mirror
from masque import Mask
from detecteur import Detector
from propagator import Propagator
from champ import Field
from pupille import Pupil
from lentille import Lens
import imageio


f = Field(256, scale=4e-5)
pup = Pupil(256, aperture_shape='circular', aperture_size=32)
cam = Detector(256)

f * pup
for i in range(200):
    dist = (1+i/25)*25e-3
    FST = Propagator('FST', f, z=dist)
    ((f@1)*FST)*cam
    image = (np.abs(cam.complex_amplitude)**2)
    plt.imshow(image)
    plt.title('Intensity @ z = {:.3g} m'.format(dist))
    plt.show()
    plt.pause(.25)
