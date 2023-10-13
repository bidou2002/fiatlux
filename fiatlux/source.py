# -*- coding: utf-8 -*-
"""
  ______                              
 / _____)                             
( (____   ___  _   _  ____ ____ _____ 
 \____ \ / _ \| | | |/ ___) ___) ___ |
 _____) ) |_| | |_| | |  ( (___| ____|
(______/ \___/|____/|_|   \____)_____)

Created on Fri Oct 13 10:05:00 2023

@author: pjanin
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from .field import Field

class Source(Field):

    # Constructor
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'unamed_source')

        self.magnitude = kwargs.get('magnitude', 0.)