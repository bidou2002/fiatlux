# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 17:19:43 2018

@author: pjanin
"""
import numpy as np


def FindCaptureRange(vector):
    diff = np.diff(vector)
    length = np.size(diff)
    X = range(0, length//2)
    x1 = 0
    x2 = 0
    idx = 0
    while (x1 == 0 or x2 == 0) and idx < length/2 - 1:
        i = X[idx]
        if ((diff[i] >= 0 and diff[i+1] <= 0) or
            (diff[i] <= 0 and diff[i+1] >= 0)):
            x1=i
        if ((diff[-i-1] >= 0 and diff[-i-2] <= 0) or
            (diff[-i-1] <= 0 and diff[-i-2] >= 0)):
            x2=length-i-2
        idx += 1
    return np.abs(x2-x1)
