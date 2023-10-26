from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np
import pyfftw
import os

from .source_dev import Source, PointSource, ExtendedSource
from .complex_amplitude import ComplexAmplitude


@dataclass
class Satellite:
    name: str = "Unknown"
    altitude: float = 0.0
    speed: float = 0.0

    pass

    def compute_path(self):
        pass
        satellite_path = []
        x0 = 0.5
        y0 = -1
        r = 0.5
        for theta in np.linspace(0, 2 * np.pi, 100):
            satellite_path += [[x0 + r * np.cos(theta), y0 + r * np.sin(theta)]]
            satellite_path += [[-x0 + r * np.cos(theta), y0 + r * np.sin(theta)]]

        for y in np.linspace(r, 3 - r, 100):
            satellite_path += [[x0, y0 + y]]
            satellite_path += [[-x0, y0 + y]]

        for y in np.linspace(3, 2 + r, 50):
            satellite_path += [[0, y0 + y]]

        for theta in np.linspace(0, np.pi, 50):
            satellite_path += [[r * np.cos(theta), y0 + y + r * np.sin(theta)]]
