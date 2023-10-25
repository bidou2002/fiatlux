from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np

from .spectrum_dev import Spectrum, SpectralType, Monochromatic


@dataclass
class Source:
    spectrum: Spectrum = Monochromatic(wavelength=550e-9, irradiance=1)


@dataclass(kw_only=True)
class PointSource:
    pass


@dataclass(kw_only=True)
class ExtendedSource:
    pass
