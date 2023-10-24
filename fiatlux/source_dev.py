from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np

from .spectrum_dev import Spectrum, SpectrumType


class SourceType(Enum):
    POINT_SOURCE = auto()
    EXTENDED_SOURCE = auto()


@dataclass
class Source:
    type: SourceType = SourceType.POINT_SOURCE
    spectrum: Spectrum = Spectrum(type=SpectrumType.MONOCHROMATIC)

    def __post_init__(self):
        match self.spectrum:
            case SpectrumType.MONOCHROMATIC:
                pass
            case SpectrumType.PHOTOMETRIC:
                pass
            case SpectrumType.BLACK_BODY:
                pass
            
