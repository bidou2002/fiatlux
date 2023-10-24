from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np


class SpectrumType(Enum):
    MONOCHROMATIC = auto()
    PHOTOMETRIC = auto()
    BLACK_BODY = auto()


class Monochromatic:
    pass


class BlackBody:
    pass


class Photometric:
    pass


@dataclass
class Spectrum:
    type: SpectrumType
    match type:
        case SpectrumType.MONOCHROMATIC:
            print("Bite")
        case SpectrumType.PHOTOMETRIC:
            print("cul")
        case SpectrumType.BLACK_BODY:
            print("couille")
    photon_number: float = None
