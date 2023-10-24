from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np

from .source_dev import Source, SourceType
from .scene_dev import Scene


@dataclass
class Field:
    size: int
    source_list: list[Source]
    scale: float = 1.0

    def __post_init__(self):
        print("bite")
        complex_amplitude_list = []
        for source in self.source_list:
            match source.type:
                case SourceType.POINT_SOURCE:
                    complex_amplitude = []
                case SourceType.EXTENDED_SOURCE:
                    complex_amplitude = []

            complex_amplitude_list += complex_amplitude
        self._complex_amplitude_list = complex_amplitude_list
