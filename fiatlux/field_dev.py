from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np

from .source_dev import Source, PointSource, ExtendedSource


@dataclass
class Field:
    field_size: int
    source_list: list[Source]
    scale: float = 1.0

    def __post_init__(self):
        print("bite")
        self._complex_amplitude_list = []
        for source in self.source_list:
            self._complex_amplitude_list += [source.compute_field(self.field_size)]
