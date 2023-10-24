from abc import abstractmethod
from dataclasses import dataclass
from enum import auto, Enum

import numpy as np

from .source_dev import Source

@dataclass
class Scene:
    source_list: list[Source]