from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np

from fiatlux.field import Field
from fiatlux.grid import Grid


@dataclass
class OpticalElement(ABC):
    """Élément optique pur : transforme un champ, ne le stocke pas."""

    @abstractmethod
    def apply(self, field: Field) -> Field:
        """Applique l'effet de l'élément sur le champ entrant."""
        ...

    @abstractmethod
    def build(self, grid: Grid) -> None:
        """Construit l'élément optique sur la grille donnée."""
        ...
