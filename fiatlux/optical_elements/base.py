from dataclasses import dataclass, field
from fiatlux.field import Field
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np


class OpticalElement(ABC):
    @abstractmethod
    def apply(self, field: Field) -> Field:
        """Applique l'effet de l'élément sur le champ entrant."""
        raise NotImplementedError(
            "La méthode apply doit être implémentée par les sous-classes."
        )
