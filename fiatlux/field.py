# field.py
from dataclasses import dataclass
import torch

from fiatlux.grid import Grid


@dataclass
class Field:
    """
    Champ électrique complexe défini sur une grille.

    L'amplitude est un tenseur complexe 2D (ny, nx).
    La grille définit la discrétisation spatiale.
    La longueur d'onde est une propriété physique du champ.
    """

    complex_amplitude: torch.Tensor
    grid: Grid
