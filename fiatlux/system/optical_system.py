from __future__ import annotations
from dataclasses import dataclass


import torch

import matplotlib.pyplot as plt

import math

from fiatlux.core.field import Field
from fiatlux.optics.detector import Detector
from fiatlux.core.source import Source
from fiatlux.optics.propagator import Propagator
from fiatlux.optics.elements.base import OpticalElement


@dataclass
class SimulationStep:
    """Snapshot du champ à une étape donnée."""

    element: OpticalElement
    field_before: Field  # champ avant application de l'élément
    field_after: Field  # champ après


class SerialSystem:
    """
    Encapsule la séquence d'éléments et orchestre la propagation.
    Les éléments n'ont jamais connaissance du champ : c'est ici
    que la logique de propagation + trace est centralisée.
    """

    def __init__(
        self,
        elements: list[OpticalElement | Propagator] = None,
    ):
        self.elements = elements

    def run(self, source: Source, detector: Detector = None) -> SimulationResult:
        """
        Propage le champ à travers tous les éléments dans l'ordre axial.
        Retourne un SimulationResult avec la trace complète.
        """
        steps: list[SimulationStep] = []

        current_field = source.generate_field(self.elements[0].grid)
        steps.append(SimulationStep(source, None, current_field))

        for element in self.elements:

            field_before = current_field
            field_after = element.apply(current_field)

            steps.append(SimulationStep(element, field_before, field_after))
            current_field = field_after

        if detector:
            detector.acquire(current_field)

        return SimulationResult(steps)

    def __str__(self) -> str:
        s = ""
        for element in self.elements:
            try:
                s += element._symbol + " "
            except:
                pass
        return s


# ─────────────────────────────────────────────
# Résultat de simulation : accès à la trace
# ─────────────────────────────────────────────


class SimulationResult:
    def __init__(self, steps: list[SimulationStep]):
        self.steps = steps

        self._element_to_step = {
            id(step.element): step for step in steps if step.element is not None
        }

    def field_at(self, element: OpticalElement) -> Field:
        try:
            return self._element_to_step[id(element)].field_after
        except KeyError:
            raise ValueError(f"Element {element} not found in simulation.")

    def intensity_at(self, element: OpticalElement) -> torch.Tensor:
        try:
            return self.field_at(element).intensity()
        except KeyError:
            raise ValueError(f"Element {element} not found in simulation.")

    def __iter__(self):
        return iter(self.steps)

    def plot(self):
        n = math.ceil(len(self.steps) ** 0.5)
        print(n)
        fig, axs = plt.subplots(n, n)
        for i, step in enumerate(self.steps):
            axs[i // n, i % n].imshow(step.field_after.intensity()[0])
            axs[i // n, i % n].set_title(f"{step.element.__str__()[:10]}")
