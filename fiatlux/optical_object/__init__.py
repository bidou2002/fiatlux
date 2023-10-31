from abc import abstractmethod
from dataclasses import dataclass

from ..physical_object.complex_amplitude import ComplexAmplitude


class OpticalObject:
    @abstractmethod
    def compute_transformation(
        self,
        complex_amplitudes: list[ComplexAmplitude],
    ) -> list[ComplexAmplitude]:
        pass

    @abstractmethod
    def compute_complex_transparency(self, size) -> list[ComplexAmplitude]:
        pass
