"""Named leading axes for ordinary (unnamed) PyTorch tensors."""

from dataclasses import dataclass, replace
import torch


@dataclass(frozen=True, eq=False)
class FieldDimension:
    """An axis descriptor. Treat its tensor values as immutable as well.

    Integration weights are explicit quadrature weights; for time axes with
    ``unit='s'`` they are durations in seconds, not normalized probabilities.
    """

    name: str
    size: int
    coordinates: torch.Tensor | None = None
    unit: str | None = None
    integration_weights: torch.Tensor | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Dimension name must be non-empty.")
        if self.name in {"wavelength", "x", "y"}:
            raise ValueError(f"Reserved dimension name: {self.name}.")
        if (
            not isinstance(self.size, int)
            or isinstance(self.size, bool)
            or self.size < 0
        ):
            raise ValueError("Dimension size must be a non-negative integer.")
        if self.unit is not None and not isinstance(self.unit, str):
            raise TypeError("Dimension unit must be a string or None.")
        for name in ("coordinates", "integration_weights"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, torch.Tensor) or value.shape != (self.size,):
                    raise ValueError(f"{name} must be a 1-D tensor of length size.")
                if not value.is_floating_point() or not torch.isfinite(value).all():
                    raise ValueError(
                        f"{name} must contain finite real floating values."
                    )
                if name == "integration_weights" and (value < 0).any():
                    raise ValueError("integration_weights must be non-negative.")
                object.__setattr__(self, name, value.clone())

    def __eq__(self, other):
        if not isinstance(other, FieldDimension):
            return NotImplemented
        if (self.name, self.size, self.unit) != (other.name, other.size, other.unit):
            return False
        for name in ("coordinates", "integration_weights"):
            a, b = getattr(self, name), getattr(other, name)
            if a is None or b is None:
                if a is not b:
                    return False
            elif a.device != b.device or a.dtype != b.dtype or not torch.equal(a, b):
                return False
        return True

    __hash__ = None

    def to(self, *, device=None, dtype=None):
        return replace(
            self,
            **{
                name: (
                    None
                    if getattr(self, name) is None
                    else getattr(self, name).to(device=device, dtype=dtype)
                )
                for name in ("coordinates", "integration_weights")
            },
        )

    def sliced(self, selection: slice):
        return replace(
            self,
            size=len(range(self.size)[selection]),
            **{
                name: (
                    None
                    if getattr(self, name) is None
                    else getattr(self, name)[selection]
                )
                for name in ("coordinates", "integration_weights")
            },
        )


def validate_dimensions(dimensions, *, device, dtype):
    dimensions = tuple(dimensions)
    if not all(isinstance(d, FieldDimension) for d in dimensions):
        raise TypeError("dimensions must contain FieldDimension descriptors.")
    if len({d.name for d in dimensions}) != len(dimensions):
        raise ValueError("Dimension names must be unique.")
    for d in dimensions:
        for value in (d.coordinates, d.integration_weights):
            if value is not None and (value.device != device or value.dtype != dtype):
                raise ValueError(
                    "Dimension tensors must share the field device and real dtype."
                )
    return dimensions
