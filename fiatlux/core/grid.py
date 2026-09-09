# grid.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
import torch


class BaseGrid(ABC):

    @abstractmethod
    def meshgrid(self) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> BaseGrid: ...


@dataclass
class Grid(BaseGrid):
    nx: int
    ny: int
    dx: float
    dy: float
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        for name in ("nx", "ny"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("dx", "dy"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite spacing.")
        self.device = torch.device(self.device)
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("Grid dtype must be torch.float32 or torch.float64.")

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial tensor shape in canonical ``(ny, nx)`` order."""
        return (self.ny, self.nx)

    @property
    def x(self) -> torch.Tensor:
        return (
            torch.arange(self.nx, device=self.device, dtype=self.dtype) - self.nx // 2
        ) * self.dx

    @property
    def y(self) -> torch.Tensor:
        return (
            torch.arange(self.ny, device=self.device, dtype=self.dtype) - self.ny // 2
        ) * self.dy

    def meshgrid(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x, y)`` coordinate arrays, both shaped ``(ny, nx)``."""
        return torch.meshgrid(self.x, self.y, indexing="xy")

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Grid:
        return Grid(
            self.nx,
            self.ny,
            self.dx,
            self.dy,
            self.device if device is None else torch.device(device),
            self.dtype if dtype is None else dtype,
        )
