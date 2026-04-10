# grid.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch

from fiatlux.core.spectrum import Spectrum


class BaseGrid(ABC):

    @abstractmethod
    def meshgrid(self) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def to(self, device: torch.device) -> BaseGrid: ...


@dataclass
class Grid(BaseGrid):
    nx: int
    ny: int
    dx: float
    dy: float
    device: torch.device = torch.device("cpu")

    @property
    def x(self) -> torch.Tensor:
        return (torch.arange(self.nx, device=self.device) - self.nx // 2) * self.dx

    @property
    def y(self) -> torch.Tensor:
        return (torch.arange(self.ny, device=self.device) - self.ny // 2) * self.dy

    def meshgrid(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.meshgrid(self.x, self.y, indexing="xy")

    def to(self, device: torch.device) -> Grid:
        return Grid(self.nx, self.ny, self.dx, self.dy, device)
