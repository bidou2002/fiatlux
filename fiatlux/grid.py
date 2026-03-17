from dataclasses import dataclass
import torch


@dataclass
class Grid:
    """Grille de discrétisation spatiale définie par l'utilisateur."""

    nx: int  # nombre de pixels en x
    ny: int  # nombre de pixels en y
    dx: float  # pas spatial en x (m)
    dy: float  # pas spatial en y (m)
    device: torch.device = torch.device("cpu")

    @property
    def x(self) -> torch.Tensor:
        return (torch.arange(self.nx, device=self.device) - self.nx // 2) * self.dx

    @property
    def y(self) -> torch.Tensor:
        return (torch.arange(self.ny, device=self.device) - self.ny // 2) * self.dy

    def meshgrid(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.meshgrid(self.x, self.y, indexing="xy")

    def to(self, device: torch.device) -> "Grid":
        return Grid(self.nx, self.ny, self.dx, self.dy, device)
