from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
from typing import Callable

from fiatlux.core.grid import Grid
from fiatlux.core.field import Field
from fiatlux.core.spectrum import Spectrum
from fiatlux.utils.zernike import zernike_basis
from fiatlux.optics.elements.mask import Mask

from fiatlux.config.registry import register_type


@dataclass
@register_type("ActuatorGrid")
class ActuatorGrid:
    """Physical layout of DM actuators."""

    n_actuators_x: int
    n_actuators_y: int
    pitch: float  # m between actuators


@dataclass
@register_type("ControlBasis")
class ControlBasis(ABC):

    @abstractmethod
    def build_command_matrix(self): ...

    @property
    @abstractmethod
    def n_modes(self): ...


@dataclass
@register_type("GaussianZonalBasis")
class GaussianZonalBasis(ControlBasis):
    actuator_grid: ActuatorGrid
    pixel_grid: Grid
    influence_width: float

    def build_command_matrix(self):
        """
        Gaussian influence function for each actuator onto the pixel grid.
        Shape : (nx*ny, n_actuators)
        """
        ag = self.actuator_grid
        pg = self.pixel_grid
        x, y = pg.meshgrid()  # (nx, ny)
        x_flat = x.flatten()  # (nx*ny,)
        y_flat = y.flatten()

        # Actuator positions in meters
        ax = (torch.arange(ag.n_actuators_x) - ag.n_actuators_x // 2 + 1 / 2) * ag.pitch
        ay = (torch.arange(ag.n_actuators_y) - ag.n_actuators_y // 2 + 1 / 2) * ag.pitch
        ax_grid, ay_grid = torch.meshgrid(ax, ay, indexing="xy")
        ax_flat = ax_grid.flatten()  # (n_actuators,)
        ay_flat = ay_grid.flatten()

        sigma = ag.pitch * self.influence_width

        # Distance from each pixel to each actuator : (nx*ny, n_actuators)
        dx = x_flat[:, None] - ax_flat[None, :]
        dy = y_flat[:, None] - ay_flat[None, :]
        r2 = dx**2 + dy**2

        influence = torch.exp(-r2 / (2 * sigma**2))

        # Normalize so each actuator has unit peak response
        influence = influence / influence.max(dim=0).values.clamp(min=1e-12)
        return influence

    def n_modes(self):
        return self.actuator_grid.n_actuators_x * self.actuator_grid.n_actuators_y


@dataclass
@register_type("SquareZonalBasis")
class SquareZonalBasis(ControlBasis):
    actuator_grid: ActuatorGrid
    pixel_grid: Grid
    influence_width: float

    def build_command_matrix(self):
        """
        Gaussian influence function for each actuator onto the pixel grid.
        Shape : (nx*ny, n_actuators)
        """
        ag = self.actuator_grid
        pg = self.pixel_grid
        x, y = pg.meshgrid()  # (nx, ny)
        x_flat = x.flatten()  # (nx*ny,)
        y_flat = y.flatten()

        # Actuator positions in meters
        ax = (torch.arange(ag.n_actuators_x) - ag.n_actuators_x // 2 + 1 / 2) * ag.pitch
        ay = (torch.arange(ag.n_actuators_y) - ag.n_actuators_y // 2 + 1 / 2) * ag.pitch
        ax_grid, ay_grid = torch.meshgrid(ax, ay, indexing="xy")
        ax_flat = ax_grid.flatten()  # (n_actuators,)
        ay_flat = ay_grid.flatten()

        # Distance from each pixel to each actuator : (nx*ny, n_actuators)
        dx = x_flat[:, None] - ax_flat[None, :]
        dy = y_flat[:, None] - ay_flat[None, :]

        influence = torch.zeros(self.pixel_grid.ny * self.pixel_grid.nx, self.n_modes())
        influence[
            (torch.abs(dx) < self.influence_width)
            & (torch.abs(dy) < self.influence_width)
        ] = 1

        return influence

    def n_modes(self):
        return self.actuator_grid.n_actuators_x * self.actuator_grid.n_actuators_y


@dataclass
@register_type("SquarePTTZonalBasis")
class SquarePTTZonalBasis(ControlBasis):
    actuator_grid: ActuatorGrid
    pixel_grid: Grid
    influence_width: float

    def build_command_matrix(self):
        """
        Gaussian influence function for each actuator onto the pixel grid.
        Shape : (nx*ny, n_actuators)
        """
        ag = self.actuator_grid
        pg = self.pixel_grid
        x, y = pg.meshgrid()  # (nx, ny)
        x_flat = x.flatten()  # (nx*ny,)
        y_flat = y.flatten()

        # Actuator positions in meters
        ax = (torch.arange(ag.n_actuators_x) - ag.n_actuators_x // 2 + 1 / 2) * ag.pitch
        ay = (torch.arange(ag.n_actuators_y) - ag.n_actuators_y // 2 + 1 / 2) * ag.pitch
        ax_grid, ay_grid = torch.meshgrid(ax, ay, indexing="xy")
        ax_flat = ax_grid.flatten()  # (n_actuators,)
        ay_flat = ay_grid.flatten()

        # Distance from each pixel to each actuator : (nx*ny, n_actuators)
        dx = x_flat[:, None] - ax_flat[None, :]
        dy = y_flat[:, None] - ay_flat[None, :]

        piston = torch.zeros(dx.shape)
        piston[
            (torch.abs(dx) < self.influence_width)
            & (torch.abs(dy) < self.influence_width)
        ] = 1

        tip = dx
        tip[
            (torch.abs(dx) < self.influence_width)
            & (torch.abs(dy) < self.influence_width)
        ] = 1

        tilt = dy
        tilt[
            (torch.abs(dx) < self.influence_width)
            & (torch.abs(dy) < self.influence_width)
        ] = 1

        influence = torch.cat([piston, tip, tilt], dim=1)

        return influence

    def n_modes(self):
        return 3 * self.actuator_grid.n_actuators_x * self.actuator_grid.n_actuators_y


@dataclass
@register_type("FourierBasis")
class FourierBasis(ControlBasis):
    pixel_grid: Grid
    frequencies: torch.Tensor  # spatial frequencies in cycles per aperture
    pupil: Mask

    def build_command_matrix(self):
        n_freqs = len(self.frequencies)

        # Pixel coordinate grids  (resolution, resolution)
        x, y = self.pixel_grid.meshgrid()  # (nx, ny)

        # All (freq_x, freq_y) pairs  →  (n_freqs, n_freqs)
        freq_x = self.frequencies[:, None].expand(n_freqs, n_freqs)
        freq_y = self.frequencies[None, :].expand(n_freqs, n_freqs)

        # Phase for every pair and every pixel  →  (n_freqs, n_freqs, resolution, resolution)
        phase = (
            2 * torch.pi * (freq_x[:, :, None, None] * x + freq_y[:, :, None, None] * y)
        )

        # Cosine for freq_x < 0, or freq_x == 0 and freq_y <= 0 (avoids cos/sin redundancy)
        # use_cosine = (freq_x < 0) | ((freq_x <= 0) & (freq_y >= 0))
        use_cosine = torch.full((n_freqs, n_freqs), True)
        center_freq = n_freqs**2 / 2
        use_cosine[: int(center_freq // n_freqs), :] = False
        use_cosine[int(center_freq // n_freqs), : int(center_freq % n_freqs)] = False

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.imshow(use_cosine)
        # Minor ticks
        ax.set_xticks(torch.arange(-0.5, n_freqs, 1), minor=True)
        ax.set_yticks(torch.arange(-0.5, n_freqs, 1), minor=True)
        # Gridlines based on minor ticks
        ax.grid(which="minor", color="w", linestyle="-", linewidth=2)

        modes = torch.where(
            use_cosine[:, :, None, None], torch.cos(phase), torch.sin(phase)
        )

        modes = (modes * self.pupil.transmission[None, None, ...]) / (
            modes * self.pupil.transmission[None, None, ...]
        ).pow(2).sum(dim=(2, 3), keepdim=True).sqrt()

        return modes.reshape(n_freqs**2, self.pixel_grid.nx * self.pixel_grid.ny).T

    # def build_command_matrix(
    #     self,
    # ) -> torch.Tensor:

    #     x, y = self.pixel_grid.meshgrid()  # (nx, ny)
    #     fx = fy = torch.arange(self.order)
    #     fx = fy = torch.arange(self.order) - self.order // 2

    #     D = self.pixel_grid.nx * self.pixel_grid.dx
    #     scale = 2 * torch.pi / D

    #     # Build all (fx, fy) pairs
    #     FX, FY = torch.meshgrid(fx, fy, indexing="ij")

    #     FX = FX.flatten()
    #     FY = FY.flatten()

    #     arg = scale * (FX[:, None, None] * x + FY[:, None, None] * y)

    #     cos_modes = (torch.cos(arg) + 1) / 2
    #     sin_modes = (torch.sin(arg) + 1) / 2

    #     modes = torch.cat([cos_modes, sin_modes], dim=0)
    #     modes = modes.flatten(start_dim=1).T
    #     mask = torch.ones(2 * len(fx) ** 2, dtype=torch.bool)
    #     mask[0] = False
    #     mask[len(fx) ** 2] = False

    #     return modes[:, mask]

    def n_modes(self):
        return len(self.frequencies) ** 2


@dataclass
@register_type("ZernikeBasis")
class ZernikeBasis(ControlBasis):
    pixel_grid: Grid
    n: int

    def build_command_matrix(
        self,
    ) -> torch.Tensor:

        modes = zernike_basis(
            nterms=self.n + 1,
            npix=self.pixel_grid.nx,
        )

        return modes[1:, ...].flatten(1, -1).T

    def n_modes(self):
        return self.n


@register_type("DeformableMirror")
class DeformableMirror(torch.nn.Module):
    """
    Deformable mirror — phase set by actuator voltages.

    The phase map is interpolated from actuator commands
    onto the pixel grid of the optical system.
    """

    def __init__(
        self,
        grid: Grid,
        actuator_grid: ActuatorGrid,
        pixel_grid: Grid,
        control_basis: ControlBasis,
        stroke: float = 1e-6,  # max phase stroke (m)
        influence_width: float = 1.5,  # actuator influence function width (in actuator pitches)
    ):
        torch.nn.Module.__init__(self)
        self.grid = grid
        self.actuator_grid = actuator_grid
        self.pixel_grid = pixel_grid
        self.control_basis = control_basis
        self.stroke = stroke
        self.influence_width = influence_width
        self.complex_transmission: torch.Tensor | None = None

        # Actuator commands ∈ [-1, 1] via tanh — maps to [-stroke, +stroke]
        self._commands = torch.zeros(self.control_basis.n_modes())

        # Precompute influence matrix (actuators → pixels) — fixed geometry
        self._command_matrix = self.control_basis.build_command_matrix()

    @property
    def commands(self) -> torch.Tensor:
        """Actuator commands ∈ [-1, 1]."""
        return self._commands

    @property
    def opd(self) -> torch.Tensor:
        """
        OPD map on the pixel grid (nx, ny) in meters.
        Obtained by interpolating actuator commands via the influence matrix.
        """
        # commands : (n_actuators,)
        # influence_matrix : (nx*ny, n_actuators)
        # opd : (nx*ny,)
        opd = self._command_matrix @ (self.commands * self.stroke)
        return opd.reshape(self.pixel_grid.nx, self.pixel_grid.ny)

    def _build(self, spectrum: Spectrum) -> None:
        self.complex_transmission = torch.exp(
            1j * 2 * torch.pi * self.opd / spectrum.wavelengths[:, None, None]
        )

    def apply(self, field: Field) -> Field:
        self._build(field.spectrum)

        return Field(
            field.complex_amplitude * self.complex_transmission,
            field.grid,
            field.spectrum,
        )

    def to_slm(self, slm: "SLM") -> None:
        """Print the DM phase onto an SLM (emulation)."""
        with torch.no_grad():
            # Inverse tanh to set raw parameter — clamp for numerical safety
            phase_clamped = self.phase.clamp(-torch.pi + 1e-4, torch.pi - 1e-4)
            slm._raw_phase.copy_(torch.atanh(phase_clamped / torch.pi))

    def flatten(self) -> None:
        """Reset all actuators to zero."""
        with torch.no_grad():
            self._commands.zero_()

    def plot(self) -> None:
        """Visualize the DM surface."""
        import matplotlib.pyplot as plt

        plt.imshow(
            self.opd,
            extent=[
                -self.pixel_grid.dy * self.pixel_grid.ny / 2,
                self.pixel_grid.dy * self.pixel_grid.ny / 2,
                -self.pixel_grid.dx * self.pixel_grid.nx / 2,
                self.pixel_grid.dx * self.pixel_grid.nx / 2,
            ],
            origin="lower",
        )
        plt.colorbar(label="OPD (m)")
        plt.title("Deformable Mirror OPD")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
