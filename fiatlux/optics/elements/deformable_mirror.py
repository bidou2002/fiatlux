from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
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

    def positions(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return x/y actuator positions centred on the optical axis."""
        ax = (
            torch.arange(self.n_actuators_x, device=device, dtype=dtype)
            - (self.n_actuators_x - 1) / 2
        ) * self.pitch
        ay = (
            torch.arange(self.n_actuators_y, device=device, dtype=dtype)
            - (self.n_actuators_y - 1) / 2
        ) * self.pitch
        return ax, ay


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
        x, y = pg.meshgrid()  # (ny, nx)
        x_flat = x.flatten()  # (nx*ny,)
        y_flat = y.flatten()

        # Actuator positions in meters
        ax, ay = ag.positions(device=pg.device, dtype=x.dtype)
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

    @property
    def n_modes(self) -> int:
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
        x, y = pg.meshgrid()  # (ny, nx)
        x_flat = x.flatten()  # (nx*ny,)
        y_flat = y.flatten()

        # Actuator positions in meters
        ax, ay = ag.positions(device=pg.device, dtype=x.dtype)
        ax_grid, ay_grid = torch.meshgrid(ax, ay, indexing="xy")
        ax_flat = ax_grid.flatten()  # (n_actuators,)
        ay_flat = ay_grid.flatten()

        # Distance from each pixel to each actuator : (nx*ny, n_actuators)
        dx = x_flat[:, None] - ax_flat[None, :]
        dy = y_flat[:, None] - ay_flat[None, :]

        influence = torch.zeros(
            self.pixel_grid.ny * self.pixel_grid.nx,
            self.n_modes,
            device=pg.device,
            dtype=pg.dtype,
        )
        influence[
            (torch.abs(dx) < self.influence_width)
            & (torch.abs(dy) < self.influence_width)
        ] = 1

        return influence

    @property
    def n_modes(self) -> int:
        return self.actuator_grid.n_actuators_x * self.actuator_grid.n_actuators_y


@dataclass
@register_type("SquarePTTZonalBasis")
class SquarePTTZonalBasis(ControlBasis):
    actuator_grid: ActuatorGrid
    pixel_grid: Grid
    influence_width: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.influence_width) or self.influence_width <= 0:
            raise ValueError("influence_width must be a positive finite length.")

    def build_command_matrix(self):
        """Build dimensionless piston/tip/tilt modes on square supports.

        Columns are ordered as all piston modes, then all x-tip modes, then all
        y-tilt modes. The matrix shape is ``(ny * nx, 3 * n_actuators)``.
        """
        ag = self.actuator_grid
        pg = self.pixel_grid
        x, y = pg.meshgrid()  # (ny, nx)
        x_flat = x.flatten()  # (nx*ny,)
        y_flat = y.flatten()

        # Actuator positions in meters
        ax, ay = ag.positions(device=pg.device, dtype=x.dtype)
        ax_grid, ay_grid = torch.meshgrid(ax, ay, indexing="xy")
        ax_flat = ax_grid.flatten()  # (n_actuators,)
        ay_flat = ay_grid.flatten()

        # Distance from each pixel to each actuator : (nx*ny, n_actuators)
        dx = x_flat[:, None] - ax_flat[None, :]
        dy = y_flat[:, None] - ay_flat[None, :]

        support = (torch.abs(dx) < self.influence_width) & (
            torch.abs(dy) < self.influence_width
        )
        piston = support.to(dx.dtype)
        tip = torch.where(support, dx / self.influence_width, 0.0)
        tilt = torch.where(support, dy / self.influence_width, 0.0)

        influence = torch.cat([piston, tip, tilt], dim=1)

        return influence

    @property
    def n_modes(self) -> int:
        return 3 * self.actuator_grid.n_actuators_x * self.actuator_grid.n_actuators_y


@dataclass
@register_type("FourierBasis")
class FourierBasis(ControlBasis):
    pixel_grid: Grid
    frequencies: torch.Tensor  # spatial frequencies in cycles per aperture
    pupil: Mask

    def build_command_matrix(self):
        frequencies = self.frequencies.to(
            device=self.pixel_grid.device, dtype=self.pixel_grid.dtype
        )
        n_freqs = len(frequencies)

        # Pixel coordinate grids  (resolution, resolution)
        x, y = self.pixel_grid.meshgrid()  # (ny, nx)

        # All (freq_x, freq_y) pairs  →  (n_freqs, n_freqs)
        freq_x = frequencies[:, None].expand(n_freqs, n_freqs)
        freq_y = frequencies[None, :].expand(n_freqs, n_freqs)

        # Phase for every pair and every pixel  →  (n_freqs, n_freqs, resolution, resolution)
        phase = (
            2 * torch.pi * (freq_x[:, :, None, None] * x + freq_y[:, :, None, None] * y)
        )

        # Cosine for freq_x < 0, or freq_x == 0 and freq_y <= 0 (avoids cos/sin redundancy)
        # use_cosine = (freq_x < 0) | ((freq_x <= 0) & (freq_y >= 0))
        use_cosine = torch.full(
            (n_freqs, n_freqs), True, device=self.pixel_grid.device
        )
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

    @property
    def n_modes(self) -> int:
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
        ).to(device=self.pixel_grid.device, dtype=self.pixel_grid.dtype)

        return modes[1:, ...].flatten(1, -1).T

    @property
    def n_modes(self) -> int:
        return self.n


@register_type("DeformableMirror")
class DeformableMirror(torch.nn.Module):
    """
    Deformable mirror controlled by modal OPD coefficients.

    Public ``commands`` are coefficients in metres of optical path difference.
    ``stroke`` is the maximum absolute OPD coefficient accepted by each mode.
    Values beyond that range are explicitly saturated before application.
    """

    def __init__(
        self,
        grid: Grid,
        actuator_grid: ActuatorGrid,
        pixel_grid: Grid,
        control_basis: ControlBasis,
        stroke: float = 1e-6,  # maximum absolute modal OPD coefficient (m)
        influence_width: float = 1.5,  # actuator influence function width (in actuator pitches)
    ):
        torch.nn.Module.__init__(self)
        self.grid = grid
        self.actuator_grid = actuator_grid
        self.pixel_grid = pixel_grid
        self.control_basis = control_basis
        if stroke <= 0:
            raise ValueError("stroke must be positive and expressed in metres of OPD.")
        self.stroke = float(stroke)
        self.influence_width = influence_width
        self.register_buffer("complex_transmission", None, persistent=False)

        # Raw and applied commands are OPD coefficients in metres. Keeping one
        # registered Parameter preserves differentiability and nn.Module.to().
        self._commands = torch.nn.Parameter(
            torch.zeros(
                self.control_basis.n_modes,
                device=self.pixel_grid.device,
                dtype=self.pixel_grid.dtype,
            )
        )

        # Precompute influence matrix (actuators → pixels) — fixed geometry
        self.register_buffer(
            "_command_matrix",
            self.control_basis.build_command_matrix().to(
                device=self.pixel_grid.device, dtype=self.pixel_grid.dtype
            ),
        )

    def _apply(self, fn, recurse=True):
        """Apply PyTorch transfers and keep grid device metadata consistent."""
        super()._apply(fn, recurse=recurse)
        device = self._commands.device
        dtype = self._commands.dtype
        self.grid = self.grid.to(device, dtype)
        self.pixel_grid = self.pixel_grid.to(device, dtype)
        if hasattr(self.control_basis, "pixel_grid"):
            self.control_basis.pixel_grid = self.control_basis.pixel_grid.to(device, dtype)
        return self

    @property
    def commands(self) -> torch.Tensor:
        """Requested modal OPD coefficients in metres."""
        return self._commands

    @property
    def applied_commands(self) -> torch.Tensor:
        """Modal OPD coefficients after physical stroke saturation."""
        return self._commands.clamp(-self.stroke, self.stroke)

    @commands.setter
    def commands(self, value: torch.Tensor) -> None:
        value = torch.as_tensor(
            value,
            device=self._commands.device,
            dtype=self._commands.dtype,
        )
        if value.shape != self._commands.shape:
            raise ValueError(
                f"commands must have shape {tuple(self._commands.shape)}, "
                f"got {tuple(value.shape)}."
            )
        if not torch.isfinite(value).all():
            raise ValueError("commands must contain only finite OPD values.")
        with torch.no_grad():
            self._commands.copy_(value)

    @property
    def opd(self) -> torch.Tensor:
        """
        OPD map on the pixel grid ``(ny, nx)`` in metres.
        Obtained by interpolating actuator commands via the influence matrix.
        """
        # commands : (n_actuators,)
        # influence_matrix : (nx*ny, n_actuators)
        # opd : (nx*ny,)
        opd = self._command_matrix @ self.applied_commands
        return opd.reshape(self.pixel_grid.ny, self.pixel_grid.nx)

    def _build(self, spectrum: Spectrum) -> None:
        self.complex_transmission = torch.exp(
            1j * 2 * torch.pi * self.opd / spectrum.wavelengths[:, None, None]
        )

    def apply(self, field: Field) -> Field:
        if field.grid != self.pixel_grid:
            raise ValueError("DM pixel grid must match the incoming field grid.")
        self._build(field.spectrum)

        return Field(
            field.complex_amplitude * self.complex_transmission,
            field.grid,
            field.spectrum,
        )

    def to_slm(self, slm: "SLM") -> None:
        """Reject an undefined generic conversion to hardware-specific units."""
        raise NotImplementedError(
            "Generic DM-to-SLM conversion is undefined. Export OPD commands "
            "with export_commands() and convert them using the calibrated "
            "hardware adapter."
        )

    def export_commands(self) -> torch.Tensor:
        """Return an independent CPU copy of applied commands in metres OPD."""
        return self.applied_commands.detach().cpu().clone()

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
