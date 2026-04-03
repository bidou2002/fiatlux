from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
from typing import Callable

from fiatlux.grid import Grid
from fiatlux.field import Field
from fiatlux.detector import Detector
from fiatlux.optical_system import SerialSystem
from fiatlux.spectrum import Spectrum
from fiatlux.optical_system import SimulationResult


@dataclass
class ActuatorGrid:
    """Physical layout of DM actuators."""

    n_actuators_x: int
    n_actuators_y: int
    pitch: float  # m between actuators


@dataclass
class ControlBasis(ABC):

    @abstractmethod
    def build_command_matrix(self): ...

    @property
    @abstractmethod
    def n_modes(self): ...


@dataclass
class ZonalBasis(ControlBasis):
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
class FourierBasis(ControlBasis):
    pixel_grid: Grid
    order: int

    def build_command_matrix(
        self,
    ) -> torch.Tensor:

        x, y = self.pixel_grid.meshgrid()  # (nx, ny)
        fx = fy = torch.arange(self.order)

        D = self.pixel_grid.nx * self.pixel_grid.dx
        scale = 2 * torch.pi / D

        # Build all (fx, fy) pairs
        FX, FY = torch.meshgrid(fx, fy, indexing="ij")

        FX = FX.flatten()
        FY = FY.flatten()

        arg = scale * (FX[:, None, None] * x + FY[:, None, None] * y)

        cos_modes = (torch.cos(arg) + 1) / 2
        sin_modes = (torch.sin(arg) + 1) / 2

        modes = torch.cat([cos_modes, sin_modes], dim=0)
        modes = modes.flatten(start_dim=1).T
        mask = torch.ones(2 * len(fx) ** 2, dtype=torch.bool)
        mask[0] = False
        mask[len(fx) ** 2] = False

        return modes[:, mask]

    def n_modes(self):
        return 2 * (self.order**2 - 1)


class DeformableMirror(torch.nn.Module):
    """
    Deformable mirror — phase set by actuator voltages.

    The phase map is interpolated from actuator commands
    onto the pixel grid of the optical system.
    """

    def __init__(
        self,
        actuator_grid: ActuatorGrid,
        pixel_grid: Grid,
        control_basis: ControlBasis,
        stroke: float = 1e-6,  # max phase stroke (m)
        influence_width: float = 1.5,  # actuator influence function width (in actuator pitches)
    ):
        torch.nn.Module.__init__(self)
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

    def _build(self, grid: Grid, spectrum: Spectrum) -> None:
        self.complex_transmission = torch.exp(
            1j * 2 * torch.pi * self.opd / spectrum.wavelengths[:, None, None]
        )

    def apply(self, field: Field) -> Field:
        self._build(field.grid, field.spectrum)

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


class InteractionMatrix:
    """
    Measures the interaction matrix of a DeformableMirror in an OpticalSystem.

    For each actuator i :
        1. Poke actuator i with amplitude ε
        2. Run the simulation
        3. Record the response (image or pupil field) as column i

    Result : M of shape (n_pixels_response, n_actuators)
    """

    def __init__(
        self,
        system: SerialSystem,
        dm: DeformableMirror,
        poke_amplitude: float = 0.1,  # in units of dm.stroke
        response_function: Callable[[SimulationResult], Field] | None = None,
    ):
        self.system = system
        self.dm = dm
        self.poke_amplitude = poke_amplitude
        self.response_function = response_function
        self.matrix: torch.Tensor | None = None  # (n_response, n_actuators)

    def _get_response(self) -> torch.Tensor:
        """Run the system and return the flattened response vector."""
        result = self.system.run()
        self.system.detector.acquire(self.response_function(result))

        return self.system.detector.image_buffer.flatten()  # (nx*ny,)

    def measure(self, verbose: bool = True) -> torch.Tensor:
        """
        Poke each actuator and record the response.
        Returns the interaction matrix of shape (n_response, n_actuators).
        """
        n_actuators = self.dm._commands.numel()
        columns = []

        # Save current DM state and flatten
        saved_commands = self.dm._commands.data.clone()
        self.dm.flatten()

        with torch.no_grad():
            for i in range(n_actuators):
                if verbose:
                    print(f"Poking actuator {i+1}/{n_actuators}", end="\r")

                # Poke actuator i
                self.dm._commands.data[i] = self.poke_amplitude

                # Record response
                response = self._get_response()
                columns.append(response)

                # Reset actuator
                self.dm._commands.data[i] = 0.0

        # Restore original DM state
        self.dm._commands.data.copy_(saved_commands)

        self.matrix = torch.stack(columns, dim=1)  # (n_response, n_actuators)

        if verbose:
            print(f"\nInteraction matrix : {self.matrix.shape}")

        return self.matrix

    def push_pull(self, verbose: bool = True) -> torch.Tensor:
        """
        Push-pull measurement for better linearity :
        column_i = (response(+ε) - response(-ε)) / (2ε)
        Cancels static aberrations and nonlinearities.
        """
        n_actuators = self.dm._commands.numel()
        columns = []

        saved_commands = self.dm._commands.data.clone()
        self.dm.flatten()

        with torch.no_grad():
            for i in range(n_actuators):
                if verbose:
                    print(f"Push-pull actuator {i+1}/{n_actuators}", end="\r")

                # Push
                self.dm._commands.data[i] = +self.poke_amplitude
                r_plus = self._get_response()

                # Pull
                self.dm._commands.data[i] = -self.poke_amplitude
                r_minus = self._get_response()

                # Differential response
                columns.append((r_plus - r_minus) / (2 * self.poke_amplitude))

                self.dm._commands.data[i] = 0.0

        self.dm._commands.data.copy_(saved_commands)
        self.matrix = torch.stack(columns, dim=1)  # (n_response, n_actuators)

        if verbose:
            print(f"\nPush-pull matrix : {self.matrix.shape}")

        return self.matrix

    def control_matrix(self, n_modes: int | None = None) -> torch.Tensor:
        """
        Computes the pseudo-inverse (control matrix) via SVD.
        n_modes : number of singular modes to keep (truncated SVD).
        Returns the control matrix of shape (n_actuators, n_response).
        """
        if self.matrix is None:
            raise RuntimeError("Call measure() or push_pull() first.")

        U, S, Vh = torch.linalg.svd(self.matrix, full_matrices=False)

        if n_modes is not None:
            U, S, Vh = U[:, :n_modes], S[:n_modes], Vh[:n_modes, :]

        return Vh.T @ torch.diag(1.0 / S) @ U.T  # (n_actuators, n_response)
