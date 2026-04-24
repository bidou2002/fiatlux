from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
from typing import Callable

from fiatlux.core.source import Source
from fiatlux.optics.detector import Detector
from fiatlux.core.field import Field
from fiatlux.system.optical_system import SerialSystem
from fiatlux.system.optical_system import SimulationResult
from fiatlux.optics.elements.deformable_mirror import DeformableMirror


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
        dm: DeformableMirror,
        poke_amplitude: float = 0.1,  # in units of dm.stroke
        acquiring_function: Callable = lambda res: res.image,
    ):
        self.dm = dm
        self.poke_amplitude = poke_amplitude
        self.acquiring_function = acquiring_function
        self.matrix: torch.Tensor | None = None  # (n_response, n_actuators)

    def _get_response(self) -> torch.Tensor:
        """Run the system and return the flattened response vector."""
        return self.acquiring_function().flatten()

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
