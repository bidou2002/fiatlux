import math
import torch
from typing import Callable

import matplotlib.pyplot as plt

from fiatlux.optics.elements.deformable_mirror import DeformableMirror


class InteractionMatrix:
    """Calibrate the local response to DM commands expressed in metres OPD.

    ``acquiring_function`` owns the forward model and takes no argument. It
    must return the measurement tensor produced by the current DM state.
    Images and arbitrary-dimensional measurements are accepted and flattened
    only when columns are assembled.

    The resulting matrix has shape ``(n_measurements, n_commands)`` and units
    measurement-unit per metre of OPD.
    """

    def __init__(
        self,
        dm: DeformableMirror,
        acquiring_function: Callable[[], torch.Tensor],
        poke_amplitude: float = 10e-9,
        reference_commands: torch.Tensor | None = None,
    ) -> None:
        if not math.isfinite(poke_amplitude) or poke_amplitude <= 0:
            raise ValueError("poke_amplitude must be a positive finite OPD in metres.")
        if not callable(acquiring_function):
            raise TypeError("acquiring_function must be callable without arguments.")

        self.dm = dm
        self.poke_amplitude = float(poke_amplitude)
        self.acquiring_function = acquiring_function
        if reference_commands is None:
            reference_commands = torch.zeros_like(dm.commands)
        else:
            reference_commands = torch.as_tensor(
                reference_commands,
                device=dm.commands.device,
                dtype=dm.commands.dtype,
            )
        if reference_commands.shape != dm.commands.shape:
            raise ValueError(
                "reference_commands must have shape "
                f"{tuple(dm.commands.shape)}, got {tuple(reference_commands.shape)}."
            )
        if not torch.isfinite(reference_commands).all():
            raise ValueError("reference_commands must contain only finite values.")
        if torch.any(reference_commands.abs() + self.poke_amplitude > dm.stroke):
            raise ValueError(
                "reference_commands +/- poke_amplitude exceed the DM stroke."
            )

        self.reference_commands = reference_commands.detach().clone()
        self.reference_response: torch.Tensor | None = None
        self.measurement_shape: torch.Size | None = None
        self._measurement_dtype: torch.dtype | None = None
        self._measurement_device: torch.device | None = None
        self.matrix: torch.Tensor | None = None

    def _get_response(self, expected_shape: torch.Size | None = None) -> torch.Tensor:
        """Acquire and validate one measurement without changing its shape."""
        response = self.acquiring_function()
        if not isinstance(response, torch.Tensor):
            raise TypeError("acquiring_function must return a torch.Tensor.")
        if response.numel() == 0:
            raise ValueError("acquiring_function returned an empty measurement.")
        if not torch.isfinite(response).all():
            raise ValueError("acquiring_function returned non-finite values.")
        if expected_shape is not None and response.shape != expected_shape:
            raise ValueError(
                "Measurement shape changed during calibration: expected "
                f"{tuple(expected_shape)}, got {tuple(response.shape)}."
            )
        if self._measurement_dtype is not None and response.dtype != self._measurement_dtype:
            raise ValueError("Measurement dtype changed during calibration.")
        if self._measurement_device is not None and response.device != self._measurement_device:
            raise ValueError("Measurement device changed during calibration.")
        if self._measurement_dtype is None:
            self._measurement_dtype = response.dtype
            self._measurement_device = response.device
        return response.detach().clone()

    def _calibrate(self, *, central: bool, verbose: bool) -> torch.Tensor:
        """Run one-sided or central finite-difference calibration."""
        n_commands = self.dm.commands.numel()
        columns = []
        saved_commands = self.dm.commands.detach().clone()
        self.reference_response = None
        self.measurement_shape = None
        self._measurement_dtype = None
        self._measurement_device = None

        with torch.no_grad():
            try:
                self.dm.commands = self.reference_commands
                if not central:
                    reference = self._get_response()
                    self.reference_response = reference
                    self.measurement_shape = reference.shape

                for i in range(n_commands):
                    if verbose:
                        method = "Push-pull" if central else "One-sided"
                        print(f"{method} command {i + 1}/{n_commands}", end="\r")

                    plus_commands = self.reference_commands.clone()
                    plus_commands[i] += self.poke_amplitude
                    self.dm.commands = plus_commands
                    response_plus = self._get_response(self.measurement_shape)

                    if central:
                        if self.measurement_shape is None:
                            self.measurement_shape = response_plus.shape
                        minus_commands = self.reference_commands.clone()
                        minus_commands[i] -= self.poke_amplitude
                        self.dm.commands = minus_commands
                        response_minus = self._get_response(self.measurement_shape)
                        column = (response_plus - response_minus) / (
                            2.0 * self.poke_amplitude
                        )
                    else:
                        column = (response_plus - reference) / self.poke_amplitude

                    columns.append(column.flatten())
            finally:
                self.dm.commands = saved_commands

        self.matrix = torch.stack(columns, dim=1)

        if verbose:
            print(f"\nInteraction matrix: {self.matrix.shape}")

        return self.matrix

    def calibrate_one_sided(self, verbose: bool = True) -> torch.Tensor:
        """Calibrate ``(response(c0 + eps) - response(c0)) / eps``."""
        return self._calibrate(central=False, verbose=verbose)

    def calibrate_push_pull(self, verbose: bool = True) -> torch.Tensor:
        """Calibrate the central derivative around the reference commands."""
        return self._calibrate(central=True, verbose=verbose)

    def compute_control_matrix(self, n_modes: int = None):
        """
        Computes the pseudo-inverse (control matrix) via SVD.
        n_modes : number of singular modes to keep (truncated SVD).
        Returns the control matrix of shape (n_actuators, n_response).
        """
        if self.matrix is None:
            raise RuntimeError(
                "Call calibrate_one_sided() or calibrate_push_pull() first."
            )

        self.U, self.S, self.Vh = torch.linalg.svd(self.matrix, full_matrices=False)

        if n_modes is not None:
            self.U, self.S, self.Vh = (
                self.U[:, :n_modes],
                self.S[:n_modes],
                self.Vh[:n_modes, :],
            )

        self.control_matrix = (
            self.Vh.T @ torch.diag(1.0 / self.S) @ self.U.T
        )  # (n_actuators, n_response)

    def plot(self):

        plt.figure()
        plt.plot(self.S)
        plt.yscale("log")
        plt.xlabel("Mode index")
        plt.ylabel("Singular value")
        plt.title("Singular values of the interaction matrix")
        plt.grid()
        plt.draw()

        N = self.dm._commands.numel()
        n = torch.tensor(N**0.5).ceil().int().item()

        fig, axes = plt.subplots(n, n, figsize=(10, 10))
        fig_out, axes_out = plt.subplots(n, n, figsize=(10, 10))

        n_reshape = int(self.control_matrix.shape[1] ** 0.5)

        for i in range(N):

            ax = axes[i // n, i % n]
            ax_out = axes_out[i // n, i % n]

            mode = self.U[:, i].reshape(n_reshape, n_reshape)
            mode_in = self.matrix[:, i].reshape(n_reshape, n_reshape)

            ax.imshow(mode_in)
            ax.set_title(f"Mode {i}")
            ax.axis("off")

            ax_out.imshow(mode)
            ax_out.set_title(f"Mode {i}")
            ax_out.axis("off")

        plt.tight_layout()
        plt.draw()
