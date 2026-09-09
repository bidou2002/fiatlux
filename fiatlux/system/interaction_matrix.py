import math
import torch
from typing import Callable

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

    @property
    def singular_values(self) -> torch.Tensor:
        """Complete singular-value spectrum from the latest inversion."""
        if not hasattr(self, "S"):
            raise RuntimeError("Call compute_control_matrix() first.")
        return self.S

    @property
    def condition_number(self) -> float:
        """Condition number of the retained singular subspace."""
        if not hasattr(self, "retained_mode_indices"):
            raise RuntimeError("Call compute_control_matrix() first.")
        if self.effective_rank == 0:
            return math.inf
        retained = self.S[self.retained_mode_indices]
        return (retained.max() / retained.min()).item()

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

    def compute_control_matrix(
        self,
        n_modes: int | None = None,
        *,
        rcond: float | None = None,
        atol: float = 0.0,
    ) -> torch.Tensor:
        """Compute a filtered SVD pseudo-inverse.

        Singular values must be larger than both ``atol`` and
        ``rcond * S.max()``. When ``rcond`` is omitted, the same
        dimension-scaled machine-precision default as a conventional
        pseudo-inverse is used. ``n_modes`` can additionally cap the number
        of retained modes, ordered from largest to smallest singular value.
        """
        if self.matrix is None:
            raise RuntimeError(
                "Call calibrate_one_sided() or calibrate_push_pull() first."
            )

        if n_modes is not None and (
            not isinstance(n_modes, int)
            or isinstance(n_modes, bool)
            or n_modes < 1
        ):
            raise ValueError("n_modes must be a positive integer or None.")
        for name, value in (("rcond", rcond), ("atol", atol)):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite non-negative value.")

        self.U, self.S, self.Vh = torch.linalg.svd(self.matrix, full_matrices=False)
        if rcond is None:
            rcond = max(self.matrix.shape) * torch.finfo(self.S.dtype).eps

        relative_threshold = rcond * self.S.max()
        self.singular_value_threshold = max(
            torch.as_tensor(atol, device=self.S.device, dtype=self.S.dtype),
            relative_threshold,
        )
        retained = self.S > self.singular_value_threshold
        if n_modes is not None:
            mode_cap = torch.arange(self.S.numel(), device=self.S.device) < n_modes
            retained &= mode_cap

        self.retained_mode_indices = torch.nonzero(retained, as_tuple=False).flatten()
        self.effective_rank = int(self.retained_mode_indices.numel())

        inverse_singular_values = torch.zeros_like(self.S)
        inverse_singular_values[retained] = self.S[retained].reciprocal()
        self.control_matrix = (
            self.Vh.T @ torch.diag(inverse_singular_values) @ self.U.T
        )
        return self.control_matrix

    def plot_singular_values(self):
        """Plot the full spectrum and mark the retained SVD modes."""
        import matplotlib.pyplot as plt

        singular_values = self.singular_values.detach().cpu()
        fig, ax = plt.subplots()
        ax.semilogy(singular_values, marker="o", label="all modes")
        if self.effective_rank:
            indices = self.retained_mode_indices.detach().cpu()
            ax.semilogy(
                indices,
                singular_values[indices],
                linestyle="none",
                marker="o",
                label="retained",
            )
        ax.axhline(
            self.singular_value_threshold.detach().cpu().item(),
            color="tab:red",
            linestyle="--",
            label="threshold",
        )
        ax.set_xlabel("Mode index")
        ax.set_ylabel("Singular value")
        ax.set_title("Interaction-matrix singular values")
        ax.grid(True)
        ax.legend()
        return fig, ax

    def plot_modes(
        self,
        measurement_shape: tuple[int, ...] | torch.Size | None = None,
        *,
        max_modes: int | None = None,
    ):
        """Plot retained measurement modes as curves or 2-D images.

        A one-dimensional measurement is plotted as a curve. A two-dimensional
        measurement is displayed as an image. Higher-dimensional measurements
        require the caller to provide a one- or two-dimensional display shape.
        """
        import matplotlib.pyplot as plt

        _ = self.singular_values
        shape = self.measurement_shape if measurement_shape is None else measurement_shape
        if shape is None:
            shape = (self.matrix.shape[0],)
        shape = tuple(shape)
        if len(shape) not in (1, 2) or math.prod(shape) != self.matrix.shape[0]:
            raise ValueError(
                "measurement_shape must be one- or two-dimensional and contain "
                f"{self.matrix.shape[0]} values."
            )
        if max_modes is not None and (
            not isinstance(max_modes, int)
            or isinstance(max_modes, bool)
            or max_modes < 1
        ):
            raise ValueError("max_modes must be a positive integer or None.")

        indices = self.retained_mode_indices
        if max_modes is not None:
            indices = indices[:max_modes]
        n_plots = max(1, indices.numel())
        n_columns = min(4, n_plots)
        n_rows = math.ceil(n_plots / n_columns)
        fig, axes = plt.subplots(
            n_rows, n_columns, squeeze=False, figsize=(3 * n_columns, 3 * n_rows)
        )
        for ax in axes.flat:
            ax.set_visible(False)
        for ax, index in zip(axes.flat, indices.tolist()):
            ax.set_visible(True)
            mode = self.U[:, index].detach().cpu().reshape(shape)
            if len(shape) == 1:
                ax.plot(mode)
                ax.set_xlabel("Measurement index")
            else:
                ax.imshow(mode.numpy())
                ax.axis("off")
            ax.set_title(f"Mode {index}")
        fig.tight_layout()
        return fig, axes

    def plot(self, measurement_shape=None, *, max_modes=None):
        """Compatibility helper returning singular-value and mode figures."""
        singular_figure = self.plot_singular_values()
        mode_figure = self.plot_modes(measurement_shape, max_modes=max_modes)
        return singular_figure, mode_figure
