"""Optional bridge between Fiatlux fields and the external FATMOSS backend."""

from __future__ import annotations

import importlib
import math
from typing import Any, Literal, Protocol

import torch

from fiatlux.core.grid import Grid


class FatmossUnavailableError(ImportError):
    """FATMOSS was requested but its source modules cannot be imported."""


class FatmossBackend(Protocol):
    """Narrow subset of FATMOSS consumed by Fiatlux."""

    def GetScreenByTimestep(self, timestep: int) -> Any: ...


class FatmossAtmosphereModel:
    """Expose sequential FATMOSS screens as Fiatlux OPD screens.

    FATMOSS currently returns optical-path screens in nanometres and stores
    spatial arrays in ``(x, y)`` order. Fiatlux uses metres and ``(ny, nx)``.
    Both conventions are explicit and configurable here so no unit or axis
    conversion is hidden in :class:`~fiatlux.optics.elements.mask.Atmosphere`.
    """

    def __init__(
        self,
        grid: Grid,
        backend: FatmossBackend,
        *,
        reference_wavelength: float = 500e-9,
        screen_unit: Literal["nm", "m"] = "nm",
        source_axes: Literal["xy", "yx"] = "xy",
        initial_timestep: int = 0,
    ) -> None:
        if not math.isfinite(reference_wavelength) or reference_wavelength <= 0:
            raise ValueError("reference_wavelength must be positive and finite.")
        if screen_unit not in ("nm", "m"):
            raise ValueError("screen_unit must be 'nm' or 'm'.")
        if source_axes not in ("xy", "yx"):
            raise ValueError("source_axes must be 'xy' or 'yx'.")
        if (
            not isinstance(initial_timestep, int)
            or isinstance(initial_timestep, bool)
            or initial_timestep < 0
        ):
            raise ValueError("initial_timestep must be a non-negative integer.")
        if not callable(getattr(backend, "GetScreenByTimestep", None)):
            raise TypeError("FATMOSS backend must define GetScreenByTimestep(timestep).")

        self.grid = grid
        self.backend = backend
        self.reference_wavelength = float(reference_wavelength)
        self.screen_unit = screen_unit
        self.source_axes = source_axes
        self.timestep = initial_timestep
        self.dtype = grid.dtype
        # ``Atmosphere`` forwards this value to ``sample_opd``. FATMOSS owns
        # its seeded RNG, so no independent torch.Generator is advertised.
        self.generator = None

    @classmethod
    def create(
        cls,
        grid: Grid,
        *,
        time_step: float,
        batch_size: int = 100,
        n_cascades: int = 3,
        seed: int | None = None,
        reference_wavelength: float = 500e-9,
        module: Any | None = None,
    ) -> FatmossAtmosphereModel:
        """Construct FATMOSS' ``PhaseScreensGenerator`` on a Fiatlux grid.

        The upstream generator currently supports square isotropic samplings.
        ``module`` is an injection point used by tests and alternate checkouts.
        """
        if grid.nx != grid.ny or grid.dx != grid.dy:
            raise ValueError("FATMOSS requires a square grid with dx == dy.")
        if not math.isfinite(time_step) or time_step <= 0:
            raise ValueError("time_step must be positive and finite seconds.")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(n_cascades, int) or isinstance(n_cascades, bool) or n_cascades < 1:
            raise ValueError("n_cascades must be a positive integer.")

        if module is None:
            try:
                module = importlib.import_module("phase_generator")
            except ImportError as error:
                raise FatmossUnavailableError(
                    "FATMOSS is optional and is not importable. Clone "
                    "https://github.com/EjjeSynho/FATMOSS and add its source "
                    "directory to PYTHONPATH before invoking this feature."
                ) from error
        factory = getattr(module, "PhaseScreensGenerator", None)
        if not callable(factory):
            raise FatmossUnavailableError(
                "The imported FATMOSS module has no PhaseScreensGenerator."
            )
        backend = factory(
            D=grid.nx * grid.dx,
            dx=grid.dx,
            dt=time_step,
            batch_size=batch_size,
            n_cascades=n_cascades,
            seed=seed,
            double_precision=grid.dtype == torch.float64,
        )
        return cls(
            grid,
            backend,
            reference_wavelength=reference_wavelength,
            screen_unit="nm",
            source_axes="xy",
        )

    def sample_opd(
        self,
        *,
        generator: torch.Generator | None = None,
        remove_piston: bool = True,
    ) -> torch.Tensor:
        """Return the next FATMOSS screen as OPD in metres on the Fiatlux grid."""
        if generator is not None:
            raise ValueError(
                "FATMOSS owns its random state; seed it when constructing the backend."
            )
        raw = self.backend.GetScreenByTimestep(self.timestep)
        self.timestep += 1
        if isinstance(raw, torch.Tensor):
            screen = raw
        elif hasattr(raw, "__dlpack__"):
            screen = torch.from_dlpack(raw)
        else:
            screen = torch.as_tensor(raw)
        if self.source_axes == "xy":
            screen = screen.transpose(-2, -1)
        if tuple(screen.shape) != self.grid.shape:
            raise ValueError(
                f"FATMOSS screen has converted shape {tuple(screen.shape)}; "
                f"expected Fiatlux grid shape {self.grid.shape}."
            )
        screen = screen.to(device=self.grid.device, dtype=self.dtype)
        if self.screen_unit == "nm":
            screen = screen * 1e-9
        if remove_piston:
            screen = screen - screen.mean()
        return screen

    def seek(self, timestep: int) -> None:
        """Select the absolute FATMOSS timestep returned by the next call."""
        if not isinstance(timestep, int) or isinstance(timestep, bool) or timestep < 0:
            raise ValueError("timestep must be a non-negative integer.")
        self.timestep = timestep
