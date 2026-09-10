"""Optional bridge between Fiatlux fields and the external FATMOSS backend."""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch

from fiatlux.core.grid import Grid


class FatmossUnavailableError(ImportError):
    """FATMOSS was requested but its source modules cannot be imported."""


class FatmossBackend(Protocol):
    """Narrow subset of FATMOSS consumed by Fiatlux."""

    def GetScreenByTimestep(self, timestep: int) -> Any: ...

    def AddLayer(self, layer: Any) -> None: ...

    def reset(self, regenerate_layers: bool = True) -> None: ...


@dataclass(frozen=True)
class FrozenFlowLayer:
    """Physical configuration of one FATMOSS frozen-flow layer."""

    r0: float
    outer_scale: float
    wind_speed: float
    wind_direction: float
    weight: float = 1.0
    altitude: float = 0.0

    def __post_init__(self) -> None:
        for name in ("r0", "outer_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite metres.")
        if not math.isfinite(self.wind_speed) or self.wind_speed < 0:
            raise ValueError("wind_speed must be finite and non-negative m/s.")
        if not math.isfinite(self.wind_direction):
            raise ValueError("wind_direction must be finite degrees.")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("weight must be positive and finite.")
        if not math.isfinite(self.altitude) or self.altitude < 0:
            raise ValueError("altitude must be finite and non-negative metres.")


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
        advance_after_sample: bool = True,
        time_step: float | None = None,
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
        if time_step is not None and (
            not math.isfinite(time_step) or time_step <= 0
        ):
            raise ValueError("time_step must be positive and finite seconds.")
        if not callable(getattr(backend, "GetScreenByTimestep", None)):
            raise TypeError("FATMOSS backend must define GetScreenByTimestep(timestep).")

        self.grid = grid
        self.backend = backend
        self.reference_wavelength = float(reference_wavelength)
        self.screen_unit = screen_unit
        self.source_axes = source_axes
        self.timestep = initial_timestep
        self.advance_after_sample = bool(advance_after_sample)
        self.time_step = None if time_step is None else float(time_step)
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
            time_step=time_step,
        )

    @classmethod
    def create_frozen_flow(
        cls,
        grid: Grid,
        layers: Sequence[FrozenFlowLayer],
        *,
        time_step: float,
        batch_size: int = 100,
        n_cascades: int = 3,
        seed: int | None = None,
        reference_wavelength: float = 500e-9,
        normalize_weights: bool = False,
        phase_generator_module: Any | None = None,
        layer_module: Any | None = None,
    ) -> FatmossAtmosphereModel:
        """Create a non-boiling FATMOSS sequence from physical layer configs."""
        if not layers:
            raise ValueError("At least one frozen-flow layer is required.")
        if not all(isinstance(layer, FrozenFlowLayer) for layer in layers):
            raise TypeError("layers must contain FrozenFlowLayer instances.")
        if not isinstance(normalize_weights, bool):
            raise TypeError("normalize_weights must be a boolean.")
        model = cls.create(
            grid,
            time_step=time_step,
            batch_size=batch_size,
            n_cascades=n_cascades,
            seed=seed,
            reference_wavelength=reference_wavelength,
            module=phase_generator_module,
        )
        if layer_module is None:
            try:
                layer_module = importlib.import_module("atmospheric_layer")
            except ImportError as error:
                raise FatmossUnavailableError(
                    "FATMOSS atmospheric_layer.py is not importable."
                ) from error
        layer_factory = getattr(layer_module, "Layer", None)
        von_karman = getattr(layer_module, "vonKarmanPSD", None)
        simple_boiling = getattr(layer_module, "SimpleBoiling", None)
        if not all(callable(item) for item in (layer_factory, von_karman, simple_boiling)):
            raise FatmossUnavailableError(
                "FATMOSS atmospheric_layer must expose Layer, vonKarmanPSD and SimpleBoiling."
            )

        total_weight = sum(config.weight for config in layers)
        weight_scale = 1.0 / total_weight if normalize_weights else 1.0
        wavelength_nm = reference_wavelength * 1e9
        for config in layers:
            spatial_psd = lambda frequency, c=config: von_karman(
                frequency, c.r0, c.outer_scale, wavelength_nm
            )
            temporal_psd = lambda frequency: simple_boiling(frequency, grid.dx)
            backend_layer = layer_factory(
                config.weight * weight_scale,
                config.altitude,
                config.wind_speed,
                config.wind_direction,
                0.0,
                spatial_psd,
                temporal_psd,
            )
            model.backend.AddLayer(backend_layer)
        model.advance_after_sample = False
        return model

    @property
    def current_time(self) -> float:
        """Physical time in seconds at the currently selected screen."""
        if self.time_step is None:
            raise RuntimeError("This model has no configured physical time_step.")
        return self.timestep * self.time_step

    def sample_opd(
        self,
        *,
        generator: torch.Generator | None = None,
        remove_piston: bool = True,
    ) -> torch.Tensor:
        """Return OPD in metres, advancing only for legacy sequential models."""
        if generator is not None:
            raise ValueError(
                "FATMOSS owns its random state; seed it when constructing the backend."
            )
        screen = self.current_opd(remove_piston=remove_piston)
        if self.advance_after_sample:
            self.advance()
        return screen

    def current_opd(self, *, remove_piston: bool = True) -> torch.Tensor:
        """Return OPD at the selected timestep without advancing time."""
        raw = self.backend.GetScreenByTimestep(self.timestep)
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

    def current_phase(self, *, remove_piston: bool = True) -> torch.Tensor:
        """Return phase in radians at ``reference_wavelength`` without advancing."""
        return self.current_opd(remove_piston=remove_piston) * (
            2.0 * math.pi / self.reference_wavelength
        )

    def advance(self, steps: int = 1) -> None:
        """Advance the temporal state independently from optical propagation."""
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("steps must be a non-negative integer.")
        self.timestep += steps

    def reset(self) -> None:
        """Reset FATMOSS' seeded state and select the first physical instant."""
        backend_reset = getattr(self.backend, "reset", None)
        if not callable(backend_reset):
            raise NotImplementedError("This FATMOSS backend does not support reset().")
        backend_reset(regenerate_layers=True)
        self.timestep = 0

    def iter_opd(
        self,
        number: int | None = None,
        *,
        remove_piston: bool = True,
    ) -> Iterator[torch.Tensor]:
        """Yield screens lazily while retaining only FATMOSS' current batch."""
        if number is not None and (
            not isinstance(number, int) or isinstance(number, bool) or number < 0
        ):
            raise ValueError("number must be a non-negative integer or None.")
        produced = 0
        while number is None or produced < number:
            screen = self.current_opd(remove_piston=remove_piston)
            self.advance()
            produced += 1
            yield screen

    def sequence_opd(
        self, number: int, *, advance: bool = True, remove_piston: bool = True
    ) -> torch.Tensor:
        """Return ``number`` consecutive OPD screens shaped ``(time, ny, nx)``."""
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise ValueError("number must be a positive integer.")
        start = self.timestep
        screens = []
        try:
            for offset in range(number):
                self.timestep = start + offset
                screens.append(self.current_opd(remove_piston=remove_piston))
        finally:
            self.timestep = start + number if advance else start
        return torch.stack(screens)

    def opd_at(self, time: float, *, remove_piston: bool = True) -> torch.Tensor:
        """Return OPD at an exact cadence time without changing temporal state."""
        start = self.timestep
        try:
            self.seek_time(time)
            return self.current_opd(remove_piston=remove_piston)
        finally:
            self.timestep = start

    def phase_at(self, time: float, *, remove_piston: bool = True) -> torch.Tensor:
        """Return phase in radians at an exact cadence time without advancing."""
        return self.opd_at(time, remove_piston=remove_piston) * (
            2.0 * math.pi / self.reference_wavelength
        )

    def seek_time(self, time: float) -> None:
        """Select a physical time that lies on the configured FATMOSS cadence."""
        if self.time_step is None:
            raise RuntimeError("This model has no configured physical time_step.")
        if not math.isfinite(time) or time < 0:
            raise ValueError("time must be finite and non-negative seconds.")
        timestep = round(time / self.time_step)
        if not math.isclose(
            time, timestep * self.time_step, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError("time must be an integer multiple of time_step.")
        self.seek(timestep)

    def seek(self, timestep: int) -> None:
        """Select the absolute FATMOSS timestep returned by the next call."""
        if not isinstance(timestep, int) or isinstance(timestep, bool) or timestep < 0:
            raise ValueError("timestep must be a non-negative integer.")
        self.timestep = timestep
