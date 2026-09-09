from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from fiatlux.optics.elements.base import OpticalElement, validate_field_grid
from fiatlux.core.grid import Grid
from fiatlux.core.field import Field
from fiatlux.core.spectrum import Spectrum
from fiatlux.optics.atmosphere import AtmosphereModel, NCPAModel

from fiatlux.config.registry import register_type

from pathlib import Path

from itertools import cycle

from fiatlux.utils.random import RandomGeneratorMixin

@dataclass
class Mask(OpticalElement, ABC):
    """
    Base mask class.
    Grid (and spectrum for chromatic masks) known at construction.
    Transmission built once at construction, cached.
    """

    grid: Grid
    transmission: torch.Tensor | None = None
    opd: torch.Tensor | None = None
    complex_transmission: torch.Tensor | None = None
    recompute: bool = False

    @abstractmethod
    def _build_transmission(self) -> None: ...

    @abstractmethod
    def _build_opd(self) -> None: ...

    def build(self, spectrum: Spectrum) -> None:
        self._build_transmission()
        self._build_opd()
        expected_spatial_shape = (self.grid.ny, self.grid.nx)
        allowed_shapes = {
            expected_spatial_shape,
            (len(spectrum.wavelengths), *expected_spatial_shape),
        }
        for name in ("transmission", "opd"):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) not in allowed_shapes
            ):
                raise ValueError(
                    f"Mask {name} must have shape (ny, nx) or "
                    f"(n_wavelengths, ny, nx); got "
                    f"{None if not isinstance(value, torch.Tensor) else tuple(value.shape)}."
                )
        self.complex_transmission = self.transmission * torch.exp(
            1j * 2 * torch.pi * self.opd / spectrum.wavelengths[:, None, None]
        )
        expected_shape = (len(spectrum.wavelengths), *expected_spatial_shape)
        if tuple(self.complex_transmission.shape) != expected_shape:
            raise ValueError(
                "Mask complex transmission has incompatible wavelength/spatial "
                f"shape {tuple(self.complex_transmission.shape)}; expected "
                f"{expected_shape}."
            )
        self._built_wavelengths = spectrum.wavelengths.detach().clone()

    def apply(self, field: Field) -> Field:
        validate_field_grid(field, self.grid, self.__class__.__name__)
        wavelengths_changed = (
            not hasattr(self, "_built_wavelengths")
            or self._built_wavelengths.shape != field.spectrum.wavelengths.shape
            or self._built_wavelengths.device != field.spectrum.wavelengths.device
            or self._built_wavelengths.dtype != field.spectrum.wavelengths.dtype
            or not torch.equal(
                self._built_wavelengths, field.spectrum.wavelengths
            )
        )
        if self.complex_transmission is None or self.recompute or wavelengths_changed:
            self.build(field.spectrum)

        return Field(
            field.complex_amplitude * self.complex_transmission,
            field.grid,
            field.spectrum,
        )

    def __repr__(self):
        return "".join(
            (
                self.__class__.__name__,
                "(" f"transmission={self.transmission.type()}, ",
                f"opd={self.opd.type()}, ",
                f"complex_transmission={self.complex_transmission.type()}, ",
                f"recompute={self.recompute}",
                ")",
            )
        )


@register_type("CircularAperture")
class CircularAperture(Mask):
    def __init__(self, grid: Grid, radius: float):
        self.radius = radius
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        x, y = self.grid.meshgrid()
        r = torch.sqrt(x**2 + y**2)
        self.transmission = r <= self.radius

    def _build_opd(self) -> None:
        self.opd = torch.zeros(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    @property
    def _symbol(self) -> str:
        return "O"


@register_type("ArbitraryAperture")
class ArbitraryAperture(Mask):
    def __init__(self, grid: Grid, transmission: torch.Tensor):
        super().__init__(grid=grid)
        self._input_transmission = transmission

    def _build_transmission(self) -> None:
        # Validate size compatibility
        if self._input_transmission.shape != (self.grid.ny, self.grid.nx):
            raise ValueError(
                f"Transmission shape {self._input_transmission.shape} "
                f"does not match grid shape {(self.grid.ny, self.grid.nx)}"
            )

        # Use provided tensor
        self.transmission = self._input_transmission.to(
            device=self.grid.device, dtype=self.grid.dtype
        )

    def _build_opd(self) -> None:
        self.opd = torch.zeros(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    @property
    def _symbol(self) -> str:
        return "O"


@register_type("ZeldaMask")
class ZeldaMask(Mask):
    def __init__(self, grid: Grid, radius: float, well_depth: float):
        self.radius = radius
        self.well_depth = well_depth
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        x, y = self.grid.meshgrid()
        r = torch.sqrt(x**2 + y**2)
        self.opd = self.well_depth * (r <= self.radius)


@register_type("ZeldaStop")
class ZeldaStop(Mask):

    def __init__(self, grid: Grid, radius: float):
        self.radius = radius
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        x, y = self.grid.meshgrid()
        r = torch.sqrt(x**2 + y**2)
        self.transmission = r <= self.radius

    def _build_opd(self) -> None:
        self.opd = torch.zeros(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)


@register_type("Piston")
class Piston(Mask):

    def __init__(self, grid: Grid, piston: float):
        self.piston = piston
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        self.opd = self.piston * torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)


@register_type("Step")
class Step(Mask):

    def __init__(self, grid: Grid, piston: float):
        self.piston = piston
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        tmp = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)
        tmp[:, : self.grid.nx // 2] = 0
        self.opd = self.piston * (torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype) - tmp)


@register_type("TipTilt")
class TipTilt(Mask):

    def __init__(self, grid: Grid, tip: float, tilt: float):
        self.tip = tip
        self.tilt = tilt
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        x, y = self.grid.meshgrid()
        self.opd = self.tip * x + self.tilt * y


class ProuhetThueMorse(Mask):

    def __init__(self, grid: Grid):
        super().__init__(grid=grid)

    def parity_popcount(self, x):
        p = torch.zeros_like(x)

        while x.any():
            p ^= x & 1
            x >>= 1

        return p

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        y, x = torch.meshgrid(
            torch.arange(self.grid.ny, device=self.grid.device),
            torch.arange(self.grid.nx, device=self.grid.device),
            indexing="ij",
        )

        # XOR-based 2D PTM
        Z = x ^ y

        self.opd = 600e-9 * self.parity_popcount(Z).to(self.grid.dtype)


@register_type("ADC")
class ADC(Mask):

    def __init__(
        self, grid: Grid, amplitude: torch.Tensor, angle: float = 0.0
    ):
        self.amplitude = amplitude
        self.angle = angle
        super().__init__(grid=grid)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        x, y = self.grid.meshgrid()

        amplitude = self.amplitude.to(device=self.grid.device, dtype=self.grid.dtype)
        angle = torch.as_tensor(self.angle, device=self.grid.device, dtype=self.grid.dtype)
        self.opd = amplitude[:, None, None] * (
            x * torch.cos(torch.deg2rad(angle))
            + y * torch.sin(torch.deg2rad(angle))
        )


class Atmosphere(RandomGeneratorMixin, Mask):
    def __init__(
        self,
        grid: Grid,
        atmosphere_model: AtmosphereModel,
        recompute: bool = True,
        remove_piston: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ):
        self.atmosphere_model = atmosphere_model
        self.remove_piston = remove_piston
        if seed is None and generator is None:
            self.generator = atmosphere_model.generator
        else:
            self._configure_generator(
                grid.device,
                seed=seed,
                generator=generator,
            )
        super().__init__(grid=grid, recompute=recompute)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(
            (self.grid.ny, self.grid.nx),
            device=self.grid.device,
            dtype=self.atmosphere_model.dtype,
        )

    def _build_opd(self) -> None:
        self.opd = self.atmosphere_model.sample_opd(
            generator=self.generator,
            remove_piston=self.remove_piston
        )


class NCPA(RandomGeneratorMixin, Mask):
    def __init__(
        self,
        grid: Grid,
        ncpa_model: NCPAModel,
        recompute: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ):
        self.ncpa_model = ncpa_model
        if seed is None and generator is None:
            self.generator = ncpa_model.generator
        else:
            self._configure_generator(
                grid.device,
                seed=seed,
                generator=generator,
            )
        super().__init__(grid=grid, recompute=recompute)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(
            (self.grid.ny, self.grid.nx),
            device=self.grid.device,
            dtype=self.ncpa_model.dtype,
        )

    def _build_opd(self) -> None:
        self.opd = self.ncpa_model.sample_opd(generator=self.generator)


class Random(RandomGeneratorMixin, Mask):
    def __init__(
        self,
        grid: Grid,
        amplitude: float,
        recompute: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ):
        self.amplitude = amplitude
        self._configure_generator(
            grid.device,
            seed=seed,
            generator=generator,
        )
        super().__init__(grid=grid, recompute=recompute)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        self.opd = self.amplitude * torch.randn(
            self.grid.shape,
            device=self.grid.device,
            dtype=self.grid.dtype,
            generator=self.generator,
        )


class HarmoniDatasetError(RuntimeError):
    """Base exception for an unusable HARMONI residual dataset."""


class HarmoniDatasetNotFoundError(HarmoniDatasetError, FileNotFoundError):
    """The configured HARMONI dataset directory cannot be read."""


class InvalidHarmoniDatasetError(HarmoniDatasetError, ValueError):
    """A HARMONI dataset exists but does not satisfy the documented format."""


class HarmoniResiduals(RandomGeneratorMixin, Mask):
    """Sequence of HARMONI residual OPD screens and its pupil support.

    ``pupil`` is a boolean transmission mask independent from the individual
    OPD screen being sampled. Supplying an authoritative pupil is preferred.
    For legacy datasets without a separate mask, the fallback support is the
    union of non-zero pixels over every loaded screen, never a single screen.
    ``support`` is an alias for ``pupil``.
    """

    def __init__(
        self,
        grid: Grid,
        dataset_path: str | Path,
        pupil: torch.Tensor | None = None,
        *,
        hdu_index: int = 0,
        opd_plane_index: int | None = 1,
        opd_scale: float = 1.0,
        rotate_quarter_turns: int = 1,
        shuffle: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ):
        self.grid = grid
        self.dataset_path = Path(dataset_path).expanduser()
        self.hdu_index = hdu_index
        self.opd_plane_index = opd_plane_index
        self.opd_scale = float(opd_scale)
        self.rotate_quarter_turns = rotate_quarter_turns
        self.files = self._dataset_files(self.dataset_path)
        self.load_datacube(
            self.files,
            hdu_index=hdu_index,
            opd_plane_index=opd_plane_index,
            opd_scale=self.opd_scale,
            rotate_quarter_turns=rotate_quarter_turns,
        )
        self.pupil = self._prepare_pupil(pupil)

        self._configure_generator(
            "cpu",
            seed=seed,
            generator=generator,
        )
        indices = torch.arange(len(self.datacube))
        if shuffle:
            indices = indices[
                torch.randperm(len(indices), generator=self.generator)
            ]
        self._indices = indices.tolist()
        self.iterator = iter(cycle(self._indices))

        super().__init__(grid=grid, recompute=True)

    def _prepare_pupil(self, pupil: torch.Tensor | None) -> torch.Tensor:
        if self.datacube.ndim != 3 or tuple(self.datacube.shape[-2:]) != self.grid.shape:
            raise ValueError(
                "HARMONI residual datacube must have shape "
                f"(n_screens, {self.grid.ny}, {self.grid.nx}); got "
                f"{tuple(self.datacube.shape)}."
            )

        if pupil is None:
            pupil = torch.any(self.datacube != 0, dim=0)
        else:
            pupil = torch.as_tensor(pupil)
            if tuple(pupil.shape) != self.grid.shape:
                raise ValueError(
                    f"HARMONI pupil must have grid shape {self.grid.shape}; "
                    f"got {tuple(pupil.shape)}."
                )
            if pupil.dtype != torch.bool and not torch.all((pupil == 0) | (pupil == 1)):
                raise ValueError("HARMONI pupil must be boolean or contain only 0 and 1.")

        return pupil.to(device=self.grid.device, dtype=torch.bool)

    @property
    def support(self) -> torch.Tensor:
        """Alias for the public boolean :attr:`pupil` mask."""
        return self.pupil

    @staticmethod
    def _dataset_files(path: Path) -> list[Path]:
        if not path.exists():
            raise HarmoniDatasetNotFoundError(
                f"HARMONI dataset directory does not exist: {path}"
            )
        if not path.is_dir():
            raise HarmoniDatasetNotFoundError(
                f"HARMONI dataset path is not a directory: {path}"
            )
        files = sorted(
            item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".fits"
        )
        if not files:
            raise InvalidHarmoniDatasetError(
                f"HARMONI dataset directory contains no FITS files: {path}"
            )
        return files

    def load_datacube(
        self,
        files: list[Path],
        *,
        hdu_index: int,
        opd_plane_index: int | None,
        opd_scale: float,
        rotate_quarter_turns: int,
    ) -> None:
        try:
            from astropy.io import fits
        except ImportError as error:
            raise ImportError(
                "HARMONI FITS data require Astropy; install it with "
                "`python -m pip install 'fiatlux[fits]'`."
            ) from error
        if not isinstance(hdu_index, int) or isinstance(hdu_index, bool) or hdu_index < 0:
            raise ValueError("hdu_index must be a non-negative integer.")
        if opd_plane_index is not None and (
            not isinstance(opd_plane_index, int)
            or isinstance(opd_plane_index, bool)
            or opd_plane_index < 0
        ):
            raise ValueError("opd_plane_index must be None or a non-negative integer.")
        if not torch.isfinite(torch.tensor(opd_scale)) or opd_scale == 0:
            raise ValueError("opd_scale must be finite and non-zero.")
        if not isinstance(rotate_quarter_turns, int) or isinstance(
            rotate_quarter_turns, bool
        ):
            raise ValueError("rotate_quarter_turns must be an integer.")

        datacube = []
        for file in files:
            with fits.open(file, memmap=False) as hdul:
                if hdu_index >= len(hdul):
                    raise InvalidHarmoniDatasetError(
                        f"{file}: missing HDU index {hdu_index}."
                    )
                data = hdul[hdu_index].data
                if data is None:
                    raise InvalidHarmoniDatasetError(
                        f"{file}: HDU {hdu_index} contains no data."
                    )
                if data.dtype.kind != "f":
                    raise InvalidHarmoniDatasetError(
                        f"{file}: residual OPD data must have a floating dtype; "
                        f"got {data.dtype}."
                    )
                if opd_plane_index is None:
                    if data.ndim != 3:
                        raise InvalidHarmoniDatasetError(
                            f"{file}: expected shape (n_screens, ny, nx) when "
                            f"opd_plane_index=None; got {data.shape}."
                        )
                    selected = data
                else:
                    if data.ndim != 4 or opd_plane_index >= data.shape[0]:
                        raise InvalidHarmoniDatasetError(
                            f"{file}: expected shape (n_products, n_screens, ny, nx) "
                            f"containing product {opd_plane_index}; got {data.shape}."
                        )
                    selected = data[opd_plane_index]

                if selected.shape[0] < 1:
                    raise InvalidHarmoniDatasetError(
                        f"{file}: residual OPD cube contains no screens."
                    )

                native = selected.astype(selected.dtype.newbyteorder("="), copy=True)
                cube = torch.from_numpy(native).to(dtype=self.grid.dtype)
                cube = torch.rot90(
                    cube, k=rotate_quarter_turns % 4, dims=(-2, -1)
                )
                if tuple(cube.shape[-2:]) != self.grid.shape:
                    raise InvalidHarmoniDatasetError(
                        f"{file}: transformed residual shape {tuple(cube.shape[-2:])} "
                        f"does not match grid shape {self.grid.shape}."
                    )
                if not torch.isfinite(cube).all():
                    raise InvalidHarmoniDatasetError(
                        f"{file}: residual OPD data contain NaN or infinite values."
                    )
                datacube.append(cube * opd_scale)
        self.datacube = torch.cat(datacube, dim=0)

    def _build_transmission(self) -> None:
        self.transmission = torch.ones(self.grid.shape, device=self.grid.device, dtype=self.grid.dtype)

    def _build_opd(self) -> None:
        self.opd = self.datacube[next(self.iterator), ...].to(
            device=self.grid.device,
            dtype=self.grid.dtype,
        )
