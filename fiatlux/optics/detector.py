from dataclasses import dataclass
import math
from fiatlux.core.dimensions import FieldDimension
import torch

from fiatlux.core.grid import Grid
from fiatlux.core.field import Field
from fiatlux.optics.elements.base import validate_field_grid
from fiatlux.utils.random import RandomGeneratorMixin


class Detector(RandomGeneratorMixin):
    def __init__(
        self,
        grid: Grid,
        exposure_time: float = 0.150,
        quantum_efficiency: float = 1,
        photon_noise: bool = False,
        readout_noise_variance: float = 0,
        dark_current: float = 0,
        offset: int = 0,
        bitdepth: int = 16,
        digitize: bool = False,
        sensitivity: float = 1.0,
        random_seed: int | None = None,
        generator: torch.Generator | None = None,
        name: str = "",
    ):
        if exposure_time < 0:
            raise ValueError("exposure_time must be non-negative.")
        if not 0 <= quantum_efficiency <= 1:
            raise ValueError("quantum_efficiency must be between 0 and 1.")
        if readout_noise_variance < 0:
            raise ValueError("readout_noise_variance must be non-negative.")
        if dark_current < 0:
            raise ValueError("dark_current must be non-negative.")
        self.grid = grid
        self.exposure_time = exposure_time
        self.quantum_efficiency = quantum_efficiency
        self.photon_noise = photon_noise
        self.readout_noise_variance = readout_noise_variance
        self.dark_current = dark_current
        self.offset = offset
        if not isinstance(bitdepth, int) or isinstance(bitdepth, bool) or bitdepth < 1:
            raise ValueError("bitdepth must be a positive integer.")
        self.bitdepth = bitdepth
        self.digitize = digitize
        self.sensitivity = sensitivity
        self.random_seed = random_seed
        self._configure_generator(
            grid.device,
            seed=random_seed,
            generator=generator,
        )
        self.name = name
        self.image_buffer = None

    def _integrated_counts(self, field: Field, integrate_over: str):
        validate_field_grid(field, self.grid, "Detector")
        dimension = field.dimension(integrate_over)
        weights = dimension.integration_weights
        if dimension.unit != "s" or weights is None:
            raise ValueError(
                "Exposure integration requires explicit integration_weights in seconds (unit='s')."
            )
        if (
            dimension.size == 0
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError(
                "Exposure weights must be finite, non-negative and non-empty."
            )
        duration = weights.sum()
        if duration <= 0:
            raise ValueError("Exposure duration must be positive.")
        axis = field.axis(integrate_over)
        rate = field.intensity().sum(dim=-3)
        shape = [1] * rate.ndim
        shape[axis] = dimension.size
        counts = (rate * weights.reshape(shape)).sum(axis) * (
            self.grid.dx * self.grid.dy
        )
        remaining = field.dimensions[:axis] + field.dimensions[axis + 1 :]
        return counts, remaining, duration

    def _validate_duration(self, duration):
        if not math.isclose(
            (
                float(duration.detach())
                if isinstance(duration, torch.Tensor)
                else duration
            ),
            self.exposure_time,
            rel_tol=1e-6,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Sum of integration weights must match detector exposure_time."
            )

    def _readout(self, counts, dimensions):
        data = self.add_noise(counts)
        self.image_buffer = DetectorImage(data, dimensions) if dimensions else data
        return self.image_buffer

    def acquire(self, field: Field, *, integrate_over: str | None = None):
        """Read electrons/ADUs after one exposure.

        Without a named reduction each latent sample is an independent exposure
        of ``exposure_time``. With a reduction, weights in seconds must sum to
        ``exposure_time``; they replace its intensity multiplier.
        """
        validate_field_grid(field, self.grid, "Detector")
        if integrate_over is None:
            counts = (
                field.intensity().sum(dim=-3)
                * (self.grid.dx * self.grid.dy)
                * self.exposure_time
            )
            remaining = field.dimensions
        else:
            counts, remaining, duration = self._integrated_counts(field, integrate_over)
            self._validate_duration(duration)
        return self._readout(counts, remaining)

    def add_photon_noise(self, expected_electrons: torch.Tensor) -> torch.Tensor:
        """Draw detected photoelectrons from their Poisson distribution."""
        return torch.poisson(
            expected_electrons,
            generator=self.generator,
        )

    def add_dark_noise(self, electrons: torch.Tensor):
        """Add Poisson dark electrons from a rate in e-/pixel/s."""
        expected_dark_electrons = torch.full_like(
            electrons,
            self.dark_current * self.exposure_time,
        )
        dark_noise = torch.poisson(
            expected_dark_electrons,
            generator=self.generator,
        )
        return electrons + dark_noise

    def add_readout_noise(self, electrons: torch.Tensor):
        """Add zero-mean Gaussian read noise with variance in electrons²."""
        noise = torch.normal(
            mean=torch.zeros_like(electrons),
            std=self.readout_noise_variance**0.5,
            generator=self.generator,
        )
        return electrons + noise

    def photons_to_electrons(self, photons: torch.Tensor):
        return self.quantum_efficiency * photons

    def electrons_to_adus(self, electrons: torch.Tensor) -> torch.Tensor:
        """Digitize electrons into integer ADUs in the configured ADC range.

        Gain and offset are applied before values are clipped to
        ``[0, 2**bitdepth - 1]``. ``int64`` is used so a 16-bit ADC can safely
        represent its full range, including 65535.
        """
        max_adu = 2**self.bitdepth - 1
        adus = torch.floor(electrons * self.sensitivity) + self.offset
        return adus.clamp(0, max_adu).to(torch.int64)

    def add_noise(self, photon_counts: torch.Tensor) -> torch.Tensor:
        """Convert integrated photon counts through the detector pipeline."""
        electrons = self.photons_to_electrons(photons=photon_counts)
        if self.photon_noise:
            electrons = self.add_photon_noise(expected_electrons=electrons)

        # add the dark noise to electrons
        electrons = self.add_dark_noise(electrons=electrons)

        # add the computed noise to electrons
        electrons = self.add_readout_noise(electrons=electrons)

        # Convert to ADU and add baseline
        if self.digitize:
            return self.electrons_to_adus(electrons=electrons)

        return electrons


@dataclass(frozen=True)
class DetectorImage:
    """Electrons (or ADUs) with unresolved latent axes and final spatial axes."""

    data: torch.Tensor
    dimensions: tuple[FieldDimension, ...]


class ExposureAccumulator:
    """Accumulate expected counts and read once in finish().

    ``track_grad=False`` detaches each chunk for bounded memory; ``True``
    retains its graph and therefore consumes memory across the whole exposure.
    Chunks must have identical remaining metadata, spectrum, grid and dtype.
    The caller supplies non-overlapping chunks from the desired exposure.
    """

    def __init__(self, detector: Detector, *, integrate_over="time", track_grad=False):
        self.detector = detector
        self.integrate_over = integrate_over
        self.track_grad = track_grad
        self._total = None
        self._duration = 0.0
        self._dimensions = None
        self._spectrum = None
        self._finished = False
        self._settings = self._detector_settings()

    def _detector_settings(self):
        d = self.detector
        return (
            d.grid,
            d.exposure_time,
            d.quantum_efficiency,
            d.photon_noise,
            d.readout_noise_variance,
            d.dark_current,
            d.offset,
            d.bitdepth,
            d.digitize,
            d.sensitivity,
        )

    def _check_open(self):
        if self._finished:
            raise RuntimeError("ExposureAccumulator has already finished.")
        if self._settings != self._detector_settings():
            raise ValueError("Detector settings changed during exposure.")

    def add(self, field: Field):
        self._check_open()
        with torch.set_grad_enabled(self.track_grad):
            counts, dimensions, duration = self.detector._integrated_counts(
                field, self.integrate_over
            )
        spectrum = (field.spectrum.wavelengths, field.spectrum.fluxes)
        if self._total is not None:
            if (
                dimensions != self._dimensions
                or counts.shape != self._total.shape
                or counts.dtype != self._total.dtype
                or counts.device != self._total.device
            ):
                raise ValueError(
                    "Chunk remaining dimensions, shape, device and dtype must match."
                )
            if any(
                a.device != b.device or a.dtype != b.dtype or not torch.equal(a, b)
                for a, b in zip(spectrum, self._spectrum)
            ):
                raise ValueError("Chunk spectra must match.")
        new_duration = self._duration + float(duration.detach())
        if new_duration > self.detector.exposure_time and not math.isclose(
            new_duration, self.detector.exposure_time, rel_tol=1e-6, abs_tol=1e-12
        ):
            raise ValueError("Chunks exceed detector exposure_time.")
        if not self.track_grad:
            counts = counts.detach()
        with torch.set_grad_enabled(self.track_grad):
            total = counts if self._total is None else self._total + counts
        self._total, self._dimensions, self._duration = total, dimensions, new_duration
        self._spectrum = tuple(v.detach().clone() for v in spectrum)
        return self

    def finish(self):
        self._check_open()
        if self._total is None:
            raise ValueError("Cannot finish an empty exposure.")
        self.detector._validate_duration(self._duration)
        result = self.detector._readout(self._total, self._dimensions)
        self._finished = True
        self._total = None
        return result
