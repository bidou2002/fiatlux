"""Vendor-independent calibration and registration helpers for phase SLMs.

This module deliberately stops at an integer image tensor.  Communication,
BMP serialization, camera acquisition, timing, and safe device shutdown belong
in the experiment-specific hardware wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as functional

from fiatlux.core.grid import Grid
from fiatlux.optics.elements.deformable_mirror import SquareZonalBasis


@dataclass(frozen=True)
class SLMCalibration:
    """Measured conversion from phase delay to integer SLM drive level.

    ``phase_radians`` must be a strictly increasing one-dimensional table
    covering at least the interval [0, 2*pi].  ``drive_levels`` contains the
    corresponding, monotonically increasing device values.  The calibration
    is wavelength-specific; polarization and other bench metadata should be
    managed alongside the calibration file by the experiment.
    """

    wavelength: float
    drive_levels: torch.Tensor
    phase_radians: torch.Tensor
    bit_depth: int = 8

    def __post_init__(self) -> None:
        if not math.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("wavelength must be a positive finite value in metres.")
        if not isinstance(self.bit_depth, int) or not 1 <= self.bit_depth <= 16:
            raise ValueError("bit_depth must be an integer between 1 and 16.")

        levels = torch.as_tensor(self.drive_levels)
        phases = torch.as_tensor(self.phase_radians)
        if levels.ndim != 1 or phases.ndim != 1 or levels.shape != phases.shape:
            raise ValueError(
                "drive_levels and phase_radians must be one-dimensional tables "
                "with identical shapes."
            )
        if phases.numel() < 2:
            raise ValueError("The calibration table must contain at least two points.")
        if not torch.isfinite(levels).all() or not torch.isfinite(phases).all():
            raise ValueError("The calibration table must contain only finite values.")
        if not bool(torch.all(phases[1:] > phases[:-1])):
            raise ValueError("phase_radians must be strictly increasing.")
        if not bool(torch.all(levels[1:] >= levels[:-1])):
            raise ValueError("drive_levels must be monotonically increasing.")
        if float(phases[0]) > 0 or float(phases[-1]) < 2 * math.pi:
            raise ValueError("phase_radians must cover the complete [0, 2*pi] range.")
        maximum = 2**self.bit_depth - 1
        if float(levels[0]) < 0 or float(levels[-1]) > maximum:
            raise ValueError(f"drive_levels must lie in [0, {maximum}].")

        object.__setattr__(self, "drive_levels", levels)
        object.__setattr__(self, "phase_radians", phases)

    @property
    def maximum_level(self) -> int:
        return 2**self.bit_depth - 1

    def phase_to_drive(self, phase: torch.Tensor) -> torch.Tensor:
        """Wrap phase modulo 2*pi and interpolate calibrated drive levels."""
        if not isinstance(phase, torch.Tensor) or not phase.is_floating_point():
            raise TypeError("phase must be a floating-point torch.Tensor.")
        if not torch.isfinite(phase).all():
            raise ValueError("phase must contain only finite values.")

        phases = self.phase_radians.to(device=phase.device, dtype=phase.dtype)
        levels = self.drive_levels.to(device=phase.device, dtype=phase.dtype)
        wrapped = torch.remainder(phase, 2 * torch.pi)
        upper = torch.searchsorted(phases, wrapped, right=False)
        upper = upper.clamp(1, phases.numel() - 1)
        lower = upper - 1
        fraction = (wrapped - phases[lower]) / (phases[upper] - phases[lower])
        drive = levels[lower] + fraction * (levels[upper] - levels[lower])
        output_dtype = torch.uint8 if self.bit_depth <= 8 else torch.int32
        return drive.round().clamp(0, self.maximum_level).to(output_dtype)


@dataclass(frozen=True)
class AffineRegistration:
    """Affine map from FIATLUX physical coordinates to SLM pixel coordinates.

    The forward convention is ``uv = xy @ matrix.T + translation``.  A full
    2x2 matrix represents rotation, unequal x/y scales, reflection, and shear.
    """

    matrix: torch.Tensor
    translation: torch.Tensor
    output_shape: tuple[int, int]

    def __post_init__(self) -> None:
        matrix = torch.as_tensor(self.matrix)
        translation = torch.as_tensor(self.translation)
        if matrix.shape != (2, 2) or translation.shape != (2,):
            raise ValueError("matrix and translation must have shapes (2, 2) and (2,).")
        if not matrix.is_floating_point() or not translation.is_floating_point():
            raise TypeError("matrix and translation must be floating-point tensors.")
        if not torch.isfinite(matrix).all() or not torch.isfinite(translation).all():
            raise ValueError("Registration parameters must be finite.")
        if abs(float(torch.linalg.det(matrix))) <= torch.finfo(matrix.dtype).eps:
            raise ValueError("The affine registration matrix must be invertible.")
        if (
            len(self.output_shape) != 2
            or not all(isinstance(value, int) and value > 0 for value in self.output_shape)
        ):
            raise ValueError("output_shape must contain two positive integers (ny, nx).")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def fit(
        cls,
        source_xy: torch.Tensor,
        measured_uv: torch.Tensor,
        output_shape: tuple[int, int],
    ) -> tuple[AffineRegistration, torch.Tensor]:
        """Fit an affine transform and return it with per-point residuals.

        At least three non-collinear point pairs are required.  Supplying more
        pairs performs a least-squares fit suitable for square-poke centroids.
        """
        source = torch.as_tensor(source_xy)
        measured = torch.as_tensor(measured_uv)
        if source.ndim != 2 or source.shape[1:] != (2,) or source.shape != measured.shape:
            raise ValueError("source_xy and measured_uv must have identical shape (N, 2).")
        if source.shape[0] < 3:
            raise ValueError("At least three point pairs are required.")
        if not source.is_floating_point() or not measured.is_floating_point():
            raise TypeError("Registration points must be floating-point tensors.")
        measured = measured.to(device=source.device, dtype=source.dtype)
        if not torch.isfinite(source).all() or not torch.isfinite(measured).all():
            raise ValueError("Registration points must be finite.")

        design = torch.cat(
            (source, torch.ones((source.shape[0], 1), device=source.device, dtype=source.dtype)),
            dim=1,
        )
        if int(torch.linalg.matrix_rank(design)) != 3:
            raise ValueError("Registration source points must not be collinear.")
        coefficients = torch.linalg.lstsq(design, measured).solution
        registration = cls(coefficients[:2].mT, coefficients[2], output_shape)
        predicted = registration.transform_points(source)
        return registration, measured - predicted

    def transform_points(self, xy: torch.Tensor) -> torch.Tensor:
        points = torch.as_tensor(xy)
        matrix = self.matrix.to(device=points.device, dtype=points.dtype)
        translation = self.translation.to(device=points.device, dtype=points.dtype)
        return points @ matrix.mT + translation

    def sampling_grid(self, source_grid: Grid) -> torch.Tensor:
        """Return the inverse affine sampling grid for ``grid_sample``."""
        dtype = source_grid.dtype
        device = source_grid.device
        ny, nx = self.output_shape
        v, u = torch.meshgrid(
            torch.arange(ny, device=device, dtype=dtype),
            torch.arange(nx, device=device, dtype=dtype),
            indexing="ij",
        )
        uv = torch.stack((u, v), dim=-1)
        matrix = self.matrix.to(device=device, dtype=dtype)
        translation = self.translation.to(device=device, dtype=dtype)
        xy = (uv - translation) @ torch.linalg.inv(matrix).mT

        # Grid coordinates are centred and spaced by dx/dy. align_corners=True
        # maps the first and last pixel centres exactly to -1 and +1.
        gx = 2 * (xy[..., 0] / source_grid.dx + source_grid.nx // 2) / max(
            source_grid.nx - 1, 1
        ) - 1
        gy = 2 * (xy[..., 1] / source_grid.dy + source_grid.ny // 2) / max(
            source_grid.ny - 1, 1
        ) - 1
        return torch.stack((gx, gy), dim=-1)

    def resample_opd(self, opd: torch.Tensor, source_grid: Grid) -> torch.Tensor:
        """Resample a continuous OPD map in metres onto the SLM pixel grid."""
        if opd.shape != source_grid.shape:
            raise ValueError(
                f"opd must have source-grid shape {source_grid.shape}; got {tuple(opd.shape)}."
            )
        if not opd.is_floating_point() or not torch.isfinite(opd).all():
            raise ValueError("opd must be a finite floating-point tensor in metres.")
        sample_grid = self.sampling_grid(source_grid).unsqueeze(0)
        sampled = functional.grid_sample(
            opd.unsqueeze(0).unsqueeze(0),
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled[0, 0]


@dataclass(frozen=True)
class SLMEncoder:
    """Convert an OPD map in metres into a calibrated integer SLM frame."""

    calibration: SLMCalibration
    registration: AffineRegistration

    def encode_opd(self, opd: torch.Tensor, source_grid: Grid) -> torch.Tensor:
        if opd.shape != source_grid.shape:
            raise ValueError(
                f"opd must have source-grid shape {source_grid.shape}; got {tuple(opd.shape)}."
            )
        if not opd.is_floating_point() or not torch.isfinite(opd).all():
            raise ValueError("opd must be a finite floating-point tensor in metres.")
        registered_opd = self.registration.resample_opd(opd, source_grid)
        phase = 2 * torch.pi * registered_opd / self.calibration.wavelength
        return self.calibration.phase_to_drive(phase)


def square_zonal_centers(basis: SquareZonalBasis) -> torch.Tensor:
    """Return physical (x, y) centres in command-matrix column order."""
    actuator_grid = basis.actuator_grid
    x, y = actuator_grid.positions(
        device=basis.pixel_grid.device,
        dtype=basis.pixel_grid.dtype,
    )
    center_x, center_y = torch.meshgrid(x, y, indexing="xy")
    return torch.stack((center_x.flatten(), center_y.flatten()), dim=-1)


def illumination_fractions(
    basis: SquareZonalBasis,
    pupil_transmission: torch.Tensor,
) -> torch.Tensor:
    """Return the intensity-weighted illuminated fraction of every square mode."""
    transmission = torch.as_tensor(
        pupil_transmission,
        device=basis.pixel_grid.device,
        dtype=basis.pixel_grid.dtype,
    )
    if transmission.shape != basis.pixel_grid.shape:
        raise ValueError(
            "pupil_transmission must have shape "
            f"{basis.pixel_grid.shape}; got {tuple(transmission.shape)}."
        )
    if not torch.isfinite(transmission).all():
        raise ValueError("pupil_transmission must contain only finite values.")
    support = basis.build_command_matrix().abs()
    area = support.sum(dim=0)
    if bool(torch.any(area == 0)):
        raise ValueError("Every square mode must contain at least one sampled pixel.")
    intensity = transmission.abs().square().flatten()
    return (intensity[:, None] * support).sum(dim=0) / area


def select_registration_pokes(
    basis: SquareZonalBasis,
    pupil_transmission: torch.Tensor,
    count: int,
    *,
    minimum_illumination: float = 0.95,
) -> torch.Tensor:
    """Select well-separated, illuminated square modes from a dense basis."""
    if not isinstance(count, int) or count < 3:
        raise ValueError("count must be an integer of at least three.")
    if not math.isfinite(minimum_illumination) or not 0 <= minimum_illumination <= 1:
        raise ValueError("minimum_illumination must lie in [0, 1].")

    fractions = illumination_fractions(basis, pupil_transmission)
    candidates = torch.where(fractions >= minimum_illumination)[0]
    if candidates.numel() < count:
        raise ValueError(
            f"Only {candidates.numel()} modes meet the illumination threshold; "
            f"cannot select {count}."
        )
    positions = square_zonal_centers(basis)[candidates]

    # Seed with a wide, non-collinear triangle. This avoids the common failure
    # where centre and opposite edge points selected by plain farthest-point
    # sampling happen to be collinear.
    centroid = positions.mean(dim=0)
    first = int(torch.linalg.vector_norm(positions - centroid, dim=1).argmax())
    second = int(torch.linalg.vector_norm(positions - positions[first], dim=1).argmax())
    baseline = positions[second] - positions[first]
    relative = positions - positions[first]
    line_distance = (
        relative[:, 0] * baseline[1] - relative[:, 1] * baseline[0]
    ).abs() / torch.linalg.vector_norm(baseline)
    third = int(line_distance.argmax())
    if not bool(line_distance[third] > 0):
        raise ValueError("Illuminated square-mode centres are collinear.")
    selected_local = [first, second, third]
    for _ in range(3, count):
        distances = torch.cdist(positions, positions[selected_local])
        nearest = distances.min(dim=1).values
        nearest[selected_local] = -1
        selected_local.append(int(nearest.argmax()))
    return candidates[torch.tensor(selected_local, device=candidates.device)]


def weighted_centroid(
    response: torch.Tensor,
    *,
    threshold_fraction: float = 0.2,
    absolute: bool = True,
) -> torch.Tensor:
    """Measure an (x, y) centroid from a flat- or push-pull response image."""
    if response.ndim != 2 or not response.is_floating_point():
        raise ValueError("response must be a two-dimensional floating-point tensor.")
    if not torch.isfinite(response).all():
        raise ValueError("response must contain only finite values.")
    if not math.isfinite(threshold_fraction) or not 0 <= threshold_fraction < 1:
        raise ValueError("threshold_fraction must lie in [0, 1).")

    weights = response.abs() if absolute else response.clamp_min(0)
    peak = weights.max()
    if not bool(peak > 0):
        raise ValueError("The response contains no measurable signal.")
    weights = torch.where(
        weights >= threshold_fraction * peak,
        weights,
        torch.zeros_like(weights),
    )
    total = weights.sum()
    y, x = torch.meshgrid(
        torch.arange(response.shape[0], device=response.device, dtype=response.dtype),
        torch.arange(response.shape[1], device=response.device, dtype=response.dtype),
        indexing="ij",
    )
    return torch.stack(((weights * x).sum() / total, (weights * y).sum() / total))
