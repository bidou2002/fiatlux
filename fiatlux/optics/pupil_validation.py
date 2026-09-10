from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PupilComparison:
    """Dimensionless overlap metrics and physical areas for two pupil masks."""

    intersection_over_union: float
    dice: float
    analytical_area: float
    reference_area: float
    relative_area_error: float
    false_positive_fraction: float
    false_negative_fraction: float


def compare_pupil_masks(
    analytical: torch.Tensor,
    reference: torch.Tensor,
    *,
    pixel_area: float,
    threshold: float = 0.5,
) -> PupilComparison:
    """Compare sampled pupil supports, including an optionally loaded FITS mask.

    Both tensors must describe the same registered sampling. Floating masks are
    converted to support masks using ``value > threshold``. ``pixel_area`` is
    in square metres, so returned areas are in square metres.
    """
    if analytical.ndim != 2 or reference.ndim != 2:
        raise ValueError("Pupil masks must both be two-dimensional.")
    if analytical.shape != reference.shape:
        raise ValueError(
            f"Pupil masks must have identical shapes; got {tuple(analytical.shape)} "
            f"and {tuple(reference.shape)}."
        )
    if not math.isfinite(pixel_area) or pixel_area <= 0:
        raise ValueError("pixel_area must be positive and finite.")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    analytical_support = analytical > threshold
    reference_support = reference.to(analytical.device) > threshold
    intersection = torch.count_nonzero(analytical_support & reference_support).item()
    union = torch.count_nonzero(analytical_support | reference_support).item()
    analytical_count = torch.count_nonzero(analytical_support).item()
    reference_count = torch.count_nonzero(reference_support).item()
    if union == 0 or analytical_count + reference_count == 0 or reference_count == 0:
        raise ValueError("The comparison requires a non-empty reference and union.")

    analytical_area = analytical_count * pixel_area
    reference_area = reference_count * pixel_area
    false_positive = torch.count_nonzero(
        analytical_support & ~reference_support
    ).item()
    false_negative = torch.count_nonzero(
        ~analytical_support & reference_support
    ).item()
    return PupilComparison(
        intersection_over_union=intersection / union,
        dice=2 * intersection / (analytical_count + reference_count),
        analytical_area=analytical_area,
        reference_area=reference_area,
        relative_area_error=(analytical_area - reference_area) / reference_area,
        false_positive_fraction=false_positive / reference_count,
        false_negative_fraction=false_negative / reference_count,
    )
