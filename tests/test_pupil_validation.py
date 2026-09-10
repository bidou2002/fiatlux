import math

import pytest
import torch

from fiatlux import ELTHarmoniPupil, Grid, compare_pupil_masks
from fiatlux.core.spectrum import Band, Spectrum


def _build(pupil: ELTHarmoniPupil) -> None:
    pupil.build(Spectrum(0, Band(1e-6, 0.0, 1.0), 1, dtype=pupil.grid.dtype))


def test_default_pupil_is_symmetric_and_has_plausible_collecting_area():
    grid = Grid(401, 401, 0.12, 0.12, dtype=torch.float64)
    pupil = ELTHarmoniPupil(grid, spider_width=0.0, petal_gap=0.0)
    _build(pupil)

    sampled_area = float(pupil.transmission.sum() * grid.dx * grid.dy)
    annular_area = math.pi * (
        pupil.outer_radius**2 - pupil.central_obscuration_radius**2
    )
    assert pupil.number_of_segments == 798
    assert sampled_area / annular_area == pytest.approx(1.0, rel=0.08)
    torch.testing.assert_close(pupil.transmission, torch.flip(pupil.transmission, (0,)))
    torch.testing.assert_close(pupil.transmission, torch.flip(pupil.transmission, (1,)))


def test_collecting_area_converges_with_sampling():
    spacings = (0.48, 0.24, 0.12)
    sizes = (101, 201, 401)
    areas = []
    for size, spacing in zip(sizes, spacings):
        pupil = ELTHarmoniPupil(
            Grid(size, size, spacing, spacing, dtype=torch.float64),
            spider_width=0.0,
            petal_gap=0.0,
        )
        _build(pupil)
        sampled_area = float(pupil.transmission.sum() * spacing**2)
        areas.append(sampled_area)

    coarse_change = abs(areas[1] - areas[0])
    fine_change = abs(areas[2] - areas[1])
    assert fine_change < coarse_change
    assert fine_change / areas[2] < 0.01


def test_reference_comparison_reports_known_overlap_metrics():
    reference = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float64)
    analytical = torch.tensor([[1, 0, 1], [1, 0, 0]], dtype=torch.float32)

    comparison = compare_pupil_masks(
        analytical, reference, pixel_area=0.25
    )

    assert comparison.intersection_over_union == pytest.approx(0.5)
    assert comparison.dice == pytest.approx(2 / 3)
    assert comparison.analytical_area == pytest.approx(0.75)
    assert comparison.reference_area == pytest.approx(0.75)
    assert comparison.relative_area_error == pytest.approx(0.0)
    assert comparison.false_positive_fraction == pytest.approx(1 / 3)
    assert comparison.false_negative_fraction == pytest.approx(1 / 3)


def test_reference_comparison_validates_registration_and_empty_masks():
    with pytest.raises(ValueError, match="identical shapes"):
        compare_pupil_masks(torch.ones(2, 2), torch.ones(3, 3), pixel_area=1.0)
    with pytest.raises(ValueError, match="non-empty"):
        compare_pupil_masks(torch.zeros(2, 2), torch.zeros(2, 2), pixel_area=1.0)


def test_m4_petal_boundaries_align_with_spiders():
    grid = Grid(301, 301, 0.16, 0.16, dtype=torch.float64)
    common = dict(
        grid=grid,
        central_obscuration_radius=5.5,
        spider_angles=(0.0, 60.0, 120.0),
    )
    spider_only = ELTHarmoniPupil(
        **common, spider_width=0.24, petal_gap=0.0
    )
    petal_only = ELTHarmoniPupil(
        **common, spider_width=0.0, petal_gap=0.24, petal_rotation=0.0
    )
    _build(spider_only)
    _build(petal_only)

    torch.testing.assert_close(spider_only.transmission, petal_only.transmission)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_elt_pupil_and_reference_comparison_run_on_cuda():
    grid = Grid(201, 201, 0.24, 0.24, device="cuda", dtype=torch.float32)
    pupil = ELTHarmoniPupil(grid)
    _build(pupil)
    comparison = compare_pupil_masks(
        pupil.transmission, pupil.transmission.clone(), pixel_area=grid.dx * grid.dy
    )

    assert comparison.intersection_over_union == 1.0
    assert pupil.transmission.dtype == torch.float32
    assert pupil.transmission.device.type == "cuda"
