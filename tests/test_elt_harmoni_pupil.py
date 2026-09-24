import math

import pytest
import torch

from fiatlux import ELTHarmoniPupil, Grid
from fiatlux.core.spectrum import Band, Spectrum


def _build(pupil: ELTHarmoniPupil) -> None:
    pupil.build(Spectrum(0, Band(1e-6, 0.0, 1.0), 1, dtype=pupil.grid.dtype))


def test_default_elt_pupil_has_798_segments_and_six_petals():
    grid = Grid(401, 401, 0.1, 0.1, dtype=torch.float64)
    pupil = ELTHarmoniPupil(grid, spider_width=0.0, petal_gap=0.02)
    _build(pupil)

    assert pupil.number_of_segments == 798
    active_petals = torch.unique(pupil.petal_index[pupil.petal_index >= 0])
    torch.testing.assert_close(active_petals, torch.arange(6))


def test_outer_circular_mask_is_optional_but_central_obscuration_remains():
    grid = Grid(401, 401, 0.1, 0.1, dtype=torch.float64)
    segments_only = ELTHarmoniPupil(grid, spider_width=0.0)
    circularly_masked = ELTHarmoniPupil(grid, spider_width=0.0, circular_mask=True)
    _build(segments_only)
    _build(circularly_masked)

    x, y = grid.meshgrid()
    squared_radius = x.square() + y.square()
    inside_obscuration = squared_radius < segments_only.central_obscuration_radius**2
    radius = torch.sqrt(squared_radius)
    outside_outer_disk = radius > segments_only.outer_radius + 0.5 * grid.dx

    assert segments_only.circular_mask is False
    assert torch.all(segments_only.transmission[inside_obscuration] == 0)
    assert torch.all(circularly_masked.transmission[inside_obscuration] == 0)
    assert torch.any(segments_only.transmission[outside_outer_disk] > 0)
    assert torch.all(circularly_masked.transmission[outside_outer_disk] == 0)
    assert torch.all(circularly_masked.transmission <= segments_only.transmission)


def test_obscuration_spiders_and_petal_gaps_block_expected_points():
    grid = Grid(501, 501, 0.1, 0.1, dtype=torch.float64)
    pupil = ELTHarmoniPupil(
        grid,
        central_obscuration_radius=5.5,
        circular_mask=True,
        spider_width=0.4,
        spider_angles=(0.0,),
        petal_gap=0.2,
        petal_rotation=30.0,
    )
    _build(pupil)
    center = grid.nx // 2

    assert pupil.transmission[center, center] == 0
    assert torch.all(pupil.transmission[center, :] == 0)
    assert torch.all(pupil.segment_index[pupil.transmission == 0] == -1)
    assert torch.all(pupil.petal_index[pupil.transmission == 0] == -1)


def test_circular_mask_requires_a_boolean():
    grid = Grid(11, 11, 1.0, 1.0)

    with pytest.raises(TypeError, match="circular_mask must be a boolean"):
        ELTHarmoniPupil(grid, circular_mask=1)


def test_center_moves_the_obscuration_and_rotation_moves_segment_centers():
    grid = Grid(301, 301, 0.2, 0.2, dtype=torch.float64)
    reference = ELTHarmoniPupil(grid, center=(0.0, 0.0), rotation=0.0)
    transformed = ELTHarmoniPupil(grid, center=(1.0, -0.6), rotation=30.0)

    reference_center = reference.segment_centers[0]
    transformed_center = transformed.segment_centers[0]
    expected_x = math.cos(math.pi / 6) * reference_center[0] - math.sin(math.pi / 6) * reference_center[1] + 1.0
    expected_y = math.sin(math.pi / 6) * reference_center[0] + math.cos(math.pi / 6) * reference_center[1] - 0.6
    torch.testing.assert_close(transformed_center, torch.stack((expected_x, expected_y)))


def test_segment_and_petal_masks_select_only_transmitted_pixels():
    grid = Grid(301, 301, 0.15, 0.15, dtype=torch.float64)
    pupil = ELTHarmoniPupil(grid, spider_width=0.2, petal_gap=0.1)
    _build(pupil)

    coordinate = pupil.segment_coordinates[100]
    assert torch.all(pupil.segment_mask(coordinate) <= (pupil.transmission > 0))
    for index in range(6):
        assert torch.all(pupil.petal_mask(index) <= (pupil.transmission > 0))
