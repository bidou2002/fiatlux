import math

import pytest
import torch

from fiatlux import Grid, HexagonalSegmentedAperture
from fiatlux.core.spectrum import Band, Spectrum


def _build(aperture: HexagonalSegmentedAperture) -> None:
    aperture.build(Spectrum(0, Band(1e-6, 0.0, 1.0), 1, dtype=aperture.grid.dtype))


def test_segment_count_and_axial_index_are_deterministic():
    aperture = HexagonalSegmentedAperture(
        Grid(401, 401, 0.01, 0.01, dtype=torch.float64),
        segment_circumradius=0.2,
        rings=2,
    )

    assert aperture.number_of_segments == 1 + 3 * 2 * (2 + 1)
    assert aperture.segment_coordinates[0] == (0, 0)
    assert aperture.index_of((0, 0)) == 0
    assert aperture.segment_coordinates == tuple(aperture.segment_coordinates)


def test_gap_and_inactive_segments_are_rendered_in_physical_coordinates():
    grid = Grid(601, 601, 0.005, 0.005, dtype=torch.float64)
    complete = HexagonalSegmentedAperture(
        grid, 0.2, 1, gap=0.04
    )
    incomplete = HexagonalSegmentedAperture(
        grid, 0.2, 1, gap=0.04, inactive_segments=[(1, 0)]
    )
    _build(complete)
    _build(incomplete)

    assert complete.number_of_segments == 7
    assert incomplete.number_of_segments == 6
    assert incomplete.transmission.sum() < complete.transmission.sum()
    with pytest.raises(KeyError, match="No active segment"):
        incomplete.index_of((1, 0))


def test_sampled_area_matches_analytical_segment_area():
    grid = Grid(801, 801, 0.004, 0.004, dtype=torch.float64)
    radius = 0.2
    aperture = HexagonalSegmentedAperture(grid, radius, 1, gap=0.02)
    _build(aperture)

    sampled_area = float(aperture.transmission.sum() * grid.dx * grid.dy)
    expected_area = aperture.number_of_segments * 3 * math.sqrt(3) * radius**2 / 2
    assert sampled_area == pytest.approx(expected_area, rel=0.015)


def test_transmission_is_symmetric_and_uses_grid_dtype_and_device():
    grid = Grid(401, 401, 0.01, 0.01, dtype=torch.float64)
    aperture = HexagonalSegmentedAperture(grid, 0.2, 2, gap=0.02)
    _build(aperture)

    assert aperture.transmission.dtype == grid.dtype
    assert aperture.transmission.device == grid.device
    torch.testing.assert_close(aperture.transmission, torch.flip(aperture.transmission, (0,)))
    torch.testing.assert_close(aperture.transmission, torch.flip(aperture.transmission, (1,)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_segmented_aperture_is_device_aware_on_cuda():
    grid = Grid(101, 101, 0.02, 0.02, device="cuda", dtype=torch.float32)
    aperture = HexagonalSegmentedAperture(grid, 0.2, 1, gap=0.01)
    _build(aperture)

    assert aperture.segment_centers.device.type == "cuda"
    assert aperture.transmission.device.type == "cuda"

