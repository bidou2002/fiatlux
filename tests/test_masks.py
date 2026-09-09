import pytest
import torch

from fiatlux.core.grid import Grid
from fiatlux.core.source import PlaneWave
from fiatlux.core.spectrum import Band, Spectrum
from fiatlux.optics.elements.mask import (
    ArbitraryAperture,
    CircularAperture,
    Piston,
    Step,
    TipTilt,
)


@pytest.fixture
def optical_state():
    grid = Grid(7, 5, 0.1, 0.2, dtype=torch.float64)
    spectrum = Spectrum(0, Band(1e-6, 0.0, 1e8), 1, dtype=torch.float64)
    field = PlaneWave(spectrum).generate_field(grid)
    return grid, spectrum, field


@pytest.mark.parametrize(
    "factory",
    [
        lambda grid: CircularAperture(grid, radius=0.25),
        lambda grid: Piston(grid, piston=100e-9),
        lambda grid: Step(grid, piston=100e-9),
        lambda grid: TipTilt(grid, tip=10e-9, tilt=-20e-9),
    ],
)
def test_standard_masks_preserve_field_contract(optical_state, factory):
    grid, _, field = optical_state
    mask = factory(grid)

    output = mask.apply(field)

    assert output.grid is field.grid
    assert output.spectrum is field.spectrum
    assert output.complex_amplitude.shape == field.complex_amplitude.shape
    assert output.complex_amplitude.dtype == torch.complex128
    assert output.complex_amplitude.device == grid.device
    assert mask.opd.shape == grid.shape
    assert mask.opd.dtype == grid.dtype


def test_circular_aperture_blocks_pixels_outside_radius(optical_state):
    grid, spectrum, _ = optical_state
    aperture = CircularAperture(grid, radius=0.15)

    aperture.build(spectrum)

    x, y = grid.meshgrid()
    assert torch.equal(aperture.transmission, x.square() + y.square() <= 0.15**2)


def test_arbitrary_aperture_rejects_wrong_shape(optical_state):
    grid, spectrum, _ = optical_state
    aperture = ArbitraryAperture(grid, torch.ones(7, 5))

    with pytest.raises(ValueError, match="does not match grid shape"):
        aperture.build(spectrum)


def test_mask_rejects_field_on_different_grid(optical_state):
    grid, _, _ = optical_state
    other_grid = Grid(7, 5, 0.2, 0.2, dtype=torch.float64)
    field = PlaneWave(Spectrum(0, Band(1e-6, 0.0, 1e8), 1)).generate_field(
        Grid(7, 5, 0.1, 0.2)
    )

    with pytest.raises(ValueError, match="grid must match"):
        CircularAperture(other_grid, radius=0.2).apply(field)
