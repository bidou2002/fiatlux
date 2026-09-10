import pytest
import torch

from fiatlux.core.field import Field
from fiatlux.core.grid import Grid
from fiatlux.core.spectrum import Band, Spectrum
from fiatlux.optics.propagator import (
    NearFieldPropagator,
    PropagationSamplingError,
)


class PassThroughNearField(NearFieldPropagator):
    def apply(self, field):
        self._validate_input(field)
        return field


def make_field(grid):
    spectrum = Spectrum(
        magnitude=0,
        band=Band(central_wavelength=1e-6, delta_wavelength=0.0, f0=1.0),
        samples=1,
    ).to(device=grid.device, dtype=grid.dtype)
    dtype = torch.complex64 if grid.dtype == torch.float32 else torch.complex128
    return Field(
        torch.ones((1, grid.ny, grid.nx), device=grid.device, dtype=dtype),
        grid,
        spectrum,
    )


@pytest.mark.parametrize("distance", [1.0, -1.0, 0.0])
def test_signed_finite_distances_are_part_of_the_base_contract(distance):
    grid = Grid(nx=8, ny=6, dx=0.1, dy=0.2)
    propagator = PassThroughNearField(distance, grid)

    assert propagator.distance == distance
    assert propagator.output_grid is grid


@pytest.mark.parametrize("distance", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_distances_are_rejected(distance):
    with pytest.raises(ValueError, match="finite"):
        PassThroughNearField(distance, Grid(8, 6, 0.1, 0.2))


@pytest.mark.parametrize("distance", [True, "1 m", torch.tensor(1.0)])
def test_distance_is_initially_a_python_real_number(distance):
    with pytest.raises(TypeError, match="real number"):
        PassThroughNearField(distance, Grid(8, 6, 0.1, 0.2))


def test_near_field_input_and_output_use_the_same_grid():
    expected = Grid(nx=8, ny=6, dx=0.1, dy=0.2)
    actual = Grid(nx=8, ny=6, dx=0.2, dy=0.2)
    propagator = PassThroughNearField(1.0, expected)

    with pytest.raises(ValueError, match="PassThroughNearField.*expected.*got"):
        propagator.apply(make_field(actual))


def test_sampling_error_is_a_value_error():
    assert issubclass(PropagationSamplingError, ValueError)
