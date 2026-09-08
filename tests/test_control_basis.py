import pytest
import torch

from fiatlux.core.grid import Grid
from fiatlux.optics.elements.deformable_mirror import (
    ActuatorGrid,
    ControlBasis,
    FourierBasis,
    GaussianZonalBasis,
    SquarePTTZonalBasis,
    SquareZonalBasis,
    ZernikeBasis,
)


class DummyPupil:
    def __init__(self, grid: Grid):
        self.transmission = torch.ones((grid.ny, grid.nx))


def make_bases():
    pixel_grid = Grid(nx=5, ny=4, dx=0.1, dy=0.2)
    actuator_grid = ActuatorGrid(n_actuators_x=2, n_actuators_y=3, pitch=0.1)
    return [
        (GaussianZonalBasis(actuator_grid, pixel_grid, 1.0), 6),
        (SquareZonalBasis(actuator_grid, pixel_grid, 1.0), 6),
        (SquarePTTZonalBasis(actuator_grid, pixel_grid, 1.0), 18),
        (FourierBasis(pixel_grid, torch.arange(3), DummyPupil(pixel_grid)), 9),
        (ZernikeBasis(pixel_grid, n=7), 7),
    ]


@pytest.mark.parametrize("basis, expected", make_bases())
def test_every_control_basis_exposes_n_modes_as_property(basis, expected):
    assert basis.n_modes == expected
    assert isinstance(basis.n_modes, int)


def test_n_modes_is_abstract_property_on_base_class():
    assert isinstance(ControlBasis.__dict__["n_modes"], property)
    assert ControlBasis.__dict__["n_modes"].fget.__isabstractmethod__
