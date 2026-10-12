import pytest
import torch

from fiatlux.core.grid import Grid
from fiatlux.optics.elements.deformable_mirror import (
    ActuatorGrid, DeformableMirror, ZernikeBasis,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(8, 6), (9, 7), (9, 9)])
def test_grid_orientation_support_and_dm(shape, dtype, device):
    nx, ny = shape
    grid = Grid(nx, ny, .1, .17, device=device, dtype=dtype)
    radius = .35
    basis = ZernikeBasis(grid, 3, pupil_radius=radius)
    modes = basis.build_command_matrix().T.reshape(3, ny, nx)
    x, y = grid.meshgrid()
    support = torch.hypot(x, y) <= radius
    torch.testing.assert_close(modes[0], torch.where(support, 2*x/radius, 0.))
    torch.testing.assert_close(modes[1], torch.where(support, 2*y/radius, 0.))
    assert modes[0, ny//2, nx//2] == 0
    assert modes[1, ny//2, nx//2] == 0
    dm = DeformableMirror(grid, ActuatorGrid(3, 1, .1), grid, basis)
    assert dm.opd.shape == grid.shape
    assert modes.dtype == dtype and modes.device == grid.device


def test_default_radius_and_physical_scaling():
    grid = Grid(8, 6, .1, .2, dtype=torch.float64)
    default = ZernikeBasis(grid, 3).build_command_matrix()
    explicit = ZernikeBasis(grid, 3, .4).build_command_matrix()
    torch.testing.assert_close(default, explicit)
    enlarged = Grid(8, 6, .2, .4, dtype=torch.float64)
    torch.testing.assert_close(default, ZernikeBasis(enlarged, 3, .8).build_command_matrix())


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf")])
def test_invalid_radius(radius):
    with pytest.raises(ValueError, match="pupil_radius"):
        ZernikeBasis(Grid(8, 6, .1, .2), 3, radius)
