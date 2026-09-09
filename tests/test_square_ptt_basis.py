import pytest
import torch

from fiatlux.core.grid import Grid
from fiatlux.optics.elements.deformable_mirror import (
    ActuatorGrid,
    DeformableMirror,
    SquarePTTZonalBasis,
)


def split_single_actuator_modes(basis: SquarePTTZonalBasis):
    matrix = basis.build_command_matrix()
    return tuple(matrix[:, index].reshape(basis.pixel_grid.shape) for index in range(3))


@pytest.mark.parametrize(
    "grid",
    [
        Grid(nx=9, ny=9, dx=0.25, dy=0.25),
        Grid(nx=9, ny=7, dx=0.25, dy=0.5),
    ],
)
def test_single_actuator_ptt_modes_have_expected_support_and_values(grid):
    width = 0.8
    basis = SquarePTTZonalBasis(
        actuator_grid=ActuatorGrid(1, 1, pitch=1.0),
        pixel_grid=grid,
        influence_width=width,
    )
    piston, tip, tilt = split_single_actuator_modes(basis)
    x, y = grid.meshgrid()
    support = (x.abs() < width) & (y.abs() < width)

    torch.testing.assert_close(piston, support.to(piston.dtype))
    torch.testing.assert_close(tip[support], x[support] / width)
    torch.testing.assert_close(tilt[support], y[support] / width)
    assert torch.count_nonzero(tip[~support]) == 0
    assert torch.count_nonzero(tilt[~support]) == 0


def test_tip_varies_only_along_x_and_tilt_only_along_y():
    grid = Grid(nx=7, ny=5, dx=0.1, dy=0.1)
    basis = SquarePTTZonalBasis(ActuatorGrid(1, 1, 1.0), grid, 1.0)
    _, tip, tilt = split_single_actuator_modes(basis)

    assert torch.all(torch.diff(tip, dim=-1) > 0)
    assert torch.all(torch.diff(tip, dim=-2) == 0)
    assert torch.all(torch.diff(tilt, dim=-2) > 0)
    assert torch.all(torch.diff(tilt, dim=-1) == 0)


@pytest.mark.parametrize("n_x,n_y", [(1, 1), (2, 4), (3, 5)])
def test_actuator_positions_are_symmetric_about_zero(n_x, n_y):
    actuator_grid = ActuatorGrid(n_x, n_y, pitch=0.5)

    x, y = actuator_grid.positions(device=torch.device("cpu"), dtype=torch.float32)

    torch.testing.assert_close(x, -x.flip(0))
    torch.testing.assert_close(y, -y.flip(0))


def test_ptt_command_matrix_and_dm_opd_use_rectangular_y_x_shape():
    grid = Grid(nx=9, ny=7, dx=0.25, dy=0.5)
    actuator_grid = ActuatorGrid(1, 1, pitch=1.0)
    basis = SquarePTTZonalBasis(actuator_grid, grid, influence_width=0.8)
    dm = DeformableMirror(grid, actuator_grid, grid, basis)

    assert dm._command_matrix.shape == (grid.ny * grid.nx, 3)
    assert dm.opd.shape == (grid.ny, grid.nx)


@pytest.mark.parametrize("width", [0.0, -1.0, float("inf"), float("nan")])
def test_ptt_basis_rejects_invalid_influence_width(width):
    with pytest.raises(ValueError, match="positive finite length"):
        SquarePTTZonalBasis(
            ActuatorGrid(1, 1, 1.0),
            Grid(nx=3, ny=3, dx=0.1, dy=0.1),
            width,
        )
