import pytest
import torch

from fiatlux.core.grid import Grid


def test_grid_coordinates_are_centered_and_use_declared_state():
    grid = Grid(5, 3, 0.2, 0.4, dtype=torch.float64)

    torch.testing.assert_close(
        grid.x, torch.tensor([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=torch.float64)
    )
    torch.testing.assert_close(
        grid.y, torch.tensor([-0.4, 0.0, 0.4], dtype=torch.float64)
    )
    assert grid.meshgrid()[0].shape == grid.shape == (3, 5)
    assert grid.x.device == grid.device


def test_grid_to_returns_new_grid_without_mutating_source():
    grid = Grid(4, 2, 0.1, 0.2)

    converted = grid.to(dtype=torch.float64)

    assert converted is not grid
    assert converted.dtype == torch.float64
    assert grid.dtype == torch.float32
    assert (converted.nx, converted.ny, converted.dx, converted.dy) == (4, 2, 0.1, 0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nx": 0},
        {"ny": -1},
        {"nx": 2.5},
        {"dx": 0.0},
        {"dy": float("nan")},
        {"dtype": torch.float16},
    ],
)
def test_grid_rejects_invalid_geometry_and_dtype(kwargs):
    defaults = {"nx": 4, "ny": 3, "dx": 0.1, "dy": 0.2}
    defaults.update(kwargs)

    with pytest.raises(ValueError):
        Grid(**defaults)
