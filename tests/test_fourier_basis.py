import pytest
import torch

from fiatlux.core.grid import Grid
from fiatlux.optics.elements.deformable_mirror import FourierBasis


class Pupil:
    def __init__(self, grid):
        self.transmission = torch.ones(grid.shape, device=grid.device, dtype=grid.dtype)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_finite_independent_modes_and_physical_frequency(dtype, device):
    grid = Grid(21, 17, .05, .07, dtype=dtype, device=device)
    basis = FourierBasis(grid, torch.tensor([-1., 0., 1.]), Pupil(grid))
    matrix = basis.build_command_matrix()
    assert torch.isfinite(matrix).all()
    assert torch.linalg.matrix_rank(matrix) == 9
    x, y = grid.meshgrid()
    phases = [2*torch.pi*(fx*x+fy*y) for fx in [-1, 0, 1] for fy in [-1, 0, 1]]
    expected = torch.stack([
        torch.cos(phase) if fx > 0 or (fx == 0 and fy >= 0) else torch.sin(phase)
        for phase, (fx, fy) in zip(phases, [(fx, fy) for fx in [-1, 0, 1] for fy in [-1, 0, 1]])
    ]).flatten(1).T
    expected /= torch.linalg.vector_norm(expected, dim=0)
    torch.testing.assert_close(matrix, expected)


def test_cycles_per_aperture_is_scale_invariant():
    frequencies = torch.tensor([-1., 0., 1.])
    grid = Grid(21, 17, .05, .07, dtype=torch.float64)
    larger = Grid(21, 17, .1, .14, dtype=torch.float64)
    a = FourierBasis(grid, frequencies, Pupil(grid), "cycles/aperture", 1.).build_command_matrix()
    b = FourierBasis(larger, frequencies, Pupil(larger), "cycles/aperture", 2.).build_command_matrix()
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("frequencies", [[0, 1, 2], [1, 0, -1], [-1, 1],
                                         [-1, 0, 0, 0, 1], [float("nan")], []])
def test_unsupported_frequency_sets_raise(frequencies):
    grid = Grid(9, 9, .1, .1)
    with pytest.raises(ValueError, match="symmetric"):
        FourierBasis(grid, torch.tensor(frequencies), Pupil(grid))


def test_zero_pupil_response_raises():
    grid = Grid(9, 9, .1, .1)
    pupil = Pupil(grid)
    pupil.transmission.zero_()
    with pytest.raises(ValueError, match="pupil response"):
        FourierBasis(grid, torch.tensor([-1., 0., 1.]), pupil).build_command_matrix()


def test_cycles_per_aperture_requires_diameter():
    grid = Grid(9, 9, .1, .1)
    with pytest.raises(ValueError, match="requires aperture_diameter"):
        FourierBasis(grid, torch.tensor([-1., 0., 1.]), Pupil(grid), "cycles/aperture")
