import pytest
import torch

from fiatlux.core.grid import Grid
from fiatlux.core.spectrum import Band, Spectrum
from fiatlux.optics.elements.deformable_mirror import (
    ActuatorGrid,
    ControlBasis,
    DeformableMirror,
    FourierBasis,
    GaussianZonalBasis,
    PupilOrthogonalizedBasis,
    SquarePTTZonalBasis,
    SquareZonalBasis,
    ZernikeBasis,
)
from fiatlux.optics.elements.mask import ArbitraryAperture


class DummyPupil:
    def __init__(self, grid: Grid):
        self.transmission = torch.ones((grid.ny, grid.nx))


class NonOrthogonalBasis(ControlBasis):
    def __init__(self, pixel_grid: Grid):
        self.pixel_grid = pixel_grid

    @property
    def n_modes(self) -> int:
        return 2

    def build_command_matrix(self) -> torch.Tensor:
        return torch.tensor(
            [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, -1.0]],
            device=self.pixel_grid.device,
            dtype=self.pixel_grid.dtype,
        )


def build_pupil(grid: Grid, transmission: torch.Tensor) -> ArbitraryAperture:
    pupil = ArbitraryAperture(grid, transmission)
    pupil.build(Spectrum(0, Band(1e-6, 0.0, 1.0), 1, dtype=grid.dtype))
    return pupil


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


def test_pupil_orthogonalized_basis_has_unit_weighted_gram_matrix():
    grid = Grid(2, 2, 0.1, 0.1, dtype=torch.float64)
    transmission = torch.tensor([[1.0, 0.5], [0.0, 1.0]], dtype=grid.dtype)
    pupil = build_pupil(grid, transmission)
    basis = PupilOrthogonalizedBasis(NonOrthogonalBasis(grid), pupil)

    matrix = basis.build_command_matrix()
    weights = transmission.square().flatten()
    weights /= weights.sum()
    gram = matrix.mT @ (weights[:, None] * matrix)

    torch.testing.assert_close(gram, torch.eye(2, dtype=grid.dtype))
    assert matrix.dtype == grid.dtype
    assert matrix.device == grid.device


def test_pupil_orthogonalization_preserves_the_controlled_subspace():
    grid = Grid(2, 2, 0.1, 0.1, dtype=torch.float64)
    pupil = build_pupil(grid, torch.ones(grid.shape, dtype=grid.dtype))
    original = NonOrthogonalBasis(grid).build_command_matrix()
    orthogonal = PupilOrthogonalizedBasis(
        NonOrthogonalBasis(grid), pupil
    ).build_command_matrix()

    original_projector = original @ torch.linalg.pinv(original)
    orthogonal_projector = orthogonal @ torch.linalg.pinv(orthogonal)
    torch.testing.assert_close(orthogonal_projector, original_projector)


def test_pupil_orthogonalized_basis_rejects_rank_loss_on_pupil():
    grid = Grid(2, 2, 0.1, 0.1, dtype=torch.float64)
    pupil = build_pupil(
        grid,
        torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=grid.dtype),
    )
    basis = PupilOrthogonalizedBasis(NonOrthogonalBasis(grid), pupil)

    with pytest.raises(ValueError, match=r"rank 1.*2 modes"):
        basis.build_command_matrix()


def test_pupil_must_be_built_before_orthogonalization():
    grid = Grid(2, 2, 0.1, 0.1)
    pupil = ArbitraryAperture(grid, torch.ones(grid.shape))
    basis = PupilOrthogonalizedBasis(NonOrthogonalBasis(grid), pupil)

    with pytest.raises(ValueError, match="pupil must be built"):
        basis.build_command_matrix()


def test_orthogonalized_basis_preserves_dm_command_autograd():
    grid = Grid(2, 2, 0.1, 0.1, dtype=torch.float64)
    pupil = build_pupil(grid, torch.ones(grid.shape, dtype=grid.dtype))
    basis = PupilOrthogonalizedBasis(NonOrthogonalBasis(grid), pupil)
    dm = DeformableMirror(grid, ActuatorGrid(2, 1, 0.1), grid, basis)

    dm.opd.square().sum().backward()

    assert dm.commands.grad is not None
    assert torch.isfinite(dm.commands.grad).all()
