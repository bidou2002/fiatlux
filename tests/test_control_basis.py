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
from fiatlux.optics.elements.segmented_aperture import ELTHarmoniPupil


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
        (FourierBasis(pixel_grid, torch.tensor([-1., 0., 1.]), DummyPupil(pixel_grid)), 9),
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



@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("kind", ["imaginary", "apodized", "bool"])
def test_orthogonalization_uses_phase_invariant_intensity(dtype, device, kind):
    grid = Grid(2, 2, .1, .1, dtype=dtype, device=device)
    amplitude = torch.ones(grid.shape, dtype=dtype, device=device)
    if kind == "apodized":
        amplitude[0, 1] = .5
    transmission = (amplitude.bool() if kind == "bool" else 1j * amplitude)
    # Exercise native complex/bool transmissions, without a mask conversion.
    pupil = DummyPupil(grid)
    pupil.grid = grid
    pupil.transmission = transmission
    matrix = PupilOrthogonalizedBasis(NonOrthogonalBasis(grid), pupil).build_command_matrix()
    weights = amplitude.square().flatten()
    weights = weights / weights.sum()
    gram = matrix.mT @ (weights[:, None] * matrix)
    torch.testing.assert_close(gram, torch.eye(2, device=device, dtype=dtype))
    pupil.transmission = amplitude
    real_matrix = PupilOrthogonalizedBasis(NonOrthogonalBasis(grid), pupil).build_command_matrix()
    torch.testing.assert_close(matrix, real_matrix)


def test_direct_svd_does_not_overflow_by_squaring_large_finite_basis():
    grid = Grid(2, 2, .1, .1)
    pupil = build_pupil(grid, torch.ones(grid.shape))

    class OverflowingBasis(NonOrthogonalBasis):
        def build_command_matrix(self):
            return 1e30 * super().build_command_matrix()

    basis = PupilOrthogonalizedBasis(OverflowingBasis(grid), pupil)
    matrix = basis.build_command_matrix()
    torch.testing.assert_close(matrix.mT @ matrix / 4, torch.eye(2), atol=2e-6, rtol=2e-6)
    report = basis.diagnose(compare_gram=True)
    assert report["rank"] == 2
    assert "non-finite" in report["gram_failure"]


@pytest.mark.parametrize("component", ["left_vectors", "values", "right_vectors"])
@pytest.mark.parametrize("nonfinite", [float('nan'), float('inf'), -float('inf')])
def test_orthogonalization_reports_nonfinite_svd(monkeypatch, component, nonfinite):
    grid = Grid(2, 2, .1, .1)
    pupil = build_pupil(grid, torch.ones(grid.shape))
    parts = dict(left_vectors=torch.ones(4, 2), values=torch.ones(2), right_vectors=torch.eye(2))
    parts[component].flatten()[0] = nonfinite
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **kw: tuple(parts.values()))
    with pytest.raises(ValueError, match="Numerical failure.*SVD.*non-finite"):
        PupilOrthogonalizedBasis(NonOrthogonalBasis(grid), pupil).build_command_matrix()


def test_orthogonalization_reports_svd_failure(monkeypatch):
    grid = Grid(2, 2, .1, .1)
    pupil = build_pupil(grid, torch.ones(grid.shape))

    def fail(*args, **kwargs):
        raise torch.linalg.LinAlgError("did not converge")

    monkeypatch.setattr(torch.linalg, "svd", fail)
    with pytest.raises(ValueError, match="Numerical failure.*did not converge"):
        PupilOrthogonalizedBasis(NonOrthogonalBasis(grid), pupil).build_command_matrix()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_elt_orthogonalization_error_for_resolved_control_subspace(dtype):
    grid = Grid(128, 128, 40/128, 40/128, dtype=dtype)
    pupil = ELTHarmoniPupil(grid)
    pupil.build(Spectrum(0, Band(1e-6, 0., 1.), 1, dtype=dtype))
    count = 120
    matrix = PupilOrthogonalizedBasis(ZernikeBasis(grid, count), pupil).build_command_matrix()
    weights = pupil.transmission.square().flatten().double()
    weights /= weights.sum()
    error = matrix.double().mT @ (weights[:, None]*matrix.double()) - torch.eye(count, dtype=torch.float64)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-13
    assert float(error.abs().max()) < tolerance
    assert matrix.dtype == dtype and matrix.device == grid.device


class MatrixBasis(ControlBasis):
    def __init__(self, grid, matrix):
        self.pixel_grid = grid
        self.matrix = matrix

    @property
    def n_modes(self):
        return self.matrix.shape[1]

    def build_command_matrix(self):
        return self.matrix


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_svd_lowdin_is_symmetric_and_preserves_zero_weight_continuation(dtype, device):
    grid = Grid(2, 2, .1, .1, dtype=dtype, device=device)
    transmission = torch.tensor([[1., .4], [0., .9]], dtype=dtype, device=device)
    pupil = build_pupil(grid, transmission)
    original = NonOrthogonalBasis(grid).build_command_matrix()
    weights = transmission.square().flatten()
    weights /= weights.sum()
    gram = original.mT @ (weights[:, None] * original)
    values, vectors = torch.linalg.eigh(gram)
    expected_transform = (vectors * values.rsqrt()) @ vectors.mT
    orthogonal = PupilOrthogonalizedBasis(NonOrthogonalBasis(grid), pupil).build_command_matrix()
    torch.testing.assert_close(orthogonal, original @ expected_transform)
    assert torch.count_nonzero(orthogonal[weights == 0]) > 0
    torch.testing.assert_close(orthogonal @ torch.linalg.pinv(orthogonal),
                               original @ torch.linalg.pinv(original))


def test_rcond_domains_are_explicit_and_legacy_cutoff_is_preserved():
    grid = Grid(2, 2, .1, .1, dtype=torch.float64)
    original = torch.tensor([[2., 0.], [0., 2e-5], [0., 0.], [0., 0.]], dtype=grid.dtype)
    pupil = build_pupil(grid, torch.ones(grid.shape, dtype=grid.dtype))
    raw = MatrixBasis(grid, original)
    legacy = PupilOrthogonalizedBasis(raw, pupil, rcond=1e-8)
    direct = PupilOrthogonalizedBasis(raw, pupil, rcond=1e-8, rcond_on="singular_values")
    old = legacy.diagnose(compare_gram=True)
    new = direct.diagnose(compare_gram=True)
    assert old["rank"] == old["gram_rank"] == 1
    assert new["rank"] == new["gram_rank"] == 2
    assert old["relative_singular_cutoff"] == 1e-4
    assert new["relative_singular_cutoff"] == 1e-8
    with pytest.raises(ValueError, match="rank 1.*2 modes.*No modes were removed"):
        legacy.build_command_matrix()
    q = direct.build_command_matrix()
    torch.testing.assert_close(q.mT @ q / 4, torch.eye(2, dtype=grid.dtype))
    default_old = PupilOrthogonalizedBasis(raw, pupil).diagnose()
    default_new = PupilOrthogonalizedBasis(raw, pupil, rcond_on="singular_values").diagnose()
    assert default_old["relative_cutoff"] == torch.finfo(grid.dtype).eps * 2
    assert default_new["relative_cutoff"] == torch.finfo(grid.dtype).eps * 4


def test_rank_diagnostics_include_unobserved_original_mode_combinations():
    grid = Grid(2, 2, .1, .1, dtype=torch.float64)
    pupil = build_pupil(grid, torch.tensor([[1., 0.], [0., 0.]], dtype=grid.dtype))
    raw = NonOrthogonalBasis(grid)
    report = PupilOrthogonalizedBasis(raw, pupil).diagnose(compare_gram=True)
    assert report["rank"] == report["gram_rank"] == 1
    assert report["positive_pixels"] == 1
    assert report["singular_values"].shape == (2,)
    assert report["right_singular_vectors"].shape == (2, 2)
    coefficients = report["right_singular_vectors"][~report["retained"]]
    hidden_direction = raw.build_command_matrix() @ coefficients.mT
    torch.testing.assert_close(hidden_direction[:1], torch.zeros(1, 1, dtype=grid.dtype), atol=1e-15, rtol=0)
    assert torch.count_nonzero(hidden_direction[1:]) > 0


def test_svd_removes_gram_roundoff_rank_ambiguity_without_lowering_rcond():
    grid = Grid(2, 2, .1, .1, dtype=torch.float64)
    pupil = build_pupil(grid, torch.ones(grid.shape, dtype=grid.dtype))
    # Full-rank A with a singular ratio 1e-10. Squaring it loses the small
    # eigenvalue in this rotated Gram matrix, even with the equivalent cutoff.
    original = torch.tensor([[1., 1.], [1e-10, -1e-10], [0., 0.], [0., 0.]], dtype=grid.dtype)
    basis = PupilOrthogonalizedBasis(MatrixBasis(grid, original), pupil,
                                     rcond_on="singular_values")
    report = basis.diagnose(compare_gram=True)
    assert report["rank"] == 2
    assert report["gram_rank"] < 2
    q = basis.build_command_matrix()
    torch.testing.assert_close(q.mT @ q/4,
                               torch.eye(2, dtype=grid.dtype), atol=2e-15, rtol=2e-15)


def test_orthogonalization_autograd_through_basis_and_apodization():
    grid = Grid(2, 2, .1, .1, dtype=torch.float64)
    original = NonOrthogonalBasis(grid).build_command_matrix().requires_grad_()
    transmission = torch.tensor([[1., .4], [0., .9]], dtype=grid.dtype, requires_grad=True)
    pupil = DummyPupil(grid)
    pupil.grid = grid
    pupil.transmission = transmission
    q = PupilOrthogonalizedBasis(MatrixBasis(grid, original), pupil).build_command_matrix()
    gradients = torch.autograd.grad((q * torch.arange(8, dtype=grid.dtype).reshape(4, 2)).sum(),
                                    (original, transmission))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients)

    def orthogonalize(matrix, amplitude):
        pupil.transmission = amplitude
        return PupilOrthogonalizedBasis(MatrixBasis(grid, matrix), pupil).build_command_matrix()

    assert torch.autograd.gradcheck(orthogonalize, (original, transmission))


@pytest.mark.parametrize("rcond", [-1., 1., float("nan"), float("inf")])
def test_invalid_rcond_is_rejected(rcond):
    grid = Grid(2, 2, .1, .1)
    pupil = build_pupil(grid, torch.ones(grid.shape))
    with pytest.raises(ValueError, match="rcond must be finite"):
        PupilOrthogonalizedBasis(NonOrthogonalBasis(grid), pupil, rcond=rcond)


def test_invalid_rcond_domain_is_rejected():
    grid = Grid(2, 2, .1, .1)
    pupil = build_pupil(grid, torch.ones(grid.shape))
    with pytest.raises(ValueError, match="rcond_on"):
        PupilOrthogonalizedBasis(NonOrthogonalBasis(grid), pupil, rcond_on="automatic")


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_full_3240_mode_elt_lowdin_orthogonality_and_span_in_float64(device):
    grid = Grid(128, 128, 37/128, 37/128, dtype=torch.float64, device=device)
    pupil = ELTHarmoniPupil(grid)
    pupil.build(Spectrum(0, Band(1e-6, 0., 1.), 1, dtype=grid.dtype))
    original = ZernikeBasis(grid, 3240).build_command_matrix()
    raw = MatrixBasis(grid, original)
    basis = PupilOrthogonalizedBasis(raw, pupil, rcond_on="singular_values")
    q = basis.build_command_matrix()
    weights = pupil.transmission.square().flatten()
    weights /= weights.sum()
    gram = q.mT @ (weights[:, None] * q)
    assert float((gram-torch.eye(3240, dtype=grid.dtype, device=device)).abs().max()) < 5e-13
    # Both subspace directions, without a huge pixel-by-pixel projector:
    # reconstruct B from Q and check its W-norm residual; an independent QR of
    # B on the WHOLE disk also checks Q's continuation into occulted regions.
    coefficients = q.mT @ (weights[:, None] * original)
    reconstruction = q @ coefficients
    weighted_error = (weights[:, None].sqrt()*(reconstruction-original)).norm()
    weighted_norm = (weights[:, None].sqrt()*original).norm()
    assert float(weighted_error/weighted_norm) < 5e-13
    disk_q, _ = torch.linalg.qr(original, mode="reduced")
    span_residual = q - disk_q @ (disk_q.mT @ q)
    assert float(span_residual.norm()/q.norm()) < 5e-12
    assert q.shape == original.shape and q.dtype == grid.dtype and q.device == grid.device


def test_zero_basis_is_reported_as_rank_zero_even_with_zero_rcond():
    grid = Grid(2, 2, .1, .1, dtype=torch.float64)
    pupil = build_pupil(grid, torch.ones(grid.shape, dtype=grid.dtype))
    basis = PupilOrthogonalizedBasis(MatrixBasis(grid, torch.zeros(4, 2, dtype=grid.dtype)),
                                     pupil, rcond=0, rcond_on="singular_values")
    report = basis.diagnose()
    assert report["rank"] == 0 and report["sigma_min_over_max"] == 0
    with pytest.raises(ValueError, match="rank 0.*2 modes"):
        basis.build_command_matrix()


def test_3240_elt_directions_below_direct_svd_cutoff_are_not_silently_retained(monkeypatch):
    grid = Grid(128, 128, 40/128, 40/128, dtype=torch.float64)
    pupil = ELTHarmoniPupil(grid)
    pupil.build(Spectrum(0, Band(1e-6, 0., 1.), 1, dtype=grid.dtype))
    basis = PupilOrthogonalizedBasis(ZernikeBasis(grid, 3240), pupil,
                                     rcond_on="singular_values")
    decomposition = basis._weighted_svd()
    # Reuse the actual, unchanged decomposition for diagnosis and construction;
    # this avoids repeating a large SVD merely to exercise the failure path.
    monkeypatch.setattr(basis, "_weighted_svd", lambda: decomposition)
    report = basis.diagnose(compare_gram=True)
    assert 0 < report["rank"] < 3240
    assert report["singular_values"][-1] < report["singular_value_threshold"]
    with pytest.raises(ValueError, match="numerical rank.*3240 modes.*No modes were removed"):
        basis.build_command_matrix()


def test_bounded_memory_benchmark_preserves_direct_singular_spectrum():
    from benchmarks.pupil_orthogonalization import streamed_qr

    generator = torch.Generator().manual_seed(12)
    matrix = torch.randn(83, 7, generator=generator, dtype=torch.float64)
    triangular, gram = streamed_qr(matrix.split(11), 7)
    torch.testing.assert_close(torch.linalg.svdvals(triangular), torch.linalg.svdvals(matrix),
                               atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(gram, matrix.mT @ matrix)
