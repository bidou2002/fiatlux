import math

import pytest
import torch

from fiatlux import (
    ActuatorGrid,
    AffineRegistration,
    Grid,
    SLMCalibration,
    SLMEncoder,
    SquareZonalBasis,
    illumination_fractions,
    select_registration_pokes,
    square_zonal_centers,
    weighted_centroid,
)


def linear_calibration(dtype=torch.float64):
    return SLMCalibration(
        wavelength=500e-9,
        drive_levels=torch.tensor([0.0, 100.0, 200.0, 255.0], dtype=dtype),
        phase_radians=torch.tensor(
            [0.0, math.pi / 2, math.pi, 2 * math.pi], dtype=dtype
        ),
    )


def test_calibration_wraps_interpolates_and_quantizes_phase():
    calibration = linear_calibration()
    phase = torch.tensor(
        [0.0, math.pi / 4, math.pi, 2 * math.pi, 2 * math.pi + math.pi / 4],
        dtype=torch.float64,
    )

    drive = calibration.phase_to_drive(phase)

    assert drive.dtype == torch.uint8
    torch.testing.assert_close(
        drive,
        torch.tensor([0, 50, 200, 0, 50], dtype=torch.uint8),
    )


@pytest.mark.parametrize(
    "phases",
    [
        [0.1, 1.0],
        [0.0, 1.0, 0.5, 2 * math.pi],
    ],
)
def test_calibration_rejects_incomplete_or_non_monotonic_phase_table(phases):
    with pytest.raises(ValueError):
        SLMCalibration(
            wavelength=500e-9,
            drive_levels=torch.linspace(0, 255, len(phases)),
            phase_radians=torch.tensor(phases),
        )


def test_affine_fit_recovers_anisotropic_scale_rotation_shear_and_translation():
    source = torch.tensor(
        [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    matrix = torch.tensor([[4.0, 0.6], [-0.3, 2.0]], dtype=torch.float64)
    translation = torch.tensor([960.25, 540.75], dtype=torch.float64)
    measured = source @ matrix.mT + translation

    registration, residuals = AffineRegistration.fit(
        source, measured, output_shape=(1080, 1920)
    )

    torch.testing.assert_close(registration.matrix, matrix)
    torch.testing.assert_close(registration.translation, translation)
    torch.testing.assert_close(residuals, torch.zeros_like(residuals), atol=1e-12, rtol=0)


def test_affine_fit_rejects_collinear_pokes():
    source = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match="must not be collinear"):
        AffineRegistration.fit(source, source, output_shape=(10, 10))


def test_identity_registration_preserves_opd_on_odd_rectangular_grid():
    grid = Grid(nx=7, ny=5, dx=0.2, dy=0.3, dtype=torch.float64)
    opd = torch.linspace(-3e-6, 3e-6, grid.nx * grid.ny, dtype=grid.dtype).reshape(
        grid.shape
    )
    registration = AffineRegistration(
        matrix=torch.tensor(
            [[1 / grid.dx, 0.0], [0.0, 1 / grid.dy]], dtype=grid.dtype
        ),
        translation=torch.tensor(
            [grid.nx // 2, grid.ny // 2], dtype=grid.dtype
        ),
        output_shape=grid.shape,
    )

    registered = registration.resample_opd(opd, grid)

    torch.testing.assert_close(registered, opd, atol=1e-15, rtol=1e-12)


def test_registration_interpolates_continuous_opd_before_phase_wrapping():
    grid = Grid(nx=2, ny=1, dx=1.0, dy=1.0, dtype=torch.float64)
    opd = torch.tensor([[0.0, 1e-6]], dtype=grid.dtype)
    registration = AffineRegistration(
        matrix=torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=grid.dtype),
        translation=torch.tensor([2.0, 0.0], dtype=grid.dtype),
        output_shape=(1, 3),
    )

    registered = registration.resample_opd(opd, grid)

    torch.testing.assert_close(
        registered[0, 1], torch.tensor(0.5e-6, dtype=grid.dtype)
    )


def test_encoder_converts_opd_to_registered_integer_frame():
    grid = Grid(3, 3, 1.0, 1.0, dtype=torch.float64)
    registration = AffineRegistration(
        torch.eye(2, dtype=grid.dtype),
        torch.tensor([1.0, 1.0], dtype=grid.dtype),
        grid.shape,
    )
    encoder = SLMEncoder(linear_calibration(), registration)
    opd = torch.zeros(grid.shape, dtype=grid.dtype)
    opd[1, 1] = 125e-9  # pi/2 at 500 nm

    frame = encoder.encode_opd(opd, grid)

    assert frame.shape == grid.shape
    assert frame.dtype == torch.uint8
    assert frame[1, 1] == 100


def test_opd_registration_is_differentiable_before_integer_encoding():
    grid = Grid(3, 3, 1.0, 1.0, dtype=torch.float64)
    registration = AffineRegistration(
        torch.eye(2, dtype=grid.dtype),
        torch.tensor([1.0, 1.0], dtype=grid.dtype),
        grid.shape,
    )
    opd = torch.linspace(-1e-6, 1e-6, 9, dtype=grid.dtype).reshape(3, 3)
    opd.requires_grad_()

    registered = registration.resample_opd(opd, grid)
    registered.square().sum().backward()

    assert opd.grad is not None
    assert torch.isfinite(opd.grad).all()


def make_dense_basis(dtype=torch.float64):
    grid = Grid(13, 13, 1.0, 1.0, dtype=dtype)
    actuator_grid = ActuatorGrid(3, 3, pitch=4.0)
    return SquareZonalBasis(actuator_grid, grid, influence_width=1.1)


def test_square_zonal_centers_follow_command_matrix_column_order():
    basis = make_dense_basis()
    centers = square_zonal_centers(basis)
    matrix = basis.build_command_matrix()
    x, y = basis.pixel_grid.meshgrid()

    measured = []
    for column in matrix.mT:
        support = column.reshape(basis.pixel_grid.shape).bool()
        measured.append(torch.stack((x[support].mean(), y[support].mean())))

    torch.testing.assert_close(centers, torch.stack(measured))


def test_illumination_filter_and_poke_selection_use_dense_basis_modes():
    basis = make_dense_basis()
    pupil = torch.ones(basis.pixel_grid.shape, dtype=basis.pixel_grid.dtype)
    pupil[:4, :] = 0

    fractions = illumination_fractions(basis, pupil)
    selected = select_registration_pokes(
        basis, pupil, count=5, minimum_illumination=0.95
    )

    assert selected.shape == (5,)
    assert selected.unique().numel() == 5
    assert torch.all(fractions[selected] >= 0.95)


def test_weighted_centroid_finds_subpixel_response_center():
    y, x = torch.meshgrid(
        torch.arange(9, dtype=torch.float64),
        torch.arange(11, dtype=torch.float64),
        indexing="ij",
    )
    response = torch.exp(-((x - 6.25) ** 2 + (y - 3.5) ** 2) / 4)

    center = weighted_centroid(response, threshold_fraction=0.0)

    torch.testing.assert_close(
        center, torch.tensor([6.25, 3.5], dtype=response.dtype), atol=2e-2, rtol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_slm_encoder_preserves_cuda_execution_until_wrapper_boundary():
    grid = Grid(3, 3, 1.0, 1.0, device=torch.device("cuda"))
    calibration = SLMCalibration(
        500e-9,
        torch.tensor([0.0, 255.0]),
        torch.tensor([0.0, 2 * math.pi]),
    )
    registration = AffineRegistration(
        torch.eye(2), torch.tensor([1.0, 1.0]), grid.shape
    )

    frame = SLMEncoder(calibration, registration).encode_opd(
        torch.zeros(grid.shape, device="cuda"), grid
    )

    assert frame.device.type == "cuda"
