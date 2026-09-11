import pytest
import torch

from fiatlux import (
    ActuatorGrid,
    DeformableMirror,
    Grid,
    InteractionMatrix,
    PlaneWave,
    ShackHartmannLensletArray,
    ShackHartmannSlopeEstimator,
    Spectrum,
    ZernikeBasis,
)
from fiatlux.core.spectrum import Band


def make_bench(*, dtype=torch.float64, device="cpu"):
    grid = Grid(24, 24, 0.05, 0.05, dtype=dtype, device=device)
    spectrum = Spectrum(
        magnitude=0,
        band=Band(650e-9, 0.0, 368.0),
        samples=1,
        dtype=dtype,
        device=device,
    )
    field = PlaneWave(spectrum).generate_field(grid)
    actuator_grid = ActuatorGrid(1, 1, 1.0)

    def mirror():
        return DeformableMirror(
            grid,
            actuator_grid,
            grid,
            ZernikeBasis(grid, n=3),
            stroke=500e-9,
        )

    sensor = ShackHartmannLensletArray(
        grid, pitch=0.2, focal_length=2.0, spot_oversampling=4
    )
    estimator = ShackHartmannSlopeEstimator(focal_length=2.0)
    return field, mirror(), mirror(), sensor, estimator


def slopes(field, mirror, sensor, estimator):
    return estimator.measure(sensor.propagate(mirror.apply(field))).slope_vector


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_flat_and_low_order_zernike_responses_calibrate_with_expected_dtype(dtype):
    field, _, correction, sensor, estimator = make_bench(dtype=dtype)
    flat = slopes(field, correction, sensor, estimator)
    torch.testing.assert_close(flat, torch.zeros_like(flat))

    calibration = InteractionMatrix(
        correction,
        lambda: slopes(field, correction, sensor, estimator),
        poke_amplitude=10e-9,
    )
    matrix = calibration.calibrate_push_pull(verbose=False)
    control = calibration.compute_control_matrix(rcond=1e-5)

    assert matrix.shape == (2 * 6 * 6, 3)
    assert matrix.dtype == dtype
    assert control.dtype == dtype
    assert calibration.effective_rank == 3
    assert torch.all(torch.linalg.vector_norm(matrix, dim=0) > 0)


def test_interaction_matrix_recovers_tip_tilt_and_defocus_coefficients():
    field, aberration, correction, sensor, estimator = make_bench()
    calibration = InteractionMatrix(
        correction,
        lambda: slopes(field, correction, sensor, estimator),
        poke_amplitude=10e-9,
    )
    calibration.calibrate_push_pull(verbose=False)
    control = calibration.compute_control_matrix(rcond=1e-5)
    injected = torch.tensor([12e-9, -8e-9, 6e-9], dtype=field.grid.dtype)
    aberration.commands = injected

    measurement = slopes(aberration.apply(field), correction, sensor, estimator)
    reconstructed = control @ measurement

    torch.testing.assert_close(reconstructed, injected, rtol=0.04, atol=0.2e-9)


def test_closed_loop_reduces_low_order_residual():
    field, aberration, correction, sensor, estimator = make_bench()
    calibration = InteractionMatrix(
        correction,
        lambda: slopes(field, correction, sensor, estimator),
        poke_amplitude=10e-9,
    )
    calibration.calibrate_push_pull(verbose=False)
    control = calibration.compute_control_matrix(rcond=1e-5)
    aberration.commands = torch.tensor(
        [30e-9, -20e-9, 15e-9], dtype=field.grid.dtype
    )

    residuals = []
    for _ in range(6):
        measurement = slopes(
            aberration.apply(field), correction, sensor, estimator
        )
        residuals.append(torch.linalg.vector_norm(measurement))
        correction.commands = correction.commands - 0.7 * (control @ measurement)

    assert residuals[-1] < 0.02 * residuals[0]
    assert all(after < before for before, after in zip(residuals, residuals[1:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_end_to_end_calibration_runs_on_cuda():
    field, _, correction, sensor, estimator = make_bench(
        dtype=torch.float32, device="cuda"
    )
    calibration = InteractionMatrix(
        correction,
        lambda: slopes(field, correction, sensor, estimator),
        poke_amplitude=10e-9,
    )
    matrix = calibration.calibrate_push_pull(verbose=False)
    control = calibration.compute_control_matrix(rcond=1e-5)

    assert matrix.device.type == "cuda"
    assert control.device.type == "cuda"
    assert calibration.effective_rank == 3
