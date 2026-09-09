import pytest
import torch

from fiatlux.system.interaction_matrix import InteractionMatrix


class LinearDM:
    def __init__(self, n_commands=2, stroke=1e-6):
        self._commands = torch.nn.Parameter(torch.zeros(n_commands))
        self.stroke = stroke

    @property
    def commands(self):
        return self._commands

    @commands.setter
    def commands(self, value):
        value = torch.as_tensor(value, dtype=self._commands.dtype)
        with torch.no_grad():
            self._commands.copy_(value)


def make_linear_calibration(reference_commands=None):
    dm = LinearDM()
    response_matrix = torch.tensor(
        [[2.0, -1.0], [0.5, 3.0], [-4.0, 2.0], [1.5, 0.25]]
    ) * 1e8
    background = torch.tensor([7.0, 11.0, -5.0, 2.0])

    def acquire():
        return (response_matrix @ dm.commands + background).reshape(2, 2)

    calibration = InteractionMatrix(
        dm=dm,
        acquiring_function=acquire,
        poke_amplitude=10e-9,
        reference_commands=reference_commands,
    )
    return calibration, dm, response_matrix, background


def test_one_sided_returns_derivative_without_static_background():
    calibration, dm, expected, background = make_linear_calibration()

    matrix = calibration.calibrate_one_sided(verbose=False)

    torch.testing.assert_close(matrix, expected, rtol=2e-6, atol=32.0)
    torch.testing.assert_close(calibration.reference_response.flatten(), background)
    assert calibration.measurement_shape == torch.Size([2, 2])


def test_push_pull_returns_central_derivative():
    calibration, _, expected, _ = make_linear_calibration(
        reference_commands=torch.tensor([100e-9, -50e-9])
    )

    matrix = calibration.calibrate_push_pull(verbose=False)

    torch.testing.assert_close(matrix, expected, rtol=2e-6, atol=32.0)


@pytest.mark.parametrize("method", ["calibrate_one_sided", "calibrate_push_pull"])
def test_original_dm_commands_are_restored(method):
    calibration, dm, _, _ = make_linear_calibration()
    original = torch.tensor([300e-9, -200e-9])
    dm.commands = original

    getattr(calibration, method)(verbose=False)

    torch.testing.assert_close(dm.commands, original)


def test_dm_commands_are_restored_when_acquisition_fails():
    dm = LinearDM()
    original = torch.tensor([300e-9, -200e-9])
    dm.commands = original

    def fail():
        raise RuntimeError("camera failure")

    calibration = InteractionMatrix(dm, fail, poke_amplitude=10e-9)

    with pytest.raises(RuntimeError, match="camera failure"):
        calibration.calibrate_one_sided(verbose=False)

    torch.testing.assert_close(dm.commands, original)


def test_measurement_shape_must_remain_constant():
    dm = LinearDM()
    calls = 0

    def changing_shape():
        nonlocal calls
        calls += 1
        return torch.zeros(2 if calls == 1 else 3)

    calibration = InteractionMatrix(dm, changing_shape, poke_amplitude=10e-9)

    with pytest.raises(ValueError, match="shape changed"):
        calibration.calibrate_one_sided(verbose=False)


@pytest.mark.parametrize("poke_amplitude", [0.0, -1e-9, float("inf")])
def test_invalid_poke_amplitude_is_rejected(poke_amplitude):
    with pytest.raises(ValueError):
        InteractionMatrix(LinearDM(), lambda: torch.zeros(1), poke_amplitude)


def test_reference_and_poke_must_fit_inside_stroke():
    with pytest.raises(ValueError, match="exceed the DM stroke"):
        InteractionMatrix(
            LinearDM(stroke=100e-9),
            lambda: torch.zeros(1),
            poke_amplitude=20e-9,
            reference_commands=torch.tensor([90e-9, 0.0]),
        )


def test_acquisition_must_return_finite_nonempty_tensor():
    dm = LinearDM()

    for bad_response in ([], torch.empty(0), torch.tensor([float("nan")])):
        calibration = InteractionMatrix(
            dm,
            lambda response=bad_response: response,
            poke_amplitude=10e-9,
        )
        with pytest.raises((TypeError, ValueError)):
            calibration.calibrate_one_sided(verbose=False)


def test_legacy_calibration_method_names_are_removed():
    calibration, _, _, _ = make_linear_calibration()

    assert not hasattr(calibration, "measure")
    assert not hasattr(calibration, "push_pull")


def calibration_from_matrix(matrix):
    calibration, _, _, _ = make_linear_calibration()
    calibration.matrix = torch.as_tensor(matrix, dtype=torch.float64)
    return calibration


def test_zero_singular_value_is_safely_discarded():
    calibration = calibration_from_matrix([[2.0, 0.0], [0.0, 0.0]])

    control = calibration.compute_control_matrix()

    assert torch.isfinite(control).all()
    torch.testing.assert_close(control, torch.tensor([[0.5, 0.0], [0.0, 0.0]], dtype=torch.float64))
    assert calibration.effective_rank == 1
    assert calibration.retained_mode_indices.tolist() == [0]


def test_relative_threshold_discards_poorly_conditioned_mode():
    calibration = calibration_from_matrix([[10.0, 0.0], [0.0, 1e-4]])

    calibration.compute_control_matrix(rcond=1e-3)

    assert calibration.effective_rank == 1
    assert calibration.retained_mode_indices.tolist() == [0]
    assert calibration.S.tolist() == pytest.approx([10.0, 1e-4])


def test_absolute_threshold_and_mode_cap_are_combined():
    calibration = calibration_from_matrix(torch.diag(torch.tensor([4.0, 2.0, 1.0])))

    calibration.compute_control_matrix(n_modes=2, rcond=0.0, atol=1.5)

    assert calibration.retained_mode_indices.tolist() == [0, 1]
    assert calibration.effective_rank == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_modes": 0},
        {"n_modes": 1.5},
        {"rcond": -1.0},
        {"rcond": float("inf")},
        {"atol": -1.0},
    ],
)
def test_invalid_svd_filter_settings_are_rejected(kwargs):
    calibration = calibration_from_matrix(torch.eye(2))

    with pytest.raises(ValueError):
        calibration.compute_control_matrix(**kwargs)


def test_svd_diagnostics_are_exposed():
    calibration = calibration_from_matrix([[4.0, 0.0], [0.0, 2.0]])

    calibration.compute_control_matrix()

    torch.testing.assert_close(
        calibration.singular_values, torch.tensor([4.0, 2.0], dtype=torch.float64)
    )
    assert calibration.condition_number == pytest.approx(2.0)


def test_diagnostics_require_control_matrix_computation():
    calibration = calibration_from_matrix(torch.eye(2))

    with pytest.raises(RuntimeError, match="compute_control_matrix"):
        _ = calibration.singular_values
    with pytest.raises(RuntimeError, match="compute_control_matrix"):
        _ = calibration.condition_number


def test_plot_modes_supports_one_dimensional_measurements():
    import matplotlib.pyplot as plt

    calibration = calibration_from_matrix(
        [[2.0, 0.0], [1.0, 1.0], [0.0, 2.0]]
    )
    calibration.measurement_shape = torch.Size([3])
    calibration.compute_control_matrix()

    figure, axes = calibration.plot_modes()

    assert len(axes.flat[0].lines) == 1
    plt.close(figure)


def test_plot_modes_supports_explicit_rectangular_shape():
    import matplotlib.pyplot as plt

    calibration = calibration_from_matrix(torch.arange(12).reshape(6, 2))
    calibration.compute_control_matrix()

    figure, axes = calibration.plot_modes(measurement_shape=(2, 3))

    assert axes.flat[0].images[0].get_array().shape == (2, 3)
    plt.close(figure)


@pytest.mark.parametrize("shape", [(2, 2), (1, 2, 3)])
def test_plot_modes_rejects_incompatible_display_shape(shape):
    calibration = calibration_from_matrix(torch.arange(12).reshape(6, 2))
    calibration.compute_control_matrix()

    with pytest.raises(ValueError, match="measurement_shape"):
        calibration.plot_modes(measurement_shape=shape)
