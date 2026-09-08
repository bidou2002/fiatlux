import pytest
import torch

from fiatlux.core.grid import Grid
from fiatlux.core.spectrum import Band, Spectrum
from fiatlux.optics.elements.deformable_mirror import (
    ActuatorGrid,
    ControlBasis,
    DeformableMirror,
)


class TwoPixelBasis(ControlBasis):
    def __init__(self, pixel_grid: Grid):
        self.pixel_grid = pixel_grid

    def build_command_matrix(self) -> torch.Tensor:
        matrix = torch.zeros(self.pixel_grid.nx * self.pixel_grid.ny, 2)
        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0
        return matrix

    def n_modes(self) -> int:
        return 2


def make_dm(stroke=1e-6):
    grid = Grid(nx=3, ny=2, dx=0.1, dy=0.2)
    return DeformableMirror(
        grid=grid,
        actuator_grid=ActuatorGrid(2, 1, 0.1),
        pixel_grid=grid,
        control_basis=TwoPixelBasis(grid),
        stroke=stroke,
    )


def test_commands_are_opd_coefficients_in_metres():
    dm = make_dm()
    dm.commands = torch.tensor([120e-9, -80e-9])

    torch.testing.assert_close(dm.commands, torch.tensor([120e-9, -80e-9]))
    assert dm.opd.shape == (2, 3)
    torch.testing.assert_close(dm.opd[0, :2], dm.commands)


def test_commands_saturate_at_physical_stroke():
    dm = make_dm(stroke=500e-9)
    dm.commands = torch.tensor([800e-9, -700e-9])

    torch.testing.assert_close(
        dm.applied_commands,
        torch.tensor([500e-9, -500e-9]),
    )


def test_setting_commands_preserves_registered_parameter():
    dm = make_dm()
    parameter = dm._commands

    dm.commands = torch.tensor([10e-9, 20e-9])

    assert dm._commands is parameter
    assert list(dm.parameters()) == [parameter]
    assert "_command_matrix" in dict(dm.named_buffers())


def test_commands_remain_differentiable_inside_stroke():
    dm = make_dm()
    loss = dm.opd.square().sum()
    loss.backward()

    assert dm._commands.grad is not None


def test_opd_to_phase_conversion_is_analytical():
    dm = make_dm()
    dm.commands = torch.tensor([100e-9, 0.0])
    wavelength = 1e-6
    spectrum = Spectrum(
        magnitude=0,
        band=Band(wavelength, 0.0, 1.0),
        samples=1,
    )

    dm._build(spectrum)

    expected_phase = torch.as_tensor(
        2 * torch.pi * 100e-9 / wavelength,
        dtype=dm.complex_transmission.real.dtype,
    )
    torch.testing.assert_close(dm.complex_transmission.angle()[0, 0, 0], expected_phase)


@pytest.mark.parametrize(
    "value",
    [
        torch.zeros(3),
        torch.tensor([float("nan"), 0.0]),
        torch.tensor([float("inf"), 0.0]),
    ],
)
def test_invalid_commands_are_rejected(value):
    with pytest.raises(ValueError):
        make_dm().commands = value


def test_non_positive_stroke_is_rejected():
    with pytest.raises(ValueError):
        make_dm(stroke=0.0)
