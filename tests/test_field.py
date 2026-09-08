import pytest
import torch

from fiatlux.core.field import Field
from fiatlux.core.grid import Grid
from fiatlux.core.spectrum import Band, Spectrum


def make_field(
    *,
    amplitude: torch.Tensor | None = None,
    grid: Grid | None = None,
    spectrum: Spectrum | None = None,
) -> Field:
    grid = grid or Grid(nx=3, ny=2, dx=0.1, dy=0.2)
    spectrum = spectrum or Spectrum(
        magnitude=0,
        band=Band(central_wavelength=1e-6, delta_wavelength=0.2e-6, f0=1),
        samples=2,
    )
    amplitude = amplitude if amplitude is not None else torch.ones(
        (2, 2, 3), dtype=torch.complex64
    )
    return Field(amplitude, grid, spectrum)


def test_to_returns_new_consistent_field_without_mutating_source():
    field = make_field()

    moved = field.to("cpu")

    assert moved is not field
    assert moved.grid is not field.grid
    assert moved.spectrum is not field.spectrum
    assert moved.grid.device == torch.device("cpu")
    assert moved.complex_amplitude.device == torch.device("cpu")
    assert moved.spectrum.wavelengths.device == torch.device("cpu")
    assert moved.spectrum.fluxes.device == torch.device("cpu")
    assert moved.complex_amplitude.dtype == field.complex_amplitude.dtype
    assert moved.spectrum.wavelengths.dtype == field.spectrum.wavelengths.dtype
    assert moved.spectrum.fluxes.dtype == field.spectrum.fluxes.dtype
    torch.testing.assert_close(moved.complex_amplitude, field.complex_amplitude)
    torch.testing.assert_close(moved.spectrum.wavelengths, field.spectrum.wavelengths)
    torch.testing.assert_close(moved.spectrum.fluxes, field.spectrum.fluxes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_to_cuda_moves_every_tensor_and_preserves_dtype():
    field = make_field()

    moved = field.to("cuda")

    assert moved.grid.device.type == "cuda"
    assert moved.complex_amplitude.device.type == "cuda"
    assert moved.spectrum.wavelengths.device.type == "cuda"
    assert moved.spectrum.fluxes.device.type == "cuda"
    assert moved.complex_amplitude.dtype == torch.complex64
    assert moved.spectrum.wavelengths.dtype == field.spectrum.wavelengths.dtype
    assert moved.spectrum.fluxes.dtype == field.spectrum.fluxes.dtype
    assert field.grid.device.type == "cpu"
    assert field.complex_amplitude.device.type == "cpu"


def test_field_addition_and_subtraction():
    left = make_field(amplitude=torch.full((2, 2, 3), 3, dtype=torch.complex64))
    right = make_field(
        amplitude=torch.full((2, 2, 3), 2, dtype=torch.complex64),
        grid=left.grid,
        spectrum=left.spectrum,
    )

    torch.testing.assert_close(
        (left + right).complex_amplitude,
        torch.full_like(left.complex_amplitude, 5),
    )
    torch.testing.assert_close(
        (left - right).complex_amplitude,
        torch.ones_like(left.complex_amplitude),
    )


def test_tensor_and_scalar_arithmetic():
    field = make_field()
    tensor = torch.full((2, 2, 3), 2, dtype=torch.complex64)

    torch.testing.assert_close((field + tensor).complex_amplitude, tensor + 1)
    torch.testing.assert_close((2 + field).complex_amplitude, tensor + 1)
    torch.testing.assert_close((field - tensor).complex_amplitude, -torch.ones_like(tensor))
    torch.testing.assert_close((2 - field).complex_amplitude, torch.ones_like(tensor))
    torch.testing.assert_close((field * tensor).complex_amplitude, tensor)
    torch.testing.assert_close((2 * field).complex_amplitude, tensor)


@pytest.mark.parametrize("difference", ["grid", "shape", "wavelengths", "fluxes"])
def test_field_arithmetic_rejects_incompatible_fields(difference):
    left = make_field()
    right = make_field(grid=left.grid, spectrum=left.spectrum)

    if difference == "grid":
        right.grid = Grid(nx=3, ny=2, dx=0.11, dy=0.2)
    elif difference == "shape":
        right.complex_amplitude = torch.ones((2, 2, 2), dtype=torch.complex64)
    elif difference == "wavelengths":
        right.spectrum = Spectrum(0, Band(1.1e-6, 0.2e-6, 1), 2)
    else:
        right.spectrum = Spectrum(1, Band(1e-6, 0.2e-6, 1), 2)

    with pytest.raises(ValueError, match="must (be|have) identical"):
        left - right


def test_unsupported_arithmetic_returns_type_error():
    field = make_field()

    with pytest.raises(TypeError):
        field + object()
    with pytest.raises(TypeError):
        field * field
