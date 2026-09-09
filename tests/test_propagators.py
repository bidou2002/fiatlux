import pytest
import torch

from fiatlux.core.grid import Grid
from fiatlux.core.source import PlaneWave
from fiatlux.core.spectrum import Band, Spectrum
from fiatlux.optics.propagator import IdentityPropagator, MFTPropagator


def make_field(grid):
    spectrum = Spectrum(
        0, Band(1e-6, 0.1e-6, 1e8), 2, device=grid.device, dtype=grid.dtype
    )
    return PlaneWave(spectrum).generate_field(grid)


def test_identity_propagator_returns_the_same_field_object():
    grid = Grid(7, 5, 0.1, 0.2)
    field = make_field(grid)

    assert IdentityPropagator(grid).apply(field) is field


@pytest.mark.parametrize(
    "dtype, complex_dtype",
    [(torch.float32, torch.complex64), (torch.float64, torch.complex128)],
)
def test_mft_transforms_shape_and_preserves_tensor_contract(dtype, complex_dtype):
    input_grid = Grid(7, 5, 0.1, 0.2, dtype=dtype)
    output_grid = Grid(8, 4, 1e-6, 2e-6, dtype=dtype)
    field = make_field(input_grid)

    output = MFTPropagator(2.0, output_grid).apply(field)

    assert output.grid is output_grid
    assert output.spectrum is field.spectrum
    assert output.complex_amplitude.shape == (2, 4, 8)
    assert output.complex_amplitude.dtype == complex_dtype


def test_mft_rejects_mixed_grid_precision():
    input_grid = Grid(7, 5, 0.1, 0.2, dtype=torch.float32)
    output_grid = Grid(8, 4, 1e-6, 2e-6, dtype=torch.float64)

    with pytest.raises(ValueError, match="same dtype"):
        MFTPropagator(2.0, output_grid).apply(make_field(input_grid))
