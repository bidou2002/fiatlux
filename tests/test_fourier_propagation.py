import pytest
import torch

from fiatlux.core.field import Field
from fiatlux.core.grid import Grid
from fiatlux.core.spectrum import Band, Spectrum
from fiatlux.optics.propagator import (
    FFTPropagator,
    MFTPropagator,
    PropagationRegime,
    PropagationSamplingError,
)


WAVELENGTH = 632.8e-9
SCALE = 0.2


def make_field(nx=9, ny=7, *, samples=1, dtype=torch.float64):
    grid = Grid(nx, ny, 20e-6, 30e-6, dtype=dtype)
    spectrum = Spectrum(
        magnitude=0,
        band=Band(WAVELENGTH, 20e-9 if samples > 1 else 0.0, 1.0),
        samples=samples,
        dtype=dtype,
    )
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    generator = torch.Generator().manual_seed(12)
    real = torch.randn(samples, ny, nx, generator=generator, dtype=dtype)
    imaginary = torch.randn(samples, ny, nx, generator=generator, dtype=dtype)
    amplitude = torch.complex(real, imaginary).to(complex_dtype)
    return Field(amplitude, grid, spectrum)


def natural_grid(field, scale=SCALE):
    wavelength = float(field.spectrum.wavelengths[0])
    return Grid(
        field.grid.nx,
        field.grid.ny,
        wavelength * abs(scale) / (field.grid.nx * field.grid.dx),
        wavelength * abs(scale) / (field.grid.ny * field.grid.dy),
        dtype=field.grid.dtype,
        device=field.grid.device,
    )


def integrated_flux(field):
    return field.intensity().sum(dim=(-2, -1)) * field.grid.dx * field.grid.dy


def test_fraunhofer_is_default_for_both_numerical_methods():
    field = make_field()
    grid = natural_grid(field)

    assert MFTPropagator(SCALE, grid).propagation is PropagationRegime.FRAUNHOFER
    assert FFTPropagator(SCALE).propagation is PropagationRegime.FRAUNHOFER


@pytest.mark.parametrize("regime", list(PropagationRegime))
@pytest.mark.parametrize("shape", [(9, 7), (8, 6)])
def test_mft_and_fft_agree_on_natural_output_grid(regime, shape):
    field = make_field(nx=shape[0], ny=shape[1])
    output_grid = natural_grid(field)

    if regime is PropagationRegime.FRAUNHOFER:
        mft = MFTPropagator(SCALE, output_grid)
        fft = FFTPropagator(SCALE)
    else:
        mft = MFTPropagator(
            output_grid=output_grid, propagation=regime, distance=SCALE
        )
        fft = FFTPropagator(propagation=regime, distance=SCALE)

    mft_field = mft.apply(field)
    fft_field = fft.apply(field)
    torch.testing.assert_close(
        mft_field.complex_amplitude,
        fft_field.complex_amplitude,
        rtol=2e-10,
        atol=2e-10,
    )


@pytest.mark.parametrize("regime", list(PropagationRegime))
def test_mft_and_fft_preserve_flux_on_natural_grid(regime):
    field = make_field()
    grid = natural_grid(field)
    if regime is PropagationRegime.FRAUNHOFER:
        propagators = [MFTPropagator(SCALE, grid), FFTPropagator(SCALE)]
    else:
        propagators = [
            MFTPropagator(output_grid=grid, propagation=regime, distance=SCALE),
            FFTPropagator(propagation=regime, distance=SCALE),
        ]

    for propagator in propagators:
        torch.testing.assert_close(
            integrated_flux(propagator.apply(field)),
            integrated_flux(field),
            rtol=2e-10,
            atol=2e-10,
        )


def test_negative_fresnel_distance_agrees_between_mft_and_fft():
    field = make_field()
    grid = natural_grid(field, scale=-SCALE)
    mft = MFTPropagator(
        output_grid=grid, propagation="fresnel", distance=-SCALE
    ).apply(field)
    fft = FFTPropagator(propagation="fresnel", distance=-SCALE).apply(field)

    torch.testing.assert_close(
        mft.complex_amplitude, fft.complex_amplitude, rtol=2e-10, atol=2e-10
    )
    torch.testing.assert_close(
        integrated_flux(fft), integrated_flux(field), rtol=2e-10, atol=2e-10
    )


def test_fresnel_mft_supports_polychromatic_fields_on_a_common_grid():
    field = make_field(samples=3)
    output = MFTPropagator(
        output_grid=natural_grid(make_field()),
        propagation="fresnel",
        distance=SCALE,
    ).apply(field)

    assert output.complex_amplitude.shape == (3, field.grid.ny, field.grid.nx)
    assert output.spectrum is field.spectrum


def test_fft_rejects_polychromatic_field_with_actionable_error():
    field = make_field(samples=3)
    with pytest.raises(
        PropagationSamplingError, match="monochromatic.*MFTPropagator"
    ):
        FFTPropagator(SCALE).apply(field)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("regime", list(PropagationRegime))
def test_fft_preserves_dtype_device_and_shape(dtype, regime):
    field = make_field(dtype=dtype)
    if regime is PropagationRegime.FRAUNHOFER:
        output = FFTPropagator(SCALE).apply(field)
    else:
        output = FFTPropagator(propagation=regime, distance=SCALE).apply(field)

    assert output.complex_amplitude.dtype == field.complex_amplitude.dtype
    assert output.complex_amplitude.device == field.complex_amplitude.device
    assert output.complex_amplitude.shape == field.complex_amplitude.shape
    assert output.grid.dtype == field.grid.dtype


@pytest.mark.parametrize("regime", list(PropagationRegime))
@pytest.mark.parametrize("method", ["mft", "fft"])
def test_propagation_is_differentiable_with_respect_to_input_field(regime, method):
    field = make_field()
    amplitude = field.complex_amplitude.detach().requires_grad_(True)
    differentiable_field = Field(amplitude, field.grid, field.spectrum)
    output_grid = natural_grid(field)
    if method == "mft" and regime is PropagationRegime.FRAUNHOFER:
        propagator = MFTPropagator(SCALE, output_grid)
    elif method == "mft":
        propagator = MFTPropagator(
            output_grid=output_grid, propagation=regime, distance=SCALE
        )
    elif regime is PropagationRegime.FRAUNHOFER:
        propagator = FFTPropagator(SCALE)
    else:
        propagator = FFTPropagator(propagation=regime, distance=SCALE)

    output = propagator.apply(differentiable_field)

    output.intensity().square().sum().backward()
    assert amplitude.grad is not None
    assert torch.isfinite(amplitude.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("regime", list(PropagationRegime))
def test_fft_runs_on_cuda(regime):
    field = make_field().to(device="cuda")
    if regime is PropagationRegime.FRAUNHOFER:
        output = FFTPropagator(SCALE).apply(field)
    else:
        output = FFTPropagator(propagation=regime, distance=SCALE).apply(field)

    assert output.complex_amplitude.device.type == "cuda"
    assert output.grid.device.type == "cuda"


def test_zero_distance_fresnel_is_exact_identity():
    field = make_field()
    fft_output = FFTPropagator(propagation="fresnel", distance=0).apply(field)
    mft_output = MFTPropagator(
        output_grid=field.grid, propagation="fresnel", distance=0
    ).apply(field)

    assert fft_output is field
    torch.testing.assert_close(
        mft_output.complex_amplitude, field.complex_amplitude, rtol=0, atol=0
    )


def test_fresnel_requires_an_explicit_finite_distance():
    field = make_field()
    with pytest.raises(TypeError, match="distance"):
        FFTPropagator(propagation="fresnel")
    with pytest.raises(ValueError, match="finite"):
        MFTPropagator(
            output_grid=field.grid, propagation="fresnel", distance=float("nan")
        )


def test_unknown_regime_is_rejected():
    with pytest.raises(ValueError, match="fraunhofer, fresnel"):
        FFTPropagator(SCALE, propagation="nearish")
