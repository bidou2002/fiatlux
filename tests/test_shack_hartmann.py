import math

import pytest
import torch

from fiatlux import (
    Field,
    Grid,
    PlaneWave,
    ShackHartmannLensletArray,
    Spectrum,
)
from fiatlux.core.spectrum import Band


def monochromatic_field(grid, wavelength=500e-9):
    spectrum = Spectrum(
        magnitude=0,
        band=Band(wavelength, 0.0, 368.0),
        samples=1,
        dtype=grid.dtype,
        device=grid.device,
    )
    return PlaneWave(spectrum).generate_field(grid)


def test_flat_wavefront_forms_regular_spots_and_conserves_flux():
    grid = Grid(16, 16, 0.1, 0.1, dtype=torch.float64)
    field = monochromatic_field(grid)
    sensor = ShackHartmannLensletArray(
        grid, pitch=0.4, focal_length=2.0
    )

    spots = sensor.propagate(field)
    assert spots.complex_amplitude.shape == (1, 4, 4, 4, 4)
    peak = spots.pixel_flux[0].reshape(16, -1).argmax(dim=1)
    assert torch.equal(peak, torch.full((16,), 2 * 4 + 2))
    input_flux = field.intensity().sum() * grid.dx * grid.dy
    torch.testing.assert_close(spots.pixel_flux.sum(), input_flux)

    mosaic = spots.mosaic()
    assert mosaic.shape == (16, 16)
    assert int((mosaic > 0).sum()) == 16


def test_known_wavefront_tilt_moves_every_spot_with_correct_sign_and_scale():
    wavelength = 500e-9
    grid = Grid(16, 16, 0.1, 0.1, dtype=torch.float64)
    field = monochromatic_field(grid, wavelength)
    sensor = ShackHartmannLensletArray(grid, pitch=0.4, focal_length=2.0)
    angle_x = wavelength / sensor.pitch
    x, _ = grid.meshgrid()
    tilted = Field(
        field.complex_amplitude
        * torch.exp(2j * math.pi * angle_x * x[None] / wavelength),
        grid,
        field.spectrum,
    )

    spots = sensor.propagate(tilted)
    peak = spots.pixel_flux[0, 0, 0].argmax()
    peak_y, peak_x = divmod(int(peak), sensor.samples_x)
    assert (peak_y, peak_x) == (2, 3)
    measured_position = (peak_x - 2) * float(spots.pixel_scale_x[0])
    assert measured_position == pytest.approx(sensor.focal_length * angle_x)


def test_rectangular_grid_polychromatic_sampling_dtype_and_autograd():
    grid = Grid(12, 8, 0.1, 0.2, dtype=torch.float64)
    spectrum = Spectrum(
        magnitude=0,
        band=Band(600e-9, 200e-9, 368.0),
        samples=3,
        dtype=torch.float64,
    )
    source_field = PlaneWave(spectrum).generate_field(grid)
    amplitude = source_field.complex_amplitude.detach().clone().requires_grad_(True)
    field = Field(amplitude, grid, spectrum)
    sensor = ShackHartmannLensletArray(grid, pitch=0.4, focal_length=3.0)

    spots = sensor.propagate(field)
    assert spots.complex_amplitude.shape == (3, 4, 3, 2, 4)
    assert spots.complex_amplitude.dtype == torch.complex128
    assert spots.pixel_scale_x.dtype == torch.float64
    assert torch.all(spots.pixel_scale_x[1:] > spots.pixel_scale_x[:-1])
    assert torch.all(spots.pixel_scale_y[1:] > spots.pixel_scale_y[:-1])
    lens_phase = sensor.lenslet_phase(spectrum.wavelengths)
    assert lens_phase.shape == (3, 2, 4)
    assert torch.all(lens_phase <= 0)
    assert lens_phase[0].abs().max() > lens_phase[-1].abs().max()
    spots.intensity.sum().backward()
    assert amplitude.grad is not None
    assert torch.isfinite(amplitude.grad).all()


def test_registration_and_valid_subapertures_are_explicit():
    grid = Grid(10, 10, 0.1, 0.1)
    valid = torch.tensor([[True, False], [True, True]])
    sensor = ShackHartmannLensletArray(
        grid,
        pitch=0.2,
        focal_length=1.0,
        n_lenslets_x=2,
        n_lenslets_y=2,
        registration_x=0.1,
        registration_y=-0.1,
        valid_subapertures=valid,
    )
    assert (sensor.start_y, sensor.start_x) == ((2, 4))

    spots = sensor.propagate(monochromatic_field(grid))
    assert spots.pixel_flux[0, 0, 1].sum() == 0
    assert torch.all(spots.pixel_flux[0, valid].sum(dim=(-2, -1)) > 0)


def test_pitch_and_registration_must_align_with_input_samples():
    grid = Grid(10, 10, 0.1, 0.1)
    with pytest.raises(ValueError, match="pitch / dx"):
        ShackHartmannLensletArray(grid, pitch=0.25, focal_length=1.0)
    with pytest.raises(ValueError, match="registration_x"):
        ShackHartmannLensletArray(
            grid, pitch=0.2, focal_length=1.0, registration_x=0.05
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_device_is_preserved():
    grid = Grid(8, 8, 0.1, 0.1, device="cuda")
    field = monochromatic_field(grid)
    spots = ShackHartmannLensletArray(
        grid, pitch=0.4, focal_length=1.0
    ).propagate(field)
    assert spots.complex_amplitude.device.type == "cuda"
    assert spots.pixel_scale_x.device.type == "cuda"
    assert spots.valid_subapertures.device.type == "cuda"
