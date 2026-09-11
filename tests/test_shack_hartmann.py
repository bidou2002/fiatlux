import math

import pytest
import torch

from fiatlux import (
    Field,
    Grid,
    PlaneWave,
    ShackHartmannDetector,
    ShackHartmannImage,
    ShackHartmannLensletArray,
    ShackHartmannSlopeEstimator,
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


def test_oversampled_spots_refine_sampling_and_conserve_flux():
    grid = Grid(8, 8, 0.1, 0.1, dtype=torch.float64)
    field = monochromatic_field(grid)
    sensor = ShackHartmannLensletArray(
        grid, pitch=0.4, focal_length=2.0, spot_oversampling=4
    )

    spots = sensor.propagate(field)
    assert spots.complex_amplitude.shape == (1, 2, 2, 16, 16)
    assert float(spots.pixel_scale_x[0]) == pytest.approx(
        field.spectrum.wavelengths[0] * 2.0 / (16 * grid.dx)
    )
    input_flux = field.intensity().sum() * grid.dx * grid.dy
    torch.testing.assert_close(spots.pixel_flux.sum(), input_flux)

    with pytest.raises(ValueError, match="spot_oversampling"):
        ShackHartmannLensletArray(
            grid, pitch=0.4, focal_length=2.0, spot_oversampling=0
        )


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


def test_centroids_recover_known_xy_tilt_sign_and_scale():
    wavelength = 500e-9
    grid = Grid(24, 16, 0.1, 0.1, dtype=torch.float64)
    field = monochromatic_field(grid, wavelength)
    sensor = ShackHartmannLensletArray(grid, pitch=0.4, focal_length=2.0)
    angle_x = wavelength / sensor.pitch
    angle_y = -wavelength / sensor.pitch
    x, y = grid.meshgrid()
    tilted = Field(
        field.complex_amplitude
        * torch.exp(2j * math.pi * (angle_x * x + angle_y * y)[None] / wavelength),
        grid,
        field.spectrum,
    )

    measurement = ShackHartmannSlopeEstimator(
        focal_length=sensor.focal_length
    ).measure(sensor.propagate(tilted))
    expected = torch.tensor([angle_x, angle_y], dtype=grid.dtype)
    torch.testing.assert_close(
        measurement.slopes,
        expected.expand_as(measurement.slopes),
        rtol=1e-12,
        atol=1e-15,
    )
    assert measurement.slope_vector.shape == (2 * 4 * 6,)
    torch.testing.assert_close(
        measurement.slope_vector[:24], expected[0].expand(24)
    )
    torch.testing.assert_close(
        measurement.slope_vector[24:], expected[1].expand(24)
    )


def test_reference_centroids_are_subtracted_and_invalid_lenslets_are_omitted():
    grid = Grid(8, 8, 0.1, 0.1, dtype=torch.float64)
    valid = torch.tensor([[True, False], [True, True]])
    sensor = ShackHartmannLensletArray(
        grid, pitch=0.4, focal_length=2.0, valid_subapertures=valid
    )
    image = sensor.propagate(monochromatic_field(grid))
    reference = torch.zeros((2, 2, 2), dtype=grid.dtype)
    reference[..., 0] = float(image.pixel_scale_x[0])

    measurement = ShackHartmannSlopeEstimator(
        focal_length=2.0, reference_centroids=reference
    ).measure(image)
    torch.testing.assert_close(
        measurement.slopes[..., 0][valid],
        torch.full((3,), -float(image.pixel_scale_x[0]) / 2, dtype=grid.dtype),
    )
    assert torch.equal(
        measurement.slopes[~valid], torch.zeros((1, 2), dtype=grid.dtype)
    )
    assert measurement.slope_vector.shape == (6,)


def test_polychromatic_centroid_uses_each_channel_physical_sampling():
    scales = torch.tensor([1.0e-6, 2.0e-6, 3.0e-6], dtype=torch.float64)
    amplitude = torch.zeros((3, 1, 1, 3, 3), dtype=torch.complex128)
    # Equal photon flux in the +1 x pixel of every channel.
    amplitude[:, 0, 0, 1, 2] = scales.reciprocal()
    image = ShackHartmannImage(
        complex_amplitude=amplitude,
        wavelengths=torch.tensor([500e-9, 600e-9, 700e-9]),
        pixel_scale_x=scales,
        pixel_scale_y=scales,
        valid_subapertures=torch.ones((1, 1), dtype=torch.bool),
    )

    measurement = ShackHartmannSlopeEstimator(focal_length=2.0).measure(image)
    torch.testing.assert_close(
        measurement.slopes[..., 0], torch.tensor([[1.0e-6]], dtype=torch.float64)
    )


def test_centroid_window_threshold_and_validation():
    grid = Grid(8, 8, 0.1, 0.1)
    sensor = ShackHartmannLensletArray(grid, pitch=0.4, focal_length=1.0)
    image = sensor.propagate(monochromatic_field(grid))
    measurement = ShackHartmannSlopeEstimator(
        focal_length=1.0, window_radius=(1, 1), threshold=0.5
    ).measure(image)
    assert torch.equal(measurement.centroids, torch.zeros_like(measurement.centroids))

    with pytest.raises(ValueError, match="window_radius"):
        ShackHartmannSlopeEstimator(focal_length=1.0, window_radius=-1)
    with pytest.raises(ValueError, match="threshold"):
        ShackHartmannSlopeEstimator(focal_length=1.0, threshold=1.0)
    with pytest.raises(ValueError, match="reference_centroids"):
        ShackHartmannSlopeEstimator(focal_length=1.0).measure(
            image, reference_centroids=torch.zeros(2, 2)
        )


def test_partial_subaperture_illumination_is_measured_masked_and_weighted():
    grid = Grid(8, 8, 0.1, 0.1, dtype=torch.float64)
    pupil = torch.ones((8, 8), dtype=grid.dtype)
    pupil[:2, :4] = 0
    pupil[:3, 4:] = 0
    sensor = ShackHartmannLensletArray(
        grid,
        pitch=0.4,
        focal_length=1.0,
        pupil_transmission=pupil,
        minimum_illumination=0.4,
    )

    expected = torch.tensor([[0.5, 0.25], [1.0, 1.0]], dtype=grid.dtype)
    torch.testing.assert_close(sensor.illumination_fractions, expected)
    assert torch.equal(
        sensor.valid_subapertures,
        torch.tensor([[True, False], [True, True]]),
    )
    measurement = ShackHartmannSlopeEstimator(focal_length=1.0).measure(
        sensor.propagate(monochromatic_field(grid))
    )
    torch.testing.assert_close(measurement.weights, expected)
    assert measurement.slope_vector.shape == (6,)

    obscured_grid = Grid(12, 12, 0.1, 0.1, dtype=torch.float64)
    central_obscuration = torch.ones((12, 12), dtype=obscured_grid.dtype)
    central_obscuration[4:8, 4:8] = 0
    obscured = ShackHartmannLensletArray(
        obscured_grid,
        pitch=0.4,
        focal_length=1.0,
        pupil_transmission=central_obscuration,
        minimum_illumination=0.5,
    )
    assert obscured.illumination_fractions[1, 1] == 0
    assert not obscured.valid_subapertures[1, 1]

    with pytest.raises(ValueError, match="minimum_illumination"):
        ShackHartmannLensletArray(
            grid, pitch=0.4, focal_length=1.0, minimum_illumination=1.1
        )


def test_detector_integrates_signal_dark_current_and_quantum_efficiency():
    grid = Grid(8, 8, 0.1, 0.1, dtype=torch.float64)
    sensor = ShackHartmannLensletArray(grid, pitch=0.4, focal_length=1.0)
    image = sensor.propagate(monochromatic_field(grid))
    detector = ShackHartmannDetector(
        exposure_time=2.0,
        pixel_scale_x=float(image.pixel_scale_x[0]),
        quantum_efficiency=0.5,
        photon_noise=False,
        dark_current=3.0,
        random_seed=5,
        device=grid.device,
    )
    frame = detector.expose(image)
    torch.testing.assert_close(
        frame.expected_electrons,
        image.pixel_flux.sum(dim=0) + 6.0,
    )


def test_detector_noise_is_reproducible_and_low_flux_is_invalid():
    grid = Grid(8, 8, 0.1, 0.1, dtype=torch.float64)
    sensor = ShackHartmannLensletArray(grid, pitch=0.4, focal_length=1.0)
    image = sensor.propagate(monochromatic_field(grid))
    kwargs = dict(
        exposure_time=1.0,
        pixel_scale_x=float(image.pixel_scale_x[0]),
        photon_noise=True,
        dark_current=0.2,
        read_noise=1.0,
        minimum_electrons=1e30,
        random_seed=12,
        device=grid.device,
    )
    first = ShackHartmannDetector(**kwargs).expose(image)
    second = ShackHartmannDetector(**kwargs).expose(image)
    torch.testing.assert_close(first.electrons, second.electrons)
    assert not first.valid_subapertures.any()

    valid_frame = ShackHartmannDetector(
        **{**kwargs, "minimum_electrons": 0.0}
    ).expose(image)
    noisy_measurement = ShackHartmannSlopeEstimator(focal_length=1.0).measure(
        valid_frame
    )
    assert noisy_measurement.valid_subapertures.all()
    assert torch.isfinite(noisy_measurement.slopes).all()


def test_saturation_is_clipped_reported_and_rejected_by_centroiding():
    grid = Grid(8, 8, 0.1, 0.1, dtype=torch.float64)
    sensor = ShackHartmannLensletArray(grid, pitch=0.4, focal_length=1.0)
    image = sensor.propagate(monochromatic_field(grid))
    frame = ShackHartmannDetector(
        exposure_time=1.0,
        pixel_scale_x=float(image.pixel_scale_x[0]),
        photon_noise=False,
        full_well=1e-12,
        device=grid.device,
    ).expose(image)

    assert frame.saturated_subapertures.all()
    assert torch.all(frame.electrons <= 1e-12)
    measurement = ShackHartmannSlopeEstimator(focal_length=1.0).measure(frame)
    assert not measurement.valid_subapertures.any()
    assert measurement.saturated_subapertures.all()
    assert measurement.slope_vector.numel() == 0


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
