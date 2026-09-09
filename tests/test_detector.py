from unittest.mock import patch

import pytest
import torch

from fiatlux.core.field import Field
from fiatlux.core.grid import Grid
from fiatlux.core.spectrum import Band, Spectrum
from fiatlux.optics.detector import Detector


def make_detector(**kwargs) -> Detector:
    return Detector(Grid(nx=3, ny=2, dx=0.1, dy=0.2), **kwargs)


def make_field(grid: Grid, intensity_per_channel: torch.Tensor) -> Field:
    n_wavelengths = len(intensity_per_channel)
    spectrum = Spectrum(
        magnitude=0,
        band=Band(central_wavelength=1e-6, delta_wavelength=0.1e-6, f0=1),
        samples=n_wavelengths,
    )
    amplitude = intensity_per_channel.sqrt()[:, None, None].expand(
        -1, grid.ny, grid.nx
    )
    return Field(amplitude, grid, spectrum)


def test_16_bit_adc_preserves_the_full_unsigned_range_without_overflow():
    detector = make_detector(bitdepth=16, digitize=True)
    electrons = torch.tensor([0.0, 32767.0, 32768.0, 65535.0, 65536.0])

    adus = detector.electrons_to_adus(electrons)

    assert adus.dtype == torch.int64
    torch.testing.assert_close(
        adus,
        torch.tensor([0, 32767, 32768, 65535, 65535], dtype=torch.int64),
    )


def test_adc_clips_negative_values_before_integer_conversion():
    detector = make_detector(bitdepth=16, digitize=True)

    adus = detector.electrons_to_adus(torch.tensor([-10.0, -0.1, 0.0]))

    torch.testing.assert_close(adus, torch.zeros(3, dtype=torch.int64))


def test_adc_applies_sensitivity_and_offset_before_clipping():
    detector = make_detector(
        bitdepth=8,
        digitize=True,
        sensitivity=2.0,
        offset=10,
    )

    adus = detector.electrons_to_adus(torch.tensor([1.9, 200.0]))

    torch.testing.assert_close(adus, torch.tensor([13, 255], dtype=torch.int64))


def test_lower_bitdepth_is_not_rescaled_to_a_16_bit_container():
    detector = make_detector(bitdepth=12, digitize=True)

    adus = detector.electrons_to_adus(torch.tensor([4095.0]))

    assert adus.item() == 4095


def test_digitize_controls_whether_adc_conversion_is_applied():
    photons = torch.tensor([1.5, 2.5])

    analog = make_detector(digitize=False).add_noise(photons)
    digital = make_detector(digitize=True).add_noise(photons)

    assert analog.is_floating_point()
    assert digital.dtype == torch.int64
    torch.testing.assert_close(digital, torch.tensor([1, 2], dtype=torch.int64))


@pytest.mark.parametrize("bitdepth", [0, -1, 1.5, True])
def test_invalid_bitdepth_is_rejected(bitdepth):
    with pytest.raises(ValueError, match="bitdepth must be a positive integer"):
        make_detector(bitdepth=bitdepth)


def test_legacy_binarize_option_is_no_longer_accepted():
    with pytest.raises(TypeError):
        make_detector(binarize=True)


def test_acquire_integrates_spectral_rate_over_pixel_area_and_exposure_time():
    detector = make_detector(exposure_time=2.0)
    field = make_field(detector.grid, torch.tensor([10.0, 15.0]))

    image = detector.acquire(field)

    expected = torch.full((2, 3), 25.0 * 0.1 * 0.2 * 2.0)
    torch.testing.assert_close(image, expected)
    assert image is detector.image_buffer


def test_acquire_rejects_an_incompatible_grid_with_expected_and_actual_details():
    detector = make_detector()
    field = make_field(
        Grid(nx=3, ny=2, dx=0.2, dy=0.2), torch.tensor([10.0])
    )

    with pytest.raises(ValueError, match=r"Detector.*expected.*got"):
        detector.acquire(field)


def test_doubling_exposure_time_doubles_source_counts():
    short = make_detector(exposure_time=1.0)
    long = make_detector(exposure_time=2.0)
    field = make_field(short.grid, torch.tensor([10.0]))

    torch.testing.assert_close(long.acquire(field), 2 * short.acquire(field))


def test_quantum_efficiency_scales_expected_electrons_once():
    ideal = make_detector(exposure_time=1.0, quantum_efficiency=1.0)
    half_qe = make_detector(exposure_time=1.0, quantum_efficiency=0.5)
    field = make_field(ideal.grid, torch.tensor([100.0]))

    torch.testing.assert_close(half_qe.acquire(field), 0.5 * ideal.acquire(field))


def test_zero_field_produces_zero_source_counts_without_detector_noise():
    detector = make_detector(exposure_time=10.0)
    field = make_field(detector.grid, torch.tensor([0.0]))

    torch.testing.assert_close(detector.acquire(field), torch.zeros((2, 3)))


def test_photon_noise_receives_integrated_detected_electron_counts():
    detector = make_detector(
        exposure_time=2.0,
        quantum_efficiency=0.5,
        photon_noise=True,
    )
    field = make_field(detector.grid, torch.tensor([100.0]))
    poisson_inputs = []

    def deterministic_poisson(values, generator=None):
        poisson_inputs.append(values.clone())
        return values

    with patch("torch.poisson", side_effect=deterministic_poisson):
        detector.acquire(field)

    expected_photoelectrons = torch.full((2, 3), 100.0 * 0.1 * 0.2 * 2.0 * 0.5)
    torch.testing.assert_close(poisson_inputs[0], expected_photoelectrons)


def test_dark_current_is_a_rate_per_pixel_and_read_noise_uses_variance():
    detector = make_detector(
        exposure_time=2.0,
        dark_current=3.0,
        readout_noise_variance=4.0,
    )
    electrons = torch.zeros((2, 3))

    with patch("torch.poisson", side_effect=lambda values, generator=None: values):
        darkened = detector.add_dark_noise(electrons)
    torch.testing.assert_close(darkened, torch.full((2, 3), 6.0))

    with patch("torch.normal", return_value=torch.zeros((2, 3))) as normal:
        detector.add_readout_noise(electrons)
    assert normal.call_args.kwargs["std"] == 2.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"exposure_time": -1},
        {"quantum_efficiency": -0.1},
        {"quantum_efficiency": 1.1},
        {"dark_current": -1},
        {"readout_noise_variance": -1},
    ],
)
def test_invalid_physical_detector_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_detector(**kwargs)
