import pytest
import torch

from fiatlux.core.grid import Grid
from fiatlux.optics.detector import Detector


def make_detector(**kwargs) -> Detector:
    return Detector(Grid(nx=3, ny=2, dx=0.1, dy=0.2), **kwargs)


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
