import math

import pytest
import torch

from fiatlux.optics.turbulence_validation import (
    estimate_translation,
    spatial_periodogram,
    spatial_structure_function,
    temporal_autocorrelation,
)


def test_spatial_periodogram_has_physical_frequency_and_parseval_normalization():
    n = 32
    dx = 0.2
    x = torch.arange(n, dtype=torch.float64)
    screen = torch.cos(2 * math.pi * 4 * x / n).repeat(n, 1)

    fx, fy, psd = spatial_periodogram(screen, dx=dx)
    peak = torch.nonzero(psd == psd.max(), as_tuple=False)[0]
    assert abs(float(fx[tuple(peak)])) == pytest.approx(4 / (n * dx))
    assert float(fy[tuple(peak)]) == pytest.approx(0.0)

    frequency_bin_area = 1.0 / ((n * dx) ** 2)
    recovered_variance = psd.sum() * frequency_bin_area
    assert float(recovered_variance) == pytest.approx(float(screen.var(correction=0)))


def test_structure_function_is_distinct_from_the_spatial_psd():
    n = 20
    spacing = 0.1
    slope = 3.0
    ramp = (slope * spacing * torch.arange(n, dtype=torch.float64)).repeat(n, 1)

    separation, structure = spatial_structure_function(
        ramp, spacing=spacing, max_lag=5, axis="x"
    )
    torch.testing.assert_close(structure, (slope * separation).square())


def test_temporal_autocorrelation_reports_lag_independently_of_spatial_statistics():
    pattern = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
    coefficients = torch.tensor([1.0, 0.0, -1.0, 0.0], dtype=torch.float64)
    sequence = coefficients[:, None, None] * pattern

    correlation = temporal_autocorrelation(sequence, max_lag=2)
    torch.testing.assert_close(
        correlation, torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)
    )


def test_wind_displacement_uses_two_dimensional_subpixel_correlation():
    n = 64
    dx = 0.1
    y = torch.arange(n, dtype=torch.float64) - n / 2
    x = torch.arange(n, dtype=torch.float64) - n / 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    reference = torch.exp(-(xx.square() + yy.square()) / (2 * 5.0**2))
    expected_y_pixels = 2.4
    expected_x_pixels = -1.7
    fy = torch.fft.fftfreq(n, dtype=torch.float64)[:, None]
    fx = torch.fft.fftfreq(n, dtype=torch.float64)[None, :]
    phase_ramp = torch.exp(
        -2j * math.pi * (fy * expected_y_pixels + fx * expected_x_pixels)
    )
    shifted = torch.fft.ifft2(torch.fft.fft2(reference) * phase_ramp).real

    shift_y, shift_x = estimate_translation(reference, shifted, dx=dx)
    assert shift_y == pytest.approx(expected_y_pixels * dx, abs=0.02 * dx)
    assert shift_x == pytest.approx(expected_x_pixels * dx, abs=0.02 * dx)
