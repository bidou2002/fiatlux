import math

import pytest
import torch

from fiatlux.core.grid import Grid
from fiatlux.optics.atmosphere import KolmogorovAtmosphereModel


def make_grid(nx=64, ny=48, dx=0.1, dy=0.12):
    return Grid(nx=nx, ny=ny, dx=dx, dy=dy)


def test_kolmogorov_psd_matches_analytical_law():
    model = KolmogorovAtmosphereModel(
        make_grid(), r0=0.2, reference_wavelength=500e-9, seed=1
    )
    fx, fy = model.frequency_grid()
    f2 = fx.square() + fy.square()
    expected = 0.023 * model.r0 ** (-5.0 / 3.0) * f2[1, 2] ** (-11.0 / 6.0)

    assert model.phase_psd.shape == (model.grid.ny, model.grid.nx)
    torch.testing.assert_close(model.phase_psd[1, 2], expected)
    assert model.phase_psd[0, 0] == 0


def test_von_karman_psd_has_finite_low_frequency_power():
    model = KolmogorovAtmosphereModel(
        make_grid(),
        r0=0.2,
        reference_wavelength=500e-9,
        outer_scale=25.0,
    )
    expected_dc = 0.023 * model.r0 ** (-5.0 / 3.0) * (1 / 25.0**2) ** (
        -11.0 / 6.0
    )
    torch.testing.assert_close(
        model.phase_psd[0, 0],
        torch.as_tensor(expected_dc, dtype=model.dtype),
    )


def test_screen_is_real_rectangular_and_reproducible():
    grid = make_grid()
    kwargs = dict(
        grid=grid,
        r0=0.2,
        reference_wavelength=500e-9,
        outer_scale=25.0,
        seed=42,
    )
    model_a = KolmogorovAtmosphereModel(**kwargs)
    model_b = KolmogorovAtmosphereModel(**kwargs)

    phase_a = model_a.sample_phase()
    phase_b = model_b.sample_phase()

    assert phase_a.shape == (grid.ny, grid.nx)
    assert not phase_a.is_complex()
    assert torch.isfinite(phase_a).all()
    assert phase_a.mean().abs() < 1e-5
    torch.testing.assert_close(phase_a, phase_b)



def test_default_dtype_matches_fiatlux_complex64_pipeline():
    model = KolmogorovAtmosphereModel(make_grid(), r0=0.2, seed=3)

    assert model.phase_psd.dtype == torch.float32
    assert model.sample_phase().dtype == torch.float32
    assert model.sample_opd().dtype == torch.float32

def test_opd_is_phase_converted_at_reference_wavelength():
    kwargs = dict(
        grid=make_grid(), r0=0.2, reference_wavelength=700e-9, seed=7
    )
    phase = KolmogorovAtmosphereModel(**kwargs).sample_phase()
    opd = KolmogorovAtmosphereModel(**kwargs).sample_opd()
    torch.testing.assert_close(opd, phase * 700e-9 / (2 * math.pi))


def test_screen_variance_follows_integrated_psd_without_renormalization():
    model = KolmogorovAtmosphereModel(
        make_grid(nx=32, ny=24),
        r0=0.2,
        reference_wavelength=500e-9,
        outer_scale=25.0,
        seed=12,
    )
    psd_without_piston = model.phase_psd.clone()
    psd_without_piston[0, 0] = 0.0
    expected_variance = psd_without_piston.sum() * model.frequency_bin_area

    measured_variance = torch.stack(
        [model.sample_phase().square().mean() for _ in range(256)]
    ).mean()

    torch.testing.assert_close(measured_variance, expected_variance, rtol=0.08, atol=0)


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"r0": 0.0},
        {"r0": -0.1},
        {"r0": 0.2, "outer_scale": 0.0},
        {"r0": 0.2, "reference_wavelength": 0.0},
    ],
)
def test_invalid_physical_parameters_are_rejected(bad_kwargs):
    with pytest.raises(ValueError):
        KolmogorovAtmosphereModel(make_grid(), **bad_kwargs)
