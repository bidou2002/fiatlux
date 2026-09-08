import pytest
import torch

from fiatlux.core.grid import Grid
from fiatlux.optics.atmosphere import NCPAModel


def make_grid(nx=48, ny=32, dx=0.08, dy=0.11):
    return Grid(nx=nx, ny=ny, dx=dx, dy=dy)


def test_opd_psd_has_requested_integrated_variance():
    model = NCPAModel(make_grid(), opd_rms=80e-9, seed=1)

    integrated_variance = model.opd_psd.sum() * model.frequency_bin_area

    assert model.opd_psd.shape == (model.grid.ny, model.grid.nx)
    assert model.opd_psd.dtype == torch.float32
    assert model.opd_psd[0, 0] == 0
    torch.testing.assert_close(
        integrated_variance,
        torch.as_tensor((80e-9) ** 2, dtype=model.dtype),
        rtol=2e-6,
        atol=0,
    )


def test_opd_psd_follows_requested_power_law():
    model = NCPAModel(
        make_grid(dx=0.1, dy=0.1),
        opd_rms=100e-9,
        spectral_index=2.5,
    )
    fx, fy = model.frequency_grid()
    frequency = torch.sqrt(fx.square() + fy.square())
    first = (1, 2)
    second = (2, 3)
    expected_ratio = (frequency[first] / frequency[second]).pow(-2.5)

    torch.testing.assert_close(
        model.opd_psd[first] / model.opd_psd[second], expected_ratio
    )


def test_opd_screen_is_real_rectangular_and_reproducible():
    kwargs = dict(
        grid=make_grid(),
        opd_rms=120e-9,
        spectral_index=3.0,
        outer_scale=2.0,
        seed=42,
    )
    opd_a = NCPAModel(**kwargs).sample_opd()
    opd_b = NCPAModel(**kwargs).sample_opd()

    assert opd_a.shape == (kwargs["grid"].ny, kwargs["grid"].nx)
    assert not opd_a.is_complex()
    assert torch.isfinite(opd_a).all()
    assert opd_a.mean().abs() < 1e-12
    torch.testing.assert_close(opd_a, opd_b)


def test_ensemble_variance_follows_psd_without_per_screen_rescaling():
    model = NCPAModel(
        make_grid(nx=32, ny=24),
        opd_rms=100e-9,
        spectral_index=3.0,
        seed=12,
    )
    variances = torch.stack(
        [model.sample_opd().square().mean() for _ in range(256)]
    )

    torch.testing.assert_close(
        variances.mean(),
        torch.as_tensor(model.opd_rms**2, dtype=model.dtype),
        rtol=0.08,
        atol=0,
    )
    assert variances.std() > 0


def test_external_generator_controls_sampling():
    model = NCPAModel(make_grid(), opd_rms=100e-9, seed=1)
    generator_a = torch.Generator().manual_seed(9)
    generator_b = torch.Generator().manual_seed(9)

    torch.testing.assert_close(
        model.sample_opd(generator=generator_a),
        model.sample_opd(generator=generator_b),
    )


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"opd_rms": 0.0},
        {"opd_rms": -1e-9},
        {"opd_rms": 1e-9, "spectral_index": 0.0},
        {"opd_rms": 1e-9, "outer_scale": 0.0},
    ],
)
def test_invalid_ncpa_parameters_are_rejected(bad_kwargs):
    with pytest.raises(ValueError):
        NCPAModel(make_grid(), **bad_kwargs)
