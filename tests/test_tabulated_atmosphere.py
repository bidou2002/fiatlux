import math

import pytest
import torch

from fiatlux import Grid, TabulatedAtmosphereModel

def make_model(power, *, frequency_step, **kwargs):
    p = torch.as_tensor(power)
    ny, nx = p.shape if p.ndim == 2 else (2, 2)
    step = frequency_step if math.isfinite(frequency_step) and frequency_step > 0 else .1
    grid = Grid(nx, ny, 1/(nx*step), 1/(ny*step), dtype=torch.float64)
    return TabulatedAtmosphereModel(grid, power, frequency_step=frequency_step, **kwargs)



def test_native_grid_and_power_conservation():
    power = torch.arange(48, dtype=torch.float64).reshape(6, 8)
    model = make_model(power, frequency_step=.2, symmetrize=True)
    assert model.grid.dx == 1 / (8 * .2)
    assert model.grid.dy == 1 / (6 * .2)
    recovered = model.psd.sum() * model.frequency_bin_area * (500e-9 / (2 * math.pi)) ** 2
    torch.testing.assert_close(recovered, power.sum() * 1e-18, atol=1e-25, rtol=1e-12)


def test_ensemble_variance_and_reference_wavelength_invariance():
    power = torch.ones((12, 16), dtype=torch.float64)
    power[6, 8] = 0
    a = make_model(power, frequency_step=.1, seed=4)
    b = make_model(power, frequency_step=.1, seed=4, reference_wavelength=2.2e-6)
    x = a.sample_opd_many(1024)
    torch.testing.assert_close(x, b.sample_opd_many(1024), atol=1e-22, rtol=1e-12)
    assert abs(float(x.square().mean()) / (191e-18) - 1) < .015
    assert float(x.mean((-2, -1)).abs().max()) < 1e-22


def test_asymmetry_requires_explicit_consent():
    p = torch.zeros((5, 7), dtype=torch.float64)
    p[1, 2] = 2
    with pytest.raises(ValueError, match="inversion symmetry"):
        make_model(p, frequency_step=.1)
    m = make_model(p, frequency_step=.1, symmetrize=True)
    assert m.power_nm2.sum() == 2
    assert m.inversion_asymmetry == 2


@pytest.mark.parametrize('power,step', [([[1, -1], [1, 1]], .1),
    ([[1, float('nan')], [1, 1]], .1), ([[1, 1], [1, 1]], 0),
    ([[1, 1], [1, 1]], float('inf')), ([1, 2], .1)])
def test_invalid_input(power, step):
    with pytest.raises(ValueError):
        make_model(power, frequency_step=step)
