import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fiatlux import Grid, TiptopSamplingError, tiptop_psd, load_harmoni_scao_config
from fiatlux.optics.tiptop import configure_tiptop_sampling, verify_tiptop_frequency_grid, _sampling


def config():
    return {"telescope": {"TelescopeDiameter": 39},
            "sources_science": {"Wavelength": [1.075e-6, 1.1e-6, 1.125e-6]},
            "sensor_science": {"PixelScale": 4, "FieldOfView": 640}}


def test_config_uses_existing_padded_grid_and_all_wavelengths():
    grid = Grid(400, 400, .25, .25, dtype=torch.float64)
    before = copy.deepcopy(grid.__dict__)
    cfg = config()
    new = configure_tiptop_sampling(grid, cfg)
    df, kref, _, _ = _sampling(new['sensor_science']['PixelScale'],
        np.array(cfg['sources_science']['Wavelength']), 39)
    assert df == pytest.approx(.01, rel=1e-12)
    assert kref > 1  # polychromatic expansion must not be assumed to be 1
    assert new['sensor_science']['FieldOfView'] * kref >= grid.nx
    assert cfg == config()
    assert grid.__dict__ == before


def test_unpadded_grid_fails_config_only_without_mutation():
    grid = Grid(400, 400, 39/400, 39/400)
    with pytest.raises(TiptopSamplingError, match='configuration limit'):
        configure_tiptop_sampling(grid, config())
    assert grid.dx == 39/400 and grid.nx == 400


@pytest.mark.parametrize('n,size', [(10, 8), (11, 8), (10, 7), (8, 8)])
def test_coordinate_validation_and_exact_crop(n, size):
    grid = Grid(size, size, 1/(size*.01), 1/(size*.01))
    f = (np.arange(n) - n//2) * .01 + 1e-10
    x, y = np.meshgrid(f, f, indexing='ij')
    freq = SimpleNamespace(nOtf=n, PSDstep=.01, kx_=x, ky_=y)
    crop = verify_tiptop_frequency_grid(grid, freq)
    np.testing.assert_allclose(x[crop,crop][:,0], np.fft.fftshift(np.fft.fftfreq(size,grid.dx)), atol=2e-10)
    freq.PSDstep = .02
    with pytest.raises(TiptopSamplingError):
        verify_tiptop_frequency_grid(grid, freq)
    freq.PSDstep = .01
    freq.kx_ = x + .001
    with pytest.raises(TiptopSamplingError):
        verify_tiptop_frequency_grid(grid, freq)


@pytest.mark.parametrize('explicit,extent,size', [(False, 100., 321), (True, 38.542, 320)])
def test_actual_p3_grid_and_variance(explicit, extent, size):
    pytest.importorskip('p3')
    pytest.importorskip('tiptop')
    # N large enough to contain AO support, independently of physical pupil.
    grid = Grid(size, size, extent/size, extent/size, dtype=torch.float64)
    original = copy.deepcopy(grid.__dict__)
    cfg = load_harmoni_scao_config()
    result = tiptop_psd(grid, config=cfg, explicit_sampling=explicit,
        overrides={'telescope': {'TelescopeDiameter': 38.542, 'ZenithAngle': 52.4},
                   'sources_science': {'Wavelength': [1.075e-6, 1.1e-6, 1.125e-6]}})
    assert result.grid is grid
    assert grid.__dict__ == original
    assert result.diagnostics['PSDstep'] == pytest.approx(1/extent, rel=1e-12)
    assert result.diagnostics['component_sum_matches_total']
    model = result.atmosphere_model(symmetrize=True, seed=78)
    assert model.grid is grid
    variance = result.opd_psd[0].sum() * result.frequency_step**2
    torch.testing.assert_close(variance, result.power_nm2[0].sum()*1e-18, atol=1e-27, rtol=1e-12)
    screens = model.sample_opd_many(32)
    assert abs(float(screens.square().mean()/variance)-1) < .03


def test_import_rejects_other_grid():
    from fiatlux import TabulatedAtmosphereModel
    grid = Grid(8,8,.1,.1)
    with pytest.raises(ValueError, match='frequency increment'):
        TabulatedAtmosphereModel(grid, torch.ones(8,8), frequency_step=.1)


def test_explicit_camera_override_is_verified_not_silently_replaced():
    pytest.importorskip('p3')
    pytest.importorskip('tiptop')
    grid = Grid(320, 320, 100/320, 100/320, dtype=torch.float64)
    with pytest.raises(TiptopSamplingError, match='No interpolation'):
        tiptop_psd(grid, overrides={'sensor_science': {'PixelScale': 4.0}})
    assert grid.nx == 320 and grid.dx == 100/320
