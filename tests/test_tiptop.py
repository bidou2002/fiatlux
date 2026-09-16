import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fiatlux import Grid, TiptopSamplingError, tiptop_psd, load_harmoni_scao_config
from fiatlux.optics.tiptop import configure_tiptop_sampling, verify_tiptop_frequency_grid, exact_frequency_indices, _sampling


def config():
    return {'telescope': {'TelescopeDiameter': 39},
            'sources_science': {'Wavelength': [1.075e-6,1.1e-6,1.125e-6]},
            'sensor_science': {'PixelScale': 4, 'FieldOfView': 640},
            'DM': {'DmPitchs': [.38]}}


@pytest.mark.parametrize('extent', [39.,100.])
def test_config_uses_existing_grid(extent):
    grid=Grid(400,400,extent/400,extent/400,dtype=torch.float64)
    before=copy.deepcopy(grid.__dict__);cfg=config()
    new=configure_tiptop_sampling(grid,cfg)
    df,kref,_,_=_sampling(new['sensor_science']['PixelScale'],np.array(cfg['sources_science']['Wavelength']),39)
    q=1/extent/df
    assert q == pytest.approx(round(q),rel=1e-11)
    assert new['sensor_science']['FieldOfView']*kref >= 400*round(q)
    assert cfg==config() and grid.__dict__==before


@pytest.mark.parametrize('q', [1,2,3,4])
@pytest.mark.parametrize('size', [7,8])
@pytest.mark.parametrize('extra', [0,1,4])
def test_exact_indices_even_odd_stride_and_crop(q,size,extra):
    grid=Grid(size,size,1/(size*.01),1/(size*.01))
    n=q*size+extra
    source=(np.arange(n)-n//2)*.01/q+1e-10
    x,y=np.meshgrid(source,source,indexing='ij')
    freq=SimpleNamespace(nOtf=n,PSDstep=.01/q,kx_=x,ky_=y)
    ix,iy=verify_tiptop_frequency_grid(grid,freq)
    target=np.fft.fftshift(np.fft.fftfreq(size,grid.dx))
    np.testing.assert_allclose(source[ix],target,atol=2e-10)
    assert ix[size//2]==n//2
    assert np.all(np.diff(ix)==q)
    # Distinct values prove selection, not averaging adjacent cells.
    values=np.arange(n*n).reshape(n,n)
    np.testing.assert_array_equal(values[np.ix_(ix,iy)],values[ix[:,None],iy[None,:]])


@pytest.mark.parametrize('step,n,offset', [(.007,40,0),(.005,8,0),(.005,24,.001)])
def test_incompatible_coordinates_fail(step,n,offset):
    grid=Grid(8,8,12.5,12.5)
    source=(np.arange(n)-n//2)*step+offset
    x,y=np.meshgrid(source,source,indexing='ij')
    with pytest.raises(TiptopSamplingError):
        verify_tiptop_frequency_grid(grid,SimpleNamespace(nOtf=n,PSDstep=step,kx_=x,ky_=y))


@pytest.mark.parametrize('size,q,poly', [(160,2,False),(161,3,False),(160,4,False),(160,None,True)])
def test_actual_p3_pupil_only_exact_subgrid_and_density(size,q,poly):
    pytest.importorskip('p3');pytest.importorskip('tiptop')
    D=38.542
    grid=Grid(size,size,D/size,D/size,dtype=torch.float64)
    original=copy.deepcopy(grid.__dict__)
    wavelengths=[1.075e-6,1.1e-6,1.125e-6] if poly else [1.1e-6]
    result=tiptop_psd(grid,sampling_ratio=q,overrides={
        'telescope':{'TelescopeDiameter':D,'ZenithAngle':52.4},
        'sources_science':{'Wavelength':wavelengths}})
    assert result.grid is grid and grid.__dict__==original
    assert result.diagnostics['PSDstep'] < 1/D
    assert result.power_nm2.shape == (1,size,size)
    target=np.fft.fftshift(np.fft.fftfreq(size,grid.dx))
    np.testing.assert_allclose(result.p3_frequency_x[result.indices_x],target,atol=2e-10)
    assert result.diagnostics['component_sum_matches_total']
    # P3's return has dk² included: divide it out BEFORE using FIATLUX df².
    dk=result.diagnostics['p3_normalization_dk']
    density=torch.fft.ifftshift(result.selected_p3_power_nm2,dim=(-2,-1))*1e-18/dk**2
    torch.testing.assert_close(result.opd_psd,density,atol=1e-30,rtol=1e-12)
    variance=density[0].sum()/D**2
    model=result.atmosphere_model(symmetrize=True,seed=78)
    assert model.grid is grid
    screens=model.sample_opd_many(64)
    assert abs(float(screens.square().mean()/variance)-1) < .04


def test_extraction_never_calls_interpolation(monkeypatch):
    # Guards the complete extraction path. P3 may independently resize its
    # static pupil/OTF inputs; that is not residual-PSD interpolation.
    def forbidden(*args,**kwargs):
        raise AssertionError('Interpolation called during PSD extraction')
    monkeypatch.setattr(np,'interp',forbidden)
    scipy=pytest.importorskip('scipy.interpolate')
    for name in ['interp1d','interpn','RegularGridInterpolator','griddata','RectBivariateSpline']:
        monkeypatch.setattr(scipy,name,forbidden)
    grid=Grid(8,8,.1,.1)
    f=np.fft.fftshift(np.fft.fftfreq(32,.1))
    x,y=np.meshgrid(f,f,indexing='ij')
    ix,iy=verify_tiptop_frequency_grid(grid,SimpleNamespace(nOtf=32,PSDstep=1/3.2,kx_=x,ky_=y))
    data=np.arange(1024).reshape(32,32)
    assert data[np.ix_(ix,iy)].shape == grid.shape


def test_explicit_incompatible_override_is_not_silently_replaced():
    pytest.importorskip('p3');pytest.importorskip('tiptop')
    grid=Grid(160,160,39/160,39/160,dtype=torch.float64)
    with pytest.raises(TiptopSamplingError):
        tiptop_psd(grid,pixel_scale=4.0)
