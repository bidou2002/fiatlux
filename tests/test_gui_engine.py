from copy import deepcopy
import threading
import time

import pytest
import torch

from fiatlux_app.engine import build_system, execute
from fiatlux_app.model import examples, loads, dumps, new_block
from fiatlux_app.server import BusyError, RunManager
from fiatlux.core.grid import Grid
from fiatlux.core.source import PlaneWave
from fiatlux.core.spectrum import Band, Spectrum
from fiatlux.optics.elements.mask import CircularAperture, Step, ZeldaMask, Atmosphere
from fiatlux.optics.elements.deformable_mirror import ActuatorGrid, DeformableMirror, ZernikeBasis
from fiatlux.optics.elements.segmented_aperture import ELTHarmoniPupil
from fiatlux.optics.atmosphere import KolmogorovAtmosphereModel
from fiatlux.optics.propagator import MFTPropagator, FFTPropagator
from fiatlux.optics.detector import Detector
from fiatlux.system.optical_system import SerialSystem


@pytest.mark.parametrize('name', list(examples()))
def test_application_matches_independently_constructed_example(name):
    config = examples()[name]
    grid = Grid(**config['grids']['pupil'], dtype=torch.float64)
    focal = Grid(**config['grids']['focal'], dtype=torch.float64)
    source = PlaneWave(Spectrum(0., Band(1.65e-6, 0., 368e6), 1, dtype=torch.float64))
    if name == '02_elt_dm':
        dm = DeformableMirror(grid, ActuatorGrid(6, 1, .5), grid, ZernikeBasis(grid, 6), stroke=1e-6)
        dm.commands = torch.tensor([40e-9, -20e-9, 60e-9, 0., 0., 0.], dtype=torch.float64)
        elements = [ELTHarmoniPupil(grid), dm, MFTPropagator(10., focal)]
    elif name == '03_zelda':
        elements = [CircularAperture(grid, .5), Step(grid, 82.5e-9), MFTPropagator(10., focal),
                    ZeldaMask(focal, 16.5e-6, 412.5e-9), MFTPropagator(10., grid)]
    else:
        elements = [CircularAperture(grid, .5), MFTPropagator(10., focal)]
    detector = Detector(grid if name == '03_zelda' else focal, exposure_time=.15,
                        photon_noise=False, readout_noise_variance=0, dark_current=0, random_seed=42)
    direct = SerialSystem(elements).run(source, detector)
    actual = execute(loads(dumps(config)))
    torch.testing.assert_close(actual.trace.final_field.complex_amplitude, direct.final_field.complex_amplitude,
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual.prepared.detector.image_buffer, detector.image_buffer, rtol=1e-12, atol=1e-12)
    assert len(actual.metadata()['steps']) == len(config['blocks'])
    image = actual.image(len(config['blocks'])-1)
    torch.testing.assert_close(torch.tensor(image['values'], dtype=torch.float64), detector.image_buffer)


def test_spectral_views_follow_core_flux_and_phase_conventions():
    config = examples()['01_circular']
    config['spectrum'].update(samples=3, delta_wavelength=.1e-6)
    actual = execute(config)
    field = actual.trace.final_field
    summed = actual.image(2, 'intensity', 'sum')
    torch.testing.assert_close(torch.tensor(summed['values'], dtype=torch.float64), field.intensity().sum(-3))
    phase = actual.image(2, 'phase', '1')
    torch.testing.assert_close(torch.tensor(phase['values'], dtype=torch.float64), field.phase()[1])
    with pytest.raises(ValueError, match='pas de phase polychromatique'):
        actual.image(2, 'phase', 'sum')


def test_dm_opd_is_optical_path_difference_and_units_only_scale_display():
    actual = execute(examples()['02_elt_dm'])
    metres = torch.tensor(actual.image(2, 'opd', opd_unit='m')['values'], dtype=torch.float64)
    nm = torch.tensor(actual.image(2, 'opd', opd_unit='nm')['values'], dtype=torch.float64)
    torch.testing.assert_close(metres, actual.prepared.system.elements[1].opd)
    torch.testing.assert_close(nm, metres*1e9)


def test_seeded_kolmogorov_is_reproducible_real_engine():
    config = examples()['01_circular']
    atmosphere = new_block('Kolmogorov', 'atmosphere')
    config['blocks'].insert(2, atmosphere)
    a, b = execute(config), execute(config)
    torch.testing.assert_close(a.trace.final_field.complex_amplitude, b.trace.final_field.complex_amplitude)
    grid = a.prepared.source_grid
    direct = Atmosphere(grid, KolmogorovAtmosphereModel(grid, .2, seed=42), recompute=False)
    torch.testing.assert_close(a.trace.steps[2].field_after.complex_amplitude,
                               direct.apply(a.trace.steps[1].field_after).complex_amplitude)


def test_fresnel_and_fft_use_public_propagator_contracts():
    config = examples()['01_circular']
    params = config['blocks'][2]['params']
    params.update(propagation='fresnel', focal_length=None, distance=0., output_grid='pupil')
    config['blocks'][-1]['params']['grid'] = 'pupil'
    actual = execute(config)
    torch.testing.assert_close(actual.trace.final_field.complex_amplitude, actual.trace.steps[1].field_after.complex_amplitude)
    config = examples()['01_circular']
    config['blocks'][2]['type'] = 'FFTPropagator'
    config['blocks'][2]['params'].pop('output_grid')
    dx = 1.65e-6*10
    config['grids']['focal'] = dict(nx=64, ny=64, dx=dx, dy=dx)
    actual = execute(config)
    expected = FFTPropagator(10.).apply(actual.trace.steps[1].field_after)
    torch.testing.assert_close(actual.trace.final_field.complex_amplitude, expected.complex_amplitude)


def test_manager_rejects_concurrent_launch_and_snapshots_configuration():
    gate = threading.Event()
    seen = []
    def runner(config):
        seen.append(config)
        gate.wait(2)
        return execute(config)
    manager = RunManager(runner)
    config = examples()['01_circular']
    try:
        run_id = manager.start(config)
        with pytest.raises(BusyError):
            manager.start(config)
        config['name'] = 'edited during execution'
        assert manager.status(run_id)['state'] == 'running'
        gate.set()
        manager.future.result(timeout=30)
        assert manager.status(run_id)['state'] == 'complete'
        assert seen[0]['name'] != config['name']
    finally:
        gate.set()
        manager.close()


def test_failed_run_keeps_configuration_and_reports_error():
    config = examples()['01_circular']
    original = deepcopy(config)
    def fail(_):
        raise RuntimeError('Bloc pupil : erreur contrôlée')
    manager = RunManager(fail)
    try:
        run_id = manager.start(config)
        with pytest.raises(RuntimeError):
            manager.future.result(timeout=2)
        assert manager.status(run_id)['error'] == 'Bloc pupil : erreur contrôlée'
        assert config == original
    finally:
        manager.close()



def test_detector_view_is_integrated_acquisition_not_a_spectral_field():
    config = examples()['01_circular']
    config['spectrum'].update(samples=3, delta_wavelength=.1e-6)
    actual = execute(config)
    image = actual.image(3)
    torch.testing.assert_close(torch.tensor(image['values'], dtype=torch.float64),
                               actual.prepared.detector.image_buffer)
    with pytest.raises(ValueError, match='intègre les canaux'):
        actual.image(3, channel='0')
