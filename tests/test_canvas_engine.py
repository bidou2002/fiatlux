from copy import deepcopy
import pytest
import torch

from fiatlux_app.engine import execute
from fiatlux_app.graph import add_visualization, compile_graph, graph_examples
from fiatlux_app.incremental import IncrementalEngine, GraphResult, NodeSnapshot
from fiatlux.core.field import Field
from fiatlux.core.dimensions import FieldDimension


def test_graph_outputs_match_direct_serial_chain_and_display_units():
    graph = graph_examples()['04_canvas']
    actual = IncrementalEngine().execute(graph)
    direct = execute(compile_graph(graph))
    torch.testing.assert_close(actual.snapshots['focus'].field.complex_amplitude,
                               direct.trace.final_field.complex_amplitude, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual.snapshots['detector'].counts, direct.prepared.detector.image_buffer)
    nm = torch.tensor(actual.image('dm', port='opd', quantity='opd')['values'], dtype=torch.float64)
    torch.testing.assert_close(nm, actual.snapshots['dm'].opd*1e9)


def test_dm_change_reuses_upstream_fields_operators_and_atmosphere():
    graph = graph_examples()['04_canvas']
    engine = IncrementalEngine()
    before = engine.execute(graph)
    operators = dict(engine.operators)
    next(n for n in graph['nodes'] if n['id']=='dm')['params']['commands'][0] += 20e-9
    after = engine.execute(graph)
    assert after.computed == ['dm', 'focus', 'detector']
    assert after.reused == ['source', 'pupil', 'atmosphere']
    for bid in after.reused:
        assert after.snapshots[bid] is before.snapshots[bid]
        assert engine.operators[bid][1] is operators[bid][1]
    torch.testing.assert_close(after.snapshots['atmosphere'].opd, before.snapshots['atmosphere'].opd)
    assert not torch.equal(after.snapshots['focus'].field.complex_amplitude, before.snapshots['focus'].field.complex_amplitude)


def test_display_layout_and_observer_addition_do_not_change_any_scientific_output():
    graph = graph_examples()['04_canvas']
    engine = IncrementalEngine()
    before = engine.execute(graph)
    graph['layout']['positions']['dm']['x'] += 300
    graph['nodes'][-1]['params'].update(colormap='magma', scale='log')
    add_visualization(graph,'extra','pupil','field','intensity',10,700)
    after = engine.execute(graph)
    assert after.computed == []
    assert all(after.snapshots[key] is value for key,value in before.snapshots.items())
    engine_result = after.image('pupil')
    torch.testing.assert_close(torch.tensor(engine_result['values'],dtype=torch.float64), before.snapshots['pupil'].field.intensity().sum(-3))


def test_realization_survives_changed_illumination_and_has_explicit_counter():
    graph = graph_examples()['04_canvas']
    engine = IncrementalEngine()
    before = engine.execute(graph)
    screen = before.snapshots['atmosphere'].opd.clone()
    graph['spectrum']['central_wavelength'] *= 1.1
    illuminated = engine.execute(graph)
    torch.testing.assert_close(illuminated.snapshots['atmosphere'].opd,screen,rtol=0,atol=0)
    atmosphere = next(n for n in graph['nodes'] if n['id']=='atmosphere')
    atmosphere['params']['realization'] += 1
    changed = engine.execute(graph)
    assert not torch.equal(changed.snapshots['atmosphere'].opd,screen)
    repeated = IncrementalEngine().execute(graph)
    torch.testing.assert_close(changed.snapshots['atmosphere'].opd,repeated.snapshots['atmosphere'].opd,rtol=0,atol=0)
    atmosphere['params']['r0'] *= 2
    changed_stats = engine.execute(graph)
    assert not torch.equal(changed_stats.snapshots['atmosphere'].opd,changed.snapshots['atmosphere'].opd)


def test_changed_grid_rebuilds_mft_but_preserves_unreferenced_upstream_grid():
    graph = graph_examples()['04_canvas']
    engine = IncrementalEngine(); before = engine.execute(graph)
    old = engine.operators['focus'][1]
    graph['grids']['focal']['dx'] *= 1.05
    after = engine.execute(graph)
    assert after.computed == ['focus','detector']
    assert engine.operators['focus'][1] is not old
    assert after.snapshots['dm'] is before.snapshots['dm']


def test_latent_selection_is_explicit_and_does_not_advance_simulation():
    graph = graph_examples()['04_canvas']
    engine = IncrementalEngine(); result=engine.execute(graph)
    field=result.snapshots['source'].field
    dimension=FieldDimension('sample',2)
    latent_field=Field(torch.stack([field.complex_amplitude,2*field.complex_amplitude]),field.grid,field.spectrum,(dimension,))
    result.snapshots['source']=NodeSnapshot('test',latent_field)
    with pytest.raises(ValueError,match='explicitement'):
        result.image('source')
    image=result.image('source',latent={'sample':1})
    torch.testing.assert_close(torch.tensor(image['values'],dtype=torch.float64),4*field.intensity().sum(-3))
    assert engine.evaluations == 1
    with pytest.raises(ValueError,match='hors limites'):
        result.image('source',latent={'sample':2})
