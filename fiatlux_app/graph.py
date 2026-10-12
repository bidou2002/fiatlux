"""Data-only v2 graph: typed connections define physics; layout never does."""
from copy import deepcopy
import json
import math
import struct

from .model import CATALOG, ConfigurationError, _fields, _number, fingerprint, parameter, validate

GRAPH_CATALOG = deepcopy(CATALOG)
GRAPH_CATALOG['Kolmogorov']['fields']['realization'] = parameter('Réalisation', 0, 'integer', minimum=0, maximum=1000000)
GRAPH_CATALOG['Visualization'] = dict(label='Visualisation', group='Observation', fields={
    'quantity': parameter('Grandeur', 'intensity', 'quantity'),
    'channel': parameter('Longueur d’onde', 'sum', 'channel'),
    'colormap': parameter('Colormap', 'green', 'colormap'),
    'scale': parameter('Échelle', 'linear', 'scale'),
    'minimum': parameter('Limite basse (vide = auto)', None, nullable=True),
    'maximum': parameter('Limite haute (vide = auto)', None, nullable=True),
    'opd_unit': parameter('Unité OPD', 'nm', 'opd_unit'),
    'latent': parameter('Sélection des dimensions', {}, 'latent'),
})
OPD_TYPES = {'CircularAperture', 'ELTHarmoniPupil', 'Kolmogorov', 'DeformableMirror', 'Step', 'ZeldaMask', 'ZeldaStop'}


def ports(kind):
    if kind == 'Visualization':
        return dict(inputs={'field': 'Field', 'opd': 'OPD', 'counts': 'Counts'}, outputs={})
    output = {'counts': 'Counts'} if kind == 'Detector' else {'field': 'Field'}
    if kind in OPD_TYPES:
        output['opd'] = 'OPD'
    return dict(inputs={} if kind == 'PlaneWave' else {'in': 'Field'}, outputs=output)


def fail(message):
    raise ConfigurationError(message)


def effective_seed(params):
    """Counter-addressed independent realization; no mutable time or RNG cursor."""
    return (params['seed'] + 104729 * params['realization']) % 2147483648


def _structure(config):
    _fields(config, ['format', 'version', 'name', 'precision', 'spectrum', 'grids', 'nodes', 'connections', 'layout'], 'Canevas')
    if config['format'] != 'fiatlux-canvas' or type(config['version']) is not int or config['version'] != 2:
        fail('Canevas : format fiatlux-canvas version 2 requis')
    if not isinstance(config['nodes'], list) or not 3 <= len(config['nodes']) <= 40:
        fail('Canevas : 3 à 40 blocs requis')
    nodes = {}
    for node in config['nodes']:
        _fields(node, ['id', 'type', 'params'], 'Bloc')
        bid, kind = node['id'], node['type']
        if not isinstance(bid, str) or not bid or len(bid) > 80 or bid in nodes:
            fail('Bloc : identifiant unique de 1 à 80 caractères requis')
        if not isinstance(kind, str) or kind not in GRAPH_CATALOG:
            fail(f'Bloc {bid} : type non pris en charge')
        _fields(node['params'], GRAPH_CATALOG[kind]['fields'], f'Bloc {bid}')
        if kind == 'Kolmogorov':
            _number(node['params']['realization'], f'Bloc {bid}.realization', True, 0, 1000000)
        if kind == 'Visualization':
            p = node['params']
            for key, choices in [('quantity', ('intensity', 'phase', 'opd')), ('colormap', ('green', 'gray', 'magma', 'diverging')),
                                 ('scale', ('linear', 'log')), ('opd_unit', ('m', 'nm'))]:
                if p[key] not in choices:
                    fail(f'Bloc {bid}.{key} : valeur non prise en charge')
            if p['channel'] != 'sum' and (not isinstance(p['channel'], str) or not p['channel'].isdigit()):
                fail(f'Bloc {bid}.channel : sum ou index spectral requis')
            for key in ('minimum', 'maximum'):
                if p[key] is not None:
                    _number(p[key], f'Bloc {bid}.{key}')
            if p['minimum'] is not None and p['maximum'] is not None and p['minimum'] >= p['maximum']:
                fail(f'Bloc {bid} : limite basse ≥ limite haute')
            if p['scale'] == 'log' and p['quantity'] != 'intensity':
                fail(f'Bloc {bid} : logarithme disponible pour l’intensité uniquement')
            if p['scale'] == 'log' and p['minimum'] is not None and p['minimum'] <= 0:
                fail(f'Bloc {bid} : limite logarithmique strictement positive requise')
            if not isinstance(p['latent'], dict):
                fail(f'Bloc {bid}.latent : objet dimension/index requis')
            for name, index in p['latent'].items():
                if not isinstance(name, str):
                    fail(f'Bloc {bid}.latent : nom de dimension requis')
                _number(index, f'Bloc {bid}.latent.{name}', True, 0)
        nodes[bid] = node
    layout = config['layout']
    _fields(layout, ['positions', 'viewport'], 'Disposition')
    if not isinstance(layout['positions'], dict) or set(layout['positions']) != set(nodes):
        fail('Disposition : une position requise pour chaque bloc')
    for bid, pose in layout['positions'].items():
        _fields(pose, ['x', 'y', 'width', 'height', 'label'], f'Disposition {bid}')
        for key in ('x', 'y', 'width', 'height'):
            _number(pose[key], f'Disposition {bid}.{key}', minimum=-100000, maximum=100000)
        if pose['width'] < 180 or pose['height'] < 110:
            fail(f'Disposition {bid} : boîte trop petite')
        if not isinstance(pose['label'], str) or len(pose['label']) > 120:
            fail(f'Disposition {bid} : nom de présentation invalide')
    _fields(layout['viewport'], ['x', 'y', 'zoom'], 'Vue du canevas')
    for key in ('x', 'y', 'zoom'):
        _number(layout['viewport'][key], f'Vue.{key}')
    if not .2 <= layout['viewport']['zoom'] <= 2.5:
        fail('Vue.zoom : valeur entre 0.2 et 2.5 requise')
    return nodes


def check_connections(config):
    """Also validates unfinished drafts, so incompatible drops are refused immediately."""
    nodes = _structure(config)
    if not isinstance(config['connections'], list) or len(config['connections']) > 64:
        fail('Connexions : tableau de 64 connexions maximum requis')
    incoming, outgoing, observers = {}, {}, {}
    for edge in config['connections']:
        _fields(edge, ['from', 'to'], 'Connexion')
        for side in ('from', 'to'):
            _fields(edge[side], ['node', 'port'], 'Port')
        a, b = edge['from'], edge['to']
        if a['node'] not in nodes or b['node'] not in nodes:
            fail('Connexion : bloc inconnu')
        source, target = nodes[a['node']], nodes[b['node']]
        outs, ins = ports(source['type'])['outputs'], ports(target['type'])['inputs']
        if a['port'] not in outs or b['port'] not in ins or outs[a['port']] != ins[b['port']]:
            fail(f'Connexion {a["node"]}.{a["port"]} → {b["node"]}.{b["port"]} : ports incompatibles')
        if target['type'] == 'Visualization':
            if b['node'] in observers:
                fail(f'Visualisation {b["node"]} : une seule entrée autorisée')
            p = target['params']
            if (a['port'] == 'opd') != (p['quantity'] == 'opd'):
                fail(f'Visualisation {b["node"]} : quantité incompatible avec le port {a["port"]}')
            if a['port'] == 'counts' and (p['quantity'] != 'intensity' or p['channel'] != 'sum'):
                fail(f'Visualisation {b["node"]} : le détecteur fournit une acquisition intégrée')
            if p['quantity'] == 'phase' and p['channel'] == 'sum':
                fail(f'Visualisation {b["node"]} : choisir une longueur d’onde pour la phase')
            if p['channel'] != 'sum' and int(p['channel']) >= config['spectrum']['samples']:
                fail(f'Visualisation {b["node"]} : canal spectral hors limites')
            observers[b['node']] = a
        else:
            if b['node'] in incoming:
                fail(f'Bloc {b["node"]} : plusieurs entrées scientifiques interdites')
            if a['node'] in outgoing:
                fail(f'Bloc {a["node"]} : branche scientifique interdite ; seules les visualisations peuvent se ramifier')
            incoming[b['node']], outgoing[a['node']] = a['node'], b['node']
    for start in nodes:
        seen, current = set(), start
        while current in outgoing:
            if current in seen:
                fail(f'Cycle scientifique détecté au bloc {current}')
            seen.add(current)
            current = outgoing[current]
    # Explicit physical grids even when the rest of the chain is unfinished.
    def output_grid(bid):
        node, p = nodes[bid], nodes[bid]['params']
        if node['type'] == 'MFTPropagator':
            return config['grids'].get(p['output_grid'])
        if node['type'] == 'FFTPropagator':
            if config['spectrum']['samples'] != 1:
                fail(f'Bloc {bid} : FFT monochromatique uniquement')
            if bid not in incoming:
                return None
            grid = output_grid(incoming[bid])
            if grid is None:
                return None
            scale = p['focal_length'] if p['propagation'] == 'fraunhofer' else p['distance']
            if scale is None or not isinstance(scale, (int, float)):
                return None
            if scale == 0:
                return grid
            wavelength = config['spectrum']['central_wavelength']
            if config['precision'] == 'float32':
                wavelength = struct.unpack('f', struct.pack('f', wavelength))[0]
            return dict(nx=grid['nx'], ny=grid['ny'], dx=wavelength*abs(scale)/(grid['nx']*grid['dx']),
                        dy=wavelength*abs(scale)/(grid['ny']*grid['dy']))
        return config['grids'].get(p.get('grid'))
    for target, source in incoming.items():
        p = nodes[target]['params']
        actual = output_grid(source)
        if 'grid' in p and actual is not None and config['grids'].get(p['grid']) != actual:
            fail(f'Connexion {source} → {target} : grille {p["grid"]} incompatible avec la sortie de {source}')
    return nodes, incoming, outgoing, observers


def compile_graph(config):
    nodes, incoming, outgoing, observers = check_connections(config)
    scientific = [n for n in nodes.values() if n['type'] != 'Visualization']
    sources = [n['id'] for n in scientific if n['type'] == 'PlaneWave']
    detectors = [n['id'] for n in scientific if n['type'] == 'Detector']
    if len(sources) != 1 or len(detectors) != 1:
        fail('Chaîne : exactement une source et un détecteur requis')
    ordered, current = [], sources[0]
    while True:
        ordered.append(deepcopy(nodes[current]))
        if current not in outgoing:
            break
        current = outgoing[current]
    if len(ordered) != len(scientific) or ordered[-1]['id'] != detectors[0]:
        missing = next((n['id'] for n in scientific if n['id'] not in [b['id'] for b in ordered]), current)
        fail(f'Bloc {missing} : chaîne scientifique incomplète ; relier tous les blocs de la source au détecteur')
    for node in nodes.values():
        if node['type'] == 'Visualization' and node['id'] not in observers:
            fail(f'Visualisation {node["id"]} : entrée non connectée')
    for block in ordered:
        if block['type'] == 'Kolmogorov':
            block['params']['seed'] = effective_seed(block['params'])
            block['params'].pop('realization')
    chain = {key: deepcopy(config[key]) for key in ('name', 'precision', 'spectrum', 'grids')}
    chain.update(format='fiatlux-chain', version=1, blocks=ordered)
    validate(chain)
    return chain


def validate_graph(config):
    compile_graph(config)
    return config


def scientific_fingerprint(config):
    chain = compile_graph(config)
    chain.pop('name')
    # Only grids actually referenced by scientific operators matter.
    used = {v for b in chain['blocks'] for k, v in b['params'].items() if k in ('grid', 'output_grid')}
    chain['grids'] = {k: v for k, v in chain['grids'].items() if k in used}
    return fingerprint(chain)


def upgrade(chain):
    if chain.get('version') == 2:
        return deepcopy(validate_graph(chain))
    validate(chain)
    graph = {key: deepcopy(chain[key]) for key in ('name', 'precision', 'spectrum', 'grids')}
    graph.update(format='fiatlux-canvas', version=2, nodes=deepcopy(chain['blocks']), connections=[],
                 layout=dict(positions={}, viewport=dict(x=30, y=55, zoom=.85)))
    for i, node in enumerate(graph['nodes']):
        if node['type'] == 'Kolmogorov':
            node['params']['realization'] = 0
        graph['layout']['positions'][node['id']] = dict(x=i*250, y=30, width=210, height=150, label='')
        if i:
            graph['connections'].append( {'from': {'node': graph['nodes'][i-1]['id'], 'port': 'field'},
                                         'to': {'node': node['id'], 'port': 'in'}})
    return graph


def new_node(kind, bid):
    return dict(id=bid, type=kind, params={key: deepcopy(value['default']) for key, value in GRAPH_CATALOG[kind]['fields'].items()})


def add_visualization(graph, bid, source, port, quantity, x, y, channel='sum'):
    node = new_node('Visualization', bid)
    node['params'].update(quantity=quantity, channel=channel, colormap='diverging' if quantity != 'intensity' else 'green')
    graph['nodes'].append(node)
    graph['layout']['positions'][bid] = dict(x=x, y=y, width=280, height=290, label='')
    graph['connections'].append({'from': {'node': source, 'port': port}, 'to': {'node': bid, 'port': port}})
    return graph


def graph_examples():
    from .model import examples
    result = {name: upgrade(config) for name, config in examples().items()}
    for config in result.values():
        target = config['nodes'][-1]['id']
        add_visualization(config, 'view-intensity', target, 'counts', 'intensity', 650, 280)
    from pathlib import Path
    root = Path(__file__).parent / 'canvas_examples'
    for p in root.glob('*.json'):
        result[p.stem] = validate_graph(json.loads(p.read_text(encoding='utf-8')))
    return result
