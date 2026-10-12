"""Versioned, data-only configuration and strict preflight validation.

This module deliberately imports no scientific or graphical dependencies.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import struct


class ConfigurationError(ValueError):
    pass


def parameter(label, default, kind='number', unit='', minimum=None, maximum=None, nullable=False):
    return dict(label=label, default=default, kind=kind, unit=unit,
                minimum=minimum, maximum=maximum, nullable=nullable)


def grid_parameter():
    return parameter('Grille du plan', 'pupil', 'grid')


CATALOG = {
    'PlaneWave': dict(label='Source plane', group='Source', fields={'grid': grid_parameter()}),
    'CircularAperture': dict(label='Pupille circulaire', group='Pupille', fields={
        'grid': grid_parameter(), 'radius': parameter('Rayon', .5, unit='m', minimum=0)}),
    'ELTHarmoniPupil': dict(label='Pupille ELT / HARMONI', group='Pupille', fields={
        'grid': grid_parameter(),
        'outer_radius': parameter('Rayon extérieur', 18.5, unit='m', minimum=0),
        'central_obscuration_radius': parameter('Obscuration centrale', 5.5, unit='m', minimum=0),
        'circular_mask': parameter('Découpe circulaire extérieure', False, 'bool'),
        'segment_circumradius': parameter('Rayon des segments', .725, unit='m', minimum=0),
        'segment_gap': parameter('Espacement des segments', .004, unit='m', minimum=0),
        'segment_count': parameter('Nombre de segments', 798, 'integer', minimum=1, maximum=798),
        'spider_width': parameter('Largeur des araignées', .5, unit='m', minimum=0),
        'petal_gap': parameter('Espacement M4', 0., unit='m', minimum=0),
        'rotation': parameter('Rotation', 0., unit='°')}),
    'Kolmogorov': dict(label='Atmosphère Kolmogorov', group='Phase', fields={
        'grid': grid_parameter(), 'r0': parameter('Paramètre de Fried', .2, unit='m', minimum=0),
        'reference_wavelength': parameter('Longueur d’onde de référence', 500e-9, unit='m', minimum=0),
        'outer_scale': parameter('Échelle externe (vide = infinie)', None, unit='m', minimum=0, nullable=True),
        'seed': parameter('Graine aléatoire', 42, 'integer', minimum=0, maximum=2147483647)}),
    'DeformableMirror': dict(label='Miroir déformable · Zernike', group='Phase', fields={
        'grid': grid_parameter(),
        'commands': parameter('Coefficients Noll 2… · OPD', [0., 0., 0.], 'array', 'm'),
        'stroke': parameter('Limite de coefficient OPD', 1e-6, unit='m', minimum=0)}),
    'Step': dict(label='Marche de phase', group='Phase', fields={
        'grid': grid_parameter(), 'piston': parameter('OPD de la marche', 82.5e-9, unit='m')}),
    'MFTPropagator': dict(label='Propagation MFT', group='Propagation', fields={
        'output_grid': parameter('Grille de sortie', 'focal', 'grid'),
        'propagation': parameter('Régime', 'fraunhofer', 'regime'),
        'focal_length': parameter('Focale · Fraunhofer', 10., unit='m', nullable=True),
        'distance': parameter('Distance · Fresnel', None, unit='m', nullable=True)}),
    'FFTPropagator': dict(label='Propagation FFT', group='Propagation', fields={
        'propagation': parameter('Régime', 'fraunhofer', 'regime'),
        'focal_length': parameter('Focale · Fraunhofer', 10., unit='m', nullable=True),
        'distance': parameter('Distance · Fresnel', None, unit='m', nullable=True)}),
    'ZeldaMask': dict(label='Masque ZELDA', group='Phase', fields={
        'grid': parameter('Grille du plan', 'focal', 'grid'),
        'radius': parameter('Rayon du masque', 16.5e-6, unit='m', minimum=0),
        'well_depth': parameter('Profondeur optique', 412.5e-9, unit='m')}),
    'ZeldaStop': dict(label='Diaphragme ZELDA', group='Pupille', fields={
        'grid': parameter('Grille du plan', 'focal', 'grid'),
        'radius': parameter('Rayon', 50e-6, unit='m', minimum=0)}),
    'Detector': dict(label='Détecteur', group='Acquisition', fields={
        'grid': parameter('Grille du plan', 'focal', 'grid'),
        'exposure_time': parameter('Temps de pose', .15, unit='s', minimum=0),
        'quantum_efficiency': parameter('Rendement quantique', 1., minimum=0, maximum=1),
        'photon_noise': parameter('Bruit photonique', False, 'bool'),
        'dark_current': parameter('Courant d’obscurité', 0., unit='e⁻/pixel/s', minimum=0),
        'readout_noise_variance': parameter('Variance du bruit de lecture', 0., unit='e⁻²', minimum=0),
        'digitize': parameter('Conversion en ADU', False, 'bool'),
        'sensitivity': parameter('Gain de conversion', 1., unit='ADU/e⁻', minimum=0),
        'offset': parameter('Offset', 0, 'integer', 'ADU'),
        'bitdepth': parameter('Résolution ADC', 16, 'integer', 'bits', minimum=1, maximum=32),
        'random_seed': parameter('Graine aléatoire', 42, 'integer', minimum=0, maximum=2147483647)}),
}


def _error(path, message):
    raise ConfigurationError(f'{path} : {message}')


def _fields(obj, keys, path):
    if not isinstance(obj, dict):
        _error(path, 'un objet JSON est requis')
    missing, extra = set(keys) - obj.keys(), obj.keys() - set(keys)
    if missing:
        _error(path, f'paramètre requis : {sorted(missing)[0]}')
    if extra:
        _error(path, f'paramètre inconnu : {sorted(extra)[0]}')


def _number(value, path, integer=False, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _error(path, 'nombre fini requis')
    if integer and not isinstance(value, int):
        _error(path, 'entier requis')
    if minimum is not None and value < minimum:
        _error(path, f'valeur ≥ {minimum} requise')
    if maximum is not None and value > maximum:
        _error(path, f'valeur ≤ {maximum} requise')


def validate(config):
    """Reject incompatible data without filling defaults or repairing values."""
    _fields(config, ['format', 'version', 'name', 'precision', 'spectrum', 'grids', 'blocks'], 'Système')
    if config['format'] != 'fiatlux-chain' or type(config['version']) is not int or config['version'] != 1:
        _error('Système', 'format fiatlux-chain version 1 requis')
    if not isinstance(config['name'], str) or not 1 <= len(config['name']) <= 120:
        _error('Nom', 'texte de 1 à 120 caractères requis')
    if config['precision'] not in ('float32', 'float64'):
        _error('Précision', 'float32 ou float64 requis')
    spec = config['spectrum']
    _fields(spec, ['central_wavelength', 'delta_wavelength', 'f0', 'magnitude', 'samples'], 'Spectre')
    for key in ('central_wavelength', 'delta_wavelength', 'f0', 'magnitude'):
        _number(spec[key], f'Spectre.{key}')
    _number(spec['samples'], 'Spectre.samples', integer=True, minimum=1, maximum=8)
    if spec['central_wavelength'] <= 0 or spec['delta_wavelength'] < 0 or spec['f0'] <= 0:
        _error('Spectre', 'λ centrale et f0 positifs, largeur de bande non négative requis')
    if spec['central_wavelength'] - spec['delta_wavelength']/2 <= 0:
        _error('Spectre', 'toutes les longueurs d’onde doivent être positives')
    if abs(spec['magnitude']) > 100:
        _error('Spectre.magnitude', 'valeur entre −100 et 100 requise')
    grids = config['grids']
    if not isinstance(grids, dict) or not 1 <= len(grids) <= 8:
        _error('Grilles', '1 à 8 grilles nommées requises')
    for name, grid in grids.items():
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,39}', name):
            _error('Grilles', 'nom de grille invalide')
        _fields(grid, ['nx', 'ny', 'dx', 'dy'], f'Grille {name}')
        for key in ('nx', 'ny'):
            _number(grid[key], f'Grille {name}.{key}', True, 2, 256)
        for key in ('dx', 'dy'):
            _number(grid[key], f'Grille {name}.{key}')
            if grid[key] <= 0:
                _error(f'Grille {name}.{key}', 'pas spatial positif requis (m)')
    blocks = config['blocks']
    if not isinstance(blocks, list) or not 3 <= len(blocks) <= 24:
        _error('Chaîne', '3 à 24 blocs requis')
    ids = set()
    for i, block in enumerate(blocks):
        _fields(block, ['id', 'type', 'params'], f'Bloc {i+1}')
        bid, kind, params = block['id'], block['type'], block['params']
        path = f'Bloc {i+1} · {bid}'
        if not isinstance(bid, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,79}', bid) or bid in ids:
            _error(path, 'identifiant unique requis')
        ids.add(bid)
        if not isinstance(kind, str) or kind not in CATALOG:
            _error(path, 'composant non pris en charge')
        fields = CATALOG[kind]['fields']
        _fields(params, fields, path)
        for key, value in params.items():
            rule, fieldpath = fields[key], f'{path}.{key}'
            if value is None and rule['nullable']:
                continue
            if rule['kind'] == 'bool':
                if type(value) is not bool:
                    _error(fieldpath, 'booléen requis')
            elif rule['kind'] == 'grid':
                if not isinstance(value, str) or value not in grids:
                    _error(fieldpath, 'référence à une grille définie requise')
            elif rule['kind'] == 'regime':
                if value not in ('fraunhofer', 'fresnel'):
                    _error(fieldpath, 'fraunhofer ou fresnel requis')
            elif rule['kind'] == 'array':
                if not isinstance(value, list) or not 1 <= len(value) <= 36:
                    _error(fieldpath, '1 à 36 coefficients OPD requis')
                for j, entry in enumerate(value):
                    _number(entry, f'{fieldpath}[{j}]')
            else:
                _number(value, fieldpath, rule['kind'] == 'integer', rule['minimum'], rule['maximum'])
        for key in ('radius', 'r0', 'reference_wavelength', 'segment_circumradius', 'outer_radius', 'stroke', 'sensitivity'):
            if key in params and params[key] <= 0:
                _error(f'{path}.{key}', 'valeur strictement positive requise')
        if kind == 'ELTHarmoniPupil' and params['central_obscuration_radius'] >= params['outer_radius']:
            _error(path, 'obscuration centrale ≥ rayon extérieur')
        if kind == 'Kolmogorov' and params['outer_scale'] is not None and params['outer_scale'] <= 0:
            _error(path, 'échelle externe positive ou vide requise')
        if kind == 'DeformableMirror':
            grid = grids[params['grid']]
            if grid['nx'] != grid['ny'] or grid['dx'] != grid['dy']:
                _error(path, 'la base Zernike actuelle exige une grille carrée isotrope')
            if any(abs(c) > params['stroke'] for c in params['commands']):
                _error(path, 'un coefficient OPD dépasse le stroke ; aucune saturation silencieuse')
        if kind in ('MFTPropagator', 'FFTPropagator'):
            if params['propagation'] == 'fraunhofer':
                if params['focal_length'] is None or params['focal_length'] == 0 or params['distance'] is not None:
                    _error(path, 'Fraunhofer exige une focale non nulle et distance vide')
            elif params['distance'] is None or params['focal_length'] is not None:
                _error(path, 'Fresnel exige distance (m) et focale vide')
            if kind == 'FFTPropagator' and spec['samples'] != 1:
                _error(path, 'FFT monochromatique uniquement ; utiliser MFT pour plusieurs λ')
    kinds = [b['type'] for b in blocks]
    if kinds[0] != 'PlaneWave' or kinds.count('PlaneWave') != 1:
        _error('Chaîne', 'exactement une source PlaneWave, en première position')
    if kinds[-1] != 'Detector' or kinds.count('Detector') != 1:
        _error('Chaîne', 'exactement un détecteur, en dernière position')
    # Track physical sampling through propagators, without allocating a field.
    current = grids[blocks[0]['params']['grid']]
    for i, block in enumerate(blocks[1:], 2):
        params, kind = block['params'], block['type']
        path = f'Bloc {i} · {block["id"]}'
        if 'grid' in params and grids[params['grid']] != current:
            _error(path, 'grille incompatible avec le plan précédent ; ajouter une propagation explicite')
        if kind == 'MFTPropagator':
            output = grids[params['output_grid']]
            if params['propagation'] == 'fresnel' and params['distance'] == 0 and output != current:
                _error(path, 'Fresnel à distance zéro exige la même grille')
            current = output
        if kind == 'FFTPropagator':
            scale = params['focal_length'] if params['propagation'] == 'fraunhofer' else params['distance']
            if scale != 0:
                wavelength = spec['central_wavelength']
                if config['precision'] == 'float32':
                    try:
                        wavelength = struct.unpack('f', struct.pack('f', wavelength))[0]
                    except (OverflowError, struct.error):
                        _error(path, 'longueur d’onde non représentable en float32')
                current = dict(nx=current['nx'], ny=current['ny'],
                               dx=wavelength*abs(scale)/(current['nx']*current['dx']),
                               dy=wavelength*abs(scale)/(current['ny']*current['dy']))
    return config


def parse_json(text):
    def reject(value):
        raise ConfigurationError(f'JSON : valeur non finie {value}')
    def pairs(entries):
        output = {}
        for key, value in entries:
            if key in output:
                raise ConfigurationError(f'JSON : clé dupliquée {key}')
            output[key] = value
        return output
    try:
        return json.loads(text, parse_constant=reject, object_pairs_hook=pairs)
    except json.JSONDecodeError as error:
        raise ConfigurationError(f'JSON ligne {error.lineno} : {error.msg}') from error


def loads(text):
    return validate(parse_json(text))


def dumps(config):
    validate(config)
    return json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def examples():
    root = Path(__file__).parent / 'examples'
    return {p.stem: loads(p.read_text(encoding='utf-8')) for p in sorted(root.glob('*.json'))}


def new_block(kind, bid):
    return dict(id=bid, type=kind, params={key: deepcopy(rule['default'])
                                         for key, rule in CATALOG[kind]['fields'].items()})
