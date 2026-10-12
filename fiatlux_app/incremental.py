"""Incremental FIATLUX evaluation with immutable snapshots and frozen atmosphere."""
from collections import OrderedDict
from dataclasses import dataclass
from copy import deepcopy
import time

from .engine import construct_element
from .graph import compile_graph
from .model import ConfigurationError, fingerprint


def local_key(block, chain):
    params = block['params']
    used = {value for name, value in params.items() if name in ('grid', 'output_grid')}
    value = dict(type=block['type'], params=params, precision=chain['precision'],
                 grids={name: chain['grids'][name] for name in used})
    if block['type'] == 'PlaneWave':
        value['spectrum'] = chain['spectrum']
    return fingerprint(value)


def path_keys(chain):
    previous = ''
    keys = []
    for block in chain['blocks']:
        # Spectrum can affect every transfer, but an unchanged local operator may be reused.
        previous = fingerprint(dict(parent=previous, local=local_key(block, chain)))
        keys.append(previous)
    return keys


@dataclass
class NodeSnapshot:
    key: str
    field: object
    opd: object = None
    counts: object = None
    dimensions: tuple = ()
    digitize: bool = False


@dataclass
class GraphResult:
    chain: dict
    snapshots: dict
    elapsed: float
    computed: list
    reused: list

    def metadata(self):
        nodes = []
        for block in self.chain['blocks']:
            snapshot = self.snapshots[block['id']]
            grid = snapshot.field.grid
            dimensions = snapshot.dimensions if snapshot.counts is not None else snapshot.field.dimensions
            nodes.append(dict(id=block['id'], type=block['type'], grid=dict(nx=grid.nx, ny=grid.ny, dx=grid.dx, dy=grid.dy),
                              opd=snapshot.opd is not None, opd_spectral=snapshot.opd is not None and snapshot.opd.ndim == 3,
                              dimensions=[dict(name=d.name, size=d.size, unit=d.unit,
                                               coordinates=None if d.coordinates is None else d.coordinates.detach().cpu().tolist()) for d in dimensions]))
        spectrum = next(iter(self.snapshots.values())).field.spectrum
        return dict(nodes=nodes, elapsed=self.elapsed, computed=self.computed, reused=self.reused,
                    wavelengths=spectrum.wavelengths.detach().cpu().tolist(), precision=self.chain['precision'])

    def image(self, node, port='field', quantity='intensity', channel='sum', opd_unit='nm', latent=None):
        import torch
        if node not in self.snapshots:
            raise ConfigurationError(f'Résultat du bloc {node} non disponible')
        snapshot = self.snapshots[node]
        field = snapshot.field
        selections = dict(latent or {})
        dimensions = snapshot.dimensions if port == 'counts' else field.dimensions
        if set(selections) != {d.name for d in dimensions}:
            raise ConfigurationError('Sélectionner explicitement chaque dimension latente présente')
        for dimension in dimensions:
            index = selections[dimension.name]
            if type(index) is not int or not 0 <= index < dimension.size:
                raise ConfigurationError(f'Dimension {dimension.name} : index hors limites')
        if port == 'counts':
            if snapshot.counts is None or quantity != 'intensity' or channel != 'sum':
                raise ConfigurationError('Le port Counts fournit uniquement l’acquisition intégrée')
            data = snapshot.counts
            for dimension in dimensions:
                data = data[selections[dimension.name]]
            unit = 'ADU/pixel' if snapshot.digitize else 'e⁻/pixel'
        elif port == 'opd':
            if snapshot.opd is None or quantity != 'opd':
                raise ConfigurationError('Cette sortie ne fournit pas d’OPD')
            data = snapshot.opd
            if data.ndim == 3:
                if channel == 'sum':
                    raise ConfigurationError('Cette OPD exige une longueur d’onde')
                data = data[self._channel(field, channel)]
            if opd_unit not in ('m', 'nm'):
                raise ConfigurationError('Unité OPD : m ou nm')
            data, unit = data * (1e9 if opd_unit == 'nm' else 1), opd_unit
        elif port == 'field':
            if snapshot.counts is not None:
                raise ConfigurationError('Le détecteur ne possède pas de port Field')
            for dimension in dimensions:
                field = field.select(dimension.name, selections[dimension.name])
            if quantity == 'phase':
                if channel == 'sum':
                    raise ConfigurationError('Phase : choisir une longueur d’onde')
                data, unit = field.phase()[self._channel(field, channel)], 'rad'
            elif quantity == 'intensity':
                intensities = field.intensity()
                data = intensities.sum(-3) if channel == 'sum' else intensities[self._channel(field, channel)]
                unit = 'photons/s/m²'
            else:
                raise ConfigurationError('Le port Field fournit intensité ou phase')
        else:
            raise ConfigurationError('Port de résultat inconnu')
        if not torch.isfinite(data).all():
            raise ConfigurationError('Données optiques non finies')
        # Only the visualization boundary converts tensors into CPU/NumPy/JSON.
        array = data.detach().cpu().numpy()
        return dict(values=array.tolist(), nx=field.grid.nx, ny=field.grid.ny, unit=unit,
                    x=field.grid.x.detach().cpu().tolist(), y=field.grid.y.detach().cpu().tolist(),
                    minimum=float(array.min()), maximum=float(array.max()), quantity=quantity, node=node, port=port)

    @staticmethod
    def _channel(field, channel):
        index = int(channel)
        if not 0 <= index < len(field.spectrum.wavelengths):
            raise ConfigurationError('Canal spectral hors limites')
        return index


class IncrementalEngine:
    def __init__(self):
        self.entries = {}
        self.operators = {}
        self.realizations = OrderedDict()
        self.evaluations = 0

    def affected(self, graph):
        chain = compile_graph(graph)
        keys = path_keys(chain)
        return [block['id'] for block, key in zip(chain['blocks'], keys)
                if block['id'] not in self.entries or self.entries[block['id']].key != key]

    def execute(self, graph):
        import torch
        from fiatlux.core.grid import Grid
        from fiatlux.core.source import PlaneWave
        from fiatlux.core.spectrum import Band, Spectrum
        from fiatlux.optics.elements.mask import Atmosphere
        from fiatlux.system.optical_system import SerialSystem
        from fiatlux.optics.detector import DetectorImage
        chain = compile_graph(graph)
        dtype = getattr(torch, chain['precision'])
        grids = {name: Grid(**values, device='cpu', dtype=dtype) for name, values in chain['grids'].items()}
        keys = path_keys(chain)
        snapshots, computed, reused = {}, [], []
        start = time.perf_counter()
        self.evaluations += 1
        current = None
        with torch.no_grad():
            for block, key in zip(chain['blocks'], keys):
                bid = block['id']
                cached = self.entries.get(bid)
                if cached is not None and cached.key == key:
                    snapshots[bid] = cached
                    current = cached.field
                    reused.append(bid)
                    continue
                try:
                    opkey = local_key(block, chain)
                    operator = self.operators.get(bid)
                    if operator is None or operator[0] != opkey:
                        if block['type'] == 'PlaneWave':
                            spec = chain['spectrum']
                            obj = PlaneWave(Spectrum(spec['magnitude'], Band(spec['central_wavelength'], spec['delta_wavelength'], spec['f0']),
                                                     spec['samples'], dtype=dtype))
                        else:
                            obj = construct_element(block, grids)
                        if block['type'] == 'Kolmogorov':
                            # A chromatic Mask.build() normally calls _build_opd again.
                            # Freeze the native sampled OPD so changing illumination never samples again.
                            if opkey not in self.realizations:
                                self.realizations[opkey] = obj.atmosphere_model.sample_opd(
                                    generator=obj.generator, remove_piston=obj.remove_piston).detach().clone()
                                if len(self.realizations) > 16:
                                    self.realizations.popitem(last=False)
                            frozen_opd = self.realizations[opkey]
                            class FrozenAtmosphere(Atmosphere):
                                def _build_opd(self):
                                    self.opd = self.fixed_opd
                            obj = FrozenAtmosphere(obj.grid, obj.atmosphere_model, recompute=False)
                            obj.fixed_opd = frozen_opd
                        self.operators[bid] = (opkey, obj)
                    else:
                        obj = operator[1]
                    if block['type'] == 'PlaneWave':
                        current = obj.generate_field(grids[block['params']['grid']])
                        snapshot = NodeSnapshot(key, current)
                    elif block['type'] == 'Detector':
                        acquired = obj.acquire(current)
                        counts = acquired.data if isinstance(acquired, DetectorImage) else acquired
                        dimensions = acquired.dimensions if isinstance(acquired, DetectorImage) else ()
                        snapshot = NodeSnapshot(key, current, counts=counts.detach().clone(), dimensions=dimensions, digitize=obj.digitize)
                    else:
                        # Real SerialSystem applies the native operator to the valid upstream Field.
                        current = SerialSystem([obj]).run_field(current).final_field
                        opd = getattr(obj, 'opd', None)
                        snapshot = NodeSnapshot(key, current, opd=None if opd is None else opd.detach().clone())
                    self.entries[bid] = snapshot
                    snapshots[bid] = snapshot
                    computed.append(bid)
                except (ValueError, TypeError, RuntimeError) as error:
                    raise ConfigurationError(f'Bloc {bid} ({block["type"]}) : {error}') from error
        active = set(snapshots)
        self.entries = {key: value for key, value in self.entries.items() if key in active}
        self.operators = {key: value for key, value in self.operators.items() if key in active}
        return GraphResult(deepcopy(chain), snapshots, time.perf_counter()-start, computed, reused)
