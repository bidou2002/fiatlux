"""Construction, execution and visualization of real FIATLUX objects."""
from dataclasses import dataclass
import time

from .model import ConfigurationError, fingerprint, validate


@dataclass
class PreparedSystem:
    configuration: dict
    source: object
    source_grid: object
    system: object
    detector: object
    blocks: list


def build_system(config):
    validate(config)
    import torch
    from fiatlux.config.loader import _validate_grid_chain
    from fiatlux.core.grid import Grid
    from fiatlux.core.source import PlaneWave
    from fiatlux.core.spectrum import Band, Spectrum
    from fiatlux.optics.atmosphere import KolmogorovAtmosphereModel
    from fiatlux.optics.elements.deformable_mirror import ActuatorGrid, DeformableMirror, ZernikeBasis
    from fiatlux.optics.elements.mask import Atmosphere, CircularAperture, Step, ZeldaMask, ZeldaStop
    from fiatlux.optics.elements.segmented_aperture import ELTHarmoniPupil
    from fiatlux.optics.propagator import MFTPropagator, FFTPropagator
    from fiatlux.optics.detector import Detector
    from fiatlux.system.optical_system import SerialSystem

    dtype = getattr(torch, config['precision'])
    grids = {name: Grid(**values, device='cpu', dtype=dtype) for name, values in config['grids'].items()}
    sp = config['spectrum']
    spectrum = Spectrum(sp['magnitude'], Band(sp['central_wavelength'], sp['delta_wavelength'], sp['f0']),
                        sp['samples'], device='cpu', dtype=dtype)
    if (not torch.isfinite(spectrum.fluxes).all() or torch.any(spectrum.fluxes < 0)
            or not torch.isfinite(spectrum.wavelengths).all() or torch.any(spectrum.wavelengths <= 0)):
        raise ConfigurationError('Spectre : flux non fini ou négatif')
    source = PlaneWave(spectrum)
    source_grid = grids[config['blocks'][0]['params']['grid']]
    objects, detector = [], None
    mask_classes = dict(CircularAperture=CircularAperture, ELTHarmoniPupil=ELTHarmoniPupil,
                        Step=Step, ZeldaMask=ZeldaMask, ZeldaStop=ZeldaStop)
    for block in config['blocks'][1:]:
        kind, params = block['type'], dict(block['params'])
        try:
            grid = grids[params.pop('grid')] if 'grid' in params else None
            if kind in mask_classes:
                obj = mask_classes[kind](grid=grid, **params)
            elif kind == 'Kolmogorov':
                model = KolmogorovAtmosphereModel(grid=grid, **params)
                obj = Atmosphere(grid, model, recompute=False)
            elif kind == 'DeformableMirror':
                commands = params.pop('commands')
                # Zernike is modal; this constructor layout does not define zonal influences.
                obj = DeformableMirror(grid, ActuatorGrid(len(commands), 1, grid.dx), grid,
                                       ZernikeBasis(grid, len(commands)), **params)
                obj.commands = torch.tensor(commands, dtype=dtype)
            elif kind == 'MFTPropagator':
                obj = MFTPropagator(output_grid=grids[params.pop('output_grid')], **params)
            elif kind == 'FFTPropagator':
                obj = FFTPropagator(**params)
            elif kind == 'Detector':
                detector = Detector(grid=grid, **params)
                continue
            else:
                raise ConfigurationError(f'Composant non pris en charge : {kind}')
            objects.append(obj)
        except (ValueError, TypeError, RuntimeError) as error:
            raise ConfigurationError(f'Bloc {block["id"]} ({kind}) : {error}') from error
    system = SerialSystem(objects)
    # Reuse the existing grid-chain preflight where it models all planes.
    # FFT owns a wavelength-dependent output_grid_for contract, handled above.
    if (hasattr(objects[0], 'grid')
            and not any(isinstance(obj, FFTPropagator) for obj in objects)):
        try:
            _validate_grid_chain(system, detector)
        except ValueError as error:
            raise ConfigurationError(f'Grilles de la chaîne : {error}') from error
    return PreparedSystem(config, source, source_grid, system, detector, config['blocks'])


class _NamedElement:
    """Error context only: optical application is entirely delegated to FIATLUX."""
    def __init__(self, element, block):
        self.element, self.block = element, block

    def apply(self, field):
        try:
            return self.element.apply(field)
        except (ValueError, TypeError, RuntimeError) as error:
            raise ConfigurationError(f'Bloc {self.block["id"]} ({self.block["type"]}) : {error}') from error


@dataclass
class RunResult:
    prepared: PreparedSystem
    trace: object
    elapsed: float

    def metadata(self):
        steps = []
        for index, (block, step) in enumerate(zip(self.prepared.blocks[:-1], self.trace.steps)):
            grid = step.field_after.grid
            obj = None if index == 0 else self.prepared.system.elements[index-1]
            steps.append(dict(id=block['id'], label=block['type'], index=index,
                              grid=dict(nx=grid.nx, ny=grid.ny, dx=grid.dx, dy=grid.dy),
                              opd=hasattr(obj, 'opd') and obj.opd is not None,
                              detector=False))
        detector = self.prepared.detector
        steps.append(dict(id=self.prepared.blocks[-1]['id'], label='Detector', index=len(steps),
                          grid=dict(nx=detector.grid.nx, ny=detector.grid.ny, dx=detector.grid.dx, dy=detector.grid.dy),
                          opd=False, detector=True))
        return dict(fingerprint=fingerprint(self.prepared.configuration), elapsed=self.elapsed, steps=steps,
                    wavelengths=self.prepared.source.spectrum.wavelengths.detach().cpu().tolist(),
                    fluxes=self.prepared.source.spectrum.fluxes.detach().cpu().tolist(),
                    precision=self.prepared.configuration['precision'])

    def image(self, step, quantity='intensity', channel='sum', opd_unit='nm'):
        import torch
        metadata = self.metadata()
        if not 0 <= step < len(metadata['steps']):
            raise ConfigurationError('Étape inconnue')
        descriptor = metadata['steps'][step]
        if quantity not in ('intensity', 'phase', 'opd'):
            raise ConfigurationError('Vue inconnue')
        if channel != 'sum' and not 0 <= int(channel) < len(metadata['wavelengths']):
            raise ConfigurationError('Longueur d’onde inconnue')
        if descriptor['detector']:
            if quantity != 'intensity':
                raise ConfigurationError('Le détecteur fournit uniquement une intensité mesurée')
            spectral = self.prepared.detector.image_buffer
            data = spectral.sum(-3) if channel == 'sum' else spectral[int(channel)]
            unit = 'ADU/pixel' if self.prepared.detector.digitize else 'e⁻/pixel'
        elif quantity == 'opd':
            if not descriptor['opd']:
                raise ConfigurationError('Cette étape ne fournit pas de carte OPD')
            if opd_unit not in ('m', 'nm'):
                raise ConfigurationError('Unité OPD : m ou nm')
            data = self.prepared.system.elements[step-1].opd
            if data.ndim == 3:
                if channel == 'sum':
                    raise ConfigurationError('Sélectionner une longueur d’onde pour cette OPD spectrale')
                data = data[int(channel)]
            data = data * (1e9 if opd_unit == 'nm' else 1)
            unit = opd_unit
        else:
            field = self.trace.steps[step].field_after
            if quantity == 'phase':
                if channel == 'sum':
                    raise ConfigurationError('La phase exige une longueur d’onde ; pas de phase polychromatique')
                data, unit = field.phase()[int(channel)], 'rad'
            else:
                spectral = field.intensity()
                # FIATLUX already encodes per-channel flux in amplitude: do not weight twice.
                data = spectral.sum(-3) if channel == 'sum' else spectral[int(channel)]
                unit = 'photons/s/m²'
        if not torch.isfinite(data).all():
            raise ConfigurationError('Le moteur a produit des données non finies')
        grid = self.trace.final_field.grid if descriptor['detector'] else self.trace.steps[step].field_after.grid
        # The visualization boundary is the only NumPy conversion.
        array = data.detach().cpu().numpy()
        return dict(values=array.tolist(), nx=grid.nx, ny=grid.ny, unit=unit,
                    x=grid.x.detach().cpu().tolist(), y=grid.y.detach().cpu().tolist(),
                    minimum=float(array.min()), maximum=float(array.max()), quantity=quantity,
                    channel=channel, step=step)


def execute(config):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError('PyTorch absent : installer FIATLUX avec python -m pip install -e .') from error
    from fiatlux.system.optical_system import SerialSystem
    prepared = build_system(config)
    start = time.perf_counter()
    with torch.no_grad():
        try:
            field = prepared.source.generate_field(prepared.source_grid)
        except (ValueError, TypeError, RuntimeError) as error:
            raise ConfigurationError(f'Bloc {config["blocks"][0]["id"]} (PlaneWave) : {error}') from error
        wrappers = [_NamedElement(element, block) for element, block in
                    zip(prepared.system.elements, config['blocks'][1:-1])]
        trace = SerialSystem(wrappers).run_field(field)
        try:
            prepared.detector.acquire(trace.final_field)
        except (ValueError, TypeError, RuntimeError) as error:
            raise ConfigurationError(f'Bloc {config["blocks"][-1]["id"]} (Detector) : {error}') from error
    return RunResult(prepared, trace, time.perf_counter()-start)
