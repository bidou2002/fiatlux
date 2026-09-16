"""Optional P3 adapter: FIATLUX owns sampling; P3 only evaluates the PSD.

Validated against astro-tiptop 1.5.1 / astro-p3 1.6.2. No PSD interpolation.
P3 fourierModel.powerSpectrumDensity returns centered OPD **bin power** in
nm²: its final multiplier is (dk * wavelength_nm / (2*pi))², where
dk = 2*kcMax_/resAO (not necessarily PSDstep, because resAO is truncated).
Recover continuous OPD density as bin_power * 1e-18 / dk² before exact
frequency extraction. FIATLUX integrates with its own df², never P3's bin area.
P3 uses (x,y,source); FIATLUX uses (source,y,x) and unshifted FFT PSDs.
"""
import ast
import configparser
import copy
from dataclasses import dataclass
from importlib.metadata import version
import math
from pathlib import Path

import numpy as np
import torch

from .tabulated_atmosphere import TabulatedAtmosphereModel

RAD2MAS = 180 * 3600 * 1000 / math.pi


class TiptopSamplingError(ValueError):
    """P3 cannot represent the requested, unchanged FIATLUX Fourier samples."""


def _cpu(value):
    return np.asarray(value.get() if hasattr(value, "get") else value)


def load_harmoni_scao_config(path=None):
    """Load TIPTOP's installed HARMONI SCAO NGS preset or an explicit INI.

    No physical AO parameters are hardcoded here. Values must be Python
    literals; unlike P3's legacy INI reader, this loader does not use eval.
    Relative asset paths in the default preset resolve against TIPTOP's root.
    """
    if path is None:
        import tiptop
        path = Path(tiptop.__file__).parent / "perfTest" / "HARMONI_SCAO.ini"
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    with Path(path).open() as stream:
        parser.read_file(stream)
    return {section: {key: ast.literal_eval(value) for key, value in parser[section].items()}
            for section in parser.sections()}


def _target(grid):
    if grid.nx != grid.ny or not math.isclose(grid.dx, grid.dy, rel_tol=1e-12):
        raise TiptopSamplingError("P3 currently supports square, isotropic target grids only; FIATLUX was not changed.")
    if grid.nx < 2:
        raise TiptopSamplingError("At least two target samples are required.")
    df = 1 / (grid.nx * grid.dx)
    f = np.fft.fftshift(np.fft.fftfreq(grid.nx, d=grid.dx))
    return df, f


def _sampling(pixel_scale, wavelengths, diameter):
    samp = wavelengths * RAD2MAS / (pixel_scale * diameter)
    k = np.ceil(2 / samp).astype(int)
    steps = pixel_scale / (wavelengths * RAD2MAS * k)
    i = int(np.argmin(steps))
    kref_float = k[i] * wavelengths[i] / wavelengths.min()
    return float(steps[i]), int(np.ceil(kref_float)), k, samp * k


def configure_tiptop_sampling(grid, config, *, sampling_ratio=None):
    """Configure an auxiliary P3 grid containing every FIATLUX frequency.

    Search integer refinement ratios and P3 camera oversampling regimes.
    Only P3's camera configuration is changed; the simulation Grid is input.
    """
    df, target = _target(grid)
    cfg = copy.deepcopy(config)
    wavelengths = np.unique(np.atleast_1d(cfg["sources_science"]["Wavelength"]).astype(float))
    if not np.isfinite(wavelengths).all() or (wavelengths <= 0).any():
        raise ValueError("Science wavelengths must be finite and positive.")
    sensor = cfg["sensor_science"]
    if sensor.get("SpectralBandwidth", 0) != 0 or len(sensor.get("Transmittance", [1])) != 1:
        raise ValueError("Supply explicit wavelengths, zero sensor bandwidth and one transmittance bin.")
    diameter = float(cfg["telescope"]["TelescopeDiameter"])
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("TelescopeDiameter must be positive and finite.")
    if sampling_ratio is not None and (isinstance(sampling_ratio, bool) or
            not isinstance(sampling_ratio, int) or sampling_ratio < 1):
        raise ValueError("sampling_ratio must be a positive integer.")
    ratios = [sampling_ratio] if sampling_ratio is not None else range(1, 65)
    for q in ratios:
        target_step = df / q
        if target_step > 1/(2*diameter)*(1+1e-12):
            continue
        candidates = []
        for factor in range(1, 65):
            for wavelength in wavelengths:
                nominal = target_step * wavelength * RAD2MAS * factor
                # Avoid ceil changing regime at an exactly integral boundary.
                for pixel in [nominal, np.nextafter(nominal, 0)]:
                    step, kref, _, _ = _sampling(pixel, wavelengths, diameter)
                    if not math.isclose(step, target_step, rel_tol=1e-11, abs_tol=1e-14):
                        continue
                    # Both endpoints matter for odd/even target and source grids.
                    negative = q * (grid.nx // 2)
                    positive = q * ((grid.nx - 1)//2)
                    minimum_size = max(2*negative, 2*positive+1)
                    # P3 must also contain its own AO-corrected support.
                    minimum_size = max(minimum_size, math.ceil(1/min(cfg["DM"]["DmPitchs"])/step))
                    fov = math.ceil(minimum_size/kref)
                    candidates.append((fov*kref, factor, float(pixel), fov))
        if candidates:
            _, _, pixel, fov = min(candidates)
            sensor.update(PixelScale=pixel, FieldOfView=fov)
            return cfg
    raise TiptopSamplingError("No compatible auxiliary grid found in the requested integer ratios / camera regimes (1..64); FIATLUX was not changed.")


def exact_frequency_indices(source, target, *, atol=2e-10, rtol=1e-10):
    """Locate actual samples, never interpolate, average or blindly stride.

    The absolute tolerance accommodates P3's documented 1e-10 cycles/m offset.
    Reject ambiguous matches, missing endpoints and nonmonotone coordinates.
    """
    source, target = np.asarray(source), np.asarray(target)
    if source.ndim != 1 or target.ndim != 1 or len(source) < 2 or len(target) < 2:
        raise TiptopSamplingError("Frequency coordinates must be one-dimensional arrays of size >= 2.")
    if not (np.isfinite(source).all() and np.isfinite(target).all() and
            np.all(np.diff(source)>0) and np.all(np.diff(target)>0)):
        raise TiptopSamplingError("Frequency coordinates must be finite and strictly increasing.")
    right = np.searchsorted(source, target)
    left = np.clip(right-1, 0, len(source)-1)
    right = np.clip(right, 0, len(source)-1)
    indices = np.where(abs(source[left]-target) <= abs(source[right]-target), left, right)
    if not np.allclose(source[indices], target, atol=atol, rtol=rtol) or len(np.unique(indices)) != len(target):
        raise TiptopSamplingError("FIATLUX frequencies are not an exact subset of P3; no interpolation allowed.")
    if np.min(np.diff(source)) <= 2*(atol+rtol*np.max(abs(target))):
        raise TiptopSamplingError("Frequency tolerance would allow ambiguous source samples.")
    return indices


def verify_tiptop_frequency_grid(grid, frequency):
    """Check integer step ratio, coverage and actual 2D P3 coordinates."""
    df, target = _target(grid)
    step = float(frequency.PSDstep)
    if not math.isfinite(step) or step <= 0:
        raise TiptopSamplingError("P3 PSDstep must be positive and finite.")
    q = df/step
    if round(q) < 1 or not math.isclose(q, round(q), rel_tol=1e-10, abs_tol=1e-10):
        raise TiptopSamplingError("Noninteger FIATLUX/P3 frequency-step ratio; no interpolation allowed.")
    n = int(frequency.nOtf)
    kx, ky = _cpu(frequency.kx_), _cpu(frequency.ky_)
    if kx.shape != (n,n) or ky.shape != (n,n):
        raise TiptopSamplingError("Unexpected P3 coordinate shapes.")
    if not (np.allclose(np.diff(kx[:, n//2]), step, rtol=1e-10, atol=1e-14)
            and np.allclose(np.diff(ky[n//2, :]), step, rtol=1e-10, atol=1e-14)):
        raise TiptopSamplingError("Actual P3 coordinate increments disagree with PSDstep.")
    ix = exact_frequency_indices(kx[:, n//2], target)
    iy = exact_frequency_indices(ky[n//2, :], target)
    if not (np.allclose(kx[np.ix_(ix,iy)], target[:,None], atol=2e-10, rtol=1e-10)
            and np.allclose(ky[np.ix_(ix,iy)], target[None,:], atol=2e-10, rtol=1e-10)):
        raise TiptopSamplingError("Actual P3 2D coordinates differ from the requested FIATLUX grid.")
    return ix, iy


@dataclass
class TiptopPSDResult:
    """Residual at each science direction, with the original FIATLUX grid."""
    grid: object
    power_nm2: torch.Tensor  # FIATLUX bin powers: sampled density * df_F² [nm²]
    frequency_step: float
    config: dict
    diagnostics: dict
    components_nm2: dict
    p3_frequency_x: np.ndarray
    p3_frequency_y: np.ndarray
    indices_x: np.ndarray
    indices_y: np.ndarray
    selected_p3_power_nm2: torch.Tensor

    @property
    def opd_psd(self):
        """Unshifted OPD density in m^4 = m²/(cycles/m)²."""
        return torch.fft.ifftshift(self.power_nm2, dim=(-2, -1)) * 1e-18 / self.frequency_step**2

    def atmosphere_model(self, source=0, *, reference_wavelength=500e-9,
                         symmetrize=False, seed=None):
        return TabulatedAtmosphereModel(self.grid, self.power_nm2[source],
            frequency_step=self.frequency_step, reference_wavelength=reference_wavelength,
            symmetrize=symmetrize, seed=seed)


def tiptop_psd(grid, *, config=None, overrides=None, sampling_ratio=None,
               pixel_scale=None, field_of_view=None, path_root=None):
    """Evaluate the HARMONI SCAO NGS residual PSD on an existing FIATLUX grid.

    `overrides` contains INI section/key dictionaries (D, zenith, wavelengths,
    atmosphere, WFS photons, etc.). HCM2 science magnitude is NOT a WFS flux:
    set sensor_HO.NumberPhotons using a calibrated guide-star photometric
    model. PixelScale/FieldOfView only control the P3 evaluator, never FIATLUX.
    sampling_ratio optionally requests an integer df_F/df_P; otherwise search.
    Incompatible coordinates fail explicitly, without any sampling override.
    Static maps are not random residual PSDs. Extra LO jitter is not modeled
    by this NGS SCAO adapter. No temporal evolution is implied by a PSD.
    """
    import tiptop
    from p3.aoSystem.aoSystem import aoSystem
    from p3.aoSystem.frequencyDomain import frequencyDomain
    from p3.aoSystem.fourierModel import fourierModel
    cfg = copy.deepcopy(load_harmoni_scao_config() if config is None else config)
    for section, values in (overrides or {}).items():
        cfg.setdefault(section, {}).update(copy.deepcopy(values))
    cfg = configure_tiptop_sampling(grid, cfg, sampling_ratio=sampling_ratio)
    sensor_overrides = (overrides or {}).get("sensor_science", {})
    pixel_scale = sensor_overrides.get("PixelScale") if pixel_scale is None else pixel_scale
    field_of_view = sensor_overrides.get("FieldOfView") if field_of_view is None else field_of_view
    if pixel_scale is not None:
        if not math.isfinite(float(pixel_scale)) or float(pixel_scale) <= 0:
            raise ValueError("PixelScale must be positive and finite.")
        cfg["sensor_science"]["PixelScale"] = float(pixel_scale)
    if field_of_view is not None:
        if isinstance(field_of_view, bool) or not isinstance(field_of_view, int) or field_of_view < 1:
            raise ValueError("FieldOfView must be a positive integer.")
        cfg["sensor_science"]["FieldOfView"] = field_of_view
    root = str(Path(tiptop.__file__).parent.parent) if path_root is None else str(path_root)
    ao = aoSystem(None, path_root=root, config_dict=cfg, psdExpansion=True, verbose=False)
    if ao.aoMode != "SCAO" or ao.ngs.nSrc != 1:
        raise ValueError("This adapter supports single-NGS SCAO only.")
    frequency = frequencyDomain(ao, computeFocalAnisoCov=False, dtype=ao.dtype)
    ix, iy = verify_tiptop_frequency_grid(grid, frequency)
    if frequency.nOtf < frequency.resAO:
        raise TiptopSamplingError("P3 auxiliary grid cannot contain its AO support; increase P3 FieldOfView without changing FIATLUX.")
    model = fourierModel(None, ao=ao, freq=frequency, calcPSF=False, display=False,
                        verbose=False, computeFocalAnisoCov=False,
                        getErrorBreakDown=True, reduce_memory=False)
    verify_tiptop_frequency_grid(grid, frequency)
    raw = _cpu(model.PSD)
    selected = raw[ix[:,None], iy[None,:], :].transpose(2,1,0).copy()
    df = 1/(grid.nx*grid.dx)
    dk = float(2*frequency.kcMax_/frequency.resAO)
    # Undo P3's integrated-bin factor, select density, integrate on FIATLUX.
    power = selected / dk**2 * df**2
    tensor = torch.as_tensor(power, dtype=grid.dtype, device=grid.device)
    if not torch.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError("P3 produced nonfinite or negative residual power.")
    diagnostics = dict(p3_version=version("astro-p3"), tiptop_version=version("astro-tiptop"),
        mode="exact frequency-grid extraction",
        nOtf=int(frequency.nOtf), target_N=grid.nx, PSDstep=float(frequency.PSDstep),
        target_df=1/(grid.nx*grid.dx), k_= _cpu(frequency.k_).tolist(),
        kRef_=int(frequency.kRef_), samp=_cpu(frequency.samp).tolist(),
        sampling_ratio=int(round(df/float(frequency.PSDstep))),
        selected_index_first=int(ix[0]), selected_index_last=int(ix[-1]),
        rms_nm=np.sqrt(power.sum((1, 2))).tolist(),
        full_p3_rms_nm=np.sqrt(raw.sum((0, 1))).tolist(),
        p3_normalization_dk=float(2*frequency.kcMax_/frequency.resAO),
        ao_cutoff_cycles_per_m=float(frequency.kcMax_))
    components = {}
    scale = (2 * float(frequency.kcMax_) / frequency.resAO * float(frequency.wvlRef) * 1e9 / (2 * math.pi))**2
    lo = int(np.ceil(frequency.nOtf / 2 - frequency.resAO / 2))
    hi = lo + frequency.resAO
    for name, attr in [("fitting", "psdFit"), ("noise", "psdNoise"),
                       ("aliasing", "psdAlias"), ("spatio_temporal", "psdSpatioTemporal"),
                       ("chromatism", "psdChromatism"), ("differential_refraction", "psdDiffRef")]:
        value = getattr(model, attr, None)
        if value is None:
            continue
        values = _cpu(value)
        if values.ndim == 2:
            values = values[:, :, None]
        full = np.zeros_like(raw)
        if values.shape[:2] == raw.shape[:2]:
            full[:] = values * scale
        elif values.shape[:2] == (frequency.resAO, frequency.resAO):
            full[lo:hi, lo:hi] = values * scale
        else:
            raise ValueError(f"Unexpected P3 component shape for {name}: {values.shape}")
        data = full[ix[:,None], iy[None,:], :].transpose(2,1,0).copy() / dk**2 * df**2
        components[name] = torch.as_tensor(data, dtype=grid.dtype, device=grid.device)
    diagnostics["component_sum_matches_total"] = bool(torch.allclose(sum(components.values()), tensor, rtol=1e-8, atol=1e-10)) if components else False
    return TiptopPSDResult(grid, tensor, df, cfg, diagnostics, components,
        _cpu(frequency.kx_)[:, int(frequency.nOtf)//2].copy(),
        _cpu(frequency.ky_)[int(frequency.nOtf)//2, :].copy(), ix, iy,
        torch.as_tensor(selected, dtype=grid.dtype, device=grid.device))
