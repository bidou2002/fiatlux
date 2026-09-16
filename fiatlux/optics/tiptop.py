"""Optional P3 adapter: FIATLUX owns sampling; P3 only evaluates the PSD.

Validated against astro-tiptop 1.5.1 / astro-p3 1.6.2. No PSD interpolation.
P3 fourierModel.powerSpectrumDensity returns centered OPD **bin power** in
nm²: its final multiplier is (dk * wavelength_nm / (2*pi))², where
dk = 2*kcMax_/resAO (not necessarily PSDstep, because resAO is truncated).
Thus canonical phase density is bin_power * (2*pi*1e-9/lambda)² / df².
We preserve P3's bin powers; we never fit an RMS or multiply by dk again.
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


def configure_tiptop_sampling(grid, config, *, explicit_sampling=False):
    """Return a deep copy of configuration; never modify grid or input config.

    Search piecewise-constant P3 oversampling regimes. The minimum PSD step
    can be supplied by any wavelength, not necessarily the shortest one.
    An explicit user PixelScale can instead be passed to tiptop_psd.
    """
    df, _ = _target(grid)
    cfg = copy.deepcopy(config)
    wavelengths = np.unique(np.atleast_1d(cfg["sources_science"]["Wavelength"]).astype(float))
    if not np.isfinite(wavelengths).all() or (wavelengths <= 0).any():
        raise ValueError("Science wavelengths must be finite and positive.")
    sensor = cfg["sensor_science"]
    if sensor.get("SpectralBandwidth", 0) != 0 or len(sensor.get("Transmittance", [1])) != 1:
        raise ValueError("Supply explicit science wavelengths with zero sensor bandwidth and one transmittance bin.")
    diameter = float(cfg["telescope"]["TelescopeDiameter"])
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("TelescopeDiameter must be positive and finite.")
    # For every channel, k=ceil(2/samp) implies PSDstep <= 1/(2D).
    if df > 1 / (2 * diameter) * (1 + 1e-12) and not explicit_sampling:
        raise TiptopSamplingError(
            f"FIATLUX df={df:.12g} cycles/m exceeds P3's configuration limit "
            f"1/(2D)={1/(2*diameter):.12g}. No PixelScale/FieldOfView pair can match. "
            "Keep FIATLUX unchanged; explicitly request explicit_sampling=True to use the pinned P3 frequency-domain adapter.")
    candidates = []
    for factor in range(1, 65):
        for wavelength in wavelengths:
            pixel = df * wavelength * RAD2MAS * factor
            step, kref, _, _ = _sampling(pixel, wavelengths, diameter)
            if math.isclose(step, df, rel_tol=1e-11, abs_tol=1e-14):
                n = math.ceil(grid.nx / kref) * kref
                candidates.append((n, factor, pixel, kref))
    if candidates:
        _, _, pixel, kref = min(candidates)
    elif explicit_sampling:
        # This nominal camera configuration only initializes P3 metadata.
        # The opt-in frequency-domain adapter supplies df and N directly.
        pixel = df * wavelengths.min() * RAD2MAS
        _, kref, _, _ = _sampling(pixel, wavelengths, diameter)
    else:
        raise TiptopSamplingError("No exact camera sampling found in P3 oversampling factors 1..64; FIATLUX was not changed.")
    sensor["PixelScale"] = float(pixel)
    sensor["FieldOfView"] = math.ceil(grid.nx / kref)
    return cfg


def verify_tiptop_frequency_grid(grid, frequency):
    """Verify actual P3 coordinates and return an exact centered crop slice.

    P3 adds a documented 1e-10 cycles/m offset to avoid singularities. Only
    this numerical offset (2e-10 absolute tolerance) is accepted at DC.
    Parity-aware integer slices align DC even when input/output sizes differ.
    """
    df, target = _target(grid)
    n = int(frequency.nOtf)
    if n < grid.nx or not math.isclose(float(frequency.PSDstep), df, rel_tol=1e-10, abs_tol=1e-14):
        raise TiptopSamplingError(f"P3 nOtf={n}, PSDstep={float(frequency.PSDstep):.12g}; FIATLUX N={grid.nx}, df={df:.12g}. No interpolation allowed.")
    start = n // 2 - grid.nx // 2
    crop = slice(start, start + grid.nx)
    kx, ky = _cpu(frequency.kx_), _cpu(frequency.ky_)
    if kx.shape != (n, n) or ky.shape != (n, n):
        raise TiptopSamplingError("Unexpected P3 coordinate shapes.")
    if not (np.allclose(kx[crop, crop], target[:, None], rtol=1e-10, atol=2e-10)
            and np.allclose(ky[crop, crop], target[None, :], rtol=1e-10, atol=2e-10)):
        raise TiptopSamplingError("Actual P3 kx_/ky_ samples do not match FIATLUX; no interpolation allowed.")
    return crop


def _explicit_frequency_domain(ao, grid):
    """Small P3-side sampling hook through fourierModel's public freq= API.

    P3 1.6.2's constructor assigns PSDstep and nOtf before constructing any
    coordinates, AO support, masks, filters or covariances. Local properties
    replace only those two assignments. All subsequent quantities are built
    by P3 from the requested sampling. No global monkeypatch or post-hoc
    coordinate replacement. This adapter is PSD-only, not a TIPTOP PSF API.
    The ideal upstream change is two optional constructor keywords at those
    assignment sites; the version gate prevents silently relying on new code.
    """
    if version("astro-p3") != "1.6.2":
        raise TiptopSamplingError("Explicit P3 sampling is validated only for astro-p3==1.6.2.")
    from p3.aoSystem.frequencyDomain import frequencyDomain
    df, _ = _target(grid)

    class TargetFrequencyDomain(frequencyDomain):
        @property
        def PSDstep(self):
            return self._target_step

        @PSDstep.setter
        def PSDstep(self, value):
            self._target_step = df

        @property
        def nOtf(self):
            return self._target_size

        @nOtf.setter
        def nOtf(self, value):
            self._target_size = grid.nx
            # Pupil/OTF sampling for this explicitly supplied frequency grid.
            self.sampRef = 1 / (float(ao.tel.D) * df)

    return TargetFrequencyDomain(ao, computeFocalAnisoCov=False, dtype=ao.dtype)


@dataclass
class TiptopPSDResult:
    """Residual at each science direction, with the original FIATLUX grid."""
    grid: object
    power_nm2: torch.Tensor  # centered (source,y,x), before optional symmetry
    frequency_step: float
    config: dict
    diagnostics: dict
    components_nm2: dict

    @property
    def opd_psd(self):
        """Unshifted OPD density in m^4 = m²/(cycles/m)²."""
        return torch.fft.ifftshift(self.power_nm2, dim=(-2, -1)) * 1e-18 / self.frequency_step**2

    def atmosphere_model(self, source=0, *, reference_wavelength=500e-9,
                         symmetrize=False, seed=None):
        return TabulatedAtmosphereModel(self.grid, self.power_nm2[source],
            frequency_step=self.frequency_step, reference_wavelength=reference_wavelength,
            symmetrize=symmetrize, seed=seed)


def tiptop_psd(grid, *, config=None, overrides=None, explicit_sampling=False,
               pixel_scale=None, field_of_view=None, path_root=None):
    """Evaluate the HARMONI SCAO NGS residual PSD on an existing FIATLUX grid.

    `overrides` contains INI section/key dictionaries (D, zenith, wavelengths,
    atmosphere, WFS photons, etc.). HCM2 science magnitude is NOT a WFS flux:
    set sensor_HO.NumberPhotons using a calibrated guide-star photometric
    model. PixelScale/FieldOfView only control the P3 evaluator, never FIATLUX.
    Config-only incompatibility fails unless explicit_sampling is opted in.
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
    cfg = configure_tiptop_sampling(grid, cfg, explicit_sampling=explicit_sampling)
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
    frequency = (_explicit_frequency_domain(ao, grid) if explicit_sampling else
                 frequencyDomain(ao, computeFocalAnisoCov=False, dtype=ao.dtype))
    crop = verify_tiptop_frequency_grid(grid, frequency)
    if frequency.nOtf < frequency.resAO:
        raise TiptopSamplingError("P3 AO support exceeds target frequency coverage; increase P3 FieldOfView for a same-df crop, not the FIATLUX grid.")
    model = fourierModel(None, ao=ao, freq=frequency, calcPSF=False, display=False,
                        verbose=False, computeFocalAnisoCov=False,
                        getErrorBreakDown=True, reduce_memory=False)
    verify_tiptop_frequency_grid(grid, frequency)
    raw = _cpu(model.PSD)
    power = np.moveaxis(raw[crop, crop, :], -1, 0).transpose(0, 2, 1).copy()
    tensor = torch.as_tensor(power, dtype=grid.dtype, device=grid.device)
    if not torch.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError("P3 produced nonfinite or negative residual power.")
    diagnostics = dict(p3_version=version("astro-p3"), tiptop_version=version("astro-tiptop"),
        mode="explicit P3 frequency domain" if explicit_sampling else "camera configuration",
        nOtf=int(frequency.nOtf), target_N=grid.nx, PSDstep=float(frequency.PSDstep),
        target_df=1/(grid.nx*grid.dx), k_= _cpu(frequency.k_).tolist(),
        kRef_=int(frequency.kRef_), samp=_cpu(frequency.samp).tolist(),
        crop_start=crop.start, crop_stop=crop.stop,
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
        data = full[crop, crop].transpose(2, 1, 0).copy()
        components[name] = torch.as_tensor(data, dtype=grid.dtype, device=grid.device)
    diagnostics["component_sum_matches_total"] = bool(torch.allclose(sum(components.values()), tensor, rtol=1e-8, atol=1e-10)) if components else False
    return TiptopPSDResult(grid, tensor, float(frequency.PSDstep), cfg, diagnostics, components)
