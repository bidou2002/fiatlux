# TIPTOP residuals on an existing FIATLUX grid

The simulation owns its grid. This adapter never changes its shape, pixel
pitch, physical extent, padding, or optical sampling. P3 evaluates a residual
PSD on compatible Fourier samples. No PSD interpolation is permitted.

Install the optional, tested versions with `pip install '.[tiptop,tutorials]'`.
The core import does not import TIPTOP, P3, scipy, astropy or matplotlib.

```python
from fiatlux import tiptop_psd

# pupil_grid and spectrum already belong to the simulation.
result = tiptop_psd(
    pupil_grid,
    overrides={
        "telescope": {"TelescopeDiameter": 38.542, "ZenithAngle": 52.4},
        "sources_science": {"Wavelength": spectrum.wavelengths.tolist()},
    },
)
model = result.atmosphere_model(symmetrize=True, seed=2026)
assert model.grid is pupil_grid
opd = model.sample_opd_many(32)  # metres, independent realizations
```

## Sampling contract and configuration limits

`df = 1/(grid.nx*grid.dx)` is derived from the computational extent, not
the telescope diameter. The adapter searches wavelength-dependent P3
oversampling regimes for camera PixelScale/FieldOfView, then verifies actual
`PSDstep`, `nOtf`, `kx_` and `ky_`. Only an exact centered subset of a larger
same-df grid is accepted. Cropping loses out-of-band power without rescaling.
Square isotropic grids are supported; other grids fail explicitly.

P3 1.6.2 computes `k=ceil(2/samp)`, hence `PSDstep <= 1/(2D)` in camera
configuration mode. In particular an unpadded extent D cannot be matched by
any PixelScale/FieldOfView pair. Polychromatic sampling has additional integer
constraints. The finite search (oversampling 1..64) reports failure rather
than assuming every mathematically possible solution was exhausted.

For incompatible configurations, **explicitly** opt into
`tiptop_psd(pupil_grid, ..., explicit_sampling=True)`. The isolated P3-side
adapter is a local `frequencyDomain` subclass passed via `fourierModel(freq=)`.
It replaces the constructor's PSDstep and nOtf assignments before P3 creates
coordinates, AO masks, piston filters and support. The physical grid in
FIATLUX is untouched. No PSD is computed on one grid and resampled onto another.
This path is gated to P3 1.6.2 because those initialization details are
version-dependent. An upstream constructor extension with two optional
sampling keywords would remove the need for the subclass. Camera k_/kRef_
metadata retain their upstream values; nOtf/PSDstep define the explicitly
injected evaluator. Do not use this PSD-only adapter as a TIPTOP PSF backend.

P3's 1e-10 cycles/m coordinate offset is tolerated (atol 2e-10); frequency
increments are checked much more tightly. An explicit incompatible camera
override still fails verification. The original Grid object is returned.

## Physical normalization

P3 `fourierModel.powerSpectrumDensity` returns centered OPD **bin powers in
nm²**, despite its legacy density docstring. The final factor is
`(dk*wavelength_nm/(2*pi))**2`, with `dk=2*kcMax_/resAO`. Integer resAO rounding
means dk is not necessarily PSDstep. Preserve P3's integrated bin powers.

The adapter transposes P3 `(x,y,source)` to `(source,y,x)`. `result.opd_psd`
is unshifted density in m⁴: `bin_power*1e-18/df**2`. The atmosphere model uses
phase density in rad² m²: `OPD_density*(2*pi/reference_wavelength)**2`.
The identity `variance=PSD.sum()*df**2` is tested. There is no target RMS.

Real screens require inversion symmetry. Asymmetric spectra fail unless
`symmetrize=True` is explicitly supplied. Pair averaging conserves total power;
the raw P3 output remains available. This is not interpolation. Full-grid
variance is distinct from pupil-weighted, piston-subtracted variance.

## Physical assumptions

- The default physical preset is loaded from installed TIPTOP's
  `perfTest/HARMONI_SCAO.ini`, the NGS preset used by the referenced ORION page.
  P3's separately shipped `HarmoniSCAO.ini` differs (profile, WFS flux, gain,
  LO sections, asset paths); they are not silently interchanged.
- Section/key overrides expose telescope, atmosphere, science channels and
  WFS parameters. A user-supplied config can be loaded separately. Relative
  asset paths resolve against `path_root` (installed TIPTOP root by default).
- Science magnitude 14 is not a WFS photon calibration. Override
  `sensor_HO.NumberPhotons` using guide-star band/throughput/frame calibration;
  retaining the preset means 500 photons/subaperture/frame.
  For a calibrated reference in the same guide-star band, use
  `photons = reference_photons * 10**(-0.4*(guide_magnitude-reference_magnitude))`
  and pass `overrides={"sensor_HO": {"NumberPhotons": [photons]}}`.
- Static OPD is not part of the atmospheric/AO residual. The adapter supports
  single-NGS SCAO; it does not add TIPTOP's separate low-order jitter kernels.
- P3 evaluates wavelength-dependent residual terms at its shortest reference
  science wavelength. The resulting OPD ensemble is shared by FIATLUX spectral
  channels; this is not a wavelength-specific chromatic residual covariance model.
- A spatial PSD does not specify a time sequence. These draws are independent;
  500 Hz is the AO loop frequency, not a screen correlation prescription.

See [tutorial 13](../tutorials/13_tiptop_residual_psds.ipynb) for equations,
padding examples, exact-grid checks, components, images and Monte-Carlo data.
