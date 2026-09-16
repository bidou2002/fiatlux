# TIPTOP residuals through exact frequency-grid extraction

**FIATLUX defines the simulation grid. P3 may use a finer and larger auxiliary
frequency grid. The adapter selects the exact P3 samples required by FIATLUX,
without PSD interpolation, neighboring-cell averaging, empirical normalization,
or modification of the FIATLUX grid.**

Install optional tested dependencies: `pip install '.[tiptop,tutorials]'`.
Core imports remain independent of TIPTOP/P3, scipy, astropy and matplotlib.

```python
from fiatlux import tiptop_psd

# Reuse the existing simulation objects, including any existing padding.
result = tiptop_psd(
    pupil_grid,
    overrides={
        "telescope": {"TelescopeDiameter": 38.542, "ZenithAngle": 52.4},
        "sources_science": {"Wavelength": spectrum.wavelengths.tolist()},
    },
)
assert result.grid is pupil_grid
model = result.atmosphere_model(symmetrize=True, seed=2026)
opd = model.sample_opd_many(32)  # metres, independent realizations
```

## Auxiliary grid contract

Derive `df_F=1/(grid.nx*grid.dx)` and centered `fftfreq` coordinates from the
actual computational extent, not telescope diameter. P3 camera configuration
is chosen so `df_F/PSDstep` is an integer q, and its auxiliary array covers both
FIATLUX frequency endpoints. An optional `sampling_ratio=q` requests a specific
integer ratio. Search otherwise starts at 1, up to 64, over P3 oversampling
regimes 1..64. Finite-search failure is explicit; no simulation grid is changed.

The camera formula `k=ceil(2/samp)` implies `PSDstep <= 1/(2D)`. This is NOT an
incompatibility with pupil-only MFT support D: for example, P3 can use step
1/(2D), approximately 2N samples, and FIATLUX can select the exact 1/D samples.
Polychromatic k_ and kRef_ are included in the search; q=2 is not assumed.
Only standard P3 configuration is used. No frequency-domain subclass, sampling
assignment hook or backend monkeypatch is required.

After P3 constructs its domain, the adapter verifies actual PSDstep, nOtf, kx_
and ky_. `exact_frequency_indices` locates every target coordinate, checks
coverage, uniqueness and coordinate equality. This handles even/odd sizes,
DC centering and integer ratios. The 1e-10 cycles/m offset used by P3 is
accepted with absolute tolerance 2e-10. Ambiguous matches fail. Noninteger
ratios, insufficient extent or shifted coordinates fail explicitly.

A centered **crop** only reduces extent; it retains df_P. **Exact frequency-grid
extraction** can additionally select an integer-stride subset and yield df_F.
Index matching, not a blind `[::2, ::2]`, determines the selection. The result
exposes source coordinate vectors and selected indices for independent checks.

## Density and normalization

P3 1.6.2 `fourierModel.powerSpectrumDensity` returns centered OPD bin powers B_P
in nm², despite its legacy density docstring. Its last multiplier is
`(dk*wavelength_nm/(2*pi))**2`, with `dk=2*kcMax_/resAO`. Integer rounding of
resAO means dk is not exactly PSDstep.

To recover the evaluated continuous density, **undo the actual dk² factor**:

```
W_h = B_P * 1e-18 / dk**2              # OPD density in m⁴
W_F = W_h[exact FIATLUX indices]       # existing samples only
variance_F = W_F.sum() * df_F**2       # m², FIATLUX quadrature
```

No neighboring cells are averaged. In particular, neither keeping selected
P3 bin powers unchanged nor dividing them by df_F² is the correct conversion
when source and target grids differ. There is no target RMS and no empirical
rescaling. P3 full-grid RMS and FIATLUX RMS are different quadratures and need
not be equal, even over similar support.

P3 `(x,y,source)` becomes FIATLUX `(source,y,x)`. Result fields:

- `opd_psd`: unshifted density W_F in m⁴.
- `selected_p3_power_nm2`: exact raw P3 bin powers at selected indices.
- `power_nm2`: centered W_F times FIATLUX bin area, expressed in nm².
- `components_nm2`: the same conversion for exposed error components.
- `frequency_step`: FIATLUX df; diagnostics contain P3 PSDstep and actual dk.

The screen model converts OPD density to phase density in rad² m² using
`(2*pi/reference_wavelength)**2`. Independent real screens require inversion
symmetry. Raw extracted samples are never averaged; optional `symmetrize=True`
is a separate screen-generation choice that averages opposite frequencies,
not neighboring 2x2 cells. It preserves the target-grid integrated power and
reports the original asymmetry. Full-grid variance is not pupil-weighted,
piston-subtracted variance.

## Physical assumptions and scope

The installed TIPTOP `perfTest/HARMONI_SCAO.ini` supplies the NGS preset referenced
by ORION. P3's separately shipped `HarmoniSCAO.ini` is a different preset and is
not silently substituted. Section/key overrides expose telescope, atmosphere,
science channels and WFS configuration. Optional PixelScale/FieldOfView overrides
are checked, not silently replaced. Custom asset paths resolve against path_root.

Science magnitude 14 is not a guide-star flux calibration. The default remains
500 photons/subaperture/frame. With a calibrated reference in the same WFS band,
use `photons=reference_photons*10**(-0.4*(guide_magnitude-reference_magnitude))`
and override `sensor_HO.NumberPhotons` explicitly.

Square isotropic grids and single-NGS SCAO are supported. Versions are pinned
for reproducible normalization. Static OPD and separate LO jitter are excluded.
P3 may internally interpolate static pupil/OTF inputs; the residual PSD is never
interpolated. Tests forbid interpolation functions on the extraction path.
P3 evaluates chromatic residual terms at its shortest reference science wavelength;
the resulting OPD ensemble is shared across FIATLUX spectral channels, not a
wavelength-specific chromatic covariance model. Independent screens do not imply
a physical time sequence; the 500 Hz loop rate is not a correlation prescription.

[Tutorial 13](../tutorials/13_tiptop_residual_psds.ipynb) includes the pupil-only
q=2 case, padded/MFT grid comparison, exact-subset diagram, component maps and
Monte-Carlo verification of the FIATLUX variance integral.
