# FIATLUX local optical-chain editor

The application is a separate `fiatlux_app` package. It runs a loopback-only
Python HTTP server and a browser interface using native HTML, JavaScript and
Canvas. No graphical dependency, frontend build, account, CDN or public service
is required. Importing `fiatlux` does not import the app.

## Install and launch

From the repository root, with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m fiatlux_app
```

On Windows, activate using `.venv\Scripts\activate` instead. The browser opens
`http://127.0.0.1:8765`. The equivalent installed command is `fiatlux-gui`.
To select a different port or open the browser yourself:

```bash
python -m fiatlux_app --port 8766 --no-browser
```

Stop with Ctrl+C. A running calculation is allowed to finish; this first version
does not promise cancellation. Bindings to public interfaces are deliberately
not available. The app is a local workstation tool, not a production web server.

## Build and inspect a chain

The initial circular-pupil example is ready to run. Select one of three saved
examples, add components from the library, drag blocks or use the Before/After
buttons, and edit parameters in the side panel. The source must be first and the
detector last. Multiple sources, multiple detectors and branching are rejected.

Grids and spectrum are editable in **Grilles & spectre**. Every spatial spacing
and wavelength is expressed in metres, exposure in seconds, and phase in
radians. Parameters are never repaired automatically. Errors include a block
identifier. Edit the configuration and run again without losing it.

**Vérifier** validates the data-only configuration, including grid-chain and
spectral compatibility; **Lancer** additionally constructs the actual scientific
objects and reports constructor or propagation errors. Calculations use a single
background worker, leaving the HTTP/UI thread responsive. Additional accidental
launches are rejected across browser tabs. Configuration edits mark previous
results as stale. A failed run keeps the configuration and last displayed map.
Only the most recent successful/active run is available for stage inspection.

Select a stage and a wavelength to inspect spectral intensity or phase. The
polychromatic intensity is `Field.intensity().sum(-3)`: channel flux is already
encoded in `PlaneWave` amplitude and must not be multiplied twice. There is no
polychromatic phase. Detector views display acquired electrons/pixel or ADU/pixel,
not field intensity. Masks and the DM expose OPD where the engine provides it.
The visualization boundary converts detached CPU tensors to NumPy/JSON;
construction and propagation stay in PyTorch.

The map uses x on the last axis, y on the penultimate axis and a lower origin.
Hover to read actual physical coordinates and values. Linear/log intensity
scaling changes display only. The logarithmic display floor is `max(I)*1e-12`
(with a positive representability floor), and does not modify calculated data.
OPD may be displayed in metres or nanometres. Commands are metres of optical
path difference, **not mechanical displacement**.

## Supported components and actual engine constraints

- PlaneWave with an explicit Band/Spectrum, magnitude and spectral samples.
- CircularAperture and analytical ELTHarmoniPupil, without external FITS files.
- KolmogorovAtmosphereModel wrapped by Atmosphere: one deterministic seeded
  screen per run, optional outer scale and piston removal using the core defaults.
- DeformableMirror with the native ZernikeBasis, Noll ordering excluding piston.
  The number of modes is the number of supplied OPD coefficients. ActuatorGrid
  is only the required constructor metadata for this modal basis, not a zonal
  actuator simulation. Coefficients beyond stroke are rejected before the core
  can silently saturate them.
- MFTPropagator and FFTPropagator. Fraunhofer is default. Fresnel requires explicit
  distance and an empty focal-length field; zero distance requires the same grid.
  Signed distance/focal length retain the core propagation convention.
- ZeldaMask, ZeldaStop and the Step aberration used in the existing ZELDA tutorial.
- Detector with exposure, QE, seeded noise, conversion gain, offset and ADC.

The inspected baseline is `main` at `a63dc8775bf1ed852b426c1207b65312eb7c8080`.
The legacy `dev_2.0` branch is not the API baseline. On this main, ZernikeBasis
still creates square arrays with its native coordinate convention; this app
therefore explicitly rejects rectangular/anisotropic DM grids and does not
change the scientific basis. Other components accept rectangular grids. FFT
requires one spectral channel and produces its natural physical grid:
`dx_out = wavelength * abs(f_or_z)/(nx * dx_in)` and similarly for y. Define a
matching downstream grid explicitly; the app does not replace an incompatible
chosen grid. MFT supports multiple wavelengths on a common explicit output grid.

The UI intentionally exposes a useful subset of ELT geometry parameters. Remaining
constructor defaults (spider angles, center, petal rotation, inactive segment list)
remain documented FIATLUX defaults; arbitrary Python and optional data backends
are not configuration features. The examples are modest CPU demonstrations
(64–96 pixel pupils), not quantitatively resolved ELT segment-gap calculations.
No CUDA claim, hardware control, AO loop, FATMOSS requirement, branches or cycles.

## Saved format

**Enregistrer** downloads a validated UTF-8 JSON file. **Ouvrir JSON** validates
before replacing the working configuration. The browser also keeps a best-effort
local draft; it does not send anything to external services. An invalid draft is
not automatically restored on reload. No pickle, dynamic imports, expression
execution or arbitrary code is accepted. Duplicate keys and NaN/infinity are
rejected. Unknown versions, components, fields and grid references fail explicitly.

Version 1 consists of:

```json
{
  "format": "fiatlux-chain",
  "version": 1,
  "name": "My optical chain",
  "precision": "float64",
  "spectrum": {
    "central_wavelength": 0.00000165,
    "delta_wavelength": 0,
    "f0": 368000000,
    "magnitude": 0,
    "samples": 1
  },
  "grids": {
    "pupil": {"nx": 64, "ny": 64, "dx": 0.015625, "dy": 0.015625},
    "focal": {"nx": 80, "ny": 80, "dx": 0.000004125, "dy": 0.000004125}
  },
  "blocks": []
}
```

`blocks` is an ordered array of `{id, type, params}`; all fields shown by the
component catalog are serialized. The skeleton above needs its source, optical
blocks and detector before it can run. Complete executable files are in
`fiatlux_app/examples/01_circular.json`, `02_elt_dm.json`, `03_zelda.json`.
The spectrum `f0` uses the core Band convention:
`photon_flux = f0 / 368 * 10**(-magnitude/2.5)` shared over `samples` channels.
Grids share the saved real dtype, on CPU. Seeds, DM commands, order and every
exposed scientific parameter are preserved. Fixed-seed same-version CPU runs
reproduce the same system. Cross-version/platform bitwise identity is not promised.

Version 1 CPU limits: 24 blocks, 8 grids, 2–256 pixels per axis, 8 spectral channels,
36 DM modes, and 1 MB input JSON. These bounds keep the first local editor modest;
they are not limits of the scientific library.

The existing `fiatlux.config` loader uses fixed pupil/focal grids, a registry and
reference/expression resolution. It cannot represent this editor's explicit
ordered block array, arbitrary named grids, custom spectral band and Atmosphere
wrapper directly. The app therefore uses a strict, data-only model and constructor
allowlist, while reusing public classes, SerialSystem propagation and the existing
grid-chain validator where its contract applies. It never dispatches user code
through the expression resolver. The scientific library is unchanged.

## Verification

```bash
python -m pip install -e '.[all,dev]'
python -m pytest -q
python -m pip install playwright
python -m playwright install chromium
python scripts/gui_smoke.py gui-captures
```

The browser smoke starts an actual localhost server, runs all three optical
examples, checks nonzero acquired images, phase/OPD views, JSON save/reload,
stale results, invalid parameter recovery, reorder/add/delete, and mobile layout.
It writes screenshots and saved example files. Tests independently construct the
three example systems using public FIATLUX objects and compare complex fields
and detector arrays numerically. They also cover spectral sums, native phase,
DM units, seeded turbulence, Fresnel identity, FFT sampling, concurrency and
configuration snapshots. `Local GUI` CI publishes the screenshots as the
`fiatlux-gui-captures` artifact. Existing Package CI continues to verify the full
core tests, isolated import and clean-kernel tutorials.
