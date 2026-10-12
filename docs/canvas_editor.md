# FIATLUX 2D optical canvas

The separate `fiatlux_app` package keeps the existing standard-library Python
HTTP backend and native HTML/JavaScript/Canvas stack. No graphics dependency is
added to the scientific library. The application now opens a freely positioned
2D canvas. The previous linear editor and v1 API remain at `/linear` for backward
compatibility. The new editor imports validated v1 configurations into v2.

## Exact installation and launch

From this checkout, with Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m fiatlux_app
```

Windows activation: `.venv\Scripts\activate`. The installed entry point
`fiatlux-gui` is equivalent. To open the browser manually or change the port:

```bash
python -m fiatlux_app --port 8766 --no-browser
```

Open `http://127.0.0.1:8766`. The server binds loopback only. Ctrl+C waits for
active calculation to finish; cancellation of an executing optical transform is
not promised. No public deployment, hardware command, AO loops or arbitrary
configuration code is implemented.

## Canvas interactions

The initial example is already laid out:

`PlaneWave → CircularAperture → Kolmogorov → DM → MFT → Detector`

Phase, native DM OPD and field intensity views branch from the appropriate ports.
Drag components from the palette (click is an accessible fallback). Drag box
headers to place them freely, including to the left of their predecessors.
**Only connections define the optical order**. Drag an output port to an input
port, or click output then input. Typed ports are Field, OPD and Counts. One
scientific successor and one scientific predecessor maximum: fan-out is allowed
only into visualization observers. Cycles, duplicate scientific inputs,
incompatible grids and wrong port types report precise node/connection errors.
A disconnected scientific draft is preserved but does not execute.

Click a box for its parameters; hover its header for physical parameters, grid,
spectrum and computation state. Drag empty canvas space to pan. Mouse wheel or
± zooms around the pointer/view center; **Tout voir** frames all boxes. Delete a
selected box or selected wire with the button or Delete key. All positions,
presentation labels, sizes, viewport, connections and local visualization settings
save with the scientific configuration. Placement and presentation never launch a
calculation. The browser keeps a best-effort local draft; full JSON save requires
a valid graph.

## Observers and visualization

A Visualization node has no scientific output. Multiple observers may read the
same cached Field or OPD. Counts exposes the real detector acquisition, already
integrated over wavelength. It cannot provide a wavelength-resolved phase.
Field intensity is `abs(E)**2`; the polychromatic intensity sums channel
intensities already weighted by the source convention. Phase uses the selected
wavelength's native `Field.phase()`, not a fabricated polychromatic phase. OPD
is offered only by a real native object's OPD tensor and is snapshotted at its
calculation revision. DM coefficients stay **metres of optical path difference**,
not mechanical displacement.

Each visual box contains the actual map, colorbar, physical x/y coordinates and
units. Local colormap, lower/upper limits, linear/log intensity scale, wavelength
and OPD unit do not propagate anything. Map enlargement opens a larger view.
The logarithmic display clips non-positive values only for rendering with
`max(I)*1e-12` as a floor; underlying arrays remain unchanged. CPU/NumPy conversion
occurs solely at the visualization transport boundary.

If a native result has latent dimensions, the inspector lists each dimension and
its size/unit; select one index explicitly per dimension. No implicit time
integration or progression is performed. The current palette's sources/screens
produce ordinary spectral fields; this UI does not introduce a temporal FATMOSS
backend simply to fabricate latent samples.

## Automatic revisions and caches

Scientific numeric/text input and DM sliders debounce for 350 ms. Changing a
scientific parameter increments the client revision immediately, marks the
changed node and downstream scientific nodes stale, and submits a complete
validated graph in the background. Invalid input supersedes outstanding work,
retains the last visible maps, displays the error, and never executes. Unchanged
upstream fields and operators remain valid. A one-worker queue coalesces pending
changes per browser session. Intermediate calculations that cannot be stopped may
finish and warm correctly keyed caches, but publication is guarded by the exact
latest revision. Old status/image responses are also guarded in the browser.
At most eight sessions and sixteen atmospheric screen cache entries are retained.

The incremental engine compiles connected scientific nodes into a native linear
chain. Each applied element still uses `SerialSystem.run_field()` and the real
FIATLUX constructors; no optical equations are replaced. Per-node cache keys
include local parameters, referenced physical grids, precision and the upstream
scientific key. Spectrum is part of the source key and therefore invalidates all
scientific descendants. A changed DM reuses source/pupil/atmosphere snapshots and
rebuilds only DM and downstream responses. A changed focal grid leaves the pupil
chain untouched and reconstructs the propagator/detector. Existing masks cache
chromatic transmissions and the DM builds its transmission on application;
changed local configuration reconstructs affected operators rather than mutating
an already published snapshot. Published Field/count/OPD snapshots cannot be
changed through a later operator mutation.

## Reproducible stochastic policy

Atmosphere uses the native Kolmogorov OPD sampler. Its screen identity includes
r0, reference wavelength, outer scale, seed, realization counter, physical grid
and precision. The effective native seed is
`(seed + 104729 * realization) % 2147483648`. **Nouvelle réalisation** increments
only that atmosphere's saved counter. Changing atmospheric statistics/grid/dtype
reconstructs a deterministic screen with the saved effective seed. Returning to
the same identity reproduces it even after cache eviction. Changing illumination,
DM commands, colormap or position reuses the same screen exactly.

Native `Mask.build()` ordinarily rebuilds OPD when the spectrum changes. The app's
small `FrozenAtmosphere` adapter overrides only `_build_opd` to return the saved
native sample; transmission and propagation remain the core implementations.
This avoids an implicit new realization during deterministic recalculation.
No FATMOSS timestep is advanced: FATMOSS is not a selectable component here.

## Version 2 document

`format: "fiatlux-canvas"`, `version: 2` contains name, precision, spectrum and
named grids, plus three independent sections:

- `nodes`: `{id, type, params}`; Visualization parameters are display-only.
- `connections`: `{from: {node, port}, to: {node, port}}`.
- `layout`: positions `{id: {x,y,width,height,label}}` and viewport `{x,y,zoom}`.

Node array order and x/y positions have no scientific meaning. Configuration is
JSON data only, with no pickle, executable expressions or arbitrary imports.
Unknown ports, components and fields are rejected. Scientific limits from v1
remain (CPU, 256 pixels/axis, 8 wavelengths, 36 DM modes), with up to 40 total
nodes and 64 connections. Current native Zernike remains square/isotropic;
FFT remains monochromatic with an explicit matching downstream grid. Fresnel
retains its existing distance/focal-length contract. Complete example:
`fiatlux_app/canvas_examples/04_canvas.json`.

## Validation and captures

```bash
python -m pip install -e '.[all,dev]'
python -m pytest -q
python -m pip install playwright
python -m playwright install chromium
python scripts/canvas_smoke.py canvas-captures
```

The browser smoke launches the real localhost server, verifies palette drop,
free placement, typed wiring/refusal, DM edits and upstream identity reuse,
local-only display changes, explicit atmospheric refresh, errors preserving maps,
save/reload of wiring/poses, zoom, tooltips, enlargement, selection/delete and
mobile layout. It records screenshots from genuine FIATLUX results. Unit tests
compare connected outputs with native direct Python chains, validate cached
operator reconstruction, latent selection, exact atmospheric preservation and
latest-revision publication under controlled overlapping work. Existing v1,
core-import, full pytest and clean-kernel tutorial CI checks remain active.
