# Named latent dimensions

`Field` stores `(*latent, wavelength, ny, nx)`. Wavelength stays special and is
represented by `Spectrum` at axis `-3`. Spatial transforms use only `(-2,-1)`.
Sources still produce ordinary three-dimensional fields.

## Construct and propagate a time buffer

```python
import torch
from fiatlux import FieldDimension, SerialSystem

time = FieldDimension(
    'time', len(opd),
    coordinates=torch.arange(len(opd), device=opd.device, dtype=opd.dtype) * dt,
    unit='s',
    integration_weights=torch.full((len(opd),), dt, device=opd.device, dtype=opd.dtype),
)
buffer = source_field.apply_opd(opd, dimensions=(time,))
propagated = system.run_field(buffer).final_field
```

`opd` is in metres, shaped `(time,ny,nx)`. No propagation loop is required.
An OPD map without latent axes has shape `(ny,nx)`. OPD descriptors may include
existing axes in a different order; matching names must have identical sizes,
coordinates, units and weights. Existing field axes retain their order, and
new axes are appended before wavelength. Independent sources with different
grids or spectra must be represented by separate fields.

`FieldDimension` is a frozen descriptor. Its input tensors are cloned; callers
must also treat its tensor attributes as read-only (PyTorch tensors themselves
are mutable). Names must be unique and nonempty; `wavelength`, `x`, and `y`
are reserved. Computational coordinates and weights use the amplitude's real
precision and device. Latent amplitudes must be complex64/complex128, paired
with float32/float64 grids and spectra. Legacy real-valued 3-D fields remain
accepted for detector compatibility. `to(dtype=torch.float64)` preserves the
complex amplitude using complex128; it never discards the imaginary part.

Available operations:

- `dimension(name)` and `axis(name)` look up latent metadata/positions.
- `select(name,index)` removes one axis; negative indices work.
- `slice(name,start,stop)` follows Python slicing and slices coordinates/weights.
- `rename_dimension(old,new)` returns new metadata.
- `expand_dimension(descriptor)` explicitly adds a broadcast axis.
- `coherent_sum(name)` and `coherent_mean(name)` reduce **complex amplitudes**.

Field addition/subtraction requires identical grids, spectra and dimension
metadata. Scalar/tensor arithmetic cannot introduce or resize an axis. There
is no implicit name alignment for field arithmetic and no ambiguous `mean`.
Plotting requires selecting all latent dimensions first.

## Atmosphere sampling

`AtmosphereModel.sample_opd_many(number)` synthesizes independent screens with
batched spatial FFTs. Existing `sample_opd` is unchanged. Seed replay is
reproducible for the same draw shape; PyTorch does not guarantee identical
random draws across batch partitions/devices. Freeze the OPD cube when comparing
loop and batched propagation.

`Atmosphere.apply_buffer(field, number, dimension='time', sample_period=dt,
advance=True)` creates a regular time descriptor in seconds with zero-order-hold
weights `dt`. Independent models require `sample_period`; `advance=False`
restores their generator state. Their coordinates start at zero per call because
they do not own a physical clock. For absolute independent-sample timestamps,
construct the descriptor explicitly and use `apply_opd`.

FATMOSS uses `sequence_opd`, starts coordinates at its selected timestep, and
uses its configured cadence. A supplied period must match that cadence. It
retains `sequence_opd`'s explicit advance and exception semantics. `advance=False`
restores the selected time; ordinary propagation never advances a buffered
atmosphere. Real FATMOSS is optional; the tests inject a translating backend.
For irregular times, provide explicit integration weights rather than inferring
an exposure from timestamps.

## Detector exposure and chunking

```python
from fiatlux import Detector, ExposureAccumulator

detector = Detector(propagated.grid, exposure_time=T * dt)
image = detector.acquire(propagated, integrate_over='time')

accumulator = ExposureAccumulator(detector, integrate_over='time', track_grad=False)
for chunk in propagated_chunks:
    accumulator.add(chunk)
image = accumulator.finish()
```

Named acquisition sums spectral intensities, weights and sums the requested
axis, and multiplies pixel area once. Weights must be finite, nonnegative,
nonempty and in seconds (`unit='s'`), with positive total duration matching
`exposure_time`. They replace the exposure-time multiplier. QE, Poisson photon
noise, dark noise, read noise and digitization run once at readout.

Without `integrate_over`, every latent sample is an independent exposure of
`exposure_time`. Remaining axes produce a `DetectorImage(data, dimensions)`;
a fully reduced image remains the legacy 2-D tensor. `image_buffer` carries the
same return object. Calibration callbacks must explicitly reduce remaining
latent dimensions before returning a measurement tensor.

The accumulator checks remaining metadata, spectra, device, dtype and exposure
duration. Supply non-overlapping chunks; it does not infer coverage from their
timestamps. Empty, incomplete, overlong or repeated-finish exposures fail
clearly. Detector settings must not change mid-exposure.

`track_grad=False` detaches chunk totals for bounded accumulator memory. Release
your own references to propagated fields as well. `track_grad=True` preserves
the graph across chunks, so memory grows with exposure history. Retaining full
autograd over arbitrarily long exposures is not constant-memory. The benchmark
also reports forward+backward separately and retains the whole graph in that mode.

## Supported optics and limits

Static masks, ZELDA, static DM, MFT (Fraunhofer/Fresnel), FFT
(Fraunhofer/Fresnel), and identity propagation preserve latent metadata. MFT
maps wavelength at `-3`; each wavelength's matrices batch all leading samples.
FFT keeps its monochromatic physical-grid restriction. Fraunhofer is still
the default. Time-dependent DM commands are not implemented.

Shack–Hartmann explicitly rejects latent fields at `propagate`; its specialized
lenslet image, detector and slope containers retain their existing shapes.
Select samples before using that sensor. Its future extension must propagate
metadata through all three products. No Pyramid or dedicated coronagraph class
is introduced.

## Validation and benchmarking

Run `python -m pytest` and the clean-kernel notebooks as described in
[tutorials.md](tutorials.md). Tutorial 12 freezes one OPD cube and checks complex
fields, explicit expected counts and chunked counts without renormalization.
Float32 BLAS summation order can differ between batch shapes; tests use an
explicit eight-epsilon absolute/relative roundoff bound, while scientific
validation runs in float64 and reports raw errors.

`tests/test_atmosphere_buffer.py` captures the exact OPD cube returned to
`Atmosphere.apply_buffer` and uses those samples for the scalar reference.
Small float32/float64 cases compare complex fields and noiseless integrated
counts for monochromatic FFT, polychromatic MFT, Fresnel MFT and a static DM.
Propagation cases use rectangular grids; the Zernike DM case uses a square
grid because the current `ZernikeBasis` constructs square modes.
Diagnostics report maximum absolute, pointwise relative and RMS errors without
renormalization. Existing tests also use opposite-phase samples to distinguish
incoherent integration from coherent averaging.

The correlated-sequence test uses the existing injected translating backend
with a seeded OPD scale. It verifies scalar/buffer sample identity, correlation,
reset/replay, nonzero start time, both advancement modes and chunk partitions.
The same test optionally exercises real FATMOSS when `phase_generator` is
importable (see [installation](fatmoss.md#installation)); otherwise only that
parameter is skipped. Run it with:

```bash
python -m pytest -q -s tests/test_atmosphere_buffer.py
```

Tutorial 12 displays buffer shapes, OPD samples, a long-exposure image and raw
error metrics. It can be executed independently if an earlier tutorial fails:

```bash
jupyter nbconvert --to notebook --execute tutorials/12_named_latent_dimensions.ipynb --output-dir /tmp/fiatlux-validation
```

```bash
python benchmarks/latent_dimensions.py --output latent-benchmark.json
```

The default matrix covers sizes 128/256/400, T=1/4/16/64/256, one/three
wavelengths, both methods/regimes and chunks 1/4/8/16/32/64. CPU measurements
use `torch.utils.benchmark`; CUDA uses events with synchronization and reports
peak allocated/reserved memory. Sampling is excluded. Input cubes and reference
images remain resident during measurement, so memory figures include them.
Use command-line selectors for smaller runs. Speedups are measurements, not
performance guarantees.
