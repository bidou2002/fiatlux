# Shack-Hartmann optical model

`ShackHartmannLensletArray` separates optical spot formation from centroid and
slope extraction. It consumes a Fiatlux `Field` and returns a
`ShackHartmannImage`; measurement algorithms are deliberately handled by the
next layer of the sensor API.

```python
from fiatlux import Grid, ShackHartmannLensletArray

grid = Grid(240, 240, 0.01, 0.01)
lenslets = ShackHartmannLensletArray(
    grid,
    pitch=0.20,
    focal_length=5e-3,
    n_lenslets_x=12,
    n_lenslets_y=12,
)
spots = lenslets.propagate(pupil_field)
detector_flux = spots.mosaic()  # photons / s / detector pixel
```

## Tensor and coordinate conventions

The complex focal field has shape
`(n_wavelengths, n_lenslets_y, n_lenslets_x, spot_ny, spot_nx)`. Lenslet
indices increase in the same y/x order as Fiatlux images. `registration_x` and
`registration_y` shift the centered array in metres and currently must be
integer multiples of the corresponding pupil-grid spacing.

The hard subaperture windows contain `pitch / dx` by `pitch / dy` samples.
These ratios must be integers. A Boolean `valid_subapertures` array can disable
known invalid lenslets; automatic partial-illumination decisions belong to the
dedicated illumination/noise layer.

For calibrated subpixel centroiding, `spot_oversampling` zero-pads each
subaperture before its focal FFT. For example, a 4×4 subaperture with
`spot_oversampling=4` produces a 16×16 spot with four times finer physical
sampling on each axis. The physical FFT normalization continues to conserve
photon flux. The default is one for backward-compatible native sampling.

## Propagation and sampling

For a thin lens of focal length `f`, its quadratic phase followed by Fresnel
propagation over `f` reduces to a Fraunhofer transform of the windowed
subaperture, up to the output quadratic phase. This is the batched transform
implemented by `propagate()`. `lenslet_phase(wavelengths)` exposes the actual
local thin-lens phase

\[
\phi_\mathrm{lens}(x,y;\lambda) =
-\frac{\pi}{\lambda f}(x^2+y^2).
\]

Natural detector sampling depends on wavelength:

\[
\Delta u_\lambda = \frac{\lambda f}{N_x\Delta x},\qquad
\Delta v_\lambda = \frac{\lambda f}{N_y\Delta y}.
\]

It is stored per channel in `pixel_scale_x` and `pixel_scale_y`. Consequently,
spectral intensity densities are not blindly added on a common physical grid.
`pixel_flux` first integrates each channel over its own pixel area, and
`mosaic()` then returns photon rate per detector pixel. The FFT normalization
obeys Parseval, so total output photon rate equals the rate inside enabled
subapertures without empirical renormalization.

All lenslets and wavelength channels are transformed in batches. Dtype, device
and autograd are preserved. Fiatlux `Field` currently has no additional
arbitrary leading batch dimension.

## Centroids and calibrated slopes

`ShackHartmannSlopeEstimator` converts a `ShackHartmannImage` into a
`ShackHartmannMeasurement`. It supports a rectangular centroid window and a
relative threshold (a fraction of each spectral spot peak). For polychromatic
data, it computes centroids using each wavelength's physical detector sampling
before combining them with photon-flux weights.

```python
from fiatlux import ShackHartmannSlopeEstimator

estimator = ShackHartmannSlopeEstimator(
    focal_length=lenslets.focal_length,
    window_radius=2,
    threshold=0.05,
    reference_centroids=reference_centroids,
)
measurement = estimator.measure(spots)
```

`measurement.centroids[..., 0]` and `[..., 1]` are detector positions in
metres, ordered `(x, y)`. The reference-subtracted `measurement.slopes` use the
same ordering and are small angles in radians. `measurement.slope_vector`
contains only valid subapertures, ordered as all x slopes in row-major lenslet
order followed by all y slopes in the same order. Empty or explicitly disabled
subapertures return zero and are false in `valid_subapertures`.

## Partial illumination and detector noise

Pass the pupil's amplitude transmission to the lenslet array to compute the
mean transmitted-power fraction of every registered subaperture. A minimum
fraction masks edge, obscured or inter-segment subapertures before propagation:

```python
lenslets = ShackHartmannLensletArray(
    grid,
    pitch=0.20,
    focal_length=5e-3,
    pupil_transmission=pupil_mask,
    minimum_illumination=0.5,
)
```

The fractions are carried by `ShackHartmannImage` and returned as
`measurement.weights`. Invalid subapertures are omitted from `slope_vector`;
the full slope map remains available with zero values at invalid positions.

`ShackHartmannDetector` converts ideal spectral photon-rate densities into a
single physical detector exposure:

```python
from fiatlux import ShackHartmannDetector

detector = ShackHartmannDetector(
    exposure_time=1e-3,          # s
    pixel_scale_x=15e-6,         # m / pixel
    quantum_efficiency=0.9,
    photon_noise=True,
    dark_current=0.01,           # electrons / pixel / s
    read_noise=1.0,              # electrons RMS / pixel / exposure
    full_well=80_000,            # electrons / pixel
    minimum_electrons=100,       # electrons / subaperture
    random_seed=1,
    device=grid.device,
)
frame = detector.expose(spots)
measurement = estimator.measure(frame)
```

Before spectral integration, each wavelength is bilinearly sampled on the
common detector grid because its natural focal-plane scale differs. Photon and
dark counts use Poisson statistics; read noise is Gaussian. Pixels are clipped
to the full-well capacity. A subaperture is invalid if it is optically masked,
below `minimum_electrons`, or contains a saturated pixel. The frame reports
both `valid_subapertures` and `saturated_subapertures` explicitly. Random state
is private to the detector and reproducible from `random_seed`.

## End-to-end calibration

The maintained `tutorials/11_shack_hartmann_end_to_end.ipynb` notebook uses the
generic `InteractionMatrix` with `measurement.slope_vector`. It calibrates tip,
tilt and defocus by push–pull, filters the SVD pseudo-inverse, reconstructs
known modal coefficients and demonstrates closed-loop convergence. The
notebook executes from a clean kernel in CI.
