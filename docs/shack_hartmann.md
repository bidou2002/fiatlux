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
