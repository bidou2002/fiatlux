# Near-field propagation contract

This document fixes the physical, numerical, and public API conventions for
finite-distance scalar propagation in Fiatlux. Implementations and tests must
follow this contract.

This feature is additive. The existing `MFTPropagator` remains Fiatlux's
default pupil-to-focal Fraunhofer propagator, with unchanged behavior and API.
The term Fresnel below selects a finite-distance physical model; it does not
replace or redefine Fraunhofer propagation.

The executable companion notebook
`tutorials/08_near_field_propagation_contract.ipynb` visualizes these choices
and checks flux conservation, Gaussian-beam spreading, and
the Fresnel sampling bound.

## Scope

The first implementation supports homogeneous free-space propagation between
parallel planes with unchanged transverse sampling. It provides:

- a paraxial Fresnel transfer-function propagator;
- independent propagation of every wavelength channel;
- rectangular grids, PyTorch autograd, and CPU/CUDA execution.

Fresnel propagation by MFT, tilted planes, refractive media, vector fields,
and non-uniform sampling are outside this first contract. A later Fresnel-MFT
contract will allow an explicitly chosen output grid without conflating the
physical Fresnel model with its numerical implementation.

The intended propagation family is therefore:

- `MFTPropagator`: existing and default Fraunhofer propagation;
- `FresnelPropagator`: first finite-distance implementation, using FFTs on an
  unchanged grid;
- a future explicitly named Fresnel-MFT propagator for arbitrary output
  sampling.

## Coordinates and sign convention

Fields use the existing Fiatlux array convention
`(n_wavelengths, ny, nx)`. The x coordinate is the last array axis and y is
the penultimate axis. Both are expressed in metres and follow `Grid`'s
centred coordinate convention.

Fiatlux adopts the time convention

\[
    \mathcal{E}(x,y,z,t) = E(x,y,z)\,e^{-i\omega t}.
\]

A plane wave travelling towards positive z therefore accumulates
`exp(+i k_z z)`. A positive propagation distance moves the field forward
along positive z; a negative distance performs backward propagation. A zero
distance is an exact identity.

The continuous transverse Fourier transform is

\[
 \widehat E(f_x,f_y)
 = \iint E(x,y)\,
   e^{-i2\pi(f_xx+f_yy)}\,dx\,dy,
\]

with spatial frequencies in cycles per metre. Implementations using
`torch.fft` must account explicitly for the centred spatial grid with
`ifftshift` before the forward FFT and `fftshift` after the inverse FFT.

## Field normalization and units

Spatial-plane complex amplitude retains the Fiatlux unit
`sqrt(photons / s / m²)`. Lossless propagation preserves

\[
    \sum |E|^2\,dx\,dy
\]

up to numerical precision. Transfer-function propagation therefore uses
unitary discrete FFT normalization (`norm="ortho"`) and does not introduce
an empirical amplitude factor.

The absolute carrier phase `exp(i 2π z / wavelength)` is retained. Tests that
only concern intensity may ignore a spatially constant phase, but complex
field comparisons may not silently remove it.

## Wavelength, dtype, device, and gradients

Each spectral channel uses its own wavelength and transfer function while
preserving wavelength order and spectral flux metadata.

- float32 grids and spectra produce complex64 fields;
- float64 grids and spectra produce complex128 fields;
- all intermediate tensors are created on the field device;
- no NumPy conversion is allowed in the numerical path;
- gradients must propagate through the input complex amplitude and distance
  whenever the distance is represented by a tensor in a future extension.

The initial public API accepts distance as a Python real number. Trainable
distance is explicitly deferred rather than supported accidentally.

## Output-grid contract

The initial near-field propagator is a same-grid Fresnel transfer-function
method implemented with FFTs.
Input and output have identical `nx`, `ny`, `dx`, `dy`, device, and
real dtype. The propagator is constructed with that grid and rejects a field
on any other grid.

This restriction is deliberate: arbitrary output sampling requires a scaled
Fresnel transform with a different normalization and aliasing contract. Such a
transform must be introduced as a separate propagator, not as an implicit
resampling option.

## Fresnel transfer-function method

The paraxial transfer function is

\[
 H_\mathrm{F}(f_x,f_y;z)
 = e^{i2\pi z/\lambda}
   e^{-i\pi\lambda z(f_x^2+f_y^2)}.
\]

The planned constructor is:

```python
FresnelPropagator(
    distance: float,
    grid: Grid,
    *,
    check_sampling: bool = True,
)
```

The method assumes paraxial content
`lambda² (f_x² + f_y²) << 1`. With sampling checks enabled, the sampled
transfer-function phase must not change by more than π between adjacent
frequency bins at either Nyquist edge. The conservative axis-wise condition is

\[
 |z| \leq
 \min\!\left(
   \frac{n_x\,dx^2}{\lambda},
   \frac{n_y\,dy^2}{\lambda}
 \right).
\]

Violation raises `PropagationSamplingError` with the wavelength and limiting
distance. It is never converted silently into a warning or hidden resampling.
`check_sampling=False` is an explicit expert override.

## Shared behavior

The class inherits `NearFieldPropagator(distance, grid)`, exposes
`output_grid == grid`, and implement `apply(field) -> Field`.

They must:

- reject non-finite distances;
- reject incompatible field grids with expected and actual sampling details;
- return an exact equivalent field at zero distance;
- preserve shape, spectrum object semantics, dtype, device, and differentiability;
- cache only tensors whose validity includes grid, wavelength, dtype, device,
  distance, and method options.

Method-specific invalid sampling raises `PropagationSamplingError`, a
subclass of `ValueError`.

## Validation matrix

The implementation is accepted only after quantitative tests cover:

1. Gaussian-beam radius and curvature versus propagation distance;
2. energy conservation;
3. forward then backward recovery for a correctly sampled field;
4. Gaussian-beam complex field agreement, including radius and curvature;
5. convergence towards Fraunhofer behavior at long distance using the
   appropriate far-field sampling;
6. rectangular and odd/even grids;
7. mono- and polychromatic fields;
8. complex64/complex128 and CPU/CUDA;
9. autograd through the input field;
10. every Fresnel sampling error path.
