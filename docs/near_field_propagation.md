# Fourier propagation contract

This document separates two independent choices in Fiatlux propagation:

1. the numerical Fourier transform: matrix Fourier transform (MFT) or FFT;
2. the physical approximation: Fraunhofer or Fresnel.

All four combinations are supported by the design. Fraunhofer is the default
regime for both numerical methods. The existing `MFTPropagator` Fraunhofer
behavior remains backward compatible.

The executable companion notebook
`tutorials/08_near_field_propagation_contract.ipynb` evaluates the four cases
and checks MFT/FFT agreement on their common sampling.

## Public API

The public API is:

```python
MFTPropagator(focal_length=f, output_grid=grid)
MFTPropagator(output_grid=grid, propagation="fresnel", distance=z)

FFTPropagator(focal_length=f)
FFTPropagator(propagation="fresnel", distance=z)
```

`propagation` accepts the public `PropagationRegime` values `FRAUNHOFER` and
`FRESNEL`, as well as their lower-case string values. Omitting it selects
Fraunhofer. Implementations must not infer the regime from the transform
method.

## Transform methods and output sampling

MFT evaluates the Fourier integral on an explicitly supplied output grid. It
is appropriate for a cropped focal region, deliberate oversampling, and
rectangular input or output planes.

FFT evaluates the same Fourier integral on its natural conjugate grid. For an
input with `nx`, `dx` and propagation scale `q`, its output sampling is

\[
    dx_2 = \frac{\lambda q}{n_x dx_1}, \qquad
    dy_2 = \frac{\lambda q}{n_y dy_1}.
\]

Here `q` is the focal length for Fraunhofer propagation through a lens and the
propagation distance for the single-transform Fresnel formulation. The FFT
propagator constructs this grid explicitly and never silently resamples it.

Because the natural FFT grid depends on wavelength and `Field` currently owns
one spatial grid, `FFTPropagator` accepts monochromatic fields. A polychromatic
request raises `PropagationSamplingError` and points users to `MFTPropagator`
for propagation onto a common physical grid.

## Coordinates and Fourier convention

Fields have shape `(n_wavelengths, ny, nx)`. The x coordinate is the last axis
and y is the penultimate axis. Coordinates are in metres and follow `Grid`'s
centred convention.

Fiatlux uses

\[
    \mathcal E(x,y,z,t)=E(x,y,z)e^{-i\omega t}
\]

and the continuous Fourier transform

\[
 \widehat E(f_x,f_y)=\iint E(x,y)
 e^{-i2\pi(f_xx+f_yy)}\,dx\,dy.
\]

A wave travelling towards positive z accumulates positive propagation phase.
FFT implementations must handle the centred grid explicitly with shifts.

## Fraunhofer regime (default)

Fraunhofer propagation evaluates the Fourier transform of the input field:

\[
 E_2(x_2,y_2) \propto
 \widehat E_1\!\left(
   \frac{x_2}{\lambda q},
   \frac{y_2}{\lambda q}
 \right).
\]

The MFT version evaluates this expression on `output_grid`. The FFT version
evaluates it on the natural conjugate grid. Fiatlux's existing focal-plane MFT
phase and normalization convention remains the backward-compatible reference.

## Fresnel regime

For signed non-zero distance z, the single-transform Fresnel formulation is

\[
 E_2(x_2,y_2)=\frac{e^{ikz}}{i\lambda z}
 e^{\frac{ik}{2z}(x_2^2+y_2^2)}
 \mathcal F\!\left\{
 E_1(x_1,y_1)e^{\frac{ik}{2z}(x_1^2+y_1^2)}
 \right\}_{f_x=x_2/(\lambda z),\,f_y=y_2/(\lambda z)}.
\]

MFT and FFT differ only in how this Fourier transform is evaluated. Both must
use the same input and output quadratic phases, global phase, physical
normalization, and sign convention.

The MFT version accepts an explicit output grid. The FFT version uses its
natural Fresnel output grid. Zero distance returns an exact equivalent field
without evaluating the singular formula. Non-finite distance and a Fresnel
request without distance are rejected.

## Units, dtype, device, and gradients

Spatial-plane complex amplitude uses `sqrt(photons / s / m²)`. Correctly
sampled lossless propagation preserves

\[
    \sum |E|^2\,dx\,dy.
\]

- float32 real quantities produce complex64 fields;
- float64 real quantities produce complex128 fields;
- tensors are created on the field device;
- no NumPy conversion occurs in the propagation path;
- wavelength order and spectrum metadata are preserved;
- gradients propagate through the input complex amplitude.

## Validation matrix

Production implementations are accepted only after tests cover:

1. the four MFT/FFT × Fraunhofer/Fresnel combinations;
2. Fraunhofer as the default for both methods;
3. MFT/FFT complex-field agreement on the natural FFT grid;
4. backward compatibility of existing Fraunhofer MFT results;
5. Fresnel Gaussian-beam radius and curvature;
6. integrated-flux conservation;
7. signed distance and zero-distance semantics;
8. rectangular and odd/even grids;
9. mono- and polychromatic sampling behavior;
10. complex64/complex128, CPU/CUDA, and autograd;
11. invalid distance and incompatible-grid error paths.
