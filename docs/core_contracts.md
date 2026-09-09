# Fiatlux core contracts

This document defines the invariants shared by the maintained Fiatlux API.
Code may rely on these rules; changes to them require tests, documentation, and
an explicit compatibility decision.

## Grid

`Grid(nx, ny, dx, dy)` describes a uniform Cartesian spatial sampling.

| Property | Contract |
|---|---|
| Array order | `(ny, nx)` |
| x direction | Last array axis; index increases with x |
| y direction | Penultimate array axis; index increases with y |
| Spacing | `dx`, `dy` in metres |
| Coordinates | `x[i] = (i - nx // 2) dx`; likewise for y |
| Optical origin | Array index `(ny // 2, nx // 2)` |
| Meshgrid return | `(x, y)`, each shaped `(ny, nx)` |
| Precision | Real `torch.float32` or `torch.float64` |
| Device | One explicit PyTorch device shared by coordinate tensors |

For even dimensions, the origin is a sample and the coordinate interval is
therefore asymmetric by one sample. This is intentional and matches the array
index convention used by the MFT.

`grid.to(device=..., dtype=...)` returns a new grid and does not mutate the
original.

## Spectrum

`Spectrum.wavelengths` and `Spectrum.fluxes` are one-dimensional tensors with
the same length, real dtype, device, and wavelength ordering. Wavelengths are
in metres. Fluxes are photon rates in photons/s assigned to each wavelength
channel; their sum is the source's total sampled photon rate.

## Field

A `Field` combines complex amplitude, a grid, and a spectrum.

| Property | Contract |
|---|---|
| Shape | `(n_wavelengths, ny, nx)` |
| Channel order | Identical to `spectrum.wavelengths` |
| Spatial amplitude units | `sqrt(photons / s / m²)` |
| `intensity()` units | `photons / s / m²` per wavelength channel |
| Integrated channel flux | `intensity()[k].sum() * dx * dy` |
| Device | Shared by amplitude, grid, wavelengths, and fluxes |
| Standard single-precision source | float32 metadata and complex64 amplitude |
| Standard double-precision source | float64 metadata and complex128 amplitude |

`field.to()` transfers the amplitude and all associated state coherently and
returns a new `Field`.

## OpticalElement

An `OpticalElement` is a same-plane transformation. By default it:

- accepts one `Field` and returns a new `Field`;
- preserves the grid and spectrum;
- preserves `(n_wavelengths, ny, nx)` shape;
- preserves device, precision, wavelength order, and field units;
- may modify amplitude and phase;
- rejects a field whose grid is incompatible with the element grid.

An exception must be explicit in the concrete element documentation. A change
of spatial sampling belongs to a propagator rather than a mask.

## Propagator

A `Propagator` maps a field between optical planes. It:

- preserves the spectrum and leading wavelength dimension;
- propagates every wavelength independently in the existing order;
- preserves device and the float/complex precision pairing;
- may replace the spatial grid and `(ny, nx)` shape;
- documents its input requirements and output grid.

For `MFTPropagator`, the input grid is `field.grid` and the output sampling is
`output_grid`. Its result shape is
`(n_wavelengths, output_grid.ny, output_grid.nx)`. Both grids must use the same
device and dtype. The implemented physical normalization preserves integrated
flux on conjugate lossless grids, as verified by the analytical tests.

`IdentityPropagator` changes no state and returns the same field object.

## Detector boundary

The detector grid must match the final field grid. A detector integrates the
spectral photon-rate density over wavelength channels, pixel area `dx * dy`,
and exposure time. Before optional digitization, its output is an electron
count per pixel.

## Runtime validation

Fiatlux rejects incompatible shapes, grids, devices, and precisions at the
closest public boundary. JSON configurations are additionally checked before
propagation so invalid optical trains fail with a path to the offending field.
