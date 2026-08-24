<p align="center">
  <img src="logos/fiatlux_logo.png" alt="Fiatlux logo" width="520">
</p>

<h1 align="center">Fiatlux</h1>

<p align="center">
  <strong>A PyTorch-based framework for Fourier optics and adaptive-optics simulations.</strong>
</p>

<p align="center">
  Build optical systems from reusable components, propagate monochromatic or polychromatic fields,
  model wavefront aberrations and correction, and inspect the field anywhere in the optical train.
</p>

---

## What is Fiatlux?

**Fiatlux** is a research-oriented Python package for numerical optical simulations.

The goal is to provide a modular environment in which an optical system can be assembled from simple, reusable objects:

```text
Source
  ↓
Field
  ↓
Aperture / Atmosphere / Phase mask / Deformable mirror / ...
  ↓
Propagator
  ↓
Detector
```

Instead of writing a new propagation script for every experiment, Fiatlux separates the simulation into a few core concepts:

- **`Grid`** — sampling of an optical plane
- **`Spectrum`** — wavelength channels and photometric flux
- **`Source`** — generation of the input optical field
- **`Field`** — complex electric field propagated through the system
- **`OpticalElement`** — pupil masks, phase masks, aberrations, deformable mirrors, etc.
- **`Propagator`** — propagation between optical planes
- **`Detector`** — image formation and detector effects
- **`SerialSystem`** — ordered optical train
- **`SimulationResult`** — access to intermediate fields throughout a simulation

Fiatlux is built around **PyTorch tensors**, allowing optical simulations to integrate naturally with modern numerical optimization and machine-learning workflows.

> [!NOTE]
> The `fiatlux2.0` branch is under active development. The API and physical normalization conventions are still being consolidated.

---

## Main capabilities

### Fourier optics

Fiatlux provides matrix Fourier transform propagation between arbitrarily sampled pupil and focal planes.

```python
from fiatlux.optics.propagator import MFTPropagator
```

Optical systems can mix propagation and optical elements in a single ordered sequence.

### Monochromatic and polychromatic fields

A `Spectrum` defines the wavelength channels propagated through the system.

```python
from fiatlux.core.spectrum import PhotometricBand, Spectrum

spectrum = Spectrum(
    magnitude=8,
    band=PhotometricBand.H,
    samples=5,
)
```

Each wavelength is propagated independently while remaining part of the same `Field`.

### Apertures and optical masks

Available mask-like elements include:

- circular apertures
- arbitrary pupil transmission maps
- piston
- differential piston / step
- tip and tilt
- ZELDA phase masks
- ZELDA stops
- atmospheric phase screens
- atmospheric-dispersion correction terms

Custom telescope pupils can be provided directly as arrays using `ArbitraryAperture`.

### Deformable mirrors

Fiatlux includes a deformable-mirror model with several control bases, including:

- Gaussian zonal influence functions
- square zonal influence functions
- piston-tip-tilt zonal modes
- Fourier modes
- Zernike modes

This makes it possible to simulate both modal and actuator-based wavefront correction.

### Adaptive optics

The package contains tools for AO-oriented simulations, including:

- deformable mirrors
- wavefront-sensor optical models
- detector acquisition
- interaction-matrix calibration
- control-matrix computation
- atmospheric and residual phase screens

Fiatlux is therefore intended not only for image formation, but also for developing and testing wavefront-sensing and correction strategies.

### Intermediate-field inspection

`SerialSystem.run()` returns a `SimulationResult`, making it possible to inspect the field after any element:

```python
result = system.run(source)

field_after_aperture = result.field_at(aperture)
focal_field = result.field_at(propagator)
```

This is useful for understanding and debugging complex optical trains.

---

## Quick start

### Installation

Clone the repository and install Fiatlux in editable mode:

```bash
git clone https://github.com/bidou2002/fiatlux.git
cd fiatlux
git switch fiatlux2.0

python -m pip install -e .
```

For development, editable installation is recommended so changes to the source code are immediately available in Python and Jupyter.

---

## Minimal example — circular-aperture PSF

The following example creates a pupil, propagates it to a focal plane, and displays the resulting PSF.

```python
import matplotlib.pyplot as plt

from fiatlux.core.grid import Grid
from fiatlux.core.spectrum import PhotometricBand, Spectrum
from fiatlux.core.source import PlaneWave
from fiatlux.optics.elements.mask import CircularAperture
from fiatlux.optics.propagator import MFTPropagator
from fiatlux.system.optical_system import SerialSystem


# Telescope
D = 1.0
N_pupil = 256

pupil_grid = Grid(
    nx=N_pupil,
    ny=N_pupil,
    dx=D / N_pupil,
    dy=D / N_pupil,
)

# Source
spectrum = Spectrum(
    magnitude=0,
    band=PhotometricBand.H,
    samples=3,
)

source = PlaneWave(spectrum)

# Pupil
aperture = CircularAperture(
    grid=pupil_grid,
    radius=D / 2,
)

# Focal plane
wavelength = PhotometricBand.H.central_wavelength
focal_length = 1.0
N_focal = 256
pixels_per_lambda_D = 4

focal_grid = Grid(
    nx=N_focal,
    ny=N_focal,
    dx=focal_length * wavelength / D / pixels_per_lambda_D,
    dy=focal_length * wavelength / D / pixels_per_lambda_D,
)

propagator = MFTPropagator(
    focal_length=focal_length,
    output_grid=focal_grid,
)

# Optical system
system = SerialSystem(
    elements=[
        aperture,
        propagator,
    ]
)

result = system.run(source)

# Broadband PSF
psf = result.field_at(propagator).intensity().sum(dim=0)
psf = psf / psf.max()

plt.imshow(psf.cpu(), origin="lower")
plt.title("Circular-aperture PSF")
plt.colorbar(label="Normalized intensity")
plt.show()
```

---

## Building an optical system

An optical train is simply an ordered list of elements:

```python
system = SerialSystem(
    elements=[
        entrance_pupil,
        atmosphere,
        deformable_mirror,
        pupil_to_focal,
        phase_mask,
        focal_to_pupil,
    ]
)
```

The same architecture can represent a simple telescope PSF calculation or a more complete adaptive-optics wavefront-sensor simulation.

---

## ELT and HARMONI simulations

Fiatlux can use real telescope pupil geometries through `ArbitraryAperture`.

For example, when HARMONI residual phase-screen data are available, an ELT pupil can be extracted from the support of the residual maps:

```python
atm_res = HarmoniResiduals(grid=pupil_grid)

pupil_mask = (
    atm_res.datacube[0] != 0
).to(torch.float32)

elt_pupil = ArbitraryAperture(
    grid=pupil_grid,
    transmission=pupil_mask,
)
```

> [!IMPORTANT]
> Examples based on `HarmoniResiduals` require the corresponding residual phase-screen FITS data to be available locally.

---

## Tutorials

Progressive notebooks are available in [`tutorials/`](tutorials/).

A recommended learning path is:

| Notebook | Topic |
|---|---|
| `00_getting_started.ipynb` | First grid, source, aperture and PSF |
| `01_aberrations.ipynb` | Piston and wavefront aberrations |
| `02_polychromatic_psf.ipynb` | Multi-wavelength propagation |
| `03_deformable_mirror.ipynb` | DM control and modal aberrations |
| `04_zelda.ipynb` | ZELDA phase-mask wavefront sensing |
| `05_interaction_matrix.ipynb` | Push-pull calibration and control matrix |
| `06_elt_pupil_from_harmoni.ipynb` | ELT pupil from HARMONI residual data |

---

## Package structure

```text
fiatlux/
├── core/
│   ├── field.py
│   ├── grid.py
│   ├── source.py
│   └── spectrum.py
│
├── optics/
│   ├── elements/
│   ├── adc.py
│   ├── atmosphere.py
│   ├── detector.py
│   └── propagator.py
│
├── system/
│   ├── optical_system.py
│   └── interaction_matrix.py
│
├── config/
└── utils/
```

The separation between **physical fields**, **optical components**, and **system orchestration** is central to the Fiatlux 2.0 architecture.

---

## Development status

Fiatlux is currently a **research and development project**.

Current work on Fiatlux 2.0 includes:

- consolidating field shape and coordinate conventions
- defining consistent physical flux normalization
- improving CPU/GPU and dtype handling
- validating propagators against analytical Fourier-optics results
- strengthening detector and deformable-mirror models
- adding automated tests
- expanding the tutorial suite
- improving packaging and configuration

The GitHub issue tracker is used to follow these developments.

---

## Contributing

Bug reports, validation cases, feature requests and contributions are welcome.

When contributing numerical or optical functionality, please consider including:

- a minimal reproducible example
- the physical convention being used
- a regression or analytical validation test where possible

---

## Authors

Fiatlux was initiated by:

- **Pierre Janin-Potiron**
- **Yoann Brûlé**
- **Olivier Fauvarque**

See the Git history for the complete list of contributors.

---

## License

Fiatlux is distributed under the terms of the [MIT License](LICENSE).
