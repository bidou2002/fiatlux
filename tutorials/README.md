# Fiatlux 2.0 tutorial notebooks

Progressive tutorial notebooks for the `fiatlux2.0` API.

1. `00_getting_started.ipynb` — Grid, Spectrum, PlaneWave, aperture, MFT, PSF
2. `01_aberrations.ipynb` — Differential piston and PSF degradation
3. `02_polychromatic_psf.ipynb` — Multi-wavelength propagation
4. `03_deformable_mirror.ipynb` — Zernike-controlled deformable mirror
5. `04_zelda.ipynb` — ZELDA phase-mask wavefront sensor
6. `05_interaction_matrix.ipynb` — Push-pull calibration and SVD control matrix

These notebooks are written against the current `fiatlux2.0` API and may expose
the core issues identified during code review, particularly monochromatic
spectrum handling, dimensional conventions, detector normalization, and DM API
consistency.
