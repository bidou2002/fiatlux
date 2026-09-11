# Fiatlux 2.0 tutorial notebooks

Progressive tutorial notebooks for the `fiatlux2.0` API.

1. `00_getting_started.ipynb` — Grid, Spectrum, PlaneWave, aperture, MFT, PSF
2. `01_aberrations.ipynb` — Differential piston and PSF degradation
3. `02_polychromatic_psf.ipynb` — Multi-wavelength propagation
4. `03_deformable_mirror.ipynb` — Zernike-controlled deformable mirror
5. `04_zelda.ipynb` — ZELDA phase-mask wavefront sensor
6. `05_interaction_matrix_modal.ipynb` — Modal push-pull calibration and SVD
7. `06_elt_pupil_from_harmoni.ipynb` — Analytical ELT/HARMONI pupil validation
8. `07_closed_loop_ao.ipynb` — Closed-loop adaptive optics with ZELDA
9. `08_near_field_propagation_contract.ipynb` — Near-field propagation contract
10. `09_fresnel_physical_validation.ipynb` — Physical Fresnel validation
11. `10_fatmoss_temporal_turbulence.ipynb` — FATMOSS temporal turbulence
12. `11_shack_hartmann_end_to_end.ipynb` — Shack-Hartmann calibrated closed loop

These notebooks are written against the current `fiatlux2.0` API and may expose
the core issues identified during code review, particularly monochromatic
spectrum handling, dimensional conventions, detector normalization, and DM API
consistency.
