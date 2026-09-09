import torch

from fiatlux.core.grid import Grid
from fiatlux.core.source import PlaneWave
from fiatlux.core.spectrum import Band, Spectrum
from fiatlux.optics.elements.mask import CircularAperture, Piston, TipTilt
from fiatlux.optics.propagator import MFTPropagator


WAVELENGTH = 1.0e-6
FOCAL_LENGTH = 2.0
DIAMETER = 1.0


def monochromatic_field(grid):
    spectrum = Spectrum(
        magnitude=0,
        band=Band(WAVELENGTH, 0.0, 1.0e8),
        samples=1,
        dtype=grid.dtype,
    )
    return PlaneWave(spectrum).generate_field(grid)


def circular_pupil_field(n_pixels=128):
    grid = Grid(
        n_pixels,
        n_pixels,
        DIAMETER / n_pixels,
        DIAMETER / n_pixels,
        dtype=torch.float64,
    )
    field = monochromatic_field(grid)
    return CircularAperture(grid, DIAMETER / 2).apply(field)


def test_circular_aperture_psf_is_centered_and_axisymmetric():
    field = circular_pupil_field()
    sampling = WAVELENGTH * FOCAL_LENGTH / DIAMETER / 20
    focal_grid = Grid(128, 128, sampling, sampling, dtype=torch.float64)

    image = MFTPropagator(FOCAL_LENGTH, focal_grid).apply(field).intensity()[0]
    center = focal_grid.nx // 2
    peak_y, peak_x = torch.unravel_index(image.argmax(), image.shape)

    assert (peak_y.item(), peak_x.item()) == (center, center)
    torch.testing.assert_close(image, image.T, rtol=1e-12, atol=1e-12)


def test_first_airy_zero_is_near_1_22_lambda_over_d():
    field = circular_pupil_field()
    sampling = WAVELENGTH * FOCAL_LENGTH / DIAMETER / 20
    focal_grid = Grid(128, 128, sampling, sampling, dtype=torch.float64)
    image = MFTPropagator(FOCAL_LENGTH, focal_grid).apply(field).intensity()[0]
    center = focal_grid.nx // 2
    radius_lambda_over_d = (
        focal_grid.x[center:] / (WAVELENGTH * FOCAL_LENGTH / DIAMETER)
    )
    radial_cut = image[center, center:]
    search = (radius_lambda_over_d > 0.9) & (radius_lambda_over_d < 1.5)
    candidates = torch.nonzero(search, as_tuple=False).flatten()
    zero_index = candidates[radial_cut[search].argmin()]
    measured_zero = radius_lambda_over_d[zero_index].item()

    assert abs(measured_zero - 1.21967) <= 0.05
    assert radial_cut[zero_index] / radial_cut[0] < 3e-4


def test_one_wavelength_piston_produces_two_pi_phase_and_same_field():
    grid = Grid(9, 7, 0.1, 0.1, dtype=torch.float64)
    field = monochromatic_field(grid)

    shifted = Piston(grid, piston=WAVELENGTH).apply(field)

    phase_shift = 2 * torch.pi * WAVELENGTH / field.spectrum.wavelengths[0]
    torch.testing.assert_close(
        phase_shift, torch.tensor(2 * torch.pi, dtype=torch.float64)
    )
    torch.testing.assert_close(
        shifted.complex_amplitude,
        field.complex_amplitude,
        rtol=1e-12,
        atol=1e-12,
    )


def test_one_cycle_tip_ramp_moves_psf_by_one_fourier_bin():
    n_pixels = 64
    field = circular_pupil_field(n_pixels)
    focal_sampling = WAVELENGTH * FOCAL_LENGTH / DIAMETER
    focal_grid = Grid(
        n_pixels,
        n_pixels,
        focal_sampling,
        focal_sampling,
        dtype=torch.float64,
    )
    tilted = TipTilt(
        field.grid,
        tip=WAVELENGTH / DIAMETER,
        tilt=0.0,
    ).apply(field)

    image = MFTPropagator(FOCAL_LENGTH, focal_grid).apply(tilted).intensity()[0]
    peak_y, peak_x = torch.unravel_index(image.argmax(), image.shape)

    assert peak_y.item() == n_pixels // 2
    assert peak_x.item() == n_pixels // 2 + 1


def test_lossless_mft_preserves_integrated_flux_on_conjugate_grids():
    n_pixels = 64
    field = circular_pupil_field(n_pixels)
    focal_grid = Grid(
        n_pixels,
        n_pixels,
        WAVELENGTH * FOCAL_LENGTH / (n_pixels * field.grid.dx),
        WAVELENGTH * FOCAL_LENGTH / (n_pixels * field.grid.dy),
        dtype=torch.float64,
    )

    propagated = MFTPropagator(FOCAL_LENGTH, focal_grid).apply(field)
    input_flux = field.intensity().sum() * field.grid.dx * field.grid.dy
    output_flux = (
        propagated.intensity().sum() * focal_grid.dx * focal_grid.dy
    )

    torch.testing.assert_close(output_flux, input_flux, rtol=1e-12, atol=1e-12)
