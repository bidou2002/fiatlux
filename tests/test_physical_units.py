import torch

from fiatlux.core.grid import Grid
from fiatlux.core.source import PlaneWave
from fiatlux.core.spectrum import Band, Spectrum
from fiatlux.optics.detector import Detector
from fiatlux.optics.propagator import IdentityPropagator, MFTPropagator
from fiatlux.system.optical_system import SerialSystem


def make_monochromatic_spectrum(wavelength=1e-6) -> Spectrum:
    return Spectrum(
        magnitude=0,
        band=Band(
            central_wavelength=wavelength,
            delta_wavelength=0.0,
            f0=3.68e8,
        ),
        samples=1,
    )


def integrated_photon_rate(field) -> torch.Tensor:
    return field.intensity().sum() * field.grid.dx * field.grid.dy


def test_plane_wave_system_matches_analytical_detector_counts():
    grid = Grid(nx=8, ny=6, dx=0.1, dy=0.2)
    spectrum = make_monochromatic_spectrum()
    source = PlaneWave(spectrum)
    identity = IdentityPropagator(grid)
    detector = Detector(
        grid,
        exposure_time=2.5,
        quantum_efficiency=0.4,
    )
    system = SerialSystem(elements=[identity])

    result = system.run(source, detector)

    expected_photon_rate = spectrum.fluxes.sum()
    expected_electrons = expected_photon_rate * 2.5 * 0.4
    torch.testing.assert_close(
        integrated_photon_rate(result.field_at(identity)),
        expected_photon_rate,
        rtol=1e-6,
        atol=0,
    )
    torch.testing.assert_close(
        detector.image_buffer.sum(),
        expected_electrons,
        rtol=1e-6,
        atol=0,
    )


def test_mft_preserves_flux_on_a_discrete_fourier_grid():
    n_pixels = 8
    wavelength = 1e-6
    focal_length = 2.0
    input_spacing = 0.1
    output_spacing = wavelength * focal_length / (n_pixels * input_spacing)
    input_grid = Grid(
        nx=n_pixels,
        ny=n_pixels,
        dx=input_spacing,
        dy=input_spacing,
    )
    output_grid = Grid(
        nx=n_pixels,
        ny=n_pixels,
        dx=output_spacing,
        dy=output_spacing,
    )
    input_field = PlaneWave(
        make_monochromatic_spectrum(wavelength)
    ).generate_field(input_grid)

    output_field = MFTPropagator(focal_length, output_grid).apply(input_field)

    torch.testing.assert_close(
        integrated_photon_rate(output_field),
        integrated_photon_rate(input_field),
        rtol=1e-5,
        atol=0,
    )
