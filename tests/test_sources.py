import pytest
import torch

from fiatlux.core.grid import Grid
from fiatlux.core.source import GaussianSource, PlaneWave
from fiatlux.core.spectrum import Band, Spectrum
from fiatlux.optics.detector import Detector


def make_spectrum(samples=3) -> Spectrum:
    return Spectrum(
        magnitude=2,
        band=Band(
            central_wavelength=1.1e-6,
            delta_wavelength=0.2e-6,
            f0=1e10,
        ),
        samples=samples,
    )


def integrated_spectral_flux(field) -> torch.Tensor:
    return field.intensity().sum(dim=(-2, -1)) * field.grid.dx * field.grid.dy


@pytest.mark.parametrize("source_type", [PlaneWave, GaussianSource])
def test_each_source_integrates_to_requested_flux_per_wavelength(source_type):
    grid = Grid(nx=7, ny=5, dx=0.03, dy=0.05)
    spectrum = make_spectrum()
    source = (
        source_type(spectrum)
        if source_type is PlaneWave
        else source_type(spectrum, waist=0.08)
    )

    field = source.generate_field(grid)

    assert field.complex_amplitude.shape == (3, 5, 7)
    assert field.complex_amplitude.is_complex()
    torch.testing.assert_close(
        integrated_spectral_flux(field),
        spectrum.fluxes,
        rtol=1e-6,
        atol=0,
    )


def test_plane_and_gaussian_sources_have_the_same_total_photon_rate():
    grid = Grid(nx=9, ny=7, dx=0.02, dy=0.03)
    spectrum = make_spectrum(samples=4)

    plane = PlaneWave(spectrum).generate_field(grid)
    gaussian = GaussianSource(spectrum, waist=0.06).generate_field(grid)

    plane_rate = plane.intensity().sum() * grid.dx * grid.dy
    gaussian_rate = gaussian.intensity().sum() * grid.dx * grid.dy
    torch.testing.assert_close(plane_rate, spectrum.fluxes.sum(), rtol=1e-6, atol=0)
    torch.testing.assert_close(gaussian_rate, plane_rate, rtol=1e-6, atol=0)


def test_plane_wave_normalization_is_independent_of_grid_sampling():
    spectrum = make_spectrum(samples=1)
    coarse = Grid(nx=5, ny=4, dx=0.1, dy=0.1)
    fine = Grid(nx=10, ny=8, dx=0.05, dy=0.05)

    coarse_rate = integrated_spectral_flux(PlaneWave(spectrum).generate_field(coarse))
    fine_rate = integrated_spectral_flux(PlaneWave(spectrum).generate_field(fine))

    torch.testing.assert_close(coarse_rate, spectrum.fluxes, rtol=1e-6, atol=0)
    torch.testing.assert_close(fine_rate, spectrum.fluxes, rtol=1e-6, atol=0)


def test_source_to_detector_applies_pixel_area_exactly_once():
    grid = Grid(nx=7, ny=5, dx=0.03, dy=0.05)
    spectrum = make_spectrum()
    field = PlaneWave(spectrum).generate_field(grid)
    detector = Detector(grid, exposure_time=1.0, quantum_efficiency=1.0)

    image = detector.acquire(field)

    torch.testing.assert_close(image.sum(), spectrum.fluxes.sum(), rtol=1e-6, atol=0)


@pytest.mark.parametrize("waist", [0.0, -1.0, float("inf"), float("nan")])
def test_gaussian_source_rejects_invalid_waist(waist):
    with pytest.raises(ValueError, match="waist must be a positive finite length"):
        GaussianSource(make_spectrum(), waist=waist)


def test_source_rejects_invalid_spectral_flux():
    spectrum = make_spectrum(samples=1)
    spectrum.fluxes[0] = -1

    with pytest.raises(ValueError, match="fluxes must be finite and non-negative"):
        PlaneWave(spectrum).generate_field(Grid(nx=3, ny=2, dx=0.1, dy=0.2))
